"""Parse Mirage demos with awpy and merge all of its tables into 2 Hz round arrays.

For every demo in data/ this writes processed/<demo>.npz holding, for each
round from freeze end to round end, sampled every 0.5 s:

    player       [steps, 10, F]  per-player state (ticks) + bomb carrier (bomb)
                                 + event counts (shots, damages, kills, footsteps)
    player_cat   [steps, 10, 2]  weapon / place ids (per-demo string tables)
    last_seen    [steps, 10, 3]  position when last spotted by the enemy team
    seen_age     [steps, 10]     seconds since last spotted (NaN = never)
    visible      [steps, 10]     spotted at some tick during this half second
    globals      [steps, G]      round clock and bomb state (rounds, bomb)
    side         [steps, 2, 3]   score / alive / equipment per side
    utility      [steps, 10, 8]  active smokes and infernos (smokes, infernos)

Slots 0-4 are the T side and 5-9 the CT side for the whole round. Round-level
arrays (round_*) give each round's step offset, length and roster. Feature
names are in features.py. After processing it writes processed/manifest.json
with the match-level train/val/test split, string vocabularies and position
statistics used by dataset.py to build the actual training windows.

Usage:
    python build_dataset.py                    # data/*.dem -> processed/
    python build_dataset.py --workers 6        # parse demos in parallel
    python build_dataset.py --overwrite        # reprocess existing .npz files
"""

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import polars as pl

from features import (
    AWPY_PLAYER_PROPS,
    G,
    GLOBAL_FEATURES,
    MAX_UTILITY,
    N_SLOTS,
    P,
    PLAYER_CATEGORICALS,
    PLAYER_FEATURES,
    S,
    SIDE_FEATURES,
    SLOTS_PER_SIDE,
    STEP_SECONDS,
    U,
    UTILITY_FEATURES,
)

# join_asof with `by` can't verify sortedness; both sides are sorted by tick.
warnings.filterwarnings("ignore", message="Sortedness of columns cannot be checked")

ROOT = Path(__file__).parent
SIDES = ("t", "ct")
SMOKE_SECONDS = 22  # fallback lifetimes when awpy finds no end event
INFERNO_SECONDS = 7

# ticks column -> PLAYER_FEATURES name, copied straight from the asof-joined state
STATE_COLUMNS = {
    "X": "X", "Y": "Y", "Z": "Z",
    "velocity_X": "vel_x", "velocity_Y": "vel_y", "velocity_Z": "vel_z",
    "yaw": "yaw", "pitch": "pitch",
    "health": "health", "armor": "armor", "has_helmet": "has_helmet", "has_defuser": "has_defuser",
    "is_alive": "is_alive", "flash_duration": "flash_duration", "is_scoped": "is_scoped",
    "is_walking": "is_walking", "ducking": "ducking", "is_defusing": "is_defusing",
    "current_equip_value": "equip_value", "spotted": "spotted",
}


class StringTable:
    """Per-demo string -> id table; id 0 is reserved for missing values."""

    def __init__(self):
        self.ids = {}

    def encode(self, series: pl.Series) -> np.ndarray:
        values = series.fill_null("").to_list()
        out = np.zeros(len(values), dtype=np.int16)
        for i, v in enumerate(values):
            if v:
                out[i] = self.ids.setdefault(v, len(self.ids) + 1)
        return out

    def strings(self) -> np.ndarray:
        return np.array([""] + list(self.ids), dtype=str)


def step_index(event_ticks: np.ndarray, freeze_end: int, step_ticks: int) -> np.ndarray:
    """Step k covers ticks (sample_k - step_ticks, sample_k]."""
    return np.ceil((event_ticks - freeze_end) / step_ticks).astype(np.int64)


def add_event_counts(player, events, id_col, feature, slot_of, freeze_end, step_ticks, values=None):
    """Accumulate an event table into player[:, :, feature] (count, or sum of `values`)."""
    if events.is_empty():
        return
    ids = events[id_col].to_list()
    slots = np.array([slot_of.get(i, -1) for i in ids], dtype=np.int64)
    steps = step_index(events["tick"].to_numpy(), freeze_end, step_ticks)
    amounts = np.ones(len(ids), np.float32) if values is None else events[values].fill_null(0).to_numpy().astype(np.float32)
    keep = (slots >= 0) & (steps >= 0) & (steps < player.shape[0])
    np.add.at(player[:, :, P[feature]], (steps[keep], slots[keep]), amounts[keep])


def pick_roster(round_ticks: pl.DataFrame):
    """Return {side: [(steamid, name, clan), ...]} for up to 5 players per side."""
    first = round_ticks.filter(pl.col("tick") == pl.col("tick").min())
    roster = {}
    for side in SIDES:
        rows = (
            first.filter(pl.col("side") == side)
            .sort(["is_alive", "steamid"], descending=[True, False])
            .head(SLOTS_PER_SIDE)
        )
        roster[side] = list(zip(rows["steamid"].to_list(), rows["name"].to_list(), rows["team_clan_name"].to_list()))
    return roster


def bomb_state(round_bomb, sample_ticks, slot_of, rnd, tickrate):
    """Replay bomb events into per-step carrier, dropped and planted state."""
    T = len(sample_ticks)
    has_bomb = np.zeros((T, N_SLOTS), np.float32)
    glob = np.zeros((T, len(GLOBAL_FEATURES)), np.float32)

    events = round_bomb.sort("tick")
    ev_ticks = events["tick"].to_numpy()
    ev_rows = events.select("event", "steamid", "X", "Y", "Z").rows()
    last = np.searchsorted(ev_ticks, sample_ticks, side="right") - 1
    for k, e in enumerate(last):
        if e < 0:
            continue
        kind, steamid, x, y, z = ev_rows[e]
        if kind == "pickup" and steamid in slot_of:
            has_bomb[k, slot_of[steamid]] = 1
        elif kind == "drop":
            glob[k, G["bomb_dropped"]] = 1
        if kind in ("drop", "plant", "defuse", "detonate") and x is not None:
            glob[k, [G["bomb_x"], G["bomb_y"], G["bomb_z"]]] = (x, y, z)

    plant_tick = rnd["bomb_plant"]
    if plant_tick is not None:
        planted = sample_ticks >= plant_tick
        glob[planted, G["bomb_planted"]] = 1
        glob[planted, G["time_since_plant"]] = (sample_ticks[planted] - plant_tick) / tickrate
        site = (rnd["bomb_site"] or "").lower()
        glob[planted, G["plant_site_a"]] = site.endswith("_a")
        glob[planted, G["plant_site_b"]] = site.endswith("_b")
    return has_bomb, glob


def utility_tokens(smokes, infernos, sample_ticks, tickrate):
    """Active smokes/infernos per step, newest first, capped at MAX_UTILITY."""
    T = len(sample_ticks)
    util = np.zeros((T, MAX_UTILITY, len(UTILITY_FEATURES)), np.float32)
    items = []
    for df, kind in ((smokes, "is_smoke"), (infernos, "is_inferno")):
        for start, end, x, y, z, side in df.select("start_tick", "end_tick", "X", "Y", "Z", "thrower_side").rows():
            items.append((start, end, x, y, z, side, kind))
    items.sort(key=lambda it: -it[0])
    for k, t in enumerate(sample_ticks):
        active = [it for it in items if it[0] <= t < it[1]][:MAX_UTILITY]
        for j, (start, _, x, y, z, side, kind) in enumerate(active):
            row = util[k, j]
            row[[U["X"], U["Y"], U["Z"]]] = (x, y, z)
            row[U["age"]] = (t - start) / tickrate
            row[U[kind]] = 1
            row[U["thrown_by_t"]] = side == "t"
            row[U["valid"]] = 1
    return util


def build_round(rnd, round_ticks, tables, team_wins, weapons, places, step_ticks, tickrate):
    freeze_end, end = rnd["freeze_end"], rnd["end"]
    sample_ticks = np.arange(freeze_end, end + 1, step_ticks, dtype=np.int64)
    T = len(sample_ticks)

    roster = pick_roster(round_ticks)
    if not roster["t"] or not roster["ct"]:
        return None
    slot_ids = [None] * N_SLOTS
    names = [""] * N_SLOTS
    clans = []
    for side_idx, side in enumerate(SIDES):
        for j, (steamid, name, _) in enumerate(roster[side]):
            slot_ids[side_idx * SLOTS_PER_SIDE + j] = steamid
            names[side_idx * SLOTS_PER_SIDE + j] = name or ""
        clans.append(roster[side][0][2] or side)
    slot_of = {sid: i for i, sid in enumerate(slot_ids) if sid is not None}
    present = np.array(list(slot_of.values()))

    # Sample each player's state at every step (latest tick at or before the sample tick).
    grid = pl.DataFrame({
        "tick": np.repeat(sample_ticks, len(present)).astype(np.int32),
        "steamid": pl.Series(np.tile(list(slot_of.keys()), T), dtype=pl.UInt64),
        "step": np.repeat(np.arange(T), len(present)),
        "slot": np.tile(present, T),
    }).sort("tick")
    state = grid.join_asof(round_ticks, on="tick", by="steamid", strategy="backward")
    steps, slots = state["step"].to_numpy(), state["slot"].to_numpy()

    player = np.zeros((T, N_SLOTS, len(PLAYER_FEATURES)), np.float32)
    for col, feature in STATE_COLUMNS.items():
        player[steps, slots, P[feature]] = state[col].cast(pl.Float32).fill_null(0).to_numpy()
    player[:, :, P["spotted"]] *= player[:, :, P["is_alive"]]  # dead players keep a stale spotted flag
    player_cat = np.zeros((T, N_SLOTS, len(PLAYER_CATEGORICALS)), np.int16)
    player_cat[steps, slots, 0] = weapons.encode(state["active_weapon_name"])
    player_cat[steps, slots, 1] = places.encode(state["place"])

    # Team-perspective visibility: when and where each player was last spotted.
    spotted = (
        round_ticks.filter(pl.col("spotted") & pl.col("is_alive"))
        .select("tick", "steamid", pl.col("tick").alias("seen_tick"), "X", "Y", "Z")
    )
    seen = grid.join_asof(spotted, on="tick", by="steamid", strategy="backward")
    seen_tick = seen["seen_tick"].cast(pl.Float64).to_numpy()  # NaN where never seen
    seen_steps, seen_slots = seen["step"].to_numpy(), seen["slot"].to_numpy()
    last_seen = np.zeros((T, N_SLOTS, 3), np.float32)
    last_seen[seen_steps, seen_slots] = seen.select("X", "Y", "Z").fill_null(0).to_numpy()
    seen_age = np.full((T, N_SLOTS), np.nan, np.float32)
    seen_age[seen_steps, seen_slots] = (seen["tick"].to_numpy() - seen_tick) / tickrate
    visible = np.zeros((T, N_SLOTS), bool)
    visible[seen_steps, seen_slots] = seen_tick > seen["tick"].to_numpy() - step_ticks

    # Event tables -> per-step counts.
    round_num = rnd["round_num"]
    t = {name: df.filter(pl.col("round_num") == round_num) for name, df in tables.items()}
    add_event_counts(player, t["shots"], "player_steamid", "shots", slot_of, freeze_end, step_ticks)
    add_event_counts(player, t["damages"], "attacker_steamid", "dmg_dealt", slot_of, freeze_end, step_ticks, "dmg_health_real")
    add_event_counts(player, t["damages"], "victim_steamid", "dmg_taken", slot_of, freeze_end, step_ticks, "dmg_health_real")
    add_event_counts(player, t["kills"], "attacker_steamid", "kills", slot_of, freeze_end, step_ticks)
    add_event_counts(player, t["footsteps"], "player_steamid", "footsteps", slot_of, freeze_end, step_ticks)

    has_bomb, glob = bomb_state(t["bomb"], sample_ticks, slot_of, rnd, tickrate)
    player[:, :, P["has_bomb"]] = has_bomb
    glob[:, G["time_since_freeze_end"]] = (sample_ticks - freeze_end) / tickrate

    side = np.zeros((T, 2, len(SIDE_FEATURES)), np.float32)
    for side_idx in range(2):
        block = player[:, side_idx * SLOTS_PER_SIDE:(side_idx + 1) * SLOTS_PER_SIDE]
        side[:, side_idx, S["score"]] = team_wins.get(clans[side_idx], 0)
        side[:, side_idx, S["alive"]] = block[:, :, P["is_alive"]].sum(1)
        side[:, side_idx, S["equip_value"]] = block[:, :, P["equip_value"]].sum(1)

    utility = utility_tokens(t["smokes"], t["infernos"], sample_ticks, tickrate)

    if rnd["winner"] in SIDES:
        winner_clan = clans[SIDES.index(rnd["winner"])]
        team_wins[winner_clan] = team_wins.get(winner_clan, 0) + 1

    return {
        "player": player, "player_cat": player_cat,
        "last_seen": last_seen, "seen_age": seen_age, "visible": visible,
        "globals": glob, "side": side, "utility": utility,
        "meta": {
            "round_num": round_num,
            "n_steps": T,
            "ticks": (rnd["start"], freeze_end, end),
            "winner": SIDES.index(rnd["winner"]) if rnd["winner"] in SIDES else -1,
            "steamids": [sid or 0 for sid in slot_ids],
            "slot_valid": [sid is not None for sid in slot_ids],
            "names": names,
            "teams": clans,
        },
    }


def measure_tickrate(parser, default: int = 64) -> int:
    """CS2 demos record at 64 ticks/s, but awpy assumes 128; measure it from game_time."""
    try:
        times = parser.parse_ticks(["game_time"], ticks=[2000, 6000]).drop_duplicates("tick").sort_values("tick")
        rate = 4000 / (times["game_time"].iloc[1] - times["game_time"].iloc[0])
        return min((64, 128), key=lambda r: abs(r - rate))
    except Exception:
        return default


def process_demo(path: Path, out_path: Path) -> dict:
    from awpy import Demo  # imported here so worker processes load it themselves
    from demoparser2 import DemoParser

    start = time.perf_counter()
    tickrate = measure_tickrate(DemoParser(str(path)))
    # The durations are upper bounds awpy uses to discard unmatched end events; real
    # smokes last ~22 s and infernos ~7 s.
    demo = Demo(path, tickrate=tickrate, smoke_duration=SMOKE_SECONDS + 5, inferno_duration=INFERNO_SECONDS + 3)
    map_name = demo.header.get("map_name")
    if map_name != "de_mirage":
        return {"file": path.name, "skipped": f"map is {map_name}"}
    demo.parse(player_props=AWPY_PLAYER_PROPS)
    step_ticks = round(STEP_SECONDS * tickrate)

    ticks = demo.ticks.sort("tick")
    tables = {
        "shots": demo.shots.select("tick", "round_num", "player_steamid"),
        "damages": demo.damages.select("tick", "round_num", "attacker_steamid", "victim_steamid", "dmg_health_real"),
        "kills": demo.kills.select("tick", "round_num", "attacker_steamid"),
        "footsteps": demo.footsteps.select("tick", "round_num", "player_steamid"),
        "bomb": demo.bomb.select("tick", "round_num", "event", "steamid", "X", "Y", "Z"),
        "smokes": demo.smokes,
        "infernos": demo.infernos,
    }
    for name, seconds in (("smokes", SMOKE_SECONDS), ("infernos", INFERNO_SECONDS)):
        tables[name] = tables[name].select(
            "round_num", "start_tick", "X", "Y", "Z", "thrower_side",
            pl.col("end_tick").fill_null(pl.col("start_tick") + int(seconds * tickrate)),
        )

    weapons, places = StringTable(), StringTable()
    team_wins = {}
    rounds = []
    for rnd in demo.rounds.sort("round_num").iter_rows(named=True):
        if rnd["freeze_end"] is None or rnd["end"] is None or rnd["end"] <= rnd["freeze_end"]:
            continue
        round_ticks = ticks.filter(
            (pl.col("round_num") == rnd["round_num"])
            & pl.col("tick").is_between(rnd["freeze_end"], rnd["end"])
            & pl.col("side").is_in(SIDES)
        )
        if round_ticks.is_empty():
            continue
        built = build_round(rnd, round_ticks, tables, team_wins, weapons, places, step_ticks, tickrate)
        if built is not None:
            rounds.append(built)

    if not rounds:
        return {"file": path.name, "skipped": "no usable rounds"}

    metas = [r["meta"] for r in rounds]
    n_steps = np.array([m["n_steps"] for m in metas])
    arrays = {key: np.concatenate([r[key] for r in rounds]) for key in rounds[0] if key != "meta"}
    arrays.update(
        round_num=np.array([m["round_num"] for m in metas], np.int32),
        round_offset=np.concatenate([[0], np.cumsum(n_steps)[:-1]]).astype(np.int64),
        round_n_steps=n_steps.astype(np.int32),
        round_ticks=np.array([m["ticks"] for m in metas], np.int64),
        round_winner=np.array([m["winner"] for m in metas], np.int8),
        round_steamids=np.array([m["steamids"] for m in metas], np.uint64),
        round_slot_valid=np.array([m["slot_valid"] for m in metas], bool),
        round_names=np.array([m["names"] for m in metas], dtype=str),
        round_teams=np.array([m["teams"] for m in metas], dtype=str),
        weapon_strings=weapons.strings(),
        place_strings=places.strings(),
        tickrate=np.array(tickrate),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(out_path)
    return {"file": path.name, "rounds": len(rounds), "steps": int(n_steps.sum()), "seconds": time.perf_counter() - start}


def process_demo_safe(path: Path, out_path: Path) -> dict:
    try:
        return process_demo(path, out_path)
    except Exception:
        return {"file": path.name, "error": traceback.format_exc()}


def match_id(stem: str) -> str:
    """Maps of the same series share a match id so they land in the same split."""
    return re.sub(r"(-m\d+)?-mirage(_\d+)?$", "", stem)


def assign_split(stem: str, val_frac: float, test_frac: float, seed: int) -> str:
    digest = hashlib.md5(f"{seed}:{match_id(stem)}".encode()).hexdigest()
    u = int(digest[:8], 16) / 0x100000000
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "val"
    return "train"


def write_manifest(out_dir: Path, val_frac: float, test_frac: float, seed: int) -> dict:
    demos, weapons, places = [], set(), set()
    pos_sum, pos_sq, pos_n = np.zeros(2), 0.0, 0
    for npz_path in sorted(out_dir.glob("*.npz")):
        if npz_path.name.endswith(".tmp.npz"):
            continue
        with np.load(npz_path) as z:
            split = assign_split(npz_path.stem, val_frac, test_frac, seed)
            demos.append({
                "file": npz_path.name,
                "match": match_id(npz_path.stem),
                "split": split,
                "rounds": int(len(z["round_num"])),
                "steps": int(z["round_n_steps"].sum()),
            })
            weapons.update(z["weapon_strings"][1:].tolist())
            places.update(z["place_strings"][1:].tolist())
            if split == "train":
                alive = z["player"][:, :, P["is_alive"]] > 0
                xy = z["player"][:, :, [P["X"], P["Y"]]][alive].astype(np.float64)
                pos_sum += xy.sum(0)
                pos_sq += (xy ** 2).sum()
                pos_n += len(xy)
    if pos_n == 0:
        sys.exit("No training demos found; cannot compute position statistics.")
    mean = pos_sum / pos_n
    std = float(np.sqrt(pos_sq / pos_n / 2 - (mean ** 2).mean()))  # isotropic, keeps the map's aspect ratio
    manifest = {
        "step_seconds": STEP_SECONDS,
        "player_features": PLAYER_FEATURES,
        "player_categoricals": PLAYER_CATEGORICALS,
        "global_features": GLOBAL_FEATURES,
        "side_features": SIDE_FEATURES,
        "utility_features": UTILITY_FEATURES,
        "weapon_vocab": ["<none>"] + sorted(weapons),
        "place_vocab": ["<none>"] + sorted(places),
        "pos_mean": mean.tolist(),
        "pos_std": std,
        "split_seed": seed,
        "demos": demos,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", nargs="?", default=str(ROOT / "data"), help="folder of .dem files (default: data/)")
    parser.add_argument("--out", default=str(ROOT / "processed"), help="output folder (default: processed/)")
    parser.add_argument("--workers", type=int, default=4, help="demos parsed in parallel (default: 4)")
    parser.add_argument("--overwrite", action="store_true", help="reprocess demos that already have an .npz")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0, help="split seed")
    args = parser.parse_args()

    out_dir = Path(args.out)
    demos = sorted(Path(args.data_dir).glob("*.dem"))
    todo = [d for d in demos if args.overwrite or not (out_dir / f"{d.stem}.npz").exists()]
    print(f"{len(demos)} demo(s) found, {len(todo)} to process")

    start = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = [pool.submit(process_demo_safe, d, out_dir / f"{d.stem}.npz") for d in todo]
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                prefix = f"[{i}/{len(todo)}] {result['file']}"
                if "error" in result:
                    print(f"{prefix}: FAILED\n{result['error']}")
                elif "skipped" in result:
                    print(f"{prefix}: skipped ({result['skipped']})")
                else:
                    print(f"{prefix}: {result['rounds']} rounds, {result['steps']} steps in {result['seconds']:.1f}s")

    manifest = write_manifest(out_dir, args.val_frac, args.test_frac, args.seed)
    counts = {s: sum(d["split"] == s for d in manifest["demos"]) for s in ("train", "val", "test")}
    print(f"Manifest written: {counts} demos per split ({time.perf_counter() - start:.1f}s)")
    for split, n in counts.items():
        if n == 0:
            print(f"  warning: the {split} split is empty; add demos or change --seed / fractions")


if __name__ == "__main__":
    main()
