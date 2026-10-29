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
    GRENADE_ITEMS,
    INFERNO_LINK_TICKS,
    MAX_UTILITY,
    MOVE_TYPE_LADDER,
    N_SLOTS,
    NON_PRIMARY,
    P,
    PISTOLS,
    PLAYER_CATEGORICALS,
    PLAYER_FEATURES,
    POP_SECONDS,
    PROJECTILE_KIND,
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
    "balance": "money", "cash_spent_this_round": "cash_spent", "round_start_equip_value": "start_equip_value",
    "active_weapon_ammo": "ammo_clip", "total_ammo_left": "mags_left",
    "is_airborne": "is_airborne", "duck_amount": "duck_amount", "zoom_lvl": "zoom_lvl",
}
KNIFE_PATTERN = "(?i)knife|bayonet|dagger|karambit"
VELOCITY_TICKS = 4  # grenade velocity is measured over the last 4 ticks


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


def bomb_state(round_bomb, sample_ticks, has_bomb, rnd, tickrate):
    """Per-step dropped / planted state and bomb position.

    The carrier comes from inventories (has_bomb [T, 10]), not from bomb events:
    pickup events can be missing (e.g. a bomb passed on during freeze time), which
    left the bomb "dropped" for a whole round while a player carried it. The bomb
    is dropped when it is not planted and no living player carries it; its
    position is that of the latest drop / plant / defuse / detonate event.
    """
    T = len(sample_ticks)
    glob = np.zeros((T, len(GLOBAL_FEATURES)), np.float32)

    planted = np.zeros(T, bool)
    plant_tick = rnd["bomb_plant"]
    if plant_tick is not None:
        planted = sample_ticks >= plant_tick
        glob[planted, G["bomb_planted"]] = 1
        glob[planted, G["time_since_plant"]] = (sample_ticks[planted] - plant_tick) / tickrate
        site = (rnd["bomb_site"] or "").lower()
        glob[planted, G["plant_site_a"]] = site.endswith("_a")
        glob[planted, G["plant_site_b"]] = site.endswith("_b")
    dropped = ~planted & ~(has_bomb > 0).any(1)
    glob[dropped, G["bomb_dropped"]] = 1

    events = round_bomb.filter(pl.col("event").is_in(["drop", "plant", "defuse", "detonate"]) & pl.col("X").is_not_null()).sort("tick")
    last = np.searchsorted(events["tick"].to_numpy(), sample_ticks, side="right") - 1
    xyz = events.select("X", "Y", "Z").to_numpy()
    on_ground = (dropped | planted) & (last >= 0)
    glob[np.ix_(on_ground, [G["bomb_x"], G["bomb_y"], G["bomb_z"]])] = xyz[last[on_ground]]
    return glob


def detonation_events(events: dict) -> dict:
    """{kind: [(entity_id, tick, x, y, z), ...]} from the detonate events awpy already parsed."""
    out = {}
    for event, kind in (("hegrenade_detonate", "is_he"), ("flashbang_detonate", "is_flash"),
                        ("smokegrenade_detonate", "is_smoke")):
        df = events.get(event)
        ok = df is not None and not df.is_empty() and {"entityid", "x", "y", "z"} <= set(df.columns)
        out[kind] = df.select("entityid", "tick", "x", "y", "z").rows() if ok else []
    return out


def grenade_throws(grenades: pl.DataFrame, detonations: dict, infernos: pl.DataFrame) -> dict:
    """Split awpy's per-tick projectile table into throws, grouped by round.

    Entity ids are reused, so a throw is a run of consecutive ticks of one entity
    and thrower. A throw is in flight until it pops. Flash, HE and smoke projectiles
    linger after going off, so their pop tick and position come from the detonate
    events. A molotov projectile disappears a few ticks before its fire starts, so
    it stays in flight (at its last position) until the thrower's fire starts.
    """
    fires = infernos.select("thrower_steamid", "start_tick").rows()
    g = (
        grenades.filter(pl.col("grenade_type").is_in(list(PROJECTILE_KIND)) & pl.col("X").is_not_null())
        .sort("entity_id", "tick")
        .with_columns(
            ((pl.col("entity_id") != pl.col("entity_id").shift())
             | (pl.col("tick") - pl.col("tick").shift() != 1)
             | (pl.col("thrower_steamid") != pl.col("thrower_steamid").shift()))
            .fill_null(True).cum_sum().alias("throw")
        )
    )
    throws = {}
    for (_,), part in g.group_by(["throw"], maintain_order=True):
        ticks = part["tick"].to_numpy()
        xyz = part.select("X", "Y", "Z").to_numpy()
        start, end = int(ticks[0]), int(ticks[-1])
        kind = PROJECTILE_KIND[part["grenade_type"][0]]
        entity = part["entity_id"][0]
        thrower = part["thrower_steamid"][0]
        pop, pop_xyz = end, xyz[-1]
        for e, tick, x, y, z in detonations.get(kind, ()):
            if e == entity and start <= tick <= end + 1:
                pop, pop_xyz = int(tick), np.array((x, y, z), np.float32)
                break
        if kind == "is_inferno":
            pop = next((s for th, s in fires if th == thrower and abs(s - end) <= INFERNO_LINK_TICKS), end)
        throws.setdefault(part["round_num"][0], []).append({
            "kind": kind, "entity": entity, "thrower": thrower, "start": start, "end": end,
            "pop": pop, "pop_xyz": pop_xyz, "ticks": ticks, "xyz": xyz,
        })
    for round_throws in throws.values():
        round_throws.sort(key=lambda th: (th["start"], th["entity"]))
        for i, th in enumerate(round_throws, 1):
            th["id"] = i
    return throws


def effect_throw_id(throws, kind, entity, thrower, start):
    """throw_id of the grenade that made a smoke cloud or fire, 0 if none matches."""
    for th in throws:
        if kind == "is_smoke" and th["kind"] == "is_smoke" and th["entity"] == entity and th["start"] <= start <= th["end"] + 1:
            return th["id"]
        if (kind == "is_inferno" and th["kind"] == "is_inferno" and th["thrower"] == thrower
                and abs(start - th["pop"]) <= INFERNO_LINK_TICKS):
            return th["id"]
    return 0


def utility_tokens(smokes, infernos, throws, sample_ticks, tickrate, slot_of):
    """Utility tokens per step, newest first, capped at MAX_UTILITY.

    Grenades in flight (decoys stay tokens while they sit on the ground), flash/HE
    detonations of the last POP_SECONDS, and active smokes/infernos.
    """
    T = len(sample_ticks)
    util = np.zeros((T, MAX_UTILITY, len(UTILITY_FEATURES)), np.float32)
    pop_ticks = int(POP_SECONDS * tickrate)
    effects = [
        (start, end, (x, y, z), kind, side == "t", effect_throw_id(throws, kind, entity, thrower, start))
        for df, kind in ((smokes, "is_smoke"), (infernos, "is_inferno"))
        for start, end, x, y, z, side, entity, thrower in df.select(
            "start_tick", "end_tick", "X", "Y", "Z", "thrower_side", "entity_id", "thrower_steamid").rows()
    ]
    for k, t in enumerate(sample_ticks):
        active = []  # (sort key, xyz, velocity, age, kind, in_flight, thrown_by_t, throw_id)
        for th in throws:
            side_t = slot_of.get(th["thrower"], N_SLOTS) < SLOTS_PER_SIDE
            if th["kind"] in ("is_flash", "is_he") and th["pop"] <= t < th["pop"] + pop_ticks:
                active.append((th["pop"], th["pop_xyz"], (0, 0, 0), (t - th["pop"]) / tickrate, th["kind"], 0, side_t, th["id"]))
            # Decoys stay tokens while they sit on the ground; the rest only until they pop
            # (a molotov's pop can come after its projectile's last tick: it then holds that position).
            last = th["end"] if th["kind"] == "is_decoy" else th["pop"] - 1
            if not th["start"] <= t <= last:
                continue
            i = np.searchsorted(th["ticks"], t, side="right") - 1
            i0 = max(np.searchsorted(th["ticks"], t - VELOCITY_TICKS, side="right") - 1, 0)
            dt = (th["ticks"][i] - th["ticks"][i0]) / tickrate
            vel = (th["xyz"][i] - th["xyz"][i0]) / dt if dt > 0 else (0, 0, 0)
            in_flight = np.linalg.norm(vel) > 1.0 if th["kind"] == "is_decoy" else True
            active.append((th["start"], th["xyz"][i], vel, (t - th["start"]) / tickrate, th["kind"], in_flight, side_t, th["id"]))
        for start, end, xyz, kind, side_t, throw_id in effects:
            if start <= t < end:
                active.append((start, xyz, (0, 0, 0), (t - start) / tickrate, kind, 0, side_t, throw_id))
        active.sort(key=lambda a: -a[0])
        for j, (_, xyz, vel, age, kind, in_flight, side_t, throw_id) in enumerate(active[:MAX_UTILITY]):
            row = util[k, j]
            row[[U["X"], U["Y"], U["Z"]]] = xyz
            row[[U["vel_x"], U["vel_y"], U["vel_z"]]] = vel
            row[U["age"]] = age
            row[U[kind]] = 1
            row[U["in_flight"]] = in_flight
            row[U["thrown_by_t"]] = side_t
            row[U["valid"]] = 1
            row[U["throw_id"]] = throw_id
    return util


def build_round(rnd, round_ticks, round_state, tables, throws, weapons, places, step_ticks, tickrate):
    """round_ticks: light props at every tick of the round; round_state: all props at sample ticks."""
    freeze_end, end = rnd["freeze_end"], rnd["end"]
    sample_ticks = np.arange(freeze_end, end + 1, step_ticks, dtype=np.int64)
    T = len(sample_ticks)

    roster = pick_roster(round_ticks)
    if not roster["t"] or not roster["ct"]:
        return "a side has no players"
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
    source = round_state.with_columns(pl.col("tick").alias("source_tick"))
    state = grid.join_asof(source, on="tick", by="steamid", strategy="backward")
    steps, slots = state["step"].to_numpy(), state["slot"].to_numpy()

    # Refuse stale state: a living player's sample must come from within the last step,
    # and the game clock must keep running (a real mid-round pause would freeze everyone).
    age = (state["tick"] - state["source_tick"]).fill_null(np.iinfo(np.int32).max).to_numpy()
    stale = (age >= step_ticks) & state["is_alive"].fill_null(False).to_numpy()
    if stale.any():
        return f"state of a living player is {age[stale].max()} ticks old at step {steps[stale].min()}"
    clock = state.group_by("step").agg(pl.col("game_time").max()).sort("step")["game_time"].to_numpy()
    paused = np.flatnonzero(np.diff(clock) < STEP_SECONDS / 2)
    if len(paused):
        return f"game clock stops at step {paused[0] + 1} (paused)"

    player = np.zeros((T, N_SLOTS, len(PLAYER_FEATURES)), np.float32)
    for col, feature in STATE_COLUMNS.items():
        player[steps, slots, P[feature]] = state[col].cast(pl.Float32).fill_null(0).to_numpy()
    player[:, :, P["spotted"]] *= player[:, :, P["is_alive"]]  # dead players keep a stale spotted flag
    player[steps, slots, P["on_ladder"]] = (state["move_type"] == MOVE_TYPE_LADDER).fill_null(False).to_numpy()

    # Inventory: grenades carried by type, zeus, and primary / secondary weapon.
    inventory = state["inventory"].fill_null([])
    for feature, items in GRENADE_ITEMS.items():
        player[steps, slots, P[feature]] = inventory.list.eval(pl.element().is_in(items).cast(pl.Int32)).list.sum().to_numpy()
    player[steps, slots, P["has_zeus"]] = inventory.list.contains("Zeus x27").to_numpy()
    player[steps, slots, P["has_bomb"]] = (inventory.list.contains("C4 Explosive") & state["is_alive"].fill_null(False)).to_numpy()
    primary = inventory.list.eval(
        pl.element().filter(~pl.element().is_in(NON_PRIMARY) & ~pl.element().str.contains(KNIFE_PATTERN))
    ).list.first()
    secondary = inventory.list.eval(pl.element().filter(pl.element().is_in(PISTOLS))).list.first()

    player_cat = np.zeros((T, N_SLOTS, len(PLAYER_CATEGORICALS)), np.int16)
    player_cat[steps, slots, 0] = weapons.encode(state["active_weapon_name"])
    player_cat[steps, slots, 1] = places.encode(state["place"])
    player_cat[steps, slots, 2] = weapons.encode(primary)
    player_cat[steps, slots, 3] = weapons.encode(secondary)

    # Who spots whom: spotted_by[step, slot, by_slot].
    spotted_by = np.zeros((T, N_SLOTS, N_SLOTS), bool)
    for k, s, seen_by in zip(steps, slots, state["approximate_spotted_by"].to_list()):
        for steamid in seen_by or ():
            by = slot_of.get(steamid)
            if by is not None:
                spotted_by[k, s, by] = True
    alive = player[:, :, P["is_alive"]] > 0
    spotted_by &= alive[:, :, None] & alive[:, None, :]

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

    glob = bomb_state(t["bomb"], sample_ticks, player[:, :, P["has_bomb"]], rnd, tickrate)
    glob[:, G["time_since_freeze_end"]] = (sample_ticks - freeze_end) / tickrate

    # Score: the game's own round counter at the first live tick, held for the whole round
    # (it counts this round's result the moment the round ends, so later ticks would leak it).
    first = state.filter(pl.col("step") == 0)
    side = np.zeros((T, 2, len(SIDE_FEATURES)), np.float32)
    for side_idx in range(2):
        block = player[:, side_idx * SLOTS_PER_SIDE:(side_idx + 1) * SLOTS_PER_SIDE]
        on_side = first.filter((pl.col("slot") // SLOTS_PER_SIDE) == side_idx)["team_rounds_total"].drop_nulls()
        if on_side.is_empty() or on_side.n_unique() != 1:
            return f"no single round counter for side {SIDES[side_idx]} ({on_side.unique().to_list()})"
        side[:, side_idx, S["score"]] = on_side[0]
        side[:, side_idx, S["alive"]] = block[:, :, P["is_alive"]].sum(1)
        side[:, side_idx, S["equip_value"]] = block[:, :, P["equip_value"]].sum(1)

    utility = utility_tokens(t["smokes"], t["infernos"], throws.get(round_num, []), sample_ticks, tickrate, slot_of)

    return {
        "player": player, "player_cat": player_cat,
        "last_seen": last_seen, "seen_age": seen_age, "visible": visible, "spotted_by": spotted_by,
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


def live_ticks(tick_df: pl.DataFrame) -> pl.Series:
    """Replacement for awpy.parsers.ticks.get_valid_ticks.

    awpy drops every tick with a timeout or waiting-for-resume flag, but those
    flags switch on as soon as a timeout is *called*, which can be mid-round
    while play continues until the round ends (found in mindfreak-vs-rooster
    round 7: 15 s of live play dropped). Timeouts only take effect during freeze
    time, which is never sampled, so only pre-match and warmup ticks are dropped
    here; build_round checks the game clock for real mid-round pauses.
    """
    valid = tick_df.filter(pl.col("is_match_started") & ~pl.col("is_warmup_period"))
    return valid.select("tick").unique(subset=["tick"])["tick"].sort()


def measure_tickrate(light_ticks: pl.DataFrame, default: int = 64) -> int:
    """CS2 demos record at 64 ticks/s, but awpy assumes 128; measure it from game_time."""
    times = light_ticks.filter(pl.col("tick").is_in([2000, 6000])).group_by("tick").agg(pl.col("game_time").first()).sort("tick")
    if times.height != 2:
        return default
    rate = 4000 / (times["game_time"][1] - times["game_time"][0])
    return min((64, 128), key=lambda r: abs(r - rate))


# Player props read at every tick: what visibility (spotted / last seen), the roster and
# the in-play filter need. Everything else in AWPY_PLAYER_PROPS is only read at sample ticks.
LIGHT_PROPS = ["X", "Y", "Z", "spotted", "is_alive", "team_name", "team_clan_name",
               "game_time", "is_match_started", "is_warmup_period"]
# Player props attached to every game event: awpy's own minimum, all its event tables use.
EVENT_PLAYER_PROPS = ["last_place_name", "X", "Y", "Z", "health", "team_name"]


def parse_demo(demo, sample_seconds: float):
    """awpy's Demo.parse, reading less of the demo (same tables, same values).

    Demo.parse attaches every requested player prop to every game event and reads
    every prop at every tick, plus a second full pass for the in-play flags. Here
    events carry only the props awpy's event tables use, one light pass reads the
    per-tick props, and the heavy props (inventory, money, ...) are read only at the
    ticks that are actually sampled. Sets demo.events / rounds / tickrate /
    in_play_ticks / ticks (light) / grenades and returns the heavy sample-tick table.
    """
    import awpy.parsers.rounds
    import awpy.parsers.utils

    def in_rounds(df):
        df = df.filter(pl.col("tick").is_in(demo.in_play_ticks))
        return awpy.parsers.rounds.apply_round_num(df=df, rounds_df=demo.rounds, tick_col="tick").filter(
            pl.col("round_num").is_not_null()
        )

    demo.events = demo.parse_events(demo.default_events, player_props=EVENT_PLAYER_PROPS)
    demo.rounds = awpy.parsers.rounds.create_round_df(demo.events)
    light = demo.parse_ticks(player_props=LIGHT_PROPS)
    demo.tickrate = measure_tickrate(light)
    demo.in_play_ticks = live_ticks(light)
    demo.ticks = awpy.parsers.utils.fix_common_names(in_rounds(light)).sort("tick")
    demo.grenades = in_rounds(demo.parse_grenades())

    # Heavy props at the latest existing tick at or before each sample tick (what the
    # per-round as-of join picks), so the result matches reading them at every tick.
    # The two ticks before it are read too: demoparser2 derives velocity_X/Y/Z at a row
    # from the positions in the two rows it output before it ((X[t-1] - X[t-2]) * tickrate),
    # so without them every velocity would be wrong.
    step_ticks = round(sample_seconds * demo.tickrate)
    present = np.unique(demo.ticks["tick"].to_numpy())
    wanted = set()
    for rnd in demo.rounds.iter_rows(named=True):
        if rnd["freeze_end"] is not None and rnd["end"] is not None and rnd["end"] > rnd["freeze_end"]:
            sample = np.arange(rnd["freeze_end"], rnd["end"] + 1, step_ticks)
            idx = np.searchsorted(present, sample, side="right") - 1
            idx = idx[idx >= 0]
            for back in (0, 1, 2):
                wanted.update(present[idx[idx >= back] - back].tolist())
    heavy = pl.from_pandas(demo.parser.parse_ticks(wanted_props=AWPY_PLAYER_PROPS, ticks=sorted(wanted)))
    return awpy.parsers.utils.fix_common_names(in_rounds(heavy)).sort("tick")


def process_demo(path: Path, out_path: Path) -> dict:
    from awpy import Demo  # imported here so worker processes load it themselves

    start = time.perf_counter()
    # The durations are upper bounds awpy uses to discard unmatched end events; real
    # smokes last ~22 s and infernos ~7 s.
    demo = Demo(path, smoke_duration=SMOKE_SECONDS + 5, inferno_duration=INFERNO_SECONDS + 3)
    map_name = demo.header.get("map_name")
    if map_name != "de_mirage":
        return {"file": path.name, "skipped": f"map is {map_name}"}
    heavy = parse_demo(demo, STEP_SECONDS)
    if "player_sound" not in demo.events:  # some servers record without it, so there are no footsteps
        return {"file": path.name, "skipped": "no player_sound events"}
    tickrate = demo.tickrate
    step_ticks = round(STEP_SECONDS * tickrate)

    ticks = demo.ticks
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
            "round_num", "entity_id", "thrower_steamid", "start_tick", "X", "Y", "Z", "thrower_side",
            pl.col("end_tick").fill_null(pl.col("start_tick") + int(seconds * tickrate)),
        )
    throws = grenade_throws(demo.grenades, detonation_events(demo.events), tables["infernos"])

    weapons, places = StringTable(), StringTable()
    rounds, skipped = [], []
    for rnd in demo.rounds.sort("round_num").iter_rows(named=True):
        if rnd["freeze_end"] is None or rnd["end"] is None or rnd["end"] <= rnd["freeze_end"]:
            continue
        in_round = (
            (pl.col("round_num") == rnd["round_num"])
            & pl.col("tick").is_between(rnd["freeze_end"], rnd["end"])
            & pl.col("side").is_in(SIDES)
        )
        round_ticks = ticks.filter(in_round)
        if round_ticks.is_empty():
            continue
        built = build_round(rnd, round_ticks, heavy.filter(in_round), tables, throws, weapons, places, step_ticks, tickrate)
        if isinstance(built, str):
            skipped.append(f"round {rnd['round_num']}: {built}")
        else:
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
    return {"file": path.name, "rounds": len(rounds), "steps": int(n_steps.sum()), "seconds": time.perf_counter() - start,
            "skipped_rounds": skipped}


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
    parser.add_argument("--workers", type=int, default=6, help="demos parsed in parallel (default: 6)")
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
                    for reason in result["skipped_rounds"]:
                        print(f"    skipped {reason}")

    manifest = write_manifest(out_dir, args.val_frac, args.test_frac, args.seed)
    counts = {s: sum(d["split"] == s for d in manifest["demos"]) for s in ("train", "val", "test")}
    print(f"Manifest written: {counts} demos per split ({time.perf_counter() - start:.1f}s)")
    for split, n in counts.items():
        if n == 0:
            print(f"  warning: the {split} split is empty; add demos or change --seed / fractions")


if __name__ == "__main__":
    main()
