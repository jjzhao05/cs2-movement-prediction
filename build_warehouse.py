"""Extract processed/*.npz (output of build_dataset.py) into Parquet tables for DuckDB.

Writes warehouse/<table>/<demo>.parquet for each demo, all in raw game units:

    rounds         one row per round: ticks, winner, teams, score, plant, economy, first death
    round_players  one row per (round, player): roster plus round totals (kills, damage, ...)
    round_steps    one row per 0.5 s step: round clock, bomb state, per-side score/alive/equipment
    player_steps   one row per (step, player): every player feature, weapon/place names,
                   visibility, plus speed, distance moved and nearest teammate/enemy
    spotted_by     one row per (step, player, enemy who has them spotted)
    utility        one row per active grenade token per step

Every table carries `demo` (the .npz file stem) and `round_num`; step-level tables
also carry `step`, and player-level tables `slot` (0-4 = T, 5-9 = CT).

Usage:
    python build_warehouse.py                              # processed/ -> warehouse/
    python build_warehouse.py --processed path/to/processed --workers 8
    python build_warehouse.py --overwrite                  # redo demos already extracted
"""

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).parent
SIDES = np.array(["t", "ct"], dtype=object)
SLOTS_PER_SIDE = 5
N_SLOTS = 10
STEP_SECONDS = 0.5
BUY_SECONDS = 5.0  # team equipment is measured this long after freeze end, once buys are done
TABLES = ("rounds", "round_players", "round_steps", "player_steps", "spotted_by", "utility")

# Player features stored as integers; the rest stay float32.
INT_FEATURES = {
    "health", "armor", "has_helmet", "has_defuser", "is_alive", "is_scoped", "is_walking",
    "ducking", "is_defusing", "equip_value", "spotted", "has_bomb", "shots", "kills", "footsteps",
    "money", "cash_spent", "start_equip_value", "ammo_clip", "mags_left", "n_flash", "n_smoke",
    "n_he", "n_molotov", "n_decoy", "has_zeus", "is_airborne", "zoom_lvl", "on_ladder",
}
KINDS = {"is_smoke": "smoke", "is_inferno": "inferno", "is_flash": "flash", "is_he": "he", "is_decoy": "decoy"}


def strings(table, idx):
    """Look up per-demo string ids; id 0 (missing) becomes null."""
    lookup = np.array([s or None for s in table.tolist()], dtype=object)
    return pa.array(lookup[idx], type=pa.string())


def nan_to_null(a):
    return pa.array(a, mask=np.isnan(a))


def feature_column(name, values):
    if name in INT_FEATURES:
        return pa.array(np.rint(values).astype(np.int32))
    return pa.array(values.astype(np.float32))


def nearest(dist, alive, mask):
    """Distance to the nearest alive player in `mask` ([10, 10] slot pairs); NaN if none or dead."""
    d = np.where(mask[None] & alive[:, None, :], dist, np.inf).min(2)
    d[~alive | np.isinf(d)] = np.nan
    return d.astype(np.float32)


def demo_tables(raw, demo, names):
    pf, gf, sf, uf = (names[k] for k in ("player_features", "global_features", "side_features", "utility_features"))
    P = {n: i for i, n in enumerate(pf)}
    U = {n: i for i, n in enumerate(uf)}
    player, side, glob = raw["player"], raw["side"], raw["globals"]
    offsets, n_steps = raw["round_offset"], raw["round_n_steps"]
    R, T = len(n_steps), len(player)
    tickrate = int(raw["tickrate"])
    step_ticks = round(STEP_SECONDS * tickrate)

    # Index arrays for the whole demo at once.
    round_of_step = np.repeat(np.arange(R), n_steps)
    step_in_round = np.arange(T) - offsets[round_of_step]
    round_num = raw["round_num"]
    slot_valid = raw["round_slot_valid"][round_of_step]  # [T, 10]
    alive = (player[:, :, P["is_alive"]] > 0) & slot_valid
    slot_side = np.arange(N_SLOTS) // SLOTS_PER_SIDE

    # ------------------------------------------------------------ derived per-step player columns
    xyz = player[:, :, [P["X"], P["Y"], P["Z"]]]
    speed = np.hypot(player[:, :, P["vel_x"]], player[:, :, P["vel_y"]]).astype(np.float32)
    moved = np.full((T, N_SLOTS), np.nan, np.float32)
    moved[1:] = np.linalg.norm(xyz[1:] - xyz[:-1], axis=-1)
    moved[offsets] = np.nan  # first step of each round has no previous step
    moved[1:][~(alive[1:] & alive[:-1])] = np.nan
    dist = np.linalg.norm(xyz[:, :, None] - xyz[:, None, :], axis=-1)  # [T, 10, 10]
    same_side = slot_side[:, None] == slot_side[None, :]
    near_mate = nearest(dist, alive, same_side & ~np.eye(N_SLOTS, dtype=bool))
    near_enemy = nearest(dist, alive, ~same_side)
    spotted_by = raw["spotted_by"]  # [T, slot, by_slot]: by_slot has slot spotted
    n_spotted_by = spotted_by.sum(2).astype(np.int8)
    n_spotting = spotted_by.sum(1).astype(np.int8)

    # ------------------------------------------------------------ player_steps
    t, s = np.nonzero(slot_valid)
    cats, cat_names = raw["player_cat"], names["player_categoricals"]
    ps = {
        "demo": pa.array([demo] * len(t)),
        "round_num": pa.array(round_num[round_of_step[t]]),
        "step": pa.array(step_in_round[t].astype(np.int32)),
        "slot": pa.array(s.astype(np.int8)),
        "side": pa.array(SIDES[slot_side[s]], type=pa.string()),
    }
    for name, i in P.items():
        ps[name.lower()] = feature_column(name, player[t, s, i])
    for j, name in enumerate(cat_names):
        table = raw["place_strings"] if name == "place" else raw["weapon_strings"]
        ps[name] = strings(table, cats[t, s, j])
    ps["visible"] = pa.array(raw["visible"][t, s].astype(np.int8))
    never_seen = np.isnan(raw["seen_age"][t, s])
    for k, axis in enumerate("xyz"):
        ps[f"last_seen_{axis}"] = pa.array(raw["last_seen"][t, s, k], mask=never_seen)
    ps["seen_age"] = nan_to_null(raw["seen_age"][t, s])
    ps["speed"] = pa.array(speed[t, s])
    ps["dist_moved"] = nan_to_null(moved[t, s])
    ps["nearest_teammate"] = nan_to_null(near_mate[t, s])
    ps["nearest_enemy"] = nan_to_null(near_enemy[t, s])
    ps["n_spotted_by"] = pa.array(n_spotted_by[t, s])
    ps["n_spotting"] = pa.array(n_spotting[t, s])

    # ------------------------------------------------------------ round_steps
    rs = {
        "demo": pa.array([demo] * T),
        "round_num": pa.array(round_num[round_of_step]),
        "step": pa.array(step_in_round.astype(np.int32)),
        "tick": pa.array((raw["round_ticks"][round_of_step, 1] + step_in_round * step_ticks).astype(np.int64)),
    }
    for i, name in enumerate(gf):
        rs[name] = pa.array(glob[:, i].astype(np.float32))
    for k, side_name in enumerate(SIDES):
        for i, name in enumerate(sf):
            rs[f"{side_name}_{name}"] = pa.array(np.rint(side[:, k, i]).astype(np.int32))
    rs["man_advantage"] = pa.array(np.rint(side[:, 0, sf.index("alive")] - side[:, 1, sf.index("alive")]).astype(np.int8))

    # ------------------------------------------------------------ spotted_by (sparse)
    st, ss, sb = np.nonzero(spotted_by)
    sp = {
        "demo": pa.array([demo] * len(st)),
        "round_num": pa.array(round_num[round_of_step[st]]),
        "step": pa.array(step_in_round[st].astype(np.int32)),
        "slot": pa.array(ss.astype(np.int8)),
        "by_slot": pa.array(sb.astype(np.int8)),
    }

    # ------------------------------------------------------------ utility (valid tokens only)
    util = raw["utility"]
    ut, uk = np.nonzero(util[:, :, U["valid"]] > 0)
    u = util[ut, uk]
    kind_cols = [U[k] for k in KINDS]
    kind = np.array(list(KINDS.values()), dtype=object)[u[:, kind_cols].argmax(1)]
    ud = {
        "demo": pa.array([demo] * len(ut)),
        "round_num": pa.array(round_num[round_of_step[ut]]),
        "step": pa.array(step_in_round[ut].astype(np.int32)),
        "token": pa.array(uk.astype(np.int8)),
        "throw_id": pa.array(u[:, U["throw_id"]].astype(np.int32)),
        "kind": pa.array(kind, type=pa.string()),
        "thrown_by": pa.array(np.where(u[:, U["thrown_by_t"]] > 0, "t", "ct").astype(object), type=pa.string()),
        "in_flight": pa.array(u[:, U["in_flight"]].astype(np.int8)),
    }
    for name in ("X", "Y", "Z", "vel_x", "vel_y", "vel_z", "age"):
        ud[name.lower()] = pa.array(u[:, U[name]].astype(np.float32))

    # ------------------------------------------------------------ rounds and round_players
    buy_step = int(round(BUY_SECONDS / STEP_SECONDS))
    rounds, players = [], []
    for r in range(R):
        o, n = int(offsets[r]), int(n_steps[r])
        buy = o + min(buy_step, n - 1)
        start_tick, freeze_end, end_tick = raw["round_ticks"][r].tolist()
        winner = int(raw["round_winner"][r])
        valid_r = raw["round_slot_valid"][r]
        alive_r = alive[o:o + n]
        died = alive_r[0] & ~alive_r[-1]
        death_step = np.where(died, np.argmin(alive_r, 0), -1)  # first step not alive
        planted = glob[o:o + n, gf.index("bomb_planted")] > 0
        plant_step = int(np.argmax(planted)) if planted.any() else None
        site = None
        if plant_step is not None:
            site = "a" if glob[o + plant_step, gf.index("plant_site_a")] > 0 else (
                "b" if glob[o + plant_step, gf.index("plant_site_b")] > 0 else None)
        first_death_side, first_death_s = None, None
        if died.any():
            first = int(death_step[died].min())
            victims = np.flatnonzero(died & (death_step == first))
            first_death_side = SIDES[slot_side[victims[0]]] if len(set(slot_side[victims])) == 1 else None
            first_death_s = first * STEP_SECONDS
        teams = raw["round_teams"][r].tolist()

        row = {
            "demo": demo, "round_num": int(round_num[r]), "tickrate": tickrate,
            "start_tick": start_tick, "freeze_end_tick": freeze_end, "end_tick": end_tick, "n_steps": n,
            "duration_s": (end_tick - freeze_end) / tickrate,
            "winner": SIDES[winner] if winner >= 0 else None,
            "t_team": teams[0], "ct_team": teams[1],
            "bomb_planted": int(plant_step is not None),
            "plant_site": site,
            "plant_s": None if plant_step is None else plant_step * STEP_SECONDS,
            "first_death_side": first_death_side, "first_death_s": first_death_s,
        }
        for k, side_name in enumerate(SIDES):
            block = slice(k * SLOTS_PER_SIDE, (k + 1) * SLOTS_PER_SIDE)
            row[f"{side_name}_score"] = int(side[o, k, sf.index("score")])
            row[f"{side_name}_players"] = int(valid_r[block].sum())
            row[f"{side_name}_money_start"] = int(player[o, block, P["money"]][valid_r[block]].sum())
            row[f"{side_name}_equip"] = int(side[buy, k, sf.index("equip_value")])
            row[f"{side_name}_cash_spent"] = int(player[o + n - 1, block, P["cash_spent"]][valid_r[block]].sum())
            row[f"{side_name}_kills"] = int(player[o:o + n, block, P["kills"]].sum())
            row[f"{side_name}_survivors"] = int(alive_r[-1, block].sum())
        rounds.append(row)

        totals = player[o:o + n].sum(0)  # [10, F] round totals of the per-step event counts
        moved_r = np.nansum(moved[o + 1:o + n], 0)
        for slot in np.flatnonzero(valid_r):
            players.append({
                "demo": demo, "round_num": int(round_num[r]), "slot": int(slot),
                "side": SIDES[slot_side[slot]],
                "steamid": int(raw["round_steamids"][r, slot]),
                "name": str(raw["round_names"][r, slot]) or None,
                "team": teams[slot_side[slot]],
                "kills": int(totals[slot, P["kills"]]),
                "shots": int(totals[slot, P["shots"]]),
                "dmg_dealt": float(totals[slot, P["dmg_dealt"]]),
                "dmg_taken": float(totals[slot, P["dmg_taken"]]),
                "footsteps": int(totals[slot, P["footsteps"]]),
                "died": int(died[slot]),
                "death_s": None if death_step[slot] < 0 else float(death_step[slot] * STEP_SECONDS),
                "money_start": int(player[o, slot, P["money"]]),
                "equip_value": int(player[buy, slot, P["equip_value"]]),
                "cash_spent": int(player[o + n - 1, slot, P["cash_spent"]]),
                "primary": strings(raw["weapon_strings"], cats[buy, slot, cat_names.index("primary")][None])[0].as_py(),
                "dist_travelled": float(moved_r[slot]),
            })

    return {
        "rounds": pa.Table.from_pylist(rounds),
        "round_players": pa.Table.from_pylist(players),
        "round_steps": pa.table(rs),
        "player_steps": pa.table(ps),
        "spotted_by": pa.table(sp),
        "utility": pa.table(ud),
    }


def extract_demo(npz_path, out_dir, names):
    start = time.perf_counter()
    with np.load(npz_path) as z:
        raw = {k: z[k] for k in z.files}
    tables = demo_tables(raw, npz_path.stem, names)
    for name, table in tables.items():
        path = out_dir / name / f"{npz_path.stem}.parquet"
        tmp = path.with_suffix(".tmp")
        pq.write_table(table, tmp, compression="zstd")
        tmp.replace(path)
    return {"file": npz_path.name, "rows": tables["player_steps"].num_rows, "seconds": time.perf_counter() - start}


def extract_demo_safe(npz_path, out_dir, names):
    try:
        return extract_demo(npz_path, out_dir, names)
    except Exception:
        return {"file": npz_path.name, "error": traceback.format_exc()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=str(ROOT / "processed"), help="output folder of build_dataset.py")
    parser.add_argument("--out", default=str(ROOT / "warehouse"), help="Parquet output folder")
    parser.add_argument("--workers", type=int, default=8, help="demos extracted in parallel")
    parser.add_argument("--overwrite", action="store_true", help="redo demos that are already extracted")
    args = parser.parse_args()

    processed, out_dir = Path(args.processed), Path(args.out)
    manifest = json.loads((processed / "manifest.json").read_text())
    names = {k: manifest[k] for k in ("player_features", "player_categoricals", "global_features", "side_features", "utility_features")}
    for table in TABLES:
        (out_dir / table).mkdir(parents=True, exist_ok=True)

    # Demo list with splits, rewritten every run so it follows the manifest.
    pq.write_table(pa.Table.from_pylist([
        {"demo": Path(d["file"]).stem, "match": d["match"], "split": d["split"], "rounds": d["rounds"], "steps": d["steps"]}
        for d in manifest["demos"]
    ]), out_dir / "demos.parquet")

    def done(stem):
        return all((out_dir / t / f"{stem}.parquet").exists() for t in TABLES)

    todo = [processed / d["file"] for d in manifest["demos"] if args.overwrite or not done(Path(d["file"]).stem)]
    print(f"{len(manifest['demos'])} demo(s) in manifest, {len(todo)} to extract")
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(extract_demo_safe, path, out_dir, names) for path in todo]
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            prefix = f"[{i}/{len(todo)}] {result['file']}"
            if "error" in result:
                print(f"{prefix}: FAILED\n{result['error']}")
            else:
                print(f"{prefix}: {result['rows']} player rows in {result['seconds']:.1f}s")
    print(f"Done in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()