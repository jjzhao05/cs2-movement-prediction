"""Load processed/ (the output of build_dataset.py) into a SQLite database.

Everything is stored in raw game units, one row per thing, so it can be queried
directly instead of going through the numpy arrays:

    demos          one row per demo (file, match, split, tickrate)
    rounds         one row per round (ticks, winner, team names)
    round_players  roster: which steamid/name sits in each slot of a round
    round_steps    one row per 0.5 s step: round clock, bomb state, per-side score/alive/equipment
    player_steps   one row per (step, player): state, weapon, place, event counts, visibility
    utility        one row per active smoke/inferno per step

Slots 0-4 are the T side and 5-9 the CT side, as in the .npz files.

Usage:
    python build_sql.py                        # processed/ -> mirage.db
    python build_sql.py --processed processed --db mirage.db

Example query (where each player was at 20 s into the round):
    SELECT r.round_num, p.name, ps.x, ps.y, ps.place
    FROM player_steps ps
    JOIN rounds r USING (round_id)
    JOIN round_players p USING (round_id, slot)
    WHERE ps.step = 40;
"""

import argparse
import json
import math
import sqlite3
import time
from pathlib import Path

import numpy as np

from features import (
    GLOBAL_FEATURES,
    MAX_UTILITY,
    N_SLOTS,
    PLAYER_FEATURES,
    SIDE_FEATURES,
    SLOTS_PER_SIDE,
    STEP_SECONDS,
    U,
)

ROOT = Path(__file__).parent
SIDES = ("t", "ct")

# Player features stored as INTEGER; the rest are REAL.
INT_PLAYER_FEATURES = {
    "health", "armor", "has_helmet", "has_defuser", "is_alive", "is_scoped", "is_walking",
    "ducking", "is_defusing", "equip_value", "spotted", "has_bomb", "shots", "kills", "footsteps",
}
INT_GLOBAL_FEATURES = {"bomb_planted", "plant_site_a", "plant_site_b", "bomb_dropped"}

PLAYER_COLUMNS = [f.lower() for f in PLAYER_FEATURES]


def sql_type(name, ints):
    return "INTEGER" if name in ints else "REAL"


SCHEMA = f"""
CREATE TABLE demos (
    demo_id     INTEGER PRIMARY KEY,
    file        TEXT NOT NULL UNIQUE,
    match       TEXT NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    tickrate    INTEGER NOT NULL,
    n_rounds    INTEGER NOT NULL,
    n_steps     INTEGER NOT NULL
);

CREATE TABLE rounds (
    round_id        INTEGER PRIMARY KEY,
    demo_id         INTEGER NOT NULL REFERENCES demos,
    round_num       INTEGER NOT NULL,
    start_tick      INTEGER NOT NULL,
    freeze_end_tick INTEGER NOT NULL,
    end_tick        INTEGER NOT NULL,
    n_steps         INTEGER NOT NULL,
    winner          TEXT CHECK (winner IN ('t', 'ct')),  -- NULL if unknown
    t_team          TEXT,
    ct_team         TEXT,
    UNIQUE (demo_id, round_num)
);

CREATE TABLE round_players (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    slot        INTEGER NOT NULL CHECK (slot BETWEEN 0 AND {N_SLOTS - 1}),
    side        TEXT NOT NULL,
    steamid     INTEGER NOT NULL,
    name        TEXT,
    PRIMARY KEY (round_id, slot)
) WITHOUT ROWID;

CREATE TABLE round_steps (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    tick        INTEGER NOT NULL,
    {",\n    ".join(f"{g} {sql_type(g, INT_GLOBAL_FEATURES)}" for g in GLOBAL_FEATURES)},
    {",\n    ".join(f"{side}_{f} INTEGER" for side in SIDES for f in SIDE_FEATURES)},
    PRIMARY KEY (round_id, step)
) WITHOUT ROWID;

CREATE TABLE player_steps (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    slot        INTEGER NOT NULL,
    {",\n    ".join(f"{c} {sql_type(f, INT_PLAYER_FEATURES)}" for c, f in zip(PLAYER_COLUMNS, PLAYER_FEATURES))},
    weapon      TEXT,
    place       TEXT,
    visible     INTEGER,  -- spotted by the enemy at some tick in this half second
    last_seen_x REAL,     -- NULL if never spotted this round
    last_seen_y REAL,
    last_seen_z REAL,
    seen_age    REAL,     -- seconds since last spotted, NULL if never
    PRIMARY KEY (round_id, step, slot)
) WITHOUT ROWID;

CREATE TABLE utility (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    token       INTEGER NOT NULL CHECK (token BETWEEN 0 AND {MAX_UTILITY - 1}),
    kind        TEXT NOT NULL CHECK (kind IN ('smoke', 'inferno')),
    thrown_by   TEXT NOT NULL,
    x REAL, y REAL, z REAL,
    age         REAL,  -- seconds since it popped
    PRIMARY KEY (round_id, step, token)
) WITHOUT ROWID;
"""

INDEXES = """
CREATE INDEX rounds_demo ON rounds (demo_id);
CREATE INDEX round_players_steamid ON round_players (steamid);
CREATE INDEX player_steps_place ON player_steps (place);
"""


def insert(conn, table, rows):
    rows = list(rows)
    if rows:
        conn.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)


def to_sql(values, ints):
    """Convert one row of numpy floats to Python ints/floats, NaN -> NULL."""
    return [None if math.isnan(v) else (int(round(v)) if i else v) for v, i in zip(values, ints)]


def load_demo(conn, demo_id, entry, z, next_round_id):
    tickrate = int(z["tickrate"])
    step_ticks = round(STEP_SECONDS * tickrate)
    weapon_strings = [s or None for s in z["weapon_strings"].tolist()]
    place_strings = [s or None for s in z["place_strings"].tolist()]
    n_steps = z["round_n_steps"]
    insert(conn, "demos", [(demo_id, entry["file"], entry["match"], entry["split"], tickrate, len(n_steps), int(n_steps.sum()))])

    player, player_cat = z["player"], z["player_cat"]
    last_seen, seen_age, visible = z["last_seen"], z["seen_age"], z["visible"]
    glob, side, util = z["globals"], z["side"], z["utility"]
    player_ints = [f in INT_PLAYER_FEATURES for f in PLAYER_FEATURES]
    global_ints = [g in INT_GLOBAL_FEATURES for g in GLOBAL_FEATURES]
    side_ints = [True] * (2 * len(SIDE_FEATURES))

    for r, offset in enumerate(z["round_offset"].tolist()):
        round_id = next_round_id + r
        n = int(n_steps[r])
        start_tick, freeze_end, end_tick = z["round_ticks"][r].tolist()
        winner = int(z["round_winner"][r])
        teams = z["round_teams"][r].tolist()
        insert(conn, "rounds", [(
            round_id, demo_id, int(z["round_num"][r]), start_tick, freeze_end, end_tick, n,
            SIDES[winner] if winner >= 0 else None, teams[0], teams[1],
        )])

        valid = z["round_slot_valid"][r]
        slots = np.flatnonzero(valid).tolist()
        insert(conn, "round_players", (
            (round_id, s, SIDES[s // SLOTS_PER_SIDE], int(z["round_steamids"][r, s]), str(z["round_names"][r, s]) or None)
            for s in slots
        ))

        rs = slice(offset, offset + n)
        g_rows, s_rows = glob[rs].tolist(), side[rs].reshape(n, -1).tolist()
        insert(conn, "round_steps", (
            [round_id, k, freeze_end + k * step_ticks] + to_sql(g_rows[k], global_ints) + to_sql(s_rows[k], side_ints)
            for k in range(n)
        ))

        p_rows, cats = player[rs].tolist(), player_cat[rs].tolist()
        ls, age, vis = last_seen[rs].tolist(), seen_age[rs].tolist(), visible[rs].tolist()

        def player_rows():
            for k in range(n):
                for s in slots:
                    never = math.isnan(age[k][s])
                    yield (
                        [round_id, k, s] + to_sql(p_rows[k][s], player_ints)
                        + [weapon_strings[cats[k][s][0]], place_strings[cats[k][s][1]], int(vis[k][s])]
                        + ([None] * 4 if never else ls[k][s] + [age[k][s]])
                    )

        insert(conn, "player_steps", player_rows())

        u = util[rs]
        ks, tokens = np.nonzero(u[..., U["valid"]] > 0)
        insert(conn, "utility", (
            (
                round_id, k, j,
                "smoke" if u[k, j, U["is_smoke"]] else "inferno",
                "t" if u[k, j, U["thrown_by_t"]] else "ct",
                float(u[k, j, U["X"]]), float(u[k, j, U["Y"]]), float(u[k, j, U["Z"]]), float(u[k, j, U["age"]]),
            )
            for k, j in zip(ks.tolist(), tokens.tolist())
        ))
    return len(n_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=str(ROOT / "processed"), help="output folder of build_dataset.py")
    parser.add_argument("--db", default=str(ROOT / "mirage.db"), help="SQLite file to (re)create")
    args = parser.parse_args()

    processed = Path(args.processed)
    manifest_path = processed / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found; run build_dataset.py first")
    manifest = json.loads(manifest_path.read_text())

    db_path = Path(args.db)
    tmp = db_path.with_suffix(".tmp.db")
    tmp.unlink(missing_ok=True)
    conn = sqlite3.connect(tmp)
    conn.execute("PRAGMA journal_mode = OFF")
    conn.execute("PRAGMA synchronous = OFF")
    conn.executescript(SCHEMA)

    start = time.perf_counter()
    round_id = 0
    for demo_id, entry in enumerate(manifest["demos"]):
        with np.load(processed / entry["file"]) as z:
            round_id += load_demo(conn, demo_id, entry, z, round_id)
        conn.commit()
        print(f"[{demo_id + 1}/{len(manifest['demos'])}] {entry['file']}: {entry['rounds']} rounds")

    conn.executescript(INDEXES)
    conn.commit()
    conn.close()
    tmp.replace(db_path)
    print(f"Wrote {db_path} ({db_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
