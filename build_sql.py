"""Load processed/ (the output of build_dataset.py) into a SQLite database.

Everything is stored in raw game units, one row per thing, so it can be queried
directly instead of going through the numpy arrays:

    demos          one row per demo (file, match, split, tickrate)
    rounds         one row per round (ticks, winner, team names)
    round_players  roster: which steamid/name sits in each slot of a round
    round_steps    one row per 0.5 s step: round clock, bomb state, per-side score/alive/equipment
    player_steps   one row per (step, player): state, economy, inventory, weapons, place,
                   event counts, visibility
    spotted_by     one row per (step, player, enemy who has them spotted)
    utility        one row per grenade token per step: in flight, just popped (flash/HE), or
                   an active smoke/inferno

Slots 0-4 are the T side and 5-9 the CT side, as in the .npz files.

The database is updated incrementally: demos in the manifest that are not in
the database yet are added, demos that left the manifest are removed, and the
split column is refreshed. Each demo is loaded in its own transaction, so an
interrupted run leaves only whole demos behind. Use --rebuild after
reprocessing demos with build_dataset.py --overwrite or changing features.py.

Usage:
    python build_sql.py                        # add new demos in processed/ to mirage.db
    python build_sql.py --rebuild              # delete mirage.db and load everything again
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
    GRENADE_ITEMS,
    GRENADE_KINDS,
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
    "money", "cash_spent", "start_equip_value", "ammo_clip", "mags_left", *GRENADE_ITEMS, "has_zeus",
    "is_airborne", "on_ladder",
}
KIND_NAMES = {kind: kind.removeprefix("is_") for kind in GRENADE_KINDS}  # is_he -> he
INT_GLOBAL_FEATURES = {"bomb_planted", "plant_site_a", "plant_site_b", "bomb_dropped"}

PLAYER_COLUMNS = [f.lower() for f in PLAYER_FEATURES]
COLUMN_SEP = ",\n    "  # backslashes are not allowed inside f-string expressions before Python 3.12


def sql_type(name, ints):
    return "INTEGER" if name in ints else "REAL"


SCHEMA = f"""
CREATE TABLE IF NOT EXISTS demos (
    demo_id     INTEGER PRIMARY KEY,
    file        TEXT NOT NULL UNIQUE,
    match       TEXT NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('train', 'val', 'test')),
    tickrate    INTEGER NOT NULL,
    n_rounds    INTEGER NOT NULL,
    n_steps     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS rounds (
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

CREATE TABLE IF NOT EXISTS round_players (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    slot        INTEGER NOT NULL CHECK (slot BETWEEN 0 AND {N_SLOTS - 1}),
    side        TEXT NOT NULL,
    steamid     INTEGER NOT NULL,
    name        TEXT,
    PRIMARY KEY (round_id, slot)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS round_steps (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    tick        INTEGER NOT NULL,
    {COLUMN_SEP.join(f"{g} {sql_type(g, INT_GLOBAL_FEATURES)}" for g in GLOBAL_FEATURES)},
    {COLUMN_SEP.join(f"{side}_{f} INTEGER" for side in SIDES for f in SIDE_FEATURES)},
    PRIMARY KEY (round_id, step)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS player_steps (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    slot        INTEGER NOT NULL,
    {COLUMN_SEP.join(f"{c} {sql_type(f, INT_PLAYER_FEATURES)}" for c, f in zip(PLAYER_COLUMNS, PLAYER_FEATURES))},
    weapon      TEXT,     -- active weapon
    place       TEXT,
    primary_weapon   TEXT,
    secondary_weapon TEXT,
    visible     INTEGER,  -- spotted by the enemy at some tick in this half second
    last_seen_x REAL,     -- NULL if never spotted this round
    last_seen_y REAL,
    last_seen_z REAL,
    seen_age    REAL,     -- seconds since last spotted, NULL if never
    PRIMARY KEY (round_id, step, slot)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS utility (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    token       INTEGER NOT NULL CHECK (token BETWEEN 0 AND {MAX_UTILITY - 1}),
    kind        TEXT NOT NULL CHECK (kind IN ({", ".join(f"'{k}'" for k in KIND_NAMES.values())})),
    in_flight   INTEGER NOT NULL,  -- 1 while thrown and not yet popped
    throw_id    INTEGER NOT NULL,  -- same id for a grenade's flight, pop and smoke/fire within a round; 0 = unknown
    thrown_by   TEXT NOT NULL,
    x REAL, y REAL, z REAL,
    vel_x REAL, vel_y REAL, vel_z REAL,
    age         REAL,  -- seconds since thrown if in flight, else since it popped
    PRIMARY KEY (round_id, step, token)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS spotted_by (
    round_id    INTEGER NOT NULL REFERENCES rounds,
    step        INTEGER NOT NULL,
    slot        INTEGER NOT NULL,  -- the player who is spotted
    by_slot     INTEGER NOT NULL,  -- the enemy who has them spotted
    PRIMARY KEY (round_id, step, slot, by_slot)
) WITHOUT ROWID;
"""



def table_columns(conn):
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
    return {t: [c[1] for c in conn.execute(f"PRAGMA table_info({t})")] for t in sorted(tables)}


with sqlite3.connect(":memory:") as _mem:  # the columns of a database built from this features.py
    _mem.executescript(SCHEMA)
    EXPECTED_COLUMNS = table_columns(_mem)

INDEXES = """
CREATE INDEX IF NOT EXISTS rounds_demo ON rounds (demo_id);
CREATE INDEX IF NOT EXISTS round_players_steamid ON round_players (steamid);
CREATE INDEX IF NOT EXISTS player_steps_place ON player_steps (place);
"""


def insert(conn, table, rows):
    rows = list(rows)
    if rows:
        conn.executemany(f"INSERT INTO {table} VALUES ({','.join('?' * len(rows[0]))})", rows)


def delete_demo(conn, demo_id):
    rounds = "SELECT round_id FROM rounds WHERE demo_id = ?"
    for table in ("utility", "spotted_by", "player_steps", "round_steps", "round_players"):
        conn.execute(f"DELETE FROM {table} WHERE round_id IN ({rounds})", (demo_id,))
    conn.execute("DELETE FROM rounds WHERE demo_id = ?", (demo_id,))
    conn.execute("DELETE FROM demos WHERE demo_id = ?", (demo_id,))


def next_id(conn, table, column):
    return conn.execute(f"SELECT COALESCE(MAX({column}), -1) + 1 FROM {table}").fetchone()[0]


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
                        + [weapon_strings[cats[k][s][0]], place_strings[cats[k][s][1]],
                           weapon_strings[cats[k][s][2]], weapon_strings[cats[k][s][3]], int(vis[k][s])]
                        + ([None] * 4 if never else ls[k][s] + [age[k][s]])
                    )

        insert(conn, "player_steps", player_rows())

        u = util[rs]
        ks, tokens = np.nonzero(u[..., U["valid"]] > 0)
        insert(conn, "utility", (
            (
                round_id, k, j,
                next(name for kind, name in KIND_NAMES.items() if u[k, j, U[kind]]),
                int(u[k, j, U["in_flight"]]),
                int(u[k, j, U["throw_id"]]),
                "t" if u[k, j, U["thrown_by_t"]] else "ct",
                *(float(u[k, j, U[f]]) for f in ("X", "Y", "Z", "vel_x", "vel_y", "vel_z", "age")),
            )
            for k, j in zip(ks.tolist(), tokens.tolist())
        ))

        insert(conn, "spotted_by", (
            (round_id, k, s, by)
            for k, s, by in zip(*(a.tolist() for a in np.nonzero(z["spotted_by"][rs])))
        ))
    return len(n_steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=str(ROOT / "processed"), help="output folder of build_dataset.py")
    parser.add_argument("--db", default=str(ROOT / "mirage.db"), help="SQLite file to create or update")
    parser.add_argument("--rebuild", action="store_true", help="delete the database and load every demo again")
    args = parser.parse_args()

    processed = Path(args.processed)
    manifest_path = processed / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"{manifest_path} not found; run build_dataset.py first")
    manifest = json.loads(manifest_path.read_text())

    db_path = Path(args.db)
    if args.rebuild:
        for suffix in ("", "-wal", "-shm"):
            Path(f"{db_path}{suffix}").unlink(missing_ok=True)
    # Autocommit mode, with one explicit transaction per demo. WAL keeps the
    # database intact if a run is interrupted halfway through a demo.
    conn = sqlite3.connect(db_path, isolation_level=None)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.executescript(SCHEMA)
    if table_columns(conn) != EXPECTED_COLUMNS:
        conn.close()
        raise SystemExit(f"{db_path} was built with an older features.py; rerun with --rebuild")

    start = time.perf_counter()
    wanted = {entry["file"]: entry for entry in manifest["demos"]}
    loaded = dict(conn.execute("SELECT file, demo_id FROM demos"))

    stale = [file for file in loaded if file not in wanted]
    for file in stale:
        conn.execute("BEGIN")
        delete_demo(conn, loaded[file])
        conn.execute("COMMIT")
        print(f"Removed {file} (no longer in the manifest)")

    # Splits are a hash of the match name, so they only change with --seed or the fractions.
    conn.executemany(
        "UPDATE demos SET split = ? WHERE file = ? AND split != ?",
        [(wanted[file]["split"], file, wanted[file]["split"]) for file in loaded if file in wanted],
    )

    new = [entry for entry in manifest["demos"] if entry["file"] not in loaded]
    demo_id, round_id = next_id(conn, "demos", "demo_id"), next_id(conn, "rounds", "round_id")
    for i, entry in enumerate(new):
        with np.load(processed / entry["file"]) as z:
            conn.execute("BEGIN")
            round_id += load_demo(conn, demo_id, entry, z, round_id)
            conn.execute("COMMIT")
        demo_id += 1
        print(f"[{i + 1}/{len(new)}] {entry['file']}: {entry['rounds']} rounds")

    conn.executescript(INDEXES)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    total = conn.execute("SELECT COUNT(*) FROM demos").fetchone()[0]
    conn.close()
    print(
        f"Updated {db_path}: +{len(new)} / -{len(stale)} demos, {total} total "
        f"({db_path.stat().st_size / 1e6:.1f} MB, {time.perf_counter() - start:.1f}s)"
    )


if __name__ == "__main__":
    main()
