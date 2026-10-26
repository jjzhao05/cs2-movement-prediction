"""Feature layout shared by build_dataset.py (writer) and dataset.py (reader).

Arrays are stored per demo in raw game units; dataset.py normalizes them and
applies team-perspective masking when building each training window.
"""

STEP_SECONDS = 0.5  # 2 Hz sampling
HORIZON_SECONDS = 10.0  # predict the next 10 s
N_SLOTS = 10  # slots 0-4 are the T side, 5-9 the CT side
SLOTS_PER_SIDE = 5
MAX_UTILITY = 10  # active smokes/infernos tracked per step

# Player props requested from awpy (it renames armor_value -> armor,
# last_place_name -> place, team_name -> side).
AWPY_PLAYER_PROPS = [
    "X", "Y", "Z",
    "velocity_X", "velocity_Y", "velocity_Z",
    "pitch", "yaw",
    "health", "armor_value", "has_helmet", "has_defuser",
    "is_alive", "flash_duration", "is_scoped", "is_walking", "ducking", "is_defusing",
    "current_equip_value", "spotted", "active_weapon_name",
    "last_place_name", "team_name", "team_clan_name",
]

# Continuous per-player features, stored as player[step, slot, feature].
PLAYER_FEATURES = [
    "X", "Y", "Z",
    "vel_x", "vel_y", "vel_z",
    "yaw", "pitch",
    "health", "armor", "has_helmet", "has_defuser",
    "is_alive", "flash_duration", "is_scoped", "is_walking", "ducking", "is_defusing",
    "equip_value",
    "spotted",  # currently seen by at least one enemy
    "has_bomb",
    # Event counts within the half-second ending at this step
    "shots", "dmg_dealt", "dmg_taken", "kills", "footsteps",
]
P = {name: i for i, name in enumerate(PLAYER_FEATURES)}

# Categorical per-player features, stored as player_cat[step, slot, feature].
PLAYER_CATEGORICALS = ["weapon", "place"]

# Round-level features that do not depend on side.
GLOBAL_FEATURES = [
    "time_since_freeze_end",  # seconds
    "bomb_planted",
    "time_since_plant",  # seconds, 0 if not planted
    "plant_site_a",
    "plant_site_b",
    "bomb_dropped",  # only the T side can see a dropped bomb
    "bomb_x", "bomb_y", "bomb_z",  # dropped or planted bomb position
]
G = {name: i for i, name in enumerate(GLOBAL_FEATURES)}

# Per-side features, stored as side[step, side (0=T, 1=CT), feature].
SIDE_FEATURES = ["score", "alive", "equip_value"]
S = {name: i for i, name in enumerate(SIDE_FEATURES)}

# Active smoke/inferno tokens, stored as utility[step, token, feature].
UTILITY_FEATURES = ["X", "Y", "Z", "age", "is_smoke", "is_inferno", "thrown_by_t", "valid"]
U = {name: i for i, name in enumerate(UTILITY_FEATURES)}
