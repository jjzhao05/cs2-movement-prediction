"""Feature layout shared by build_dataset.py (writer) and dataset.py (reader).

Arrays are stored per demo in raw game units; dataset.py normalizes them and
applies team-perspective masking when building each training window.
"""

STEP_SECONDS = 0.5  # 2 Hz sampling
HORIZON_SECONDS = 10.0  # predict the next 10 s
N_SLOTS = 10  # slots 0-4 are the T side, 5-9 the CT side
SLOTS_PER_SIDE = 5
MAX_UTILITY = 16  # grenades in flight, recent pops and active smokes/infernos tracked per step
POP_SECONDS = 1.0  # flash/HE detonations stay visible as tokens for this long

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
    "balance", "cash_spent_this_round", "round_start_equip_value",
    "active_weapon_ammo", "total_ammo_left",
    "is_airborne", "duck_amount", "zoom_lvl", "move_type",
    "inventory", "approximate_spotted_by",
    "game_time",  # used to detect mid-round pauses, not stored
    "team_rounds_total",  # the team's score, read at the first live tick of each round
]
# Highest score a team can reach: regulation ends 12-12 at worst, each tied MR3
# overtime adds 3 per team, so after 4 tied overtimes it is 24-24 and the 5th is won at 28.
MAX_SCORE = 28
MOVE_TYPE_LADDER = 9  # demoparser2 move_type while climbing a ladder

# Inventory item names (as demoparser2 reports them) counted per player.
GRENADE_ITEMS = {
    "n_flash": ("Flashbang",),
    "n_smoke": ("Smoke Grenade",),
    "n_he": ("High Explosive Grenade",),
    "n_molotov": ("Molotov", "Incendiary Grenade"),
    "n_decoy": ("Decoy Grenade",),
}
PISTOLS = (
    "Glock-18", "USP-S", "P2000", "P250", "Five-SeveN", "Tec-9", "CZ75-Auto",
    "Desert Eagle", "Dual Berettas", "R8 Revolver",
)
NON_PRIMARY = PISTOLS + sum(GRENADE_ITEMS.values(), ()) + ("C4 Explosive", "Zeus x27")  # plus every knife

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
    # Economy and inventory
    "money", "cash_spent", "start_equip_value",
    "ammo_clip",  # rounds in the active weapon's magazine
    "mags_left",  # spare magazines for the active weapon
    *GRENADE_ITEMS,  # grenades carried, by type
    "has_zeus",
    # Movement state
    "is_airborne", "duck_amount", "zoom_lvl", "on_ladder",
]
P = {name: i for i, name in enumerate(PLAYER_FEATURES)}

# Categorical per-player features, stored as player_cat[step, slot, feature]. All
# but "place" index the per-demo weapon string table.
PLAYER_CATEGORICALS = ["weapon", "place", "primary", "secondary"]

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

# Utility tokens, stored as utility[step, token, feature]: grenades in flight
# (in_flight=1, age = seconds since thrown), flash/HE detonations of the last
# POP_SECONDS and active smokes/infernos (age = seconds since they popped).
# Molotov and incendiary both count as is_inferno.
UTILITY_FEATURES = [
    "X", "Y", "Z", "vel_x", "vel_y", "vel_z", "age",
    "is_smoke", "is_inferno", "is_flash", "is_he", "is_decoy",
    "in_flight", "thrown_by_t", "valid",
    "throw_id",  # 1, 2, ... per round in throw order, shared by a grenade's flight, pop and smoke/fire; 0 = unknown
]
U = {name: i for i, name in enumerate(UTILITY_FEATURES)}
INFERNO_LINK_TICKS = 16  # a fire belongs to the thrower's molotov that burst within this many ticks
GRENADE_KINDS = ("is_smoke", "is_inferno", "is_flash", "is_he", "is_decoy")
PROJECTILE_KIND = {  # awpy grenades.grenade_type of thrown grenades
    "CSmokeGrenadeProjectile": "is_smoke",
    "CMolotovProjectile": "is_inferno",
    "CFlashbangProjectile": "is_flash",
    "CHEGrenadeProjectile": "is_he",
    "CDecoyProjectile": "is_decoy",
}

# Who spots whom, stored as spotted_by[step, slot, by_slot] (True if by_slot's
# player has slot's player spotted at the sample tick).
