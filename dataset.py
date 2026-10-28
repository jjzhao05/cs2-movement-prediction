"""PyTorch dataset of team-perspective prediction windows built from processed/.

One example = (demo, round, step t, POV side). A new prediction point is made
every 0.5 s. The input is the round history up to t (full round, or the last
`history_seconds`), shown from the POV team's point of view: teammates are
always visible, but an enemy's state is only shown while the team has it
spotted; otherwise the model gets the enemy's last known position and how long
ago it was seen. The target is the next 10 s (20 steps) of X/Y for each POV
teammate, in game units, masked after death or round end.

Example:
    from torch.utils.data import DataLoader
    from dataset import MirageWindows, collate_windows

    train = MirageWindows("processed", split="train", history_seconds=None)  # full round
    loader = DataLoader(train, batch_size=64, shuffle=True, collate_fn=collate_windows)
    batch = next(iter(loader))
    batch["players"].shape   # [B, H, 10, F]  slots 0-4 = own team, 5-9 = enemies
    batch["target_xy"].shape # [B, 20, 5, 2]

Pass mask_enemies=False for the full-information ablation and
history_seconds=3 or 10 for the short-context models.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from features import (
    HORIZON_SECONDS,
    MAX_SCORE,
    MAX_UTILITY,
    N_SLOTS,
    PLAYER_CATEGORICALS,
    SLOTS_PER_SIDE,
    G,
    P,
    S,
    U,
)

# Normalized per-player inputs: PLAYER_FEATURES with yaw split into sin/cos,
# followed by the perspective features added per window.
PLAYER_INPUTS = [
    "x", "y", "z", "vel_x", "vel_y", "vel_z", "yaw_sin", "yaw_cos", "pitch",
    "health", "armor", "has_helmet", "has_defuser", "is_alive", "flash_duration",
    "is_scoped", "is_walking", "ducking", "is_defusing", "equip_value", "spotted",
    "has_bomb", "shots", "dmg_dealt", "dmg_taken", "kills", "footsteps",
    "money", "cash_spent", "start_equip_value", "ammo_clip", "mags_left",
    "n_flash", "n_smoke", "n_he", "n_molotov", "n_decoy", "has_zeus",
    "is_airborne", "duck_amount", "zoom_lvl", "on_ladder",
    "is_teammate", "visible", "ever_seen", "seen_age",
]
GLOBAL_INPUTS = [
    "time_since_freeze_end", "bomb_planted", "time_since_plant", "plant_site_a", "plant_site_b",
    "bomb_dropped", "bomb_x", "bomb_y", "bomb_z",
    "own_score", "enemy_score", "own_alive", "enemy_alive", "own_equip", "enemy_equip",
    "pov_is_t",
]
UTILITY_INPUTS = [
    "x", "y", "z", "vel_x", "vel_y", "vel_z", "age",
    "is_smoke", "is_inferno", "is_flash", "is_he", "is_decoy",
    "in_flight", "own_team", "valid",
]  # throw_id is returned separately as "utility_ids" (renumbered per window, 0 = none)

N_BASE = len(PLAYER_INPUTS) - 4
ALIVE = PLAYER_INPUTS.index("is_alive")
TEAMMATE = PLAYER_INPUTS.index("is_teammate")
# Enemy features a team never sees, even with the enemy in view: economy and inventory.
PRIVATE = [PLAYER_INPUTS.index(f) for f in (
    "money", "cash_spent", "start_equip_value", "ammo_clip", "mags_left",
    "n_flash", "n_smoke", "n_he", "n_molotov", "n_decoy", "has_zeus",
)]
PRIVATE_CATS = [PLAYER_CATEGORICALS.index("primary"), PLAYER_CATEGORICALS.index("secondary")]


class MirageWindows(Dataset):
    def __init__(
        self,
        processed_dir="processed",
        split="train",
        history_seconds=None,
        horizon_seconds=HORIZON_SECONDS,
        pov="both",
        mask_enemies=True,
    ):
        """
        Args:
            processed_dir: output folder of build_dataset.py.
            split: "train", "val", "test", or a list of them.
            history_seconds: None for the full round so far, else e.g. 3 or 10.
            horizon_seconds: how far ahead to predict (10 s = 20 steps).
            pov: "both", "t" or "ct" - which team's perspective to build windows from.
            mask_enemies: False gives the full-information ablation.
        """
        self.dir = Path(processed_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        step = self.manifest["step_seconds"]
        self.history_steps = None if history_seconds is None else int(round(history_seconds / step)) + 1
        self.horizon_steps = int(round(horizon_seconds / step))
        self.mask_enemies = mask_enemies
        self.pos_mean = np.array(self.manifest["pos_mean"], np.float32)
        self.pos_std = np.float32(self.manifest["pos_std"])
        self.weapon_vocab = self.manifest["weapon_vocab"]
        self.place_vocab = self.manifest["place_vocab"]

        splits = {split} if isinstance(split, str) else set(split)
        povs = {"both": (0, 1), "t": (0,), "ct": (1,)}[pov]
        self.demos = []
        index = []
        for entry in self.manifest["demos"]:
            if entry["split"] not in splits:
                continue
            demo_idx = len(self.demos)
            demo = self._load(self.dir / entry["file"])
            demo["file"] = entry["file"]
            self.demos.append(demo)
            index.append(self._windows(demo, demo_idx, povs))
        if not self.demos:
            raise ValueError(f"No demos in split {split!r}; run build_dataset.py first")
        self.index = np.concatenate(index)  # columns: demo, round, step, pov

    # ------------------------------------------------------------------ loading

    def _load(self, path):
        with np.load(path) as z:
            raw = {k: z[k] for k in z.files}
        p = raw["player"]
        player = np.zeros(p.shape[:2] + (N_BASE,), np.float32)
        player[..., 0:2] = (p[..., [P["X"], P["Y"]]] - self.pos_mean) / self.pos_std
        player[..., 2] = p[..., P["Z"]] / self.pos_std
        player[..., 3:6] = p[..., [P["vel_x"], P["vel_y"], P["vel_z"]]] / 250.0
        yaw = np.deg2rad(p[..., P["yaw"]])
        player[..., 6], player[..., 7] = np.sin(yaw), np.cos(yaw)
        player[..., 8] = p[..., P["pitch"]] / 90.0
        player[..., 9] = p[..., P["health"]] / 100.0
        player[..., 10] = p[..., P["armor"]] / 100.0
        scale = {
            "flash_duration": 5.0, "equip_value": 5000.0, "dmg_dealt": 100.0, "dmg_taken": 100.0,
            "money": 16000.0, "cash_spent": 16000.0, "start_equip_value": 5000.0,
            "ammo_clip": 30.0, "mags_left": 4.0, "zoom_lvl": 2.0,
        }
        for i, name in enumerate(PLAYER_INPUTS[11:N_BASE], start=11):
            player[..., i] = p[..., P[name]] / scale.get(name, 1.0)

        # Remap per-demo string ids onto the global vocabularies.
        cat = raw["player_cat"].astype(np.int64)
        for j, name in enumerate(PLAYER_CATEGORICALS):
            strings, vocab = (raw["place_strings"], self.place_vocab) if name == "place" else (raw["weapon_strings"], self.weapon_vocab)
            lookup = {s: i for i, s in enumerate(vocab)}
            remap = np.array([lookup.get(s, 0) if s else 0 for s in strings], np.int64)
            cat[..., j] = remap[cat[..., j]]

        last_seen = np.zeros(raw["last_seen"].shape, np.float32)
        last_seen[..., 0:2] = (raw["last_seen"][..., 0:2] - self.pos_mean) / self.pos_std
        last_seen[..., 2] = raw["last_seen"][..., 2] / self.pos_std
        ever_seen = ~np.isnan(raw["seen_age"])
        # log scale without a cap: 0.5 s -> 0.08, 10 s -> 0.5, 115 s (a whole round) -> 1.0
        seen_age = np.where(ever_seen, np.log1p(np.nan_to_num(raw["seen_age"])) / np.log1p(115.0), 0.0).astype(np.float32)

        g = raw["globals"]
        glob = np.zeros((len(g), 9), np.float32)
        glob[:, 0] = g[:, G["time_since_freeze_end"]] / 115.0
        glob[:, 1] = g[:, G["bomb_planted"]]
        glob[:, 2] = g[:, G["time_since_plant"]] / 40.0
        glob[:, 3] = g[:, G["plant_site_a"]]
        glob[:, 4] = g[:, G["plant_site_b"]]
        glob[:, 5] = g[:, G["bomb_dropped"]]
        glob[:, 6:8] = (g[:, [G["bomb_x"], G["bomb_y"]]] - self.pos_mean) / self.pos_std
        glob[:, 8] = g[:, G["bomb_z"]] / self.pos_std
        known_bomb = (g[:, G["bomb_dropped"]] + g[:, G["bomb_planted"]]) > 0
        glob[~known_bomb, 6:9] = 0

        side = raw["side"].copy()
        side[..., S["score"]] /= MAX_SCORE  # in [0, 1] up to five overtimes
        side[..., S["alive"]] /= 5.0
        side[..., S["equip_value"]] /= 25000.0

        u = raw["utility"]
        util = u.copy()  # same layout as UTILITY_FEATURES; thrown_by_t becomes own_team per window
        util[..., [U["X"], U["Y"]]] = (u[..., [U["X"], U["Y"]]] - self.pos_mean) / self.pos_std
        util[..., U["Z"]] = u[..., U["Z"]] / self.pos_std
        util[..., [U["vel_x"], U["vel_y"], U["vel_z"]]] = u[..., [U["vel_x"], U["vel_y"], U["vel_z"]]] / 1000.0
        util[..., U["age"]] = u[..., U["age"]] / 20.0
        util[u[..., U["valid"]] == 0, : U["age"] + 1] = 0

        return {
            "player": player,
            "player_cat": cat,
            "last_seen": last_seen,
            "ever_seen": ever_seen,
            "seen_age": seen_age,
            "visible": raw["visible"],
            "spotted_by": raw["spotted_by"],
            "globals": glob,
            "side": side,
            "utility": util,
            "xy": p[..., [P["X"], P["Y"]]].copy(),  # raw game units for targets
            "alive": p[..., P["is_alive"]] > 0,
            "round_offset": raw["round_offset"],
            "round_n_steps": raw["round_n_steps"],
            "round_num": raw["round_num"],
            "slot_valid": raw["round_slot_valid"],
        }

    def _windows(self, demo, demo_idx, povs):
        rows = []
        for r, (offset, n) in enumerate(zip(demo["round_offset"], demo["round_n_steps"])):
            steps = np.arange(n - 1)  # need at least one future step
            for pov in povs:
                own = slice(pov * SLOTS_PER_SIDE, (pov + 1) * SLOTS_PER_SIDE)
                alive = demo["alive"][offset:offset + n - 1, own] & demo["slot_valid"][r, own]
                keep = steps[alive.any(1)]
                rows.append(np.stack([np.full_like(keep, demo_idx), np.full_like(keep, r), keep, np.full_like(keep, pov)], 1))
        return np.concatenate(rows) if rows else np.zeros((0, 4), np.int64)

    # ------------------------------------------------------------------ windows

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        demo_idx, r, t, pov = (int(v) for v in self.index[i])
        demo = self.demos[demo_idx]
        offset, n = int(demo["round_offset"][r]), int(demo["round_n_steps"][r])
        start = 0 if self.history_steps is None else max(0, t - self.history_steps + 1)
        hist = slice(offset + start, offset + t + 1)
        H = t + 1 - start

        # Own team first: slots 0-4 = POV team, 5-9 = enemies.
        order = np.r_[5:10, 0:5] if pov == 1 else np.arange(N_SLOTS)
        players = np.zeros((H, N_SLOTS, len(PLAYER_INPUTS)), np.float32)
        players[..., :N_BASE] = demo["player"][hist][:, order]
        cats = demo["player_cat"][hist][:, order].copy()
        spotted_by = demo["spotted_by"][hist][:, order][:, :, order].copy()  # [H, slot, by_slot]
        visible = demo["visible"][hist][:, order]
        ever_seen = demo["ever_seen"][hist][:, order]
        seen_age = demo["seen_age"][hist][:, order]

        teammate = np.zeros(N_SLOTS, bool)
        teammate[:SLOTS_PER_SIDE] = True
        players[..., TEAMMATE] = teammate
        players[..., TEAMMATE + 1] = visible | teammate
        players[..., TEAMMATE + 2] = ever_seen | teammate
        players[..., TEAMMATE + 3] = np.where(teammate, 0.0, seen_age)

        globals_ = np.zeros((H, len(GLOBAL_INPUTS)), np.float32)
        globals_[:, :9] = demo["globals"][hist]
        side = demo["side"][hist]
        own, enemy = (1, 0) if pov == 1 else (0, 1)
        globals_[:, 9:11] = side[:, [own, enemy], S["score"]]
        globals_[:, 11:13] = side[:, [own, enemy], S["alive"]]
        globals_[:, 13:15] = side[:, [own, enemy], S["equip_value"]]
        globals_[:, 15] = pov == 0

        utility = demo["utility"][hist].copy()
        utility[..., U["thrown_by_t"]] = (utility[..., U["thrown_by_t"]] == (pov == 0)) & (utility[..., U["valid"]] > 0)

        if self.mask_enemies:
            hidden = ~teammate & ~visible  # [H, 10]
            alive = players[..., ALIVE].copy()
            last_seen = demo["last_seen"][hist][:, order]
            players[..., :N_BASE][hidden] = 0
            players[..., 0:3][hidden] = np.where(ever_seen[hidden, None], last_seen[hidden], 0)
            players[..., ALIVE] = alive  # deaths are public via the kill feed
            cats[hidden] = 0
            players[:, SLOTS_PER_SIDE:, PRIVATE] = 0  # enemy money and inventory are never visible
            cats[:, SLOTS_PER_SIDE:, PRIVATE_CATS] = 0
            spotted_by[:, :SLOTS_PER_SIDE, SLOTS_PER_SIDE:] = False  # a team can't tell which enemy sees them
            enemy_flying = (utility[..., U["in_flight"]] > 0) & (utility[..., U["thrown_by_t"]] == 0)
            utility[enemy_flying] = 0  # enemy grenades count once they pop
            globals_[:, 14] = 0  # enemy equipment value
            if pov == 1:  # CTs can't see a dropped bomb
                dropped = globals_[:, 5] > 0
                globals_[dropped, 5:9] = 0

        # Grenade identity: renumber the throws this window shows 1, 2, ... by first appearance, so the
        # ids link a grenade's flight, pop and smoke/fire without revealing hidden enemy throws.
        raw_ids = utility[..., U["throw_id"]].astype(np.int64) * (utility[..., U["valid"]] > 0)
        seen = raw_ids[raw_ids > 0]
        _, first = np.unique(seen, return_index=True)
        lut = np.zeros(raw_ids.max() + 1, np.int64)
        lut[seen[np.sort(first)]] = np.arange(1, len(first) + 1)
        utility_ids = lut[raw_ids]
        utility = utility[..., : U["throw_id"]]

        # Targets: next horizon_steps positions of the POV team, in game units.
        own_slots = order[:SLOTS_PER_SIDE]
        future = np.arange(t + 1, t + 1 + self.horizon_steps)
        in_round = future < n
        fut_idx = offset + np.minimum(future, n - 1)
        target_xy = demo["xy"][fut_idx][:, own_slots]
        target_mask = demo["alive"][fut_idx][:, own_slots] & in_round[:, None] & demo["slot_valid"][r, own_slots]
        target_xy[~target_mask] = 0
        anchor_xy = demo["xy"][offset + t][own_slots]
        anchor_alive = demo["alive"][offset + t][own_slots] & demo["slot_valid"][r, own_slots]

        return {
            "players": torch.from_numpy(players),
            "player_cats": torch.from_numpy(cats),
            "globals": torch.from_numpy(globals_),
            "utility": torch.from_numpy(np.ascontiguousarray(utility)),
            "utility_ids": torch.from_numpy(utility_ids),
            "spotted_by": torch.from_numpy(spotted_by),
            "slot_valid": torch.from_numpy(demo["slot_valid"][r, order].copy()),
            "target_xy": torch.from_numpy(target_xy),
            "target_mask": torch.from_numpy(target_mask),
            "anchor_xy": torch.from_numpy(anchor_xy),
            "anchor_alive": torch.from_numpy(anchor_alive),
            "meta": torch.tensor([demo_idx, int(demo["round_num"][r]), t, pov]),
        }


def collate_windows(batch):
    """Left-pad histories to the longest in the batch; hist_mask marks real steps."""
    H = max(item["players"].shape[0] for item in batch)
    out = {}
    for key in ("players", "player_cats", "globals", "utility", "utility_ids", "spotted_by"):
        first = batch[0][key]
        padded = first.new_zeros((len(batch), H) + first.shape[1:])
        for b, item in enumerate(batch):
            padded[b, H - item[key].shape[0]:] = item[key]
        out[key] = padded
    hist_mask = torch.zeros(len(batch), H, dtype=torch.bool)
    for b, item in enumerate(batch):
        hist_mask[b, H - item["players"].shape[0]:] = True
    out["hist_mask"] = hist_mask
    for key in ("slot_valid", "target_xy", "target_mask", "anchor_xy", "anchor_alive", "meta"):
        out[key] = torch.stack([item[key] for item in batch])
    return out
