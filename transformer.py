"""Factorized spatio-temporal transformer that forecasts 6 paths per player for the next 10 s.

One example is a whole round from one team's point of view (dataset.MirageRounds).
At every 0.5 s step there are 27 tokens: 10 players (own team first, enemies masked
to what the team knows), 1 round-state token and 16 grenade tokens. Each block runs

    time attention    each player / round-state track over its own past (causal;
                      the 3 s / 10 s models give every step its own window of raw
                      steps, so stacked blocks never see further back than the window)
    entity attention  all 27 tokens of the same step, with a learned bias for
                      who-spots-whom pairs
    feed-forward

so one forward pass predicts at every step of the round. A decoder turns each
living teammate's token into 6 paths (20 future x/y offsets, each with a Laplace
scale) and a probability per path, with a separate output head per path. Training
is winner-takes-all: only the path closest to what happened is pulled toward it
(distance), its Laplace scales are fit with the path held fixed, and the
probabilities are trained to pick it. The probability the model gives the real
path is the surprise score psi = -log p(actual path) (nats, positions in units of
100 game units).

Grenade tokens take part in entity attention only: their slots are sorted newest
first, so slot j is not the same grenade from step to step.

Usage:
    python transformer.py                          # full-round history, team view
    python transformer.py --history 3              # last 3 s only (also: 10)
    python transformer.py --full-info              # ablation: enemies always visible
    python transformer.py --eval-only --out checkpoints/transformer_full.pt
    python transformer.py --max-rounds 20 --epochs 1   # quick pipeline check

Metrics match baselines.ipynb (minADE/minFDE@6, top-1, miss rate, Brier-minFDE at
3, 5 and 10 s, all and moving players) plus psi; the final val/test evaluation also
reports callout accuracy and off-map rate.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import GLOBAL_INPUTS, PLAYER_INPUTS, UTILITY_INPUTS, MirageRounds, collate_rounds
from features import HORIZON_SECONDS, MAX_UTILITY, N_SLOTS, SLOTS_PER_SIDE, STEP_SECONDS

ROOT = Path(__file__).parent
H = int(round(HORIZON_SECONDS / STEP_SECONDS))  # 20 future steps
HORIZONS = {3: 6, 5: 10, 10: 20}  # seconds -> steps, as in baselines.ipynb
MISS_UNITS = 200
MOVING_SPEED = 100  # u/s at prediction time
BRIER_SCALE = 200
OFFSET_SCALE = 100.0  # model offsets and Laplace scales are in units of 100 game units
TRACKS = N_SLOTS + 1  # players + round state: the tokens that get time attention
TOKENS = TRACKS + MAX_UTILITY
UTILITY_VALID = UTILITY_INPUTS.index("valid")
HISTORY_STEPS = {"3": 7, "10": 21, "full": None}  # same step counts as MirageWindows(history_seconds=...)


@dataclass
class Config:
    n_weapons: int
    n_places: int
    d_model: int = 128
    heads: int = 4
    blocks: int = 4
    decoder_layers: int = 2
    modes: int = 6
    dropout: float = 0.1
    history_steps: int | None = None  # None = the whole round so far


def mlp(n_in, d):
    return nn.Sequential(nn.Linear(n_in, d), nn.GELU(), nn.Linear(d, d))


def sinusoid(steps, d):
    """[T, d] sinusoidal embedding of step indices."""
    freqs = torch.exp(torch.arange(0, d, 2, device=steps.device) * (-math.log(10000.0) / d))
    angles = steps[:, None].float() * freqs
    return torch.cat([angles.sin(), angles.cos()], -1)


class Embed(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d_model
        self.player = mlp(len(PLAYER_INPUTS), d)
        self.weapon = nn.Embedding(c.n_weapons, d)  # active weapon
        self.primary = nn.Embedding(c.n_weapons, d)
        self.secondary = nn.Embedding(c.n_weapons, d)
        self.place = nn.Embedding(c.n_places, d)
        self.team = nn.Embedding(2, d)  # enemy / teammate
        self.round_state = mlp(len(GLOBAL_INPUTS), d)
        self.utility = mlp(len(UTILITY_INPUTS), d)
        self.kind = nn.Embedding(3, d)  # player / round state / grenade

    def forward(self, batch):
        players, cats = batch["players"], batch["player_cats"].long()
        teammate = (torch.arange(N_SLOTS, device=players.device) < SLOTS_PER_SIDE).long()  # slots 0-4: own team
        p = (self.player(players) + self.weapon(cats[..., 0]) + self.place(cats[..., 1])
             + self.primary(cats[..., 2]) + self.secondary(cats[..., 3])
             + self.team(teammate) + self.kind.weight[0])
        g = self.round_state(batch["globals"])[:, :, None] + self.kind.weight[1]
        u = self.utility(batch["utility"]) + self.kind.weight[2]
        T = players.shape[1]
        pos = sinusoid(torch.arange(T, device=players.device), p.shape[-1])[None, :, None]
        return torch.cat([p, g], 2) + pos, u + pos  # tracks [B, T, 11, d], grenades [B, T, 16, d]


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d_model
        self.heads = c.heads
        self.norm_time, self.norm_ent, self.norm_ffn = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.time_attn = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.ent_attn = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(4 * d, d))
        self.spot_bias = nn.Parameter(torch.zeros(c.heads, 2))  # per head: "query is spotted by key", "query spots key"

    def forward(self, tracks, grenades, time_mask, ent_pad, spotted_by):
        B, T, K, d = tracks.shape
        y = tracks.transpose(1, 2).reshape(B * K, T, d)
        h = self.norm_time(y)
        y = y + self.time_attn(h, h, h, attn_mask=time_mask, need_weights=False)[0]
        tracks = y.reshape(B, K, T, d).transpose(1, 2)

        x = torch.cat([tracks, grenades], 2).reshape(B * T, TOKENS, d)
        bias = x.new_zeros(B * T, self.heads, TOKENS, TOKENS)
        spot = spotted_by.reshape(B * T, 1, N_SLOTS, N_SLOTS).to(x.dtype)
        bias[:, :, :N_SLOTS, :N_SLOTS] = (self.spot_bias[:, 0, None, None] * spot
                                         + self.spot_bias[:, 1, None, None] * spot.transpose(-1, -2))
        bias = bias.masked_fill(ent_pad[:, None, None, :], float("-inf")).reshape(B * T * self.heads, TOKENS, TOKENS)
        h = self.norm_ent(x)
        x = x + self.ent_attn(h, h, h, attn_mask=bias, need_weights=False)[0]
        x = x + self.ffn(self.norm_ffn(x))
        x = x.reshape(B, T, TOKENS, d)
        return x[:, :, :TRACKS], x[:, :, TRACKS:]


class DecoderLayer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        d = c.d_model
        self.modes = c.modes
        self.norm_self, self.norm_cross, self.norm_ffn = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, c.heads, dropout=c.dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(c.dropout), nn.Linear(4 * d, d))

    def forward(self, q, context, pad):
        # q: [B*T, 5 * modes, d] (5 teammates x 6 modes); context: [B*T, 27, d]
        N, Q, d = q.shape
        y = q.reshape(N * SLOTS_PER_SIDE, self.modes, d)  # the 6 modes of one player talk to each other
        h = self.norm_self(y)
        q = (y + self.self_attn(h, h, h, need_weights=False)[0]).reshape(N, Q, d)
        q = q + self.cross_attn(self.norm_cross(q), context, context, key_padding_mask=pad, need_weights=False)[0]
        return q + self.ffn(self.norm_ffn(q))


class MirageTransformer(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.config = c
        self.embed = Embed(c)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.blocks))
        # Modes need to differ from the start: with small shared queries and one shared head all
        # 6 paths came out nearly identical (mode collapse) and training was ~2x slower.
        self.mode_queries = nn.Parameter(torch.randn(c.modes, c.d_model))
        self.decoder = nn.ModuleList(DecoderLayer(c) for _ in range(c.decoder_layers))
        self.norm_out = nn.LayerNorm(c.d_model)
        # One output head per mode. Per step: dx, dy, Laplace scale x, y; then a logit.
        self.head_weight = nn.Parameter(torch.randn(c.modes, c.d_model, 4 * H + 1) * 0.1 * c.d_model ** -0.5)
        self.head_bias = nn.Parameter(torch.zeros(c.modes, 4 * H + 1))

    def encode(self, tracks, grenades, pad, spotted_by):
        """Run the blocks. Full history: one causal pass over the round. A history window W:
        every step gets its own window of the last W raw steps, so stacking blocks can't
        widen what a step sees (a shared windowed pass would reach blocks * (W - 1) steps back)."""
        B, T, _, d = tracks.shape
        W = self.config.history_steps
        device = tracks.device
        if W is None:
            causal = torch.ones(T, T, dtype=torch.bool, device=device).triu(1)  # True = not allowed
            for block in self.blocks:
                tracks, grenades = block(tracks, grenades, causal, pad, spotted_by)
            return tracks, grenades

        idx = torch.arange(T, device=device)[:, None] + torch.arange(W, device=device)  # [T, W] in padded time

        def windows(x):  # [B, T, ...] -> [B * T, W, ...]: step t's last W steps, oldest first, left-padded
            x = torch.cat([x.new_zeros((B, W - 1) + x.shape[2:]), x], 1)
            return x[:, idx].reshape((B * T, W) + x.shape[2:])

        # Time mask per window: causal, and left padding is never attended (a padding position
        # only attends to itself, so no attention row is empty).
        padding = idx < W - 1  # [T, W]
        causal = torch.ones(W, W, dtype=torch.bool, device=device).triu(1)
        mask = causal | padding[:, None, :]
        mask &= ~torch.eye(W, dtype=torch.bool, device=device) | ~padding[:, :, None]
        heads = self.config.heads
        mask = mask[None, :, None, None].expand(B, T, TRACKS, heads, W, W).reshape(B * T * TRACKS * heads, W, W)

        tw, gw, sw = windows(tracks), windows(grenades), windows(spotted_by)
        pw = windows(pad.reshape(B, T, TOKENS)).reshape(B * T * W, TOKENS)
        for block in self.blocks:
            tw, gw = block(tw, gw, mask, pw, sw)
        return tw[:, -1].reshape(B, T, TRACKS, d), gw[:, -1].reshape(B, T, MAX_UTILITY, d)

    def forward(self, batch):
        """Returns offsets [B, T, 5, M, H, 2] (game units, from each player's current position),
        scales [B, T, 5, M, H, 2] (Laplace, units of 100) and logits [B, T, 5, M]."""
        c = self.config
        tracks, grenades = self.embed(batch)
        B, T = tracks.shape[:2]
        pad = torch.cat([
            ~batch["slot_valid"][:, None, :].expand(B, T, N_SLOTS),
            torch.zeros(B, T, 1, dtype=torch.bool, device=tracks.device),  # round state: always there
            batch["utility"][..., UTILITY_VALID] <= 0,
        ], 2).reshape(B * T, TOKENS)
        tracks, grenades = self.encode(tracks, grenades, pad, batch["spotted_by"])

        context = torch.cat([tracks, grenades], 2).reshape(B * T, TOKENS, -1)
        own = tracks[:, :, :SLOTS_PER_SIDE].reshape(B * T, SLOTS_PER_SIDE, 1, -1)
        q = (own + self.mode_queries).reshape(B * T, SLOTS_PER_SIDE * c.modes, -1)
        for layer in self.decoder:
            q = layer(q, context, pad)
        q = self.norm_out(q).reshape(B * T, SLOTS_PER_SIDE, c.modes, -1)
        out = torch.einsum("nsmd,mdo->nsmo", q, self.head_weight) + self.head_bias
        out = out.reshape(B, T, SLOTS_PER_SIDE, c.modes, 4 * H + 1)
        offsets = out[..., : 2 * H].reshape(B, T, SLOTS_PER_SIDE, c.modes, H, 2).cumsum(-2) * OFFSET_SCALE
        scales = F.softplus(out[..., 2 * H: 4 * H].reshape(B, T, SLOTS_PER_SIDE, c.modes, H, 2)) + 0.01
        return offsets, scales, out[..., -1]


# ---------------------------------------------------------------------- loss and metrics

def targets(batch):
    """Target offsets [B, T, 5, H, 2] (game units), step mask [B, T, 5, H] and which predictions count [B, T, 5]."""
    target = (batch["target_xy"] - batch["anchor_xy"][:, :, None]).permute(0, 1, 3, 2, 4)
    mask = batch["target_mask"].permute(0, 1, 3, 2) & (batch["anchor_alive"] & batch["step_mask"][..., None])[..., None]
    return target, mask, mask.any(-1)


def laplace_nll(offsets, scales, target, mask):
    """Per-mode Laplace negative log-likelihood summed over valid steps and x/y: [B, T, 5, M]."""
    err = (offsets - target[:, :, :, None]).abs() / OFFSET_SCALE
    nll = (err / scales + torch.log(2 * scales)).sum(-1)  # [B, T, 5, M, H]
    return (nll * mask[:, :, :, None]).sum(-1)


def loss_fn(outputs, batch):
    """Winner-takes-all over the 6 paths: the closest path is pulled toward the real one by its
    distance (ADE), its Laplace scales are fit with the path held fixed (so uncertainty can't slow
    the path down, which fitting both with the NLL did), and the probabilities learn to pick it."""
    offsets, scales, logits = outputs
    target, mask, valid = targets(batch)
    m = mask[:, :, :, None].float()
    dist = ((offsets - target[:, :, :, None]) ** 2).sum(-1).add(1e-6).sqrt()  # [B, T, 5, M, H]
    ade = (dist * m).sum(-1) / m.sum(-1).clamp(min=1)
    best = ade.argmin(-1).detach()

    def pick(x):
        return x.gather(-1, best[..., None])[..., 0][valid]

    distance = pick(ade).mean() / OFFSET_SCALE
    nll = laplace_nll(offsets.detach(), scales, target, mask) / m.sum(-1).clamp(min=1)  # per valid step
    uncertainty = pick(nll).mean()
    classification = F.cross_entropy(logits[valid], best[valid])
    return distance + uncertainty + classification, {
        "ade": distance.item() * OFFSET_SCALE, "nll": uncertainty.item(), "ce": classification.item()}


class MapLookup:
    """Walkable area and callout lookup built from the training split, as in baselines.ipynb.

    Walkable = the 64-unit cells a living player stood in (at every step that has a future),
    grown by one cell. A point's callout = the game's callout at the nearest of those positions.
    """

    CELL = 64

    def __init__(self, train: MirageRounds):
        from scipy.ndimage import binary_dilation
        from sklearn.neighbors import NearestNeighbors

        xy, place = [], []
        for demo in train.demos:
            for r, (offset, n) in enumerate(zip(demo["round_offset"].tolist(), demo["round_n_steps"].tolist())):
                alive = demo["alive"][offset:offset + n - 1] & demo["slot_valid"][r]
                t, s = np.nonzero(alive)  # same positions, same order as the notebook's training samples
                xy.append(demo["xy"][offset + t, s])
                place.append(demo["player_cat"][offset + t, s, 1])
        xy, place = np.concatenate(xy).astype(np.float32), np.concatenate(place)
        cells = np.floor(xy / self.CELL).astype(int)
        self.lo = cells.min(0) - 2
        walkable = np.zeros(cells.max(0) - self.lo + 3, bool)
        walkable[tuple((cells - self.lo).T)] = True
        self.walkable = binary_dilation(walkable, iterations=1)
        known = place > 0
        self.index = NearestNeighbors(n_neighbors=1).fit(xy[known])
        self.labels = place[known]

    def off_map(self, points):
        """[..., 2] game units -> bool [...]: outside the walkable area."""
        c = np.floor(points.reshape(-1, 2) / self.CELL).astype(int) - self.lo
        inside = ((c >= 0) & (c < self.walkable.shape)).all(1)
        off = ~inside
        off[inside] = ~self.walkable[c[inside, 0], c[inside, 1]]
        return off.reshape(points.shape[:-1])

    def callout(self, points):
        """[..., 2] game units -> callout ids [...] of the nearest training position."""
        nearest = self.index.kneighbors(points.reshape(-1, 2), return_distance=False)[:, 0]
        return self.labels[nearest].reshape(points.shape[:-1])


class Metrics:
    """Accumulates the baselines.ipynb metrics (plus psi) over batches.

    With a MapLookup it also reports callout accuracy (top-1 path, any of the 6), the
    lookup's ceiling (the true endpoint's looked-up callout vs the game's) and the
    off-map rate of every predicted point; these are slower, so only the final evaluation uses them.
    """

    NAMES = ("minADE@6", "minFDE@6", "ADE top-1", "FDE top-1", "miss rate %", "Brier-minFDE@6", "psi (nats)")
    MAP_NAMES = ("callout top-1 %", "callout @6 %", "callout ceiling %", "off-map %")

    def __init__(self, maps: MapLookup | None = None):
        self.maps = maps
        self.sums = {(s, h): np.zeros(len(self.NAMES)) for s in ("all", "moving") for h in HORIZONS}
        self.counts = {key: 0 for key in self.sums}
        self.map_sums = {key: np.zeros(len(self.MAP_NAMES)) for key in self.sums}
        self.map_counts = {key: np.zeros(len(self.MAP_NAMES)) for key in self.sums}

    @torch.no_grad()
    def update(self, outputs, batch):
        offsets, scales, logits = (o.float() for o in outputs)
        target, mask, _ = targets(batch)
        log_prob = logits.log_softmax(-1)
        moving = batch["anchor_speed"] >= MOVING_SPEED
        for sec, h in HORIZONS.items():
            ok = mask[..., h - 1]  # alive and in the round through the horizon
            err = ((offsets[..., :h, :] - target[:, :, :, None, :h]) ** 2).sum(-1).sqrt()  # [B, T, 5, M, h]
            ade, fde = err.mean(-1), err[..., -1]
            top = log_prob.argmax(-1, keepdim=True)
            closest = fde.argmin(-1, keepdim=True)
            p_closest = log_prob.gather(-1, closest)[..., 0].exp()
            nll = laplace_nll(offsets[..., :h, :], scales[..., :h, :], target[..., :h, :], mask[..., :h])
            psi = -torch.logsumexp(log_prob - nll, -1)
            values = torch.stack([
                ade.min(-1).values, fde.min(-1).values,
                ade.gather(-1, top)[..., 0], fde.gather(-1, top)[..., 0],
                100.0 * (fde.min(-1).values > MISS_UNITS).float(),
                fde.min(-1).values + BRIER_SCALE * (1 - p_closest) ** 2,
                psi,
            ], -1)
            for subset, sel in (("all", ok), ("moving", ok & moving)):
                self.sums[(subset, sec)] += values[sel].double().sum(0).cpu().numpy()
                self.counts[(subset, sec)] += int(sel.sum())
            if self.maps is not None:
                self._update_maps(sec, h, ok, moving, offsets, top, target, batch)

    def _update_maps(self, sec, h, ok, moving, offsets, top, target, batch):
        sel_all = ok.cpu().numpy()
        anchor = batch["anchor_xy"].float()[:, :, :, None, None]  # [B, T, 5, 1, 1, 2]
        points = (anchor + offsets[..., :h, :]).cpu().numpy()[sel_all]  # [n, M, h, 2] absolute
        true_end = (batch["anchor_xy"].float() + target[..., h - 1, :]).cpu().numpy()[sel_all]  # [n, 2]
        true_place = batch["target_place"].permute(0, 1, 3, 2)[..., h - 1].cpu().numpy()[sel_all]  # [n]
        top_idx = top[..., 0].cpu().numpy()[sel_all]
        ends = self.maps.callout(points[:, :, -1])  # [n, M]
        labelled = true_place > 0
        per_sample = np.stack([
            ends[np.arange(len(ends)), top_idx] == true_place,
            (ends == true_place[:, None]).any(1),
            self.maps.callout(true_end) == true_place,
            self.maps.off_map(points).mean((1, 2)),  # same number of points per sample
        ], 1).astype(np.float64)
        weight = np.stack([labelled, labelled, labelled, np.ones_like(labelled)], 1)
        for subset, sel in (("all", np.ones(len(per_sample), bool)), ("moving", moving.cpu().numpy()[sel_all])):
            self.map_sums[(subset, sec)] += (per_sample * weight)[sel].sum(0)
            self.map_counts[(subset, sec)] += weight[sel].sum(0)

    def table(self):
        rows = {}
        for (subset, sec), total in self.sums.items():
            n = max(self.counts[(subset, sec)], 1)
            row = {name: float(v / n) for name, v in zip(self.NAMES, total)}
            if self.maps is not None:
                counts = np.maximum(self.map_counts[(subset, sec)], 1)
                row |= {name: float(100 * v / c) for name, v, c in zip(self.MAP_NAMES, self.map_sums[(subset, sec)], counts)}
            rows[f"{subset} {sec}s"] = row | {"samples": self.counts[(subset, sec)]}
        return rows


# ---------------------------------------------------------------------- training

def log(path, message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def batches_by_length(lengths, batch_size, rng):
    """Shuffled batches of rounds with similar length, to keep padding small."""
    order = rng.permutation(len(lengths))
    chunks = [order[i:i + batch_size * 50] for i in range(0, len(order), batch_size * 50)]
    batches = []
    for chunk in chunks:
        chunk = chunk[np.argsort(lengths[chunk])]
        batches += [chunk[i:i + batch_size].tolist() for i in range(0, len(chunk), batch_size)]
    rng.shuffle(batches)
    return batches


def round_lengths(ds):
    return np.array([int(ds.demos[d]["round_n_steps"][r]) - 1 for d, r, _ in ds.rounds])


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, ds, device, batch_size, amp, maps=None):
    model.eval()
    metrics = Metrics(maps)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=collate_rounds)
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = model(batch)
        metrics.update(outputs, batch)
    return metrics.table()


def pick_device(name):
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=str(ROOT / "processed"))
    parser.add_argument("--history", choices=list(HISTORY_STEPS), default="full", help="history window in seconds")
    parser.add_argument("--full-info", action="store_true", help="ablation: enemies are never masked")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8, help="rounds per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5, help="epochs without validation improvement before stopping")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or dml (DirectML)")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    parser.add_argument("--max-rounds", type=int, help="use only this many training rounds (quick checks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", help="checkpoint path (default: checkpoints/transformer_<history>[_fullinfo].pt)")
    parser.add_argument("--eval-only", action="store_true", help="evaluate an existing checkpoint on val and test")
    args = parser.parse_args()

    name = f"transformer_{args.history}" + ("_fullinfo" if args.full_info else "")
    out = Path(args.out or ROOT / "checkpoints" / f"{name}.pt")
    log_path = ROOT / "logs" / f"{name}.log"
    log_path.parent.mkdir(exist_ok=True)
    out.parent.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    amp = device.type == "cuda"

    data = {split: MirageRounds(args.processed, split=split, mask_enemies=not args.full_info) for split in ("train", "val", "test")}
    train = data["train"]
    if args.max_rounds:
        train.rounds = train.rounds[rng.permutation(len(train.rounds))[: args.max_rounds]]
    manifest = train.manifest

    if args.eval_only:
        ckpt = torch.load(out, map_location="cpu")
        model = MirageTransformer(Config(**ckpt["config"])).to(device)
        model.load_state_dict(ckpt["state"])
    else:
        config = Config(
            n_weapons=len(manifest["weapon_vocab"]), n_places=len(manifest["place_vocab"]),
            d_model=args.d_model, heads=args.heads, blocks=args.blocks, decoder_layers=args.decoder_layers,
            dropout=args.dropout, history_steps=HISTORY_STEPS[args.history],
        )
        model = MirageTransformer(config).to(device)
        params = sum(p.numel() for p in model.parameters())
        log(log_path, f"=== {name}: {params / 1e6:.2f}M parameters, {len(train)} training rounds, device {device}, {vars(args)}")

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lengths = round_lengths(train)
        steps_per_epoch = math.ceil(len(train) / args.batch)
        total = steps_per_epoch * args.epochs
        warmup = steps_per_epoch  # one epoch
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        best, stale, history = math.inf, 0, []
        start_all = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            start, total_loss, n = time.perf_counter(), 0.0, 0
            loader = DataLoader(train, batch_sampler=batches_by_length(lengths, args.batch, rng),
                                collate_fn=collate_rounds, num_workers=args.workers)
            for batch in loader:
                batch = to_device(batch, device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    loss, parts = loss_fn(model(batch), batch)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total_loss += loss.item()
                n += 1
            val = evaluate(model, data["val"], device, args.batch, amp)
            score = val["all 10s"]["minADE@6"]
            history.append({"epoch": epoch + 1, "loss": total_loss / max(n, 1), "val_minADE@6_10s": score,
                            "seconds": time.perf_counter() - start})
            log(log_path, f"epoch {epoch + 1:2d}/{args.epochs}: loss {total_loss / max(n, 1):.3f}, "
                          f"val minADE@6 (10s) {score:.1f}, psi {val['all 10s']['psi (nats)']:.1f} ({history[-1]['seconds']:.0f}s)")
            if score < best:
                best, stale = score, 0
                torch.save({"state": model.state_dict(), "config": asdict(config), "history": history,
                            "args": vars(args), "demos": manifest["demos"]}, out)
            else:
                stale += 1
                if stale >= args.patience:
                    log(log_path, f"early stop: no improvement for {args.patience} epochs")
                    break
        log(log_path, f"trained {len(history)} epochs in {time.perf_counter() - start_all:.0f}s, best val minADE@6 (10s) {best:.1f}")
        model.load_state_dict(torch.load(out, map_location="cpu")["state"])

    start = time.perf_counter()
    maps = MapLookup(data["train"])
    results = {split: evaluate(model, data[split], device, args.batch, amp, maps) for split in ("val", "test")}
    log(log_path, f"final evaluation (with callout / off-map metrics) took {time.perf_counter() - start:.0f}s")
    for split, table in results.items():
        log(log_path, f"{split}:")
        for row, values in table.items():
            log(log_path, f"  {row:11s} " + "  ".join(f"{k} {v:.1f}" for k, v in values.items() if k != "samples")
                + f"  (n={values['samples']})")
    (out.with_suffix(".metrics.json")).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
