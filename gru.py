"""Per-player GRU baseline: train, evaluate and save its predictions (CPU or GPU).

Sees only one player: their last 10 s of positions and per-step movement, plus side, round time
and whether the bomb is planted (no teammates, enemies or utility). It outputs 6 paths as
cumulative per-step offsets plus a probability for each path. Trained winner-takes-all: the
regression loss only updates the closest path, and the probabilities learn to pick it. Early
stopping on validation minADE@6 at 10 s.

baselines.ipynb imports this module and reuses checkpoints/per_player_gru.pt when it was trained
on exactly the same samples (checked with a hash), so the GRU can be trained here, e.g. on a GPU
cluster, and the notebook only evaluates it.

Usage:
    python gru.py --device cuda                 # train, then save test predictions for app.py
    python gru.py --device cuda --resume        # continue after a job time limit
    python gru.py --eval-only                   # only predict with the saved checkpoint

Writes checkpoints/per_player_gru.pt (best epoch), .last.pt (every epoch, for --resume),
.history.json and .predictions_test.npz; progress goes to logs/per_player_gru.log.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import G, HORIZON_SECONDS, P, S, SLOTS_PER_SIDE, STEP_SECONDS

ROOT = Path(__file__).parent
H = int(round(HORIZON_SECONDS / STEP_SECONDS))  # 20 future steps
L = 20  # history steps (10 s) given to the kNN and GRU baselines
K = 6  # predicted paths per player
MOVING_SPEED = 100  # u/s at prediction time
OFFSET_SCALE = 100.0  # one output unit = 100 game units of movement per 0.5 s step


# ---------------------------------------------------------------------- per-player samples

def load_split(split, processed="processed"):
    """Per-player samples of one split, straight from processed/*.npz.

    One row per player alive at step t (with at least one future step, as in dataset.py):
    `hist` is the player's last L positions (oldest first, padded with the round's first
    position), `fut` the next H positions, `mask` the future steps where the player is alive and
    the round still running, `meta` = (demo index within the split, round index, step, slot).
    """
    processed = Path(processed)
    manifest = json.loads((processed / "manifest.json").read_text())
    place_ids = {name: i for i, name in enumerate(manifest["place_vocab"])}
    parts = []
    entries = [e for e in manifest["demos"] if e["split"] == split]
    for demo_idx, entry in enumerate(entries):
        with np.load(processed / entry["file"]) as z:
            p, glob, side, slot_valid = z["player"], z["globals"], z["side"], z["round_slot_valid"]
            # Per-demo callout ids -> global place_vocab ids (0 = no callout).
            remap = np.array([place_ids.get(s, 0) if s else 0 for s in z["place_strings"].tolist()])
            place = remap[z["player_cat"][..., 1]]  # [steps, 10]
            for r, (off, n) in enumerate(zip(z["round_offset"].tolist(), z["round_n_steps"].tolist())):
                xy = p[off:off + n, :, [P["X"], P["Y"]]]
                alive = (p[off:off + n, :, P["is_alive"]] > 0) & slot_valid[r]
                steps = np.arange(n)
                fut = steps[:, None] + np.arange(1, H + 1)
                fut_xy = xy[np.minimum(fut, n - 1)]  # [n, H, 10, 2]
                fut_mask = alive[np.minimum(fut, n - 1)] & (fut < n)[..., None]  # [n, H, 10]
                hist_xy = xy[np.maximum(steps[:, None] - np.arange(L - 1, -1, -1), 0)]  # [n, L, 10, 2]
                round_place = place[off:off + n]
                fut_place = round_place[np.minimum(fut, n - 1)]  # [n, H, 10]
                t, s = np.nonzero(alive[: n - 1])  # need at least one future step, as in dataset.py
                parts.append({
                    "anchor": xy[t, s],
                    "vel": p[off + t, s][:, [P["vel_x"], P["vel_y"]]],
                    "hist": hist_xy[t, :, s],
                    "fut": fut_xy[t, :, s],
                    "mask": fut_mask[t, :, s],
                    "place": round_place[t, s],
                    "fut_place": fut_place[t, :, s],
                    "side": (s >= SLOTS_PER_SIDE).astype(np.float32),  # 0 = T, 1 = CT
                    "time": glob[off + t, G["time_since_freeze_end"]],
                    "planted": glob[off + t, G["bomb_planted"]],
                    # economy and score: the player's own and their team's (enemy equipment is hidden from a team)
                    "score_own": side[off + t, s // SLOTS_PER_SIDE, S["score"]],
                    "score_enemy": side[off + t, 1 - s // SLOTS_PER_SIDE, S["score"]],
                    "equip": p[off + t, s, P["equip_value"]],
                    "team_equip": side[off + t, s // SLOTS_PER_SIDE, S["equip_value"]],
                    "meta": np.stack([np.full_like(t, demo_idx), np.full_like(t, r), t, s], 1),
                })
    d = {k: np.concatenate([part[k] for part in parts]) for k in parts[0]}
    for k in d:
        if k not in ("mask", "meta", "place", "fut_place"):
            d[k] = d[k].astype(np.float32)
    d["fut"][~d["mask"]] = 0  # positions after death are meaningless
    d["moving"] = np.linalg.norm(d["vel"], axis=1) >= MOVING_SPEED
    return d


# ---------------------------------------------------------------------- model

def gru_inputs(d, pos_mean, pos_std):
    pos = (d["hist"] - pos_mean) / pos_std
    step = np.diff(d["hist"], axis=1, prepend=d["hist"][:, :1]) / 125.0  # 125 units = running for 0.5 s
    static = np.column_stack([d["side"], d["time"] / 115.0, d["planted"]])
    return torch.from_numpy(np.concatenate([pos, step, np.repeat(static[:, None], L, 1)], -1).astype(np.float32))


class PlayerGRU(nn.Module):
    def __init__(self, n_in, hidden=128):
        super().__init__()
        self.gru = nn.GRU(n_in, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(), nn.Linear(256, K * (2 * H + 1)))

    def forward(self, x):
        _, h = self.gru(x)
        out = self.head(h[-1]).view(-1, K, 2 * H + 1)
        offsets = out[..., :-1].reshape(-1, K, H, 2).cumsum(2) * OFFSET_SCALE
        return offsets, out[..., -1]


def wta_loss(offsets, logits, target, mask):
    err = ((offsets - target[:, None]) ** 2).sum(-1).add(1e-6).sqrt()  # [B, K, H]
    m = mask[:, None].float()
    ade = (err * m).sum(-1) / m.sum(-1).clamp(min=1)
    best = ade.argmin(1)
    return ade.gather(1, best[:, None]).mean() / OFFSET_SCALE + F.cross_entropy(logits, best)


def tensors(d, pos_mean, pos_std):
    keep = d["mask"][:, 0]  # samples with no valid future step carry no signal
    return (
        gru_inputs({k: v[keep] for k, v in d.items()}, pos_mean, pos_std),
        torch.from_numpy(d["fut"][keep] - d["anchor"][keep][:, None]) * torch.from_numpy(d["mask"][keep])[..., None],
        torch.from_numpy(d["mask"][keep]),
    )


def fingerprint(x, y, m):
    """Hash of the training tensors: a checkpoint is only reused for exactly the same samples."""
    return hashlib.sha1(b"".join(t.numpy().tobytes() for t in (x, y, m))).hexdigest()


@torch.no_grad()
def predict(model, d, pos_mean, pos_std, device="cpu", batch_size=4096):
    """6 absolute paths [N, K, H, 2] and their probabilities [N, K]."""
    model.eval()
    x = gru_inputs(d, pos_mean, pos_std)
    offsets, probs = [], []
    for i in range(0, len(x), batch_size):
        o, logit = model(x[i:i + batch_size].to(device))
        offsets.append(o.float().cpu())
        probs.append(logit.float().softmax(-1).cpu())
    return d["anchor"][:, None, None] + torch.cat(offsets).numpy(), torch.cat(probs).numpy()


def min_ade(pred, d, h=H):
    """minADE@6 over the first h steps, for samples alive through step h (as in baselines.ipynb)."""
    ok = d["mask"][:, h - 1]
    err = np.linalg.norm(pred[ok][:, :, :h] - d["fut"][ok][:, None, :h], axis=-1)
    return float(err.mean(-1).min(1).mean())


# ---------------------------------------------------------------------- training

def fit(train, val, pos_mean, pos_std, checkpoint, device="cpu", epochs=30, batch=512, patience=4, lr=1e-3,
        retrain=False, resume=False, seed=0, log=print):
    """Train (or reuse) the GRU. Returns (model, checkpoint dict with history and timings).

    An existing checkpoint is reused when its data hash matches and retrain is False. With
    resume, training continues from <checkpoint>.last.pt.
    """
    checkpoint = Path(checkpoint)
    last = checkpoint.with_suffix(".last.pt")
    x, y, m = tensors(train, pos_mean, pos_std)
    sha = fingerprint(x, y, m)
    model = PlayerGRU(x.shape[-1]).to(device)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False) if checkpoint.exists() and not retrain else None
    if ckpt is not None and ckpt.get("data_sha1") == sha and not (resume and last.exists()):
        model.load_state_dict(ckpt["state"])
        log(f"GRU: loaded {checkpoint} ({len(ckpt['history'])} epochs, originally trained in "
            f"{ckpt['timings']['fit / train s']:.0f}s)")
        return model, ckpt

    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_score, best_state, stale, history, first, spent = math.inf, None, 0, [], 0, 0.0
    if resume and last.exists():
        state = torch.load(last, map_location="cpu", weights_only=False)
        if state["data_sha1"] != sha:
            raise SystemExit(f"{last} was trained on different data; drop --resume to start over")
        model.load_state_dict(state["state"])
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["torch_rng"])
        best_score, best_state, stale = state["best_score"], state["best_state"], state["stale"]
        history, first, spent = state["history"], state["epoch"], state["seconds"]
        log(f"GRU: resumed from {last} after epoch {first}")

    # On a GPU the whole training set fits in memory, which avoids a host-to-device copy per batch.
    on_device = torch.device(device).type == "cuda"
    if on_device:
        x, y, m = x.to(device), y.to(device), m.to(device)
    log(f"GRU: training on {len(x)} samples, batch {batch}, up to {epochs} epochs, device {device}")
    train_start = time.perf_counter()
    for epoch in range(first, epochs):
        if stale >= patience:
            break
        model.train()
        start, total = time.perf_counter(), 0.0
        perm = torch.randperm(len(x), device=x.device)
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            xb, yb, mb = (x[b], y[b], m[b]) if on_device else (x[b].to(device), y[b].to(device), m[b].to(device))
            loss = wta_loss(*model(xb), yb, mb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(b)
        score = min_ade(predict(model, val, pos_mean, pos_std, device)[0], val)
        history.append((total / len(perm), score, time.perf_counter() - start))
        log(f"GRU epoch {epoch + 1:2d}/{epochs}: train loss {total / len(perm):.3f}, "
            f"val minADE@6 (10s) {score:.1f} ({history[-1][2]:.0f}s)")
        if score < best_score:
            best_score, best_state, stale = score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.with_suffix(".history.json").write_text(json.dumps(
            [{"epoch": i + 1, "loss": h[0], "val_minADE@6_10s": h[1], "seconds": h[2]} for i, h in enumerate(history)], indent=2))
        torch.save({"state": model.state_dict(), "opt": opt.state_dict(), "torch_rng": torch.get_rng_state(),
                    "best_score": best_score, "best_state": best_state, "stale": stale, "history": history,
                    "epoch": epoch + 1, "seconds": spent + time.perf_counter() - train_start, "data_sha1": sha}, last)
        if stale >= patience:
            log(f"GRU: early stop, no improvement for {patience} epochs")
    model.load_state_dict(best_state)
    timings = {
        "fit / train s": spent + time.perf_counter() - train_start,
        "epochs": len(history),
        "s per epoch": float(np.mean([h[2] for h in history])),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    ckpt = {"state": best_state, "history": history, "timings": timings, "train_samples": len(x), "data_sha1": sha}
    torch.save(ckpt, checkpoint)
    log(f"GRU: trained {len(history)} epochs in {timings['fit / train s']:.0f}s, best val minADE@6 (10s) "
        f"{best_score:.1f}, saved {checkpoint}")
    return model, ckpt


# ---------------------------------------------------------------------- cli

def main():
    from transformer import MapLookup, pick_device, save_player_predictions
    from dataset import MirageRounds

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processed", default=str(ROOT / "processed"))
    parser.add_argument("--out", default=str(ROOT / "checkpoints" / "per_player_gru.pt"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or dml (DirectML)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--patience", type=int, default=4, help="epochs without validation improvement before stopping")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="continue training from <out>.last.pt")
    parser.add_argument("--retrain", action="store_true", help="train even if a checkpoint for this data exists")
    parser.add_argument("--eval-only", action="store_true", help="only predict with the saved checkpoint")
    parser.add_argument("--no-predictions", action="store_true", help="skip writing <out>.predictions_test.npz")
    args = parser.parse_args()

    log_path = ROOT / "logs" / "per_player_gru.log"
    log_path.parent.mkdir(exist_ok=True)

    def log(message):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    device = pick_device(args.device)
    manifest = json.loads((Path(args.processed) / "manifest.json").read_text())
    pos_mean, pos_std = np.array(manifest["pos_mean"], np.float32), np.float32(manifest["pos_std"])
    start = time.perf_counter()
    data = {split: load_split(split, args.processed) for split in ("train", "val", "test")}
    log(f"=== GRU: data loaded in {time.perf_counter() - start:.0f}s: "
        + ", ".join(f"{s} {len(d['anchor'])} samples" for s, d in data.items()) + f", device {device}")

    out = Path(args.out)
    if args.eval_only:
        if not out.exists():
            raise SystemExit(f"{out} not found; train first")
        model = PlayerGRU(gru_inputs(data["val"], pos_mean, pos_std).shape[-1]).to(device)
        model.load_state_dict(torch.load(out, map_location="cpu", weights_only=False)["state"])
    else:
        model, _ = fit(data["train"], data["val"], pos_mean, pos_std, out, device, args.epochs, args.batch, args.patience,
                       args.lr, args.retrain, args.resume, args.seed, log)

    pred, prob = predict(model, data["test"], pos_mean, pos_std, device)
    log(f"GRU test minADE@6 (10s) {min_ade(pred, data['test']):.1f}")
    if not args.no_predictions:
        maps = MapLookup(MirageRounds(args.processed, split="train"))
        path = save_player_predictions(out.with_suffix(".predictions_test.npz"), pred, prob, data["test"]["meta"],
                                       args.processed, "test", maps)
        log(f"saved test predictions to {path} ({path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
