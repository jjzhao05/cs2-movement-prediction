"""Plot saved transformer predictions on the Mirage map, and break down their metrics.

Works from the files transformer.py writes after training (no model or GPU needed):
    checkpoints/<name>.predictions_<split>.npz   every prediction of a split
    checkpoints/<name>.history.json              per-epoch loss and validation metrics

Usage:
    python visualize.py rounds  checkpoints/transformer_full.predictions_test.npz          # list rounds
    python visualize.py round   checkpoints/transformer_full.predictions_test.npz 12       # one step of round 12
    python visualize.py round   checkpoints/transformer_full.predictions_test.npz 12 --step 40
    python visualize.py round   checkpoints/transformer_full.predictions_test.npz 12 --gif # animate the round
    python visualize.py metrics checkpoints/transformer_full.predictions_test.npz          # plots + CSVs

Everything is written to plots/<predictions file name>/ unless --out is given. Add
--team-view to a round plot to hide enemies the team could not see.

The background is the map drawn by build_map.py (assets/), else awpy's radar image
(`awpy get maps`), else a density map of where players stood.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
MAP = "de_mirage"
# Mirage overview settings (pos_x, pos_y, scale) from the game, used if awpy's map data is missing.
RADAR_FALLBACK = {"pos_x": -3230.0, "pos_y": 1713.0, "scale": 5.0}
SIDES = ("T", "CT")
MISS_UNITS = 200
MOVING_SPEED = 100
COLORS = plt.get_cmap("tab10").colors


def load(path):
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    data["horizons"] = {int(k): v for k, v in json.loads(str(data["horizons"])).items()}
    data["step_seconds"] = float(data["step_seconds"])
    return data


def col(data, group, name):
    """Index of a named feature, e.g. col(data, "player", "health") -> index into data["player"][..., i]."""
    return list(data[f"{group}_features"]).index(name)


def round_slice(data, i):
    start = int(data["rounds_start"][i])
    return slice(start, start + int(data["rounds_n_steps"][i]))


def round_title(data, i):
    own, enemy = data["rounds_teams"][i]
    side = SIDES[int(data["rounds_pov"][i])]
    return f"{own} ({side}) vs {enemy} - round {data['rounds_round_num'][i]} - {data['rounds_file'][i]}"


# ---------------------------------------------------------------------- map background

class Background:
    """Radar image (or a density map of player positions) drawn in game units."""

    def __init__(self, data):
        self.image, self.extent = None, None
        maps_dir = Path.home() / ".awpy" / "maps"
        image_path = maps_dir / f"{MAP}.png"
        drawn = ROOT / "assets" / f"{MAP}_map"  # from build_map.py
        if drawn.with_suffix(".png").exists():
            self.image = plt.imread(drawn.with_suffix(".png"))
            self.extent = tuple(json.loads(drawn.with_suffix(".json").read_text())["extent"])
        elif image_path.exists():
            settings = dict(RADAR_FALLBACK)
            try:
                settings |= json.loads((maps_dir / "map-data.json").read_text())[MAP]
            except (OSError, KeyError, ValueError):
                pass
            self.image = plt.imread(image_path)
            h, w = self.image.shape[:2]
            x0, y1, s = settings["pos_x"], settings["pos_y"], settings["scale"]
            self.extent = (x0, x0 + w * s, y1 - h * s, y1)
        else:
            xy = data["all_xy"][data["all_alive"]]
            self.density, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=256)
            self.extent = (xe[0], xe[-1], ye[0], ye[-1])

    def draw(self, ax):
        if self.image is not None:
            ax.imshow(self.image, extent=self.extent, zorder=0)
        else:
            ax.imshow(np.log1p(self.density.T), origin="lower", extent=self.extent, cmap="Greys", zorder=0)
        ax.set_xlim(self.extent[:2])
        ax.set_ylim(self.extent[2:])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])


# ---------------------------------------------------------------------- one round

SMOKE_RADIUS, INFERNO_RADIUS = 144, 120  # game units, roughly


def draw_utility(ax, data, t, pov):
    u = data["utility"][t]
    for row in u[u[:, col(data, "utility", "valid")] > 0]:
        x, y = row[col(data, "utility", "X")], row[col(data, "utility", "Y")]
        own = bool(row[col(data, "utility", "thrown_by_t")]) == (pov == 0)
        edge = "tab:blue" if own else "tab:red"
        if row[col(data, "utility", "in_flight")]:
            vx, vy = row[col(data, "utility", "vel_x")], row[col(data, "utility", "vel_y")]
            ax.annotate("", (x + 0.3 * vx, y + 0.3 * vy), (x, y), arrowprops=dict(arrowstyle="->", color=edge, lw=1), zorder=5)
            ax.scatter(x, y, s=15, color=edge, zorder=5)
        elif row[col(data, "utility", "is_smoke")]:
            ax.add_patch(plt.Circle((x, y), SMOKE_RADIUS, color="lightgray", ec=edge, alpha=0.75, zorder=2))
        elif row[col(data, "utility", "is_inferno")]:
            ax.add_patch(plt.Circle((x, y), INFERNO_RADIUS, color="orange", ec=edge, alpha=0.55, zorder=2))
        else:  # flash / HE / decoy pop
            ax.scatter(x, y, marker="*", s=120, color="yellow" if row[col(data, "utility", "is_flash")] else "orangered",
                       edgecolor=edge, zorder=5)


def draw_bomb(ax, data, t):
    g = data["globals"][t]
    if g[col(data, "global", "bomb_planted")] or g[col(data, "global", "bomb_dropped")]:
        ax.scatter(g[col(data, "global", "bomb_x")], g[col(data, "global", "bomb_y")], marker="s", s=70,
                   color="black", edgecolor="yellow", zorder=6, label="bomb")


def view_line(ax, data, t, slot, color, length=150):
    x, y = data["all_xy"][t, slot]
    yaw = np.deg2rad(data["player"][t, slot, col(data, "player", "yaw")])
    ax.plot([x, x + length * np.cos(yaw)], [y, y + length * np.sin(yaw)], color=color, lw=1, zorder=5)


def draw_step(ax, data, background, i, step, trail=10, show_all_paths=True, team_view=False):
    """Teammates' 6 predicted paths (opacity = probability), what really happened (dashed),
    their recent trail and view direction, grenades, the bomb and the enemies.

    Enemies: solid x = spotted by the team right now; faint x = true position while hidden
    (omitted with team_view); hollow circle = where the team last saw a hidden enemy.
    """
    s = round_slice(data, i)
    T = s.stop - s.start
    step = min(max(step, 0), T - 1)
    t = s.start + step
    pov = int(data["rounds_pov"][i])
    ax.clear()
    background.draw(ax)
    draw_utility(ax, data, t, pov)
    draw_bomb(ax, data, t)

    names = data["rounds_names"][i]
    alive = data["all_alive"][t]
    xy = data["all_xy"][t]
    weapons = data["weapon_vocab"]
    hp = col(data, "player", "health")
    for e in range(5, 10):
        if not alive[e]:
            continue
        seen = data["enemy_visible"][t, e]
        if seen or not team_view:
            ax.scatter(*xy[e], marker="x", s=60, c="red", alpha=1.0 if seen else 0.35, zorder=5)
            view_line(ax, data, t, e, "red")
        if not seen and not np.isnan(data["seen_age"][t, e]):
            ax.scatter(*data["last_seen"][t, e, :2], s=70, facecolor="none", edgecolor="red", ls="--", zorder=5)
            ax.annotate(f"{data['seen_age'][t, e]:.0f}s", data["last_seen"][t, e, :2], fontsize=6, color="red", zorder=6)
    ax.scatter([], [], marker="x", c="red", label="enemy spotted (faint: hidden, true position)")
    ax.scatter([], [], facecolor="none", edgecolor="red", label="enemy last seen (age)")

    for p in range(5):
        if not data["anchor_alive"][t, p]:
            continue
        color = COLORS[p]
        past = data["all_xy"][max(s.start, t - trail):t + 1, p]
        ax.plot(past[:, 0], past[:, 1], color=color, lw=1.5, alpha=0.6, zorder=3)
        anchor = data["anchor_xy"][t, p]
        probs = data["pred_probs"][t, p]
        paths = anchor + data["pred_offsets"][t, p].astype(np.float32)  # [M, H, 2]
        modes = range(len(probs)) if show_all_paths else [int(probs.argmax())]
        for m in modes:
            ax.plot(paths[m, :, 0], paths[m, :, 1], color=color, lw=1 + 3 * probs[m],
                    alpha=0.25 + 0.75 * probs[m] / probs.max(), zorder=4)
            ax.scatter(*paths[m, -1], color=color, s=10 + 60 * probs[m], zorder=4)
        mask = data["target_mask"][t, p]
        future = np.vstack([anchor, data["target_xy"][t, p][mask]])
        ax.plot(future[:, 0], future[:, 1], color="black", lw=1.2, ls="--", zorder=4)
        view_line(ax, data, t, p, color)
        weapon = weapons[data["player_cat"][t, p, 0]]
        ax.scatter(*anchor, color=color, edgecolor="black", s=80, zorder=6,
                   label=f"{names[p]} {data['player'][t, p, hp]:.0f}hp {weapon} (p top {probs.max():.2f})")

    clock = data["clock"][t]
    planted = " - bomb planted" if data["bomb_planted"][t] else ""
    own, enemy = data["side"][t, :, col(data, "side", "score")].astype(int)
    alive_own, alive_enemy = data["side"][t, :, col(data, "side", "alive")].astype(int)
    winner = int(data["rounds_winner"][i])
    result = "" if winner < 0 else (" - won" if winner == pov else " - lost")
    ax.set_title(f"{round_title(data, i)}\nscore {own}-{enemy}, alive {alive_own}v{alive_enemy}{result}\n"
                 f"step {step}/{T - 1}, {clock:.1f} s into the round{planted}\n"
                 "solid: 6 predicted 10 s paths (thicker = likelier)   dashed: what happened", fontsize=8)
    ax.legend(loc="lower left", fontsize=6, framealpha=0.8)


def plot_round(data, i, out_dir, step=None, gif=False, fps=4, top_only=False, team_view=False):
    background = Background(data)
    fig, ax = plt.subplots(figsize=(9, 9))
    T = int(data["rounds_n_steps"][i])
    stem = f"round{i:04d}_{data['rounds_file'][i].removesuffix('.npz')}_r{data['rounds_round_num'][i]}_{SIDES[data['rounds_pov'][i]]}"
    if gif:
        from matplotlib.animation import FuncAnimation, PillowWriter

        anim = FuncAnimation(fig, lambda k: draw_step(ax, data, background, i, k, show_all_paths=not top_only,
                                                    team_view=team_view), frames=T)
        path = out_dir / f"{stem}.gif"
        anim.save(path, writer=PillowWriter(fps=fps), dpi=80)
    else:
        step = T // 2 if step is None else step
        draw_step(ax, data, background, i, step, show_all_paths=not top_only, team_view=team_view)
        path = out_dir / f"{stem}_step{step:03d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------- metrics

def errors(data, chunk=4096):
    """Per (step, teammate) and future step h: best-of-6 and top-1 errors in game units.

    Returns fde_min, fde_top, ade_min [N, 5, H], valid [N, 5, H] (alive and in the round
    through h) and p_top [N, 5] (the most likely path's probability), top_is_best [N, 5, H].
    """
    N, _, M, H, _ = data["pred_offsets"].shape
    fde_min, fde_top, ade_min = (np.zeros((N, 5, H), np.float32) for _ in range(3))
    top_is_best = np.zeros((N, 5, H), bool)
    top = data["pred_probs"].argmax(-1)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        pred = data["pred_offsets"][a:b].astype(np.float32)  # [n, 5, M, H, 2]
        target = data["target_xy"][a:b] - data["anchor_xy"][a:b, :, None]  # [n, 5, H, 2]
        err = np.linalg.norm(pred - target[:, :, None], axis=-1)  # [n, 5, M, H]
        ade = np.cumsum(err, -1) / np.arange(1, H + 1)
        fde_min[a:b] = err.min(2)
        ade_min[a:b] = ade.min(2)
        fde_top[a:b] = np.take_along_axis(err, top[a:b, :, None, None], 2)[:, :, 0]
        top_is_best[a:b] = err.argmin(2) == top[a:b, :, None]
    valid = data["target_mask"] & data["anchor_alive"][..., None]
    p_top = data["pred_probs"].max(-1)
    return fde_min, fde_top, ade_min, valid, p_top, top_is_best


def summarize(fde_min, fde_top, ade_min, valid, h, sel):
    ok = valid[..., h - 1] & sel
    n = int(ok.sum())
    if n == 0:
        return {"samples": 0}
    return {"samples": n, "minADE@6": float(ade_min[..., h - 1][ok].mean()), "minFDE@6": float(fde_min[..., h - 1][ok].mean()),
            "FDE top-1": float(fde_top[..., h - 1][ok].mean()),
            "miss rate %": float(100 * (fde_min[..., h - 1][ok] > MISS_UNITS).mean())}


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def plot_metrics(data, out_dir, history_path=None):
    fde_min, fde_top, ade_min, valid, p_top, top_is_best = errors(data)
    N, _, H = valid.shape
    step = data["step_seconds"]
    seconds = np.arange(1, H + 1) * step

    # Which round / side each (step, teammate) belongs to.
    pov = np.repeat(data["rounds_pov"], data["rounds_n_steps"])[:, None].repeat(5, 1)
    moving = data["anchor_speed"] >= MOVING_SPEED
    everyone = np.ones((N, 5), bool)

    # 1. Error at every 0.5 s of the horizon.
    rows = []
    for h in range(1, H + 1):
        for name, sel in (("all", everyone), ("moving", moving)):
            rows.append({"seconds": h * step, "subset": name} | summarize(fde_min, fde_top, ade_min, valid, h, sel))
    write_csv(out_dir / "error_by_horizon.csv", rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, ls in (("all", "-"), ("moving", "--")):
        part = [r for r in rows if r["subset"] == name and r["samples"]]
        x = [r["seconds"] for r in part]
        ax.plot(x, [r["minFDE@6"] for r in part], ls=ls, color=COLORS[0], label=f"minFDE@6 ({name})")
        ax.plot(x, [r["FDE top-1"] for r in part], ls=ls, color=COLORS[1], label=f"FDE top-1 ({name})")
        ax.plot(x, [r["minADE@6"] for r in part], ls=ls, color=COLORS[2], label=f"minADE@6 ({name})")
    ax.axhline(MISS_UNITS, color="gray", lw=0.8, ls=":", label=f"miss threshold ({MISS_UNITS} u)")
    ax.set_xlabel("seconds ahead")
    ax.set_ylabel("error (game units, 1 u ~ 1.9 cm)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "error_by_horizon.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Breakdowns at 3, 5 and 10 s: side, time in round, bomb, callout.
    clock = data["clock"][:, None].repeat(5, 1)
    planted = data["bomb_planted"][:, None].repeat(5, 1)
    groups = [("all", "all", everyone), ("moving", "moving", moving), ("still", "still", ~moving)]
    groups += [("side", s, pov == k) for k, s in enumerate(SIDES)]
    groups += [("bomb", "planted" if v else "not planted", planted == v) for v in (False, True)]
    for lo in range(0, 120, 15):
        groups.append(("clock", f"{lo}-{lo + 15}s", (clock >= lo) & (clock < lo + 15)))
    place_vocab = data["place_vocab"]
    place = data["anchor_place"]
    counts = np.bincount(place[data["anchor_alive"]].ravel(), minlength=len(place_vocab))
    for pid in np.argsort(-counts)[:25]:
        if pid > 0 and counts[pid]:
            groups.append(("callout", str(place_vocab[pid]), place == pid))
    alive = data["side"][:, :, col(data, "side", "alive")].astype(int)
    advantage = (alive[:, 0] - alive[:, 1])[:, None].repeat(5, 1)
    for a in range(-4, 5):
        groups.append(("advantage", f"{a:+d}", advantage == a))
    weapon = data["player_cat"][:, :5, 0]
    counts = np.bincount(weapon[data["anchor_alive"]].ravel(), minlength=len(data["weapon_vocab"]))
    for wid in np.argsort(-counts)[:15]:
        if counts[wid]:
            groups.append(("weapon", str(data["weapon_vocab"][wid]), weapon == wid))
    names = np.repeat(data["rounds_names"][:, :5], data["rounds_n_steps"], axis=0)
    for name in np.unique(names):
        if name:
            groups.append(("player", str(name), names == name))
    rows = []
    for sec, h in data["horizons"].items():
        for kind, value, sel in groups:
            rows.append({"horizon s": sec, "group": kind, "value": value} | summarize(fde_min, fde_top, ade_min, valid, h, sel))
    write_csv(out_dir / "breakdown.csv", rows)

    h10 = data["horizons"][10]
    for kind in ("side", "clock", "callout", "bomb", "advantage", "weapon", "player"):
        part = [r for r in rows if r["horizon s"] == 10 and r["group"] == kind and r["samples"] >= 50]
        if kind in ("callout", "weapon", "player"):
            part.sort(key=lambda r: r["minFDE@6"])
        fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(part) + 2), 4.5))
        x = np.arange(len(part))
        ax.bar(x - 0.2, [r["minFDE@6"] for r in part], 0.4, label="minFDE@6")
        ax.bar(x + 0.2, [r["FDE top-1"] for r in part], 0.4, label="FDE top-1")
        ax.set_xticks(x, [r["value"] for r in part], rotation=60 if len(part) > 9 else 0, ha="right" if len(part) > 9 else "center", fontsize=8)
        ax.set_ylabel("10 s error (game units)")
        ax.set_title(f"10 s error by {kind}" + (" (own alive - enemy alive)" if kind == "advantage" else ""))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
        fig.savefig(out_dir / f"error_by_{kind}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 3. Where on the map the 10 s best-of-6 error is large.
    ok = valid[..., h10 - 1]
    xy, err = data["anchor_xy"][ok], fde_min[..., h10 - 1][ok]
    background = Background(data)
    bins = 64
    x0, x1, y0, y1 = background.extent
    total, xe, ye = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[x0, x1], [y0, y1]], weights=err)
    count, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=bins, range=[[x0, x1], [y0, y1]])
    mean = np.where(count >= 20, total / np.maximum(count, 1), np.nan)
    fig, ax = plt.subplots(figsize=(9, 9))
    background.draw(ax)
    im = ax.imshow(mean.T, origin="lower", extent=(xe[0], xe[-1], ye[0], ye[-1]), cmap="magma", alpha=0.75, zorder=1)
    fig.colorbar(im, ax=ax, fraction=0.04, label="mean minFDE@6 at 10 s (game units), by where the player stood")
    ax.set_title("Where 10 s predictions go wrong (cells with >= 20 samples)")
    fig.savefig(out_dir / "error_map_10s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Are the path probabilities calibrated? P(top path is the closest one) vs its predicted probability.
    ok = valid[..., h10 - 1]
    p, hit = p_top[ok], top_is_best[..., h10 - 1][ok]
    edges = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, edges) - 1, 0, 9)
    centers, freq, n = [], [], []
    for k in range(10):
        sel = idx == k
        if sel.sum() >= 20:
            centers.append(p[sel].mean())
            freq.append(hit[sel].mean())
            n.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.plot(centers, freq, marker="o")
    ax.set_xlabel("probability of the most likely path")
    ax.set_ylabel("how often it was the closest path (10 s)")
    ax.set_title("Path probability calibration")
    ax.grid(alpha=0.3)
    fig.savefig(out_dir / "calibration_10s.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    write_csv(out_dir / "calibration_10s.csv", [{"p top (mean)": c, "top was closest": f, "samples": k} for c, f, k in zip(centers, freq, n)])

    # 5. Training curves.
    if history_path and Path(history_path).exists():
        history = json.loads(Path(history_path).read_text())
        epochs = [e["epoch"] for e in history]
        fig, ax1 = plt.subplots(figsize=(7, 4.5))
        ax1.plot(epochs, [e["loss"] for e in history], color=COLORS[0], label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("train loss", color=COLORS[0])
        ax2 = ax1.twinx()
        ax2.plot(epochs, [e["val_minADE@6_10s"] for e in history], color=COLORS[1], label="val minADE@6 (10 s)")
        if all("val" in e for e in history):
            ax2.plot(epochs, [e["val"]["all 10s"]["minFDE@6"] for e in history], color=COLORS[2], label="val minFDE@6 (10 s)")
        ax2.set_ylabel("val error (game units)")
        fig.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)
        fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_dir / 'training_curves.png'}")
    print(f"plots in {out_dir}/")


# ---------------------------------------------------------------------- cli

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["rounds", "round", "metrics"])
    parser.add_argument("predictions", help="a .predictions_<split>.npz file from transformer.py")
    parser.add_argument("round", nargs="?", type=int, help="round index from `rounds` (round command)")
    parser.add_argument("--step", type=int, help="step to draw (default: middle of the round)")
    parser.add_argument("--gif", action="store_true", help="animate the whole round")
    parser.add_argument("--fps", type=int, default=4, help="gif frames per second (one frame = 0.5 s of game time)")
    parser.add_argument("--top-only", action="store_true", help="draw only the most likely path per player")
    parser.add_argument("--team-view", action="store_true", help="hide hidden enemies' true positions (show what the team knew)")
    parser.add_argument("--history", help="history.json for training curves (default: next to the predictions)")
    parser.add_argument("--out", help="output folder (default: plots/<predictions file name>/)")
    args = parser.parse_args()

    path = Path(args.predictions)
    data = load(path)
    if args.command == "rounds":
        for i in range(len(data["rounds_n_steps"])):
            print(f"{i:4d}  {data['rounds_n_steps'][i]:4d} steps  {round_title(data, i)}")
        return

    out_dir = Path(args.out or ROOT / "plots" / path.name.removesuffix(".npz"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "round":
        if args.round is None:
            parser.error("round needs a round index (see the rounds command)")
        plot_round(data, args.round, out_dir, args.step, args.gif, args.fps, args.top_only, args.team_view)
    else:
        history = args.history or path.with_name(path.name.split(".predictions_")[0] + ".history.json")
        plot_metrics(data, out_dir, history)


if __name__ == "__main__":
    main()
