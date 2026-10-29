"""Draw a sharp, dark-themed map of Mirage from the game's own radar image.

The radar (from awpy's data release: `awpy get`, or the files in ~/.awpy/<version>/)
gives the real layout: walls, boxes, stairs and bombsite markings. Its outline is traced
as straight-edged vector polygons and drawn at high resolution, so walls stay hard at any
zoom; the floor is recoloured to the app's slate palette while keeping the radar's detail.
Callout labels go at the middle of each callout's positions in processed/.

Writes assets/de_mirage_map.png and assets/de_mirage_map.json (extent in game units,
callout label positions), which app.py and visualize.py use as the map background.

Usage:
    python build_map.py
    python build_map.py --radar path/to/de_mirage.png --scale 3
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from scipy import ndimage
from skimage import measure, morphology

from features import P

ROOT = Path(__file__).parent
MAP = "de_mirage"
BG = "#0E1117"
FLOOR = LinearSegmentedColormap.from_list("floor", ["#141922", "#222A36", "#323D4C", "#4A5768", "#6B7A8F", "#97A6B9"])
EDGE = "#B4C4D6"


def find_radar():
    """Newest awpy radar image and its overview settings (pos_x, pos_y, scale)."""
    for version in sorted((Path.home() / ".awpy").glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        radar, meta = version / "radars" / f"{MAP}.png", version / "map_data.json"
        if radar.exists() and meta.exists():
            return radar, json.loads(meta.read_text())[MAP]
    old = Path.home() / ".awpy" / "maps"  # awpy 2 layout
    if (old / f"{MAP}.png").exists():
        return old / f"{MAP}.png", json.loads((old / "map-data.json").read_text())[MAP]
    raise SystemExit("No radar image found: download awpy's map data (images.zip and map_data.json) into ~/.awpy/<version>/")


def callouts(processed):
    """Label position per callout: the median of every living player position there."""
    xy, place = [], []
    for path in sorted(Path(processed).glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as z:
            p = z["player"]
            alive = p[:, :, P["is_alive"]] > 0
            xy.append(p[alive][:, [P["X"], P["Y"]]])
            place.append(z["place_strings"][z["player_cat"][:, :, 1][alive]])
    if not xy:
        return []
    xy, place = np.concatenate(xy), np.concatenate(place)
    out = []
    for name in np.unique(place):
        pts = xy[place == name]
        if name and len(pts) >= 2000:
            out.append({"name": str(name), "label": pretty(str(name)), "x": float(np.median(pts[:, 0])),
                        "y": float(np.median(pts[:, 1])), "n": int(len(pts))})
    return out


def pretty(name):
    """'TopofMid' -> 'Top of Mid', 'CTSpawn' -> 'CT Spawn', 'BombsiteA' -> 'A Site'."""
    name = re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[A-Z])(?=[A-Z][a-z])", " ", name)
    name = re.sub(r"(?<=[a-z])of ", " of ", name)
    return re.sub(r"Bombsite ([AB])", r"\1 Site", name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--radar", help="radar .png (default: newest in ~/.awpy)")
    parser.add_argument("--processed", default=str(ROOT / "processed"), help="for callout label positions")
    parser.add_argument("--out", default=str(ROOT / "assets" / f"{MAP}_map"))
    parser.add_argument("--scale", type=float, default=2.4, help="output pixels per radar pixel")
    args = parser.parse_args()

    radar_path, settings = find_radar()
    if args.radar:
        radar_path = Path(args.radar)
    rgba = np.asarray(Image.open(radar_path).convert("RGBA")).astype(np.float32) / 255
    h, w = rgba.shape[:2]
    x0, y1, s = settings["pos_x"], settings["pos_y"], settings["scale"]
    extent = (x0, x0 + w * s, y1 - h * s, y1)

    # Map area: opaque radar pixels, minus the faint background grid dots.
    solid = morphology.remove_small_objects(rgba[..., 3] > 0.5, max_size=400)
    # Enclosed transparent slits (around the bombsite markers) are floor; thick holes are pillars.
    holes, n = ndimage.label(ndimage.binary_fill_holes(solid) & ~solid)
    thickness = ndimage.maximum(ndimage.distance_transform_edt(holes > 0), holes, np.arange(1, n + 1))
    slits = np.isin(holes, np.arange(1, n + 1)[np.asarray(thickness) <= 4])
    solid |= slits
    alpha = np.where(slits, 1.0, np.where(solid, rgba[..., 3], 0))

    # Floor: the radar's brightness through the slate palette, a little of its own colour kept
    # (bombsite and spawn markings), with contrast stretched over the map area.
    lum = rgba[..., :3] @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(lum[solid], [1, 99.5])
    _, nearest = ndimage.distance_transform_edt(slits | ~solid, return_indices=True)  # fill slits from the floor next to them
    lum = np.where(slits, lum[tuple(nearest)], lum)
    floor = FLOOR(np.clip((lum - lo) / (hi - lo), 0, 1))[..., :3]
    color = np.where(slits[..., None], rgba[..., :3][tuple(nearest)], rgba[..., :3])
    floor = 0.8 * floor + 0.2 * color
    image = np.dstack([floor, alpha])

    W, H = int(w * args.scale), int(h * args.scale)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.imshow(image, extent=extent, interpolation="lanczos", zorder=1)

    # Walls: straight-edged outline of the map area, a soft shadow under a sharp light edge.
    for contour in measure.find_contours(alpha, 0.5):
        if len(contour) < 12:
            continue
        contour = measure.approximate_polygon(contour, tolerance=0.6)
        row, col = contour[:, 0], contour[:, 1]
        px, py = x0 + (col + 0.5) * s, y1 - (row + 0.5) * s
        ax.plot(px, py, color="black", lw=4.5, alpha=0.45, solid_joinstyle="miter", zorder=2)
        ax.plot(px, py, color=EDGE, lw=1.4, alpha=0.95, solid_joinstyle="miter", zorder=3)

    ax.set_xlim(extent[:2])
    ax.set_ylim(extent[2:])
    ax.axis("off")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=100, facecolor=BG)
    plt.close(fig)

    labels = callouts(args.processed)
    meta = {"map": MAP, "extent": [float(v) for v in extent], "radar": str(radar_path), "callouts": labels}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out.with_suffix('.png')} ({out.with_suffix('.png').stat().st_size / 1e6:.1f} MB, {W}x{H}) "
          f"and {out.with_suffix('.json')} ({len(labels)} callouts)")


if __name__ == "__main__":
    main()
