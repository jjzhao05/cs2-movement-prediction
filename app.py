"""Mirage Replay: an interactive 2D replay of the transformer's saved predictions.

    streamlit run app.py
    streamlit run app.py -- checkpoints/transformer_full.predictions_test.npz

Pick a predictions file and a round in the sidebar. The round plays in the browser
(assets/replay.html), smoothly interpolated between the 0.5 s samples like a video: every
teammate's 6 predicted 10 s paths (arrows, opacity = probability), what really happened,
enemies (true positions or only what the team knew), smokes, fires, flashes, the bomb and
view directions, with both teams' loadouts, the focused player's path probabilities and the
error at 3 / 5 / 10 s alongside. Space plays / pauses, arrow keys step, scroll zooms.

Everything comes from the .predictions_<split>.npz file; no model or GPU is needed.
The map is assets/de_mirage_map.png from build_map.py (Mirage's radar, redrawn); awpy's
original radar is a second style when its data is in ~/.awpy/<version>/.
"""

import base64
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from build_map import find_radar
from visualize import MAP, SIDES, col, errors, load

ROOT = Path(__file__).parent
SIDE_COLORS = {"T": "#E3B341", "CT": "#5DADE2"}
PLAYER = ROOT / "assets" / "replay.html"
BG = "#0E1117"
UTILITY_KINDS = ("smoke", "inferno", "flash", "he", "decoy")

st.set_page_config(page_title="Mirage Replay", page_icon="🗺️", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 0.5rem; max-width: 100%;}
.badge {display:inline-block; padding:2px 9px; border-radius:999px; font-size:0.78rem; font-weight:600; margin-right:6px;}
.psub {color:#8B94A3; font-size:0.8rem;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------- data

@st.cache_resource(show_spinner="Loading predictions…")
def open_file(path: str):
    data = load(path)
    fde_min, fde_top, _, valid, _, _ = errors(data)
    return data, fde_min, fde_top, valid


DRAWN_MAP = ROOT / "assets" / f"{MAP}_map"  # from build_map.py


def awpy_radar():
    """(radar png, overview settings) from awpy's data in ~/.awpy, or None."""
    try:
        return find_radar()
    except (SystemExit, OSError, KeyError, ValueError):
        return None


def map_styles():
    return [name for name, ok in (("Drawn", DRAWN_MAP.with_suffix(".png").exists()), ("Radar", awpy_radar() is not None))
            if ok] or ["Positions"]


@st.cache_data(show_spinner=False)
def callouts():
    try:
        return json.loads(DRAWN_MAP.with_suffix(".json").read_text())["callouts"]
    except (OSError, KeyError, ValueError):
        return []


@st.cache_data(show_spinner=False)
def radar_image(path: str, style: str):
    """(data URI, extent x0 x1 y0 y1) of the map background, as a compact PNG."""
    from PIL import Image

    data = open_file(path)[0]
    if style == "Drawn":
        image = Image.open(DRAWN_MAP.with_suffix(".png")).convert("RGB")
        extent = tuple(json.loads(DRAWN_MAP.with_suffix(".json").read_text())["extent"])
    elif style == "Radar":
        image_path, settings = awpy_radar()
        radar = Image.open(image_path).convert("RGBA")
        image = Image.new("RGBA", radar.size, BG)
        image.alpha_composite(radar)
        image = image.convert("RGB")
        w, h = image.size
        x0, y1, s = settings["pos_x"], settings["pos_y"], settings["scale"]
        extent = (x0, x0 + w * s, y1 - h * s, y1)
    else:  # draw the walkable area from every position in the file
        from matplotlib.colors import LinearSegmentedColormap
        from scipy.ndimage import gaussian_filter

        xy = data["all_xy"][data["all_alive"]]
        pad, cell = 150, 8
        x0, x1 = xy[:, 0].min() - pad, xy[:, 0].max() + pad
        y0, y1 = xy[:, 1].min() - pad, xy[:, 1].max() + pad
        density, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(int((x1 - x0) / cell), int((y1 - y0) / cell)),
                                       range=[[x0, x1], [y0, y1]])
        floor = gaussian_filter((density > 0).astype(float), 2.0)
        heat = gaussian_filter(np.log1p(density), 1.5)
        values = np.clip(floor * 1.6, 0, 1) * 0.55 + heat / max(heat.max(), 1e-9) * 0.45
        cmap = LinearSegmentedColormap.from_list("floor", [BG, "#232A35", "#4A5464", "#7D8898"])
        image = Image.fromarray((cmap(values.T[::-1])[..., :3] * 255).astype(np.uint8))
        extent = (float(x0), float(x1), float(y0), float(y1))
    image.thumbnail((1800, 1800))
    buf = io.BytesIO()
    image.quantize(colors=96, method=Image.Quantize.MEDIANCUT).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), extent


@st.cache_data(show_spinner=False)
def round_table(path: str):
    data, fde_min, fde_top, valid = open_file(path)
    h10 = data["horizons"][10] - 1
    rows = []
    for i, (start, n) in enumerate(zip(data["rounds_start"].tolist(), data["rounds_n_steps"].tolist())):
        s = slice(start, start + n)
        ok = valid[s, :, h10]
        winner = int(data["rounds_winner"][i])
        rows.append({
            "index": i, "match": str(data["rounds_match"][i]), "round": int(data["rounds_round_num"][i]),
            "side": SIDES[int(data["rounds_pov"][i])], "team": str(data["rounds_teams"][i][0]),
            "enemy": str(data["rounds_teams"][i][1]),
            "result": "?" if winner < 0 else ("Won" if winner == data["rounds_pov"][i] else "Lost"),
            "length": n * data["step_seconds"],
            "err10": float(fde_min[s, :, h10][ok].mean()) if ok.any() else np.nan,
        })
    return rows


MODEL_NAMES = {"transformer_full": "Transformer", "transformer_3": "Transformer (3 s history)",
               "transformer_10": "Transformer (10 s history)", "transformer_full_fullinfo": "Transformer (full info)",
               "knn": "kNN (k=6)", "per_player_gru": "Per-player GRU", "constant_velocity": "Constant velocity",
               "stay_still": "Stay still"}


def model_name(path):
    """'checkpoints/knn.predictions_test.npz' -> 'kNN (k=6) · test'."""
    stem, _, split = Path(path).name.removesuffix(".npz").partition(".predictions_")
    return f"{MODEL_NAMES.get(stem, stem)} · {split}" if split else Path(path).name


def prediction_files():
    """Transformer first, then the baselines from strongest to simplest, then anything else."""
    files = list((ROOT / "checkpoints").glob("*.predictions_*.npz"))
    order = list(MODEL_NAMES)
    files.sort(key=lambda p: (order.index(p.name.split(".")[0]) if p.name.split(".")[0] in order else len(order), p.name))
    extra = [Path(a) for a in sys.argv[1:] if a.endswith(".npz")]
    return [str(p) for p in extra + files]


# ---------------------------------------------------------------------- round data for the player

@st.cache_data(show_spinner="Preparing round…", max_entries=24)
def round_payload(path: str, i: int, style: str):
    """Everything the browser player needs for one round, as compact JSON (ints where possible)."""
    data, fde_min, fde_top, valid = open_file(path)
    start, n = int(data["rounds_start"][i]), int(data["rounds_n_steps"][i])
    s = slice(start, start + n)
    pov = int(data["rounds_pov"][i])
    Pf = lambda name: data["player"][s, :, col(data, "player", name)]  # noqa: E731
    ints = lambda a: np.rint(a).astype(int).tolist()  # noqa: E731

    U = lambda name: col(data, "utility", name)  # noqa: E731
    utility = []
    for u in data["utility"][s]:
        items = []
        for row in u[u[:, U("valid")] > 0]:
            kind = next((k for k, name in enumerate(UTILITY_KINDS) if row[U(f"is_{name}")]), 4)
            items.append([int(row[U("X")]), int(row[U("Y")]), kind, int(bool(row[U("thrown_by_t")]) == (pov == 0)),
                          int(row[U("in_flight")]), int(row[U("vel_x")]), int(row[U("vel_y")]), round(float(row[U("age")]), 1)])
        utility.append(items)

    g = data["globals"][s]
    G = lambda name: g[:, col(data, "global", name)]  # noqa: E731
    bomb_state = np.where(G("bomb_planted") > 0, 2, np.where(G("bomb_dropped") > 0, 1, 0))
    planted = np.nonzero(bomb_state == 2)[0]

    mask = data["target_mask"][s]
    target = np.rint(data["target_xy"][s]).astype(int)
    truth = [[target[t, p][mask[t, p]].tolist() for p in range(5)] for t in range(n)]
    seen_age = data["seen_age"][s]
    last_seen = np.rint(data["last_seen"][s][..., :2]).astype(int)

    def err(arr, sec):
        h = data["horizons"][sec] - 1
        return np.where(valid[s, :, h], np.rint(arr[s, :, h]), -1).astype(int).tolist()

    h10 = data["horizons"][10] - 1
    ok = valid[s, :, h10]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # steps where nobody has a 10 s future
        team_min = np.nanmean(np.where(ok, fde_min[s, :, h10], np.nan), 1)
        team_top = np.nanmean(np.where(ok, fde_top[s, :, h10], np.nan), 1)
    uri, extent = radar_image(path, style)
    winner = int(data["rounds_winner"][i])
    return json.dumps({
        "meta": {"team": str(data["rounds_teams"][i][0]), "enemy": str(data["rounds_teams"][i][1]), "side": SIDES[pov],
                 "round": int(data["rounds_round_num"][i]), "won": None if winner < 0 else winner == pov},
        "image": uri, "extent": list(extent), "callouts": [{k: c[k] for k in ("label", "x", "y")} for c in callouts()],
        "steps": n, "stepSeconds": data["step_seconds"], "horizon": int(data["pred_offsets"].shape[3]),
        "names": [str(v) for v in data["rounds_names"][i]],
        "weapons": [str(v) for v in data["weapon_vocab"]], "places": [str(v) for v in data["place_vocab"]],
        "clock": np.round(data["clock"][s], 2).tolist(),
        "score": data["side"][s, :, col(data, "side", "score")].astype(int).tolist(),
        "aliveCount": data["side"][s, :, col(data, "side", "alive")].astype(int).tolist(),
        "xy": ints(data["all_xy"][s]), "alive": data["all_alive"][s].astype(int).tolist(),
        "hp": ints(Pf("health")), "armor": ints(Pf("armor")), "helmet": ints(Pf("has_helmet")), "kit": ints(Pf("has_defuser")),
        "yaw": ints(Pf("yaw")), "money": ints(Pf("money")), "hasBomb": ints(Pf("has_bomb")),
        "nades": np.stack([Pf(k) for k in ("n_flash", "n_smoke", "n_he", "n_molotov")], -1).astype(int).tolist(),
        "weapon": data["player_cat"][s][:, :, 0].astype(int).tolist(), "place": data["player_cat"][s][:, :, 1].astype(int).tolist(),
        "visible": data["enemy_visible"][s].astype(int).tolist(),
        "lastSeen": [[None if np.isnan(seen_age[t, p]) else [*last_seen[t, p].tolist(), round(float(seen_age[t, p]), 1)]
                      for p in range(10)] for t in range(n)],
        "utility": utility,
        "bomb": [[int(x), int(y), int(b)] for x, y, b in zip(G("bomb_x"), G("bomb_y"), bomb_state)],
        "plantStep": int(planted[0]) if len(planted) else -1,
        "paths": ints(data["anchor_xy"][s][:, :, None, None] + data["pred_offsets"][s].astype(np.float32)),
        "probs": np.round(data["pred_probs"][s], 3).tolist(),
        "endPlace": data["pred_end_place"][s].astype(int).tolist() if "pred_end_place" in data else None,
        "truth": truth,
        "err": {sec: {"min": err(fde_min, sec), "top": err(fde_top, sec)} for sec in (3, 5, 10)},
        "teamErr": {"min": np.nan_to_num(np.rint(team_min), nan=-1).astype(int).tolist(),
                    "top": np.nan_to_num(np.rint(team_top), nan=-1).astype(int).tolist()},
    }, separators=(",", ":"))


# ---------------------------------------------------------------------- app

def main():
    files = prediction_files()
    with st.sidebar:
        st.markdown("## 🗺️ Mirage Replay")
        if files:
            path = st.selectbox("Model", files, format_func=model_name,
                                help="Every .predictions_<split>.npz in checkpoints/: the transformer (transformer.py) "
                                     "and the baselines (last cells of baselines.ipynb).")
        else:
            path = st.text_input("Predictions file", placeholder="checkpoints/transformer_full.predictions_test.npz")
        if not path or not Path(path).exists():
            st.info("Train with transformer.py, then pick its .predictions_<split>.npz file here "
                    "(or pass it: `streamlit run app.py -- <file>`).")
            st.stop()

        data, fde_min, fde_top, valid = open_file(path)
        rounds = round_table(path)

        st.markdown("#### Round")
        matches = sorted({r["match"] for r in rounds})
        match = st.selectbox("Match", ["All matches"] + matches)
        c1, c2 = st.columns(2)
        side = c1.segmented_control("Side", ["T", "CT"], selection_mode="multi", default=["T", "CT"])
        result = c2.segmented_control("Result", ["Won", "Lost"], selection_mode="multi", default=["Won", "Lost"])
        sort = st.selectbox("Sort by", ["Match order", "Hardest to predict", "Easiest to predict", "Longest"])
        shown = [r for r in rounds if (match == "All matches" or r["match"] == match)
                 and r["side"] in (side or SIDES) and (r["result"] in (result or ["Won", "Lost"]) or r["result"] == "?")]
        key = {"Hardest to predict": lambda r: -np.nan_to_num(r["err10"]),
               "Easiest to predict": lambda r: np.nan_to_num(r["err10"], nan=1e9), "Longest": lambda r: -r["length"]}.get(sort)
        if key:
            shown = sorted(shown, key=key)
        if not shown:
            st.warning("No rounds match these filters.")
            st.stop()
        label = lambda r: (f"R{r['round']} · {r['team']} ({r['side']}) vs {r['enemy']} · {r['result']}"  # noqa: E731
                           + ("" if np.isnan(r["err10"]) else f" · {r['err10']:.0f}u"))
        i = st.selectbox(f"Round ({len(shown)})", [r["index"] for r in shown], format_func=lambda k: label(rounds[k]))

        st.markdown("#### View")
        styles = map_styles()
        style = (st.segmented_control("Map", styles, default=styles[0]) or styles[0]) if len(styles) > 1 else styles[0]
        height = st.slider("Player height", 600, 1300, 860, 20)

        st.divider()
        h10 = data["horizons"][10] - 1
        ok = valid[:, :, h10]
        st.caption(f"{len(rounds)} round views · best-of-6 error at 10 s **{fde_min[:, :, h10][ok].mean():.0f}u**, "
                   f"most likely path **{fde_top[:, :, h10][ok].mean():.0f}u** ({int(ok.sum()):,} predictions)")

    r = rounds[i]
    side_color = SIDE_COLORS[r["side"]]
    result_color = {"Won": "#3FB950", "Lost": "#F85149"}.get(r["result"], "#8B94A3")
    st.markdown(f"""<h3 style="margin:0 0 0.1rem">{r['team']} <span class="badge" style="background:{side_color}22;color:{side_color}">{r['side']}</span>
vs {r['enemy']} · Round {r['round']} <span class="badge" style="background:{result_color}22;color:{result_color}">{r['result']}</span></h3>
<div class="psub" style="margin-bottom:0.4rem">{r['match']} · {model_name(path)} · Space: play / pause · ← →: step · scroll: zoom · click a teammate to focus</div>""",
                unsafe_allow_html=True)
    html = PLAYER.read_text(encoding="utf-8").replace("/*__DATA__*/", round_payload(path, i, style), 1)
    components.html(html, height=height, scrolling=False)


main()
