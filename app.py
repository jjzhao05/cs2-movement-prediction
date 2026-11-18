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
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import streamlit as st

from build_map import find_radar
from visualize import MAP, SIDES, col, errors, load

ROOT = Path(__file__).parent
PLAYER = ROOT / "assets" / "replay.html"
BG = "#0F1114"
UTILITY_KINDS = ("smoke", "inferno", "flash", "he", "decoy")
ICONS = ROOT / "assets" / "icons"  # CS2 HUD icons (weapons/, hud/) from github.com/lexogrine/cs2-react-hud
# weapon_vocab name -> weapons/<file>.svg; knives without their own icon use the default one.
WEAPON_ICONS = {
    "AK-47": "ak47", "AUG": "aug", "AWP": "awp", "Bayonet": "knife_bayonet", "Bowie Knife": "knife_survival_bowie",
    "Butterfly Knife": "knife_butterfly", "C4 Explosive": "c4", "CZ75-Auto": "cz75a", "Classic Knife": "knife_css",
    "Decoy Grenade": "decoy", "Desert Eagle": "deagle", "Dual Berettas": "elite", "FAMAS": "famas",
    "Falchion Knife": "knife_falchion", "Five-SeveN": "fiveseven", "Flashbang": "flashbang", "Flip Knife": "knife_flip",
    "Galil AR": "galilar", "Glock-18": "glock", "Gut Knife": "knife_gut", "High Explosive Grenade": "hegrenade",
    "Huntsman Knife": "knife_tactical", "Incendiary Grenade": "incgrenade", "Karambit": "knife_karambit",
    "Kukri Knife": "knife", "M4A1-S": "m4a1_silencer", "M4A4": "m4a1", "M9 Bayonet": "knife_m9_bayonet",
    "MAC-10": "mac10", "MAG-7": "mag7", "MP5-SD": "mp5sd", "MP7": "mp7", "MP9": "mp9", "Molotov": "molotov",
    "Navaja Knife": "knife_gypsy_jackknife", "Negev": "negev", "Nomad Knife": "knife_outdoor", "Nova": "nova",
    "P2000": "hkp2000", "P250": "p250", "P90": "p90", "Paracord Knife": "knife_cord", "R8 Revolver": "revolver",
    "SSG 08": "ssg08", "Sawed-Off": "sawedoff", "Shadow Daggers": "knife_push", "Skeleton Knife": "knife_skeleton",
    "Smoke Grenade": "smokegrenade", "Stiletto Knife": "knife_stiletto", "Survival Knife": "knife_canis",
    "Talon Knife": "knife_widowmaker", "Tec-9": "tec9", "UMP-45": "ump45", "USP-S": "usp_silencer",
    "Ursus Knife": "knife_ursus", "XM1014": "xm1014", "Zeus x27": "taser", "knife": "knife", "knife_t": "knife_t",
    "M249": "m249", "G3SG1": "g3sg1", "SCAR-20": "scar20", "SG 553": "sg556", "PP-Bizon": "bizon",
}
HUD_ICONS = {"armor": "icon_armor_full_default", "helmet": "icon_armor_helmet_default", "bomb": "icon_c4_default",
             "kit": "icon_defuse_default", "skull": "icon_skull_default"}
GRENADE_ICONS = ("flashbang", "smokegrenade", "hegrenade", "molotov")  # the order of the payload's "nades"
# Sidebar settings -> the player's options.
PATHS = {"All": "all", "Top": "top", "None": "none"}
ENEMIES = {"True positions": False, "Team's view": True}
LAYERS = {"Actual path": "truth", "Grenades": "nades", "View": "view", "Names": "names", "Callouts": "callouts"}

st.set_page_config(page_title="Mirage replay", page_icon=":material/map:", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.6rem; padding-bottom: 0.5rem; max-width: 100%;}
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
            "winner": None if winner < 0 else SIDES[winner],
            "length": n * data["step_seconds"],
            "err10": float(fde_min[s, :, h10][ok].mean()) if ok.any() else np.nan,
        })
    return rows


MODEL_NAMES = {"transformer_full": "Transformer", "transformer_3": "Transformer (3 s history)",
               "transformer_10": "Transformer (10 s history)", "transformer_full_fullinfo": "Transformer (full info)",
               "knn": "kNN (k=6)", "per_player_gru": "Per-player GRU", "constant_velocity": "Constant velocity",
               "stay_still": "Stay still"}


def model_name(path):
    """'checkpoints/knn.predictions_test.npz' -> 'kNN (k=6), test'."""
    stem, _, split = Path(path).name.removesuffix(".npz").partition(".predictions_")
    return f"{MODEL_NAMES.get(stem, stem)}, {split}" if split else Path(path).name


def prediction_files():
    """Transformer first, then the baselines from strongest to simplest, then anything else."""
    files = list((ROOT / "checkpoints").glob("*.predictions_*.npz"))
    order = list(MODEL_NAMES)
    files.sort(key=lambda p: (order.index(p.name.split(".")[0]) if p.name.split(".")[0] in order else len(order), p.name))
    extra = [Path(a) for a in sys.argv[1:] if a.endswith(".npz")]
    return [str(p) for p in extra + files]


@st.cache_data(show_spinner=False)
def icons():
    """{name: [data URI, width / height]} for the player, drawn as masks so they take the text colour."""
    out = {}
    for name, path in [(n, ICONS / "hud" / f"{f}.svg") for n, f in HUD_ICONS.items()] + \
                      [(f, ICONS / "weapons" / f"{f}.svg") for f in {*WEAPON_ICONS.values(), *GRENADE_ICONS}]:
        if not path.exists():
            continue
        svg = path.read_bytes()
        if b'xmlns="http://www.w3.org/2000/svg"' not in svg:  # without it a browser won't draw the file as an image
            svg = svg.replace(b"<svg", b'<svg xmlns="http://www.w3.org/2000/svg"', 1)
        box = re.search(rb'viewBox="([^"]+)"', svg)
        w, h = [float(v) for v in box.group(1).replace(b",", b" ").split()[2:]] if box else (1.0, 1.0)
        out[name] = ["data:image/svg+xml;base64," + base64.b64encode(svg).decode(), round(w / h, 3)]
    return out


# ---------------------------------------------------------------------- round data for the player

@st.cache_data(show_spinner="Preparing round…", max_entries=24)
def round_payload(path: str, i: int, style: str):
    """Everything the browser player needs for one round view, as plain JSON-ready values (ints where possible)."""
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
                          int(row[U("in_flight")]), int(row[U("vel_x")]), int(row[U("vel_y")]), round(float(row[U("age")]), 1),
                          int(row[U("throw_id")])])
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
    # Main weapon: the primary, else the pistol, else whatever is in hand (0 = <none> in weapon_vocab).
    cat = data["player_cat"][s]
    main_weapon = np.where(cat[..., 2] > 0, cat[..., 2], np.where(cat[..., 3] > 0, cat[..., 3], cat[..., 0]))
    uri, extent = radar_image(path, style)
    winner = int(data["rounds_winner"][i])
    return {
        "steamids": [str(v) for v in data["rounds_steamids"][i]],
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
        "weapon": main_weapon.astype(int).tolist(), "place": data["player_cat"][s][:, :, 1].astype(int).tolist(),
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
    }


PER_PLAYER = ("paths", "probs", "endPlace", "truth")  # payload fields that only the predicted team's 5 slots have


def both_payload(t_view, ct_view):
    """Both sides at once: the T view, with the CT view's predictions filled into its 5 enemy slots
    (players matched by steam id, since each view lists its own team first). None if they don't line up."""
    if t_view["steps"] != ct_view["steps"]:
        return None
    slot_in_t = {sid: k for k, sid in enumerate(t_view["steamids"])}
    src = [None] * 5  # T view enemy slot 5 + e  <-  CT view own slot src[e]
    for j, sid in enumerate(ct_view["steamids"][:5]):
        k = slot_in_t.get(sid)
        if k is not None and k >= 5:
            src[k - 5] = j
    if None in src:
        return None

    def extend(a, b):  # [steps][5] + the CT view's matching 5 -> [steps][10]
        return [row + [b[t][j] for j in src] for t, row in enumerate(a)]

    out = dict(t_view, both=True, meta=dict(t_view["meta"]))
    for key in PER_PLAYER:
        if t_view[key] is not None and ct_view[key] is not None:
            out[key] = extend(t_view[key], ct_view[key])
    out["err"] = {sec: {kind: extend(t_view["err"][sec][kind], ct_view["err"][sec][kind]) for kind in ("min", "top")}
                  for sec in t_view["err"]}
    both = lambda a, b: round((a + b) / 2) if a >= 0 and b >= 0 else max(a, b)  # noqa: E731  (-1 = no value)
    out["teamErr"] = {kind: [both(a, b) for a, b in zip(t_view["teamErr"][kind], ct_view["teamErr"][kind])]
                      for kind in ("min", "top")}
    return out


# ---------------------------------------------------------------------- app

def match_index(rounds):
    """{match: {round number: {side: index into rounds}}}, matches and rounds in order."""
    out = {}
    for r in sorted(rounds, key=lambda r: (r["match"], r["round"])):
        out.setdefault(r["match"], {}).setdefault(r["round"], {}).setdefault(r["side"], r["index"])
    return out


def first_round(by_match):
    st.session_state.round = next(iter(by_match[st.session_state.match]))


def step_round(by_match, delta):
    """Prev / next arrows: the neighbouring round of the same match (they stop at its first and last round)."""
    state = st.session_state
    round_nums = list(by_match[state.match])
    k = round_nums.index(state.round) + delta
    if 0 <= k < len(round_nums):
        state.round = round_nums[k]


def pick_round():
    """Round strip: jump to the clicked round."""
    picked = st.session_state["round_strip"].picked
    if picked is not None:
        st.session_state.round = picked


# CS2-style round history: one square per round in the winning side's colour, gaps at halftime and at
# each overtime half, as many as fit and the rest scrolled sideways; click a square to load that round.
# The arrows either side step to the previous / next round.
_ROUND_STRIP = st.components.v2.component(
    "round_strip",
    html="""
<div class="bar">
  <button type="button" class="nav prev" title="Previous round"><svg viewBox="0 0 16 16"><path d="M10.5 3 5.5 8l5 5"/></svg></button>
  <div class="strip"></div>
  <button type="button" class="nav next" title="Next round"><svg viewBox="0 0 16 16"><path d="M5.5 3l5 5-5 5"/></svg></button>
</div>
""",
    css="""
.bar { display: flex; align-items: flex-start; gap: 6px; }
.nav { flex: 0 0 28px; height: 28px; margin-top: 3px; padding: 0; border: 1px solid var(--st-border-color);
  border-radius: 2px; background: transparent; color: var(--st-text-color); cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center; }
.nav:hover:not(:disabled) { background: var(--st-secondary-background-color); }
.nav:disabled { opacity: .3; cursor: default; }
.nav svg { width: 14px; height: 14px; fill: none; stroke: currentColor; stroke-width: 2; }
.strip { flex: 0 1 auto; min-width: 0; display: flex; gap: 3px; overflow-x: auto; padding: 3px 3px 6px;
  scrollbar-width: thin; scrollbar-color: #2A2E35 transparent; }
.strip button { flex: 0 0 28px; width: 28px; height: 28px; padding: 0; border: 0; border-radius: 2px;
  font: inherit; font-size: 11px; font-weight: 700; color: #0F1114; cursor: pointer; opacity: .55; }
.strip button:hover { opacity: .85; }
.strip button.on { opacity: 1; outline: 2px solid var(--st-text-color); outline-offset: 2px; }
.strip .T { background: #DEB453; }
.strip .CT { background: #6FA8DC; }
.strip .none { background: #272B31; color: #7C838D; }
.strip .half { flex: 0 0 8px; }
""",
    js="""
export default function (component) {
  const { data, parentElement, setTriggerValue } = component
  const strip = parentElement.querySelector(".strip")
  if (!strip || !data) return
  const prev = parentElement.querySelector(".prev"), next = parentElement.querySelector(".next")
  prev.disabled = !data.hasPrev; next.disabled = !data.hasNext
  prev.onclick = () => setTriggerValue("step", -1)
  next.onclick = () => setTriggerValue("step", 1)
  strip.replaceChildren()
  let current = null
  for (const r of data.rounds) {
    if (r.halfBefore) strip.appendChild(Object.assign(document.createElement("span"), { className: "half" }))
    const b = document.createElement("button")
    if (r.n === data.current) current = b
    b.type = "button"
    b.className = (r.winner || "none") + (r.n === data.current ? " on" : "")
    b.textContent = r.n
    b.title = `Round ${r.n}: ${r.winner ? r.winner + " won" : "result unknown"}`
    b.onclick = () => setTriggerValue("picked", r.n)
    strip.appendChild(b)
  }
  // keep the current round in view (scrolls the strip only, never the page)
  if (current) {
    const left = current.offsetLeft - strip.offsetLeft, right = left + current.offsetWidth
    if (left < strip.scrollLeft) strip.scrollLeft = left - 3
    else if (right > strip.scrollLeft + strip.clientWidth) strip.scrollLeft = right - strip.clientWidth + 3
  }
}
""",
)


def round_strip(by_match, rounds):
    """The current match's rounds as a strip, with prev / next arrows."""
    state = st.session_state
    round_nums = list(by_match[state.match])
    k = round_nums.index(state.round)
    # MR12: sides swap after round 12; overtime starts at round 25 and swaps every 3 rounds.
    half = lambda n: n == 13 or (n >= 25 and (n - 25) % 3 == 0)  # noqa: E731
    strip = [{"n": n, "winner": rounds[next(iter(views.values()))]["winner"], "halfBefore": half(n)}
             for n, views in by_match[state.match].items()]
    return _ROUND_STRIP(data={"current": state.round, "rounds": strip,
                              "hasPrev": k > 0, "hasNext": k < len(round_nums) - 1},
                        key="round_strip", on_picked_change=pick_round,
                        on_step_change=lambda: step_round(by_match, st.session_state["round_strip"].step or 0))


def main():
    files = prediction_files()
    with st.sidebar:
        st.markdown("### Mirage replay")
        if files:
            path = st.selectbox("Model", files, format_func=model_name)
        else:
            path = st.text_input("Predictions file", placeholder="checkpoints/transformer_full.predictions_test.npz")
        if not path or not Path(path).exists():
            st.info("Train with transformer.py, then pick its .predictions_<split>.npz file here "
                    "(or pass it: `streamlit run app.py -- <file>`).")
            st.stop()

        data, fde_min, fde_top, valid = open_file(path)
        rounds = round_table(path)

        # Display settings: widgets keep their values across reruns, so every round opens with them.
        st.markdown("#### Settings")
        paths = st.segmented_control("Predicted paths", list(PATHS), default="Top", key="paths") or "Top"
        enemies = st.segmented_control("Enemies", list(ENEMIES), default="True positions", key="enemies") or "True positions"
        # Set the starting value through Session State only (not also as default=), which Streamlit warns about.
        if "layers" not in st.session_state:
            st.session_state.layers = list(LAYERS)
        layers = st.pills("Show", list(LAYERS), selection_mode="multi", key="layers")
        styles = map_styles()
        style = (st.segmented_control("Map", styles, default=styles[0], key="style") or styles[0]) if len(styles) > 1 else styles[0]
        height = st.slider("Replay window height", 500, 1300, 730, 10, key="height", format="%d px")
        options = {"paths": PATHS[paths], "team": ENEMIES[enemies], **{key: name in layers for name, key in LAYERS.items()}}

        st.divider()
        h10 = data["horizons"][10] - 1
        ok = valid[:, :, h10]
        st.caption(f"{len(rounds)} round views, {int(ok.sum()):,} predictions  \n"
                   f"10 s error, best of 6: **{fde_min[:, :, h10][ok].mean():.0f}u**  \n"
                   f"10 s error, top path: **{fde_top[:, :, h10][ok].mean():.0f}u**")

    # Round picker above the player: match, round slider, prev / next, and whose side to replay.
    by_match = match_index(rounds)
    matches = list(by_match)
    state = st.session_state
    # Keep the picked match / round / side across reruns, falling back when the file doesn't have them.
    if state.get("match") not in by_match:
        state.match = matches[0]
    if state.get("round") not in by_match[state.match]:
        state.round = next(iter(by_match[state.match]))
    if state.get("side") is None:  # also when the side control is clicked off
        state.side = state.get("side_last", "T")

    # One bar: match, round strip with prev / next arrows, side. Labels are hidden so every control lines up.
    c_match, c_strip, c_side = st.columns([2, 6.1, 2], vertical_alignment="center")
    c_match.selectbox("Match", matches, key="match", on_change=first_round, args=(by_match,), label_visibility="collapsed")
    with c_strip:
        round_strip(by_match, rounds)
    views = by_match[state.match][state.round]
    side = c_side.segmented_control("Side", (*SIDES, "Both"), key="side", label_visibility="collapsed", width="stretch")
    payload = None
    if side == "Both" and all(s in views for s in SIDES):
        payload = both_payload(round_payload(path, views["T"], style), round_payload(path, views["CT"], style))
    if payload is None and side not in views:  # one side's view only (or Both couldn't be matched up)
        side = "T" if "T" in views else next(iter(views))
    i = views[side if side in views else "T"]
    state.side_last = side
    payload = payload or round_payload(path, i, style)
    r = rounds[i]
    err10 = [rounds[k]["err10"] for k in views.values()] if payload.get("both") else [r["err10"]]
    err10 = [e for e in err10 if not np.isnan(e)]

    # The title and round info are drawn in the player's own header, next to the clock and playback controls.
    # The key lets the player pick up where it was when only the settings changed (that reloads it).
    view = {"options": options, "key": f"{path}|{i}|{side}|{style}", "icons": icons(),
            "err10": round(sum(err10) / len(err10)) if err10 else None,
            "weaponIcons": {name: WEAPON_ICONS[name] for name in data["weapon_vocab"].tolist() if name in WEAPON_ICONS}}
    html = (PLAYER.read_text(encoding="utf-8").replace("/*__DATA__*/", json.dumps(payload, separators=(",", ":")), 1)
            .replace("/*__VIEW__*/{}", json.dumps(view), 1))
    st.iframe(html, height=height)


main()
