# CS2 Movement Prediction — `de_mirage`

Forecasts where each pro player will move over the next **10 seconds**, using only what their
team can see. A transformer predicts **6 possible paths per player**, each with a probability,
and is compared against simple baselines.

## Contents

- [Pipeline](#pipeline)
- [Quick start](#quick-start)
- [Data](#data)
- [Models](#models)
- [Metrics](#metrics)
- [Notes](#notes)

---

## Pipeline

```
.rar archives ──extract_rars.py──> data/*.dem ──build_dataset.py──> processed/*.npz ──build_sql.py──> mirage.db
                                                                          │
                                                        dataset.py ──> baselines.ipynb, transformer.py
```

| File | What it does |
|---|---|
| `extract_rars.py` | Pulls Mirage demos out of `.rar` archives into `data/` |
| `build_dataset.py` | Parses demos (awpy + demoparser2) into per-round arrays sampled every 0.5 s |
| `build_sql.py` | Loads `processed/` into SQLite (incremental; `--rebuild` after changing features) |
| `features.py` | The feature layout shared by every step |
| `dataset.py` | PyTorch datasets with team-perspective masking: `MirageWindows` (one window) and `MirageRounds` (whole rounds) |
| `baselines.ipynb` | Stay still, constant velocity, kNN, per-player GRU |
| `gru.py` | The per-player GRU on its own: training (CPU or GPU, resumable), test predictions |
| `transformer.py` | The transformer: model, training and evaluation |
| `visualize.py` | Draws saved predictions on the map (PNG or GIF) and breaks down the metrics |
| `app.py` | Mirage Replay: interactive Streamlit replay of saved predictions (player in `assets/replay.html`) |
| `build_map.py` | Draws the Mirage map in `assets/` from the game's radar |
| `main.py` | Runs extract → dataset → sql → baselines |
| `run_notebook.py` | Executes a notebook in place (used for `baselines.ipynb`) |
| `delta/` | Setup and Slurm jobs for training on NCSA Delta ([Training on Delta](#training-on-delta)) |

---

## Quick start

```bash
pip install awpy demoparser2 polars numpy torch scikit-learn scikit-image pandas nbclient ipykernel matplotlib streamlit
python main.py --skip baselines      # parse demos in data/ and build mirage.db
python transformer.py                # train the full-round transformer
python gru.py --device cuda          # train the per-player GRU baseline (GPU)
python main.py --only baselines      # evaluate all baselines, reusing the trained GRU
```

### Useful flags

| Script | Flag | Effect |
|---|---|---|
| `main.py` | `--overwrite --rebuild` | Reparse everything (needed after changing `features.py`) |
| `main.py` | `--workers N` | Number of parallel parse workers |
| `transformer.py` | `--history 3\|10\|full` | History window per step (seconds, or the full round) |
| `transformer.py` | `--full-info` | Disable enemy masking |
| `transformer.py` | `--device cuda\|cpu\|dml` | Compute device |
| `transformer.py` | `--eval-only` | Skip training, evaluate the saved checkpoint |
| `transformer.py` | `--resume` | Continue from the last finished epoch (e.g. after a job time limit) |
| `transformer.py` | `--save-predictions val test` | Splits whose every prediction is saved (default `val test`) |
| `gru.py` | `--device cuda\|cpu\|dml` | Compute device (on a GPU the whole training set is kept in GPU memory) |
| `gru.py` | `--resume` | Continue from the last finished epoch |
| `gru.py` | `--retrain` | Train even if a checkpoint for this training data exists |
| `gru.py` | `--eval-only` | Skip training, write test predictions from the saved checkpoint |

Progress is logged to `logs/`. A training run writes to `checkpoints/`:

| File | Contents |
|---|---|
| `<name>.pt` | Best model (by val minADE@6 at 10 s) |
| `<name>.last.pt` | Latest epoch with optimizer state, for `--resume` |
| `<name>.history.json` | Per epoch: loss, learning rate, full validation metrics |
| `<name>.metrics.json` | Final val/test metrics, including callout accuracy and off-map rate |
| `<name>.predictions_<split>.npz` | Every prediction (6 paths, probabilities, uncertainty) with the true futures, all 10 players' positions, round clock and callouts |

`visualize.py` works from these files alone (no model or GPU), so a long run never has to be
repeated just to look at it:

```bash
python visualize.py rounds  checkpoints/transformer_full.predictions_test.npz          # list rounds
python visualize.py round   checkpoints/transformer_full.predictions_test.npz 12 --gif # animate round 12
python visualize.py metrics checkpoints/transformer_full.predictions_test.npz          # plots + CSVs
```

`metrics` writes error vs. seconds ahead, breakdowns by side / round time / bomb / callout, a map of
where predictions go wrong, path-probability calibration and training curves to `plots/`.

### Mirage Replay

```bash
streamlit run app.py
```

An interactive 2D replay of any model's test predictions. Pick the **model** in the sidebar: every
`checkpoints/*.predictions_<split>.npz` is listed, i.e. the transformer (written by `transformer.py`),
the per-player GRU (written by `gru.py`) and stay still, constant velocity, kNN and the GRU again
(written by the last cells of `baselines.ipynb`). All go through `transformer.save_player_predictions`
or `Predictions`, so they share one format:

- **Map:** zoom, pan and hover. Each teammate's 6 predicted 10 s paths are arrows (bolder =
  likelier), with what really happened dashed. Also shown: enemies (true positions, or
  only what the team knew), smokes, molotovs, flashes, the bomb, view directions, deaths
  and callout labels.
- **Playback:** runs in the browser (`assets/replay.html`) at 60 fps, gliding players and
  paths between the 0.5 s samples like a video. Play / pause at 0.5x–4x, step 0.5 s at a time,
  or drag the timeline, which shows the team's 10 s error over the round with kills marked.
  Keys: Space play / pause, ← → step, Home / End.
- **Deaths:** both teams' deaths are marked where they fell (ringed ✕ with the name), listed in a
  kill feed for a few seconds, and marked on the timeline (enemies on top, your team below).
- **Side panel:** both teams' health, armor, weapon, grenades, money and callout. Focus a
  player to see their 6 paths' probabilities, where each one ends, and the error at 3 / 5 / 10 s.
- **Rounds:** filter by match, side and result; sort by hardest or easiest to predict.

The map is drawn by `build_map.py` from the game's own Mirage radar (awpy's data release,
`images.zip` + `map_data.json` from github.com/pnxenopoulos/awpy-data, unpacked into
`~/.awpy/<version>/`): the outline is traced as straight vector edges and rendered at 2.4x,
recoloured to the app's dark theme, with callout labels placed from `processed/`. The result
is saved in `assets/`. The original radar is offered as a second map style.

### Training on Delta

Parsing stays local (it needs `data/` and awpy); only `processed/` (~270 MB) goes to the cluster.
Keep the repo on project storage rather than home, e.g. `/projects/<account>/$USER/`.

```bash
# on a Delta login node (login.delta.ncsa.illinois.edu), once
mkdir -p /projects/<account>/$USER && cd /projects/<account>/$USER
git clone https://github.com/jjzhao05/cs2-movement-prediction.git && cd cs2-movement-prediction
bash delta/setup.sh                 # venv in ~/venvs/cs2 with CUDA PyTorch
echo 'export ACCOUNT=<account>-delta-gpu' >> ~/.bashrc && source ~/.bashrc   # `accounts` lists yours
```

Copy `processed/` into the clone with [Globus](https://www.globus.org) (Delta asks for it rather
than `scp`): run Globus Connect Personal on this machine, then in the Globus web app transfer
`processed/` from your personal endpoint to the *NCSA Delta* collection at
`/projects/<account>/<user>/cs2-movement-prediction/processed/`.

```bash
# on Delta
bash delta/submit.sh smoke          # 1 epoch on 20 rounds: checks venv, GPU and data
bash delta/submit.sh all            # transformer, its 3 ablations and the GRU, one GPU job each
squeue -u $USER                     # queue; output in logs/slurm-<name>-<jobid>.out
```

Each job runs `delta/train.slurm`: one GPU on `gpuA40x4` (change `PARTITION` / `TIME` in
`delta/config.sh`), 16 CPUs, 64 GB. Ten minutes before the time limit it stops, requeues itself
and continues from the last finished epoch with `--resume`, so long runs need no babysitting;
every restart appends to the same log. Bring `checkpoints/` and `logs/` back with a Globus
transfer the other way and use `visualize.py` / `app.py` locally. After reparsing, copy `processed/` again and delete
`checkpoints/*.last.pt` on Delta before submitting, so no job resumes a run trained on the old data.

---

## Data

Matches are from across October 2026: 269 series across 38 tournaments, each series
contributing exactly one Mirage map (**269 maps total**).

| Tournament | Maps |
|---|---|
| Digital Crusade Draculan Season 7 | 22 |
| CCT 2026 South America Series 7 | 18 |
| NODWIN Clutch Series 13 | 14 |
| PGL Masters Bucharest 2026 | 14 |
| Winline MPKBK CIS LAN Season 9 | 14 |
| CCT 2026 Europe Series 11 Closed Qualifier | 13 |
| CCT 2026 Europe Series 12 | 13 |
| CCT 2026 Europe Series 12 Closed Qualifier | 13 |
| CCT 2026 Europe Series 11 | 12 |
| ESL Challenger League Season 52 – North America Cup 3 | 12 |
| Stake Pulse Beat III | 12 |
| CCT 2026 Challengers Europe Series 8 | 10 |
| FRAG University 2026 | 8 |
| Stake Ranked Episode 6 Closed Qualifier | 8 |
| Thunderpick World Championship 2026 | 8 |
| 1win Private Club Season 2 | 7 |
| ESL Challenger League Season 52 – Asia-Pacific Cup 3 | 7 |
| ESL Challenger League Season 52 – Europe Cup 8 | 6 |
| NODWIN Clutch Series 13 Closed Qualifier | 6 |
| Circuit X Curitiba Season 4 | 5 |
| ESL Challenger League Season 52 – South America Cup 3 | 5 |
| RLG Tesla Cup 2026 | 5 |
| 1win Private Club Season 2 Closed Qualifier | 4 |
| Elisa Breakout Series 2 | 4 |
| ESL Pro League Season 24 | 4 |
| Gluck Yungstar 2026 Series 3 | 4 |
| CCT 2026 Europe Series 10 | 3 |
| ESL Challenger League Season 52 – Europe Cup 9 | 3 |
| Stake Ranked Episode 5 | 3 |
| DFRAG Invitational 2026 | 2 |
| ESL Challenger League Season 52 – Oceania Cup 3 | 2 |
| NSTLGA Emerald Cup 2026 | 2 |
| ANZC Locals Brisbane 2026 Season 2 | 1 |
| CIS Esports League CyberClash Season 2 | 1 |
| DFRAG Logitech G Pro Series 2026 | 1 |
| MaceMedia Void Autumn Cup 2026 | 1 |
| Mesa Pro Series Halloween Cup 2026 | 1 |
| Roman Imperium Cup IX | 1 |
| **Total** | **269** |

Sampled every **0.5 s** of live play, per round:

| Group | Features |
|---|---|
| **Players** (10) | Position, velocity, view angles, health/armor, weapons, callout, money, grenades carried, ammo, airborne/ducking/ladder, shots/damage/kills/footsteps in the last 0.5 s, who spots whom |
| **Round** | Clock, bomb state and position, score (the game's round counter), alive counts, equipment |
| **Grenades** (up to 16) | In flight (position, velocity), flash/HE detonations, active smokes and fires, linked by a per-round `throw_id` |

**Splits** are by match (hash of the match name), so train/val/test never share a match.

**Team perspective** (`mask_enemies=True`):

- Enemies are only visible while spotted; otherwise the model sees their last known position and
  time since seen.
- Enemy money and inventory are always hidden.
- Enemy grenades appear once they pop.

---

## Models

### Baselines — `baselines.ipynb`

- Stay still
- Constant velocity
- kNN over similar past situations (position, movement, time, side, score, economy)
- Per-player GRU (`gru.py`): one player's last 10 s → 6 paths with probabilities, trained
  winner-takes-all. The notebook reuses `checkpoints/per_player_gru.pt` when it was trained on the
  same samples (checked by hash), so the GRU can be trained on a GPU with `gru.py` and the notebook
  only evaluates it.

All four save their test predictions to `checkpoints/<model>.predictions_test.npz` for the replay app.

### Transformer — `transformer.py`

- **Tokens:** 27 per step — 10 players, round state, 16 grenades.
- **Blocks:** causal time attention per player, then attention across all tokens of a step.
- **Decoder:** outputs 6 paths (20 steps each, with Laplace uncertainty) plus probabilities for
  every living teammate at every step.
- **Training:** winner-takes-all.
- **History variants:** the 3 s / 10 s variants give each step its own strict history window.

---

## Metrics

Reported at **3, 5 and 10 s**, for all players and for moving players (≥ 100 u/s).
1 game unit ≈ 1.9 cm.

| Metric | Description |
|---|---|
| minADE@6 / minFDE@6 | Average / final displacement of the best of 6 paths |
| Top-1 ADE / FDE | Displacement of the most probable path |
| Miss rate | Fraction with final error > 200 units |
| Brier-minFDE | minFDE penalised by the best path's probability |
| Callout accuracy | Predicted callout matches the actual one |
| Off-map rate | Predictions that leave the playable area |
| Surprise ψ | −log p(actual path) — transformer only |

---

## Notes

- **Performance:** parsing takes ~7 s per demo with 4 workers (~2 GB RAM per worker). Training is
  much faster on a GPU; on CPU the full-round model takes ~4–5 min per epoch on the current 11 demos.
- **awpy quirks:** it drops ticks once a timeout is called mid-round, and its bomb pickup events
  can be missing. `build_dataset.py` works around both (see its docstrings).
- **Not in git:** `data/`, `processed/`, `mirage.db`, `logs/`, `checkpoints/`, `plots/`.
