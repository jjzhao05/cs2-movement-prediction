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
| `transformer.py` | The transformer: model, training and evaluation |
| `main.py` | Runs extract → dataset → sql → baselines |

---

## Quick start

```bash
pip install awpy demoparser2 polars numpy torch scikit-learn pandas nbconvert
python main.py --skip baselines      # parse demos in data/ and build mirage.db
python transformer.py                # train the full-round transformer
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

Progress is logged to `logs/`; checkpoints go to `checkpoints/`.

---

## Data

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
- Per-player GRU

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
- **Not in git:** `data/`, `processed/`, `mirage.db`, `logs/`, `checkpoints/`.
