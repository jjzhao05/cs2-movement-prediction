#!/bin/bash
# Submit training jobs to Delta, one GPU each.
#
#     bash delta/submit.sh smoke          # 1 epoch on 20 rounds (checks the setup, ~5 min)
#     bash delta/submit.sh transformer    # full-round transformer
#     bash delta/submit.sh ablations      # 3 s and 10 s history, full-information
#     bash delta/submit.sh gru            # per-player GRU baseline
#     bash delta/submit.sh all            # transformer + ablations + gru
set -euo pipefail
cd "$(dirname "$0")/.."
source delta/config.sh
[ -n "$ACCOUNT" ] || { echo "export ACCOUNT=<project>-delta-gpu in ~/.bashrc (\`accounts\` lists yours)"; exit 1; }
[ -f processed/manifest.json ] || { echo "processed/ is missing: copy it here first (see README, Delta)"; exit 1; }
[ -x "$VENV/bin/python" ] || { echo "no venv at $VENV: run bash delta/setup.sh first"; exit 1; }
mkdir -p logs checkpoints

run() {  # run <job name> <time> <command...>
    local name=$1 time=$2
    shift 2
    sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$time" --job-name="$name" delta/train.slurm "$@"
}

transformer() { run tf_full "$TIME" python transformer.py --workers 8; }
ablations() {
    run tf_3 "$TIME" python transformer.py --history 3 --workers 8
    run tf_10 "$TIME" python transformer.py --history 10 --workers 8
    run tf_fullinfo "$TIME" python transformer.py --full-info --workers 8
}
gru() { run gru "$TIME" python gru.py --device cuda; }

case "${1:-}" in
    smoke) run smoke 00:30:00 python transformer.py --max-rounds 20 --epochs 1 --workers 8 \
               --out checkpoints/smoke/transformer.pt --save-predictions ;;
    transformer) transformer ;;
    ablations) ablations ;;
    gru) gru ;;
    all) transformer; ablations; gru ;;
    *) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
echo "queued; watch with: squeue -u \$USER   and   tail -f logs/slurm-<name>-<jobid>.out"
