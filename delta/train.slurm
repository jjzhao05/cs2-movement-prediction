#!/bin/bash
# Runs one training command on a GPU node; submit.sh passes the command and the account.
# Ten minutes before the time limit the job stops training, requeues itself and, when it
# starts again, continues from the last finished epoch (the command always gets --resume).
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64g
#SBATCH --output=logs/slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --signal=B:USR1@600
#SBATCH --requeue
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
source delta/config.sh
module load "$PYTHON_MODULE" 2>/dev/null || true
source "$VENV/bin/activate"

echo "$(date)  job $SLURM_JOB_ID (restart ${SLURM_RESTART_COUNT:-0}) on $(hostname): $* --resume"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$@" --resume &
pid=$!
trap 'echo "$(date)  near the time limit: requeueing"; kill -TERM $pid; wait $pid; scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait $pid
status=$?
echo "$(date)  exit $status"
exit $status
