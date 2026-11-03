# Settings for NCSA Delta, sourced by setup.sh, submit.sh and train.slurm.
# Any of these can be overridden from the environment; put `export ACCOUNT=bbxx-delta-gpu`
# in ~/.bashrc on Delta rather than editing this file, so `git pull` stays clean.

ACCOUNT="${ACCOUNT:-}"                    # GPU allocation, e.g. bbxx-delta-gpu (`accounts` on a login node lists yours)
PARTITION="${PARTITION:-gpuA40x4}"        # or gpuA100x4 (faster, charged more per GPU hour)
TIME="${TIME:-12:00:00}"                  # per job; jobs requeue themselves near the limit and resume
VENV="${VENV:-$HOME/venvs/cs2}"           # Python environment made by setup.sh
PYTHON_MODULE="${PYTHON_MODULE:-python}"  # module that provides python3 >= 3.10 (`module avail python`)
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"  # CUDA build of PyTorch
