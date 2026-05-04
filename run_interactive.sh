#!/bin/bash
# Usage: ./run_interactive.sh dspy_inference.py --version base --prompt "Explain segfaults"
# Or:    ./run_interactive.sh inference.py --version base --prompt "Hello"

if [ $# -eq 0 ]; then
    echo "Usage: ./run_interactive.sh <script.py> [args...]"
    exit 1
fi

SCRIPT=$1
shift

# This runs exactly one srun step, which allocates the GPU, loads the correct
# module environment, activates the venv, and runs python.
srun -p grete:interactive -G 1g.20gb -t 00:15:00 -A intern_agc_emmy --export=NONE /bin/bash -l -c "
  set +u
  source /etc/profile || true
  set -u
  module purge
  module load gcc/13.2.0 python/3.11.9
  source .venv_gpu/bin/activate
  python3 $SCRIPT $@
"
