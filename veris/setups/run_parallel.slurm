#!/bin/bash
#SBATCH --job-name=veris-parallel
#SBATCH --partition=aegir
#SBATCH --constraint=v3
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --output=veris-parallel-%j.log

# Submit from the project root: sbatch veris/setups/run_parallel.slurm
# CLI arguments override the small smoke-run defaults, e.g. --nx 256 --steps 10.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [[ -f .venv-latest/bin/activate ]]; then
    source .venv-latest/bin/activate
elif [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
else
    echo "ERROR submit from the Veris project root with an installed virtual environment" >&2
    exit 1
fi
export JAX_PLATFORMS=cpu
export JAX_NUM_CPU_DEVICES=1
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
srun --cpu-bind=cores python -m veris.setups.run_parallel \
    --nx 64 --mesh 1 "${SLURM_NTASKS:-2}" --steps 2 --evp-steps 4 \
    "$@" --backend cpu
