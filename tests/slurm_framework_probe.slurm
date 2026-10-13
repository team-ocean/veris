#!/bin/bash
#SBATCH --job-name=veris-framework
#SBATCH --partition=aegir
#SBATCH --constraint=v3
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

# Submit from the project root with an absolute result directory as argument 1.
# Override --nodes=2 --ntasks-per-node=1 for actual cross-node acceptance.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source .venv-latest/bin/activate
task_result_root="$1"
mkdir -p "$task_result_root"
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu
export JAX_NUM_CPU_DEVICES=1
export JAX_CPU_COLLECTIVES_IMPLEMENTATION=gloo
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
for task_kind in island dynamics growth custom; do
    for task_cyclic in 0 1; do
        task_case_root="$task_result_root/$task_kind-$task_cyclic"
        mkdir -p "$task_case_root"
        srun --cpu-bind=cores python tests/distributed_framework_probe.py \
            --kind "$task_kind" --cyclic-y "$task_cyclic" \
            --output-root "$task_case_root"
    done
done
