#!/bin/bash
# submit_sweep_E.sh — degree-matched preferential attachment and stochastic block model simulations as a SLURM array job.
#
# 27 tasks = 3 m × 3 f_s × 3 N; each writes one .npz with 11×11 grids for both models.
# Output: 10_degree_matched/output/sweep_E_m{m}_fs{f_s}_N{N}.npz (27 files)
#
# Usage (from the repository root on the cluster):
#   mkdir -p logs
#   sbatch 10_degree_matched/submit_sweep_E.sh
#   EXCLUDE_SELF=1 sbatch 10_degree_matched/submit_sweep_E.sh   # neighbors-only twin -> 10_degree_matched/output_exclude_self/
#
# Runtime: up to 8 h per task, 32 cores.
#SBATCH --job-name=sweep_E
#SBATCH --array=0-26
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --output=logs/sweep_E_%A_%a.out
#SBATCH --error=logs/sweep_E_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Environment: set for your cluster (these lines are for Stanford Sherlock) ──
module load python/3.12.1
source "${VENV:-$HOME/venvs/misperception}/bin/activate"

cd "${SLURM_SUBMIT_DIR:-.}"      # repository root
FLAGS=""
if [[ "${EXCLUDE_SELF:-0}" == 1 ]]; then FLAGS="--exclude-self"; fi


# ── Index arithmetic: 27 tasks = 3 m × 3 f_s × 3 N ───────────────────────────
M_VALUES=(2 5 10)
FS_VALUES=(0.5 0.6667 0.8)
N_VALUES=(100 500 1000)

IDX=${SLURM_ARRAY_TASK_ID:-0}
M=${M_VALUES[$((IDX / 9))]}
FS=${FS_VALUES[$(( (IDX / 3) % 3 ))]}
N=${N_VALUES[$((IDX % 3))]}

echo "Task $IDX: m=$M  f_s=$FS  N=$N  $FLAGS ($(date))"

python3 10_degree_matched/run_sweep_E.py \
    --m      "$M" \
    --fs     "$FS" \
    --N      "$N" \
    --numsim 200 \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    $FLAGS

echo "Task $IDX done ($(date))"
