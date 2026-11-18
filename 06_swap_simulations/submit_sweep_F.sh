#!/bin/bash
# submit_sweep_F.sh — media-bias swap simulations on preferential attachment networks as a SLURM array job.
#
# 6 tasks = 3 m × 2 f_s, N = 1000; each writes one .npz (6×6 homophily grid × ~96 swap fractions).
# Output: 06_swap_simulations/output/sweep_F_m{m}_fs{f_s}_N1000.npz (6 files)
#
# Usage (from the repository root on the cluster):
#   mkdir -p logs
#   sbatch 06_swap_simulations/submit_sweep_F.sh
#   EXCLUDE_SELF=1 sbatch 06_swap_simulations/submit_sweep_F.sh   # neighbors-only twin -> 06_swap_simulations/output_exclude_self/
#
# Runtime: up to 8 h per task, 36 cores.
#SBATCH --job-name=sweep_F
#SBATCH --array=0-5
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=36
#SBATCH --mem=32G
#SBATCH --output=logs/sweep_F_%A_%a.out
#SBATCH --error=logs/sweep_F_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Environment: set for your cluster (these lines are for Stanford Sherlock) ──
module load python/3.12.1
source "${VENV:-$HOME/venvs/misperception}/bin/activate"

cd "${SLURM_SUBMIT_DIR:-.}"      # repository root
FLAGS=""
if [[ "${EXCLUDE_SELF:-0}" == 1 ]]; then FLAGS="--exclude-self"; fi


# ── Index arithmetic: 6 tasks = 3 m × 2 f_s ──────────────────────────────────
M_VALUES=(2 5 10)
FS_VALUES=(0.6667 0.8)

IDX=${SLURM_ARRAY_TASK_ID:-0}
M=${M_VALUES[$((IDX / 2))]}
FS=${FS_VALUES[$((IDX % 2))]}

echo "Task $IDX: m=$M  f_s=$FS  N=1000  $FLAGS ($(date))"

python3 06_swap_simulations/run_sweep_F.py \
    --m      "$M" \
    --fs     "$FS" \
    --N      1000 \
    --numsim 200 \
    --workers "${SLURM_CPUS_PER_TASK:-36}" \
    $FLAGS

echo "Task $IDX done ($(date))"
