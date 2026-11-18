#!/bin/bash
# submit_sweep_D.sh — finite-N preferential attachment simulations as a SLURM array job.
#
# 27 tasks = 3 m × 3 f_s × 3 N; each writes 19 .npz files (no rescaling + 9 per-agent + 9 group-mean variants).
# Output: 08_finite_n_pa/output/sweep_D_m{m}_fs{f_s}_N{N}_{variant}.npz (513 files)
#
# Usage (from the repository root on the cluster):
#   mkdir -p logs
#   sbatch 08_finite_n_pa/submit_sweep_D.sh
#   EXCLUDE_SELF=1 sbatch 08_finite_n_pa/submit_sweep_D.sh   # neighbors-only twin -> 08_finite_n_pa/output_exclude_self/
#
# Runtime: N = 1000 tasks with m = 5 or 10 take more than 4 h; 8 cores.
#SBATCH --job-name=sweep_D
#SBATCH --array=0-26
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/sweep_D_%A_%a.out
#SBATCH --error=logs/sweep_D_%A_%a.err
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

python3 08_finite_n_pa/run_sweep_D.py \
    --m      "$M" \
    --fs     "$FS" \
    --N      "$N" \
    --numsim 200 \
    --n-points 21 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    $FLAGS

echo "Task $IDX done ($(date))"
