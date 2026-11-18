#!/bin/bash
# submit_sweep_C.sh — finite-N stochastic block model simulations as a SLURM array job.
#
# 27 tasks = 3 H_out × 3 f_s × 3 N; each writes 19 .npz files (no rescaling + 9 per-agent + 9 group-mean variants).
# Output: 07_finite_n_sbm/output/sweep_C_hout{H_out}_fs{f_s}_N{N}_{variant}.npz (513 files)
#
# Usage (from the repository root on the cluster):
#   mkdir -p logs
#   sbatch 07_finite_n_sbm/submit_sweep_C.sh
#   EXCLUDE_SELF=1 sbatch 07_finite_n_sbm/submit_sweep_C.sh   # neighbors-only twin -> 07_finite_n_sbm/output_exclude_self/
#
# Runtime: up to ~2 h per task (N = 1000), 8 cores.
#SBATCH --job-name=sweep_C
#SBATCH --array=0-26
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/sweep_C_%A_%a.out
#SBATCH --error=logs/sweep_C_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Environment: set for your cluster (these lines are for Stanford Sherlock) ──
module load python/3.12.1
source "${VENV:-$HOME/venvs/misperception}/bin/activate"

cd "${SLURM_SUBMIT_DIR:-.}"      # repository root
FLAGS=""
if [[ "${EXCLUDE_SELF:-0}" == 1 ]]; then FLAGS="--exclude-self"; fi


# ── Index arithmetic: 27 tasks = 3 H_out × 3 f_s × 3 N ───────────────────────
H_OUT_VALUES=(0.1 0.3 0.5)
FS_VALUES=(0.5 0.6667 0.8)
N_VALUES=(100 500 1000)

IDX=${SLURM_ARRAY_TASK_ID:-0}
HOUT=${H_OUT_VALUES[$((IDX / 9))]}
FS=${FS_VALUES[$(( (IDX / 3) % 3 ))]}
N=${N_VALUES[$((IDX % 3))]}

echo "Task $IDX: H_out=$HOUT  f_s=$FS  N=$N  $FLAGS ($(date))"

python3 07_finite_n_sbm/run_sweep_C.py \
    --hout   "$HOUT" \
    --fs     "$FS" \
    --N      "$N" \
    --numsim 200 \
    --n-points 21 \
    --workers "${SLURM_CPUS_PER_TASK:-8}" \
    $FLAGS

echo "Task $IDX done ($(date))"
