#!/bin/bash
# submit_sweep_E_G3x3.sh — exact degree-matched baselines on the broadcast-model grid as a SLURM array job.
#
# The broadcast model (11_broadcast_media) needs peer baselines at h_ss, h_oo ∈ {0.25, 0.5, 0.75}, N = 1000.
# The main degree-matched grid (0, 0.1, …, 1) lacks 0.25/0.75; without this run, run_sweep_G.py
# falls back to bilinear interpolation. 9 tasks = 3 m × 3 f_s, 200 sims per grid point.
# Output: 10_degree_matched/output_G3x3/sweep_E_m{m}_fs{f_s}_N1000.npz (9 files)
#
# Usage (from the repository root on the cluster):
#   mkdir -p logs
#   sbatch 10_degree_matched/submit_sweep_E_G3x3.sh
#
# Runtime: well under 8 h per task, 9 cores.
#SBATCH --job-name=sweep_E_G3x3
#SBATCH --array=0-8
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=9
#SBATCH --mem=16G
#SBATCH --output=logs/sweep_E_G3x3_%A_%a.out
#SBATCH --error=logs/sweep_E_G3x3_%A_%a.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ── Environment: set for your cluster (these lines are for Stanford Sherlock) ──
module load python/3.12.1
source "${VENV:-$HOME/venvs/misperception}/bin/activate"

cd "${SLURM_SUBMIT_DIR:-.}"      # repository root


M_VALUES=(2 5 10)
FS_VALUES=(0.5 0.6667 0.8)

IDX=${SLURM_ARRAY_TASK_ID:-0}
M=${M_VALUES[$((IDX / 3))]}
FS=${FS_VALUES[$((IDX % 3))]}

echo "Task $IDX: m=$M  f_s=$FS  N=1000  grid=0.25,0.5,0.75  ($(date))"

python3 10_degree_matched/run_sweep_E.py \
    --m      "$M" \
    --fs     "$FS" \
    --N      1000 \
    --numsim 200 \
    --h-vals 0.25,0.5,0.75 \
    --workers "${SLURM_CPUS_PER_TASK:-9}" \
    --outdir 10_degree_matched/output_G3x3

echo "Task $IDX done ($(date))"
