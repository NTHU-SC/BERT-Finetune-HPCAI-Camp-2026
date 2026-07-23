#!/bin/bash
# TWCC account. Override it at submission time when using another project:
#   sbatch -A <project-id> run_train.sh ...
#SBATCH --account=ACD110018
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gp1d
#SBATCH --time=00:15:00
#SBATCH -J bert-train
#SBATCH -o bert-train.out
#SBATCH -e bert-train.err

set -eo pipefail

# Slurm batch shells do not source the login profile that defines `module`.
source /etc/profile.d/modules.sh
module purge
module load miniconda3/conda24.5.0_py3.9

: "${WORK:?WORK must be set on TWCC}"
CAMP_ENV="${CAMP_ENV:-$WORK/venvs/camp-ai}"
export HF_HOME="${HF_HOME:-$WORK/.cache/huggingface}"

if [[ ! -x "$CAMP_ENV/bin/python" ]]; then
    echo "Missing camp environment: $CAMP_ENV" >&2
    echo "Follow README.md section 1 before submitting a job." >&2
    exit 1
fi

conda activate "$CAMP_ENV"

model="${1:-google-bert/bert-base-uncased}"
output="${2:-./output_model}"
shift $(( $# >= 2 ? 2 : $# ))

python Train.py --model "$model" --output "$output" "$@"
