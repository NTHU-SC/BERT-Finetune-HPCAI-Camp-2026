#!/bin/bash
#SBATCH -p amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:r9700:1
#SBATCH --partition=cscamp
#SBATCH --time=00:00:05
#SBATCH -J bert-inf
#SBATCH -o bert-inf.out
#SBATCH -e bert-inf.err

# Slurm batch shells do not source the login profile that defines `module`.
source /etc/profile.d/modules.sh
module purge
module load rocm/7.2.0

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # The camp environment is created once with the commands in README.md.
    source "$HOME/venvs/camp-ai/bin/activate"
fi

model_flag=""

if [[ $1 != "" ]]; then
    model_flag="--model $1"
fi

python Inference.py $model_flag
