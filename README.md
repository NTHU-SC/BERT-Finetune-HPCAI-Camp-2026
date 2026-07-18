# BERT Finetune — HPCAI Camp (AMD)

This version is for the CSCC AMD environment.  Submit GPU jobs only to the
`amd` Slurm partition; it provides AMD Radeon AI PRO R9700 GPUs through ROCm.

## 1. Log in and prepare the project

```bash
ssh CSCC
cd "$HOME"
git clone https://github.com/NTHU-SC/BERT-Finetune-HPCAI-Camp-2026.git
cd BERT-Finetune-HPCAI-Camp-2026
```

All following commands are run from this project directory under your own
`$HOME`; do not work in another participant's directory.

## 2. Create a personal Python environment

Do this once.  It creates an environment under your own home directory and
does not modify the system Python or anyone else's environment.

```bash
module load rocm/7.2.0
python3 -m venv "$HOME/venvs/camp-ai"
source "$HOME/venvs/camp-ai/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install transformers datasets scikit-learn evaluate accelerate \
  --upgrade huggingface_hub
```

Verify that the ROCm PyTorch build can access a GPU.  PyTorch keeps the
`torch.cuda` API name for ROCm, so `True` is the expected result.

```bash
module load rocm/7.2.0
source "$HOME/venvs/camp-ai/bin/activate"
python -c 'import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

If Hugging Face asks for authentication, create a read-only token at
<https://huggingface.co/settings/tokens> and run `huggingface-cli login`.

## 3. Train and infer with Slurm

The supplied scripts request one `r9700` GPU in the `amd` partition and use
16 CPU cores, matching the partition's default CPU-per-GPU allocation.

```bash
sbatch --partition=cscamp run_train.sh google-bert/bert-base-uncased ./output_model
sbatch --partition=cscamp run_inf.sh ./output_model/checkpoint-13
```

The outputs are written to `bert-train.out`, `bert-train.err`,
`bert-inf.out`, and `bert-inf.err`.  Useful Slurm commands are:

```bash
squeue -u "$USER"
sacct -j <job-id>
scancel <job-id>
```

For a quick interactive check (not a training run), request an AMD GPU:

```bash
srun -p amd --gres=gpu:r9700:1 --cpus-per-task=16 --pty bash
module load rocm/7.2.0
source "$HOME/venvs/camp-ai/bin/activate"
python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

## Goal and submission

Train through `Train.py` and improve accuracy by tuning allowed
hyperparameters and training strategy.  Do not change the model, dataset,
dataset contents, or `Inference.py`.  Submit `bert-inf.out`, `bert-inf.err`,
and a readable report describing the configurations and experiments that led
to your best result.
