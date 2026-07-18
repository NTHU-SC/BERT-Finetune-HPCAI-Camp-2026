# BERT Finetune — HPCAI Camp (AMD)

This version is for the CSCC camp AMD environment. Submit GPU jobs only through
the supplied scripts, which request one AMD GPU from the `cscamp` Slurm
partition.

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

The supplied scripts request one GPU and 16 CPU cores in the `cscamp`
partition. Each job has a one-minute time limit.

The default training configuration was measured on this environment: it uses a
fixed-seed random sample of 10,000 training examples, 500 separate validation
examples, five epochs, batch size 128, a learning rate of `5e-5`, and dynamic
padding to 64 tokens. It keeps the dataset contents unchanged and saves only
the final model as `checkpoint-final`.

```bash
sbatch run_train.sh google-bert/bert-base-uncased ./output_model
sbatch run_inf.sh ./output_model/checkpoint-final
```

On a warm cache, the measured training job completed in 51 seconds and the
inference job in 28 seconds. The measured test accuracy was 40.7051%. Queue
time is not included and depends on current cluster usage.

The outputs are written to `bert-train.out`, `bert-train.err`,
`bert-inf.out`, and `bert-inf.err`.  Useful Slurm commands are:

```bash
squeue -u "$USER"
sacct -j <job-id>
scancel <job-id>
```

For a quick interactive check (not a training run), request an AMD GPU:

```bash
srun -p cscamp --gres=gpu:1 -n 1 -c 1 -t 00:01:00 --pty bash
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
