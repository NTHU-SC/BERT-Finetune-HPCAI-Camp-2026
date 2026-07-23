# BERT Finetune - HPCAI Camp 2026（TWCC）

本版流程適用於台灣杉二號的 `twcc` login node 與 NVIDIA `gp1d` GPU partition。所有專案、Python 環境與 Hugging Face cache 都放在 `$WORK`；不要在 login node 上直接訓練或推論。

## 先決條件

登入後確認工作空間與可用 Slurm project：

```bash
ssh twcc
echo "$WORK"
sacctmgr -n show user "$USER" format=User,Account  # 若此指令可用
```

這份 repo 預設使用 `ACD110018`。若你有不同的 project，提交時以 `sbatch -A <project-id> ...` 覆寫即可。

## 1. 建立固定版本的環境

下列指令只需執行一次。請不要使用 `conda create ... python` 或 `pip install torch` 的「最新版本」：目前會安裝 Python 3.14 與 CUDA 13 的 PyTorch，和 `gp1d` 的 NVIDIA driver 不相容。

```bash
module load miniconda3/conda24.5.0_py3.9

export CAMP_ENV="$WORK/venvs/camp-ai"
export CONDA_PKGS_DIRS="$WORK/.conda/pkgs"
conda create --yes --prefix "$CAMP_ENV" python=3.12
conda activate "$CAMP_ENV"

python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

`torch==2.5.1` 的 CUDA 12.1 wheel 可在 `gp1d` 的 CUDA 12.2 driver 上使用。

## 2. 取得程式碼與設定快取

```bash
cd "$WORK"
git clone https://github.com/NTHU-SC/BERT-Finetune-HPCAI-Camp-2026.git
cd BERT-Finetune-HPCAI-Camp-2026

module load miniconda3/conda24.5.0_py3.9
conda activate "$WORK/venvs/camp-ai"
python -m pip install -r requirements.txt

# 讓模型與資料集快取也留在 $WORK。
export HF_HOME="$WORK/.cache/huggingface"
```

`requirements.txt` 釘選其餘相容套件版本，請不要自行把其中任何一項升級到最新 major version。

若 Hugging Face 要求登入，建立 read-only token 後執行：

```bash
huggingface-cli login
```

## 3. 確認 GPU 環境

只能在 GPU node 上檢查 CUDA。以下指令會建立一個很短的互動式工作：

```bash
srun -A ACD110018 -p gp1d --gres=gpu:1 -n 1 -c 4 -t 00:05:00 --pty bash
module load miniconda3/conda24.5.0_py3.9
conda activate "$WORK/venvs/camp-ai"
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
exit
```

最後一行必須輸出 GPU 型號，且 `torch.cuda.is_available()` 必須是 `True`。若是 `False`，請重新依照第 1 節安裝固定的 CUDA 12.1 PyTorch，不要開始訓練。

## 4. 使用 Slurm 訓練

在 repo 目錄中執行：

```bash
export CAMP_ENV="$WORK/venvs/camp-ai"
export HF_HOME="$WORK/.cache/huggingface"
sbatch run_train.sh google-bert/bert-base-uncased ./output_model
```

若使用不同 project：

```bash
sbatch -A <project-id> run_train.sh google-bert/bert-base-uncased ./output_model
```

輸出檔：

- `bert-train.out`：training config、訓練與 validation accuracy。
- `bert-train.err`：warning 與 error。
- `output_model/checkpoint-final`：最後訓練完成的模型。

預設會使用 10,000 筆 train、500 筆 validation。可調整參數：

```bash
sbatch run_train.sh google-bert/bert-base-uncased ./experiment-lr3e-5 \
  --learning-rate 3e-5 --epochs 4
```

可用參數為：`--train-samples`、`--eval-samples`、`--batch-size`、`--epochs`、`--learning-rate`、`--seed`。每次實驗請用新的 output directory，避免覆寫模型。

> `run_train.sh` 會將其後的所有參數傳給 `Train.py`。第一個參數是 model、第二個參數是 output directory。

## 5. 使用 Slurm 推論

訓練完成後：

```bash
sbatch run_inf.sh ./output_model/checkpoint-final
```

查看結果：

```bash
cat bert-inf.out
cat bert-inf.err
```

`bert-inf.out` 中的 `The generation accuracy is ...` 是 test split accuracy。

## 規則與繳交

- 不可更換 BERT model 或 dataset。
- 不可手動修改資料內容。
- 不可修改 `Inference.py`。
- 繳交 `bert-inf.out`、`bert-inf.err` 與可閱讀的 HackMD report。

Report 至少說明調整的參數、每次實驗的 accuracy／耗時與最後選擇。

## 常用 Slurm 指令

```bash
squeue -u "$USER"
sacct -j <job-id>
scancel <job-id>
```

若工作失敗，先閱讀對應的 `bert-train.err` 或 `bert-inf.err`。
