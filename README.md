# BERT Finetune - HPCAI Camp 2026

[toc]

## Overview

* 環境設置：使用 Miniconda 建立虛擬環境並安裝固定版本的必要套件。
* 使用 Slurm 提交 GPU 訓練與推論 job。
* **主要操作：調整學習率、批次大小、訓練輪數與訓練資料量等超參數。**
* 目標：在不更換 BERT model 與 dataset 的前提下，達到超過基準準確率（30.95%）的表現。
* Tips：使用 `tmux`，避免訓練過程因 SSH 斷線而中斷。

> 操作過程中遇到問題，請保留 `bert-train.out`、`bert-train.err`、`bert-inf.out` 與 `bert-inf.err`，再詢問隊輔或助教。

## 環境設置

### 1. 登入台灣杉二號 TWCC

```bash
ssh <username>@ln01.twcc.ai
```

以下操作都使用自己的 `$HOME`，不要使用其他人的目錄。

### 2. 安裝相關套件

以下只需執行一次。請固定 Python 與 PyTorch 版本；`gp1d` 的 NVIDIA driver 與最新 CUDA 13 PyTorch 不相容。

```bash
# 載入 Miniconda 模組
module load miniconda3/conda24.5.0_py3.9

# 建立並啟動個人虛擬環境
export CAMP_ENV="$HOME/venvs/camp-ai"
export CONDA_PKGS_DIRS="$HOME/.conda/pkgs"
conda create --yes --prefix "$CAMP_ENV" python=3.12
conda activate "$CAMP_ENV"

# 確認目前使用的是自己的環境
which pip
which python
```

```bash
# 安裝 GPU 與 Python 套件
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

> `torch==2.5.1+cu121` 可在 `gp1d` 的 CUDA 12.2 driver 上執行。不要使用沒有版本限制的 `pip install torch`。

### 3. 準備 script

下載本次營隊使用的 repository：

```bash
cd "$HOME"
git clone https://github.com/NTHU-SC/BERT-Finetune-HPCAI-Camp-2026.git
cd BERT-Finetune-HPCAI-Camp-2026

module load miniconda3/conda24.5.0_py3.9
conda activate "$HOME/venvs/camp-ai"
python -m pip install -r requirements.txt

# 將 Hugging Face 模型與資料集快取放在自己的 home directory
export HF_HOME="$HOME/.cache/huggingface"
```

相關檔案介紹：

```text
├── Train.py          # 訓練腳本
├── Inference.py      # 推論腳本
├── run_train.sh      # Slurm 訓練提交腳本
├── run_inf.sh        # Slurm 推論提交腳本
└── requirements.txt  # 固定版本的 Python 套件
```

* `Train.py` 與 `Inference.py` 是主要程式。
* `run_train.sh` 與 `run_inf.sh` 會在 `gp1d` partition 申請一張 NVIDIA GPU。
* 預設 project 是 `GOV115003`；若使用不同 project，可在 `sbatch` 時加上 `-A <project-id>` 覆寫。

若 Hugging Face 要求登入或 token 已過期：

1. 到 <https://huggingface.co/settings/tokens> 建立 read-only token。
2. 在終端機執行：

   ```bash
   huggingface-cli login
   ```

3. 貼上 token（格式為 `hf_...`）。

第一次使用 GPU 時，先用短的互動式工作確認環境：

```bash
srun -A GOV115003 -p gp1d --gres=gpu:1 -n 1 -c 4 -t 00:05:00 --pty bash
module load miniconda3/conda24.5.0_py3.9
conda activate "$HOME/venvs/camp-ai"
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
exit
```

`torch.cuda.is_available()` 必須輸出 `True`。若為 `False`，請重新確認第 2 節的固定版 PyTorch 安裝，不要在 login node 直接訓練。

### 4. 送出 job 開始訓練

先確認目前位置在 repo 內：

```bash
cd "$HOME/BERT-Finetune-HPCAI-Camp-2026"
export CAMP_ENV="$HOME/venvs/camp-ai"
export HF_HOME="$HOME/.cache/huggingface"
```

```bash
# 方法 1：使用 Slurm 送出 job
sbatch run_train.sh <model_path> <output_dir>

# model_path：想 train / inference 的 model 路徑或 Hugging Face model 名稱
# output_dir：training checkpoint 的存放位置

# 範例
sbatch run_train.sh google-bert/bert-base-uncased ./output_model
```

若使用其他 project：

```bash
sbatch -A <project-id> run_train.sh google-bert/bert-base-uncased ./output_model
```

```bash
# 方法 2：在互動式 GPU shell 直接執行
srun -A GOV115003 -p gp1d --gres=gpu:1 -n 1 -c 4 -t 00:15:00 --pty bash
module load miniconda3/conda24.5.0_py3.9
conda activate "$HOME/venvs/camp-ai"
cd "$HOME/BERT-Finetune-HPCAI-Camp-2026"
python Train.py --model google-bert/bert-base-uncased --output ./output_model
exit
```

訓練完成後：

* `bert-train.out`：訓練輸出、training config 與 validation accuracy。
* `bert-train.err`：warning 與 error。
* `./output_model/checkpoint-final`：訓練完成的模型。

```bash
cat bert-train.out
cat bert-train.err
ls ./output_model/checkpoint-final
```

### 5. 送出 job 開始推論

確認訓練沒有問題後，使用訓練完成的 checkpoint：

```bash
sbatch run_inf.sh ./output_model/checkpoint-final
```

或先依第 4 節的「方法 2」取得互動式 GPU shell，再執行：

```bash
python Inference.py --model ./output_model/checkpoint-final
```

推論完成後：

```bash
cat bert-inf.out
cat bert-inf.err
```

`bert-inf.out` 中的 `The generation accuracy is ...` 即為完整 test split 的 accuracy。

### 6. 得到 Baseline 結果

`Inference.py` 會輸入完整 test split 的 3,432 筆句子給 BERT 分析。使用 Hugging Face Hub 上的 base model 時，可作為未微調的 baseline；請以完成訓練後的 checkpoint 進行正式比較。

## 目標

> Accuracy 越高越好！

透過 `Train.py` 微調 BERT。可以嘗試：

* 調整 hyperparameters。
* 調整 train / validation 資料量與分配。
* 提升 training epoch。
* 調整 batch size、learning rate 或 optimizer 相關策略。

可直接傳入的訓練參數：

```bash
sbatch run_train.sh google-bert/bert-base-uncased ./experiment-lr3e-5 \
  --train-samples 10000 \
  --eval-samples 500 \
  --batch-size 128 \
  --epochs 4 \
  --learning-rate 3e-5 \
  --seed 42
```

可調整的參數為 `--train-samples`、`--eval-samples`、`--batch-size`、`--epochs`、`--learning-rate` 與 `--seed`。建議一次只改一到兩個設定，並使用不同的 output directory 方便比較。

請注意：

* 不可以更換 model（BERT）與 dataset。
* 不可以手動修改資料內容。
* 不可以修改 `Inference.py`。

### Report

報告調整與優化的內容，沒有固定格式，但至少說明：

* 設定了哪些參數？
* 使用了哪些技巧？
* 各次實驗的 accuracy 與花費時間？
* 如何分配嘗試次數與 training 時間？
* 最後為何選擇這個模型繳交？

## 繳交檔案

* `bert-inf.out`
* `bert-inf.err`
* report：提供可閱讀的 HackMD 連結。

若不是透過 Slurm 執行推論，可將輸出同時寫入檔案：

```bash
python Inference.py --model <model_path> 2>&1 | tee bert-inf.out
```

## 繳交方法

使用 Google Form 繳交：**[Google Form（請在此替換為正式連結）](https://example.com/replace-with-google-form-url)**。

* 可以先繳交，之後再重新編輯。
* 系統取最佳結果。

### 如何將台灣杉的檔案 Pull 下來？

* `scp`
* VS Code Remote-SSH 側邊欄直接下載
* Copy & Paste

其他操作請依自己的環境調整；繳交時請保留完整 stdout 與 stderr。

## 附錄：常用 Slurm 指令

```text
Submitted batch job 730425
```

* `Submitted`：表示提交成功。
* `batch job`：表示提交的是批次工作。
* `730425`：Job ID。

```bash
# 查看指定工作狀態
squeue -j 730425

# 查看工作輸出
cat bert-train.out
cat bert-train.err

# 取消正在排隊或執行的工作
scancel 730425

# 查看工作詳細資訊
sacct -j 730425
```
