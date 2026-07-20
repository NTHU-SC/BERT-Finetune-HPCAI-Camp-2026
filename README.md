# BERT Finetune - HPCAI Camp

[toc]

## 總覽

- 環境設置：在 CSCC 的 AMD GPU 環境建立自己的 Python virtual environment。
- 使用 Slurm 提交訓練與推論工作。
- **主要操作：調整 learning rate、batch size、training epoch、訓練資料量等超參數。**
- 目標：在不更換模型與資料集的前提下，提升 test accuracy。
- Tips：使用 `tmux` 可以避免 SSH 斷線時中斷正在做的安裝或查看工作。

> 操作過程中遇到任何問題，請保留 `bert-train.out`、`bert-train.err`，再詢問
> 隊輔或助教。這兩個檔案通常能顯示完整的錯誤原因。

## 環境設置

### 1. 登入 CSCC

```bash
ssh CSCC
```

之後所有檔案都請放在自己的 home directory。`$HOME` 會自動代表你的目錄，
請不要使用其他同學的路徑。

### 2. 建立個人 Python 環境並安裝套件

以下只需執行一次。它會在自己的 home directory 建立環境，不會修改系統 Python
或其他人的環境。

```bash
module load rocm/7.2.0
python3 -m venv "$HOME/venvs/camp-ai"
source "$HOME/venvs/camp-ai/bin/activate"

# 確認目前使用的是自己的環境
which pip
which python

# 安裝套件
python -m pip install --upgrade pip
python -m pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/rocm7.2
python -m pip install transformers datasets scikit-learn evaluate accelerate \
  --upgrade huggingface_hub
```

確認安裝：

```bash
python -c 'import torch; print(torch.__version__); print(torch.version.hip)'
python -c 'import transformers; print(transformers.__version__)'
```

若 Hugging Face 要求登入，請到 <https://huggingface.co/settings/tokens> 建立一個
read-only token，然後執行：

```bash
huggingface-cli login
```

### 3. 準備腳本

下載本次營隊使用的 repository：

```bash
cd "$HOME"
git clone https://github.com/NTHU-SC/BERT-Finetune-HPCAI-Camp-2026.git
cd BERT-Finetune-HPCAI-Camp-2026
```

相關檔案介紹：

```text
├── Train.py          # 訓練腳本
├── Inference.py      # 推論與 accuracy 計算腳本
├── run_train.sh      # Slurm 訓練工作腳本
└── run_inf.sh        # Slurm 推論工作腳本
```

- `Train.py` 和 `Inference.py` 是主要程式。
- `run_train.sh` 和 `run_inf.sh` 會向 Slurm 的 `cscamp` partition 申請一張 AMD GPU。
- 每個工作會申請 16 個 CPU core，最長執行五分鐘。

第一次使用 GPU 時，可以先用互動式工作確認 ROCm PyTorch 正常：

```bash
srun -p cscamp --gres=gpu:1 -n 1 -c 1 -t 00:05:00 --pty bash
module load rocm/7.2.0
source "$HOME/venvs/camp-ai/bin/activate"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
exit
```

ROCm 版 PyTorch 仍使用 `torch.cuda` 這個 API 名稱；看到 `True` 即表示 GPU 可用。

### 4. 送出工作開始訓練

先確認目前位置在 repository 內：

```bash
cd "$HOME/BERT-Finetune-HPCAI-Camp-2026"
```

使用 Slurm 送出訓練工作：

```bash
sbatch run_train.sh <model_path> <output_dir>
```

- `model_path`：要訓練的 model 名稱或路徑。
- `output_dir`：訓練完成後存放 checkpoint 的目錄。

範例：

```bash
sbatch run_train.sh google-bert/bert-base-uncased ./output_model
```

提交成功會看到：

```text
Submitted batch job <job-id>
```

訓練完成後會出現：

- `bert-train.out`：訓練輸出、每個 epoch 的 `eval_accuracy` 與 training config。
- `bert-train.err`：警告與錯誤訊息。
- `./output_model/checkpoint-final`：訓練好的最終模型。

可用以下指令查看：

```bash
cat bert-train.out
cat bert-train.err
ls ./output_model/checkpoint-final
```

若要直接執行 Python 版本，請先透過 `srun` 取得 GPU，再在該 shell 中執行：

```bash
python Train.py --model google-bert/bert-base-uncased --output ./output_model
```

### 5. 送出工作開始推論

確認訓練完成後，使用剛剛的 checkpoint 送出推論工作：

```bash
sbatch run_inf.sh ./output_model/checkpoint-final
```

或是在已取得 GPU 的互動式 shell 中直接執行：

```bash
python Inference.py --model ./output_model/checkpoint-final
```

推論完成後查看：

```bash
cat bert-inf.out
cat bert-inf.err
```

`bert-inf.out` 裡的 `The generation accuracy is ...` 就是完整 test split 的
accuracy。預設設定在 warm cache 的參考執行中，訓練約需 51 秒、推論約需 28 秒，
test accuracy 為 40.7051%；首次下載模型、資料集或多人同時使用 GPU 時，時間可能
較長。

## 目標

> Accuracy 越高越好！

請透過 `Train.py` 微調 BERT。可以嘗試：

- 調整 hyperparameters。
- 調整訓練資料量與 validation 資料量。
- 調整 training epoch、batch size、learning rate。
- 調整 optimizer 或其他訓練策略。

目前 `Train.py` 可使用的參數如下：

```text
--train-samples   預設 10000
--eval-samples    預設 500
--batch-size      預設 128
--epochs          預設 5
--learning-rate   預設 5e-5
--seed            預設 42
```

例如，在互動式 GPU shell 中嘗試不同設定：

```bash
python Train.py \
  --model google-bert/bert-base-uncased \
  --output ./experiment-lr3e-5 \
  --learning-rate 3e-5 \
  --epochs 4
```

建議一次只改一到兩個設定，並為每次實驗使用不同的 output directory。這樣較容易
比較 `eval_accuracy`，也不會覆蓋之前的 checkpoint。若發生 GPU out-of-memory，
先將 `--batch-size` 降為 64 或 32。

請注意：

- 不可以更換 model（BERT）與 dataset。
- 不可以手動修改資料內容。
- 不可以修改 `Inference.py`。

## Report

報告你調整和優化的內容，沒有固定格式，但至少應說明：

- 調整了哪些參數？
- 使用了哪些技巧？
- 每次實驗的 accuracy 與花費時間？
- 如何分配嘗試次數與 training 時間？
- 最後為什麼選擇這個模型繳交？

## 繳交檔案

- `bert-inf.out`
- `bert-inf.err`
- report：提供可閱讀的 HackMD 連結

請透過[營隊繳交 Google 表單](https://docs.google.com/forms/d/e/1FAIpQLSfMzM9dPfWS5OMAoQ8md5lQY9zLLyE1xfyPTL8en3Ko7M73Rg/viewform?usp=publish-editor)上傳。
若不是透過 Slurm 執行推論，可把輸出同時寫入檔案：

```bash
python Inference.py --model <model_path> 2>&1 | tee bert-inf.out
```

## 附錄：常用 Slurm 指令

```bash
# 查看自己的工作
squeue -u "$USER"

# 查看指定工作狀態與資源使用情形
sacct -j <job-id>

# 取消正在排隊或執行的工作
scancel <job-id>
```

`PENDING` 表示正在等資源，`RUNNING` 表示正在執行，`COMPLETED` 表示工作成功。
如果看到 `FAILED` 或 `TIMEOUT`，請先查看 `bert-train.err` 或 `bert-inf.err`。
