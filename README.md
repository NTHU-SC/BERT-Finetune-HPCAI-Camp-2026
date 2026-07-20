# BERT 微調實作：HPCAI Camp AMD 環境

這個實作會帶你把預訓練好的 BERT，微調成能辨識推文情緒的三分類
模型（negative、neutral、positive）。你不需要從零訓練模型；重點是理解
資料、超參數、GPU 與 Slurm 如何一起影響一次實驗的結果。

完成本教學後，你應該能夠：

- 在 CSCC 的 AMD GPU 節點建立自己的 Python 環境。
- 用 Slurm 送出訓練與推論工作，而不是佔用登入節點。
- 從輸出檔判斷工作是否成功、讀出 accuracy 與 checkpoint。
- 有系統地調整訓練資料量、batch size、epoch、learning rate 等設定。

> 遇到問題時，先保留 `bert-train.err`、`bert-train.out`，再向隊輔或助教求助。
> 這兩個檔案通常比只貼終端機最後一行更容易找出問題。

## 先認識這個專案

專案根目錄中的檔案各自負責不同工作：

```text
.
├── Train.py        # 載入資料、微調 BERT、驗證並儲存模型
├── Inference.py    # 使用訓練好的模型在測試集計算 accuracy
├── run_train.sh    # 向 Slurm 申請 GPU 後執行 Train.py
├── run_inf.sh      # 向 Slurm 申請 GPU 後執行 Inference.py
└── README.md       # 本教學
```

資料集是 `mteb/tweet_sentiment_extraction`。`Train.py` 會依固定 seed 將
訓練集分成兩個不重疊的部分：預設使用 10,000 筆做訓練、500 筆做驗證。
驗證集的 accuracy 會在每個 epoch 後寫進訓練輸出；最後的真正成績則由
`Inference.py` 在完整 test split 上計算。

本活動中請遵守以下限制：不要更換 BERT 模型、資料集或資料內容，也不要
修改 `Inference.py`。可以調整訓練策略和超參數，並在報告中說明實驗過程。

## 第一次使用：建立自己的環境

請在自己的 home directory 工作。以下的 `$HOME` 會自動代表你的 home
directory，因此不要把它換成其他同學的路徑。

先登入 CSCC，下載專案：

```bash
ssh CSCC
cd "$HOME"
git clone https://github.com/NTHU-SC/BERT-Finetune-HPCAI-Camp-2026.git
cd BERT-Finetune-HPCAI-Camp-2026
```

接著建立只屬於自己的 virtual environment。這個步驟只需做一次；日後只要
重新啟用它即可。

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

確認目前使用的是剛建立的環境，而不是系統 Python：

```bash
which python
python -c 'import torch, transformers; print(torch.__version__); print(transformers.__version__)'
```

`which python` 的結果應該位於 `$HOME/venvs/camp-ai/` 下方。若 Hugging Face
要求登入，請到 <https://huggingface.co/settings/tokens> 建立 read-only token，
再執行：

```bash
huggingface-cli login
```

### 為什麼登入後還看不到 GPU？

登入節點是用來編輯檔案、安裝套件與提交工作，不是用來跑訓練的地方。GPU
由 Slurm 管理，所以要先取得資源配置。可用下面指令做一次互動式檢查：

```bash
srun -p cscamp --gres=gpu:1 -n 1 -c 1 -t 00:05:00 --pty bash
module load rocm/7.2.0
source "$HOME/venvs/camp-ai/bin/activate"
python -c 'import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
exit
```

ROCm 版 PyTorch 仍沿用 `torch.cuda` 這個 API 名稱；第一行輸出 `True` 即表示
程式已經看得到 AMD GPU。

## 第一次訓練：把工作交給 Slurm

回到專案根目錄後，執行：

```bash
sbatch run_train.sh google-bert/bert-base-uncased ./output_model
```

這不是立刻在目前終端機執行訓練，而是提交一個 batch job。`run_train.sh` 會
申請 `cscamp` partition 的一張 GPU、16 個 CPU core，並給工作最長五分鐘的
執行時間。第一次使用某個模型或資料集時，下載與建立快取可能較慢；之後通常
會快很多。

提交成功後會看到類似：

```text
Submitted batch job 123456
```

這個數字是 job ID。常用的觀察方式如下：

```bash
# 查看自己正在排隊或執行的工作
squeue -u "$USER"

# 查看工作完成後的狀態與實際資源使用情形
sacct -j 123456

# 取消尚未結束的工作
scancel 123456
```

工作結束後，Slurm 會在專案根目錄寫出：

- `bert-train.out`：訓練進度、驗證 accuracy、訓練設定與 checkpoint 資訊。
- `bert-train.err`：警告與錯誤訊息；工作失敗時先從這裡開始看。
- `output_model/checkpoint-final`：本次訓練完成後儲存的模型。

可以用以下指令快速查看結果：

```bash
tail -n 60 bert-train.out
tail -n 60 bert-train.err
ls ./output_model/checkpoint-final
```

訓練輸出中的 `eval_accuracy` 是保留的 validation split 結果，用來比較實驗
方向是否有幫助；它不是最後繳交成績。每次訓練開始時也會印出 `Training Config:`，
請保留這一段，因為它記錄了本次實驗真正使用的設定。

## 執行推論並取得成績

模型訓練完成後，將 checkpoint 提供給推論腳本：

```bash
sbatch run_inf.sh ./output_model/checkpoint-final
```

完成後查看：

```bash
cat bert-inf.out
cat bert-inf.err
```

`bert-inf.out` 中的 `The generation accuracy is ...` 是完整 test split 的
accuracy。先前在 warm cache 的參考執行中，預設設定的訓練工作約需 51 秒、
推論工作約需 28 秒，test accuracy 為 40.7051%。這些數字會隨當下排程、首次
下載與 GPU 使用狀況而變動，請以自己的輸出檔為準。

## 怎麼開始調參？

好的實驗不是一次改很多東西，而是先建立一個可重現的 baseline，再一次只改
一到兩個因素。`Train.py` 提供下列參數：

| 參數 | 預設值 | 影響 |
| --- | ---: | --- |
| `--train-samples` | 10000 | 使用多少筆資料訓練；越多通常越完整，但更花時間。 |
| `--eval-samples` | 500 | 保留多少筆做 validation，不會加入訓練。 |
| `--batch-size` | 128 | 每次更新所用的樣本數；過大可能耗盡 GPU 記憶體。 |
| `--epochs` | 5 | 完整看過訓練資料的次數；太多可能過度擬合。 |
| `--learning-rate` | `5e-5` | 每次更新的步伐；過大可能不穩，過小可能學得太慢。 |
| `--seed` | 42 | 控制抽樣與初始化的隨機性，方便重現結果。 |

在取得 GPU 的互動式 shell 中，可直接嘗試不同參數。例如：

```bash
python Train.py \
  --model google-bert/bert-base-uncased \
  --output ./experiment-lr3e-5 \
  --learning-rate 3e-5 \
  --epochs 4
```

建議的實驗節奏是：

1. 先跑預設設定，確認訓練、checkpoint 與推論流程都正確。
2. 用較少的 `--train-samples` 或 `--epochs` 快速探索方向。
3. 選出 validation accuracy 較佳的少數設定，再用較完整的資料量重跑。
4. 對候選 checkpoint 執行 `Inference.py`，以 test accuracy 做最後比較。

每個實驗請使用不同的 `--output` 目錄，例如 `./experiment-bs64`、
`./experiment-epoch3`。這樣 checkpoint 不會互相覆蓋，也能回頭比較輸出。
若發生 CUDA out-of-memory，先降低 `--batch-size`，例如從 128 改成 64 或 32。

## 除錯小抄

| 現象 | 先做什麼 |
| --- | --- |
| job 一直是 `PENDING` | 用 `squeue -u "$USER"` 查看原因；通常是在等 GPU。 |
| job 顯示 `FAILED` 或 `TIMEOUT` | 用 `sacct -j <job-id>` 搭配 `bert-*.err` 確認原因。 |
| `torch.cuda.is_available()` 是 `False` | 確認在 Slurm 配置內、已 `module load rocm/7.2.0`，並啟用自己的 venv。 |
| 找不到模型或資料集 | 檢查網路／Hugging Face token，之後重新送出工作。 |
| GPU 記憶體不足 | 降低 `--batch-size`，再重新訓練。 |

## 繳交前檢查

請確認保留以下檔案：

- `bert-inf.out`
- `bert-inf.err`
- 一份可閱讀的報告（例如 HackMD 連結）

報告不需要固定格式，但至少應說明：你嘗試了哪些設定、每個設定的結果、如何
選出最後模型，以及遇到的問題和解法。請以活動公告的繳交方式為準。
