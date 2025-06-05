# 中醫問答模型評測系統 - 安裝設置指南

## 📋 系統要求

- Python 3.8+
- Windows/Linux/macOS
- 至少 8GB RAM（推薦 16GB+）
- 如果使用GPU推理，需要CUDA環境

## 🚀 快速開始

### 1. 環境準備

```bash
# 安裝cuda 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda


```bash
# 創建虛擬環境（推薦）
conda create -n tcm_eval python==3.9
conda activate tcm_eval

# 或使用 venv
python -m venv tcm_eval
# Windows
tcm_eval\Scripts\activate
# Linux/macOS
source tcm_eval/bin/activate
```

### 2. 安裝依賴

```bash
# 基礎依賴（必須）
pip install pandas numpy scipy matplotlib seaborn
# 模型推理依賴（根據你的模型選擇）
# 如果使用 Transformers + PyTorch
pip install accelerate protobuf sentencepiece
pip install torch==2.2.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.36.0
pip install markupsafe==2.1.3
pip install jinja2==3.1.2

# 如果使用 Transformers + CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 如果使用其他框架，請安裝相應依賴
```

### 3. 下載評測腳本

將以下文件保存到同一目錄：
- `tcm_evaluation.py` - 主評測系統
- `quick_evaluation.py` - 快速評測腳本
- `tcm_exam_question.csv` - 你的中醫問答資料集

### 4. 配置路徑

編輯 `quick_evaluation.py` 中的路徑配置：

```python
# 修改這些路徑為你的實際路徑
MODEL_PATH = r"C:\Users\user\Desktop\LLaMA-Factory\Breeze_lindan TCPM QA"
DATASET_PATH = "tcm_exam_question.csv"
OUTPUT_DIR = "quick_evaluation_results"
SAMPLE_SIZE = 500  # 快速測試用，設為 None 使用全部數據
```

### 5. 運行評測

```bash
# 快速評測
python quick_evaluation.py

# 或使用命令行版本（完整功能）
python tcm_evaluation.py --model_path "你的模型路徑" --dataset_path "tcm_exam_question.csv" --generate_charts
```

## 📊 評測流程說明

### 第一次使用流程

1. **模型載入**: 系統會載入你的微調模型
2. **資料集檢查**: 自動檢查資料集格式和內容
3. **樣本評測**: 對選定數量的題目進行推理
4. **指標計算**: 計算準確率、信心分數等指標
5. **結果保存**: 生成CSV、JSON、圖表等多種格式的結果

### 評測指標說明

- **準確率**: 模型答對題目的百分比
- **信心分數**: 模型對其答案的置信度
- **回應時間**: 模型推理每道題的平均時間
- **分類別分析**: 按中醫科目分類的表現
- **錯誤分析**: 詳細的錯誤案例分析

## 🔧 模型適配

### 支援的模型格式

1. **Transformers 格式** (推薦)
   - Llama, Qwen, ChatGLM 等
   - 使用 AutoModelForCausalLM 載入

2. **自定義格式**
   - 修改 `get_model_answer()` 方法
   - 實現你的推理邏輯

### 模型適配範例

如果你的模型需要特殊載入方式，修改 `load_model()` 方法：

```python
def load_model(self):
    """自定義模型載入"""
    try:
        # 你的自定義載入邏輯
        from your_model_library import YourModel
        
        self.model = YourModel.from_pretrained(self.model_path)
        self.tokenizer = YourTokenizer.from_pretrained(self.model_path)
        
        self.logger.info("自定義模型載入成功")
        
    except Exception as e:
        self.logger.error(f"載入模型失敗: {e}")
        self.model = None
```

### 提示詞客製化

修改 `format_prompt()` 方法來調整提示詞格式：

```python
def format_prompt(self, question: str) -> str:
    """客製化提示詞"""
    # 範例：加入角色設定和特殊格式
    prompt = f"""<|system|>
你是一位資深的中醫師，具有豐富的臨床經驗和深厚的理論基礎。

<|user|>
請根據以下中醫問題選擇正確答案，只需回答選項字母（A、B、C或D）：

{question}

<|assistant|>
"""
    return prompt
```

## 🛠️ 進階功能

### 1. 批量模型比較

```python
from tcm_evaluation import BatchEvaluator

# 準備多個模型路徑
models = [
    ("baseline", "path/to/baseline/model"),
    ("fine_tuned", "path/to/fine_tuned/model"),
    ("ensemble", "path/to/ensemble/model")
]

# 批量評測
batch_evaluator = BatchEvaluator("tcm_exam_question.csv")
results = batch_evaluator.evaluate_multiple_models(
    model_paths=[path for _, path in models],
    model_names=[name for name, _ in models],
    sample_size=1000
)
```

### 2. 自定義評測指標

```python
def calculate_custom_metrics(self, results):
    """自定義評測指標"""
    # 計算 F1 分數（多分類）
    from sklearn.metrics import classification_report
    
    y_true = [r.correct_answer for r in results]
    y_pred = [r.model_answer for r in results]
    
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # 計算各選項的準確率
    option_accuracy = {}
    for option in ['A', 'B', 'C', 'D']:
        option_results = [r for r in results if r.correct_answer == option]
        if option_results:
            correct = sum(1 for r in option_results if r.is_correct)
            option_accuracy[f'option_{option}'] = correct / len(option_results)
    
    return {
        'classification_report': report,
        'option_accuracy': option_accuracy
    }
```

### 3. 錯誤分析增強

```python
def analyze_errors(self, results):
    """詳細錯誤分析"""
    incorrect_results = [r for r in results if not r.is_correct]
    
    # 按錯誤類型分類
    error_patterns = {
        'A_to_B': 0, 'A_to_C': 0, 'A_to_D': 0,
        'B_to_A': 0, 'B_to_C': 0, 'B_to_D': 0,
        # ... 其他組合
    }
    
    for result in incorrect_results:
        pattern = f"{result.correct_answer}_to_{result.model_answer}"
        if pattern in error_patterns:
            error_patterns[pattern] += 1
    
    # 找出高頻錯誤模式
    frequent_errors = sorted(
        error_patterns.items(), 
        key=lambda x: x[1], 
        reverse=True
    )[:5]
    
    return {
        'total_errors': len(incorrect_results),
        'error_patterns': error_patterns,
        'frequent_errors': frequent_errors
    }
```

## 📈 結果解讀指南

### 準確率標準

- **90%+**: 優秀，接近專業水平
- **80-90%**: 良好，具備基本中醫知識
- **70-80%**: 中等，需要進一步改進
- **<70%**: 較差，建議重新訓練

### 信心分數分析

- **高信心+正確**: 模型表現最佳
- **高信心+錯誤**: 需要關注的過度自信案例
- **低信心+正確**: 潛在的知識盲點
- **低信心+錯誤**: 模型不確定的領域

### t檢定結果解讀

- **p < 0.05**: 統計顯著差異
- **Cohen's d**:
  - 0.2: 小效應
  - 0.5: 中等效應
  - 0.8: 大效應

## 🔍 故障排除

### 常見問題

1. **ModuleNotFoundError: No module named 'transformers'**
   ```bash
   pip install transformers torch
   ```

2. **CUDA out of memory**
   ```python
   # 減少batch_size或使用CPU
   torch_dtype=torch.float16  # 使用半精度
   device_map="cpu"  # 強制使用CPU
   ```

3. **模型載入失敗**
   ```python
   # 檢查模型路徑和權限
   # 確保模型文件完整
   # 檢查Python版本相容性
   ```

4. **評測速度慢**
   ```python
   # 使用較小樣本
   SAMPLE_SIZE = 100
   
   # 使用GPU加速
   device_map="auto"
   
   # 減少max_new_tokens
   max_new_tokens=5
   ```

### 調試模式

啟用詳細日誌：

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 效能優化

1. **GPU記憶體優化**
   ```python
   # 使用gradient checkpointing
   model.gradient_checkpointing_enable()
   
   # 清理緩存
   torch.cuda.empty_cache()
   ```

2. **推理加速**
   ```python
   # 使用compiled模型 (PyTorch 2.0+)
   model = torch.compile(model)
   
   # 使用TensorRT或ONNX
   # 批量推理
   ```

## 📝 輸出文件說明

### 文件結構
```
evaluation_results/
├── detailed_results_20241205_143022.csv     # 詳細結果
├── evaluation_report_20241205_143022.json   # JSON報告
├── summary_report_20241205_143022.txt       # 文字總結
├── evaluation_charts.png                    # 視覺化圖表
└── evaluation_20241205_143022.log          # 運行日誌
```

### CSV文件欄位說明

- `question_id`: 題目編號
- `question`: 題目內容
- `correct_answer`: 正確答案
- `model_answer`: 模型答案
- `is_correct`: 是否正確
- `confidence_score`: 信心分數
- `response_time`: 回應時間
- `category`: 題目類別

## 🎯 最佳實踐

### 評測策略

1. **分階段評測**
   - 先用小樣本快速測試
   - 確認無誤後進行全量評測
   - 定期重新評測追蹤進步

2. **多維度分析**
   - 整體準確率
   - 分類別表現
   - 錯誤模式分析
   - 信心校準

3. **結果驗證**
   - 人工檢查部分結果
   - 交叉驗證
   - 與專家評估對比

### 報告撰寫

1. **執行摘要**: 關鍵指標和結論
2. **方法描述**: 評測設置和參數
3. **結果分析**: 詳細數據和圖表
4. **改進建議**: 基於結果的優化方向

## 🚀 下一步

1. 運行快速評測確認系統正常
2. 調整樣本大小進行全面評測
3. 分析結果並識別改進點
4. 如有多個模型版本，進行比較分析
5. 根據結果調整訓練策略

## 💡 提示

- 首次運行建議使用小樣本（100-500題）
- 記錄每次評測的設置和結果
- 定期備份評測結果
- 建立評測基準線用於後續比較

---

有任何問題請查看日誌文件或聯繫技術支援！