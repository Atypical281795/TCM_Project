
import sys
sys.path.append('.')
from tcm_evaluation import TCMEvaluationSystem

# 測試載入
evaluator = TCMEvaluationSystem(
    model_path=r"C:\Users\user\Desktop\TCM_Project\SFT_weights\Breeze_lindan_TCPM_QA",
    dataset_path="tcm_exam_question.csv",
    output_dir="test_output"
)

# 檢查模型是否載入
if evaluator.model is not None:
    print("✅ 模型載入成功！")
    print(f"模型類型: {type(evaluator.model).__name__}")
    print(f"Tokenizer類型: {type(evaluator.tokenizer).__name__}")
    
    # 測試一個問題
    test_question = "下列何者為麻黃湯的組成？ (A)麻黃、桂枝、杏仁、甘草 (B)麻黃、桂枝、芍藥、甘草 (C)麻黃、石膏、杏仁、甘草 (D)麻黃、細辛、附子"
    answer, confidence, time = evaluator.get_model_answer(test_question)
    print(f"\n測試問題回答: {answer}")
    print(f"信心分數: {confidence:.3f}")
    print(f"回應時間: {time:.3f}秒")
else:
    print("❌ 模型載入失敗")
