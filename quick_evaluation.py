#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中醫問答模型快速評測腳本
簡化版本，適合快速測試
"""

import sys
import os
from pathlib import Path

# plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# 確保可以導入主評測系統
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from tcm_evaluation import TCMEvaluationSystem, ModelComparator
except ImportError:
    print("錯誤: 無法導入評測系統，請確保 tcm_evaluation.py 在同一目錄下")
    sys.exit(1)

def quick_evaluation():
    """快速評測函數"""
    
    # 配置參數 - 請根據你的實際情況修改
    MODEL_PATH = r"C:\Users\user\Desktop\TCM_Project\SFT_weights\Breeze_lindan_TCPM_QA"
    DATASET_PATH = "tcm_exam_question.csv"
    OUTPUT_DIR = "quick_evaluation_results"
    SAMPLE_SIZE = 500  # 快速測試用較小樣本，設為 None 使用全部數據
    
    print("=" * 60)
    print("中醫問答模型快速評測")
    print("=" * 60)
    print(f"模型路徑: {MODEL_PATH}")
    print(f"資料集: {DATASET_PATH}")
    print(f"樣本數: {SAMPLE_SIZE if SAMPLE_SIZE else '全部'}")
    print(f"輸出目錄: {OUTPUT_DIR}")
    print()
    
    try:
        # 1. 初始化評測系統
        print("🚀 初始化評測系統...")
        evaluator = TCMEvaluationSystem(
            model_path=MODEL_PATH,
            dataset_path=DATASET_PATH,
            output_dir=OUTPUT_DIR
        )
        
        # 2. 執行評測
        print("📝 開始評測...")
        results = evaluator.evaluate_sample(sample_size=SAMPLE_SIZE, random_seed=42)
        
        if not results:
            print("❌ 評測失敗：沒有獲得任何結果")
            return
        
        # 3. 計算指標
        print("📊 計算評測指標...")
        metrics = evaluator.calculate_metrics(results)
        
        # 4. 生成圖表
        print("📈 生成視覺化圖表...")
        evaluator.generate_visualizations(results, metrics)
        
        # 5. 保存結果
        print("💾 保存評測結果...")
        evaluator.save_results(results, metrics)
        
        # 6. 顯示關鍵結果
        print("\n" + "=" * 50)
        print("✅ 評測完成！關鍵結果：")
        print("=" * 50)
        
        overall = metrics.get('overall', {})
        print(f"📈 總準確率: {overall.get('accuracy', 0):.2f}%")
        print(f"📝 評測題數: {overall.get('total_questions', 0)}")
        print(f"🎯 正確題數: {overall.get('correct_answers', 0)}")
        print(f"💪 平均信心分數: {overall.get('avg_confidence', 0):.3f}")
        print(f"⏱️  平均回應時間: {overall.get('avg_response_time', 0):.3f}秒")
        
        # 7. 各類別表現
        if 'by_category' in metrics and metrics['by_category']:
            print("\n📋 各類別表現:")
            for category, stats in metrics['by_category'].items():
                print(f"  {category}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
        
        # 8. 輸出文件位置
        print(f"\n📁 詳細結果保存在: {OUTPUT_DIR}/")
        print("   - detailed_results_*.csv: 詳細評測結果")
        print("   - evaluation_report_*.json: JSON格式報告")
        print("   - summary_report_*.txt: 人類可讀總結")
        print("   - evaluation_charts.png: 視覺化圖表")
        
        print("\n🎉 評測完成！")
        
    except FileNotFoundError as e:
        print(f"❌ 文件不存在: {e}")
        print("請檢查模型路徑和資料集路徑是否正確")
    except Exception as e:
        print(f"❌ 評測過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

def compare_models():
    """模型比較範例"""
    
    # 配置 - 請根據實際情況修改
    MODEL1_PATH = r"C:\Users\user\Desktop\專題\SFT_weights\Breeze_lindan TCPM QA"  # 你的微調模型
    MODEL2_PATH = r"C:\Users\user\Desktop\專題\SFT_weights\Breeze_lindan TCPM QA"  # 基礎模型（如果有的話）
    DATASET_PATH = "tcm_exam_question.csv"
    OUTPUT_DIR = "model_comparison_results"
    SAMPLE_SIZE = 300  # 比較時建議使用較小樣本以節省時間
    
    print("=" * 60)
    print("模型比較評測（含t檢定）")
    print("=" * 60)
    
    if not os.path.exists(MODEL2_PATH):
        print("⚠️  未找到比較模型，請修改 MODEL2_PATH 或跳過比較")
        return
    
    try:
        # 評測模型1
        print("📝 評測模型1...")
        evaluator1 = TCMEvaluationSystem(MODEL1_PATH, DATASET_PATH, OUTPUT_DIR)
        results1 = evaluator1.evaluate_sample(sample_size=SAMPLE_SIZE, random_seed=42)
        metrics1 = evaluator1.calculate_metrics(results1)
        
        # 評測模型2
        print("📝 評測模型2...")
        evaluator2 = TCMEvaluationSystem(MODEL2_PATH, DATASET_PATH, OUTPUT_DIR)
        results2 = evaluator2.evaluate_sample(sample_size=SAMPLE_SIZE, random_seed=42)
        metrics2 = evaluator2.calculate_metrics(results2)
        
        # 執行t檢定
        print("📊 執行成對樣本t檢定...")
        t_test_result = evaluator1.paired_t_test(results1, results2)
        
        # 保存結果
        evaluator1.save_results(results1, metrics1, t_test_result, "model1_")
        evaluator2.save_results(results2, metrics2, output_prefix="model2_")
        
        # 顯示比較結果
        print("\n" + "=" * 50)
        print("📊 模型比較結果：")
        print("=" * 50)
        
        acc1 = metrics1['overall']['accuracy']
        acc2 = metrics2['overall']['accuracy']
        
        print(f"模型1準確率: {acc1:.2f}%")
        print(f"模型2準確率: {acc2:.2f}%")
        print(f"準確率差異: {acc1 - acc2:+.2f}%")
        
        print(f"\nt檢定結果:")
        print(f"  t統計量: {t_test_result['t_statistic']:.4f}")
        print(f"  p值: {t_test_result['p_value']:.6f}")
        print(f"  統計顯著: {'是' if t_test_result['significant'] else '否'}")
        print(f"  效應量(Cohen's d): {t_test_result['cohen_d']:.4f}")
        print(f"  結果解釋: {t_test_result['interpretation']}")
        
    except Exception as e:
        print(f"❌ 比較評測失敗: {e}")

def main():
    """主函數"""
    print("中醫問答模型評測工具")
    print("1. 快速評測")
    print("2. 模型比較（含t檢定）")
    print("3. 退出")
    
    while True:
        choice = input("\n請選擇操作 (1-3): ").strip()
        
        if choice == '1':
            quick_evaluation()
            break
        elif choice == '2':
            compare_models()
            break
        elif choice == '3':
            print("再見！")
            break
        else:
            print("請輸入有效選項 (1-3)")

if __name__ == "__main__":
    main()