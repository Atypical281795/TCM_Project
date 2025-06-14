#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中醫問答模型評測系統
支援多種評測指標、成對樣本t檢定、詳細報告生成
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import re
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import argparse
import matplotlib.font_manager

# 統計分析
from scipy import stats
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import platform

# 設定中文字體
def setup_matplotlib_chinese():
    system = platform.system()
    if system == 'Windows':
        # Windows 字體設定
        font_options = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Microsoft JhengHei']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        
        for font in font_options:
            if font in available_fonts:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                return
        
        # 使用字體檔案
        import os
        font_path = r"C:\Windows\Fonts\msyh.ttc"
        if os.path.exists(font_path):
            font_prop = fm.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
    elif system == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

# 執行字體設定
setup_matplotlib_chinese()

# 模型推理 (根據你使用的框架調整)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers 未安裝，請根據需要安裝相關依賴")

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class EvaluationResult:
    """評測結果數據結構"""
    question_id: int
    question: str
    correct_answer: str
    model_answer: str
    is_correct: bool
    confidence_score: float
    response_time: float
    category: str

class TCMEvaluationSystem:
    """中醫問答模型評測系統"""
    
    def __init__(self, model_path: str, dataset_path: str, output_dir: str = "evaluation_results"):
        """
        初始化評測系統
        
        Args:
            model_path: 模型路徑
            dataset_path: 數據集路徑 
            output_dir: 輸出目錄
        """
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 設置日誌
        self.setup_logging()
        
        # 載入數據集
        self.dataset = self.load_dataset()
        
        # 載入模型 (如果使用transformers)
        self.model = None
        self.tokenizer = None
        if TRANSFORMERS_AVAILABLE:
            self.load_model()
    
    def setup_logging(self):
        """設置日誌系統"""
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_dataset(self) -> pd.DataFrame:
        """載入中醫問答資料集"""
        try:
            # 使用 utf-8-sig 以正確處理含 BOM 的 CSV 檔
            df = pd.read_csv(self.dataset_path, encoding='utf-8-sig')
            self.logger.info(f"成功載入資料集，共 {len(df)} 筆題目")
            
            # 清理數據
            df = df.dropna(subset=['question', 'answer'])
            df['question'] = df['question'].astype(str)
            df['answer'] = df['answer'].astype(str)
            
            # 添加類別資訊（基於filename）
            if 'filename' in df.columns:
                df['category'] = df['filename'].apply(self.extract_category)
            else:
                df['category'] = 'general'
            
            return df
        except Exception as e:
            self.logger.error(f"載入資料集失敗: {e}")
            raise
    
    def extract_category(self, filename: str) -> str:
        """從檔名提取類別"""
        if isinstance(filename, str):
            # 提取中醫科目
            categories = {
                '內科': 'internal_medicine',
                '外科': 'surgery', 
                '婦科': 'gynecology',
                '兒科': 'pediatrics',
                '針灸': 'acupuncture',
                '方劑': 'prescriptions',
                '本草': 'materia_medica',
                '診斷': 'diagnosis'
            }
            
            for key, value in categories.items():
                if key in filename:
                    return value
            
            return 'general'
        return 'general'
    
    def load_model(self):
        """載入模型（根據實際使用的框架調整）"""
        try:
            if not TRANSFORMERS_AVAILABLE:
                self.logger.warning("Transformers 不可用，將使用模擬模式")
                return
            
            self.logger.info(f"載入模型: {self.model_path}")
            
            # 首先嘗試標準載入方式
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    trust_remote_code=True,
                    use_fast=False  # 使用慢速 tokenizer
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.logger.info("使用 AutoModel 載入成功")
                
            except Exception as e1:
                self.logger.warning(f"AutoModel 載入失敗: {e1}")
                self.logger.info("嘗試使用 LlamaForCausalLM...")
                
                # 嘗試 Llama 模型載入方式
                from transformers import LlamaTokenizer, LlamaForCausalLM
                
                self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path)
                self.model = LlamaForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.logger.info("使用 LlamaForCausalLM 載入成功")
            
            # 設置特殊token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 驗證模型載入
            if self.model is not None and self.tokenizer is not None:
                self.logger.info("✅ 模型和 tokenizer 載入成功")
                # 測試推理
                test_input = self.tokenizer("測試", return_tensors="pt")
                self.logger.info("✅ 模型推理測試通過")
            else:
                raise Exception("模型或 tokenizer 為 None")
            
        except Exception as e:
            self.logger.error(f"載入模型失敗: {e}")
            self.model = None
            self.tokenizer = None
            # 不要默默使用模擬模式，而是提醒用戶
            self.logger.warning("⚠️  將使用隨機答案模式進行測試")
    
    def format_prompt(self, question: str) -> str:
        """格式化提示詞"""
        prompt = f"""你是一位專業的中醫師，請根據以下問題選擇正確答案。請只回答選項字母（A、B、C或D）。

問題：{question}

答案："""
        return prompt
    
    def get_model_answer(self, question: str) -> Tuple[str, float, float]:
        """
        獲取模型答案
        
        Returns:
            (答案, 信心分數, 回應時間)
        """
        start_time = time.time()
        
        try:
            if self.model is None or self.tokenizer is None:
                # 模擬模式（用於測試）
                if not hasattr(self, '_warned_simulation'):
                    self.logger.warning("⚠️  使用模擬模式：模型未載入，將返回隨機答案")
                    self._warned_simulation = True
                time.sleep(0.1)  # 模擬推理時間
                mock_answers = ['A', 'B', 'C', 'D']
                answer = np.random.choice(mock_answers)
                confidence = np.random.uniform(0.5, 1.0)
                response_time = time.time() - start_time
                return answer, confidence, response_time
            
            # 實際模型推理
            prompt = self.format_prompt(question)
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # 只需要生成 A/B/C/D
                    temperature=0.1,
                    do_sample=False,  # 使用 greedy decoding
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解碼答案
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer_text = generated_text[len(prompt):].strip()
            
            # 提取答案選項
            answer = self.extract_answer(answer_text)
            
            # 計算信心分數（簡化版本）
            confidence = self.calculate_confidence(answer_text)
            
            response_time = time.time() - start_time
            
            return answer, confidence, response_time
            
        except Exception as e:
            self.logger.error(f"模型推理失敗: {e}")
            response_time = time.time() - start_time
            return "ERROR", 0.0, response_time
    
    def extract_answer(self, text: str) -> str:
        """從模型輸出中提取答案選項"""
        # 尋找A、B、C、D選項
        pattern = r'\b([ABCD])\b'
        matches = re.findall(pattern, text.upper())
        
        if matches:
            return matches[0]
        
        # 如果沒找到，返回未知
        return "UNKNOWN"
    
    def calculate_confidence(self, text: str) -> float:
        """計算信心分數（簡化版本）"""
        # 這是一個簡化的信心分數計算
        # 實際應用中可以使用模型的logits或其他指標
        
        if "ERROR" in text or "UNKNOWN" in text:
            return 0.0
        
        # 基於回答的確定性給分
        confident_phrases = ["確定", "明確", "顯然", "肯定"]
        uncertain_phrases = ["可能", "或許", "大概", "不確定"]
        
        confidence = 0.7  # 基礎分數
        
        for phrase in confident_phrases:
            if phrase in text:
                confidence += 0.1
        
        for phrase in uncertain_phrases:
            if phrase in text:
                confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def evaluate_sample(self, sample_size: Optional[int] = None, random_seed: int = 42) -> List[EvaluationResult]:
        """評測樣本"""
        np.random.seed(random_seed)
        
        # 選擇評測樣本
        if sample_size is None:
            eval_df = self.dataset.copy()
        else:
            eval_df = self.dataset.sample(n=min(sample_size, len(self.dataset)), random_state=random_seed)
        
        results = []
        total_questions = len(eval_df)
        
        self.logger.info(f"開始評測 {total_questions} 道題目...")
        
        for idx, (_, row) in enumerate(eval_df.iterrows(), 1):
            try:
                question = row['question']
                correct_answer = str(row['answer']).upper()
                category = row.get('category', 'general')
                
                # 獲取模型答案
                model_answer, confidence, response_time = self.get_model_answer(question)
                
                # 判斷是否正確
                is_correct = model_answer.upper() == correct_answer
                
                # 記錄結果
                result = EvaluationResult(
                    question_id=row.get('num', idx),
                    question=question,
                    correct_answer=correct_answer,
                    model_answer=model_answer,
                    is_correct=is_correct,
                    confidence_score=confidence,
                    response_time=response_time,
                    category=category
                )
                
                results.append(result)
                
                # 進度顯示
                if idx % 100 == 0 or idx == total_questions:
                    accuracy = sum(r.is_correct for r in results) / len(results) * 100
                    self.logger.info(f"進度: {idx}/{total_questions} ({idx/total_questions*100:.1f}%), 當前準確率: {accuracy:.2f}%")
                
            except Exception as e:
                self.logger.error(f"評測第 {idx} 題時發生錯誤: {e}")
                continue
        
        self.logger.info(f"評測完成！共評測 {len(results)} 道題目")
        return results
    
    def calculate_metrics(self, results: List[EvaluationResult]) -> Dict:
        """計算評測指標"""
        if not results:
            return {}
        
        # 基本指標
        total_questions = len(results)
        correct_answers = sum(1 for r in results if r.is_correct)
        accuracy = correct_answers / total_questions * 100
        
        # 按類別統計
        category_stats = {}
        for category in set(r.category for r in results):
            category_results = [r for r in results if r.category == category]
            if category_results:
                category_correct = sum(1 for r in category_results if r.is_correct)
                category_stats[category] = {
                    'total': len(category_results),
                    'correct': category_correct,
                    'accuracy': category_correct / len(category_results) * 100,
                    'avg_confidence': np.mean([r.confidence_score for r in category_results]),
                    'avg_response_time': np.mean([r.response_time for r in category_results])
                }
        
        # 信心分數分析
        confidence_scores = [r.confidence_score for r in results]
        response_times = [r.response_time for r in results]
        
        # 按答案正確性分組的信心分數
        correct_confidence = [r.confidence_score for r in results if r.is_correct]
        incorrect_confidence = [r.confidence_score for r in results if not r.is_correct]
        
        metrics = {
            'overall': {
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'accuracy': accuracy,
                'avg_confidence': np.mean(confidence_scores),
                'std_confidence': np.std(confidence_scores),
                'avg_response_time': np.mean(response_times),
                'std_response_time': np.std(response_times)
            },
            'by_category': category_stats,
            'confidence_analysis': {
                'correct_avg_confidence': np.mean(correct_confidence) if correct_confidence else 0,
                'incorrect_avg_confidence': np.mean(incorrect_confidence) if incorrect_confidence else 0,
                'confidence_correct_correlation': np.corrcoef(
                    [r.confidence_score for r in results],
                    [1 if r.is_correct else 0 for r in results]
                )[0, 1] if len(results) > 1 else 0
            }
        }
        
        return metrics
    
    def paired_t_test(self, results1: List[EvaluationResult], results2: List[EvaluationResult]) -> Dict:
        """
        執行成對樣本t檢定
        
        Args:
            results1: 第一組結果（如基礎模型）
            results2: 第二組結果（如微調模型）
        """
        if len(results1) != len(results2):
            raise ValueError("兩組結果的樣本數量必須相同")
        
        # 提取準確率分數 (1 for correct, 0 for incorrect)
        scores1 = [1 if r.is_correct else 0 for r in results1]
        scores2 = [1 if r.is_correct else 0 for r in results2]
        
        # 執行成對樣本t檢定
        t_stat, p_value = ttest_rel(scores2, scores1)  # scores2 - scores1
        
        # 計算效應量 (Cohen's d for paired samples)
        differences = np.array(scores2) - np.array(scores1)
        cohen_d = np.mean(differences) / np.std(differences) if np.std(differences) > 0 else 0
        
        # 計算置信區間
        n = len(differences)
        se = np.std(differences) / np.sqrt(n)
        ci_95 = stats.t.interval(0.95, n-1, loc=np.mean(differences), scale=se)
        
        result = {
            'sample_size': n,
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences),
            't_statistic': t_stat,
            'p_value': p_value,
            'cohen_d': cohen_d,
            'ci_95_lower': ci_95[0],
            'ci_95_upper': ci_95[1],
            'significant': p_value < 0.05,
            'interpretation': self.interpret_t_test_result(t_stat, p_value, cohen_d)
        }
        
        return result
    
    def interpret_t_test_result(self, t_stat: float, p_value: float, cohen_d: float) -> str:
        """解釋t檢定結果"""
        interpretation = []
        
        # 顯著性
        if p_value < 0.001:
            interpretation.append("差異極度顯著 (p < 0.001)")
        elif p_value < 0.01:
            interpretation.append("差異高度顯著 (p < 0.01)")
        elif p_value < 0.05:
            interpretation.append("差異顯著 (p < 0.05)")
        else:
            interpretation.append("差異不顯著 (p ≥ 0.05)")
        
        # 效應量
        abs_d = abs(cohen_d)
        if abs_d < 0.2:
            interpretation.append("效應量很小")
        elif abs_d < 0.5:
            interpretation.append("效應量小")
        elif abs_d < 0.8:
            interpretation.append("效應量中等")
        else:
            interpretation.append("效應量大")
        
        # 方向
        if t_stat > 0:
            interpretation.append("第二組表現優於第一組")
        else:
            interpretation.append("第一組表現優於第二組")
        
        return "; ".join(interpretation)
    
    def generate_visualizations(self, results: List[EvaluationResult], metrics: Dict, output_prefix: str = ""):
        """生成視覺化圖表"""
        # 設置圖表風格
        plt.style.use('seaborn-v0_8')
        
        # 1. 準確率分布圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 按類別的準確率
        if 'by_category' in metrics:
            categories = list(metrics['by_category'].keys())
            accuracies = [metrics['by_category'][cat]['accuracy'] for cat in categories]
            
            axes[0, 0].bar(categories, accuracies, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('各類別準確率')
            axes[0, 0].set_ylabel('準確率 (%)')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 添加數值標籤
            for i, v in enumerate(accuracies):
                axes[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
        
        # 2. 信心分數分布
        confidence_scores = [r.confidence_score for r in results]
        axes[0, 1].hist(confidence_scores, bins=20, color='lightgreen', alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('信心分數分布')
        axes[0, 1].set_xlabel('信心分數')
        axes[0, 1].set_ylabel('頻率')
        
        # 3. 正確vs錯誤答案的信心分數比較
        correct_confidence = [r.confidence_score for r in results if r.is_correct]
        incorrect_confidence = [r.confidence_score for r in results if not r.is_correct]
        
        axes[1, 0].boxplot([correct_confidence, incorrect_confidence], 
                          labels=['正確答案', '錯誤答案'])
        axes[1, 0].set_title('正確與錯誤答案的信心分數比較')
        axes[1, 0].set_ylabel('信心分數')
        
        # 4. 回應時間分布
        response_times = [r.response_time for r in results]
        axes[1, 1].hist(response_times, bins=20, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('回應時間分布')
        axes[1, 1].set_xlabel('回應時間 (秒)')
        axes[1, 1].set_ylabel('頻率')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'{output_prefix}evaluation_charts.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 信心分數vs準確率散點圖
        plt.figure(figsize=(10, 6))
        
        # 分組數據
        correct_x = [r.confidence_score for r in results if r.is_correct]
        correct_y = [1] * len(correct_x)
        incorrect_x = [r.confidence_score for r in results if not r.is_correct]
        incorrect_y = [0] * len(incorrect_x)
        
        plt.scatter(correct_x, correct_y, color='green', alpha=0.6, label='正確', s=50)
        plt.scatter(incorrect_x, incorrect_y, color='red', alpha=0.6, label='錯誤', s=50)
        
        plt.xlabel('信心分數')
        plt.ylabel('答案正確性')
        plt.title('信心分數與答案正確性關係')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / f'{output_prefix}confidence_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results: List[EvaluationResult], metrics: Dict, 
                    t_test_result: Optional[Dict] = None, output_prefix: str = ""):
        """保存評測結果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 保存詳細結果到CSV
        results_df = pd.DataFrame([
            {
                'question_id': r.question_id,
                'question': r.question,
                'correct_answer': r.correct_answer,
                'model_answer': r.model_answer,
                'is_correct': r.is_correct,
                'confidence_score': r.confidence_score,
                'response_time': r.response_time,
                'category': r.category
            }
            for r in results
        ])
        
        csv_file = self.output_dir / f'{output_prefix}detailed_results_{timestamp}.csv'
        results_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        self.logger.info(f"詳細結果已保存至: {csv_file}")
        
        # 2. 保存評測報告
        report = {
            'evaluation_info': {
                'model_path': str(self.model_path),
                'dataset_path': str(self.dataset_path),
                'evaluation_time': timestamp,
                'total_questions_in_dataset': len(self.dataset),
                'evaluated_questions': len(results)
            },
            'metrics': metrics,
            't_test_result': t_test_result
        }
        
        json_file = self.output_dir / f'{output_prefix}evaluation_report_{timestamp}.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        self.logger.info(f"評測報告已保存至: {json_file}")
        
        # 3. 生成人類可讀的總結報告
        self.generate_summary_report(results, metrics, t_test_result, output_prefix, timestamp)
    
    def generate_summary_report(self, results: List[EvaluationResult], metrics: Dict,
                              t_test_result: Optional[Dict], output_prefix: str, timestamp: str):
        """生成總結報告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("中醫問答模型評測報告")
        report_lines.append("=" * 60)
        report_lines.append(f"評測時間: {timestamp}")
        report_lines.append(f"模型路徑: {self.model_path}")
        report_lines.append(f"資料集路徑: {self.dataset_path}")
        report_lines.append("")
        
        # 整體表現
        overall = metrics.get('overall', {})
        report_lines.append("整體表現:")
        report_lines.append(f"  總題數: {overall.get('total_questions', 0)}")
        report_lines.append(f"  正確題數: {overall.get('correct_answers', 0)}")
        report_lines.append(f"  準確率: {overall.get('accuracy', 0):.2f}%")
        report_lines.append(f"  平均信心分數: {overall.get('avg_confidence', 0):.3f}")
        report_lines.append(f"  平均回應時間: {overall.get('avg_response_time', 0):.3f}秒")
        report_lines.append("")
        
        # 各類別表現
        if 'by_category' in metrics:
            report_lines.append("各類別表現:")
            for category, stats in metrics['by_category'].items():
                report_lines.append(f"  {category}:")
                report_lines.append(f"    題數: {stats['total']}")
                report_lines.append(f"    準確率: {stats['accuracy']:.2f}%")
                report_lines.append(f"    平均信心分數: {stats['avg_confidence']:.3f}")
            report_lines.append("")
        
        # 信心分數分析
        if 'confidence_analysis' in metrics:
            conf_analysis = metrics['confidence_analysis']
            report_lines.append("信心分數分析:")
            report_lines.append(f"  正確答案平均信心分數: {conf_analysis['correct_avg_confidence']:.3f}")
            report_lines.append(f"  錯誤答案平均信心分數: {conf_analysis['incorrect_avg_confidence']:.3f}")
            report_lines.append(f"  信心分數與準確率相關係數: {conf_analysis['confidence_correct_correlation']:.3f}")
            report_lines.append("")
        
        # t檢定結果
        if t_test_result:
            report_lines.append("成對樣本t檢定結果:")
            report_lines.append(f"  樣本數: {t_test_result['sample_size']}")
            report_lines.append(f"  平均差異: {t_test_result['mean_difference']:.4f}")
            report_lines.append(f"  t統計量: {t_test_result['t_statistic']:.4f}")
            report_lines.append(f"  p值: {t_test_result['p_value']:.6f}")
            report_lines.append(f"  Cohen's d: {t_test_result['cohen_d']:.4f}")
            report_lines.append(f"  95%置信區間: [{t_test_result['ci_95_lower']:.4f}, {t_test_result['ci_95_upper']:.4f}]")
            report_lines.append(f"  是否顯著: {'是' if t_test_result['significant'] else '否'}")
            report_lines.append(f"  結果解釋: {t_test_result['interpretation']}")
            report_lines.append("")
        
        # 錯誤分析（前10個錯誤案例）
        incorrect_results = [r for r in results if not r.is_correct]
        if incorrect_results:
            report_lines.append("錯誤案例分析（前10例）:")
            for i, result in enumerate(incorrect_results[:10], 1):
                report_lines.append(f"  案例 {i}:")
                report_lines.append(f"    題目ID: {result.question_id}")
                report_lines.append(f"    問題: {result.question[:100]}...")
                report_lines.append(f"    正確答案: {result.correct_answer}")
                report_lines.append(f"    模型答案: {result.model_answer}")
                report_lines.append(f"    信心分數: {result.confidence_score:.3f}")
                report_lines.append(f"    類別: {result.category}")
                report_lines.append("")
        
        # 保存報告
        report_file = self.output_dir / f'{output_prefix}summary_report_{timestamp}.txt'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"總結報告已保存至: {report_file}")
        
        # 打印關鍵指標到控制台
        print("\n" + "="*50)
        print("評測完成！關鍵指標：")
        print("="*50)
        print(f"總準確率: {overall.get('accuracy', 0):.2f}%")
        print(f"評測題數: {overall.get('total_questions', 0)}")
        print(f"平均信心分數: {overall.get('avg_confidence', 0):.3f}")
        if t_test_result:
            print(f"t檢定 p值: {t_test_result['p_value']:.6f}")
            print(f"統計顯著性: {'是' if t_test_result['significant'] else '否'}")
        print("="*50)

def main():
    """主函數 - 命令行介面"""
    parser = argparse.ArgumentParser(description='中醫問答模型評測系統')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型路徑')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='數據集路徑（CSV文件）')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='輸出目錄（默認: evaluation_results）')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='評測樣本數量（默認: 全部）')
    parser.add_argument('--compare_model_path', type=str, default=None,
                       help='比較模型路徑（用於t檢定）')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='隨機種子（默認: 42）')
    parser.add_argument('--generate_charts', action='store_true',
                       help='生成視覺化圖表')
    
    args = parser.parse_args()
    
    print("初始化評測系統...")
    
    # 主要模型評測
    evaluator = TCMEvaluationSystem(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir
    )
    
    print("開始評測主要模型...")
    results = evaluator.evaluate_sample(
        sample_size=args.sample_size,
        random_seed=args.random_seed
    )
    
    print("計算評測指標...")
    metrics = evaluator.calculate_metrics(results)
    
    # 比較模型評測（如果提供）
    t_test_result = None
    if args.compare_model_path:
        print("開始評測比較模型...")
        compare_evaluator = TCMEvaluationSystem(
            model_path=args.compare_model_path,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir
        )
        
        compare_results = compare_evaluator.evaluate_sample(
            sample_size=args.sample_size,
            random_seed=args.random_seed
        )
        
        print("執行成對樣本t檢定...")
        t_test_result = evaluator.paired_t_test(results, compare_results)
        
        # 保存比較模型結果
        compare_metrics = compare_evaluator.calculate_metrics(compare_results)
        compare_evaluator.save_results(compare_results, compare_metrics, output_prefix="compare_")
        
        if args.generate_charts:
            compare_evaluator.generate_visualizations(compare_results, compare_metrics, "compare_")
    
    # 生成視覺化圖表
    if args.generate_charts:
        print("生成視覺化圖表...")
        evaluator.generate_visualizations(results, metrics)
    
    # 保存結果
    print("保存評測結果...")
    evaluator.save_results(results, metrics, t_test_result)
    
    print("評測完成！")

if __name__ == "__main__":
    main()


# 使用範例腳本

# 使用範例:

# 1. 基本評測:
# python tcm_evaluation.py --model_path "C:\Users\user\Desktop\LLaMA-Factory\Breeze_lindan TCPM QA" --dataset_path "tcm_exam_question.csv" --generate_charts

# 2. 樣本評測:
# python tcm_evaluation.py --model_path "path/to/model" --dataset_path "tcm_exam_question.csv" --sample_size 1000 --generate_charts

# 3. 模型比較（t檢定）:
# python tcm_evaluation.py --model_path "path/to/fine_tuned_model" --dataset_path "tcm_exam_question.csv" --compare_model_path "path/to/base_model" --generate_charts

# 4. 自定義輸出目錄:
# python tcm_evaluation.py --model_path "path/to/model" --dataset_path "tcm_exam_question.csv" --output_dir "my_evaluation_results" --generate_charts

# 依賴安裝:
# pip install pandas numpy scipy matplotlib seaborn transformers torch


# 額外的工具函數和類別

class ModelComparator:
    """模型比較工具"""
    
    def __init__(self, evaluation_results_dir: str):
        self.results_dir = Path(evaluation_results_dir)
    
    def load_evaluation_results(self, result_file: str) -> List[EvaluationResult]:
        """載入評測結果"""
        # 評測結果以 utf-8-sig 編碼保存，讀取時需指定相同編碼
        df = pd.read_csv(self.results_dir / result_file, encoding='utf-8-sig')
        
        results = []
        for _, row in df.iterrows():
            result = EvaluationResult(
                question_id=row['question_id'],
                question=row['question'],
                correct_answer=row['correct_answer'],
                model_answer=row['model_answer'],
                is_correct=row['is_correct'],
                confidence_score=row['confidence_score'],
                response_time=row['response_time'],
                category=row['category']
            )
            results.append(result)
        
        return results
    
    def compare_multiple_models(self, result_files: List[str], model_names: List[str]) -> Dict:
        """比較多個模型"""
        if len(result_files) != len(model_names):
            raise ValueError("結果文件數量必須與模型名稱數量相同")
        
        all_results = {}
        for file, name in zip(result_files, model_names):
            all_results[name] = self.load_evaluation_results(file)
        
        comparison = {}
        
        # 整體比較
        for name, results in all_results.items():
            accuracy = sum(1 for r in results if r.is_correct) / len(results) * 100
            avg_confidence = np.mean([r.confidence_score for r in results])
            avg_time = np.mean([r.response_time for r in results])
            
            comparison[name] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'avg_response_time': avg_time,
                'total_questions': len(results)
            }
        
        # 成對t檢定
        pairwise_tests = {}
        models = list(all_results.keys())
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                model1, model2 = models[i], models[j]
                
                scores1 = [1 if r.is_correct else 0 for r in all_results[model1]]
                scores2 = [1 if r.is_correct else 0 for r in all_results[model2]]
                
                t_stat, p_value = ttest_rel(scores2, scores1)
                pairwise_tests[f"{model2}_vs_{model1}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return {
            'model_comparison': comparison,
            'pairwise_tests': pairwise_tests
        }

class BatchEvaluator:
    """批量評測工具"""
    
    def __init__(self, dataset_path: str, output_dir: str = "batch_evaluation"):
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def evaluate_multiple_models(self, model_paths: List[str], model_names: List[str],
                                sample_size: Optional[int] = None):
        """批量評測多個模型"""
        results = {}
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"評測模型: {model_name}")
            
            evaluator = TCMEvaluationSystem(
                model_path=model_path,
                dataset_path=self.dataset_path,
                output_dir=self.output_dir / model_name
            )
            
            model_results = evaluator.evaluate_sample(sample_size=sample_size)
            metrics = evaluator.calculate_metrics(model_results)
            
            evaluator.save_results(model_results, metrics, output_prefix=f"{model_name}_")
            evaluator.generate_visualizations(model_results, metrics, f"{model_name}_")
            
            results[model_name] = {
                'results': model_results,
                'metrics': metrics
            }
        
        # 生成比較報告
        self.generate_comparison_report(results)
        
        return results
    
    def generate_comparison_report(self, results: Dict):
        """生成比較報告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 準備比較數據
        comparison_data = []
        for model_name, data in results.items():
            metrics = data['metrics']
            overall = metrics.get('overall', {})
            
            comparison_data.append({
                'model_name': model_name,
                'accuracy': overall.get('accuracy', 0),
                'avg_confidence': overall.get('avg_confidence', 0),
                'avg_response_time': overall.get('avg_response_time', 0),
                'total_questions': overall.get('total_questions', 0)
            })
        
        # 保存比較表格
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('accuracy', ascending=False)
        
        csv_file = self.output_dir / f'model_comparison_{timestamp}.csv'
        comparison_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        # 生成比較圖表
        plt.figure(figsize=(12, 8))
        
        # 準確率比較
        plt.subplot(2, 2, 1)
        plt.bar(comparison_df['model_name'], comparison_df['accuracy'], color='skyblue', alpha=0.7)
        plt.title('模型準確率比較')
        plt.ylabel('準確率 (%)')
        plt.xticks(rotation=45)
        
        # 信心分數比較
        plt.subplot(2, 2, 2)
        plt.bar(comparison_df['model_name'], comparison_df['avg_confidence'], color='lightgreen', alpha=0.7)
        plt.title('平均信心分數比較')
        plt.ylabel('信心分數')
        plt.xticks(rotation=45)
        
        # 回應時間比較
        plt.subplot(2, 2, 3)
        plt.bar(comparison_df['model_name'], comparison_df['avg_response_time'], color='orange', alpha=0.7)
        plt.title('平均回應時間比較')
        plt.ylabel('回應時間 (秒)')
        plt.xticks(rotation=45)
        
        # 綜合散點圖
        plt.subplot(2, 2, 4)
        plt.scatter(comparison_df['avg_confidence'], comparison_df['accuracy'], 
                   s=comparison_df['avg_response_time']*1000, alpha=0.6)
        
        for i, model in enumerate(comparison_df['model_name']):
            plt.annotate(model, (comparison_df.iloc[i]['avg_confidence'], 
                               comparison_df.iloc[i]['accuracy']))
        
        plt.xlabel('平均信心分數')
        plt.ylabel('準確率 (%)')
        plt.title('綜合表現比較（氣泡大小=回應時間）')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'model_comparison_charts_{timestamp}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比較報告已保存至: {self.output_dir}")

# 配置文件模板
CONFIG_TEMPLATE = """
# 中醫問答模型評測配置文件
# 請根據實際情況修改以下配置

[models]
# 主要評測模型
main_model_path = "C:\\Users\\user\\Desktop\\LLaMA-Factory\\Breeze_lindan TCPM QA"

# 比較模型（可選）
compare_model_path = ""

# 批量評測模型列表
batch_models = [
    {"name": "model1", "path": "path/to/model1"},
    {"name": "model2", "path": "path/to/model2"}
]

[data]
# 資料集路徑
dataset_path = "tcm_exam_question.csv"

# 評測樣本數量（None表示全部）
sample_size = None

# 隨機種子
random_seed = 42

[output]
# 輸出目錄
output_dir = "evaluation_results"

# 是否生成圖表
generate_charts = true

# 是否生成詳細報告
generate_detailed_report = true

[evaluation]
# 是否進行t檢定
perform_t_test = true

# 是否按類別分析
analyze_by_category = true

# 信心分數閾值
confidence_threshold = 0.7
"""

if __name__ == "__main__":
    # 生成配置文件
    config_file = Path("tcm_evaluation_config.ini")
    if not config_file.exists():
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(CONFIG_TEMPLATE)
        print(f"配置文件已生成: {config_file}")
    
    main()