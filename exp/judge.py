#!/usr/bin/env python3
"""
评估：LLM-as-Judge (Pairwise Comparison)

成对比较两个方法生成的规则，判断哪个更好。

用法:
    python judge.py samples.json --methods bgag,baseline
    python judge.py samples.json --methods bgag,random --workers 5

输出:
    judged_{input_name}_{timestamp}.json
"""

import json
import argparse
import os
import random as rand_module
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from openai import OpenAI


# ============================================================
# 配置
# ============================================================

JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", os.getenv("TRANSLATE_API_KEY", "sk-1234"))
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://api.deepseek.com/v1")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "deepseek-reasoner")

# Pairwise Comparison Prompt
PAIRWISE_PROMPT = """You are a cybersecurity expert evaluating detection rules.

## Task
Given detection context, compare two candidate rules and determine which one better fulfills the detection context.

## detection context
{context}

## Rule A
```
{rule_a}
```

## Rule B
```
{rule_b}
```

## Evaluation Criteria
1. Does the rule accurately capture the threat/behavior described?
2. Is the detection logic correct and complete?
3. Are the field names, operators, and syntax appropriate?

## Instructions
- Choose the rule that better satisfies the detection context,
- If both rules are equally good (or equally bad), choose "TIE"
- Provide a brief reason for your choice

## Output Format (JSON only, no other text)
{{"winner": "A" or "B" or "TIE", "reason": "one sentence explanation"}}"""


# ============================================================
# Judge 类
# ============================================================

class LLMJudge:
    """LLM Pairwise Judge"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def compare(self, context: str, rule_a: str, rule_b: str) -> Dict:
        """比较两个规则"""
        if not rule_a or not rule_b:
            return {"winner": "TIE", "reason": "Empty rule", "error": True}
        
        # 随机交换顺序，消除位置偏见
        swapped = rand_module.random() < 0.5
        if swapped:
            rule_a, rule_b = rule_b, rule_a
        
        prompt = PAIRWISE_PROMPT.format(
            context=context,
            rule_a=rule_a,
            rule_b=rule_b
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=20000
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            # 如果交换了顺序，反转结果
            if swapped:
                winner = result.get("winner", "TIE")
                if winner == "A":
                    result["winner"] = "B"
                elif winner == "B":
                    result["winner"] = "A"
            
            result["swapped"] = swapped
            return result
            
        except json.JSONDecodeError:
            return {"winner": "TIE", "reason": "Parse error", "error": True}
        except Exception as e:
            return {"winner": "TIE", "reason": str(e), "error": True}


# ============================================================
# 并发处理
# ============================================================

class Counter:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()
    
    def increment(self):
        with self.lock:
            self.value += 1
            return self.value


def judge_sample(
    sample: dict,
    method_a: str,
    method_b: str,
    method_results: Dict[str, dict],
    judge: LLMJudge,
    counter: Counter,
    total: int
) -> dict:
    """评估单个样本"""
    
    idx = counter.increment()
    sample_id = sample['id']
    context = sample.get('input_text', '')
    
    # 获取两个方法的规则
    result_a = method_results.get(method_a, {}).get('results', {}).get(sample_id, {})
    result_b = method_results.get(method_b, {}).get('results', {}).get(sample_id, {})
    
    rule_a = result_a.get('generated_rule', '')
    rule_b = result_b.get('generated_rule', '')
    
    if not rule_a or not rule_b:
        print(f"[{idx}/{total}] {sample_id} - SKIP (missing rule)")
        return {
            "id": sample_id,
            "winner": "TIE",
            "reason": "Missing rule",
            "error": True
        }
    
    judgment = judge.compare(context, rule_a, rule_b)
    
    winner = judgment.get('winner', 'TIE')
    winner_name = method_a.upper() if winner == "A" else (method_b.upper() if winner == "B" else "TIE")
    # print(f"[{idx}/{total}] {sample_id}: {winner_name}")
    
    return {
        "id": sample_id,
        **judgment
    }


def run_judging(input_file: str, method_a: str, method_b: str, output_file: str, workers: int):
    """运行评估"""
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    methods = data.get('methods', {})
    
    comp_name = f"{method_a}_vs_{method_b}"
    
    print("=" * 60)
    print(f"LLM-as-Judge: {comp_name}")
    print(f"Model: {JUDGE_MODEL}")
    print(f"Samples: {len(samples)}, Workers: {workers}")
    print("=" * 60)
    
    # 检查方法是否存在
    if method_a not in methods:
        print(f"Error: Method '{method_a}' not found")
        return
    if method_b not in methods:
        print(f"Error: Method '{method_b}' not found")
        return
    
    if not JUDGE_API_KEY:
        print("Error: JUDGE_API_KEY not set")
        return
    
    judge = LLMJudge(JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL)
    
    # 并发评估
    all_results = []
    counter = Counter()
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_sample,
                sample, method_a, method_b, methods, judge, counter, len(samples)
            ): sample for sample in samples
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                sample = futures[future]
                print(f"\n✗ [{sample['id']}] Error: {e}")
                all_results.append({
                    "id": sample['id'],
                    "winner": "TIE",
                    "error": True
                })
    
    # 统计
    wins = {method_a: 0, method_b: 0, "tie": 0}
    errors = 0
    
    for r in all_results:
        if r.get('error'):
            errors += 1
        winner = r.get('winner', 'TIE')
        if winner == "A":
            wins[method_a] += 1
        elif winner == "B":
            wins[method_b] += 1
        else:
            wins["tie"] += 1
    
    total_valid = wins[method_a] + wins[method_b] + wins["tie"]
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {comp_name}")
    print("=" * 60)
    print(f"  {method_a.upper():12s}: {wins[method_a]:3d}")
    print(f"  {method_b.upper():12s}: {wins[method_b]:3d}")
    print(f"  {'TIE':12s}: {wins['tie']:3d}")
    
    if total_valid > 0:
        print(f"\n  Win Rate:")
        print(f"    {method_a.upper()}: {wins[method_a]/total_valid*100:.1f}%")
        print(f"    {method_b.upper()}: {wins[method_b]/total_valid*100:.1f}%")
    
    if errors > 0:
        print(f"\n  Errors: {errors}")
    
    # 保存结果
    output = {
        "experiment": data['experiment'],
        "comparison": comp_name,
        "judge_model": JUDGE_MODEL,
        "judge_timestamp": datetime.now().isoformat(),
        "summary": {
            comp_name: {
                "wins": wins,
                "total": total_valid,
                "win_rate": {
                    method_a: wins[method_a] / total_valid * 100 if total_valid > 0 else 0,
                    method_b: wins[method_b] / total_valid * 100 if total_valid > 0 else 0
                }
            }
        },
        "samples": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Saved to {output_file}")


def run_judging_with_retry(input_file: str, method_a: str, method_b: str, output_file: str, workers: int):
    """加载已有结果，只重试error样本"""
    
    # 加载输入数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    samples = data['samples']
    methods = data.get('methods', {})
    
    # 加载已有结果
    with open(output_file, 'r', encoding='utf-8') as f:
        existing_output = json.load(f)
    
    existing_results = {r['id']: r for r in existing_output.get('samples', [])}
    
    # 找出需要重试的样本（有error标记的）
    retry_samples = []
    for sample in samples:
        sample_id = sample['id']
        existing = existing_results.get(sample_id)
        if existing and existing.get('error'):
            retry_samples.append(sample)
    
    comp_name = f"{method_a}_vs_{method_b}"
    
    if not retry_samples:
        print("=" * 60)
        print(f"No errors found in {output_file}")
        print("All samples have valid results. No retry needed.")
        print("=" * 60)
        return
    
    print("=" * 60)
    print(f"LLM-as-Judge: {comp_name} (RETRY MODE)")
    print(f"Model: {JUDGE_MODEL}")
    print(f"Total samples: {len(samples)}, Retry samples: {len(retry_samples)}, Workers: {workers}")
    print("=" * 60)
    
    # 检查方法是否存在
    if method_a not in methods:
        print(f"Error: Method '{method_a}' not found")
        return
    if method_b not in methods:
        print(f"Error: Method '{method_b}' not found")
        return
    
    if not JUDGE_API_KEY:
        print("Error: JUDGE_API_KEY not set")
        return
    
    judge = LLMJudge(JUDGE_API_KEY, JUDGE_BASE_URL, JUDGE_MODEL)
    
    # 并发重新评估error样本
    retry_results = []
    counter = Counter()
        
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                judge_sample,
                sample, method_a, method_b, methods, judge, counter, len(retry_samples)
            ): sample for sample in retry_samples
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                retry_results.append(result)
            except Exception as e:
                sample = futures[future]
                print(f"\n✗ [{sample['id']}] Error: {e}")
                retry_results.append({
                    "id": sample['id'],
                    "winner": "TIE",
                    "error": True,
                    "reason": str(e)
                })
    
    # 更新结果：用新结果替换旧的error结果
    retry_results_dict = {r['id']: r for r in retry_results}
    
    all_results = []
    for sample in samples:
        sample_id = sample['id']
        if sample_id in retry_results_dict:
            # 使用新的重试结果
            all_results.append(retry_results_dict[sample_id])
        elif sample_id in existing_results:
            # 保留原有结果
            all_results.append(existing_results[sample_id])
        else:
            # 不应该发生，但以防万一
            all_results.append({
                "id": sample_id,
                "winner": "TIE",
                "reason": "Missing result",
                "error": True
            })
    
    # 重新统计
    wins = {method_a: 0, method_b: 0, "tie": 0}
    errors = 0
    
    for r in all_results:
        if r.get('error'):
            errors += 1
        winner = r.get('winner', 'TIE')
        if winner == "A":
            wins[method_a] += 1
        elif winner == "B":
            wins[method_b] += 1
        else:
            wins["tie"] += 1
    
    total_valid = wins[method_a] + wins[method_b] + wins["tie"]
    
    print(f"\n{'=' * 60}")
    print(f"UPDATED RESULTS: {comp_name}")
    print("=" * 60)
    print(f"  {method_a.upper():12s}: {wins[method_a]:3d}")
    print(f"  {method_b.upper():12s}: {wins[method_b]:3d}")
    print(f"  {'TIE':12s}: {wins['tie']:3d}")
    
    if total_valid > 0:
        print(f"\n  Win Rate:")
        print(f"    {method_a.upper()}: {wins[method_a]/total_valid*100:.1f}%")
        print(f"    {method_b.upper()}: {wins[method_b]/total_valid*100:.1f}%")
    
    if errors > 0:
        print(f"\n  Remaining Errors: {errors} (retry failed)")
    
    # 保存更新后的结果
    output = {
        "experiment": data['experiment'],
        "comparison": comp_name,
        "judge_model": JUDGE_MODEL,
        "judge_timestamp": datetime.now().isoformat(),
        "retry_info": {
            "retried_samples": len(retry_samples),
            "retry_timestamp": datetime.now().isoformat()
        },
        "summary": {
            comp_name: {
                "wins": wins,
                "total": total_valid,
                "win_rate": {
                    method_a: wins[method_a] / total_valid * 100 if total_valid > 0 else 0,
                    method_b: wins[method_b] / total_valid * 100 if total_valid > 0 else 0
                }
            }
        },
        "samples": all_results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="LLM-as-Judge Pairwise Comparison")
    parser.add_argument("input", help="Input JSON file")
    parser.add_argument("--methods", required=True,
                        help="Two methods to compare (e.g., bgag,baseline)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=5)
    
    args = parser.parse_args()
    
    # 解析方法
    methods = args.methods.split(",")
    if len(methods) != 2:
        print("Error: --methods must specify exactly 2 methods (e.g., bgag,baseline)")
        return
    
    method_a, method_b = methods[0].strip(), methods[1].strip()
    
    output_file = args.output
    if not output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(args.input).stem
        output_file = f"judged_{method_a}_vs_{method_b}_{base_name}.json"
        
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Loading and retrying errors...")
        run_judging_with_retry(args.input, method_a, method_b, output_file, args.workers)
    else:
        run_judging(args.input, method_a, method_b, output_file, args.workers)


if __name__ == "__main__":
    main()
