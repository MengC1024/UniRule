#!/usr/bin/env python3
"""
第二步：生成规则

每次运行生成一种方法的结果，追加到同一文件中。

用法:
    python generate.py samples.json --method bgag
    python generate.py samples.json --method baseline
    python generate.py samples.json --method random
    # 新增消融实验方法 (请确保后台连接了对应的 MCP Server)
    python generate.py samples.json --method intent_only
    python generate.py samples.json --method logic_only

支持的方法:
    - bgag: 启用 MCP 检索 (完整版)
    - baseline: 不启用 MCP 检索
    - random: 随机选择训练集中的规则
    - intent_only: 启用 MCP (需配合 Intent-Only Server)
    - logic_only: 启用 MCP (需配合 Logic-Only Server)

输出:
    更新 samples.json 中的 methods 字段
"""

import json
import sqlite3
import argparse
import requests
import sseclient
import time
import re
import random
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


# ============================================================
# 配置
# ============================================================

AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "http://localhost:20001")
DB_DIR = os.getenv("DB_DIR", "../databases")

# Prompt 模板
PROMPT_TEMPLATES = {
    "context": """You are a security detection engineer. Generate a {language} detection rule based on the following detection requirement:

{input_text}

Requirements:
- Output a valid, executable {language} detection rule
- The rule should accurately detect the threat/behavior described above
- Include appropriate field selections and filters for {language}
- Output the rule in a code block

Generate the {language} rule:""",

    "cti": """You are a security detection engineer. Based on the following Cyber Threat Intelligence (CTI) report, generate a {language} detection rule:

{input_text}

Requirements:
- Extract the key indicators and behaviors from the CTI
- Output a valid, executable {language} detection rule
- Focus on detecting the attack techniques and tactics described
- Include appropriate field selections and filters for {language}
- Output the rule in a code block

Generate the {language} rule:""",

    "intent": """You are a security detection engineer. Generate a {language} detection rule to detect the following malicious intent:

{input_text}

Requirements:
- Output a valid, executable {language} detection rule
- The rule should detect the attack/threat intent described above
- Include appropriate field selections and filters for {language}
- Output the rule in a code block

Generate the {language} rule:""",

    "detection_logic": """You are a security detection engineer. Generate a {language} detection rule that implements the following detection logic:

{input_text}

Requirements:
- Output a valid, executable {language} detection rule
- Implement the technical detection mechanism described above
- Use appropriate {language} syntax and functions
- Output the rule in a code block

Generate the {language} rule:"""
}


# ============================================================
# Agent 交互
# ============================================================

def create_session(enable_mcp: bool, max_retries: int = 3) -> Optional[str]:
    """创建 Agent 会话"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{AGENT_BASE_URL}/sessions",
                json={"enable_mcp": enable_mcp},
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("session_id")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
                continue
            print(f"    Session error: {e}")
            return None


def chat_stream(session_id: str, message: str, max_retries: int = 2) -> dict:
    """发送消息并获取流式响应"""
    result = {
        "success": False,
        "content": "",
        "tool_calls": [],
        "error": None
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{AGENT_BASE_URL}/chat/stream",
                json={"session_id": session_id, "message": message},
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            
            client = sseclient.SSEClient(response)
            current_tool_calls = []
            
            for event in client.events():
                if event.data:
                    try:
                        data = json.loads(event.data)
                        event_type = data.get("type")
                        
                        if event_type == "content":
                            result["content"] += data.get("content", "")
                        
                        elif event_type == "tool_call_start":
                            # 记录工具调用开始
                            current_tool_calls.append({
                                "tool_name": data.get("tool_name"),
                                "arguments": data.get("arguments"),
                                "result": None
                            })
                        
                        elif event_type == "tool_call_result":
                            # 记录工具调用结果
                            tool_name = data.get("tool_name")
                            for tc in current_tool_calls:
                                if tc["tool_name"] == tool_name and tc["result"] is None:
                                    tc["result"] = data.get("result")
                                    break
                        
                        elif event_type == "done":
                            result["success"] = True
                        
                        elif event_type == "finish":
                            result["success"] = True
                        
                        elif event_type == "error":
                            result["error"] = data.get("error")
                            
                    except json.JSONDecodeError:
                        continue
            
            result["tool_calls"] = current_tool_calls
            
            if not result["error"]:
                result["success"] = True
            
            return result
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and attempt < max_retries - 1:
                time.sleep(2)
                continue
            result["error"] = str(e)
            return result
        except Exception as e:
            result["error"] = str(e)
            return result
    
    return result


def extract_rule(response: str, language: str) -> str:
    """从响应中提取规则"""
    # 尝试提取代码块
    patterns = [
        rf'```{language}\s*(.*?)```',
        rf'```spl\s*(.*?)```',
        rf'```\s*(.*?)```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0].strip()
    
    # 没有代码块，尝试找规则特征
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if language == "splunk" and ('index=' in line or 'source=' in line or '|' in line):
            return line
        elif language == "snort" and line.startswith(('alert', 'log', 'pass', 'drop')):
            return line
    
    return response.strip()


# ============================================================
# 生成方法
# ============================================================

def generate_bgag(input_text: str, language: str, input_type: str) -> dict:
    """BGAG: 启用 MCP 检索"""
    session_id = create_session(enable_mcp=True)
    if not session_id:
        return {"success": False, "generated_rule": "", "error": "Session failed"}
    
    prompt = PROMPT_TEMPLATES.get(input_type, PROMPT_TEMPLATES["context"])
    prompt = prompt.format(language=language, input_text=input_text)
    
    chat_result = chat_stream(session_id, prompt)
    
    generated_rule = ""
    if chat_result["success"] and chat_result["content"]:
        generated_rule = extract_rule(chat_result["content"], language)
    
    return {
        "success": chat_result["success"],
        "generated_rule": generated_rule,
        "tool_calls": chat_result["tool_calls"],
        "error": chat_result.get("error")
    }


def generate_baseline(input_text: str, language: str, input_type: str) -> dict:
    """Baseline: 不启用 MCP 检索"""
    session_id = create_session(enable_mcp=False)
    if not session_id:
        return {"success": False, "generated_rule": "", "error": "Session failed"}
    
    prompt = PROMPT_TEMPLATES.get(input_type, PROMPT_TEMPLATES["context"])
    prompt = prompt.format(language=language, input_text=input_text)
    
    chat_result = chat_stream(session_id, prompt)
    
    generated_rule = ""
    if chat_result["success"] and chat_result["content"]:
        generated_rule = extract_rule(chat_result["content"], language)
    
    return {
        "success": chat_result["success"],
        "generated_rule": generated_rule,
        "tool_calls": [],
        "error": chat_result.get("error")
    }


def generate_random_rag(input_text: str, language: str, input_type: str, 
                        random_rules: List[dict], k: int = 5) -> dict:
    """Random-RAG: 随机选 k 个规则作为参考，让 LLM 生成"""
    session_id = create_session(enable_mcp=False)
    if not session_id:
        return {"success": False, "generated_rule": "", "error": "Session failed"}
    
    if not random_rules or len(random_rules) < k:
        return {"success": False, "generated_rule": "", "error": "Not enough random rules"}
    
    # 随机选 k 个规则
    selected_rules = random.sample(random_rules, k)
    
    # 构建参考规则文本
    reference_text = "\n\n".join([
        f"Example {i+1}:\nRule: {r.get('rule', '')}"
        for i, r in enumerate(selected_rules)
    ])
    
    # 使用带参考的 prompt
    base_prompt = PROMPT_TEMPLATES.get(input_type, PROMPT_TEMPLATES["context"])
    base_prompt = base_prompt.format(language=language, input_text=input_text)
    
    prompt = f"""Here are some example {language} detection rules for reference:

{reference_text}

---

{base_prompt}"""
    
    chat_result = chat_stream(session_id, prompt)
    
    generated_rule = ""
    if chat_result["success"] and chat_result["content"]:
        generated_rule = extract_rule(chat_result["content"], language)
    
    return {
        "success": chat_result["success"],
        "generated_rule": generated_rule,
        "reference_rules": [r['id'] for r in selected_rules],
        "error": chat_result.get("error")
    }


def get_random_rules(language: str, num: int = 100) -> List[dict]:
    """从训练集获取随机规则"""
    db_path = Path(DB_DIR) / f"{language}_rules.db"
    
    if not db_path.exists():
        return []
    
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, rule, intent, detection_logic
        FROM rules 
        WHERE rule IS NOT NULL AND rule != ''
        ORDER BY RANDOM()
        LIMIT ?
    """, (num,))
    
    rows = cursor.fetchall()
    rules = [dict(row) for row in rows]
    conn.close()
    
    return rules


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


def process_sample(
    sample: dict,
    method: str,
    language: str,
    input_type: str,
    random_rules: List[dict],
    counter: Counter,
    total: int,
    random_rag_k: int = 5
) -> dict:
    """处理单个样本"""
    
    idx = counter.increment()
    sample_id = sample['id']
    input_text = sample.get('input_text', '')
    
    if not input_text:
        print(f"[{idx}/{total}] {sample_id[:8]}... - SKIP (no input)")
        return {"id": sample_id, "success": False, "error": "No input text"}
    
    start_time = time.time()
    
    # === 方法分发逻辑 ===
    if method == "bgag":
        # 完整版 (需后台运行完整 MCP Server)
        result = generate_bgag(input_text, language, input_type)
        
    elif method in ["intent_only", "logic_only"]:
        # 消融版 (需后台运行对应的消融 MCP Server)
        # 对 Client 而言，逻辑也是 "Enable MCP"，所以直接复用 generate_bgag
        result = generate_bgag(input_text, language, input_type)
        
    elif method == "baseline":
        result = generate_baseline(input_text, language, input_type)
        
    elif method == "random_rag":
        result = generate_random_rag(input_text, language, input_type, random_rules, k=random_rag_k)
        
    elif method == "random":
        if random_rules:
            random_rule = random.choice(random_rules)
            result = {
                "success": True,
                "generated_rule": random_rule['rule'],
                "from_id": random_rule['id'],
                "error": None
            }
        else:
            result = {"success": False, "generated_rule": "", "error": "No random rules"}
    else:
        result = {"success": False, "error": f"Unknown method: {method}"}
    
    elapsed = time.time() - start_time
    
    if result.get("success"):
        rule_len = len(result.get("generated_rule", ""))
        print(f"[{idx}/{total}] {sample_id[:8]}... - OK ({rule_len} chars, {elapsed:.1f}s)")
    else:
        print(f"[{idx}/{total}] {sample_id[:8]}... - FAILED: {result.get('error', 'unknown')[:50]}")
    
    return {"id": sample_id, **result}


def run_generation(input_file: str, method: str, workers: int, random_rag_k: int = 5):
    """运行生成"""
    
    # 加载数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    language = data['experiment']['language']
    input_type = data['experiment']['input_type']
    samples = data['samples']
    
    print("=" * 60)
    print(f"Generating: {method.upper()}")
    print(f"Language: {language}, Input Type: {input_type}")
    print(f"Samples: {len(samples)}, Workers: {workers}")
    if method == "random_rag":
        print(f"Random RAG k: {random_rag_k}")
        
    # === 安全提示 ===
    if method in ["intent_only", "logic_only"]:
        print("\n⚠️  [ATTENTION] Ablation Study Mode")
        print(f"Please ensure the Agent is connected to the '{method.upper()}' MCP Server!")
        print("Waiting 3 seconds to confirm...")
        time.sleep(3)
        
    print("=" * 60)
    
    # 检查是否已经有这个方法的结果
    if 'methods' in data and method in data['methods']:
        print(f"\n⚠ Method '{method}' already exists. Overwriting...")
    
    # 准备随机规则（random 和 random_rag 都需要）
    random_rules = []
    if method in ["random", "random_rag"]:
        random_rules = get_random_rules(language, 200)
        print(f"Loaded {len(random_rules)} random rules from training set")
    
    # 并发处理
    all_results = []
    counter = Counter()
    
    print()
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_sample,
                sample, method, language, input_type, random_rules, counter, len(samples), random_rag_k
            ): sample for sample in samples
        }
        
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
            except Exception as e:
                sample = futures[future]
                print(f"\n✗ [{sample['id']}] Exception: {e}")
                all_results.append({
                    "id": sample['id'],
                    "success": False,
                    "error": str(e)
                })
    
    # 转换为 dict（按 sample_id）
    results_dict = {r['id']: r for r in all_results}
    
    # 统计
    success_count = sum(1 for r in all_results if r.get('success'))
    print(f"\n{'=' * 60}")
    print(f"SUCCESS: {success_count}/{len(samples)}")
    print("=" * 60)
    
    # 更新数据
    if 'methods' not in data:
        data['methods'] = {}
    
    data['methods'][method] = {
        "timestamp": datetime.now().isoformat(),
        "success_count": success_count,
        "total_count": len(samples),
        "results": results_dict
    }
    
    # 保存
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Updated {input_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate rules for one method")
    parser.add_argument("input", help="Input JSON file from extract.py")
    parser.add_argument("--method", required=True, 
                        choices=["bgag", "baseline", "random", "random_rag", "intent_only", "logic_only"],
                        help="Generation method")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (default: 1, Agent may not support high concurrency)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of random rules for random_rag method (default: 5)")
    
    args = parser.parse_args()
    
    run_generation(args.input, args.method, args.workers, args.k)


if __name__ == "__main__":
    main()
