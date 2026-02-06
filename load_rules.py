#!/usr/bin/env python3
"""
规则数据加载工具
将各种格式的规则文件导入统一数据库
支持训练集/测试集分割
支持双语义空间构建：
  - intent: 威胁意图 φ_threat(r)
  - detection_logic: 检测逻辑 φ_detection(r)
支持CTI报告生成（用于评估）
"""

import json
import sqlite3
import os
import random
import argparse
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

# OpenAI库用于LLM调用
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DB_DIR = os.getenv("DB_DIR", "./databases")

# 数据库写入锁
db_lock = Lock()


def get_db_connection(language: str, split: str = "train") -> sqlite3.Connection:
    """获取数据库连接
    
    Args:
        language: 规则语言
        split: train 或 test
    """
    db_dir = Path(DB_DIR)
    db_dir.mkdir(parents=True, exist_ok=True)
    
    if split == "train":
        db_path = db_dir / f"{language}_rules.db"
    else:
        db_path = db_dir / f"{language}_rules_test.db"
    
    conn = sqlite3.connect(str(db_path))
    return conn


def init_unified_schema(cursor):
    """初始化统一的规则表结构
    
    所有语言使用相同的核心schema：
    - id: 规则唯一标识
    - context: 检测上下文（原始描述）
    - rule: 规则内容
    - intent: 威胁意图 φ_threat(r)
    - detection_logic: 检测逻辑 φ_detection(r)
    - cti: CTI报告（LLM生成，用于评估）
    - metadata: 其他元数据（JSON）
    """
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rules (
            id TEXT PRIMARY KEY,
            context TEXT,
            rule TEXT,
            intent TEXT,
            detection_logic TEXT,
            cti TEXT,
            metadata TEXT
        )
    """)
    
    # 全文搜索
    cursor.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS rules_fts USING fts5(
            id UNINDEXED, context, rule, intent, detection_logic
        )
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_context ON rules(context)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_intent ON rules(intent)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detection_logic ON rules(detection_logic)")


def migrate_add_intent_column(cursor):
    """迁移：为旧数据库添加intent列"""
    try:
        cursor.execute("ALTER TABLE rules ADD COLUMN intent TEXT")
        print("  ✓ Added intent column to rules table")
    except sqlite3.OperationalError:
        # 列已存在
        pass


def migrate_add_detection_logic_column(cursor):
    """迁移：为旧数据库添加detection_logic列"""
    try:
        cursor.execute("ALTER TABLE rules ADD COLUMN detection_logic TEXT")
        print("  ✓ Added detection_logic column to rules table")
    except sqlite3.OperationalError:
        # 列已存在
        pass


def migrate_add_cti_column(cursor):
    """迁移：为旧数据库添加cti列"""
    try:
        cursor.execute("ALTER TABLE rules ADD COLUMN cti TEXT")
        print("  ✓ Added cti column to rules table")
    except sqlite3.OperationalError:
        # 列已存在
        pass


# ============================================================
# LLM Client
# ============================================================

def create_llm_client(base_url: Optional[str] = None, 
                      api_key: Optional[str] = None) -> Optional['OpenAI']:
    """创建LLM客户端
    
    Args:
        base_url: API base URL (默认从环境变量 OPENAI_BASE_URL)
        api_key: API key (默认从环境变量 OPENAI_API_KEY)
    """
    if OpenAI is None:
        print("Warning: openai library not installed. Run: pip install openai")
        return None
    
    return OpenAI(
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        api_key=api_key or os.getenv("OPENAI_API_KEY", "not-needed")
    )


# ============================================================
# Intent Generation（威胁意图 φ_threat(r)）
# ============================================================

INTENT_GENERATION_PROMPT = """Given this {language} detection rule:

{rule}

Describe the malicious intent this rule aims to detect. Focus on the threat semantics: what attack or malicious activity, and what is the adversary's goal.

Output plain text only, no markdown formatting."""

def generate_intent_single(client: 'OpenAI', 
                           rule: str, 
                           language: str,
                           model: str = "gpt-4o-mini") -> Optional[str]:
    """为单条规则生成威胁意图描述 φ_threat(r)
    
    Args:
        client: OpenAI客户端
        rule: 规则内容
        language: 规则语言
        model: 模型名称
    
    Returns:
        检测意图文本，失败返回None
    """
    prompt = INTENT_GENERATION_PROMPT.format(language=language, rule=rule)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error generating intent: {e}")
        return None


def generate_intent_batch(language: str,
                          split: str = "train",
                          base_url: Optional[str] = None,
                          api_key: Optional[str] = None,
                          model: str = "gpt-4o-mini",
                          max_workers: int = 10,
                          limit: Optional[int] = None,
                          force: bool = False):
    """批量生成检测意图
    
    Args:
        language: 规则语言
        split: train 或 test
        base_url: API base URL
        api_key: API key
        model: 模型名称
        max_workers: 并发线程数
        limit: 限制处理数量（用于测试）
        force: 强制重新生成（忽略已有intent）
    """
    client = create_llm_client(base_url, api_key)
    if client is None:
        return
    
    conn = get_db_connection(language, split)
    cursor = conn.cursor()
    
    # 确保intent列存在
    migrate_add_intent_column(cursor)
    conn.commit()
    
    # 获取需要处理的规则
    if force:
        cursor.execute("SELECT id, rule FROM rules")
    else:
        cursor.execute("SELECT id, rule FROM rules WHERE intent IS NULL OR intent = ''")
    
    rules = cursor.fetchall()
    
    if limit:
        rules = rules[:limit]
    
    if not rules:
        print(f"No rules to process for {language} ({split})")
        conn.close()
        return
    
    print(f"Generating intents for {len(rules)} {language} rules ({split})...")
    print(f"  Model: {model}, Workers: {max_workers}")
    
    # 进度统计
    total = len(rules)
    completed = 0
    failed = 0
    start_time = time.time()
    
    def process_rule(rule_data):
        """处理单条规则"""
        rule_id, rule_text = rule_data
        intent = generate_intent_single(client, rule_text, language, model)
        return rule_id, intent
    
    def update_db(rule_id: str, intent: str):
        """更新数据库"""
        nonlocal completed, failed
        with db_lock:
            try:
                update_conn = get_db_connection(language, split)
                update_cursor = update_conn.cursor()
                update_cursor.execute(
                    "UPDATE rules SET intent = ? WHERE id = ?",
                    (intent, rule_id)
                )
                update_conn.commit()
                update_conn.close()
                if intent:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    DB error for {rule_id}: {e}")
                failed += 1
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_rule, rule): rule[0] for rule in rules}
        
        for future in as_completed(futures):
            rule_id = futures[future]
            try:
                rid, intent = future.result()
                update_db(rid, intent)
                
                # 进度显示
                done = completed + failed
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {done}/{total} ({rate:.1f} rules/sec)")
                    
            except Exception as e:
                print(f"    Error processing {rule_id}: {e}")
                failed += 1
    
    elapsed = time.time() - start_time
    conn.close()
    
    print(f"✓ Completed: {completed} succeeded, {failed} failed in {elapsed:.1f}s")


# ============================================================
# Detection Logic Generation（检测逻辑 φ_detection(r)）
# ============================================================

DETECTION_LOGIC_GENERATION_PROMPT = """Given this {language} detection rule:

{rule}

Describe the detection logic this rule implements. Focus on the technical mechanism: what system events, data patterns, or conditions does it match.

Output plain text only, no markdown formatting."""


def generate_detection_logic_single(client: 'OpenAI', 
                                    rule: str, 
                                    language: str,
                                    model: str = "gpt-4o-mini") -> Optional[str]:
    """为单条规则生成检测逻辑描述 φ_detection(r)
    
    Args:
        client: OpenAI客户端
        rule: 规则内容
        language: 规则语言
        model: 模型名称
    
    Returns:
        检测逻辑描述文本，失败返回None
    """
    prompt = DETECTION_LOGIC_GENERATION_PROMPT.format(language=language, rule=rule)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error generating detection_logic: {e}")
        return None


def generate_detection_logic_batch(language: str,
                                   split: str = "train",
                                   base_url: Optional[str] = None,
                                   api_key: Optional[str] = None,
                                   model: str = "gpt-4o-mini",
                                   max_workers: int = 10,
                                   limit: Optional[int] = None,
                                   force: bool = False):
    """批量生成检测逻辑描述
    
    Args:
        language: 规则语言
        split: train 或 test
        base_url: API base URL
        api_key: API key
        model: 模型名称
        max_workers: 并发线程数
        limit: 限制处理数量（用于测试）
        force: 强制重新生成（忽略已有detection_logic）
    """
    client = create_llm_client(base_url, api_key)
    if client is None:
        return
    
    conn = get_db_connection(language, split)
    cursor = conn.cursor()
    
    # 确保detection_logic列存在
    migrate_add_detection_logic_column(cursor)
    conn.commit()
    
    # 获取需要处理的规则
    if force:
        cursor.execute("SELECT id, rule FROM rules")
    else:
        cursor.execute("SELECT id, rule FROM rules WHERE detection_logic IS NULL OR detection_logic = ''")
    
    rules = cursor.fetchall()
    
    if limit:
        rules = rules[:limit]
    
    if not rules:
        print(f"No rules to process for {language} ({split})")
        conn.close()
        return
    
    print(f"Generating detection_logic for {len(rules)} {language} rules ({split})...")
    print(f"  Model: {model}, Workers: {max_workers}")
    
    # 进度统计
    total = len(rules)
    completed = 0
    failed = 0
    start_time = time.time()
    
    def process_rule(rule_data):
        """处理单条规则"""
        rule_id, rule_text = rule_data
        detection_logic = generate_detection_logic_single(client, rule_text, language, model)
        return rule_id, detection_logic
    
    def update_db(rule_id: str, detection_logic: str):
        """更新数据库"""
        nonlocal completed, failed
        with db_lock:
            try:
                update_conn = get_db_connection(language, split)
                update_cursor = update_conn.cursor()
                update_cursor.execute(
                    "UPDATE rules SET detection_logic = ? WHERE id = ?",
                    (detection_logic, rule_id)
                )
                update_conn.commit()
                update_conn.close()
                if detection_logic:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    DB error for {rule_id}: {e}")
                failed += 1
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_rule, rule): rule[0] for rule in rules}
        
        for future in as_completed(futures):
            rule_id = futures[future]
            try:
                rid, detection_logic = future.result()
                update_db(rid, detection_logic)
                
                # 进度显示
                done = completed + failed
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {done}/{total} ({rate:.1f} rules/sec)")
                    
            except Exception as e:
                print(f"    Error processing {rule_id}: {e}")
                failed += 1
    
    elapsed = time.time() - start_time
    conn.close()
    
    print(f"✓ Completed: {completed} succeeded, {failed} failed in {elapsed:.1f}s")


# ============================================================
# CTI Report Generation（CTI报告生成，用于评估）
# ============================================================

CTI_GENERATION_PROMPT = """You are a threat intelligence analyst. Given a detection rule and its description, write a brief CTI summary about the underlying threat.

Rule Description: {description}
Rule Content: {rule}
{metadata_section}

In 2-4 sentences, explain: what this threat is, why attackers use it, and its potential impact.

Do not include detection-specific details like log fields, query syntax, or event IDs."""


def generate_cti_single(client: 'OpenAI',
                        description: str,
                        rule: str,
                        language: str,
                        metadata: Optional[Dict] = None,
                        model: str = "gpt-4o-mini") -> Optional[str]:
    """为单条规则生成CTI报告
    
    Args:
        client: OpenAI客户端
        description: 规则描述
        rule: 规则内容
        language: 规则语言
        metadata: 规则元数据
        model: 模型名称
    
    Returns:
        CTI报告文本，失败返回None
    """
    # 构建metadata部分
    metadata_section = ""
    if metadata:
        relevant_fields = []
        
        # 提取有用的元数据字段
        if metadata.get('TTP'):
            relevant_fields.append(f"MITRE ATT&CK: {', '.join(metadata['TTP'])}")
        if metadata.get('tags', {}).get('mitre_attack_id'):
            relevant_fields.append(f"MITRE ATT&CK: {', '.join(metadata['tags']['mitre_attack_id'])}")
        if metadata.get('data_source'):
            sources = metadata['data_source']
            if isinstance(sources, list):
                relevant_fields.append(f"Data Sources: {', '.join(sources)}")
            else:
                relevant_fields.append(f"Data Sources: {sources}")
        if metadata.get('references'):
            refs = metadata['references'][:3]  # 最多3个引用
            relevant_fields.append(f"References: {', '.join(refs)}")
        
        if relevant_fields:
            metadata_section = "Additional Context:\n" + "\n".join(f"- {f}" for f in relevant_fields)
    
    prompt = CTI_GENERATION_PROMPT.format(
        description=description,
        rule=rule,
        metadata_section=metadata_section
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error generating CTI: {e}")
        return None


def generate_cti_batch(language: str,
                       split: str = "test",
                       base_url: Optional[str] = None,
                       api_key: Optional[str] = None,
                       model: str = "gpt-4o-mini",
                       max_workers: int = 10,
                       limit: Optional[int] = None,
                       force: bool = False):
    """批量生成CTI报告
    
    Args:
        language: 规则语言
        split: train 或 test（默认test，因为CTI主要用于评估）
        base_url: API base URL
        api_key: API key
        model: 模型名称
        max_workers: 并发线程数
        limit: 限制处理数量（用于测试）
        force: 强制重新生成（忽略已有cti）
    """
    client = create_llm_client(base_url, api_key)
    if client is None:
        return
    
    conn = get_db_connection(language, split)
    cursor = conn.cursor()
    
    # 确保cti列存在
    migrate_add_cti_column(cursor)
    conn.commit()
    
    # 获取需要处理的规则
    if force:
        cursor.execute("SELECT id, context, rule, metadata FROM rules")
    else:
        cursor.execute("SELECT id, context, rule, metadata FROM rules WHERE cti IS NULL OR cti = ''")
    
    rules = cursor.fetchall()
    
    if limit:
        rules = rules[:limit]
    
    if not rules:
        print(f"No rules to process for {language} ({split})")
        conn.close()
        return
    
    print(f"Generating CTI reports for {len(rules)} {language} rules ({split})...")
    print(f"  Model: {model}, Workers: {max_workers}")
    
    # 进度统计
    total = len(rules)
    completed = 0
    failed = 0
    start_time = time.time()
    
    def process_rule(rule_data):
        """处理单条规则"""
        rule_id, context, rule_text, metadata_str = rule_data
        metadata = json.loads(metadata_str) if metadata_str else None
        cti = generate_cti_single(client, context, rule_text, language, metadata, model)
        return rule_id, cti
    
    def update_db(rule_id: str, cti: str):
        """更新数据库"""
        nonlocal completed, failed
        with db_lock:
            try:
                update_conn = get_db_connection(language, split)
                update_cursor = update_conn.cursor()
                update_cursor.execute(
                    "UPDATE rules SET cti = ? WHERE id = ?",
                    (cti, rule_id)
                )
                update_conn.commit()
                update_conn.close()
                if cti:
                    completed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"    DB error for {rule_id}: {e}")
                failed += 1
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_rule, rule): rule[0] for rule in rules}
        
        for future in as_completed(futures):
            rule_id = futures[future]
            try:
                rid, cti = future.result()
                update_db(rid, cti)
                
                # 进度显示
                done = completed + failed
                if done % 10 == 0 or done == total:
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    print(f"  Progress: {done}/{total} ({rate:.1f} rules/sec)")
                    
            except Exception as e:
                print(f"    Error processing {rule_id}: {e}")
                failed += 1
    
    elapsed = time.time() - start_time
    conn.close()
    
    print(f"✓ Completed: {completed} succeeded, {failed} failed in {elapsed:.1f}s")


# ============================================================
# 规则加载函数
# ============================================================

def load_snort_rules(json_path: str, test_ratio: float = 0.2, seed: int = 42):
    """加载Snort规则"""
    print(f"Loading Snort rules from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules = data if isinstance(data, list) else data.get('parsed_rules', data.get('rules', []))
    
    # 随机分割
    random.seed(seed)
    random.shuffle(rules)
    split_idx = int(len(rules) * (1 - test_ratio))
    train_rules = rules[:split_idx]
    test_rules = rules[split_idx:]
    
    for split, split_rules in [("train", train_rules), ("test", test_rules)]:
        conn = get_db_connection("snort", split)
        cursor = conn.cursor()
        init_unified_schema(cursor)
        migrate_add_intent_column(cursor)
        migrate_add_detection_logic_column(cursor)
        migrate_add_cti_column(cursor)
        
        cursor.execute("DELETE FROM rules")
        try:
            cursor.execute("DELETE FROM rules_fts")
        except:
            pass
        
        for rule in split_rules:
            rule_id = str(rule.get('sid', ''))
            if not rule_id:
                continue
            
            context = rule.get('msg', '')
            
            if 'raw_rule' in rule:
                rule_text = rule['raw_rule']
            else:
                contents = rule.get('content', [])
                content_str = ' '.join([f'content:"{c}";' for c in contents]) if contents else ''
                rule_text = f"alert (msg:\"{context}\"; {content_str} sid:{rule_id};)"
            
            metadata = {
                'rev': rule.get('rev'),
                'content': rule.get('content', []),
                'flow': rule.get('flow'),
                'metadata': rule.get('metadata')
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO rules (id, context, rule, intent, detection_logic, cti, metadata)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """, (rule_id, context, rule_text, json.dumps(metadata, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        print(f"  ✓ {split}: {len(split_rules)} rules")
    
    print(f"✓ Loaded Snort rules (train: {len(train_rules)}, test: {len(test_rules)})")


def load_splunk_rules(json_path: str, test_ratio: float = 0.2, seed: int = 42):
    """加载Splunk规则"""
    print(f"Loading Splunk rules from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules = data if isinstance(data, list) else data.get('rules', [])
    rules = [r for r in rules if r.get('rule')]
    
    random.seed(seed)
    random.shuffle(rules)
    split_idx = int(len(rules) * (1 - test_ratio))
    train_rules = rules[:split_idx]
    test_rules = rules[split_idx:]
    
    for split, split_rules in [("train", train_rules), ("test", test_rules)]:
        conn = get_db_connection("splunk", split)
        cursor = conn.cursor()
        init_unified_schema(cursor)
        migrate_add_intent_column(cursor)
        migrate_add_detection_logic_column(cursor)
        migrate_add_cti_column(cursor)
        
        cursor.execute("DELETE FROM rules")
        try:
            cursor.execute("DELETE FROM rules_fts")
        except:
            pass
        
        for rule in split_rules:
            rule_id = rule.get('id', rule.get('name', ''))
            if not rule_id:
                continue
            
            context = rule.get('description', '')
            rule_text = rule.get('rule', '')
            
            metadata = {
                'name': rule.get('name'),
                'type': rule.get('type'),
                'status': rule.get('status'),
                'TTP': rule.get('TTP', []),
                'tags': rule.get('tags', {}),
                'data_source': rule.get('data_source', []),
                'references': rule.get('references', [])
            }
            
            cursor.execute("""
                INSERT OR REPLACE INTO rules (id, context, rule, intent, detection_logic, cti, metadata)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """, (rule_id, context, rule_text, json.dumps(metadata, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        print(f"  ✓ {split}: {len(split_rules)} rules")
    
    print(f"✓ Loaded Splunk rules (train: {len(train_rules)}, test: {len(test_rules)})")


def load_elastic_rules(json_path: str, test_ratio: float = 0.2, seed: int = 42):
    """加载Elastic Detection Rules"""
    print(f"Loading Elastic rules from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules = data if isinstance(data, list) else data.get('rules', [])
    rules = [r for r in rules if r.get('rule')]
    
    random.seed(seed)
    random.shuffle(rules)
    split_idx = int(len(rules) * (1 - test_ratio))
    train_rules = rules[:split_idx]
    test_rules = rules[split_idx:]
    
    for split, split_rules in [("train", train_rules), ("test", test_rules)]:
        conn = get_db_connection("elastic", split)
        cursor = conn.cursor()
        init_unified_schema(cursor)
        migrate_add_intent_column(cursor)
        migrate_add_detection_logic_column(cursor)
        migrate_add_cti_column(cursor)
        
        cursor.execute("DELETE FROM rules")
        try:
            cursor.execute("DELETE FROM rules_fts")
        except:
            pass
        
        for idx, rule in enumerate(split_rules):
            rule_id = rule.get('id', f"es_{idx}")
            context = rule.get('description', '')
            rule_text = rule.get('rule', '')
            
            metadata = {k: v for k, v in rule.items() if k not in ['rule', 'description']}
            
            cursor.execute("""
                INSERT OR REPLACE INTO rules (id, context, rule, intent, detection_logic, cti, metadata)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """, (rule_id, context, rule_text, json.dumps(metadata, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        print(f"  ✓ {split}: {len(split_rules)} rules")
    
    print(f"✓ Loaded Elastic rules (train: {len(train_rules)}, test: {len(test_rules)})")


def load_generic_rules(json_path: str, language: str, 
                       context_field: str = "description",
                       rule_field: str = "rule",
                       id_field: str = "id",
                       test_ratio: float = 0.2, 
                       seed: int = 42):
    """通用规则加载器"""
    print(f"Loading {language} rules from {json_path}...")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    rules = data if isinstance(data, list) else data.get('rules', [])
    rules = [r for r in rules if r.get(rule_field)]
    
    random.seed(seed)
    random.shuffle(rules)
    split_idx = int(len(rules) * (1 - test_ratio))
    train_rules = rules[:split_idx]
    test_rules = rules[split_idx:]
    
    for split, split_rules in [("train", train_rules), ("test", test_rules)]:
        conn = get_db_connection(language, split)
        cursor = conn.cursor()
        init_unified_schema(cursor)
        migrate_add_intent_column(cursor)
        migrate_add_detection_logic_column(cursor)
        migrate_add_cti_column(cursor)
        
        cursor.execute("DELETE FROM rules")
        try:
            cursor.execute("DELETE FROM rules_fts")
        except:
            pass
        
        for idx, rule in enumerate(split_rules):
            rule_id = str(rule.get(id_field, f"{language}_{idx}"))
            context = rule.get(context_field, '')
            rule_text = rule.get(rule_field, '')
            
            metadata = {k: v for k, v in rule.items() 
                       if k not in [rule_field, context_field, id_field]}
            
            cursor.execute("""
                INSERT OR REPLACE INTO rules (id, context, rule, intent, detection_logic, cti, metadata)
                VALUES (?, ?, ?, NULL, NULL, NULL, ?)
            """, (rule_id, context, rule_text, json.dumps(metadata, ensure_ascii=False)))
        
        conn.commit()
        conn.close()
        print(f"  ✓ {split}: {len(split_rules)} rules")
    
    print(f"✓ Loaded {language} rules (train: {len(train_rules)}, test: {len(test_rules)})")


def show_stats():
    """显示所有数据库统计"""
    db_dir = Path(DB_DIR)
    if not db_dir.exists():
        print("No databases found.")
        return
    
    print("\n" + "=" * 60)
    print("Database Statistics")
    print("=" * 60)
    
    for db_file in sorted(db_dir.glob("*.db")):
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM rules")
            count = cursor.fetchone()[0]
            
            # 统计 intent 覆盖率
            try:
                cursor.execute("SELECT COUNT(*) FROM rules WHERE intent IS NOT NULL AND intent != ''")
                intent_count = cursor.fetchone()[0]
                intent_pct = (intent_count / count * 100) if count > 0 else 0
            except:
                intent_count = 0
                intent_pct = 0
            
            # 统计 detection_logic 覆盖率
            try:
                cursor.execute("SELECT COUNT(*) FROM rules WHERE detection_logic IS NOT NULL AND detection_logic != ''")
                detection_logic_count = cursor.fetchone()[0]
                detection_logic_pct = (detection_logic_count / count * 100) if count > 0 else 0
            except:
                detection_logic_count = 0
                detection_logic_pct = 0
            
            # 统计 cti 覆盖率
            try:
                cursor.execute("SELECT COUNT(*) FROM rules WHERE cti IS NOT NULL AND cti != ''")
                cti_count = cursor.fetchone()[0]
                cti_pct = (cti_count / count * 100) if count > 0 else 0
            except:
                cti_count = 0
                cti_pct = 0
            
            cursor.execute("SELECT id, substr(context, 1, 50) FROM rules LIMIT 1")
            sample = cursor.fetchone()
            
            conn.close()
            
            print(f"\n{db_file.name}:")
            print(f"  Rules: {count}")
            print(f"  Intents: {intent_count} ({intent_pct:.1f}%)")
            print(f"  Detection Logic: {detection_logic_count} ({detection_logic_pct:.1f}%)")
            print(f"  CTI Reports: {cti_count} ({cti_pct:.1f}%)")
            if sample:
                print(f"  Sample: [{sample[0]}] {sample[1]}...")
        except Exception as e:
            print(f"\n{db_file.name}: Error - {e}")
    
    print("\n" + "=" * 60)


def show_sample(language: str, split: str = "test", field: str = "intent", n: int = 1):
    """显示样本数据
    
    Args:
        language: 规则语言
        split: train 或 test
        field: 用于过滤的字段（只显示该字段非空的规则）
        n: 显示数量
    """
    conn = get_db_connection(language, split)
    cursor = conn.cursor()
    
    valid_fields = ['context', 'rule', 'intent', 'detection_logic', 'cti']
    if field not in valid_fields:
        print(f"Invalid field. Choose from: {valid_fields}")
        return
    
    cursor.execute(f"""
        SELECT id, context, rule, intent, detection_logic, cti FROM rules 
        WHERE {field} IS NOT NULL AND {field} != ''
        LIMIT ?
    """, (n,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print(f"No samples found with {field} for {language} ({split})")
        return
    
    print(f"\n{'=' * 70}")
    print(f"Samples: {language} ({split}) - filtered by {field}")
    print('=' * 70)
    
    for row in rows:
        rule_id, context, rule, intent, detection_logic, cti = row
        print(f"\n[ID: {rule_id}]")
        
        print(f"\n--- Context ---\n{context if context else '(empty)'}")
        print(f"\n--- Rule ---\n{rule if rule else '(empty)'}")
        print(f"\n--- Intent (φ_threat) ---\n{intent if intent else '(empty)'}")
        print(f"\n--- Detection Logic (φ_detection) ---\n{detection_logic if detection_logic else '(empty)'}")
        print(f"\n--- CTI ---\n{cti if cti else '(empty)'}")
        
        print("\n" + "-" * 70)


def main():
    parser = argparse.ArgumentParser(description="Load detection rules into database")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # load命令
    load_parser = subparsers.add_parser("load", help="Load rules from file")
    load_parser.add_argument("language", choices=["snort", "splunk", "elastic", "generic"],
                            help="Rule language")
    load_parser.add_argument("path", help="Path to rules JSON file")
    load_parser.add_argument("--test-ratio", type=float, default=0.2,
                            help="Test set ratio (default: 0.2)")
    load_parser.add_argument("--seed", type=int, default=42,
                            help="Random seed (default: 42)")
    load_parser.add_argument("--context-field", default="description",
                            help="Context field name for generic loader")
    load_parser.add_argument("--rule-field", default="rule",
                            help="Rule field name for generic loader")
    load_parser.add_argument("--id-field", default="id",
                            help="ID field name for generic loader")
    load_parser.add_argument("--lang-name", default=None,
                            help="Language name for generic loader")
    
    # generate-intent命令
    intent_parser = subparsers.add_parser("generate-intent", 
                                          help="Generate threat intent (φ_threat)")
    intent_parser.add_argument("language", help="Rule language (e.g., splunk, snort)")
    intent_parser.add_argument("--split", choices=["train", "test"], default="train",
                              help="Dataset split (default: train)")
    intent_parser.add_argument("--base-url", default=None,
                              help="OpenAI API base URL (or set OPENAI_BASE_URL)")
    intent_parser.add_argument("--api-key", default=None,
                              help="OpenAI API key (or set OPENAI_API_KEY)")
    intent_parser.add_argument("--model", default="gpt-4o-mini",
                              help="Model name (default: gpt-4o-mini)")
    intent_parser.add_argument("--workers", type=int, default=10,
                              help="Number of concurrent workers (default: 10)")
    intent_parser.add_argument("--limit", type=int, default=None,
                              help="Limit number of rules to process (for testing)")
    intent_parser.add_argument("--force", action="store_true",
                              help="Force regenerate even if intent exists")
    
    # generate-detection-logic命令
    detection_logic_parser = subparsers.add_parser("generate-detection-logic", 
                                                   help="Generate detection logic (φ_detection)")
    detection_logic_parser.add_argument("language", help="Rule language (e.g., splunk, snort)")
    detection_logic_parser.add_argument("--split", choices=["train", "test"], default="train",
                                        help="Dataset split (default: train)")
    detection_logic_parser.add_argument("--base-url", default=None,
                                        help="OpenAI API base URL (or set OPENAI_BASE_URL)")
    detection_logic_parser.add_argument("--api-key", default=None,
                                        help="OpenAI API key (or set OPENAI_API_KEY)")
    detection_logic_parser.add_argument("--model", default="gpt-4o-mini",
                                        help="Model name (default: gpt-4o-mini)")
    detection_logic_parser.add_argument("--workers", type=int, default=10,
                                        help="Number of concurrent workers (default: 10)")
    detection_logic_parser.add_argument("--limit", type=int, default=None,
                                        help="Limit number of rules to process (for testing)")
    detection_logic_parser.add_argument("--force", action="store_true",
                                        help="Force regenerate even if detection_logic exists")
    
    # generate-cti命令
    cti_parser = subparsers.add_parser("generate-cti", 
                                       help="Generate CTI reports (for evaluation)")
    cti_parser.add_argument("language", help="Rule language (e.g., splunk, snort)")
    cti_parser.add_argument("--split", choices=["train", "test"], default="test",
                           help="Dataset split (default: test)")
    cti_parser.add_argument("--base-url", default=None,
                           help="OpenAI API base URL (or set OPENAI_BASE_URL)")
    cti_parser.add_argument("--api-key", default=None,
                           help="OpenAI API key (or set OPENAI_API_KEY)")
    cti_parser.add_argument("--model", default="gpt-4o-mini",
                           help="Model name (default: gpt-4o-mini)")
    cti_parser.add_argument("--workers", type=int, default=10,
                           help="Number of concurrent workers (default: 10)")
    cti_parser.add_argument("--limit", type=int, default=None,
                           help="Limit number of rules to process (for testing)")
    cti_parser.add_argument("--force", action="store_true",
                           help="Force regenerate even if CTI exists")
    
    # stats命令
    subparsers.add_parser("stats", help="Show database statistics")
    
    # sample命令
    sample_parser = subparsers.add_parser("sample", help="Show sample data")
    sample_parser.add_argument("language", help="Rule language")
    sample_parser.add_argument("--split", choices=["train", "test"], default="test",
                              help="Dataset split (default: test)")
    sample_parser.add_argument("--field", choices=["context", "rule", "intent", "detection_logic", "cti"],
                              default="intent", help="Field to display (default: intent)")
    sample_parser.add_argument("-n", type=int, default=1,
                              help="Number of samples (default: 1)")
    
    args = parser.parse_args()
    
    if args.command == "load":
        if args.language == "snort":
            load_snort_rules(args.path, args.test_ratio, args.seed)
        elif args.language == "splunk":
            load_splunk_rules(args.path, args.test_ratio, args.seed)
        elif args.language == "elastic":
            load_elastic_rules(args.path, args.test_ratio, args.seed)
        elif args.language == "generic":
            lang_name = args.lang_name or Path(args.path).stem.split('_')[0]
            load_generic_rules(
                args.path, lang_name,
                context_field=args.context_field,
                rule_field=args.rule_field,
                id_field=args.id_field,
                test_ratio=args.test_ratio,
                seed=args.seed
            )
    elif args.command == "generate-intent":
        generate_intent_batch(
            language=args.language,
            split=args.split,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_workers=args.workers,
            limit=args.limit,
            force=args.force
        )
    elif args.command == "generate-detection-logic":
        generate_detection_logic_batch(
            language=args.language,
            split=args.split,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_workers=args.workers,
            limit=args.limit,
            force=args.force
        )
    elif args.command == "generate-cti":
        generate_cti_batch(
            language=args.language,
            split=args.split,
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            max_workers=args.workers,
            limit=args.limit,
            force=args.force
        )
    elif args.command == "stats":
        show_stats()
    elif args.command == "sample":
        show_sample(
            language=args.language,
            split=args.split,
            field=args.field,
            n=args.n
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()