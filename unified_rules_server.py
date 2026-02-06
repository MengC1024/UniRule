#!/usr/bin/env python3
"""
Dual Semantic Space Retrieval MCP Server
双语义空间检索服务

提供两个语义空间的检索：
- intent (φ_threat): 威胁意图空间
- detection_logic (φ_detection): 检测逻辑空间

Agent 根据用户输入类型选择合适的空间进行检索
"""

import json
import sqlite3
import os
import sys
import numpy as np
import requests
import time
from typing import Optional, Literal
from pathlib import Path
from fastmcp import FastMCP


# ============================================================
# 配置
# ============================================================

DB_DIR = os.getenv("DB_DIR", "./databases")
SERVER_MODE = os.getenv("SERVER_MODE", "http")
HTTP_HOST = os.getenv("HTTP_HOST", "0.0.0.0")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8000"))

# Embedding API 配置
EMBEDDING_API_URL = os.getenv(
    "EMBEDDING_API_URL",
    "http://localhost:8007/v1/embeddings"
)
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "qwen3-embedding-8b")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")  # 添加 API Key 支持

# API 调用配置
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "60"))
EMBEDDING_MAX_RETRIES = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))

# 支持的规则语言
SUPPORTED_LANGUAGES = ["snort", "sigma", "splunk", "elastic", "suricata", "yara"]

# 语义空间类型
SEMANTIC_SPACES = ["intent", "detection_logic"]


# ============================================================
# 双语义空间索引
# ============================================================

class DualSemanticIndex:
    """双语义空间向量索引
    
    维护两个语义空间的 embedding 索引：
    - intent: 威胁意图 (φ_threat)
    - detection_logic: 检测逻辑 (φ_detection)
    """
    
    def __init__(self, db_dir: str = DB_DIR):
        self.db_dir = Path(db_dir)
        self.api_url = EMBEDDING_API_URL
        self.model_name = EMBEDDING_MODEL_NAME
        self.api_key = EMBEDDING_API_KEY
        self.embedding_dim = None
        
        # 规则数据 (共享)
        self.rules: list[str] = []
        self.languages: list[str] = []
        self.rule_ids: list[str] = []
        
        # Intent 空间 (φ_threat)
        self.intents: list[str] = []
        self.intent_embeddings: Optional[np.ndarray] = None
        
        # Detection Logic 空间 (φ_detection)
        self.detection_logics: list[str] = []
        self.detection_logic_embeddings: Optional[np.ndarray] = None
        
        self._initialized = False
    
    def _get_headers(self) -> dict:
        """获取 API 请求头"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _test_api_connection(self):
        """测试 Embedding API 连接"""
        print(f"Testing Embedding API connection...")
        print(f"  URL: {self.api_url}")
        print(f"  Model: {self.model_name}")
        print(f"  API Key: {'configured' if self.api_key else 'not set'}")
        
        try:
            test_embedding = self._get_embedding("test")
            self.embedding_dim = len(test_embedding)
            print(f"✓ API connection successful (dim={self.embedding_dim})")
            return True
        except Exception as e:
            print(f"✗ API connection failed: {e}")
            raise RuntimeError(f"Cannot connect to Embedding API: {e}")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取单个文本的 embedding"""
        for attempt in range(EMBEDDING_MAX_RETRIES):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json={"model": self.model_name, "input": text},
                    timeout=EMBEDDING_TIMEOUT
                )
                response.raise_for_status()
                data = response.json()
                return np.array(data['data'][0]['embedding'], dtype=np.float32)
            except Exception as e:
                if attempt < EMBEDDING_MAX_RETRIES - 1:
                    time.sleep((attempt + 1) * 2)
                else:
                    raise RuntimeError(f"API call failed: {e}")
    
    def _get_embeddings_batch(self, texts: list[str], space_name: str) -> np.ndarray:
        """批量获取 embeddings"""
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, self.embedding_dim or 1)
        
        print(f"  Encoding {len(texts)} texts for {space_name} space...")
        
        all_embeddings = []
        total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
        
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
            batch_texts = texts[i:i + EMBEDDING_BATCH_SIZE]
            batch_num = i // EMBEDDING_BATCH_SIZE + 1
            
            print(f"    [{batch_num}/{total_batches}] Processing {len(batch_texts)} texts...", end=" ", flush=True)
            
            for attempt in range(EMBEDDING_MAX_RETRIES):
                try:
                    response = requests.post(
                        self.api_url,
                        headers=self._get_headers(),
                        json={"model": self.model_name, "input": batch_texts},
                        timeout=EMBEDDING_TIMEOUT
                    )
                    response.raise_for_status()
                    data = response.json()
                    
                    batch_embeddings = [
                        np.array(item['embedding'], dtype=np.float32)
                        for item in data['data']
                    ]
                    all_embeddings.extend(batch_embeddings)
                    print(f"✓")
                    break
                except Exception as e:
                    if attempt < EMBEDDING_MAX_RETRIES - 1:
                        print(f"⚠️  Retry...")
                        time.sleep((attempt + 1) * 2)
                    else:
                        raise RuntimeError(f"Batch {batch_num} failed: {e}")
        
        return np.array(all_embeddings, dtype=np.float32)
    
    def load_from_databases(self):
        """从所有数据库加载并构建双语义空间索引"""
        print("Loading dual semantic index...")
        
        self._test_api_connection()
        
        # 重置数据
        self.rules = []
        self.languages = []
        self.rule_ids = []
        self.intents = []
        self.detection_logics = []
        
        for lang in SUPPORTED_LANGUAGES:
            db_path = self.db_dir / f"{lang}_rules.db"
            if not db_path.exists():
                continue
            
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            try:
                # 加载有 intent 或 detection_logic 的规则
                cursor.execute("""
                    SELECT id, rule, intent, detection_logic 
                    FROM rules 
                    WHERE (intent IS NOT NULL AND intent != '')
                       OR (detection_logic IS NOT NULL AND detection_logic != '')
                """)
                rows = cursor.fetchall()
                
                count = 0
                for rule_id, rule, intent, detection_logic in rows:
                    # 至少有一个字段才加入
                    if intent or detection_logic:
                        self.rule_ids.append(rule_id)
                        self.rules.append(rule)
                        self.languages.append(lang)
                        self.intents.append(intent or "")
                        self.detection_logics.append(detection_logic or "")
                        count += 1
                
                print(f"  {lang}: {count} rules loaded")
            except sqlite3.OperationalError as e:
                print(f"  {lang}: skipped ({e})")
            finally:
                conn.close()
        
        if not self.rules:
            print("Warning: No rules found. Run 'load_rules.py generate-intent/detection-logic' first.")
            self._initialized = True
            return
        
        # 构建 Intent 空间索引
        intent_texts = [t for t in self.intents if t]
        if intent_texts:
            print(f"\nBuilding Intent (φ_threat) index...")
            # 为空的 intent 使用占位符
            texts_to_embed = [t if t else "[EMPTY]" for t in self.intents]
            self.intent_embeddings = self._get_embeddings_batch(texts_to_embed, "intent")
        
        # 构建 Detection Logic 空间索引
        dl_texts = [t for t in self.detection_logics if t]
        if dl_texts:
            print(f"\nBuilding Detection Logic (φ_detection) index...")
            texts_to_embed = [t if t else "[EMPTY]" for t in self.detection_logics]
            self.detection_logic_embeddings = self._get_embeddings_batch(texts_to_embed, "detection_logic")
        
        self._initialized = True
        print(f"\n✓ Index ready: {len(self.rules)} rules from {len(set(self.languages))} languages")
        print(f"  Intent space: {sum(1 for t in self.intents if t)} entries")
        print(f"  Detection Logic space: {sum(1 for t in self.detection_logics if t)} entries")
    
    def search(self, query: str, space: str, k: int = 5, language: Optional[str] = None) -> list[dict]:
        """在指定语义空间中检索
        
        Args:
            query: 查询文本
            space: 语义空间 ("intent" 或 "detection_logic")
            k: 返回数量
            language: 可选，限制返回的语言
        
        Returns:
            list of {intent, detection_logic, rule, language, score}
        """
        if not self._initialized:
            self.load_from_databases()
        
        # 选择空间
        if space == "intent":
            embeddings = self.intent_embeddings
            descriptions = self.intents
        else:  # detection_logic
            embeddings = self.detection_logic_embeddings
            descriptions = self.detection_logics
        
        if embeddings is None or len(embeddings) == 0:
            return []
        
        # 获取查询向量
        query_embedding = self._get_embedding(query)
        
        # 计算余弦相似度
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = embeddings_norm @ query_norm
        
        # 过滤空描述
        for i, desc in enumerate(descriptions):
            if not desc:
                similarities[i] = -np.inf
        
        # 如果指定语言，过滤
        if language:
            language = language.lower()
            for i, lang in enumerate(self.languages):
                if lang != language:
                    similarities[i] = -np.inf
        
        # 获取 top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if similarities[idx] == -np.inf:
                continue
            results.append({
                "id": self.rule_ids[idx],
                "intent": self.intents[idx] if self.intents[idx] else None,
                "detection_logic": self.detection_logics[idx] if self.detection_logics[idx] else None,
                "rule": self.rules[idx],
                "language": self.languages[idx],
                "score": float(similarities[idx])
            })
        
        return results
    
    def get_stats(self) -> dict:
        """获取索引统计"""
        if not self._initialized:
            self.load_from_databases()
        
        lang_counts = {}
        for lang in self.languages:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        return {
            "total_rules": len(self.rules),
            "languages": lang_counts,
            "intent_count": sum(1 for t in self.intents if t),
            "detection_logic_count": sum(1 for t in self.detection_logics if t),
            "embedding_dim": self.embedding_dim,
            "api_url": self.api_url,
            "model_name": self.model_name
        }


# ============================================================
# MCP Server
# ============================================================

# 全局索引实例
index = DualSemanticIndex()

# 创建 FastMCP 服务器
mcp = FastMCP("semantic-retrieval-server")


@mcp.tool()
def search_rules(
    query: str, 
    space: str = "intent",
    k: int = 5, 
    language: Optional[str] = None
) -> str:
    """Search for similar detection rules in semantic space.
    
    Two semantic spaces available:
    
    - "intent" (φ_threat): Threat semantics space
      Use when query describes: attack techniques, threat actors, malicious goals,
      or what the adversary is trying to achieve.
      Examples: "credential theft attack", "ransomware C2 communication", 
      "adversary disabling security controls", "data exfiltration attempt"
    
    - "detection_logic" (φ_detection): Detection mechanism space  
      Use when query describes: specific events, log sources, technical patterns,
      or how to detect something.
      Examples: "PowerShell event 4104 script block", "AWS CloudTrail API calls",
      "process spawned by cmd.exe", "network connection to port 445"
    
    If the query mixes both (e.g., "detect credential dumping via LSASS"), 
    prefer "intent" as it captures the core threat semantics.
    
    Args:
        query: Search query text
        space: Semantic space - "intent" or "detection_logic"  
        k: Number of results (1-20)
        language: Optional, filter by rule language (snort, splunk, elastic)
    
    Returns:
        JSON with {id, intent, detection_logic, rule, language, score}
    """
    # 参数验证
    k = max(1, min(20, k))
    
    if space not in SEMANTIC_SPACES:
        return json.dumps({
            "error": f"Invalid space: {space}",
            "valid_spaces": SEMANTIC_SPACES
        }, indent=2)
    
    if language and language.lower() not in SUPPORTED_LANGUAGES:
        return json.dumps({
            "error": f"Unsupported language: {language}",
            "supported": SUPPORTED_LANGUAGES
        }, indent=2)
    
    try:
        results = index.search(query, space=space, k=k, language=language)
        return json.dumps({
            "query": query,
            "space": space,
            "total": len(results),
            "results": results
        }, indent=2, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"Search failed: {str(e)}"}, indent=2)


@mcp.tool()
def get_index_stats() -> str:
    """获取语义索引的统计信息
    
    返回索引中的规则数量、语言分布、各空间覆盖率等信息。
    """
    try:
        stats = index.get_stats()
        return json.dumps(stats, indent=2)
    except Exception as e:
        return json.dumps({"error": f"Failed to get stats: {str(e)}"}, indent=2)


# ============================================================
# 主函数
# ============================================================

def main():
    """主函数"""
    print("=" * 60)
    print("Dual Semantic Space Retrieval MCP Server")
    print("=" * 60)
    print(f"Embedding API: {EMBEDDING_API_URL}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Spaces: {SEMANTIC_SPACES}")
    print("=" * 60)
    
    # 启动时加载索引
    try:
        index.load_from_databases()
    except Exception as e:
        print(f"\n❌ Failed to initialize index: {e}")
        print("\nPlease check:")
        print(f"  1. Embedding API is running at {EMBEDDING_API_URL}")
        print(f"  2. Database directory exists: {DB_DIR}")
        print(f"  3. Run 'python load_rules.py generate-intent <lang>' first")
        print(f"  4. Run 'python load_rules.py generate-detection-logic <lang>' first")
        sys.exit(1)
    
    if not index.rules:
        print("\n⚠️  No rules indexed. Please run:")
        print("   python load_rules.py generate-intent <language>")
        print("   python load_rules.py generate-detection-logic <language>")
    
    mode = SERVER_MODE.lower()
    
    if mode == "stdio":
        print("\n🚀 Starting stdio mode...", flush=True)
        mcp.run()
    elif mode == "http":
        print(f"\n🚀 Starting HTTP mode on {HTTP_HOST}:{HTTP_PORT}...", flush=True)
        mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT)
    else:
        print(f"❌ Error: Unknown SERVER_MODE '{mode}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()