#!/usr/bin/env python3
"""
Unified Rules MCP Agent - Web Backend
为前端 HTML 页面提供 API 服务
支持多种规则语言（Snort, Splunk, Elastic等）
"""

import asyncio
import json
import os
import time
import uuid
from typing import Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from openai import AsyncOpenAI
from fastmcp import Client
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-1234")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

# System Prompt - 可通过环境变量自定义
DEFAULT_SYSTEM_PROMPT = """You are a detection rule generation system.

Task: Given a detection context and target language, generate a detection rule.

Output the rule in a code block."""

SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

# 是否使用 [DONE] 标记（实验模式下建议关闭）
USE_DONE_MARKER = os.getenv("USE_DONE_MARKER", "false").lower() == "true"

# 每轮对话最大工具调用次数
MAX_AGENT_ROUND = int(os.getenv("MAX_AGENT_ROUND", "5"))

# 数据模型
class SessionCreate(BaseModel):
    enable_mcp: bool = True
    system_prompt: Optional[str] = None  # 允许每个session自定义prompt


class ChatMessage(BaseModel):
    session_id: str
    message: str


# 会话管理
class Session:
    def __init__(self, session_id: str, enable_mcp: bool = True, system_prompt: Optional[str] = None):
        self.session_id = session_id
        self.enable_mcp = enable_mcp
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        self.mcp_client: Optional[Client] = None
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
        self.tools = []
        self.conversation_history = []
        self.is_finished = False
        
    async def initialize(self):
        """初始化会话"""
        if self.enable_mcp:
            try:
                self.mcp_client = Client(MCP_SERVER_URL,timeout=120)
                await self.mcp_client.__aenter__()
                
                # 获取工具
                mcp_tools = await self.mcp_client.list_tools()
                print(f"已获取 {len(mcp_tools)} 个工具")
                print(mcp_tools)
                self.tools = [self._convert_tool_to_openai(tool) for tool in mcp_tools]
                
            except Exception as e:
                print(f"MCP 连接失败: {e}")
                self.enable_mcp = False
        
        # 初始化对话历史
        self.conversation_history = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
        
    async def cleanup(self):
        """清理会话"""
        if self.mcp_client:
            try:
                await self.mcp_client.__aexit__(None, None, None)
            except:
                pass
    
    def _convert_tool_to_openai(self, mcp_tool) -> dict:
        """转换工具格式"""
        return {
            "type": "function",
            "function": {
                "name": mcp_tool.name,
                "description": mcp_tool.description,
                "parameters": mcp_tool.inputSchema
            }
        }
    
    async def call_tool(self, tool_name: str, arguments: dict) -> tuple[bool, str, Optional[str], float]:
        """调用工具，返回 (success, result, error, duration)"""
        try:
            start_time = time.time()
            result = await self.mcp_client.call_tool(tool_name, arguments)
            duration = (time.time() - start_time) * 1000
            return True, result.data, None, duration
        except Exception as e:
            return False, None, str(e), 0


# 全局会话存储
sessions: Dict[str, Session] = {}


# FastAPI 应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    yield
    # 清理所有会话
    for session in sessions.values():
        await session.cleanup()


app = FastAPI(title="Unified Rules MCP Agent API", lifespan=lifespan)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_html():
    """提供前端 HTML 页面"""
    html_path = "agent_frontend.html"
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"message": "请将 HTML 文件保存为 agent_frontend.html"}


@app.post("/sessions")
async def create_session(session_create: SessionCreate):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    session = Session(
        session_id, 
        session_create.enable_mcp,
        session_create.system_prompt
    )
    
    try:
        await session.initialize()
        sessions[session_id] = session
        
        return {
            "session_id": session_id,
            "enable_mcp": session.enable_mcp,
            "tools_count": len(session.tools)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建会话失败: {str(e)}")


@app.post("/chat/stream")
async def chat_stream(chat_message: ChatMessage):
    """流式聊天"""
    session_id = chat_message.session_id
    user_message = chat_message.message
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    session = sessions[session_id]
    
    if session.is_finished:
        raise HTTPException(status_code=400, detail="会话已结束")
    
    async def event_stream():
        """SSE 事件流"""
        try:
            # 添加用户消息
            session.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # 最多 MAX_AGENT_ROUND 轮工具调用，+1 轮用于无工具时的最终输出
            for round_num in range(MAX_AGENT_ROUND + 1):
                # 判断是否还允许使用工具
                allow_tools = session.enable_mcp and (round_num < MAX_AGENT_ROUND)
                
                # 如果达到工具调用上限，通知前端
                if session.enable_mcp and round_num == MAX_AGENT_ROUND:
                    yield f"data: {json.dumps({'type': 'warning', 'message': f'已达到最大工具调用轮次({MAX_AGENT_ROUND})，将直接输出结果'})}\n\n"
                
                # 调用 OpenAI API
                response = await session.openai_client.chat.completions.create(
                    model=MODEL,
                    messages=session.conversation_history,
                    tools=session.tools if allow_tools else None,
                    tool_choice="auto" if allow_tools else None,
                    stream=True
                )
                logging.debug(f"Round {round_num}, allow_tools={allow_tools}, TOOLS:{session.tools if allow_tools else 'None'}")
                
                assistant_message_content = ""
                assistant_tool_calls = []
                
                # 处理流式响应
                async for chunk in response:
                    delta = chunk.choices[0].delta
                    
                    # 内容流
                    if delta.content:
                        assistant_message_content += delta.content
                        yield f"data: {json.dumps({'type': 'content', 'content': delta.content})}\n\n"
                    
                    # 工具调用流（仅在允许工具时处理）
                    if allow_tools and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.index is not None:
                                idx = tool_call_delta.index
                                
                                # 新的工具调用
                                if idx >= len(assistant_tool_calls):
                                    assistant_tool_calls.append({
                                        "id": tool_call_delta.id or "",
                                        "type": "function",
                                        "function": {
                                            "name": tool_call_delta.function.name or "",
                                            "arguments": ""
                                        }
                                    })
                                
                                # 累积参数
                                if tool_call_delta.function.arguments:
                                    assistant_tool_calls[idx]["function"]["arguments"] += tool_call_delta.function.arguments
                
                # 完成内容输出
                if assistant_message_content:
                    yield f"data: {json.dumps({'type': 'content', 'content': '', 'done': True})}\n\n"
                
                # 添加助手消息到历史
                assistant_message = {
                    "role": "assistant",
                    "content": assistant_message_content
                }
                
                if assistant_tool_calls:
                    assistant_message["tool_calls"] = assistant_tool_calls
                
                session.conversation_history.append(assistant_message)
                
                # 如果没有工具调用，结束
                if not assistant_tool_calls:
                    # 可选：检查 [DONE] 标记
                    if USE_DONE_MARKER and "[DONE]" in assistant_message_content:
                        session.is_finished = True
                        yield f"data: {json.dumps({'type': 'finish', 'message': '任务完成'})}\n\n"
                    else:
                        yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                
                # 执行工具调用
                all_success = True
                for tool_call in assistant_tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args_str = tool_call["function"]["arguments"]
                    
                    try:
                        tool_args = json.loads(tool_args_str)
                    except:
                        tool_args = {}
                    
                    # 发送工具调用开始事件
                    yield f"data: {json.dumps({'type': 'tool_call_start', 'tool_name': tool_name, 'arguments': tool_args, 'reasoning': f'Calling {tool_name}'})}\n\n"
                    
                    # 调用工具
                    success, result, error, duration = await session.call_tool(tool_name, tool_args)
                    
                    # 发送工具结果事件
                    yield f"data: {json.dumps({'type': 'tool_call_result', 'tool_name': tool_name, 'result': result, 'success': success, 'error': error, 'duration_ms': duration})}\n\n"
                    
                    if not success:
                        all_success = False
                    
                    # 添加工具结果到历史
                    session.conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "name": tool_name,
                        "content": result if success else f"Error: {error}"
                    })
                
                if not all_success:
                    yield f"data: {json.dumps({'type': 'warning', 'message': 'Some tool calls failed'})}\n\n"
                
                # 继续下一轮
            
        except Exception as e:
            print(f"处理错误: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    if session_id in sessions:
        session = sessions[session_id]
        await session.cleanup()
        del sessions[session_id]
        return {"message": "会话已删除"}
    else:
        raise HTTPException(status_code=404, detail="会话不存在")


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "sessions_count": len(sessions),
        "mcp_url": MCP_SERVER_URL,
        "model": MODEL
    }


@app.get("/config")
async def get_config():
    """获取当前配置（用于调试）"""
    return {
        "model": MODEL,
        "mcp_url": MCP_SERVER_URL,
        "use_done_marker": USE_DONE_MARKER,
        "max_agent_round": MAX_AGENT_ROUND,
        "system_prompt_length": len(SYSTEM_PROMPT)
    }


if __name__ == "__main__":
    import uvicorn
    
    # 检查 API Key
    if OPENAI_API_KEY == "your-api-key-here":
        print("❌ 错误: 请设置 OPENAI_API_KEY 环境变量")
        print("\n设置方法:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  python agent_backend.py")
        exit(1)
    
    print("🚀 启动 Unified Rules MCP Agent 后端服务")
    print(f"📡 MCP 服务器: {MCP_SERVER_URL}")
    print(f"🤖 LLM 模型: {MODEL}")
    print(f"🔧 最大工具调用轮次: {MAX_AGENT_ROUND}")
    print(f"🌐 访问地址: http://localhost:20001")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=20001,
        log_level="info"
    )