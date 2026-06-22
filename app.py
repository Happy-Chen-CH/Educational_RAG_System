"""
EduRAG Web API 服务
===================
基于 FastAPI 的 Web 后端，薄封装 IntegratedQASystem 的核心能力，
提供 SSE 流式问答、对话历史管理、学科列表等 RESTful API。

启动方式:
    python app.py
    浏览器打开 http://localhost:8000

API 端点:
    POST /api/chat              — SSE 流式问答
    GET  /api/history/{session_id}     — 获取对话历史
    DELETE /api/history/{session_id}   — 清空对话历史
    GET  /api/sources            — 获取学科列表
    GET  /api/conversations      — 列出所有会话（供侧边栏用）
"""

import json
import uuid
import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# —— 确保项目根目录在 sys.path 中，使各模块可正确导入 ——
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import IntegratedQASystem
from base import Config

# ==================== 全局实例 ====================

qa: IntegratedQASystem = None
config: Config = None


# ==================== 请求模型 ====================

class ChatRequest(BaseModel):
    query: str
    source_filter: str | None = None
    session_id: str | None = None


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动时初始化 QA 系统，关闭时清理数据库连接。"""
    global qa, config
    config = Config()
    qa = IntegratedQASystem()
    yield
    if qa and hasattr(qa, "mysql_client") and qa.mysql_client:
        qa.mysql_client.close()


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title="EduRAG API",
    description="教育领域智能问答系统 Web API",
    version="1.0.0",
    lifespan=lifespan,
)

# 允许跨域访问（开发阶段宽松配置）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== API 端点 ====================


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    SSE 流式问答接口。

    接收用户查询，调用 IntegratedQASystem.query() 获取答案，
    通过 Server-Sent Events 流式返回，前端可逐字渲染。

    请求体 (JSON):
        - query (str): 用户查询文本
        - source_filter (str, optional): 学科类别过滤
        - session_id (str, optional): 会话 ID

    返回:
        text/event-stream: 逐 token 产出答案
    """
    # 未提供 session_id 时自动生成
    sid = req.session_id or str(uuid.uuid4())

    async def event_stream():
        try:
            for token, is_complete in qa.query(
                req.query, source_filter=req.source_filter, session_id=sid
            ):
                if token:
                    # 将答案拆分为小片段逐字推送，营造流式体验
                    chunk_size = 3
                    for i in range(0, len(token), chunk_size):
                        chunk = token[i : i + chunk_size]
                        yield f"data: {json.dumps({'token': chunk, 'session_id': sid, 'done': False}, ensure_ascii=False)}\n\n"
                        await asyncio.sleep(0.015)
                if is_complete:
                    yield f"data: {json.dumps({'token': '', 'session_id': sid, 'done': True}, ensure_ascii=False)}\n\n"
                    break
        except Exception as e:
            yield f"data: {json.dumps({'token': f'系统错误: {str(e)}', 'session_id': sid, 'done': True}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲
        },
    )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """获取指定会话的最近 5 轮对话历史。"""
    history = qa.get_session_history(session_id)
    return {"session_id": session_id, "history": history}


@app.delete("/api/history/{session_id}")
async def clear_history(session_id: str):
    """清空指定会话的全部对话历史。"""
    success = qa.clear_session_history(session_id)
    return {"session_id": session_id, "success": success}


@app.get("/api/sources")
async def get_sources():
    """返回系统中所有可用的学科类别列表。"""
    return {"sources": config.VALID_SOURCES}


@app.get("/api/conversations")
async def get_conversations():
    """
    列出所有会话及其最近一次提问（供侧边栏历史列表用）。
    按最近活动时间倒序排列，最多返回 100 条。
    """
    try:
        qa.mysql_client.cursor.execute("""
            SELECT c.session_id, c.question, c.timestamp
            FROM conversations c
            JOIN (
                SELECT session_id, MAX(id) AS max_id
                FROM conversations
                GROUP BY session_id
            ) latest ON c.id = latest.max_id
            ORDER BY c.timestamp DESC
            LIMIT 100
        """)
        rows = qa.mysql_client.cursor.fetchall()
        return [
            {
                "session_id": row[0],
                "latest_question": row[1],
                "latest_time": str(row[2]) if row[2] else "",
            }
            for row in rows
        ]
    except Exception:
        # MySQL 不可用时返回空列表，不阻塞前端渲染
        return []


# ==================== 静态文件挂载 ====================

static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(static_dir, exist_ok=True)

# 静态文件挂载须在 API 路由之后，否则会屏蔽 API 路由
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


# ==================== 启动入口 ====================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
