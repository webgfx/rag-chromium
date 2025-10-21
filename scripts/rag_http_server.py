#!/usr/bin/env python3
"""
HTTP/WebSocket server for Chromium RAG - enables remote connections.
Run this on a server machine and connect from clients via HTTP/WebSocket.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict
import logging

# Add project root to path (parent of scripts directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from copilot_rag_interface import CopilotRAGInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


class QueryResponse(BaseModel):
    result: str
    sources: list
    query_time: float


app = FastAPI(
    title="Chromium RAG Server",
    description="HTTP/WebSocket server for Chromium RAG queries",
    version="2.0.0"
)

# Enable CORS for remote access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG interface
rag_interface = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup."""
    global rag_interface
    
    logger.info("🚀 Starting Chromium RAG HTTP Server...")
    logger.info("🔥 Pre-warming models for instant queries...")
    
    start_time = time.time()
    rag_interface = CopilotRAGInterface(preload=True)
    
    # Warmup query
    try:
        _ = rag_interface.query("test", top_k=1, use_cache=False)
    except Exception as e:
        logger.warning(f"⚠️  Warmup query failed: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"✅ Server ready in {elapsed:.1f}s - queries will be FAST!")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "Chromium RAG Server",
        "version": "2.0.0",
        "status": "running",
        "endpoints": {
            "http_query": "/api/search (POST)",
            "websocket": "/ws",
            "mcp_websocket": "/mcp (WebSocket)"
        }
    }


@app.get("/health")
async def health():
    """Health check for monitoring."""
    return {"status": "healthy", "ready": rag_interface is not None}


@app.post("/api/search", response_model=QueryResponse)
async def search(request: QueryRequest):
    """HTTP endpoint for RAG queries."""
    if rag_interface is None:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        start_time = time.time()
        result = rag_interface.query(request.query, top_k=request.top_k)
        query_time = time.time() - start_time
        
        # Extract sources from result
        sources = []
        if hasattr(result, 'metadata'):
            sources = result.metadata.get('sources', [])
        
        return QueryResponse(
            result=str(result),
            sources=sources,
            query_time=query_time
        )
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time queries."""
    await websocket.accept()
    logger.info("WebSocket client connected")
    
    try:
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            query = request.get("query", "")
            top_k = request.get("top_k", 5)
            
            if not query:
                await websocket.send_json({"error": "Query is required"})
                continue
            
            try:
                start_time = time.time()
                result = rag_interface.query(query, top_k=top_k)
                query_time = time.time() - start_time
                
                await websocket.send_json({
                    "result": str(result),
                    "query_time": query_time,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Query error: {e}")
                await websocket.send_json({
                    "error": str(e),
                    "status": "error"
                })
    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")


@app.websocket("/mcp")
async def mcp_websocket(websocket: WebSocket):
    """MCP-compatible WebSocket endpoint."""
    await websocket.accept()
    logger.info("MCP WebSocket client connected")
    
    try:
        # Send initialize message
        await websocket.send_json({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {
                "name": "rag-chromium",
                "version": "2.0.0",
                "capabilities": {
                    "prompts": {
                        "chromium-rag": {
                            "description": "Search Chromium commit history"
                        }
                    }
                }
            }
        })
        
        while True:
            data = await websocket.receive_text()
            request = json.loads(data)
            
            method = request.get("method")
            params = request.get("params", {})
            request_id = request.get("id")
            
            if method == "prompts/get":
                # Handle prompt request
                prompt_name = params.get("name")
                arguments = params.get("arguments", {})
                query = arguments.get("query", "")
                
                if not query:
                    await websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": "Query is required"}
                    })
                    continue
                
                try:
                    start_time = time.time()
                    result = rag_interface.query(query, top_k=5)
                    query_time = time.time() - start_time
                    
                    await websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "result": {
                            "description": f"Chromium RAG results for: {query}",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": {
                                        "type": "text",
                                        "text": str(result)
                                    }
                                }
                            ],
                            "metadata": {
                                "query_time": query_time
                            }
                        }
                    })
                except Exception as e:
                    logger.error(f"Query error: {e}")
                    await websocket.send_json({
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32603, "message": str(e)}
                    })
            
            elif method == "prompts/list":
                # List available prompts
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": {
                        "prompts": [
                            {
                                "name": "chromium-rag",
                                "description": "Search Chromium commit history",
                                "arguments": [
                                    {
                                        "name": "query",
                                        "description": "Search query",
                                        "required": True
                                    }
                                ]
                            }
                        ]
                    }
                })
            
            else:
                # Unknown method
                await websocket.send_json({
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"}
                })
    
    except WebSocketDisconnect:
        logger.info("MCP WebSocket client disconnected")
    except Exception as e:
        logger.error(f"MCP WebSocket error: {e}")


def main():
    """Start the HTTP server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chromium RAG HTTP Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"HTTP endpoint: http://{args.host}:{args.port}/api/search")
    logger.info(f"WebSocket endpoint: ws://{args.host}:{args.port}/ws")
    logger.info(f"MCP WebSocket endpoint: ws://{args.host}:{args.port}/mcp")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )


if __name__ == "__main__":
    main()
