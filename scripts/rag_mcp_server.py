#!/usr/bin/env python3
"""
GPU-Optimized MCP server for Chromium RAG with fast query responses.
Pre-warms models on startup for instant queries.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Add project root to path (parent of scripts directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

from copilot_rag_interface import CopilotRAGInterface


class OptimizedMCPServer:
    """GPU-optimized MCP server with model pre-warming."""
    
    def __init__(self):
        self.name = "rag-chromium-optimized"
        self.version = "2.0.0"
        
        print("🚀 Initializing Optimized Chromium RAG MCP Server...", file=sys.stderr)
        
        # Pre-warm models for instant first query
        start_time = time.time()
        self.rag_interface = CopilotRAGInterface(preload=True)
        
        # Do a test query to fully warm everything
        print("🔥 Warming up with test query...", file=sys.stderr)
        try:
            _ = self.rag_interface.query("test", top_k=1, use_cache=False)
        except Exception as e:
            print(f"⚠️  Warmup query failed: {e}", file=sys.stderr)
        
        elapsed = time.time() - start_time
        print(f"✅ Server ready in {elapsed:.1f}s - queries will be FAST!", file=sys.stderr)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests."""
        
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        try:
            if method == "initialize":
                return self._handle_initialize(request_id)
            
            elif method == "tools/list":
                return self._handle_tools_list(request_id)
            
            elif method == "tools/call":
                return await self._handle_tool_call(request_id, params)
            
            elif method == "resources/list":
                return self._handle_resources_list(request_id)
            
            else:
                return self._error_response(request_id, -32601, f"Method not found: {method}")
        
        except Exception as e:
            print(f"❌ Error handling {method}: {e}", file=sys.stderr)
            return self._error_response(request_id, -32603, str(e))
    
    def _handle_initialize(self, request_id) -> Dict[str, Any]:
        """Handle initialize request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {},
                    "resources": {}
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        }
    
    def _handle_tools_list(self, request_id) -> Dict[str, Any]:
        """List available tools."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": [
                    {
                        "name": "query_chromium_rag",
                        "description": "Query the Chromium codebase knowledge base (150K+ commits). Use this when you need information about Chrome/Chromium internals, implementations, or commit history. Returns relevant code snippets and commit information.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Your question about Chromium (e.g., 'How does Chrome handle WebGPU on Qualcomm?')"
                                },
                                "top_k": {
                                    "type": "number",
                                    "description": "Number of results to return (default: 5)",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    },
                    {
                        "name": "get_chromium_context",
                        "description": "Get broader context about Chromium features or subsystems. Better for understanding overall architecture or getting comprehensive overviews.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Topic to get context about (e.g., 'WebGPU architecture')"
                                },
                                "top_k": {
                                    "type": "number",
                                    "description": "Number of context items (default: 10)",
                                    "default": 10
                                }
                            },
                            "required": ["query"]
                        }
                    }
                ]
            }
        }
    
    async def _handle_tool_call(self, request_id, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tool call."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "query_chromium_rag":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            
            print(f"📊 Query: '{query}' (k={top_k})", file=sys.stderr)
            
            # Use GPU-optimized query with caching
            result = self.rag_interface.query(query, top_k=top_k, use_cache=True)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
        
        elif tool_name == "get_chromium_context":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 10)
            
            print(f"📚 Context: '{query}' (k={top_k})", file=sys.stderr)
            
            # Use GPU-optimized answer generation
            result = self.rag_interface.generate_answer(query, top_k=top_k, use_cache=True)
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": result
                        }
                    ]
                }
            }
        
        else:
            return self._error_response(request_id, -32602, f"Unknown tool: {tool_name}")
    
    def _handle_resources_list(self, request_id) -> Dict[str, Any]:
        """List available resources."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "resources": []
            }
        }
    
    def _error_response(self, request_id, code: int, message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    async def run(self):
        """Run the MCP server on stdio."""
        print("🎯 MCP server listening on stdio...", file=sys.stderr)
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse JSON-RPC request
                request = json.loads(line)
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response to stdout
                print(json.dumps(response), flush=True)
                
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON: {e}", file=sys.stderr)
                continue
            
            except KeyboardInterrupt:
                print("\n👋 Shutting down...", file=sys.stderr)
                break
            
            except Exception as e:
                print(f"❌ Unexpected error: {e}", file=sys.stderr)
                continue


async def main():
    """Main entry point."""
    server = OptimizedMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
