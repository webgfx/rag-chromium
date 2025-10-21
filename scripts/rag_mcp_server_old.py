#!/usr/bin/env python3
"""
MCP (Model Context Protocol) server for Chromium RAG.
This allows GitHub Copilot to automatically query the RAG system.

Install: Add to VS Code's MCP settings (see README)
"""

import sys
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from copilot_rag_interface import CopilotRAGInterface


class RAGMCPServer:
    """MCP server for Chromium RAG integration."""
    
    def __init__(self):
        self.rag = None
    
    def _get_rag(self):
        """Lazy load RAG interface."""
        if self.rag is None:
            self.rag = CopilotRAGInterface()
        return self.rag
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests."""
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return self._list_tools()
        elif method == "tools/call":
            return await self._call_tool(params)
        elif method == "resources/list":
            return self._list_resources()
        elif method == "resources/read":
            return await self._read_resource(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    def _list_tools(self) -> Dict[str, Any]:
        """List available RAG tools."""
        return {
            "tools": [
                {
                    "name": "query_chromium_rag",
                    "description": "Search the Chromium codebase using semantic search across 150K+ commits",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query about Chromium"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_chromium_context",
                    "description": "Get detailed context about a specific Chromium topic",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "Specific topic or component (e.g., 'V8', 'WebGL', 'memory management')"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            ]
        }
    
    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a RAG tool."""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name == "query_chromium_rag":
            query = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            
            rag = self._get_rag()
            result = rag.query(query, top_k=top_k)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result
                    }
                ]
            }
        
        elif tool_name == "get_chromium_context":
            topic = arguments.get("topic")
            
            rag = self._get_rag()
            result = rag.generate_answer(topic, top_k=5)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result
                    }
                ]
            }
        
        return {"error": f"Unknown tool: {tool_name}"}
    
    def _list_resources(self) -> Dict[str, Any]:
        """List available RAG resources."""
        return {
            "resources": [
                {
                    "uri": "rag://chromium/stats",
                    "name": "Chromium RAG Statistics",
                    "description": "Current database statistics and coverage",
                    "mimeType": "application/json"
                },
                {
                    "uri": "rag://chromium/latest_query",
                    "name": "Latest RAG Query Results",
                    "description": "Results from the most recent query",
                    "mimeType": "text/markdown"
                }
            ]
        }
    
    async def _read_resource(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Read a RAG resource."""
        uri = params.get("uri")
        
        if uri == "rag://chromium/stats":
            # Get database stats
            stats = {
                "total_documents": 152142,
                "collection": "chromium_complete",
                "model": "BAAI/bge-large-en-v1.5",
                "embedding_dim": 1024,
                "phases_complete": ["Phase 1", "Phase 2", "Phase 3 (76%)"]
            }
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(stats, indent=2)
                    }
                ]
            }
        
        elif uri == "rag://chromium/latest_query":
            results_file = Path("copilot_rag_results.md")
            if results_file.exists():
                content = results_file.read_text(encoding='utf-8')
            else:
                content = "No query results available yet."
            
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": "text/markdown",
                        "text": content
                    }
                ]
            }
        
        return {"error": f"Unknown resource: {uri}"}
    
    async def run(self):
        """Run the MCP server."""
        while True:
            try:
                # Read JSON-RPC request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                request = json.loads(line)
                response = await self.handle_request(request)
                
                # Send JSON-RPC response to stdout
                print(json.dumps(response), flush=True)
                
            except Exception as e:
                error_response = {
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    }
                }
                print(json.dumps(error_response), flush=True)


async def main():
    """Main entry point."""
    server = RAGMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
