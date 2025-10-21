#!/usr/bin/env python3
"""
Fast-start MCP server for Chromium RAG with lazy initialization.
Starts immediately and loads models on first query.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path (parent of scripts directory)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Suppress logging to stderr during initialization
import logging
logging.basicConfig(level=logging.ERROR)

from copilot_rag_interface import CopilotRAGInterface


class FastMCPServer:
    """Fast-starting MCP server with lazy initialization."""
    
    def __init__(self):
        self.name = "rag-chromium"
        self.version = "2.0.0"
        self.rag_interface = None  # Lazy init on first query
    
    def _ensure_initialized(self):
        """Initialize RAG system on first use."""
        if self.rag_interface is None:
            print("🔄 Loading RAG system...", file=sys.stderr)
            self.rag_interface = CopilotRAGInterface(preload=False)
            print("✅ RAG system ready!", file=sys.stderr)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP protocol requests."""
        
        method = request.get("method")
        params = request.get("params", {})
        request_id = request.get("id")
        
        # Initialize message
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "prompts": {
                            "listChanged": False
                        }
                    },
                    "serverInfo": {
                        "name": self.name,
                        "version": self.version
                    }
                }
            }
        
        # List prompts
        elif method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "prompts": [
                        {
                            "name": "chromium-rag",
                            "description": "Search Chromium commit history for relevant code changes and implementations",
                            "arguments": [
                                {
                                    "name": "query",
                                    "description": "Your search query about Chromium code or features",
                                    "required": True
                                }
                            ]
                        }
                    ]
                }
            }
        
        # Get prompt (execute query)
        elif method == "prompts/get":
            prompt_name = params.get("name")
            arguments = params.get("arguments", {})
            query = arguments.get("query", "")
            
            if not query:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Query parameter is required"
                    }
                }
            
            try:
                # Initialize on first query
                self._ensure_initialized()
                
                # Execute query
                print(f"🔍 Query: {query}", file=sys.stderr)
                result = self.rag_interface.query(query, top_k=5)
                
                return {
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
                        ]
                    }
                }
            except Exception as e:
                print(f"❌ Error: {e}", file=sys.stderr)
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Query failed: {str(e)}"
                    }
                }
        
        # Notifications (no response needed)
        elif method == "notifications/initialized":
            return None
        
        # Unknown method
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def run(self):
        """Run the MCP server (stdio transport)."""
        print(f"🚀 {self.name} MCP server v{self.version} starting...", file=sys.stderr)
        
        while True:
            try:
                # Read request from stdin
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                
                if not line:
                    break
                
                # Parse request
                request = json.loads(line.strip())
                
                # Handle request
                response = await self.handle_request(request)
                
                # Send response (if not a notification)
                if response is not None:
                    print(json.dumps(response), flush=True)
            
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}", file=sys.stderr)
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }
                print(json.dumps(error_response), flush=True)
            
            except Exception as e:
                print(f"❌ Server error: {e}", file=sys.stderr)


def main():
    """Entry point."""
    server = FastMCPServer()
    asyncio.run(server.run())


if __name__ == "__main__":
    main()
