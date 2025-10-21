#!/usr/bin/env python3
"""
Client for connecting to remote Chromium RAG HTTP server.
Can be used as a drop-in replacement for local MCP server.
"""

import asyncio
import json
import sys
import argparse
import websockets
import requests
from typing import Any, Dict


class RemoteRAGClient:
    """Client for remote Chromium RAG server."""
    
    def __init__(self, server_url: str):
        """
        Initialize remote client.
        
        Args:
            server_url: Base URL of the server (e.g., "http://192.168.1.100:8080")
        """
        self.server_url = server_url.rstrip('/')
        self.http_endpoint = f"{self.server_url}/api/search"
        self.ws_endpoint = self.server_url.replace('http://', 'ws://').replace('https://', 'wss://')
    
    def query_http(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query via HTTP POST.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Query results
        """
        try:
            response = requests.post(
                self.http_endpoint,
                json={"query": query, "top_k": top_k},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    async def query_websocket(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query via WebSocket.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Query results
        """
        ws_url = f"{self.ws_endpoint}/ws"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send query
                await websocket.send(json.dumps({
                    "query": query,
                    "top_k": top_k
                }))
                
                # Receive result
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            return {"error": str(e)}
    
    async def query_mcp(self, query: str) -> Dict[str, Any]:
        """
        Query via MCP WebSocket protocol.
        
        Args:
            query: Search query
            
        Returns:
            Query results
        """
        mcp_url = f"{self.ws_endpoint}/mcp"
        
        try:
            async with websockets.connect(mcp_url) as websocket:
                # Wait for initialized message
                init_msg = await websocket.recv()
                print(f"Connected to MCP server: {init_msg}", file=sys.stderr)
                
                # Send prompt request
                request = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "prompts/get",
                    "params": {
                        "name": "chromium-rag",
                        "arguments": {
                            "query": query
                        }
                    }
                }
                
                await websocket.send(json.dumps(request))
                
                # Receive result
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            return {"error": str(e)}


def main():
    """CLI interface for remote RAG client."""
    parser = argparse.ArgumentParser(description="Remote Chromium RAG Client")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--server", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--method", choices=["http", "websocket", "mcp"], default="http", help="Connection method")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    client = RemoteRAGClient(args.server)
    
    print(f"🔍 Querying remote server: {args.server}")
    print(f"📝 Query: {args.query}")
    print()
    
    if args.method == "http":
        result = client.query_http(args.query, args.top_k)
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        else:
            print(f"✅ Query completed in {result.get('query_time', 0):.2f}s")
            print()
            print(result.get('result', 'No results'))
    
    elif args.method == "websocket":
        result = asyncio.run(client.query_websocket(args.query, args.top_k))
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        else:
            print(f"✅ Query completed in {result.get('query_time', 0):.2f}s")
            print()
            print(result.get('result', 'No results'))
    
    elif args.method == "mcp":
        result = asyncio.run(client.query_mcp(args.query))
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)
        else:
            print("✅ MCP query completed")
            print()
            # Extract result from MCP response
            if "result" in result and "messages" in result["result"]:
                for msg in result["result"]["messages"]:
                    if "content" in msg and "text" in msg["content"]:
                        print(msg["content"]["text"])


if __name__ == "__main__":
    main()
