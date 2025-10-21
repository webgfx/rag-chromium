# VS Code Settings for Optimized Chromium RAG

## 📋 Quick Setup

Copy this to your VS Code `settings.json`:

```json
{
  "github.copilot.chat.mcp.servers": {
    "chromium-rag": {
      "command": "python",
      "args": [
        "E:\\rag-chromium\\scripts\\rag_mcp_server_optimized.py"
      ]
    }
  }
}
```

**Note:** For deployed/remote installations, adjust the path to match your deployment location (e.g., `E:\\rag_chromium_20251021\\scripts\\...`)

## 🔍 How to Open Settings

### Method 1: Command Palette
1. Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
2. Type: "Preferences: Open User Settings (JSON)"
3. Press Enter
4. Paste the configuration above

### Method 2: Settings UI
1. Press `Ctrl+,` (Windows/Linux) or `Cmd+,` (Mac)
2. Search for: "mcp"
3. Click "Edit in settings.json"
4. Paste the configuration above

## ✅ Verify Setup

After saving settings:

1. **Restart VS Code** (important!)
2. Open GitHub Copilot Chat
3. Try asking: "Query the Chromium codebase about WebGPU"
4. First query will take 45-60s (one-time warmup)
5. Subsequent queries: 10-20s ⚡

## 🎯 Test Queries

Try these to see the optimization in action:

```
@chromium-rag How does Chrome handle WebGPU on Qualcomm?
@chromium-rag Explain V8 garbage collection optimization
@chromium-rag How does Blink renderer work?
```

**Pro tip:** Repeat a query to see instant cache hits (<50ms)!

## 🔧 Alternative: Remote Server

If you're using a remote server instead:

```json
{
  "github.copilot.chat.mcp.servers": {
    "chromium-rag-remote": {
      "command": "python",
      "args": [
        "E:\\rag-chromium\\dist\\chromium-rag-client.py",
        "ws://YOUR_SERVER_IP:8080/mcp"
      ]
    }
  }
}
```

## 📊 Performance Comparison

| Scenario | Old Performance | New Performance | Speedup |
|----------|----------------|-----------------|---------|
| First query (cold) | 2-3 minutes | 45-60 seconds | 2-3x |
| New query (warm) | 30-45 seconds | 10-20 seconds | 3-4x |
| Cached query | 30-45 seconds | <50ms | 600x+ |
| GPU utilization | 60-70% | 95%+ | Better |

## 🐛 Troubleshooting

### Server doesn't start?
```bash
# Test manually
cd E:\rag-chromium
python scripts/rag_mcp_server_optimized.py

# Should see:
# 🚀 Initializing Optimized Chromium RAG MCP Server...
# 🔥 Pre-warming models...
# ✅ Server ready in 45.2s
```

### Still slow?
```bash
# Check GPU
nvidia-smi

# Expected:
# - GPU Memory: ~13-14 GB used
# - GPU Util: 80-100% during query
```

### Cache not working?
Query cache is automatic. Verify with:
```python
from copilot_rag_interface import CopilotRAGInterface
rag = CopilotRAGInterface(preload=True)

# First run
rag.query("test query", top_k=5)  # ~12s

# Second run (should be instant)
rag.query("test query", top_k=5)  # <50ms
```

## 💡 Tips

1. **Keep VS Code open** - Server stays warm between queries
2. **Group similar queries** - Cache works better
3. **Start with smaller top_k** - Faster results
4. **Monitor GPU usage** - Should see 80-100% during queries

## 📚 More Info

- Full guide: `PERFORMANCE_OPTIMIZATION.md`
- Benchmark: `python benchmark_performance.py`
- Debug: Check VS Code Output → "MCP Servers"
