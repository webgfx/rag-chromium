# Remote Server Deployment Guide

This guide explains how to deploy the Chromium RAG system as a remote HTTP/WebSocket server that multiple clients can connect to.

## Architecture

```
┌─────────────┐         ┌─────────────────────┐         ┌──────────────────┐
│   Client 1  │         │                     │         │                  │
│  (VS Code)  │────────>│                     │         │  Qdrant Database │
└─────────────┘         │   RAG HTTP Server   │<───────>│  (244K commits)  │
                        │   (GPU Machine)     │         │                  │
┌─────────────┐         │                     │         └──────────────────┘
│   Client 2  │────────>│   Port 8080         │
│  (VS Code)  │         │                     │         ┌──────────────────┐
└─────────────┘         │   Endpoints:        │         │                  │
                        │   - HTTP API        │<───────>│  Embedding Model │
┌─────────────┐         │   - WebSocket       │         │  (11GB, GPU)     │
│   Client 3  │────────>│   - MCP WebSocket   │         │                  │
│  (VS Code)  │         │                     │         └──────────────────┘
└─────────────┘         └─────────────────────┘
```

## Setup on Server Machine

### 1. Deploy the Package

```powershell
# Transfer the package to your server
robocopy "E:\rag-chromium\deployment\rag-chromium-20251022" "\\SERVER\path\" /E /Z /MT:8

# Or use the deployment package on the server
cd \\SERVER\path\rag-chromium-20251022
```

### 2. Install Dependencies

```powershell
.\quick-deploy.bat
```

### 3. Start HTTP Server

```powershell
.\start-http-server.bat
```

This will start the server on `0.0.0.0:8080`, making it accessible from any machine on the network.

**Server Endpoints:**
- HTTP API: `http://YOUR-SERVER-IP:8080/api/search`
- WebSocket: `ws://YOUR-SERVER-IP:8080/ws`
- MCP WebSocket: `ws://YOUR-SERVER-IP:8080/mcp`
- Health Check: `http://YOUR-SERVER-IP:8080/health`

### 4. Configure Firewall

Make sure port 8080 is open:

```powershell
# Windows Firewall
New-NetFirewallRule -DisplayName "Chromium RAG Server" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Allow
```

## Setup on Client Machines

### Option 1: HTTP API (Simple)

Query directly from command line:

```powershell
python scripts\rag_remote_client.py "WebGPU implementation" --server http://192.168.1.100:8080
```

### Option 2: VS Code with MCP WebSocket

Configure VS Code to connect to the remote server.

**Add to `settings.json`:**

```json
{
  "github.copilot.chat.mcp.servers": {
    "rag-chromium": {
      "command": "python",
      "args": [
        "C:\\path\\to\\rag_remote_client.py",
        "--server", "http://192.168.1.100:8080",
        "--method", "mcp"
      ]
    }
  }
}
```

**Then use in Copilot:**
```
@rag-chromium How does Chrome handle WebGPU?
```

### Option 3: Direct HTTP Queries

Use the HTTP API from any programming language:

**Python:**
```python
import requests

response = requests.post(
    "http://192.168.1.100:8080/api/search",
    json={"query": "WebGPU implementation", "top_k": 5}
)
result = response.json()
print(result['result'])
```

**JavaScript/Node.js:**
```javascript
fetch('http://192.168.1.100:8080/api/search', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    query: 'WebGPU implementation',
    top_k: 5
  })
})
.then(r => r.json())
.then(data => console.log(data.result));
```

**PowerShell:**
```powershell
$body = @{
    query = "WebGPU implementation"
    top_k = 5
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://192.168.1.100:8080/api/search" `
    -Method Post -Body $body -ContentType "application/json"

Write-Host $response.result
```

## Advanced Configuration

### Custom Port

```powershell
python scripts\rag_http_server.py --host 0.0.0.0 --port 9000
```

### Run as Background Service

**Windows (using NSSM):**
```powershell
# Install NSSM: https://nssm.cc/download
nssm install ChromiumRAG "python" "C:\path\to\scripts\rag_http_server.py"
nssm start ChromiumRAG
```

**Linux (systemd):**
```bash
sudo nano /etc/systemd/system/chromium-rag.service
```

```ini
[Unit]
Description=Chromium RAG HTTP Server
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/rag-chromium
ExecStart=/usr/bin/python3 /path/to/scripts/rag_http_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable chromium-rag
sudo systemctl start chromium-rag
```

### HTTPS/SSL

For production, use a reverse proxy like nginx:

```nginx
server {
    listen 443 ssl;
    server_name rag.example.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

## Performance

**Server Requirements:**
- GPU: NVIDIA with 16GB+ VRAM recommended
- RAM: 32GB+
- CPU: Modern multi-core processor
- Network: Gigabit ethernet

**Expected Performance:**
- First query (cold start): 45-60s
- Subsequent queries: 10-20s
- Concurrent queries: Handles multiple clients efficiently
- Network latency: Add 10-50ms depending on connection

## Monitoring

Check server health:
```powershell
curl http://192.168.1.100:8080/health
```

View logs:
```powershell
# Server logs are printed to console
# Or redirect to file:
python scripts\rag_http_server.py > server.log 2>&1
```

## Troubleshooting

**Can't connect from client:**
- Check firewall rules on server
- Verify server is running: `curl http://SERVER-IP:8080/health`
- Test from server itself first: `curl http://localhost:8080/health`

**Slow queries:**
- Check GPU utilization on server
- Ensure server has enough RAM
- Check network latency: `ping SERVER-IP`

**Server crashes:**
- Check GPU memory: may need to reduce batch size in config.yaml
- Check system logs for out-of-memory errors
- Ensure Python dependencies are installed: `.\quick-deploy.bat`

## Security Considerations

**For Internal Networks:**
- Current setup is suitable for trusted internal networks
- No authentication required

**For Public Access:**
- Add authentication (API keys, OAuth)
- Use HTTPS/WSS (secure WebSocket)
- Implement rate limiting
- Use a reverse proxy (nginx, Caddy)
- Consider VPN access instead of public exposure

## Migration from Local to Remote

1. **Backup local setup** (already working)
2. **Deploy server** on GPU machine
3. **Test with one client** using `rag_remote_client.py`
4. **Update VS Code settings** for all clients
5. **Monitor performance** and adjust as needed

## Benefits of Remote Deployment

✅ **Centralized Resources:**
- One GPU server serves multiple users
- Share the 13.47 GB database across team
- Consistent results for everyone

✅ **Cost Efficiency:**
- No need for GPU on every machine
- Shared infrastructure and maintenance

✅ **Scalability:**
- Add more clients without additional resources
- Upgrade server hardware benefits everyone

✅ **Maintenance:**
- Update database in one place
- Single point for monitoring and debugging
