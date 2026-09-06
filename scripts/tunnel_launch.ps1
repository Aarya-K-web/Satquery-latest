param([string]$Tool="ngrok", [int]$Port=8001)
if (-not (Get-Command uvicorn -ErrorAction SilentlyContinue)) { Write-Error "uvicorn not found"; exit 1 }
Write-Host "Starting VQA server on :$Port ..."
Start-Process python -ArgumentList "-m","uvicorn","src.vqa_server:app","--host","0.0.0.0","--port","$Port" -WindowStyle Hidden
Start-Sleep 4
try { Invoke-RestMethod "http://127.0.0.1:$Port/health" -TimeoutSec 5 | ConvertTo-Json -Compress | Write-Host } catch { Write-Warning "health check failed: $_" }
if ($Tool -eq "ngrok") {
  if (-not (Get-Command ngrok -ErrorAction SilentlyContinue)) { Write-Host "Install ngrok: https://ngrok.com/download"; exit 0 }
  Write-Host "Exposing via ngrok http $Port — copy Forwarding URL and set VQA_SERVER_URL=https://<id>.ngrok-free.app/infer"
  ngrok http $Port
} else {
  if (-not (Get-Command cloudflared -ErrorAction SilentlyContinue)) { Write-Host "Install cloudflared: https://developers.cloudflare.com/cloudflare-one/connections/connect/networks/downloads/"; exit 0 }
  Write-Host "Exposing via cloudflared --url http://localhost:$Port"
  cloudflared tunnel --url "http://localhost:$Port"
}
