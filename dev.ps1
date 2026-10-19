# Windows: start backend (FastAPI :8000) and frontend (Vite :5173). Close the window to stop.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "backend\.venv\Scripts\python.exe")) {
    Write-Host "Creating backend\.venv (Python 3.11 or 3.12 recommended for TensorFlow)..."
    py -3.11 -m venv backend\.venv
    backend\.venv\Scripts\pip install -r backend\requirements.txt
}
if (-not (Test-Path "frontend\node_modules")) {
    Push-Location frontend; npm install; Pop-Location
}

$back = Start-Process -PassThru -NoNewWindow -WorkingDirectory backend `
    -FilePath "$PSScriptRoot\backend\.venv\Scripts\python.exe" -ArgumentList "-m","uvicorn","app.main:app","--port","8000"
Write-Host "Backend on http://localhost:8000  -  App on http://localhost:5173"
try {
    Push-Location frontend; npm run dev
} finally {
    Pop-Location
    Stop-Process -Id $back.Id -ErrorAction SilentlyContinue
}
