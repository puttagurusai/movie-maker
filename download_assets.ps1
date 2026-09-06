# download_assets.ps1
# Small assets into models/. Large HF models: see README.md (one link per model).
# Run:  .\download_assets.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Models = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $Models | Out-Null

Write-Host "=== Downloading small assets into: $Models ===" -ForegroundColor Cyan

$files = @(
    @{
        Url  = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
        Dest = (Join-Path $Models "face_landmarker.task")
        Size = "~4 MB"
        Note = "Live MediaPipe capture"
    },
    @{
        Url  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
        Dest = (Join-Path $Models "kokoro-v1.0.onnx")
        Size = "~310 MB"
        Note = "Optional legacy TTS (Parler is primary)"
    },
    @{
        Url  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
        Dest = (Join-Path $Models "voices-v1.0.bin")
        Size = "~27 MB"
        Note = "Optional legacy TTS voices"
    }
)

foreach ($file in $files) {
    if (Test-Path $file.Dest) {
        Write-Host "Already exists: $($file.Dest)" -ForegroundColor Yellow
        continue
    }
    Write-Host "Downloading $($file.Dest) ($($file.Size)) — $($file.Note) ..." -ForegroundColor White
    try {
        & curl.exe -L --progress-bar -o $file.Dest $file.Url
        if ((Test-Path $file.Dest) -and ((Get-Item $file.Dest).Length -gt 1000)) {
            Write-Host "OK: $($file.Dest)" -ForegroundColor Green
        } else {
            Write-Host "Failed or empty: $($file.Dest)" -ForegroundColor Red
        }
    } catch {
        Write-Host "Failed: $($file.Dest)" -ForegroundColor Red
        Write-Host $file.Url
    }
}

Write-Host ""
Write-Host "=== Large models (manual / huggingface-cli) ===" -ForegroundColor Cyan
Write-Host "Parler:    https://huggingface.co/parler-tts/parler-tts-mini-v1"
Write-Host "wav2arkit: https://huggingface.co/myned-ai/wav2arkit_cpu"
Write-Host "HuBERT:    https://huggingface.co/facebook/hubert-base-ls960"
Write-Host "MoMask:    https://github.com/EricGuo5513/momask-codes"
Write-Host "           weights https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing"
Write-Host "Audio2Emotion: NGC Maxine audio2emotion-v2.2"
Write-Host ""
Write-Host "Full table: README.md  →  Model downloads"
Write-Host "Done."
