# Run face pipeline after reboot

## 1) Web viewer (2 terminals)

```powershell
cd C:\me\proj\projface_v1\web_viewer
python -m http.server 8000
```

```powershell
cd C:\me\proj\projface_v1\web_viewer
python ws_bridge.py
```

Browser: http://localhost:8000/face_viewer.html

## 2) Talking face (orchestrator)

```powershell
cd C:\me\proj\projface_v1
python orchestrator_agents.py
```

## 3) Smoke test (no Blender)

```powershell
cd C:\me\proj\projface_v1
python tools/smoke_test_pipeline.py
```

## Tokens

Use markers in text: `[laugh]`, `[eww]`, `[gasp]`, `[sigh]`, …

- Speech words → Parler TTS  
- Tokens → `temp/vocal_sounds/*.wav` (real SFX, not Parler) + lips + expression  

Rebuild real vocals if needed:

```powershell
python tools/build_real_vocals.py
```
