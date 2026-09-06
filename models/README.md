# models/

Large weights are **not** stored in git. Download each model with the **one link** below into the matching folder.

| Folder / file | Model | One link |
|---------------|--------|----------|
| `parler-tts-mini-v1/` | Parler-TTS Mini v1 | https://huggingface.co/parler-tts/parler-tts-mini-v1 |
| `wav2arkit_cpu/` | wav2arkit ONNX lips | https://huggingface.co/myned-ai/wav2arkit_cpu |
| `brain/hubert-base-ls960/` | HuBERT base | https://huggingface.co/facebook/hubert-base-ls960 |
| `brain/*.pt` | ProjFace Brain modules | Copy from your training export (not public HF) |
| `audio2emotion_v2.2/` | NVIDIA Audio2Emotion | https://catalog.ngc.nvidia.com/orgs/nvidia/teams/maxine/resources/audio2emotion-v2.2 |
| `face_landmarker.task` | MediaPipe Face Landmarker | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |
| `kokoro-v1.0.onnx` | Optional Kokoro TTS | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx |
| `voices-v1.0.bin` | Optional Kokoro voices | https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin |

## Hugging Face CLI examples

```powershell
pip install -U "huggingface_hub[cli]"

huggingface-cli download parler-tts/parler-tts-mini-v1 --local-dir models/parler-tts-mini-v1
huggingface-cli download myned-ai/wav2arkit_cpu --local-dir models/wav2arkit_cpu
huggingface-cli download facebook/hubert-base-ls960 --local-dir models/brain/hubert-base-ls960
```

Small helper for MediaPipe (+ optional Kokoro):

```powershell
.\download_assets.ps1
```

## Minimum to talk + lips + catalog body

1. Parler  
2. wav2arkit  

MoMask, Brain, Audio2Emotion are optional upgrades.
