# models/brain — Brain weights + emotion_26d stats

## Required files

```
models/brain/
  emotion_stats.json      # from extract_emotion_stats.py (mean/std per emotion)
  brain.pt                # trained Brain state_dict
  hubert-base-ls960/      # facebook/hubert-base-ls960 (local HF snapshot)
```

### Linked from training output (this machine)

```
train.parquet  →  brain output/results (1)/myned_dataset/parquet/train.parquet
brain weights  →  brain output/results (1)/checkpoints/brain_latest.pt
                  copied to models/brain/brain.pt
```

### Build emotion_stats.json (once, offline)

Where your `train.parquet` lives (must have columns `emotion_label`, `emotion_26d`):

```powershell
python extract_emotion_stats.py path\to\train.parquet models\brain\emotion_stats.json
```

### Runtime emotion_26d

```python
from emotion_manager import EmotionManager
emo_mgr = EmotionManager("models/brain/emotion_stats.json")
t = emo_mgr.get_emotion_26d_tensor("thinking", 0.75, device="cuda")  # [1, 26]
```

### Brain 4 inputs

| # | Input | Built by |
|---|--------|----------|
| 1 | Audio WAV | Parler TTS |
| 2 | Emotion label + intensity | JSON / Groq |
| 3 | Emotion 26-D `[1,26]` | `emotion_manager.py` |
| 4 | Prosody `[T,3]` | `prosody_gpu.py` |

### Enable

```python
# orchestrator.py
LIPSYNC_ENGINE = "brain"
```

or:

```powershell
python orchestrator_brain.py
```
