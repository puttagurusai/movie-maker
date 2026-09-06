# Story paragraph → MP4

## One command

Blender must be open with **face stream receiver** running (`N-panel → Avatar → Start` / `bpy.ops.face.stream_receiver()`).

```bash
python run_movie_pipeline.py ^
  --story "On a sunny city street a person walks forward, stops to wave hello, then says goodbye." ^
  --title street_hello_30s ^
  --target-s 30 ^
  --no-brain ^
  --join-timeline ^
  --preview-mp4
```

Or from a prepared beat sheet (recommended for a reliable ~30s cut):

```bash
python run_movie_pipeline.py ^
  --script temp/stories/demo_30s.json ^
  --title street_hello_30s ^
  --target-s 30 ^
  --no-brain ^
  --join-timeline ^
  --preview-mp4
```

## What happens

1. **Director** — story → shots (LLM with `--llm`, or rules / script JSON)  
2. **Bake** — TTS + face + body (catalog / MoMask) + camera plans  
3. **Package** — `temp/movies/<title>_<ts>/movie_package.json`  
4. **`--join-timeline`** — push all shots onto one Blender Session timeline + **SessionCam** history  
5. **`--preview-mp4`** — Workbench playblast + speech mix → MP4  

Output MP4 path is printed; usually under `temp/movies/.../preview_*.mp4`.

## From Blender sidebar (after a session is already on the timeline)

- **Export preview** — fast playblast of current Session  
- **Export movie** — higher-quality package path when available  

Or chat: `export preview` / `export movie`.

## Camera notes (fixed)

- Live take no longer freezes over **SessionCam** history.  
- After each take / `rest`, cameras re-bind so **Play / scrub / export** move with the timeline.  
- Multi-cam clips switch `scene.camera` from SessionCam take table.

## Tips

| Flag | Use |
|------|-----|
| `--target-s 30` | Aim ~30 seconds total |
| `--no-brain` | Faster lips-only face |
| `--llm` | Smarter director (needs `llm_fw` config) |
| `--play` | Live play in Blender after bake (slower) |
| `--preview-res 1280x720` | Sharper preview (slower) |

MoMask shots need GPU time; prefer **catalog** waves/talk for a quick demo, keep **one** walk as MoMask.
