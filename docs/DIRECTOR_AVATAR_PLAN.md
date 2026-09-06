# Director-level human avatar (face + body)

**Goal:** You (or an LLM) act as **director**:

> “Say this line, with this emotion, while doing this action.”

The avatar performs **dialogue + face emotion + body action** together — not a random robot body bolted on.

---

## Backup (do not overwrite lightly)

| Item | Path |
|------|------|
| Face-only working snapshot | `web_viewer/backup_face_only_20260723_231047/` |
| Restore | `web_viewer/backup_face_only_20260723_231047/RESTORE.ps1` |

Restore anytime:

```powershell
cd web_viewer\backup_face_only_20260723_231047
.\RESTORE.ps1
```

---

## What we already have (face director layer)

| Channel | Status | Driven by |
|---------|--------|-----------|
| Dialogue text | ✅ | Parler TTS |
| Emotion label | ✅ | `emotion` + `intensity` → face agents |
| Lips | ✅ | wav2arkit |
| Face expression | ✅ | brows/eyes/cheeks/brain + `[laugh]` etc. |
| Head pose | ✅ | head_agent nods/breath |
| Body | ❌ removed | needs **matching** body mesh + action system |

Current sentence format:

```json
{"text": "Hello [laugh] nice to meet you", "emotion": "happy", "intensity": 0.8}
```

**Director format extends this with `actions`.**

---

## Target director script (one “beat”)

```json
{
  "beats": [
    {
      "text": "Welcome in — great to see you.",
      "emotion": "happy",
      "intensity": 0.75,
      "actions": ["wave", "look_camera"],
      "action_timing": "start"
    },
    {
      "text": "Hmm [hmm] that is a tough one.",
      "emotion": "thinking",
      "intensity": 0.65,
      "actions": ["think_chin", "weight_shift"],
      "action_timing": "during"
    },
    {
      "text": "Oh [eww] that is disgusting.",
      "emotion": "disgusted",
      "intensity": 0.85,
      "actions": ["recoil", "hands_reject"],
      "action_timing": "start"
    }
  ]
}
```

### Fields

| Field | Meaning |
|-------|---------|
| `text` | What to **say** (may include face tokens `[laugh]` `[eww]`) |
| `emotion` | Face + posture mood |
| `intensity` | 0–1 strength |
| `actions` | List of **body/performance** verbs from the catalog |
| `action_timing` | `start` \| `during` \| `end` (default `during`) |

Face tokens stay for **micro reactions** (laugh sound + face burst).  
`actions` are for **full-body / gesture direction**.

---

## Action catalog (phase 1)

Performance verbs the body agent will understand. Start small; expand later.

### Social / talk
- `idle` — breath + small sway  
- `talk_open` — open hand gestures while speaking  
- `talk_emphasize` — beat gesture on energy peaks  
- `wave` — greeting wave  
- `nod_yes` / `shake_no` — agreement / denial (body + head)  
- `shrug` — “I don’t know”  
- `point_forward` — indicate  

### Emotion body
- `recoil` — disgust / surprise pull back  
- `celebrate` — happy open arms (soft)  
- `slump` — sad / tired  
- `tense` — angry / assertive upright  
- `think_chin` — thinking hand near chin  
- `hands_reject` — push away (disgust/no)  

### Stage
- `look_camera` — face/torso toward camera  
- `look_left` / `look_right`  
- `weight_shift` — idle leg/hip shift (needs full body)  

### Phase 2 (later)
- Sit / stand / walk / object interaction / lip-sync hand props  

---

## How to get a body that matches THIS face

**Rule:** Never glue a random female robot/Mixamo character to this head again.  
The body must be **male, similar proportions, neck compatible**, and we attach **our** `face.glb` at the neck (body head hidden).

### Recommended paths (pick one)

#### A) Best visual match — MetaHuman / ScanStore ecosystem (recommended)

Your head identity comes from ScanStore → MetaHuman-style maps.

1. Create a **male MetaHuman** in Unreal (or use free MH sample) close in age/ethnicity to the face.
2. Export body + skeleton (or use MH LOD mesh).
3. Hide MH head; attach our ARKit head at `neck_01` / head joint.
4. Scale neck carefully; match skin tone on body albedo if possible.

**Pros:** Pro quality, male, skinned correctly.  
**Cons:** Export pipeline work (Blender/Unreal).

#### B) Fast path — Male Mixamo / RPM **body only** (temporary)

1. Download a **male** Mixamo character (not Xbot robot), T-pose or A-pose.
2. Export FBX → GLB.
3. In viewer: hide head meshes / scale head bone ~0; parent `face.glb` to neck.
4. Recolor body materials toward ScanStore skin.

**Pros:** Fast skeleton + gestures.  
**Cons:** Face/body identity mismatch until you swap to A.

#### C) You provide a body

Drop a rigged male GLB/FBX with standard humanoid bones into:

```
web_viewer/body_assets/body_male_v1.glb
```

We wire attach + control; no placeholder robots.

### Hard requirements for any body mesh

- [ ] Male proportions  
- [ ] Humanoid skeleton (Mixamo or UE Mannequin bone names documented)  
- [ ] Neck bone usable for head attach  
- [ ] Torso + arms minimum (legs optional phase 1)  
- [ ] Head of body **hidden** when our face is attached  
- [ ] Skin color roughly compatible with ScanStore albedo  

---

## Control architecture

```
Director script (JSON / LLM)
        │
        ▼
┌───────────────────┐
│  Director runtime │  parse beats → timeline
└─────────┬─────────┘
          │
    ┌─────┴──────────────────────┐
    ▼                            ▼
 Speech + face                 Body performance
 (existing FaceCoordinator)    (BodyDirector — NEW)
    │                            │
    │  viseme / emotion / head   │  action clips + procedural
    ▼                            ▼
 UDP 9001  ────────────────►  ws_bridge  ──►  viewer
                                              face.glb + body.glb
```

### Layers of body motion (quality stack)

1. **Procedural** (always on) — breath, sway, talk energy arms, emotion posture  
2. **Action clips** (director verbs) — short Mixamo/custom animations blended  
3. **Emotion bias** — sad = slump, happy = open, angry = tense (multiplies posture)  
4. **Face sync** — head nod/gaze stay on face agents; body doesn’t fight them  

---

## Implementation phases

### Phase 0 — DONE
- Face-only backup  
- Director plan + schema + sample script  

### Phase 1 — Schema + face already obeys director emotions
- Accept `actions` in JSON (ignore until body exists)  
- LLM prompt teaches director fields  
- No wrong body loaded  

### Phase 2 — Matching body asset
- User/you place approved `body_male_v1.glb`  
- Neck attach, scale, skin tune  
- Idle + talk procedural only  

### Phase 3 — Action library
- Map catalog verbs → clip or procedural recipe  
- Blend with speech timing (`action_timing`)  

### Phase 4 — Full director LLM
- Chat → beats with text + emotion + actions  
- Optional timeline editor JSON  

---

## Success criteria (“director level”)

- [ ] Director can write a script of beats without code changes  
- [ ] Each beat: correct **words**, **face emotion**, and **body action**  
- [ ] Face never loses identity (our head + maps)  
- [ ] Body is male / matched enough that it doesn’t break immersion  
- [ ] Can restore face-only in one script if body experiment fails  

---

## Immediate next step

1. Keep using face-only (safe).  
2. Acquire **male** body mesh (path A or C preferred).  
3. Drop into `web_viewer/body_assets/` and we wire attach + Phase 2–3.

Do **not** auto-download robot/female demo bodies.
