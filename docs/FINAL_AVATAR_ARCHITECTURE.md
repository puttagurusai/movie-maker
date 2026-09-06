# Final goal architecture: Director + morphable full avatar

## Vision (one sentence)

**Character description** → build/morph a full human (head + body) → **director script** (say / feel / do) drives speech, face, and body on that character.

```
User / LLM
   │
   ├─ character card (who they look like)
   └─ director beats (what they perform)
            │
            ▼
   ┌────────────────────┐
   │  Avatar runtime    │
   │  identity | face   │
   │  body | voice      │
   └─────────┬──────────┘
             │
     web viewer / Unreal
```

---

## Two independent problems (do not mix them)

| Problem | Question | System |
|---------|----------|--------|
| **A. Identity (morphable look)** | Who is this person? Shape, skin, age, gender, clothes | Character / DNA / morph targets / textures |
| **B. Performance (director)** | What do they say, feel, do right now? | Beats → TTS + face agents + body actions |

You already have a strong start on **B (face only)**.  
**A** is the missing foundation for “body + head morphable by description.”

If you bolt a random body onto one fixed head, you never get “any character from description.”  
If you only morph look without director beats, you get a mannequin, not a performer.

---

## Target data model

### 1) Character card (identity — slow to change)

```json
{
  "id": "hero_01",
  "description": "30-year-old South Asian man, short black hair, light beard, athletic, blue shirt",
  "params": {
    "gender": "male",
    "age": 0.35,
    "height": 0.55,
    "weight": 0.45,
    "skin_tone": [0.72, 0.52, 0.40],
    "face_shape": { "...morphs or DNA vector..." },
    "body_shape": { "...body morphs..." },
    "hair": "short_black",
    "outfit": "casual_blue_shirt"
  },
  "voice": {
    "style": "parler_style_prompt_or_clone_id",
    "pitch": 0.0
  },
  "assets": {
    "rig_id": "ue_metahuman_or_mixamo_v1",
    "face_arkit": true
  }
}
```

`description` (text) is compiled **once** (or on change) into `params` + optional baked textures/meshes.

### 2) Director beat (performance — every line)

```json
{
  "character_id": "hero_01",
  "text": "Hello [laugh] good to meet you",
  "emotion": "happy",
  "intensity": 0.8,
  "actions": ["wave", "look_camera", "talk_open"],
  "action_timing": "start"
}
```

You already have this shape in `face_agents/director_schema.py` and `docs/DIRECTOR_AVATAR_PLAN.md`.

---

## Pipeline layers (final state)

```
┌─────────────────────────────────────────────────────────────┐
│ L0  DIRECTOR / LLM                                          │
│     chat or script → character_id + beats[]                 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ L1  CHARACTER COMPILER                                      │
│     text description → character card (params + assets)     │
│     cache: characters/{id}/                                 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ L2  AVATAR ASSET LAYER (morphable head+body, ONE rig)       │
│     load base humanoid → apply face/body morphs + textures  │
│     export or stream GLB / UE mesh with ARKit face          │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ L3  PERFORMANCE LAYER (what you largely have for face)      │
│     TTS (voice from character card)                         │
│     face agents: lips / eyes / brows / cheeks / head        │
│     body agents: actions + emotion posture + idle           │
│     tokens: [laugh] [eww] …                                 │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│ L4  RUNTIME VIEWER                                          │
│     Three.js or Unreal: skeleton + morphs + blendshapes     │
│     UDP/WS: viseme | emotion | head | body | identity       │
└─────────────────────────────────────────────────────────────┘
```

---

## How “morphable head + body from description” actually works

There is no single free API that magically outputs a perfect unique human from a sentence. Real products use one of these **identity backends**:

### Path A — Parametric base (recommended architecture)

**One fixed topology** (same vertices, same bones, same ARKit shape keys) for every character.

| Piece | Role |
|-------|------|
| Base mesh | Male/female templates (or one unisex) |
| Face morphs | 50–200 shape keys / MetaHuman DNA / FLAME |
| Body morphs | Height, weight, muscle, proportions (SMPL-X style or MH body) |
| Textures | Albedo/normal from library or AI-textured maps |
| Clothes | Swap outfits as separate skinned meshes |

**Character description → numbers:**

```
LLM/parser: "tall thin elderly man, pale skin"
  → age↑, weight↓, height↑, skin_tone light, face morphs elderly
```

Then runtime: `mesh.morphTargetInfluences[i] = params[i]` (and body equivalent).

**Why this fits your pipeline:**  
You already drive **ARKit-52** on a fixed face topology. The end state is the same idea for **full body**: fixed rig, variable morphs.

### Path B — MetaHuman / Character Creator (production quality)

- Description → pick or edit MH/CC preset  
- Export full character (or DNA)  
- Performance still uses your director beats + ARKit + body anim  

Best visual quality; heavier tooling (Unreal/CC).

### Path C — Generate new mesh per character (AI 3D)

- Image/text → 3D mesh (Tripo, Meshy, etc.)  
- Then **auto-rig** (Mixamo / Mesh2Motion)  
- **Hard part:** no stable ARKit face → lips/expression break  

Use only for backgrounds, not as your main talking hero until re-topologized onto Path A.

**For your final goal, Path A (or B feeding A) is correct.** Path C is optional later.

---

## How director performance sits on top

Every frame (conceptually):

```
identity_params   → shape of head/body (slow; change on character switch)
emotion + intensity → face expression agents + body posture bias
audio               → lips (wav2arkit) + energy gestures
actions[]           → body clips or procedural recipes (wave, shrug, …)
tokens []           → reaction audio + face bursts
```

Order of authority (resolve conflicts):

1. **Lips** (speech always wins on jaw/mouth)  
2. **Director actions** (timed gestures)  
3. **Emotion posture** (sad slump, angry tense)  
4. **Idle** (breath, sway)  
5. **Identity morphs** (do not animate every frame unless intentional)

---

## Mapping onto *this* repo (concrete)

| Module today | Evolves into |
|--------------|--------------|
| `face.glb` + ARKit | `avatar_base.glb` (full body + ARKit head morphs) |
| `face_agents/*` | keep; add `body_agent` again on real rig |
| `director_schema.py` | beats + `character_id` |
| `orchestrator_agents.py` | load character card → set identity → play beats |
| `parler_voice.py` | voice style from character card |
| `web_viewer/face_viewer.html` | avatar viewer: morph identity + drive bones + ARKit |
| `temp/director_script_example.json` | full scripts with character |
| `docs/DIRECTOR_AVATAR_PLAN.md` | performance half of the design |

New modules (to add over time):

```
characters/
  schema.py              # CharacterCard
  compile_description.py # text → params (LLM + rules)
  library/               # cached cards + baked maps
avatar/
  identity_apply.py      # push morph weights / textures to viewer
  body_actions.py        # action catalog → bones/clips
  rig_contract.md        # required bones + ARKit names
```

---

## Rig contract (non-negotiable for “any character”)

Every character asset must share:

1. **Same skeleton** (or retargetable standard: Mixamo / UE Mannequin / MH)  
2. **ARKit-compatible face blendshapes** (your current 52 names)  
3. **Body morph channels** with a fixed name list (or body DNA vector)  
4. **Neck/head joint** stable for gaze/head agent  
5. **Meters** scale, T/A-pose bind  

Without (1)+(2), director performance will not transfer across characters.

---

## Phased plan to reach the goal

### Phase 1 — Director performance on **one** fixed full avatar (current + body)
- Keep face agents + director beats  
- Add **one** male (or MH) full body with **your** ARKit head topology or MH head  
- Body actions work; identity fixed  
- **Exit:** “director can make this one human speak/act”

### Phase 2 — Morphable **identity** on that same rig
- Add body morphs + face morphs (or MH DNA / FLAME + body)  
- Character card drives morphs at load time  
- Description → params (LLM structured output)  
- **Exit:** “same performance, different look without re-exporting code”

### Phase 3 — Character library + voice binding
- Multiple cards, outfits, hair  
- Voice style / clone per character  
- Cache compiled assets  

### Phase 4 — Full product loop
- User: “Be a tired old professor who is sarcastic”  
- System: compile character + default emotion posture  
- User/LLM: director script for the scene  
- Avatar performs  

### Phase 5 (optional) — Photoreal upgrade path
- MetaHuman or ScanStore-quality maps on parametric body  
- Still same director + ARKit performance layer  

---

## Example end-to-end session

```json
{
  "character": {
    "description": "Young athletic man, warm brown skin, short hair, friendly"
  },
  "beats": [
    {
      "text": "Hey — welcome in.",
      "emotion": "happy",
      "intensity": 0.8,
      "actions": ["wave", "look_camera"]
    },
    {
      "text": "Hmm [hmm] that is a hard problem.",
      "emotion": "thinking",
      "intensity": 0.6,
      "actions": ["think_chin"]
    }
  ]
}
```

Runtime:

1. Compile description → morph weights + skin + hair + voice  
2. Load base avatar, apply identity  
3. For each beat: TTS → face agents + body actions  

---

## What not to do

| Trap | Why it fails |
|------|----------------|
| New random mesh per character with no shared ARKit | Lips/expressions break |
| Face-only morphs + unrelated body | “Floating head” / identity break |
| Director actions without shared skeleton | Can’t reuse wave/shrug |
| Expect pure AI mesh gen to replace parametric rig | Unstable for production talking avatars |

---

## Success criteria (definition of done)

- [ ] Change character description → avatar **look** updates (head + body)  
- [ ] Same director beat works on **any** character on the shared rig  
- [ ] Speech lips + emotion face + body actions run together  
- [ ] Reaction tokens still work  
- [ ] Can swap characters mid-session without rewriting orchestrator  

---

## Immediate next step in *this* project

1. **Lock the product split:** identity vs performance (this doc).  
2. **Choose identity backend:**  
   - **Parametric / MH** (right for final goal)  
   - not “download one static Mixamo only” (OK only as Phase 1 temporary body).  
3. **Phase 1:** one full-body ARKit-capable avatar + director body actions.  
4. **Phase 2:** morph channels + `character` card from description.

Performance director format is already started (`director_schema.py`).  
Identity morphability is the major new workstream.
