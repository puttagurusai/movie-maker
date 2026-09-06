# Session Timeline (append-only continuous body)

## What you get

On session start, Blender creates an **empty named timeline** (`Session_<session_id>`)
with a start marker. Each generated clip is then **appended** (not replaced) and
**labeled on the timeline** so you can see which frame range is which motion.

In one chat session, each turn **appends** motion into a single Blender Action:

| Item | Value |
|------|--------|
| Master Action | `Session_Timeline` |
| Turn 1 | keys on frames `1 … L1` only |
| Turn 2 | keys on frames `L1+1 … L1+L2` only (**clip 1 not re-baked**) |
| Turn N | only the new range |
| Scrub | frame 1 → end = continuous session (no picking each clip) |

Live play still runs the **new** segment only. After the clip settles, **Space / scrub plays the session master** (`Session_<id>`) from frame 1 through every labeled clip — not the last received Action. Camera keys bake onto the same frames; Space also replays the camera.

### MoMask inpaint (edit a slice, keep the rest)

Type in chat (after a clip was generated so `*_ric.npy` exists):

```
inpaint 40-80 a person waves with the right hand
inpaint 1.0s-2.5s a person turns left
```

Uses official `edit_t2m.py`: only that frame/time range is regenerated.

## How to use in Blender

1. Reload `blender_receiver.py` and start the stream receiver.
2. Run `python orchestrator_agents.py` and chat several turns.
3. Console: `MASTER append clip#N label='#N walk  [115-230]' …`
4. Timeline markers: start/end of each clip (`#2 wave  [115-230]`).
5. Select armature → **Action Editor** / **Dope Sheet** → action `Session_<session_id>`.
6. Set timeline to frame **1**, press **Play** — full session 1→2→…→N.
7. Type `reset` / `new scene` in chat to create a new named timeline.

## Length rules (bugfix)

Bake span is **not** raw `Action.frame_range` (that can be 480–2000 of extrapolated junk).

Order of authority:
1. Pipeline `action_frames` / director `motion_length` (MoMask length + rest pads)
2. Tight range from **actual keyframes** on the source Action  
3. Hard cap **220** frames per clip (~11s @ 20fps)

Console should look like:
```text
bake plan src='momask_…' sample=1-114 span=114 (action_frames=114|…) → dest 1-114
SESSION APPEND-ONLY clip#2 … dest 115-230 span=116 …
```

If you see span=480 or dest …-2000, reload `blender_receiver.py` and type `reset` once.

## Cost

```
Turn k costs O(length of clip k) only — never O(sum of all previous clips).
```

## Root continuity

When appending clip N, pelvis start is offset to match the end of clip N−1 so the character does not teleport (local pelvis continuity inside `Session_Timeline`).

## Next: user access / fix a bad gap (roadmap)

Goal: if frames 80–120 look bad, regenerate that gap while keeping continuity.

| Layer | Needs Blender? | Role |
|-------|----------------|------|
| **Timeline scrub / review** | Yes (now) | See whole session, pick bad range by eye |
| **Export session** | Optional | Export `Session_Timeline` → BVH/FBX for external tools |
| **Gap regen (MoMask edit)** | Pipeline, not full Blender UI | Mask [t0,t1], regenerate middle, **re-append only that span** on master Action |
| **Lightweight web/timeline UI** | No Blender for *edit request* | User marks gap + prompt; worker calls edit + append-only patch |
| **Blender as final viewport** | Yes for pixel-accurate pose | Still best DCC for polish, NLA, lighting, render |

**Is Blender necessary?**

- **For v1 continuous session + review:** yes — Action Editor / Dope Sheet is the timeline.
- **For “user access” to fix gaps long-term:** Blender is **not** required for the *request* (mark range + new caption). The **pipeline** can:
  1. Read master range / joints from export or in-memory session state  
  2. MoMask temporal inpaint or re-gen that segment  
  3. **Patch only those frames** on `Session_Timeline` (same append-only idea, overwrite a slice)  
- Blender remains the **viewer / render / optional polish** host, not the only way to *command* a fix.

### Recommended user-access stages

1. **Now (done):** append-only `Session_Timeline` + scrub whole session.  
2. **Soon:** CLI/API `session patch --start-frame F0 --end-frame F1 --prompt "…"` that overwrites only that range with continuity at edges.  
3. **Later:** simple UI (range markers + prompt) without opening Action Editor.  
4. **Optional:** MoMask inpaint for seamless middle; rest ease at cut edges.

Never re-bake the whole 10-clip session to fix one gap — only rewrite the bad frame range.
