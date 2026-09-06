# Canonical body (locked)

| Field | Value |
|-------|--------|
| **rig_id** | `smplx_male_55` |
| **motion engine (primary)** | **DART** → SMPL-X parameters |
| **mesh source** | `models/smplx` / DART locked-head layout |
| **head (phase 1)** | SMPL-X default head |
| **head (phase 2)** | Project ARKit face → neck attach |
| **not used for body drive** | MotionLCM, MetaHuman, Mixamo (unless retarget later) |

## Order of work

1. DART env + checkpoints + demo on default head  
2. Attach our face  
3. Director / TTS / agents glue  
4. Optional appearance upgrades  

## Code locations

- DART: `third_party/DART/`  
- Setup: `scripts/setup_dart.ps1`  
- Guide: `docs/DART_SMPL_X_SETUP.md`  
