# Look / realistic scene kits

Pipeline flow: **LookAgent → `type=look` → Blender `bpy`** builds `Look_Set`.

Minecraft look is gone when assets are present: **PBR textures + real glTF props** (Poly Haven CC0).

## Layout

```
assets/looks/
  hdri/           # sky / lighting
  textures/       # grass, asphalt, brick, plaster, wood, concrete, sand
  models/         # pine_tree, island_tree, urban_facade, street_lamp, bench, chair, table
  presets.json
```

## Download / refresh

```bash
python tools/download_look_assets.py
```

Uses Poly Haven public API (CC0). Needs network once.

## Locations (any scene → nearest kit)

| Prompt words | Kit | Assets used |
|--------------|-----|-------------|
| park / garden | `exterior_park` | grass tex + trees + benches |
| forest / woods | `exterior_forest` | dirt/concrete tex + pine/island trees |
| street / city | `exterior_street` | asphalt + sidewalk + **apartment facade GLB** + lamps |
| playground | `exterior_playground` | grass + pad + bench |
| station / platform | `exterior_station` | concrete platform + bench + lamp |
| beach | `exterior_beach` | sand textures + dunes |
| room / office | `interior_simple` | plaster walls + wood floors when tex present |
| studio | `studio_cyc` | soft HDRI |

Fallback: if a GLB is missing, colored/textured blocks still build (never crash).

## Product order

1. Scene (this)  
2. T2M / speech / face / camera on that set  
