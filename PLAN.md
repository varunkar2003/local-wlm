# local-wlm — GPU Physics Simulation & 3D Video Pipeline

Goal: build a physics simulation engine pipeline that produces high-quality 3D
simulation videos (rigid bodies, fluids, soft bodies, destruction), using only
free GPU resources for the heavy rendering.

## Core stack

| Layer | Tool | Why |
|---|---|---|
| Physics | [Genesis](https://genesis-world.readthedocs.io/) (`genesis-world`, v1.0) | Unified multi-physics: rigid, MPM, SPH/PBD fluids, FEM soft bodies. 10–80x faster than Isaac/MuJoCo-MJX on GPU. Pure Python API. |
| Fast preview render | Genesis built-in rasterizer (pyrender) | Headless-capable, renders straight to MP4. |
| Beauty render | Genesis ray tracer (LuisaRender/Nyx) or Blender Cycles | Cinematic quality for final content. Both run on a free T4. |
| Post | ffmpeg | Encode, speed-ramp, concat, upscale. |

## Free GPU strategy

| Platform | Free allowance | Role |
|---|---|---|
| **Kaggle** | ~30 h/week, 2× T4 (16 GB), 9 h sessions, background execution | **Primary workhorse** — batch sims + final renders |
| **Google Colab** | T4, ~15–30 h/week (dynamic), 12 h sessions | Interactive experiments, second pool |
| **Lightning AI Studios** | ~22 GPU-h/month | Overflow / persistent-env convenience |
| Local Mac (Metal) | unlimited | Scene design + low-res physics iteration (Genesis supports Apple Metal backend) |

Rotating Kaggle + Colab gives ~50+ free GPU hours/week — far more than enough,
since simulation+render of a 10 s clip at 1080p takes minutes on a T4, not hours.

## Workflow

1. **Design locally** (Mac, `backend=gs.cpu` or `gs.metal`, low res, viewer on)
   — iterate on scene layout, materials, camera moves cheaply.
2. **Push to GitHub** — the cloud notebook clones this repo.
3. **Render in the cloud** (Kaggle/Colab, `backend=gs.gpu`, headless, 1080p+)
   — run `notebooks/cloud_gpu_runner.ipynb`, download MP4s from outputs.
4. **Post-process** locally with ffmpeg / your editor.

## Phases

### Phase 0 — Hello physics (you are here)
- [x] Project scaffold
- [ ] `pip install genesis-world` locally, run `sims/dominoes.py` (CPU is fine)
- [ ] Verify an MP4 lands in `out/`

### Phase 1 — Cloud render loop
- [ ] Create GitHub repo, push
- [ ] Run `notebooks/cloud_gpu_runner.ipynb` on Kaggle (Settings → Accelerator → GPU T4 ×2)
- [ ] Render `sims/water_splash.py` at 1080p on the T4

### Phase 2 — Content-quality visuals
- [x] Mesh-based humanoid (Unitree G1 from MuJoCo Menagerie) instead of capsules
- [x] Sim exporter: water particles + link poses per frame (`--export` flag)
- [x] Blender import script: rebuilds robot from STLs, liquid surface via
      geometry nodes, water/metal shaders, Cycles GPU render
      (`blender/import_swim_g1.py` — first full test runs on Kaggle)
- [ ] Run the Blender pipeline end-to-end on Kaggle (notebook Phase 2 cells)
- [ ] Try Genesis's ray-traced camera (LuisaRender backend) for caustics
- [ ] HDRI environment lighting, depth of field, motion blur

### Phase 3 — Content pipeline at scale
- [ ] Parametrize scenes (object counts, materials, seeds) → batch-render
      variations in one Kaggle session
- [ ] Scene ideas: domino chains, water filling glass, sand/MPM pours,
      cloth tearing, soft-body drops, building collapse, mixed-material
      "satisfying" loops
- [ ] ffmpeg pipeline: loop-perfect cuts, 4K upscale, vertical crops for shorts

## macOS caveats
- Rendering needs an awake display (pyglet/Cocoa crashes with "list index out of
  range" if the screen sleeps). Wrap long renders in `caffeinate -dimsu python ...`.
- Genesis on Mac uses the Metal/CPU backend; the LuisaRender ray tracer and some
  solvers are CUDA-only → design locally, beauty-render in the cloud.
- Keep local particle counts low (e.g. `particle_size=0.02`); crank density on the T4.
