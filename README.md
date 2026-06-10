# local-wlm

GPU physics simulation → 3D video pipeline, built on [Genesis](https://genesis-world.readthedocs.io/)
and rendered on free cloud GPUs (Kaggle / Colab). See [PLAN.md](PLAN.md) for the full roadmap.

## Quick start (local, Mac)

```bash
# Genesis needs Python 3.10+ (macOS system Python is 3.9 — use uv)
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
python sims/dominoes.py        # writes out/dominoes.mp4
```

## Cloud render (free GPU)

1. Push this repo to GitHub.
2. Upload `notebooks/cloud_gpu_runner.ipynb` to [Kaggle](https://www.kaggle.com/code)
   (Accelerator: GPU T4 ×2, ~30 free hrs/week) or Colab.
3. Set `REPO_URL` in the notebook and run all cells — 1080p MP4s appear in the output.

## Layout

- `sims/` — simulation scenes, each renders straight to MP4 (`--gpu` flag for cloud);
  `swimmer_g1.py` also takes `--export DIR` to dump frames for the Blender pipeline
- `assets/unitree_g1/` — mesh-based humanoid (MuJoCo Menagerie, Apache-2.0)
- `blender/` — import + Cycles beauty-render scripts (Phase 2)
- `notebooks/` — cloud GPU runner
- `out/`, `export/` — rendered videos and sim exports (gitignored)
