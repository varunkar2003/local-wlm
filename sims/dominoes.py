"""Rigid-body domino chain -> MP4.

Local preview:  python sims/dominoes.py            (CPU/Metal, 720p)
Cloud render:   python sims/dominoes.py --gpu --res 1920 1080
"""

import argparse
import os

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="use GPU backend (cloud)")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--seconds", type=float, default=6.0)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--out", default="out/dominoes.mp4")
args = parser.parse_args()

gs.init(backend=gs.gpu if args.gpu else gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1 / args.fps, substeps=4),
    show_viewer=False,
)

scene.add_entity(gs.morphs.Plane())

N = 14
SPACING = 0.09
SIZE = (0.02, 0.06, 0.12)  # thin, wide, tall
for i in range(N):
    scene.add_entity(
        gs.morphs.Box(
            pos=(i * SPACING, 0.0, SIZE[2] / 2 + 0.001),
            size=SIZE,
            # first domino starts tilted past its tipping point
            euler=(0, 18, 0) if i == 0 else (0, 0, 0),
        ),
        surface=gs.surfaces.Default(color=(0.9, 0.25 + 0.05 * (i % 3), 0.2)),
    )

cam = scene.add_camera(
    res=tuple(args.res),
    pos=(N * SPACING / 2, -1.6, 0.7),
    lookat=(N * SPACING / 2, 0.0, 0.1),
    fov=35,
    GUI=False,
)

scene.build()

os.makedirs(os.path.dirname(args.out), exist_ok=True)
cam.start_recording()
for _ in range(int(args.seconds * args.fps)):
    scene.step()
    cam.render()
cam.stop_recording(save_to_filename=args.out, fps=args.fps)
print(f"saved {args.out}")
