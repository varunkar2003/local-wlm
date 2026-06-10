"""SPH water block collapsing onto the floor -> MP4.

Local preview:  python sims/water_splash.py --particle-size 0.02
Cloud render:   python sims/water_splash.py --gpu --particle-size 0.008 --res 1920 1080
"""

import argparse
import os

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="use GPU backend (cloud)")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--particle-size", type=float, default=0.015)
parser.add_argument("--seconds", type=float, default=4.0)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--out", default="out/water_splash.mp4")
args = parser.parse_args()

gs.init(backend=gs.gpu if args.gpu else gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.6, -0.6, 0.0),
        upper_bound=(0.6, 0.6, 1.2),
        particle_size=args.particle_size,
    ),
    show_viewer=False,
)

scene.add_entity(gs.morphs.Plane())

scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(pos=(0.0, 0.0, 0.55), size=(0.35, 0.35, 0.5)),
    surface=gs.surfaces.Default(color=(0.35, 0.7, 1.0), vis_mode="particle"),
)

# an obstacle for the splash to interact with
scene.add_entity(
    gs.morphs.Sphere(pos=(0.0, 0.0, 0.12), radius=0.12, fixed=True),
    surface=gs.surfaces.Default(color=(0.95, 0.6, 0.15)),
)

cam = scene.add_camera(
    res=tuple(args.res),
    pos=(1.6, -1.6, 0.9),
    lookat=(0.0, 0.0, 0.25),
    fov=35,
    GUI=False,
)

scene.build()

steps_per_frame = max(1, round((1 / args.fps) / 4e-3))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
cam.start_recording()
for _ in range(int(args.seconds * args.fps)):
    for _ in range(steps_per_frame):
        scene.step()
    cam.render()
cam.stop_recording(save_to_filename=args.out, fps=args.fps)
print(f"saved {args.out}")
