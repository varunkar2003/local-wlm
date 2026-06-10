"""A humanoid swimming through an SPH water pool -> MP4.

The humanoid is kinematically driven (scripted swim stroke + forward glide);
the water responds physically and splashes around the body.

Local preview:  python sims/swimmer.py --particle-size 0.05 --seconds 2
Cloud render:   python sims/swimmer.py --gpu --particle-size 0.015 --res 1920 1080
"""

import argparse
import math
import os

import numpy as np

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="use GPU backend (cloud)")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--particle-size", type=float, default=0.035)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--out", default="out/swimmer.mp4")
args = parser.parse_args()

DT = 4e-3
WATER_LEVEL = 0.36

gs.init(backend=gs.gpu if args.gpu else gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=DT, substeps=10),
    sph_options=gs.options.SPHOptions(
        lower_bound=(-0.9, -0.6, 0.0),
        upper_bound=(1.7, 0.6, 0.9),
        particle_size=args.particle_size,
    ),
    show_viewer=False,
)

scene.add_entity(gs.morphs.Plane())

# pool of water filling the SPH domain up to WATER_LEVEL
scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos=(0.4, 0.0, WATER_LEVEL / 2),
        size=(2.55, 1.15, WATER_LEVEL),
    ),
    surface=gs.surfaces.Default(color=(0.35, 0.7, 1.0), vis_mode="particle"),
)

# humanoid lying prone (face down), head pointing +x, at the water surface
humanoid = scene.add_entity(
    gs.morphs.MJCF(
        file="xml/humanoid.xml",
        pos=(0.0, 0.0, WATER_LEVEL - 0.03),
        euler=(0, 90, 0),
    ),
)

cam = scene.add_camera(
    res=tuple(args.res),
    pos=(1.3, -2.2, 1.0),
    lookat=(0.5, 0.0, 0.3),
    fov=40,
    GUI=False,
)

scene.build()


def dof_indices(joint_names):
    idx = []
    for name in joint_names:
        idx.extend(np.atleast_1d(humanoid.get_joint(name).dof_idx_local).tolist())
    return idx


root_dofs = dof_indices(["root"])  # 6 dofs: tx ty tz + rotation
limb_joints = [
    "hip_y_right", "hip_y_left",
    "knee_right", "knee_left",
    "shoulder1_right", "shoulder1_left",
    "shoulder2_right", "shoulder2_left",
    "elbow_right", "elbow_left",
]
limb_dofs = dof_indices(limb_joints)

# undriven joints must be held too, or gravity folds the body over time
all_joint_names = [j.name for j in humanoid.joints]
held_dofs = dof_indices(
    [n for n in all_joint_names if n != "root" and n not in limb_joints]
)
held_zeros = np.zeros(len(held_dofs))
n_dofs = len(root_dofs) + len(limb_dofs) + len(held_dofs)
all_dofs = root_dofs + limb_dofs + held_dofs
zero_vel = np.zeros(n_dofs)

root0 = np.array(humanoid.get_dofs_position(root_dofs))

GLIDE_SPEED = 0.15          # m/s forward
KICK_HZ = 1.4               # flutter-kick frequency
STROKE_HZ = 0.7             # arm-stroke frequency


def pose_at(t):
    """Scripted freestyle-ish stroke: alternating flutter kick + paddling arms."""
    k = 2 * math.pi * KICK_HZ * t
    s = 2 * math.pi * STROKE_HZ * t
    root = root0.copy()
    root[0] += GLIDE_SPEED * t                 # glide forward
    root[2] += 0.015 * math.sin(s * 2)         # gentle bob at the surface
    limbs = np.array([
        0.28 * math.sin(k),                    # hip_y_right  (flutter kick)
        -0.28 * math.sin(k),                   # hip_y_left   (anti-phase)
        -0.25 + 0.2 * math.sin(k + 1.0),       # knee_right
        -0.25 - 0.2 * math.sin(k + 1.0),       # knee_left
        -0.5 + 0.6 * math.sin(s),              # shoulder1_right
        -0.5 + 0.6 * math.sin(s + math.pi),    # shoulder1_left (alternating)
        0.4 * math.sin(s + 0.5),               # shoulder2_right
        -0.4 * math.sin(s + math.pi + 0.5),    # shoulder2_left
        -0.8 + 0.4 * math.sin(s + 1.2),        # elbow_right
        -0.8 + 0.4 * math.sin(s + math.pi + 1.2),  # elbow_left
    ])
    return root, limbs


steps_per_frame = max(1, round((1 / args.fps) / DT))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
cam.start_recording()

t = 0.0
for _ in range(int(args.seconds * args.fps)):
    for _ in range(steps_per_frame):
        root, limbs = pose_at(t)
        humanoid.set_dofs_position(root, root_dofs)
        humanoid.set_dofs_position(limbs, limb_dofs)
        humanoid.set_dofs_position(held_zeros, held_dofs)
        humanoid.set_dofs_velocity(zero_vel, all_dofs)
        scene.step()
        t += DT
    cam.render()

cam.stop_recording(save_to_filename=args.out, fps=args.fps)
print(f"saved {args.out}")
