"""Unitree G1 humanoid (full visual meshes) swimming through SPH water -> MP4.

Same kinematic-puppet approach as swimmer.py, but with a mesh-based humanoid
that looks like an actual figure instead of capsules.

Local preview:  python sims/swimmer_g1.py --particle-size 0.05 --seconds 2
Cloud render:   python sims/swimmer_g1.py --gpu --particle-size 0.015 --res 1920 1080
"""

import argparse
import math
import os
from pathlib import Path

import numpy as np

import genesis as gs

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true", help="use GPU backend (cloud)")
parser.add_argument("--res", type=int, nargs=2, default=[1280, 720])
parser.add_argument("--particle-size", type=float, default=0.035)
parser.add_argument("--seconds", type=float, default=5.0)
parser.add_argument("--fps", type=int, default=60)
parser.add_argument("--out", default="out/swimmer_g1.mp4")
parser.add_argument(
    "--export", metavar="DIR", default=None,
    help="instead of rendering, export water particles + link poses per frame "
         "for the Blender pipeline (see blender/import_swim_g1.py)",
)
args = parser.parse_args()

DT = 4e-3
WATER_LEVEL = 0.36
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

water = scene.add_entity(
    material=gs.materials.SPH.Liquid(),
    morph=gs.morphs.Box(
        pos=(0.4, 0.0, WATER_LEVEL / 2),
        size=(2.55, 1.15, WATER_LEVEL),
    ),
    surface=gs.surfaces.Default(color=(0.35, 0.7, 1.0), vis_mode="particle"),
)

# G1 lying prone (face down), head pointing +x, at the water surface
robot = scene.add_entity(
    gs.morphs.MJCF(
        file=str(PROJECT_ROOT / "assets/unitree_g1/g1.xml"),
        pos=(0.0, 0.0, WATER_LEVEL - 0.02),
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
        idx.extend(np.atleast_1d(robot.get_joint(name).dof_idx_local).tolist())
    return idx


root_dofs = dof_indices(["floating_base_joint"])  # 6 dofs
limb_joints = [
    "right_hip_pitch_joint", "left_hip_pitch_joint",
    "right_knee_joint", "left_knee_joint",
    "right_ankle_pitch_joint", "left_ankle_pitch_joint",
    "right_shoulder_pitch_joint", "left_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "left_shoulder_roll_joint",
    "right_elbow_joint", "left_elbow_joint",
    "waist_yaw_joint",
]
limb_dofs = dof_indices(limb_joints)

all_joint_names = [j.name for j in robot.joints]
held_dofs = dof_indices(
    [n for n in all_joint_names if n != "floating_base_joint" and n not in limb_joints]
)
held_zeros = np.zeros(len(held_dofs))
all_dofs = root_dofs + limb_dofs + held_dofs
zero_vel = np.zeros(len(all_dofs))

root0 = np.array(robot.get_dofs_position(root_dofs))

GLIDE_SPEED = 0.18
KICK_HZ = 1.8
STROKE_HZ = 0.8


def pose_at(t):
    """Freestyle-style crawl: full reach-to-hip arm pull (alternating),
    flutter kick with ankle whip, subtle torso sway. Angles in radians."""
    k = 2 * math.pi * KICK_HZ * t
    s = 2 * math.pi * STROKE_HZ * t
    root = root0.copy()
    root[0] += GLIDE_SPEED * t
    root[2] += 0.015 * math.sin(s * 2)
    limbs = np.array([
        0.25 * math.sin(k),                    # right_hip_pitch (flutter kick)
        -0.25 * math.sin(k),                   # left_hip_pitch  (anti-phase)
        0.12 + 0.18 * math.sin(k + 1.0),       # right_knee
        0.12 - 0.18 * math.sin(k + 1.0),       # left_knee
        -0.2 + 0.3 * math.sin(k + 1.8),        # right_ankle_pitch (whip)
        -0.2 - 0.3 * math.sin(k + 1.8),        # left_ankle_pitch
        -1.6 + 1.3 * math.sin(s),              # right_shoulder_pitch: reach (-2.9)
        -1.6 + 1.3 * math.sin(s + math.pi),    #   -> pull -> hand at hip (-0.3)
        -0.25,                                 # right_shoulder_roll (arm slightly out)
        0.25,                                  # left_shoulder_roll
        0.7 + 0.5 * math.sin(s + 0.9),         # right_elbow (bends during pull)
        0.7 + 0.5 * math.sin(s + math.pi + 0.9),  # left_elbow
        0.12 * math.sin(s),                    # waist_yaw (torso sway with stroke)
    ])
    return root, limbs


steps_per_frame = max(1, round((1 / args.fps) / DT))
n_frames = int(args.seconds * args.fps)

if args.export:
    os.makedirs(f"{args.export}/water", exist_ok=True)
    link_names = [link.name for link in robot.links]
    traj_pos, traj_quat = [], []
else:
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cam.start_recording()

t = 0.0
for frame in range(n_frames):
    for _ in range(steps_per_frame):
        root, limbs = pose_at(t)
        robot.set_dofs_position(root, root_dofs)
        robot.set_dofs_position(limbs, limb_dofs)
        robot.set_dofs_position(held_zeros, held_dofs)
        robot.set_dofs_velocity(zero_vel, all_dofs)
        scene.step()
        t += DT
    if args.export:
        particles = np.asarray(water.get_particles_pos(), dtype=np.float32)
        np.save(f"{args.export}/water/frame_{frame:04d}.npy", particles)
        traj_pos.append(np.asarray(robot.get_links_pos()))
        traj_quat.append(np.asarray(robot.get_links_quat()))
    else:
        cam.render()

if args.export:
    import json

    np.savez_compressed(
        f"{args.export}/robot_traj.npz",
        pos=np.stack(traj_pos),    # (frames, links, 3)
        quat=np.stack(traj_quat),  # (frames, links, 4) wxyz
    )
    with open(f"{args.export}/meta.json", "w") as f:
        json.dump(
            {
                "fps": args.fps,
                "n_frames": n_frames,
                "link_names": link_names,
                "particle_size": args.particle_size,
                "mjcf": "assets/unitree_g1/g1.xml",
            },
            f,
            indent=2,
        )
    print(f"exported {n_frames} frames to {args.export}/")
else:
    cam.stop_recording(save_to_filename=args.out, fps=args.fps)
    print(f"saved {args.out}")
