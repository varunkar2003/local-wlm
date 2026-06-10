"""Import a Genesis swim export into Blender and render it with Cycles.

This is the "simulate ugly, render beautiful" half of the pipeline:
- rebuilds the Unitree G1 from its visual STL meshes (parsed from the MJCF)
- animates each link from the exported trajectory (robot_traj.npz)
- turns the raw SPH particles into a smooth liquid surface
  (Geometry Nodes: Points -> Volume -> Mesh) with a glass/water shader
- sets up camera, lighting, and renders the animation on GPU if available

Usage (headless):
    blender -b -P blender/import_swim_g1.py -- \
        --export-dir export/swim --out out/blender/ --render

Omit --render to just build the .blend for interactive tweaking:
    blender -P blender/import_swim_g1.py -- --export-dir export/swim
"""

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
parser = argparse.ArgumentParser()
parser.add_argument("--export-dir", default="export/swim")
parser.add_argument("--out", default="out/blender/")
parser.add_argument("--render", action="store_true")
parser.add_argument("--samples", type=int, default=64)
parser.add_argument("--res", type=int, nargs=2, default=[1920, 1080])
args = parser.parse_args(argv)

EXPORT = PROJECT_ROOT / args.export_dir
meta = json.loads((EXPORT / "meta.json").read_text())
traj = np.load(EXPORT / "robot_traj.npz")
MJCF = PROJECT_ROOT / meta["mjcf"]
N_FRAMES = meta["n_frames"]

# ---------------------------------------------------------------- scene reset
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.frame_start = 1
scene.frame_end = N_FRAMES
scene.render.fps = meta["fps"]

# ------------------------------------------------------------- parse the MJCF
# map: body name -> list of visual mesh names; mesh name -> STL file
tree = ET.parse(MJCF)
root = tree.getroot()
meshdir = root.find("compiler").get("meshdir", "")
mesh_files = {
    (m.get("name") or Path(m.get("file")).stem): MJCF.parent / meshdir / m.get("file")
    for m in root.iter("mesh")
}
body_meshes = {}
for body in root.iter("body"):
    visuals = [
        g.get("mesh")
        for g in body.findall("geom")
        if g.get("class") == "visual" and g.get("mesh")
    ]
    if visuals:
        body_meshes[body.get("name")] = visuals

# ------------------------------------------------------------- build the robot
robot_mat = bpy.data.materials.new("RobotBody")
robot_mat.use_nodes = True
bsdf = robot_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.85, 0.86, 0.88, 1.0)
bsdf.inputs["Metallic"].default_value = 0.7
bsdf.inputs["Roughness"].default_value = 0.35

link_names = meta["link_names"]
link_empties = {}
for name in link_names:
    empty = bpy.data.objects.new(f"link_{name}", None)
    scene.collection.objects.link(empty)
    empty.rotation_mode = "QUATERNION"
    link_empties[name] = empty
    for mesh_name in body_meshes.get(name, []):
        stl = mesh_files[mesh_name]
        if not stl.exists():
            continue
        before = set(bpy.data.objects)
        if hasattr(bpy.ops.wm, "stl_import"):  # Blender >= 4.1
            bpy.ops.wm.stl_import(filepath=str(stl))
        else:
            bpy.ops.import_mesh.stl(filepath=str(stl))
        for obj in set(bpy.data.objects) - before:
            obj.parent = empty
            obj.data.materials.clear()
            obj.data.materials.append(robot_mat)
            for poly in obj.data.polygons:
                poly.use_smooth = True

# keyframe link transforms (Genesis and Blender both use wxyz quaternions)
pos, quat = traj["pos"], traj["quat"]
for li, name in enumerate(link_names):
    empty = link_empties[name]
    for f in range(N_FRAMES):
        empty.location = pos[f, li]
        empty.rotation_quaternion = quat[f, li]
        empty.keyframe_insert("location", frame=f + 1)
        empty.keyframe_insert("rotation_quaternion", frame=f + 1)

# ------------------------------------------------------------------ the water
# one mesh whose vertices are the SPH particles, swapped per frame by a
# frame-change handler; Geometry Nodes reconstructs a smooth liquid surface
first = np.load(EXPORT / "water" / "frame_0000.npy")
n_particles = len(first)
water_mesh = bpy.data.meshes.new("WaterParticles")
water_mesh.from_pydata([tuple(p) for p in first], [], [])
water_obj = bpy.data.objects.new("Water", water_mesh)
scene.collection.objects.link(water_obj)

_water_cache = {}


def _set_water_frame(scene_, _depsgraph=None):
    f = min(max(scene_.frame_current - 1, 0), N_FRAMES - 1)
    if f not in _water_cache:
        _water_cache.clear()  # keep memory bounded
        _water_cache[f] = np.load(EXPORT / "water" / f"frame_{f:04d}.npy")
    pts = _water_cache[f]
    water_mesh.vertices.foreach_set("co", pts.ravel())
    water_mesh.update()


bpy.app.handlers.frame_change_pre.append(_set_water_frame)

# geometry nodes: points -> volume -> mesh
gn = water_obj.modifiers.new("LiquidSurface", "NODES")
ng = bpy.data.node_groups.new("LiquidSurface", "GeometryNodeTree")
gn.node_group = ng
ng.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
ng.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
n_in = ng.nodes.new("NodeGroupInput")
n_out = ng.nodes.new("NodeGroupOutput")
m2p = ng.nodes.new("GeometryNodeMeshToPoints")
p2v = ng.nodes.new("GeometryNodePointsToVolume")
v2m = ng.nodes.new("GeometryNodeVolumeToMesh")
smooth = ng.nodes.new("GeometryNodeSetShadeSmooth")
r = meta["particle_size"]
p2v.inputs["Radius"].default_value = r * 1.6
p2v.inputs["Voxel Size"].default_value = r * 0.9
v2m.inputs["Voxel Size"].default_value = r * 0.9
ng.links.new(n_in.outputs[0], m2p.inputs["Mesh"])
ng.links.new(m2p.outputs["Points"], p2v.inputs["Points"])
ng.links.new(p2v.outputs["Volume"], v2m.inputs["Volume"])
ng.links.new(v2m.outputs["Mesh"], smooth.inputs["Geometry"])
ng.links.new(smooth.outputs["Geometry"], n_out.inputs[0])

water_mat = bpy.data.materials.new("Water")
water_mat.use_nodes = True
wb = water_mat.node_tree.nodes["Principled BSDF"]
wb.inputs["Base Color"].default_value = (0.6, 0.85, 0.95, 1.0)
wb.inputs["Transmission Weight"].default_value = 1.0
wb.inputs["Roughness"].default_value = 0.02
wb.inputs["IOR"].default_value = 1.33
water_obj.data.materials.append(water_mat)

# --------------------------------------------------- floor, light, camera, sky
floor = bpy.data.meshes.new("Floor")
floor.from_pydata(
    [(-20, -20, -0.001), (20, -20, -0.001), (20, 20, -0.001), (-20, 20, -0.001)],
    [],
    [(0, 1, 2, 3)],
)
floor_obj = bpy.data.objects.new("Floor", floor)
scene.collection.objects.link(floor_obj)
floor_mat = bpy.data.materials.new("Floor")
floor_mat.use_nodes = True
floor_mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (
    0.08, 0.1, 0.12, 1.0,
)
floor.materials.append(floor_mat)

sun = bpy.data.objects.new("Sun", bpy.data.lights.new("Sun", "SUN"))
sun.data.energy = 4.0
sun.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(sun)

world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.5, 0.7, 0.95, 1)
world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.8

cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
cam.location = (1.3, -2.4, 1.0)
scene.collection.objects.link(cam)
scene.camera = cam
look = bpy.data.objects.new("CamTarget", None)
look.location = (0.5, 0.0, 0.25)
scene.collection.objects.link(look)
track = cam.constraints.new("TRACK_TO")
track.target = look

# --------------------------------------------------------------------- render
scene.render.engine = "CYCLES"
scene.cycles.samples = args.samples
scene.render.resolution_x, scene.render.resolution_y = args.res
prefs = bpy.context.preferences.addons.get("cycles")
if prefs:  # enable any GPU present (CUDA on Kaggle/Colab, Metal locally)
    for backend in ("OPTIX", "CUDA", "METAL"):
        try:
            prefs.preferences.compute_device_type = backend
            prefs.preferences.get_devices()
            for d in prefs.preferences.devices:
                d.use = True
            scene.cycles.device = "GPU"
            break
        except Exception:
            continue

out_dir = PROJECT_ROOT / args.out
out_dir.mkdir(parents=True, exist_ok=True)
scene.render.filepath = str(out_dir / "frame_")
scene.render.image_settings.file_format = "PNG"

if args.render:
    bpy.ops.render.render(animation=True)
    print(f"rendered {N_FRAMES} frames to {out_dir}/")
    print(f"encode with: ffmpeg -framerate {meta['fps']} -i {out_dir}/frame_%04d.png "
          f"-c:v libx264 -pix_fmt yuv420p out/swimmer_blender.mp4")
else:
    blend = out_dir / "swim.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend))
    print(f"saved {blend} — open in Blender to tweak, or rerun with --render")
