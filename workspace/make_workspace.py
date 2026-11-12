"""Generate a 'hand above a table in a factory' variant of the Shadow Hand block scene."""
import argparse, os, re, sys
import mujoco
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cube_utils import cube_model_path, ASSETS

ENV = "HandManipulateBlockRotateXYZ_ContinuousTouchSensors-v1"
STOCK_MOUNT = '<body name="robot0:hand mount" pos="1 1.25 0.15" euler="1.5708 0 3.14159">'
STOCK_OBJ = '<body name="object" pos="1 0.87 0.2">'
STOCK_TGT = '<body name="target" pos="1 0.87 0.2">'
CX, CY = 1.0, 0.87  # cube and table centre in the world frame

p = argparse.ArgumentParser()
p.add_argument("--cube-size", type=float, default=2.5, help="2.5 = stock block")
p.add_argument("--mount-pos", default="1 0.87 0.55", help="x y z of the forearm mount")
p.add_argument("--mount-euler", default="3.14159 0 0", help="3.14159 0 0 = fingers pointing down")
p.add_argument("--palm-height", type=float, default=None, help="cm from the table to the palm body")
p.add_argument("--table-height", type=float, default=75.0, help="cm from the factory floor to the table top")
p.add_argument("--plain", action="store_true", help="no table or factory (floor at the table height)")
p.add_argument("--free-cube", action="store_true", help="keep the stock free joint (cube can slide and tip)")
p.add_argument("--damping", type=float, default=0.01)
p.add_argument("--frictionloss", type=float, default=0.0, help="constant resisting torque on the cube hinge (N m)")
args = p.parse_args()

hand_dir = os.path.join(ASSETS, "hand")
half = args.cube_size / 100.0
scene_stock = open(cube_model_path(ENV, args.cube_size)).read()
robot_stock = open(os.path.join(hand_dir, "robot_touch_sensors_92.xml")).read()
assert "robot_touch_sensors_92.xml" in scene_stock and STOCK_MOUNT in robot_stock

SKY = ('<texture name="ws:sky" type="skybox" builtin="gradient" rgb1="0.30 0.32 0.36" '
       'rgb2="0.08 0.09 0.11" width="512" height="512"></texture>')
ASSET = ('<texture name="ws:floor_tex" type="2d" builtin="checker" rgb1="0.33 0.35 0.38" '
         'rgb2="0.27 0.29 0.32" width="256" height="256" mark="edge" markrgb="0.2 0.22 0.25"></texture>\n'
         '<material name="ws:floor_mat" texture="ws:floor_tex" texrepeat="12 12" reflectance="0.15"></material>\n')


def geom(name, kind, pos, size, rgba, collide=False, euler=None):
    extra = "" if collide else ' contype="0" conaffinity="0"'
    if euler:
        extra += f' euler="{euler}"'
    return (f'<geom name="{name}" type="{kind}" pos="{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f}" '
            f'size="{size}" rgba="{rgba}"{extra}></geom>\n')


def factory_xml(th):
    fz = -th  # factory floor height; the table top is at z = 0
    R, H = 2.2, 3.0  # half-width of the room, wall height
    g = []
    g.append(geom("table_top", "box", (CX, CY, -0.02), "0.45 0.35 0.02", "0.62 0.64 0.67 1", collide=True))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(geom(f"table_leg{i}", "box", (CX + sx * 0.41, CY + sy * 0.31, (fz - 0.04) / 2),
                      f"0.03 0.03 {(th - 0.04) / 2:.3f}", "0.18 0.19 0.21 1"))
    g.append(f'<geom name="factory_floor" type="plane" pos="{CX} {CY} {fz}" size="{R} {R} 0.1" '
             f'material="ws:floor_mat" condim="3"></geom>\n')
    wall, mid = "0.46 0.48 0.52 1", fz + H / 2
    g.append(geom("wall_back", "box", (CX, CY - R, mid), f"{R} 0.05 {H / 2}", wall))
    g.append(geom("wall_front", "box", (CX, CY + R, mid), f"{R} 0.05 {H / 2}", wall))
    g.append(geom("wall_left", "box", (CX - R, CY, mid), f"0.05 {R} {H / 2}", wall))
    g.append(geom("wall_right", "box", (CX + R, CY, mid), f"0.05 {R} {H / 2}", wall))
    yel = "0.95 0.75 0.05 1"
    g.append(geom("stripe_back", "box", (CX, CY - R + 0.06, fz + 0.15), f"{R} 0.01 0.12", yel))
    g.append(geom("stripe_front", "box", (CX, CY + R - 0.06, fz + 0.15), f"{R} 0.01 0.12", yel))
    g.append(geom("stripe_left", "box", (CX - R + 0.06, CY, fz + 0.15), f"0.01 {R} 0.12", yel))
    g.append(geom("stripe_right", "box", (CX + R - 0.06, CY, fz + 0.15), f"0.01 {R} 0.12", yel))
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        g.append(geom(f"column{i}", "cylinder", (CX + sx * (R - 0.25), CY + sy * (R - 0.25), mid),
                      f"0.15 {H / 2}", "0.25 0.30 0.40 1"))
    # two machines with control panels and indicator lights, against the back wall
    yb = CY - R + 0.55
    g.append(geom("machine1", "box", (CX - 1.2, yb, fz + 0.7), "0.5 0.5 0.7", "0.20 0.38 0.62 1"))
    g.append(geom("panel1", "box", (CX - 1.2, yb + 0.51, fz + 1.0), "0.25 0.01 0.18", "0.05 0.06 0.07 1"))
    g.append(geom("light1g", "cylinder", (CX - 1.35, yb, fz + 1.45), "0.04 0.04", "0.1 0.9 0.2 1"))
    g.append(geom("light1r", "cylinder", (CX - 1.05, yb, fz + 1.45), "0.04 0.04", "0.9 0.1 0.1 1"))
    g.append(geom("machine2", "box", (CX + 1.2, yb, fz + 0.6), "0.6 0.5 0.6", "0.85 0.35 0.10 1"))
    g.append(geom("panel2", "box", (CX + 1.2, yb + 0.51, fz + 0.9), "0.3 0.01 0.15", "0.05 0.06 0.07 1"))
    g.append(geom("pipe0", "cylinder", (CX, CY - R + 0.25, fz + 2.4), f"0.06 {R}", "0.6 0.62 0.65 1",
                  euler="0 1.5708 0"))
    # conveyor along the right wall, with boxes on it
    cx = CX + R - 0.6
    g.append(geom("conv_base", "box", (cx, CY, fz + 0.2), "0.25 1.3 0.2", "0.20 0.21 0.24 1"))
    g.append(geom("conv_belt", "box", (cx, CY, fz + 0.45), "0.3 1.4 0.05", "0.08 0.08 0.09 1"))
    g.append(geom("conv_rail0", "box", (cx - 0.32, CY, fz + 0.55), "0.02 1.4 0.1", yel))
    g.append(geom("conv_rail1", "box", (cx + 0.32, CY, fz + 0.55), "0.02 1.4 0.1", yel))
    for i, k in enumerate((-0.8, 0.0, 0.7)):
        g.append(geom(f"crate{i}", "box", (cx, CY + k, fz + 0.60), "0.1 0.1 0.1", "0.70 0.55 0.35 1"))
    # yellow safety line on the floor around the table
    for i, (px, py, sx, sy) in enumerate(((0, 0.7, 0.75, 0.015), (0, -0.7, 0.75, 0.015),
                                          (0.75, 0, 0.015, 0.7), (-0.75, 0, 0.015, 0.7))):
        g.append(geom(f"floor_line{i}", "box", (CX + px, CY + py, fz + 0.003), f"{sx} {sy} 0.003", yel))
    return "".join(g)


def add_factory(scene):
    th = args.table_height / 100.0
    for fname in ("shared.xml", "shared_asset.xml"):  # dark skybox, in our own copies only
        text = open(os.path.join(hand_dir, fname)).read()
        new, n = re.subn(r'<texture[^>]*type="skybox"[^>]*>(\s*</texture>)?', SKY, text)
        if n:
            ws = fname.replace(".xml", "_ws.xml")
            open(os.path.join(hand_dir, ws), "w").write(new)
            assert f'file="{fname}"' in scene
            scene = scene.replace(f'file="{fname}"', f'file="{ws}"')
            print(f"skybox replaced (found in {fname})")
    scene, n = re.subn(r'(<include file="shared_asset[^"]*"></include>)',
                       lambda m: m.group(1) + "\n" + ASSET, scene, count=1)
    assert n == 1, "asset include not found"
    assert re.search(r'<geom name="floor0"[^>]*></geom>', scene), "floor0 geom not found"
    scene = re.sub(r'<geom name="floor0"[^>]*></geom>', "@@FACTORY@@", scene)
    return scene.replace("@@FACTORY@@", factory_xml(th))


scene_base = scene_stock.replace("robot_touch_sensors_92.xml", "robot_touch_sensors_92_ws.xml")
for stock, name in ((STOCK_OBJ, "object"), (STOCK_TGT, "target")):
    assert stock in scene_base, f"{name} body not found"
    scene_base = scene_base.replace(stock, f'<body name="{name}" pos="1 0.87 {half:.4f}">')
if not args.free_cube:
    joint = (f'<joint name="object:yaw" type="hinge" axis="0 0 1" '
             f'damping="{args.damping}" frictionloss="{args.frictionloss}"></joint>')
    scene_base, n = re.subn(r'<joint name="object:joint"[^>]*>(\s*</joint>)?', joint, scene_base)
    assert n == 1, "object:joint not found in the scene XML"
    old = f'<body name="object" pos="1 0.87 {half:.4f}">'
    assert old in scene_base
    scene_base = scene_base.replace(old, f'<body name="object" pos="1 0.87 {half + 0.001:.4f}">')
if not args.plain:
    scene_base = add_factory(scene_base)


def write(mount_pos):
    out = os.path.join(hand_dir, "manipulate_block_touch_sensors_ws.xml")
    open(out, "w").write(scene_base)
    new = f'<body name="robot0:hand mount" pos="{mount_pos}" euler="{args.mount_euler}">'
    open(os.path.join(hand_dir, "robot_touch_sensors_92_ws.xml"), "w").write(
        robot_stock.replace(STOCK_MOUNT, new))
    return out


mp = [float(v) for v in args.mount_pos.split()]
out_scene = write(" ".join(f"{v:.4f}" for v in mp))
if args.palm_height is not None:
    m = mujoco.MjModel.from_xml_path(out_scene)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    pid = next((i for i in range(m.nbody)
                if "palm" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")), None)
    assert pid is not None, "no body with 'palm' in its name"
    mp[2] += args.palm_height / 100.0 - d.xpos[pid][2]
    out_scene = write(" ".join(f"{v:.4f}" for v in mp))
    print(f"palm body placed {args.palm_height} cm above the table (mount z = {mp[2]:.4f})")
print("wrote", out_scene)