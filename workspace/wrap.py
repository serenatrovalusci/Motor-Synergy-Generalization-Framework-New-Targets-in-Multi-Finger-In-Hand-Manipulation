"""Sweep palm height and x/y shift; score how often thumb + 2 fingers can touch the cube's SIDES."""
import re, sys
import numpy as np
import mujoco

xml = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 300
model = mujoco.MjModel.from_xml_path(xml)
data = mujoco.MjData(model)
rng = np.random.default_rng(0)
BODY, GEOM, JNT, SITE = (mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM,
                         mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE)
nm = lambda k, i: mujoco.mj_id2name(model, k, i) or ""

mount = next(i for i in range(model.nbody) if "hand mount" in nm(BODY, i))
palm = next(i for i in range(model.nbody) if "palm" in nm(BODY, i))
cube_b = mujoco.mj_name2id(model, BODY, "object")
cube_g = mujoco.mj_name2id(model, GEOM, "object")
table_g = mujoco.mj_name2id(model, GEOM, "table_top")
half = model.geom_size[cube_g][0]
fing = [j for j in range(model.njnt) if nm(JNT, j).startswith("robot0:")
        and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and "WRJ" not in nm(JNT, j)]
FING = ["ff", "mf", "rf", "lf", "th"]
tip_site = {}
for s in range(model.nsite):
    for f in FING:
        if re.search(rf"{f}tip$", nm(SITE, s).lower()) and f not in tip_site:
            tip_site[f] = s
use_sites = len(tip_site) == 5
dist_b = {f: next(i for i in range(model.nbody) if nm(BODY, i).endswith(f + "distal")) for f in FING}


def tips():
    if use_sites:
        return np.array([data.site_xpos[tip_site[f]] for f in FING])
    return np.array([data.xpos[dist_b[f]] + data.xmat[dist_b[f]].reshape(3, 3) @ np.array([0, 0, 0.026])
                     for f in FING])


def hits_table():
    for c in data.contact[:data.ncon]:
        if c.dist < -0.002 and table_g in (c.geom1, c.geom2):
            o = c.geom2 if c.geom1 == table_g else c.geom1
            if nm(BODY, model.geom_bodyid[o]).startswith("robot0:"):
                return True
    return False


lo = np.array([model.jnt_range[j][0] for j in fing])
hi = np.array([model.jnt_range[j][1] for j in fing])
Q = rng.uniform(lo, hi, size=(N, len(fing)))
base = model.body_pos[mount].copy()
z0 = None
heights = [6, 7, 8, 9, 10]
ys = [-14, -12, -10, -8, -6, -4, -2]
xs = [-4, -2, 0, 2, 4]
res = []
for h in heights:
    for dy in ys:
        for dx in xs:
            model.body_pos[mount] = base
            data.qpos[:] = model.qpos0
            mujoco.mj_forward(model, data)
            model.body_pos[mount] = base + np.array([dx / 100, dy / 100, h / 100 - data.xpos[palm][2]])
            ok = 0
            for q in Q:
                data.qpos[:] = model.qpos0
                for j, v in zip(fing, q):
                    data.qpos[model.jnt_qposadr[j]] = v
                mujoco.mj_forward(model, data)
                if hits_table():
                    continue
                t = tips()
                c = data.xpos[cube_b]
                d = np.linalg.norm(np.maximum(np.abs(t - c) - half, 0.0), axis=1)
                side = (d < 0.015) & (t[:, 2] < 2 * half - 0.003)   # beside the cube, not on top
                ok += bool(side[4] and side[:4].sum() >= 2)
            res.append((ok / N, h, dx, dy))
res.sort(reverse=True)
print(f"fingertips from {'sites' if use_sites else 'distal bodies (approx.)'}; {N} random finger poses per cell")
print("share of poses where thumb + at least 2 fingers touch the cube's sides:\n")
for r, h, dx, dy in res[:8]:
    print(f"  {100 * r:5.1f}%   palm {h:2d} cm   x shift {dx:+d} cm   y shift {dy:+d} cm")
r, h, dx, dy = res[0]
print(f'\ntry: python workspace/make_workspace.py --palm-height {h} '
      f'--mount-pos "{base[0] + dx / 100:.4f} {base[1] + dy / 100:.4f} 0.55" --mount-euler "1.5708 0 0"')
