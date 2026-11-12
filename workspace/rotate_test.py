"""Can the fingers rotate the yaw-only cube from this pose? Random search + CEM, open loop."""
import argparse
import numpy as np
import mujoco

p = argparse.ArgumentParser()
p.add_argument("xml")
p.add_argument("--seconds", type=float, default=4.0)
p.add_argument("--init", type=int, default=300)
p.add_argument("--gens", type=int, default=8)
p.add_argument("--pop", type=int, default=100)
p.add_argument("--video", default=None)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

rng = np.random.default_rng(args.seed)
model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)
ACT, JNT, BODY, GEOM = (mujoco.mjtObj.mjOBJ_ACTUATOR, mujoco.mjtObj.mjOBJ_JOINT,
                        mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM)

names = [mujoco.mj_id2name(model, ACT, i) or "" for i in range(model.nu)]
wrist = [i for i, n in enumerate(names) if "WRJ" in n]
fing = [i for i in range(model.nu) if i not in wrist]
print(f"{model.nu} actuators: {len(wrist)} wrist (held at 0), {len(fing)} finger")
lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
mid, half = (hi + lo) / 2, (hi - lo) / 2
ctrl_base = np.zeros(model.nu)
ctrl_base[wrist] = np.clip(0.0, lo[wrist], hi[wrist])

jid = mujoco.mj_name2id(model, JNT, "object:yaw")
assert jid >= 0, "no 'object:yaw' joint; regenerate the workspace without --free-cube"
qadr = model.jnt_qposadr[jid]
table_g = mujoco.mj_name2id(model, GEOM, "table_top")
robot_geom = np.array([(mujoco.mj_id2name(model, BODY, model.geom_bodyid[g]) or "").startswith("robot0:")
                       for g in range(model.ngeom)])

dt_ctrl = 0.04
n_sub = max(1, int(round(dt_ctrl / model.opt.timestep)))
n_ctrl = int(round(args.seconds / dt_ctrl))
nf = len(fing)


def ctrl_at(x, t):
    f, b, A, ph = x[0], x[1:1 + nf], x[1 + nf:1 + 2 * nf], x[1 + 2 * nf:]
    a = np.clip(b + A * np.sin(2 * np.pi * f * t + ph), -1, 1)
    c = ctrl_base.copy()
    c[fing] = mid[fing] + a * half[fing]
    return c


def rollout(x, frames=None, renderer=None, cam=None):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    y0, hit = data.qpos[qadr], False
    for k in range(n_ctrl):
        data.ctrl[:] = ctrl_at(x, k * dt_ctrl)
        for _ in range(n_sub):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            return 0.0, hit
        if table_g >= 0 and not hit:
            for c in data.contact[:data.ncon]:
                if table_g in (c.geom1, c.geom2):
                    o = c.geom2 if c.geom1 == table_g else c.geom1
                    if robot_geom[o]:
                        hit = True
                        break
        if frames is not None:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())
    return float(data.qpos[qadr] - y0), hit


def sample(mean, std, n):
    X = rng.normal(mean, std, size=(n, len(mean)))
    X[:, 0] = np.clip(X[:, 0], 0.3, 3.0)
    X[:, 1:1 + nf] = np.clip(X[:, 1:1 + nf], -1, 1)
    X[:, 1 + nf:1 + 2 * nf] = np.clip(X[:, 1 + nf:1 + 2 * nf], 0, 1)
    return X


mean = np.concatenate([[1.2], np.zeros(nf), np.full(nf, 0.5), np.full(nf, np.pi)])
std = np.concatenate([[0.6], np.full(nf, 0.5), np.full(nf, 0.3), np.full(nf, np.pi)])
X = sample(mean, std, args.init)
res = [rollout(x) for x in X]
d = -np.degrees([r[0] for r in res])   # positive = tightening (clockwise from above)
print(f"\nrandom motions ({args.init}): best {d.max():.0f} deg | median {np.median(d):.1f} deg | "
      f"over 10 deg: {100 * np.mean(d > 10):.0f}% | over 45 deg: {100 * np.mean(d > 45):.0f}% | "
      f"over 90 deg: {100 * np.mean(d > 90):.0f}%")
best_x, best_d, best_hit = X[d.argmax()], d.max(), res[int(d.argmax())][1]

elite_n = max(5, args.pop // 10)
pool, pool_d = X, d
for g in range(args.gens):
    order = np.argsort(-pool_d)[:elite_n]
    mean, std = pool[order].mean(0), pool[order].std(0) + 0.02
    X = sample(mean, std, args.pop)
    res = [rollout(x) for x in X]
    d = -np.degrees([r[0] for r in res])   # positive = tightening (clockwise from above)
    pool, pool_d = X, d
    if d.max() > best_d:
        best_x, best_d, best_hit = X[d.argmax()], d.max(), res[int(d.argmax())][1]
    print(f"round {g + 1}/{args.gens}: best so far {best_d:.0f} deg | this round median {np.median(d):.1f} deg")

print(f"\nBEST: {best_d:.0f} deg net rotation in {args.seconds:.0f} s "
      f"({'hand touched the table' if best_hit else 'no hand-table contact'})")
np.save("workspace/best_motion.npy", best_x)

if args.video:
    import imageio
    renderer = mujoco.Renderer(model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [1.0, 0.87, 0.05]
    cam.distance, cam.azimuth, cam.elevation = 0.55, 135, -30
    frames = []
    rollout(best_x, frames, renderer, cam)
    imageio.mimsave(args.video, frames, fps=25)
    print("saved", args.video)