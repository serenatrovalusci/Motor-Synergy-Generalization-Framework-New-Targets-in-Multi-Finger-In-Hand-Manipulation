"""Play or record a Shadow Hand agent. Optional: change the cube size."""
import argparse, os, re, shutil, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
try:
    import sac_her_pipeline  # also loads mujoco_compat
except Exception as e:
    print("note: could not import sac_her_pipeline:", e)


def cube_model_path(env_id, size_cm):
    """Copy the block XML with the cube scaled; return its absolute path."""
    assert "Block" in env_id, "--cube-size only works with the Block environments"
    half = size_cm / 100.0  # README 'cm' = MuJoCo half-size (2.5 -> 0.025)
    pkg = os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets")
    out = os.path.join(HERE, "assets")
    os.makedirs(out, exist_ok=True)
    if not os.path.isdir(os.path.join(out, "hand")):
        shutil.copytree(os.path.join(pkg, "hand"), os.path.join(out, "hand"))
    for sub in ("stls", "textures"):
        src, link = os.path.join(pkg, sub), os.path.join(out, sub)
        assert os.path.isdir(src), f"{src} not found; assets are: {os.listdir(pkg)}"
        if not os.path.lexists(link):
            os.symlink(src, link)
    base = "manipulate_block_touch_sensors.xml" if "TouchSensors" in env_id else "manipulate_block.xml"
    text = open(os.path.join(pkg, "hand", base)).read()
    for name, h in (("object", half), ("object_hidden", half - 0.001), ("target", half)):
        pat = re.compile(r'(<geom name="%s" type="box" size=")[^"]*(")' % name)
        text, n = pat.subn(lambda m: m.group(1) + " ".join([f"{h:.4f}"] * 3) + m.group(2), text)
        assert n == 1, f"could not find geom {name} in {base}"
    path = os.path.join(out, "hand", base.replace(".xml", f"_cube{size_cm:g}.xml"))
    open(path, "w").write(text)
    return path

def patch_block_xml(xml_path, env_id):
    """Make the block env load our XML: replace its module-level *_XML constant."""
    import importlib
    for mod in ("manipulate_block", "manipulate_block_touch_sensors"):
        importlib.import_module(f"gymnasium_robotics.envs.shadow_dexterous_hand.{mod}")
    base = "manipulate_block_touch_sensors.xml" if "TouchSensors" in env_id else "manipulate_block.xml"
    patched, seen = 0, []
    for name, m in list(sys.modules.items()):
        if not name.startswith("gymnasium_robotics.envs.shadow_dexterous_hand"):
            continue
        for attr in dir(m):
            val = getattr(m, attr, None)
            if attr.endswith("XML") and isinstance(val, str):
                seen.append(f"{name}.{attr} = {val}")
                if os.path.basename(val) == base:
                    setattr(m, attr, xml_path)
                    patched += 1
    assert patched, "no XML constant found to patch. Candidates:\n" + "\n".join(seen)
    print(f"patched {patched} XML constant(s)")

p = argparse.ArgumentParser()
p.add_argument("--env-id", default="HandManipulateEggRotate_ContinuousTouchSensors-v1")
p.add_argument("--model-dir", default=None, help="folder with best_model.zip + vecnorm_best.pkl; omit for random actions")
p.add_argument("--cube-size", type=float, default=None, help="cube size as in the README: 2.5 = stock, 3.0 = large, 2.0 = small")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--video", default=None, help="save an mp4 here instead of opening a window")
p.add_argument("--no-render", action="store_true", help="no window and no video (fast evaluation)")
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

kw = {}
if args.cube_size is not None:
    xml_path = cube_model_path(args.env_id, args.cube_size)
    patch_block_xml(xml_path, args.env_id)
    print("using cube XML:", xml_path)


def make(render_mode=None):
    return gym.make(args.env_id, render_mode=render_mode, **kw)


mode = "rgb_array" if args.video else (None if args.no_render else "human")
env = make(mode)
import mujoco
_m = env.unwrapped.model
_g = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_GEOM, "object")
print("object half-size:", _m.geom_size[_g], " mass (kg):", _m.body_mass[_m.geom_bodyid[_g]])

model = norm = None
if args.model_dir:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    dummy = DummyVecEnv([lambda: make()])
    norm = VecNormalize.load(os.path.join(args.model_dir, "vecnorm_best.pkl"), dummy)
    norm.training, norm.norm_reward = False, False
    model = SAC.load(os.path.join(args.model_dir, "best_model.zip"), env=norm, device="cpu")
    print("agent action space:", model.action_space.shape)

frames, wins = [], 0
for ep in range(args.episodes):
    obs, info = env.reset(seed=args.seed + ep)
    done, steps = False, 0
    while not done:
        if model is not None:
            action, _ = model.predict(norm.normalize_obs(obs), deterministic=True)
        else:
            action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
        steps += 1
        if args.video:
            frames.append(env.render())
        elif mode == "human":
            time.sleep(0.03)
    ok = bool(info.get("is_success", False))
    wins += ok
    print(f"episode {ep+1}: steps={steps} success={ok}")
print(f"successes: {wins}/{args.episodes}  ({100*wins/args.episodes:.0f}%)")

if args.video and frames:
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(args.video)), exist_ok=True)
    imageio.mimsave(args.video, frames, fps=25)
    print("saved", args.video)
env.close()