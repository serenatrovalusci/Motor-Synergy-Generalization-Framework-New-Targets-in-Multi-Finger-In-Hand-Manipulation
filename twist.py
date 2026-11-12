#!/usr/bin/env python3
"""Twist task: tighten a yaw-only cube by goal_deg with the Shadow Hand fingers.

  python twist.py selftest
  python twist.py --task train --env-id HandTwistCube-v0 [any sac_her_pipeline.py option]
"""
import os
import sys
import time

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML = os.path.join(ROOT, "demo", "assets", "hand", "manipulate_block_touch_sensors_ws.xml")
ENV_ID = "HandTwistCube-v0"


class TwistEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, xml_path=DEFAULT_XML, goal_deg=360.0, tighten_sign=-1, lock_wrist=True,
                 random_yaw=True, n_settle=10, render_mode=None):
        super().__init__()
        m = self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(m)
        self.render_mode = render_mode
        self.n_sub = max(1, int(round(0.04 / m.opt.timestep)))  # one env step = 0.04 s
        self.goal = np.radians(goal_deg)
        self.sign, self.lock_wrist = tighten_sign, lock_wrist
        self.random_yaw, self.n_settle = random_yaw, n_settle

        # cube hinge and hand joints
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "object:yaw")
        if jid < 0:
            raise ValueError("no 'object:yaw' joint: regenerate with make_workspace.py (without --free-cube)")
        self.yaw_q, self.yaw_v = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
        hand = [j for j in range(m.njnt)
                if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "").startswith("robot0:")
                and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        self.hand_q = np.array([m.jnt_qposadr[j] for j in hand])
        self.hand_v = np.array([m.jnt_dofadr[j] for j in hand])

        # touch sensors only (the XML also contains other sensors)
        touch = [i for i in range(m.nsensor) if m.sensor_type[i] == mujoco.mjtSensor.mjSENS_TOUCH]
        self.touch_idx = np.array([m.sensor_adr[i] + k for i in touch for k in range(m.sensor_dim[i])],
                                  dtype=int)
        self.n_touch = len(self.touch_idx)

        # actuators: 20 position actuators, absolute control around the mid-range
        self.nu = m.nu
        assert self.nu == 20, f"expected 20 actuators, found {self.nu}"
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(self.nu)]
        self.wrist = np.array([i for i, n in enumerate(names) if "WRJ" in n])
        self.lo, self.hi = m.actuator_ctrlrange[:, 0].copy(), m.actuator_ctrlrange[:, 1].copy()
        self.center, self.half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2

        dim = len(self.hand_q) * 2 + 3 + self.n_touch
        inf = np.inf
        self.action_space = spaces.Box(-1.0, 1.0, (self.nu,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(-inf, inf, (dim,), dtype=np.float32),
            "achieved_goal": spaces.Box(-inf, inf, (1,), dtype=np.float32),
            "desired_goal": spaces.Box(-inf, inf, (1,), dtype=np.float32),
        })
        self.yaw0, self.best = 0.0, 0.0
        self._renderer = self._viewer = None

    # ---- helpers ----
    def _angle(self):
        """Tightening angle in radians since reset (positive = tightening)."""
        return self.sign * (self.data.qpos[self.yaw_q] - self.yaw0)

    def _apply(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1, 1)
        ctrl = self.center + a * self.half
        if self.lock_wrist:
            ctrl[self.wrist] = 0.0
        self.data.ctrl[:] = np.clip(ctrl, self.lo, self.hi)
        for _ in range(self.n_sub):
            mujoco.mj_step(self.model, self.data)

    def _obs(self):
        d = self.data
        yaw = d.qpos[self.yaw_q]
        touch = np.log1p(np.maximum(d.sensordata[self.touch_idx], 0.0))
        vec = np.concatenate([d.qpos[self.hand_q], d.qvel[self.hand_v],
                              [np.sin(yaw), np.cos(yaw), d.qvel[self.yaw_v]], touch]).astype(np.float32)
        return {"observation": vec,
                "achieved_goal": np.array([self.best], dtype=np.float32),
                "desired_goal": np.array([self.goal], dtype=np.float32)}

    # ---- gym API ----
    def compute_reward(self, achieved_goal, desired_goal, info=None):
        ach, des = np.asarray(achieved_goal), np.asarray(desired_goal)
        return np.where(ach[..., 0] >= des[..., 0] - 1e-6, 0.0, -1.0).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        if self.random_yaw:
            self.data.qpos[self.yaw_q] = self.np_random.uniform(-np.pi, np.pi)
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self.n_settle):
            self._apply(np.zeros(self.nu))
        self.yaw0, self.best = float(self.data.qpos[self.yaw_q]), 0.0
        return self._obs(), {"is_success": 0.0}

    def step(self, action):
        self._apply(action)
        ang = self._angle()
        self.best = max(self.best, ang)
        obs = self._obs()
        reward = float(self.compute_reward(obs["achieved_goal"], obs["desired_goal"]))
        info = {"is_success": float(self.best >= self.goal),
                "angle_deg": float(np.degrees(ang)), "best_deg": float(np.degrees(self.best))}
        return obs, reward, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, 480, 640)
                self._cam = mujoco.MjvCamera()
                self._cam.lookat[:] = [1.0, 0.87, 0.05]
                self._cam.distance, self._cam.azimuth, self._cam.elevation = 0.55, 135, -30
            self._renderer.update_scene(self.data, camera=self._cam)
            return self._renderer.render()
        if self.render_mode == "human":
            import mujoco.viewer
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
        if self._viewer is not None:
            self._viewer.close()


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point=TwistEnv, max_episode_steps=150,
                 kwargs={"goal_deg": float(os.environ.get("TWIST_GOAL_DEG", "360"))})


def selftest():
    env = gym.make(ENV_ID)
    base = env.unwrapped
    obs, _ = env.reset(seed=0)
    print("observation:", obs["observation"].shape, "| action space:", env.action_space.shape)
    print(f"sensors in the XML: {base.model.nsensor}, touch sensors used: {base.n_touch}")
    t0, n = time.time(), 0
    for ep in range(3):
        env.reset(seed=ep)
        done = False
        while not done:
            _, r, term, trunc, info = env.step(env.action_space.sample())
            done, n = term or trunc, n + 1
        print(f"random actions, episode {ep}: best={info['best_deg']:.0f} deg, "
              f"final={info['angle_deg']:+.0f} deg")
    print(f"{n / (time.time() - t0):.0f} env steps per second")
    env.close()

    path = os.path.join(ROOT, "workspace", "best_motion.npy")
    if not os.path.exists(path):
        print("no workspace/best_motion.npy, skipping the replay check")
        return
    x = np.load(path)
    fing = [i for i in range(base.nu) if i not in base.wrist]
    nf = len(fing)
    if x.size != 1 + 3 * nf:
        print("best_motion.npy has an unexpected size, skipping the replay check")
        return
    f, b, A, ph = x[0], x[1:1 + nf], x[1 + nf:1 + 2 * nf], x[1 + 2 * nf:]
    for label, kw in (("with settle steps and random start", {}),
                      ("without settle steps (like rotate_test.py)", dict(n_settle=0, random_yaw=False))):
        e = gym.make(ENV_ID, **kw)
        e.reset(seed=0)
        at = {}
        for t in range(150):
            action = np.zeros(base.nu)
            action[fing] = np.clip(b + A * np.sin(2 * np.pi * f * t * 0.04 + ph), -1, 1)
            _, _, _, _, info = e.step(action)
            at[t + 1] = info["angle_deg"]
        print(f"replay {label}: {at[100]:+.0f} deg after 100 steps, {at[150]:+.0f} deg after 150, "
              f"best {info['best_deg']:.0f} deg")
        e.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        selftest()
    else:
        import sac_her_pipeline
        sac_her_pipeline.main()