"""Open a scene XML in the MuJoCo viewer. Add --static to see the start pose without physics."""
import sys, time
import mujoco, mujoco.viewer

static = "--static" in sys.argv
model = mujoco.MjModel.from_xml_path(sys.argv[1])
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for i in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
    if name and any(k in name for k in ("palm", "distal", "object", "mount")):
        x, y, z = data.xpos[i]
        print(f"{name:30s} x={x:.3f} y={y:.3f} z={z:.3f}")
print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])

with mujoco.viewer.launch_passive(model, data) as v:
    v.cam.lookat[:] = [1.0, 0.87, 0.10]
    v.cam.distance = 0.9
    v.cam.azimuth = 150
    v.cam.elevation = -20
    while v.is_running():
        if not static:
            for _ in range(8):
                mujoco.mj_step(model, data)
        v.sync()
        time.sleep(0.016)