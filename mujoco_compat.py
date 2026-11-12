"""
Compatibility patch for gymnasium_robotics on Python 3.14.

gymnasium_robotics.utils.mujoco_utils.get_joint_qpos/get_joint_qvel assert
`joint_type in (mjJNT_HINGE, mjJNT_SLIDE)`, which fails with recent mujoco
bindings because the joint type comes back as a plain int. Importing this
module replaces both functions with versions that compare ints explicitly.
Import it before creating any Shadow Hand environment.
"""

import numpy as np
import mujoco
import gymnasium_robotics.utils.mujoco_utils as mujoco_utils

def safe_get_joint_qpos(model, data, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        return np.array([])
    
    jnt_type = int(model.jnt_type[joint_id])
    qpos_adr = model.jnt_qposadr[joint_id]

    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return data.qpos[qpos_adr : qpos_adr + 7]
    elif jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return data.qpos[qpos_adr : qpos_adr + 4]
    else:
        return np.array([data.qpos[qpos_adr]])

def safe_get_joint_qvel(model, data, name):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id == -1:
        return np.array([])

    jnt_type = int(model.jnt_type[joint_id])
    dof_adr = model.jnt_dofadr[joint_id]

    if jnt_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return data.qvel[dof_adr : dof_adr + 6]
    elif jnt_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return data.qvel[dof_adr : dof_adr + 3]
    else:
        return np.array([data.qvel[dof_adr]])

# Override gymnasium_robotics functions completely
mujoco_utils.get_joint_qpos = safe_get_joint_qpos
mujoco_utils.get_joint_qvel = safe_get_joint_qvel
