"""在 0.195 m 完整钢梯场景中短时运行规则 FSM 教师，观察前足能否上台。

不训练、不长跑：默认 6 s 仿真（300 个 50 Hz 控制步），只记录状态链、
四足最高点、是否真正踩到第一级踏板顶面，以及 IK 裁剪次数。
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import fsm_stairs_controller as fsm  # noqa: E402

SCENE_NAME = sys.argv[1] if len(sys.argv) > 1 else "full"
DURATION_S = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0

model, keyframe_id, scene_path = fsm.load_scene(SCENE_NAME)
data = mujoco.MjData(model)
controller = fsm.RuleFsmController(model, keyframe_id, SCENE_NAME)
controller.reset(data)

step_geom = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "up_step_1"
)
step_top = float(model.geom_pos[step_geom, 2] + model.geom_size[step_geom, 2])
step_x_min = float(model.geom_pos[step_geom, 0] - model.geom_size[step_geom, 0])
step_x_max = float(model.geom_pos[step_geom, 0] + model.geom_size[step_geom, 0])

physics_dt = float(model.opt.timestep)
substeps = max(1, int(round(fsm.CONTROL_DT_S / physics_dt)))
previous_target = controller._target_joint_rad.copy()
foot_ids = controller.foot_geom_ids
states: list[str] = []
max_foot_z = np.full(4, -1e9)
best_front_on_top = np.zeros(4, dtype=bool)
max_base_z = -1e9
max_base_x = -1e9
max_abs_pitch = 0.0
max_abs_roll = 0.0
min_base_z = 1e9
failure_info = None

n_control = int(round(DURATION_S / fsm.CONTROL_DT_S))
for step in range(n_control):
    controller.update_control_state(data)
    target = controller._target_joint_rad.copy()
    for sub in range(substeps):
        progress = (sub + 1) / float(substeps)
        fsm.write_motor_command(
            controller, data, fsm.cosine_blend(previous_target, target, progress)
        )
        mujoco.mj_step(model, data)
    previous_target = target
    if not states or states[-1] != controller.state.value:
        states.append(controller.state.value)
    feet = np.asarray(data.geom_xpos[foot_ids], dtype=np.float64)
    max_foot_z = np.maximum(max_foot_z, feet[:, 2])
    on_top = (
        (feet[:, 0] > step_x_min)
        & (feet[:, 0] < step_x_max)
        & (np.abs(feet[:, 2] - (step_top + fsm.FOOT_RADIUS_M)) < 0.03)
    )
    best_front_on_top |= on_top
    max_base_z = max(max_base_z, float(data.qpos[2]))
    max_base_x = max(max_base_x, float(data.qpos[0]))
    pitch, roll = fsm.quaternion_to_pitch_roll(data.qpos[3:7])
    max_abs_pitch = max(max_abs_pitch, abs(pitch))
    max_abs_roll = max(max_abs_roll, abs(roll))
    min_base_z = min(min_base_z, float(data.qpos[2]))
    if controller.state == fsm.FsmState.FAILED:
        failure_info = {
            "step": step,
            "t": float(data.time),
            "event": str(getattr(controller, "_quasi_event_name", "")),
            "base": np.round(np.asarray(data.qpos[:3]), 4).tolist(),
            "pitch_deg": round(float(np.degrees(pitch)), 1),
            "roll_deg": round(float(np.degrees(roll)), 1),
            "ik_clip": int(controller.metrics.ik_clip_count),
            "ik_max": round(float(controller.metrics.ik_clip_max_rad), 4),
        }
        break

print("SCENE=%s step_top=%.4f x=[%.4f, %.4f] ran=%.2fs" % (
    SCENE_NAME, step_top, step_x_min, step_x_max, float(data.time)))
print("STATES=" + " -> ".join(states))
print("MAX_FOOT_Z(m)=" + " ".join("%.3f" % v for v in max_foot_z))
print("ON_TOP_ONCE=" + " ".join(str(bool(v)) for v in best_front_on_top))
print("MAX_BASE_Z=%.3f MAX_BASE_X=%.3f IK_CLIP=%d IK_MAX=%.3f" % (
    max_base_z, max_base_x, controller.metrics.ik_clip_count,
    controller.metrics.ik_clip_max_rad))
print("MAX_PITCH_DEG=%.1f MAX_ROLL_DEG=%.1f MIN_BASE_Z=%.3f" % (
    np.degrees(max_abs_pitch), np.degrees(max_abs_roll), min_base_z))
print("FAILURE=%s" % failure_info)
