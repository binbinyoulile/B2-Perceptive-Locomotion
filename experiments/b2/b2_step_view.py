"""B2 跨级姿态实时预览：MuJoCo 3D 窗口，可拖拽/缩放，检查机头与护板是否干涉。

用法（在仓库根目录）：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_step_view.py

窗口里：
    - 左键拖拽 = 旋转视角，滚轮 = 缩放，右键拖拽 = 平移；
    - 机器人用 ``B2_CROSS_STEP`` 的目标角 + PD 保持，落地后会稳定在跨级站姿；
    - 关闭窗口即退出。

说明：
    姿态来自 ``configs/robot_config.B2_CROSS_STEP``（由 b2_step_pose.py 搜索得到）。
    这里不做步态，只做静态保持，方便肉眼确认：
      1) 机头/雷达与二级踏板有没有留空；
      2) 前腿小腿护板会不会顶到一级踏面的前沿；
      3) 后足在平地上有没有多余滑移。
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_CROSS_STEP, B2_ROBOT  # noqa: E402

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

SCENE = ROOT / "simulations" / "mujoco" / "b2_stairs_static" / "scene.xml"
LEGS = ("FL", "FR", "RL", "RR")


def main() -> int:
    import os

    os.chdir(SCENE.parent)
    model = mujoco.MjModel.from_xml_path(SCENE.name)
    data = mujoco.MjData(model)

    pose = B2_CROSS_STEP
    a = math.radians(-pose.pitch_deg)  # 绕 Y 负角 = 抬头
    data.qpos[:] = 0.0
    data.qpos[0:3] = pose.base_xyz
    data.qpos[3] = math.cos(a / 2.0)
    data.qpos[5] = math.sin(a / 2.0)
    for i, leg in enumerate(LEGS):
        for k, joint in enumerate(("hip", "thigh", "calf")):
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint")
            data.qpos[model.jnt_qposadr[j]] = pose.joint_rad[3 * i + k]
    mujoco.mj_forward(model, data)

    # ctrl 顺序按执行器名映射到目标角（B2 的执行器顺序是 FR/FL/RR/RL）
    target = np.empty(model.nu)
    for a_i in range(model.nu):
        j = int(model.actuator_trnid[a_i, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        leg, part = name.split("_")[0], name.split("_")[1]
        target[a_i] = pose.joint_rad[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]

    print(f"姿态 {pose.name}: pitch {pose.pitch_deg:.1f}°, 机身 {pose.base_xyz}")
    print(f"净空 机头 {pose.head_clearance_m*100:.1f} cm / 机身 {pose.body_clearance_m*100:.1f} cm"
          f" / 前腿护板 {pose.shin_clearance_m*100:.1f} cm")
    print("鼠标：左键旋转，滚轮缩放，右键平移；关闭窗口退出。")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        # 侧前方 3/4 视角：同时看得到一级踏面、机头与后腿
        viewer.cam.lookat[:] = (0.12, 0.468, 0.30)
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 115.0
        viewer.cam.elevation = -14.0
        while viewer.is_running():
            tick = time.perf_counter()
            for _ in range(10):  # 0.02 s 控制周期，0.002 s 物理步长
                for a_i in range(model.nu):
                    j = int(model.actuator_trnid[a_i, 0])
                    dof = int(model.jnt_dofadr[j])
                    q = data.qpos[model.jnt_qposadr[j]]
                    tau = (
                        pose.kp * (target[a_i] - q)
                        - pose.kd * data.qvel[dof]
                        + B2_ROBOT.fsm.gravity_feedforward_scale * data.qfrc_bias[dof]
                    )
                    lo, hi = model.actuator_ctrlrange[a_i]
                    data.ctrl[a_i] = min(hi, max(lo, tau))
                mujoco.mj_step(model, data)
            viewer.sync()
            sleep = 0.02 - (time.perf_counter() - tick)
            if sleep > 0:
                time.sleep(sleep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
