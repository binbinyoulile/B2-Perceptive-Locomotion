"""AS2 跨级站姿的交互式 3D 预览（真实物理 + PD 保持，可自由旋转/缩放）。

文件目的：
    在 MuJoCo 交互窗口里看 AS2 站在"0.195 m 高 / 0.18 m 深"钢梯上的跨级姿态
    （前足踩二级、后足留一级），用于肉眼确认净空、腿型与机身姿态。
    窗口里跑的是真实物理：12 个关节用 kp/kd + 重力前馈跟踪跨级姿态的关节角，
    不是把 qpos 冻住的假动画。

输入：
    simulations/mujoco/as2_stairs_static/scene.xml、experiments/as2/as2_stance_probe.json
输出：
    无（只开窗口，按 Esc 或关窗退出）
依赖：
    mujoco 3.16（需要 OpenGL 窗口）、numpy
运行：
    D:\\Python313\\python.exe .\\experiments\\as2\\as2_stance_view.py

关键假设：
    - 用 as2_stance_probe.py 搜出来的跨级姿态（pitch 12°，机身 x 0.369，髋高 0.380）。
    - 视角跟随机身，鼠标左键旋转、右键平移、滚轮缩放（MuJoCo 默认操作）。
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

import mujoco

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulations" / "mujoco" / "as2_stairs_static" / "scene.xml"
POSE_JSON = ROOT / "experiments" / "as2" / "as2_stance_probe.json"
LEGS = ("FL", "FR", "RL", "RR")
BODY_Y = 0.468
STEP1_TOP = 0.195
KP, KD = 400.0, 12.0


def load_model():
    previous = Path.cwd()
    os.chdir(SCENE.parent)
    try:
        return mujoco.MjModel.from_xml_path(SCENE.name)
    finally:
        os.chdir(previous)


def pose_from_json() -> tuple[tuple[float, float], float]:
    """从探针结果里读跨级姿态；没有结果文件时退回搜索到的固定值。"""
    if POSE_JSON.exists():
        data = json.loads(POSE_JSON.read_text(encoding="utf-8"))
        cmd = data.get("straddle", {}).get("cmd")
        if cmd:
            return (float(cmd[0]), float(cmd[1])), float(cmd[2])
    return (0.369, STEP1_TOP + 0.380), 12.0


def main() -> int:
    import mujoco.viewer

    model = load_model()
    data = mujoco.MjData(model)
    (bx, bz), pitch = pose_from_json()

    # 前足踩二级、后足留一级（与探针 S3 完全一致）
    feet = {
        "FL": (0.491832, BODY_Y + 0.1654, 0.3803 + 0.028),
        "FR": (0.491832, BODY_Y - 0.1654, 0.3803 + 0.028),
        "RL": (0.306570 + 0.062, BODY_Y + 0.1654, STEP1_TOP + 0.028),
        "RR": (0.306570 - 0.062, BODY_Y - 0.1654, STEP1_TOP + 0.028),
    }
    a = math.radians(-pitch)
    c, s = math.cos(a), math.sin(a)
    data.qpos[:] = 0.0
    data.qpos[0], data.qpos[1], data.qpos[2] = bx, BODY_Y, bz
    data.qpos[3], data.qpos[5] = math.cos(a / 2.0), math.sin(a / 2.0)
    target = np.zeros(model.nu)
    for i, leg in enumerate(LEGS):
        fx, _fy, fz = feet[leg]
        dxw, dzw = fx - bx, fz - bz
        lx = dxw * c - dzw * s
        lz = dxw * s + dzw * c
        hip_x = 0.2119 * (1 if leg in ("FL", "FR") else -1)
        dx, dz = lx - hip_x, lz
        r = math.hypot(dx, dz)
        l1, l2 = 0.212, 0.21344
        q2 = -math.acos(max(-1.0, min(1.0, (r * r - l1 * l1 - l2 * l2) / (2 * l1 * l2))))
        q1 = math.atan2(-dx, -dz) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        for jn, val in (("hip", 0.0), ("thigh", q1), ("calf", q2)):
            j = model.joint(f"{leg}_{jn}_joint")
            data.qpos[model.jnt_qposadr[j.id]] = val
        for adr in range(model.nu):
            name = model.actuator(adr).name
            if not name.startswith(f"{leg}_"):
                continue
            target[adr] = {"hip": 0.0, "thigh": q1, "calf": q2}[name.split("_")[1]]

    mujoco.mj_forward(model, data)
    print("AS2 跨级姿态：前足踩二级（0.3803 m）、后足留一级（0.195 m）")
    print(f"机身 ({bx:.3f}, {BODY_Y:.3f}, {bz:.3f})，抬头 {pitch:.0f}°，kp/kd {KP:.0f}/{KD:.0f}")
    print("鼠标左键旋转 / 右键平移 / 滚轮缩放；关窗退出。")

    substeps = max(1, int(round(1.0 / (50.0 * model.opt.timestep))))

    if "--check" in sys.argv:
        # 无窗口自检：跑 1.5 s 物理，确认这个姿态在 PD 下站得住
        for _ in range(int(1.5 * 50)):
            for adr in range(model.nu):
                j = int(model.actuator_trnid[adr, 0])
                dof = int(model.jnt_dofadr[j])
                tau = (KP * (target[adr] - data.qpos[model.jnt_qposadr[j]])
                       - KD * data.qvel[dof] + data.qfrc_bias[dof])
                lo, hi = model.actuator_ctrlrange[adr]
                data.ctrl[adr] = min(hi, max(lo, tau))
            for _ in range(substeps):
                mujoco.mj_step(model, data)
        print(f"自检 1.5 s：机身 z {data.qpos[2]:.4f} m（指令 {bz:.4f}），x {data.qpos[0]:.4f}")
        return 0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = (0.40, BODY_Y, 0.36)
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 118.0
        viewer.cam.elevation = -12.0
        next_ctrl = 0.0
        while viewer.is_running():
            now = time.perf_counter()
            if now >= next_ctrl:
                for adr in range(model.nu):
                    j = int(model.actuator_trnid[adr, 0])
                    dof = int(model.jnt_dofadr[j])
                    q = data.qpos[model.jnt_qposadr[j]]
                    tau = KP * (target[adr] - q) - KD * data.qvel[dof] + data.qfrc_bias[dof]
                    lo, hi = model.actuator_ctrlrange[adr]
                    data.ctrl[adr] = min(hi, max(lo, tau))
                next_ctrl = now + 1.0 / 50.0
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            viewer.sync()
            viewer.cam.lookat[:] = (float(data.qpos[0]), BODY_Y, float(data.qpos[2]) - 0.08)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
