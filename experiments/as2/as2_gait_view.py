"""AS2 爬梯动作的 3D 动态演示：跨级姿态 → 四拍抬腿/收腿尝试（真实物理，PD 跟踪）。

文件目的：
    在 MuJoCo 交互窗口里看 AS2 在 0.195 m 台阶上的完整动作序列：
      起手跨级站姿（前足二级、后足一级） → 落地 0.8 s →
      右后腿上二级 → 左后腿上二级 → 前足前移 → 前足上三级 → 后腿跟到三级。
    动作序列与 ``as2_stair_gait.py`` 完全同一份（import 复用），所以窗口里看到的就是
    报告里的数字来源，不是另一套假动画。

输入：
    simulations/mujoco/as2_stairs_static/scene.xml
输出：
    无（只开窗口）；按 Esc 或关窗退出
依赖：
    mujoco 3.16（需要 OpenGL 窗口）、numpy、同目录 as2_stair_gait / as2_stance_probe
运行：
    D:\\Python313\\python.exe .\\experiments\\as2\\as2_gait_view.py [--check]

关键假设：
    - 50 Hz 控制、kp/kd = 400/12；窗口里按真实时间播放（不是慢放）。
    - 序列在某一段跌落时停止，等 1.5 s 后自动回到起手姿态重播。

已知限制：
    - 这是"人写的动作序列"演示；它失败了不代表 AS2 一定爬不上去。
    - 窗口需要显示器；--check 用无窗口模式跑 1.5 s 自检。
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

import as2_stair_gait as gait  # noqa: E402
import as2_stance_probe as probe  # noqa: E402


def reset(w: gait.Walker, robot, feet0) -> None:
    body0 = gait.body_for(feet0, 12.0)
    feet_pre = {k: np.array(v, dtype=float) for k, v in feet0.items()}
    for k in feet_pre:
        feet_pre[k][2] -= 0.003  # 预压 3mm，保证第一帧就有四足接触
    w.place(body0, feet_pre)
    w.run([gait.Segment("settle", 0.8, body0, {k: tuple(v) for k, v in feet0.items()})])


def main() -> int:
    model, data = probe.load_scene()
    robot = probe.read_robot(model)
    feet0 = {k: np.array(v, dtype=float) for k, v in gait.straddle_feet(robot).items()}
    _, segs = gait.plan(robot)
    w = gait.Walker(model, data, robot)

    if "--check" in sys.argv:
        reset(w, robot, feet0)
        print(f"自检：起手 z {w.data.qpos[2]:.4f}，接触数 {w.data.ncon}，"
              f"前足目标 {gait.T2:.3f}、后足 {gait.T1-0.062:.3f}")
        return 0

    import mujoco.viewer

    print("AS2 爬梯演示：跨级站姿 → 四拍抬腿/收腿。关窗退出。")
    while True:
        reset(w, robot, feet0)
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                          show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = (0.40, gait.BODY_Y, 0.40)
            viewer.cam.distance = 1.9
            viewer.cam.azimuth = 25.0
            viewer.cam.elevation = -18.0
            for seg in segs:
                body_src = w.body.copy()
                body_dst = np.array(seg.body, dtype=float)
                feet_src = {k: v.copy() for k, v in w.feet.items()}
                feet_dst = {k: np.array(seg.feet.get(k, tuple(feet_src[k])), dtype=float)
                            for k in gait.LEGS}
                steps = max(1, int(round(seg.duration * w.ctrl_hz)))
                for n in range(steps):
                    if not viewer.is_running():
                        return 0
                    t0 = time.perf_counter()
                    u = (n + 1) / steps
                    smooth = u * u * (3.0 - 2.0 * u)
                    body = body_src + (body_dst - body_src) * smooth
                    feet = {}
                    for leg in gait.LEGS:
                        src, dst = feet_src[leg], feet_dst[leg]
                        pos = src + (dst - src) * smooth
                        if (np.linalg.norm(dst[:2] - src[:2]) > 1e-4
                                or abs(dst[2] - src[2]) > 1e-4):
                            pos[2] += seg.lift_m * math.sin(math.pi * u)
                        feet[leg] = pos
                    w.step(w.targets(tuple(body), feet))
                    viewer.cam.lookat[:] = (float(data.qpos[0]), gait.BODY_Y,
                                            float(data.qpos[2]) - 0.10)
                    viewer.sync()
                    rest = 1.0 / w.ctrl_hz - (time.perf_counter() - t0)
                    if rest > 0:
                        time.sleep(rest)
                w.body = body_dst
                w.feet = {k: v.copy() for k, v in feet_dst.items()}
                print(f"  {seg.name:<26} x {data.qpos[0]:+.3f}  z {data.qpos[2]:.3f}  "
                      f"ncon {data.ncon}")
                if data.qpos[2] < 0.15:
                    print("  跌落，1.5 s 后重播")
                    time.sleep(1.5)
                    break
        # 关窗后重新打开，直到用户关窗
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
