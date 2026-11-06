"""B2 窄踏面爬梯尝试的实时播放（MuJoCo 3D 窗口，可拖拽/缩放）。

播放的是 b2_stair_gait.py 里的"C 方案"：从"四足同踩一级"出发，
每拍先把重心挪到剩下三只脚的形心，再抬一只脚到上一级，四拍推进一阶。
本脚本会循环播放，方便反复观察它在哪里卡住。

用法：D:\\Python313\\python.exe .\\experiments\\b2\\b2_gait_view.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "b2"))

from configs.robot_config import B2_ROBOT  # noqa: E402
import b2_stair_gait as G  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402


def build_plan():
    """返回 (初始机身, 初始足位, 动作段列表)。"""
    body0 = (G.TREAD1_CENTER, 0.50, 0.0)
    feet0 = G.feet_on_tread(G.TREAD1_CENTER, G.TREAD1_TOP)
    feet = {k: np.array(v, dtype=float) for k, v in feet0.items()}
    body = np.array(body0, dtype=float)
    segments = []
    for leg in ("FL", "RR", "FR", "RL"):
        others = [k for k in G.LEGS if k != leg]
        cx = float(np.mean([feet[k][0] for k in others]))
        shifted = body.copy()
        shifted[0] += 0.7 * (cx - body[0])
        segments.append(G.Segment(0.8, body=tuple(shifted)))
        feet[leg][0] += G.STEP_PITCH
        feet[leg][2] += G.STEP_PITCH
        shifted[0] += G.STEP_PITCH / 4.0
        shifted[1] += 0.02
        shifted[2] += 2.0
        segments.append(G.Segment(1.2, body=tuple(shifted),
                                  feet={leg: tuple(feet[leg])}, lift_m=0.09))
        body = shifted
        segments.append(G.Segment(0.3, body=tuple(body)))
    return body0, feet0, segments


def main() -> int:
    model, data = G.load_scene()
    body0, feet0, segments = build_plan()

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = (0.35, 0.468, 0.35)
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 120.0
        viewer.cam.elevation = -12.0
        print("播放：四足同踩一级 → 每拍挪重心 → 抬脚上一级（会循环）")
        print("鼠标：左键旋转、滚轮缩放、右键平移；关闭窗口退出。")
        while viewer.is_running():
            walker = G.Walker(model, data)
            walker.place(body0, feet0)
            walker.run(segments, sync=viewer.sync, realtime=True)
            if not viewer.is_running():
                break
            t_end = time.perf_counter() + 2.0
            while viewer.is_running() and time.perf_counter() < t_end:
                walker.run([G.Segment(0.1, body=tuple(walker.body))],
                           sync=viewer.sync, realtime=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
