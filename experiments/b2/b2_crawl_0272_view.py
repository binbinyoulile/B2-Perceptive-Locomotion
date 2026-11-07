"""B2 错位对角 + 四拍爬升的 3D 动态演示（0.197 级高 / 0.272 踏面）。

文件目的：
    把 b2_crawl_0272.py 里的静态姿态与四拍序列在 MuJoCo 交互窗口里按真实时间播一遍，
    用来肉眼确认：错位对角能不能站住、第一拍抬起后机身掉多少、第几步彻底跌下来。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（新尺寸钢梯）
输出：
    无（只开窗口）；Esc 或关窗退出
依赖：mujoco 3.16（需 OpenGL）、numpy、同目录 b2_crawl_0272 / b2_step_pose
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_crawl_0272_view.py

关键假设：
    - 静态姿态取 b2_crawl_0272.py 搜出的值（pitch 0°、机身 x 0.360、z 0.680）；
      若该脚本重跑后数值变了，这里的常量要同步（它只是演示用的起点）。
已知限制：
    - 演示不重跑搜索；序列失败是真实结果，不是渲染问题。
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

import b2_crawl_0272 as C  # noqa: E402

BODY0 = (0.360, 0.680, 0.0)   # b2_crawl_0272.py 搜出的静态解（pitch, x, z 顺序见 build 调用）


def main() -> int:
    import mujoco.viewer

    model, data = C.base.load_scene()
    rb = C.Robot(model, data)
    feet = {k: np.array(v, dtype=float) for k, v in C.stance_feet().items()}
    body = np.array(BODY0, dtype=float)
    rb.build(tuple(body), feet)
    print("B2 错位对角：FL 二级 / FR 一级前 / RR 一级后 / RL 地面；kp/kd 800/20")
    print("关窗退出。")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.cam.lookat[:] = (0.50, C.BODY_Y, 0.45)
        viewer.cam.distance = 2.4
        viewer.cam.azimuth = 25.0
        viewer.cam.elevation = -16.0
        last = [0.0]

        def sync():
            now = time.perf_counter()
            if now - last[0] >= 1.0 / 50.0:
                viewer.cam.lookat[:] = (float(data.qpos[0]), C.BODY_Y, float(data.qpos[2]) - 0.15)
                viewer.sync()
                last[0] = now

        rb.build(tuple(body), feet)
        rb.hold(rb.targets(tuple(body), feet), 1.2, sync)
        for name, leg, dst, lift in C.PLAN:
            others = [k for k in C.LEGS if k != leg]
            cx = float(np.mean([feet[k][0] for k in others]))
            shift = body.copy()
            shift[0] += 0.4 * (cx - body[0])
            rb.hold(rb.targets(tuple(shift), feet), 1.0, sync)
            body = shift
            src = feet[leg].copy()
            dst_v = np.array(dst, dtype=float)
            for n in range(75):
                u = (n + 1) / 75
                sm = u * u * (3 - 2 * u)
                pos = src + (dst_v - src) * sm
                pos[2] += lift * math.sin(math.pi * u)
                f = {k: v for k, v in feet.items()}
                f[leg] = pos
                rb.hold(rb.targets(tuple(body), f), 0.02, sync)
            feet[leg] = dst_v
            rb.hold(rb.targets(tuple(body), feet), 0.6, sync)
            print(f"  {name}: z {data.qpos[2]:.3f}")
            if not viewer.is_running() or data.qpos[2] < 0.20:
                break
        time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
