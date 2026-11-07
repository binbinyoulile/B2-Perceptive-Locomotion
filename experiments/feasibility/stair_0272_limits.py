"""0.197 m 级高 / 0.272 m 踏面下的三机几何极限表（纯解析，不跑物理）。

文件目的：
    现场实测把踏面从 0.18 m 放宽到 0.272 m 后，先算清楚"哪些约束真的松了"：
    四足同踩一级的支撑深度、抬腿时的重心余量、跨级腿的伸展率、机头相对上一级的净空。
    这些数决定后面该选哪台狗、走哪种步态。

输入：
    configs/robot_config.py（Go2、B2 的官方参数）
    三个官方 MJCF 里读出来的腿长/髋距（与 multi_robot_compare.py 同源）
输出：
    experiments/feasibility/stair_0272_limits.txt
依赖：numpy、mujoco（只用来读模型尺寸）
运行：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\stair_0272_limits.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_ROBOT, GO2_ROBOT  # noqa: E402

import mujoco  # noqa: E402

# 现场实测钢梯
RISE = 0.197
TREAD = 0.272
WIDTH = 0.81


def read_as2() -> dict:
    path = ROOT / "models" / "unitree_as2_mujoco" / "xml" / "as2.xml"
    prev = Path.cwd()
    os.chdir(path.parent)
    try:
        m = mujoco.MjModel.from_xml_path(path.name)
    finally:
        os.chdir(prev)
    calf = m.body("FL_calf")
    thigh = m.body("FL_thigh")
    l1 = float(np.linalg.norm(calf.pos))
    foot = next(i for i in range(m.ngeom)
                if m.geom_bodyid[i] == calf.id
                and m.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE)
    l2 = float(-m.geom_pos[foot, 2])
    hip_x = float(m.body("FL_hip").pos[0])
    foot_y = float(m.body("FL_hip").pos[1] + thigh.pos[1])
    return dict(name="AS2", hip_x=hip_x, l1=l1, l2=l2, foot_r=float(m.geom_size[foot, 0]),
                foot_y=foot_y, mass=17.64, torque=90.0,
                stand=0.33, head_fwd=0.326, head_low=0.035)


def row(name, hip_x, l1, l2, foot_r, foot_y, mass, torque, stand, head_fwd, head_low):
    reach = l1 + l2
    usable = TREAD - 2 * foot_r
    half = usable / 2
    # 四足同踩一级：前/后足在踏面里，髋在机身两端
    dx = hip_x - half
    dz = stand - foot_r
    r_req = math.hypot(dx, dz)
    # 抬头 0° 时，机头下沿相对"上一级"的路面：站在本级时上面两级的高度差
    head_z = stand + head_low          # 机头下沿离地高度（和踏面高度直接比）
    # 跨级：前足高一级、后足本级
    dx_st = hip_x - TREAD / 2
    r_st = math.hypot(dx_st, dz + RISE / 2)
    # 同踩时的重心窗口：支撑深的一半（保守按 ±usable/2）
    return dict(name=name, hip=2 * hip_x, l1=l1, l2=l2, reach=reach, usable=usable,
                dx=dx, r_req=r_req, ext=r_req / reach * 100, head_clear=head_z,
                r_st=r_st, ext_st=r_st / reach * 100, win=usable / 2, mass=mass,
                torque=torque, twr=torque / mass, head_fwd=head_fwd,
                foot_y=foot_y, head_low=head_low)


def main() -> int:
    rows = [
        row("Go2", 0.1934, GO2_ROBOT.thigh_length_m, GO2_ROBOT.calf_length_m,
            GO2_ROBOT.foot_radius_m, 0.1420, 15.206, 45.43, 0.2577, 0.32, -0.107),
        row("B2", 0.3285, B2_ROBOT.thigh_length_m, B2_ROBOT.calf_length_m,
            B2_ROBOT.foot_radius_m, 0.1916, 69.003, 320.0, 0.4065, 0.342, -0.065),
        row(**{k: v for k, v in read_as2().items()
               if k in ("name", "hip_x", "l1", "l2", "foot_r", "foot_y", "mass",
                        "torque", "stand", "head_fwd", "head_low")}),
    ]
    lines = []

    def say(t=""):
        lines.append(t)

    say("=" * 92)
    say(f"几何极限表：级高 {RISE:.3f} m / 踏面 {TREAD:.3f} m / 梯宽 {WIDTH:.2f} m")
    say("=" * 92)
    say(f"{'机型':<6}{'髋距':>7}{'可达':>7}{'同踩支撑深':>11}{'腿需伸展':>9}"
        f"{'跨级伸展':>9}{'机头下沿':>9}{'重心窗口':>9}{'力矩/质量':>9}")
    for r in rows:
        say(f"{r['name']:<6}{r['hip']:>7.3f}{r['reach']:>7.3f}{r['usable']:>11.3f}"
            f"{r['ext']:>8.0f}%{r['ext_st']:>8.0f}%{r['head_clear']:>9.3f}"
            f"{r['win']:>9.3f}{r['twr']:>9.1f}")
    say()
    say("口径：")
    say("  同踩支撑深 = 踏面 - 2×足半径（足心离边缘留一个半径）；")
    say("  腿需伸展 = 四足同踩时，髋到足的距离 ÷ 腿长（100% 即伸直、不可行）；")
    say("  跨级伸展 = 前足高一级、后足本级时后腿的伸展率；")
    say("  机头下沿 = 机身高度 + 机头下沿偏移（直接和 0.197/0.394/0.591 三个踏面高度比）；")
    say("  重心窗口 = ±(同踩支撑深/2)，即四足同踩时重心能偏多少的上限。")
    say()
    say("对照（旧尺寸 0.18/0.195）：B2 支撑深 0.116、窗口 ±3.7cm；现在 0.208 / ±10.4cm。")

    text = "\n".join(lines)
    out = Path(__file__).with_suffix(".txt")
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
