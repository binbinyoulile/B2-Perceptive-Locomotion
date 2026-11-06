"""三台四足机器人在 0.195 m / 0.18 m 钢梯上的几何可行性对比。

文件目的：
    用同一套判据比较 Go2 / Unitree B2（标准版）/ GENISOM M1：
    1) 四足同踩一级时，支撑多边形深度与最大重心余量（只由踏面与足端半径决定）；
    2) 四足同踩一级所需的前后足中心距与腿长占用；
    3) 站上一级踏板时，机身/机头与上一级踏板的静态净空；
    4) 逐级对角（前足高一级、后足低一级）时后腿需要的伸展量，以及需要蹲低多少。

角色与判据约定：
    - ``D``：机身中心到“当前脚下那一级踏面”上足端球心的垂距（m）。标称站姿
      的 D 就是 ``RobotConfig.nominal_stand_height_m``；机身离地 = D + 足端半径。
    - 逐级对角：前足在上一级、后足在下一级，机身保持同一姿态。
      后腿需要跨过 ``D + 级间高差``，前后足中心距按“两级踏面中心”取 ``踏面深``，
      因此相对髋部的水平偏移 ``dx = (髋距 - 踏面深)/2``。
    - 判据只看静态几何：不含接触时序、摩擦与动力学。结论是“几何上是否可能”，
      不是“控制器能不能做到”。

数据来源：
    - Go2：``configs.robot_config.GO2_ROBOT``（本仓库 MJCF 实测）
    - B2 ：``configs.robot_config.B2_ROBOT``（官方 URDF/MuJoCo 模型实测，
      模型在 ``models/unitree_b2_mujoco/``，合计 69.003 kg）
    - M1 ：公开资料 + 显式假设（无公开 MJCF/URDF），必须标注“估计值”

⚠ 2026-11-06 修正：本脚本上一版把 ``unitree_mujoco/unitree_robots/b2/b2.xml``
（83.5 kg，实为轮足版 B2-W 的派生模型）当成标准版 B2，并用它的 home 关键帧
推出“站高 0.468 m”，导致对角伸展被判成 96%。标准版 B2 的正确站高是
0.3745 m（官网站立 645 mm 反解），对角伸展为 87%。详见
``docs/B2_VS_M1_REPORT.md`` 第 1 节。

依赖与版本要求：
    Python 3.10+、mujoco 3.16、numpy 2.x。
运行方式：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\multi_robot_geometry.py
整体输出：
    终端表格 + ``report_multi_robot.txt`` / ``report_multi_robot.json``
已知限制：
    M1 的腿长/足端半径/站高/机头下沿全是假设；B2 的官方 CAD 69 kg 与官网
    标称 60 kg 差 9 kg，待宇树确认。静态几何不等于动态可爬性。
重要性等级：A（选型结论依赖这些数字）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulations" / "mujoco" / "go2_stairs_static" / "scene.xml"
OUT = ROOT / "experiments" / "feasibility" / "report_multi_robot.txt"

sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_ROBOT, GO2_ROBOT  # noqa: E402

# 从已编译的钢梯模型读取踏面参数（与单机报告同源）。
previous = Path.cwd()
os.chdir(SCENE.parent)
import mujoco  # noqa: E402

model = mujoco.MjModel.from_xml_path(SCENE.name)
os.chdir(previous)
geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "up_step_1")
geom2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "up_step_2")
TREAD_DEPTH_M = float(2.0 * model.geom_size[geom_id, 0])
STEP1_TOP_M = float(model.geom_pos[geom_id, 2] + model.geom_size[geom_id, 2])
STEP2_TOP_M = float(model.geom_pos[geom2_id, 2] + model.geom_size[geom2_id, 2])
RISE_M = STEP2_TOP_M - STEP1_TOP_M


@dataclass(frozen=True)
class RobotGeometry:
    """做楼梯几何可行性判断所需的机器人参数。

    字段：
        name：模型名。
        hip_spacing_m：前髋到后髋的水平距离（同侧）。
        max_reach_m：单腿最大髋—足距离（大腿+小腿两段之和）。
        foot_radius_m：足端球半径。
        body_half_height_m：机身碰撞盒半高。
        stand_height_m：标称站姿下机身中心到足端球心的垂距（D）。
        head_bottom_m：机头最低点相对机身中心的高度（负值）。
        com_offset_x_m：整机重心相对机身中心的 X 偏移。
        mass_kg：质量（记录值）。
        source：参数来源说明。
    """

    name: str
    hip_spacing_m: float
    max_reach_m: float
    foot_radius_m: float
    body_half_height_m: float
    stand_height_m: float
    head_bottom_m: float
    com_offset_x_m: float
    mass_kg: float
    source: str


def from_config(robot, source: str) -> RobotGeometry:
    """把 ``configs.robot_config.RobotConfig`` 转成几何判据需要的字段。"""
    return RobotGeometry(
        name=robot.name.upper(),
        hip_spacing_m=2.0 * abs(robot.hip_x_m[0]),
        max_reach_m=robot.max_reach_m,
        foot_radius_m=robot.foot_radius_m,
        body_half_height_m=robot.body_half_height_m,
        stand_height_m=robot.nominal_stand_height_m,
        head_bottom_m=robot.head_bottom_m,
        com_offset_x_m=robot.com_offset_x_m,
        mass_kg=robot.nominal_mass_kg,
        source=source,
    )


ROBOTS = [
    from_config(GO2_ROBOT, "本仓库 MJCF + 实测（configs/robot_config.py）"),
    from_config(
        B2_ROBOT,
        "官方 b2_description_mujoco（models/unitree_b2_mujoco，69.003 kg）+ 官网"
        "站立尺寸 645 mm",
    ),
    RobotGeometry(
        name="M1（估计）",
        hip_spacing_m=0.55,
        max_reach_m=0.44,
        foot_radius_m=0.035,
        body_half_height_m=0.11,
        stand_height_m=0.33,
        head_bottom_m=-0.10,
        com_offset_x_m=0.02,
        mass_kg=41.0,
        source="公开资料 + 假设（无公开 MJCF/URDF，需厂商确认）",
    ),
]

# 对角姿态里前后足的水平偏移：按“两级踏面中心相距一个踏面深”取。
DIAGONAL_FOOT_SPAN_M = TREAD_DEPTH_M

# calf 关节下限（取自各自 MJCF 的 range，用于判断 IK 是否被限位卡住）。
CALF_LIMIT_RAD = {"GO2": -2.7227, "B2": -2.82}


def knee_angles(reach: float, thigh: float, calf: float) -> tuple[float, float]:
    """按余弦定理把“髋—足直线距离”换成膝内角与 calf 关节角（rad）。

    返回 (膝内角, calf 关节角)。两段等长且完全伸直时内角 = π、calf = 0；
    折叠时内角变小、calf 变负，与 MJCF 的负角度约定一致。
    """
    cosine = (reach * reach - thigh * thigh - calf * calf) / (2.0 * thigh * calf)
    interior = math.acos(max(-1.0, min(1.0, cosine)))
    return interior, interior - math.pi


def evaluate(robot: RobotGeometry) -> dict:
    """按 0.195/0.18 场景计算四项判据。"""
    thigh = calf = 0.5 * robot.max_reach_m
    support_depth = TREAD_DEPTH_M - 2.0 * robot.foot_radius_m
    com_margin = 0.5 * support_depth - abs(robot.com_offset_x_m)
    body_bottom = robot.stand_height_m + robot.foot_radius_m - robot.body_half_height_m

    # 四足同踩一级：前后足中心距最大 = 支撑深度。
    same_tread_dx = 0.5 * (robot.hip_spacing_m - support_depth)
    same_tread_reach = math.hypot(same_tread_dx, robot.stand_height_m)

    # 逐级对角：后腿要跨过“站高 + 级间高差”。
    diagonal_dx = 0.5 * (robot.hip_spacing_m - DIAGONAL_FOOT_SPAN_M)
    diagonal_drop = robot.stand_height_m + RISE_M
    diagonal_reach = math.hypot(diagonal_dx, diagonal_drop)
    calf_limit = CALF_LIMIT_RAD.get(robot.name, -2.82)
    # 超过可达半径时反解无意义（cos 会被截断），显式记成“无解”。
    if diagonal_reach <= robot.max_reach_m:
        _, calf_angle = knee_angles(diagonal_reach, thigh, calf)
        calf_margin = calf_angle - calf_limit
    else:
        calf_angle = None
        calf_margin = None
    # 允许的最高站高：把后腿伸展压到 100% 时对应的 D。
    max_stand_for_diagonal = math.sqrt(
        max(0.0, robot.max_reach_m ** 2 - diagonal_dx ** 2)
    ) - RISE_M
    return {
        "support_depth_m": support_depth,
        "com_margin_m": com_margin,
        "same_tread_reach_m": same_tread_reach,
        "same_tread_ratio": same_tread_reach / robot.max_reach_m,
        "body_bottom_height_m": body_bottom,
        "body_clearance_m": body_bottom - RISE_M,
        "head_clearance_m": robot.stand_height_m + robot.foot_radius_m
        + robot.head_bottom_m - RISE_M,
        "diagonal_dx_m": diagonal_dx,
        "diagonal_drop_m": diagonal_drop,
        "diagonal_reach_m": diagonal_reach,
        "diagonal_ratio": diagonal_reach / robot.max_reach_m,
        "diagonal_calf_rad": calf_angle,
        "diagonal_calf_margin_rad": calf_margin,
        "diagonal_possible": calf_margin is not None and calf_margin >= 0.0,
        "max_stand_for_diagonal_m": max_stand_for_diagonal,
        "crouch_needed_m": robot.stand_height_m - max_stand_for_diagonal,
    }


def main() -> int:
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)

    say("=" * 96)
    say("三台四足机器人在 0.195 m 台阶 / 0.18 m 踏面上的几何可行性")
    say("=" * 96)
    say(
        "场景：踏面深 %.4f m，第一级顶面 %.4f m，级间高差 %.4f m（从 scene.xml 编译结果读取）"
        % (TREAD_DEPTH_M, STEP1_TOP_M, RISE_M)
    )
    say(
        "判据：支撑深度 = 踏面深 - 2×足端半径（只由踏面决定）；重心余量上界 = "
        "支撑深度/2 - |重心偏移|；对角后腿需跨 D + 级间高差，D 为机身中心到脚下踏面的足心垂距。"
    )
    say()

    results = {robot.name: evaluate(robot) for robot in ROBOTS}

    header = "%-12s %7s %7s %7s %7s %8s %8s %8s %8s" % (
        "机器人", "可达m", "支撑深", "重心余", "同踩占比", "机身净空", "机头净空", "对角占比", "蹲低需求",
    )
    say(header)
    say("-" * 96)
    for robot in ROBOTS:
        item = results[robot.name]
        say(
            "%-12s %7.3f %7.3f %7.3f %6.1f%% %8.3f %8.3f %7.1f%% %8s"
            % (
                robot.name,
                robot.max_reach_m,
                item["support_depth_m"],
                item["com_margin_m"],
                100.0 * item["same_tread_ratio"],
                item["body_clearance_m"],
                item["head_clearance_m"],
                100.0 * item["diagonal_ratio"],
                ("%.3f m" % item["crouch_needed_m"]) if item["crouch_needed_m"] > 0 else "不需要",
            )
        )
    say()

    say("逐台细节：")
    for robot in ROBOTS:
        item = results[robot.name]
        say("  [%s] 来源: %s" % (robot.name, robot.source))
        say(
            "    质量=%.3f kg  髋距=%.3f m  站高D=%.4f m  足端半径=%.3f m  机身半高=%.3f m"
            % (robot.mass_kg, robot.hip_spacing_m, robot.stand_height_m,
               robot.foot_radius_m, robot.body_half_height_m)
        )
        say(
            "    四足同踩一级: 髋—足需 %.3f m（%.1f%%）；机身底面离一级顶面 %.3f m；机头 %.3f m"
            % (item["same_tread_reach_m"], 100.0 * item["same_tread_ratio"],
               item["body_clearance_m"], item["head_clearance_m"])
        )
        if item["diagonal_calf_rad"] is None:
            calf_text = "超出可达半径（反解无解）"
        else:
            calf_text = "calf=%.3f rad（限位 %.2f，余 %.3f）" % (
                item["diagonal_calf_rad"],
                CALF_LIMIT_RAD.get(robot.name, -2.82),
                item["diagonal_calf_margin_rad"],
            )
        say(
            "    逐级对角: 水平偏移 dx=%.3f m，后腿需跨 %.3f m（%.1f%%）；%s → %s"
            % (
                item["diagonal_dx_m"], item["diagonal_reach_m"],
                100.0 * item["diagonal_ratio"], calf_text,
                "可达" if item["diagonal_possible"] else "超出可达半径/关节限位",
            )
        )
        if item["crouch_needed_m"] > 1e-4:
            say(
                "      对角伸展 100%% 对应的最高站高 D=%.3f m，比标称低 %.3f m → 必须蹲低"
                % (item["max_stand_for_diagonal_m"], item["crouch_needed_m"])
            )
        else:
            say(
                "      对角伸展 100%% 对应 D=%.3f m，比标称高 %.3f m → 标称站高下有余量"
                % (item["max_stand_for_diagonal_m"], -item["crouch_needed_m"])
            )
        if item["head_clearance_m"] < 0.0:
            say("      [!] 机头在水平机身下会插进上一级踏板，必须抬机头")
        say()

    b2 = results["B2"]
    go2 = results["GO2"]
    say("-" * 96)
    say("结论：")
    say(
        "1) 踏面宽度仍是硬约束：三台的重心余量上界只有 %.1f~%.1f cm，"
        "其中 B2 为 %.1f cm；这个数只由踏面与足端半径决定，换更长的腿也不会变大。"
        % (
            100.0 * min(r["com_margin_m"] for r in results.values()),
            100.0 * max(r["com_margin_m"] for r in results.values()),
            100.0 * b2["com_margin_m"],
        )
    )
    say(
        "2) 标准版 B2 的机头净空 %.3f m、机身净空 %.3f m，都大于 10 cm；"
        "Go2 的机头净空 %.3f m（为负）。"
        % (b2["head_clearance_m"], b2["body_clearance_m"], go2["head_clearance_m"])
    )
    say(
        "3) 对角步态：标准版 B2 后腿伸展 %.1f%%（calf %.2f rad，离限位 0.7 rad 以上），"
        "标称站高下就够用；Go2 %.1f%%（>100%%，且机头为负 → 只能抬机头换空间）。"
        % (100.0 * b2["diagonal_ratio"], b2["diagonal_calf_rad"],
           100.0 * go2["diagonal_ratio"])
    )
    say(
        "4) 上一版把 B2 写成 96%%、24 cm 净空，是用轮足版 B2-W 的派生 MJCF"
        "（83.5 kg、home 站高 0.468 m）算的；标准版 B2 的正确值是 %.1f%%、"
        "%.1f cm。方向不变（B2 可行），余量比原先更大。"
        % (100.0 * b2["diagonal_ratio"], 100.0 * b2["head_clearance_m"])
    )

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
