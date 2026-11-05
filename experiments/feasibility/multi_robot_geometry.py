"""三台四足机器人在 0.195 m / 0.18 m 钢梯上的几何可行性对比。

文件目的：
    用同一套判据比较 Go2 / Unitree B2 / GENISOM M1：
    1) 四足同踩一级时，支撑多边形深度与最大重心余量（只由踏面与足端半径决定）；
    2) 四足同踩一级所需的前后足中心距与腿长占用；
    3) 站上一级踏板时，机身/机头与上一级踏板的静态净空；
    4) 逐级对角（前足高一级、后足低一级）时后腿需要的伸展量。
整体输入：
    台阶几何从 scene.xml 编译结果读取；机器人几何来自：
      - Go2：configs/robot_config.py（本仓库实测）
      - B2 ：unitree_mujoco/unitree_robots/b2/b2.xml（官方 MJCF，2026-11 取）
      - M1 ：公开资料（41 kg、站立 920×480×585 mm、标称爬梯 25 cm）+ 显式假设
整体输出：
    终端表格 + experiments/feasibility/report_multi_robot.txt
依赖与版本要求：
    Python 3.10+、mujoco 3.16、numpy 2.x。
运行方式：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\multi_robot_geometry.py
关键假设：
    M1 没有公开 MJCF/URDF（genisom_model 只有 zsl-1/zsl-1w，MATRiX 机器人
    列表里也没有 m1），其腿长、足端半径、机头下沿位置均为区间假设，报告中
    必须标注“估计值”，不能当成实测。
已知限制：
    只做静态几何，不含动力学与接触时序；支撑多边形按足端目标点计算。
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

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulations" / "mujoco" / "go2_stairs_static" / "scene.xml"
OUT = ROOT / "experiments" / "feasibility" / "report_multi_robot.txt"

sys.path.insert(0, str(ROOT))
from configs.robot_config import GO2_ROBOT  # noqa: E402

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
        body_half_length_m：机身碰撞盒半长（含头尾凸出）。
        body_half_height_m：机身碰撞盒半高。
        stand_height_m：稳定站立时机身中心到足端球心的垂距。
        head_bottom_m：机头最低点相对机身中心的高度（负值）。
        com_offset_x_m：整机重心相对机身中心的 X 偏移。
        mass_kg：质量（记录值）。
        source：参数来源说明。
    """

    name: str
    hip_spacing_m: float
    max_reach_m: float
    foot_radius_m: float
    body_half_length_m: float
    body_half_height_m: float
    stand_height_m: float
    head_bottom_m: float
    com_offset_x_m: float
    mass_kg: float
    source: str


ROBOTS = [
    RobotGeometry(
        name="Go2",
        hip_spacing_m=2.0 * abs(GO2_ROBOT.hip_x_m[0]),
        max_reach_m=GO2_ROBOT.thigh_length_m + GO2_ROBOT.calf_length_m,
        foot_radius_m=GO2_ROBOT.foot_radius_m,
        body_half_length_m=GO2_ROBOT.body_half_length_m,
        body_half_height_m=GO2_ROBOT.body_half_height_m,
        stand_height_m=0.2577,
        head_bottom_m=-0.107,
        com_offset_x_m=0.0211,
        mass_kg=GO2_ROBOT.nominal_mass_kg,
        source="本仓库 MJCF + 实测（configs/robot_config.py）",
    ),
    RobotGeometry(
        name="Unitree B2",
        hip_spacing_m=2.0 * 0.3285,
        max_reach_m=0.35 + 0.35,
        foot_radius_m=0.032,
        body_half_length_m=0.25,
        body_half_height_m=0.075,
        stand_height_m=0.468,  # MJCF home: base z=0.5，足端球心 z=0.032
        head_bottom_m=-0.07,
        com_offset_x_m=0.0213,
        mass_kg=83.5,  # MJCF 合计；厂商标称约 60 kg，差异见报告
        source="unitree_mujoco/unitree_robots/b2/b2.xml（官方 MJCF）",
    ),
    RobotGeometry(
        name="GENISOM M1（估计）",
        hip_spacing_m=0.55,
        max_reach_m=0.44,
        foot_radius_m=0.035,
        body_half_length_m=0.46,
        body_half_height_m=0.11,
        stand_height_m=0.33,
        head_bottom_m=-0.10,
        com_offset_x_m=0.02,
        mass_kg=41.0,
        source="公开资料 + 假设（无公开 MJCF/URDF，需厂商确认）",
    ),
]


def evaluate(robot: RobotGeometry) -> dict:
    """按 0.195/0.18 场景计算四项判据。

    返回字典：
        support_depth_m：四足同踩一级时支撑多边形的前后深度上界。
        com_margin_m：重心余量上界（支撑深度的一半减去重心偏移）。
        required_reach_f3_m：四足同踩一级时单腿需要的髋—足距离。
        reach_ratio_f3：该距离占最大可达半径的比例。
        body_clearance_m：站上一级时，机身底面到上一级踏板的净空。
        head_clearance_m：同上，但用机头最低点（更容易先撞）。
        diagonal_rear_drop_m：对角姿态下后腿需要的垂直伸展（含级间高差）。
        diagonal_rear_reach_m：假设后足在髋后 0.15 m 时的髋—足距离。
        diagonal_ratio：该距离占最大可达半径的比例。
        diagonal_level_possible：水平机身时对角姿态是否可达。
    """
    reach_max = robot.max_reach_m
    support_depth = TREAD_DEPTH_M - 2.0 * robot.foot_radius_m
    com_margin = 0.5 * support_depth - abs(robot.com_offset_x_m)
    # 四足同踩一级：前后足中心距最大 = support_depth，每条腿要向后/向前
    # 伸展 (hip_spacing - support_depth)/2，同时还要下降 stand_height。
    required_reach = math.hypot(
        0.5 * (robot.hip_spacing_m - support_depth), robot.stand_height_m
    )
    body_clearance = (
        robot.foot_radius_m + robot.stand_height_m
        - robot.body_half_height_m - RISE_M
    )
    head_clearance = (
        robot.foot_radius_m + robot.stand_height_m
        + robot.head_bottom_m - RISE_M
    )
    diagonal_drop = robot.stand_height_m + RISE_M
    diagonal_reach = math.hypot(0.15, diagonal_drop)
    return {
        "support_depth_m": support_depth,
        "com_margin_m": com_margin,
        "required_reach_f3_m": required_reach,
        "reach_ratio_f3": required_reach / reach_max,
        "body_clearance_m": body_clearance,
        "head_clearance_m": head_clearance,
        "diagonal_rear_drop_m": diagonal_drop,
        "diagonal_rear_reach_m": diagonal_reach,
        "diagonal_ratio": diagonal_reach / reach_max,
        "diagonal_level_possible": diagonal_reach <= reach_max,
    }


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main() -> int:
    lines = []
    def say(text: str = "") -> None:
        lines.append(text)

    say("=" * 92)
    say("三台四足机器人在 0.195 m 台阶 / 0.18 m 踏面上的几何可行性")
    say("=" * 92)
    say(
        "场景：踏面深 %.4f m，第一级顶面 %.4f m，级间高差 %.4f m（从 scene.xml 编译结果读取）"
        % (TREAD_DEPTH_M, STEP1_TOP_M, RISE_M)
    )
    say()
    say(
        "判据说明：支撑深度 = 踏面深 - 2×足端半径，与腿长无关；重心余量上界 = "
        "支撑深度/2 - |重心偏移|。"
    )
    say()
    header = (
        "%-18s %8s %8s %8s %9s %9s %9s %9s"
        % ("机器人", "可达m", "支撑深", "重心余", "同踩需伸", "占比", "机头净空", "对角占比")
    )
    say(header)
    say("-" * 92)
    results = {}
    for robot in ROBOTS:
        item = evaluate(robot)
        results[robot.name] = item
        say(
            "%-18s %8.3f %8.3f %8.3f %9.3f %8.1f%% %9.3f %8.1f%%"
            % (
                robot.name,
                robot.max_reach_m,
                item["support_depth_m"],
                item["com_margin_m"],
                item["required_reach_f3_m"],
                100.0 * item["reach_ratio_f3"],
                item["head_clearance_m"],
                100.0 * item["diagonal_ratio"],
            )
        )
    say()
    say("详细数值：")
    for robot in ROBOTS:
        item = results[robot.name]
        say("  [%s] 来源: %s" % (robot.name, robot.source))
        say(
            "    质量(记录)=%.1f kg  髋距=%.3f m  站高=%.3f m  足端半径=%.3f m"
            % (
                robot.mass_kg,
                robot.hip_spacing_m,
                robot.stand_height_m,
                robot.foot_radius_m,
            )
        )
        say(
            "    四足同踩一级: 需要髋-足 %.3f m（占可达 %.1f%%）；机身底面净空 %.3f m；"
            "机头最低点净空 %.3f m"
            % (
                item["required_reach_f3_m"],
                100.0 * item["reach_ratio_f3"],
                item["body_clearance_m"],
                item["head_clearance_m"],
            )
        )
        say(
            "    逐级对角(水平机身): 后腿需伸展 %.3f m（占可达 %.1f%%）→ %s"
            % (
                item["diagonal_rear_reach_m"],
                100.0 * item["diagonal_ratio"],
                "可达" if item["diagonal_level_possible"] else "不可达，必须抬机头或蹲低",
            )
        )
        if item["head_clearance_m"] < 0.0:
            say("    [!] 机头在水平机身下会与上一级踏板重叠，必须抬机头（Go2 实测需要 14°~30°）")
        say()

    say("-" * 92)
    say("结论要点：")
    say("1) 0.18 m 踏面把四足同踩一级的支撑深度压到 %.2f~%.2f m，重心余量上界只有 "
        "%.1f~%.1f cm；这个数只由踏面与足端半径决定，换更长的腿不会变大。"
        % (
            min(r["support_depth_m"] for r in results.values()),
            max(r["support_depth_m"] for r in results.values()),
            100.0 * min(r["com_margin_m"] for r in results.values()),
            100.0 * max(r["com_margin_m"] for r in results.values()),
        ))
    say("2) 换狗真正改善的是“机身/机头净空”和“腿部伸展余量”：Go2 机头净空为负，"
        "B2 约 0.24 m；对角姿态的伸展占比 Go2 >100%%（水平时不可达），B2 约 %.0f%%。"
        % (100.0 * results["Unitree B2"]["diagonal_ratio"]))
    say("3) 因此三台车在这个场景都应采用“逐级对角”，但 Go2 需要抬机头 14°~30°、"
        "B2 需要蹲低或抬机头来换取后腿余量。踏面宽度是比腿长更硬的约束。")

    text = "\n".join(lines)
    OUT.write_text(text, encoding="utf-8")
    (OUT.with_suffix(".json")).write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
