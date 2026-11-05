"""Go2 在 0.195 m / 0.18 m 钢梯上的物理可行性测试（不训练、不长跑）。

文件目的：
    用 MuJoCo 的几何与接触数据回答三个问题：
    1) 给定四足落点，反解出的关节角是否越界、足端目标是否超出腿长；
    2) 在该姿态下，机身/大腿/小腿是否与相邻踏板发生碰撞；
    3) 是否存在“静力学稳定”（重心投影落在支撑多边形内）的姿态，
       以及静态保持所需力矩与足底摩擦是否在模型能力内。

整体输入：
    simulations/mujoco/go2_stairs_static/scene.xml（完整三层钢梯，含碰撞）。
整体输出：
    experiments/feasibility/report_195mm.txt / .json，以及终端 ASCII 摘要。
依赖与版本要求：
    Python 3.10+、mujoco 3.16、numpy 2.x。
运行方式：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\go2_195mm_feasibility.py
关键假设：
    1. 台阶几何全部从已编译模型读取，不硬编码 XML 数字；
    2. 足端球心目标 = 踏板顶面 + 足端半径；不使用外力辅助；
    3. “运动学可行”= 四足目标可达 + 关节在限位内 + 无机器人—踏板碰撞；
       “静态可站”= 在此基础上，重心投影落在四足支撑多边形内。
已知限制：
    这是静态/准静态几何与力矩测试，不证明动态攀爬一定成功；动态失败需要
    单独的控制实验。本脚本只用于排除/确认“物理死局”。
重要性等级：A（决定是否值得继续在 Go2 上投入训练）。
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulations" / "mujoco" / "go2_stairs_static" / "scene.xml"
OUT_DIR = ROOT / "experiments" / "feasibility"

sys.path.insert(0, str(ROOT))
import fsm_stairs_controller as fsm  # noqa: E402  复用教师自己的解析 IK 与几何常量

JOINT_NAMES = fsm.JOINT_NAMES
FOOT_NAMES = fsm.FOOT_GEOM_NAMES
HIP_X = fsm.HIP_X_M
HIP_Y = fsm.HIP_Y_M
THIGH_Y_OFFSET = fsm.LEG_THIGH_Y_OFFSET_M
LEG_L = fsm.THIGH_LENGTH_M
FOOT_R = fsm.FOOT_RADIUS_M
MAX_REACH = 2.0 * LEG_L
STANCE_BASE_ABOVE_FOOT_M = 0.2577  # 实测：稳定站立时 base 到足端球心的垂距
# 规则教师 leg_inverse_kinematics 内部把髋—足距离夹到 >=0.12 m（其安全下限），
# 而机构真实下限由膝限位决定（本文件运行时按 joint range 计算，约 0.0886 m）。
# 两者之差是“控制器自限”，不是物理极限，报告里必须分开写。
CONTROLLER_MIN_RADIUS_M = 0.12


def _quat_pitch(pitch_rad: float) -> np.ndarray:
    """绕机体 +Y 轴旋转的四元数（wxyz）。正角度 = 机头向下。"""
    return np.array(
        [math.cos(pitch_rad / 2.0), 0.0, math.sin(pitch_rad / 2.0), 0.0]
    )


def _rot_y(pitch_rad: float) -> np.ndarray:
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


class Rig:
    """封装一个已编译模型、关节地址、足端/踏板几何集合。"""

    def __init__(self, scene_path: Path) -> None:
        previous = Path.cwd()
        try:
            os.chdir(scene_path.parent)
            self.model = mujoco.MjModel.from_xml_path(scene_path.name)
        finally:
            os.chdir(previous)
        self.data = mujoco.MjData(self.model)
        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        self.qpos_adr = np.array(
            [self.model.jnt_qposadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.dof_adr = np.array(
            [self.model.jnt_dofadr[jid] for jid in self.joint_ids], dtype=np.int32
        )
        self.joint_low = np.asarray(self.model.jnt_range[self.joint_ids, 0])
        self.joint_high = np.asarray(self.model.jnt_range[self.joint_ids, 1])
        self.torque_low = np.asarray(self.model.actuator_ctrlrange[:, 0])
        self.torque_high = np.asarray(self.model.actuator_ctrlrange[:, 1])
        self.foot_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_NAMES
            ],
            dtype=np.int32,
        )
        self.base_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        self.thigh_bodies = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_thigh")
                for leg in FOOT_NAMES
            ],
            dtype=np.int32,
        )
        self.stair_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "steel_stair_collision"
        )
        self.stair_geoms = set()
        if self.stair_body >= 0:
            for geom_id in range(self.model.ngeom):
                if int(self.model.geom_bodyid[geom_id]) == self.stair_body:
                    self.stair_geoms.add(int(geom_id))
        self.robot_geoms = set()
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            while body_id > 0:
                if body_id == self.base_body:
                    self.robot_geoms.add(int(geom_id))
                    break
                parent = int(self.model.body_parentid[body_id])
                if parent == body_id:
                    break
                body_id = parent
        # 只统计机器人子树的真实质量（钢梯是独立静态 body，不能算进机器人）
        self.robot_mass = float(self.model.body_subtreemass[self.base_body])
        self.foot_id_set = {int(v) for v in self.foot_ids}

    def geom_label(self, geom_id: int) -> str:
        """给未命名碰撞体一个可读标签（body 名 + 类型 + id）。"""
        name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name:
            return name
        body = int(self.model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        type_name = {
            int(mujoco.mjtGeom.mjGEOM_BOX): "box",
            int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cyl",
            int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
            int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
        }.get(int(self.model.geom_type[geom_id]), "geom")
        return "%s/%s#%d" % (body_name or "?", type_name, geom_id)

    # -- 几何读取 ---------------------------------------------------------
    def treads(self) -> list[dict]:
        """从模型读出所有“踏板/平台”碰撞盒的世界包围盒与顶面高度。"""
        out = []
        for geom_id in sorted(self.stair_geoms):
            if int(self.model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name is None or not (
                name.startswith("up_step")
                or name.startswith("down_step")
                or name == "platform"
            ):
                continue
            pos = np.asarray(self.model.geom_pos[geom_id], dtype=np.float64)
            size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
            out.append(
                {
                    "name": name,
                    "center": pos.tolist(),
                    "half": size.tolist(),
                    "x_min": float(pos[0] - size[0]),
                    "x_max": float(pos[0] + size[0]),
                    "y_min": float(pos[1] - size[1]),
                    "y_max": float(pos[1] + size[1]),
                    "top_z": float(pos[2] + size[2]),
                }
            )
        out.sort(key=lambda item: item["top_z"])
        return out

    # -- 姿态设置 ---------------------------------------------------------
    def set_pose(
        self,
        base_xyz: np.ndarray,
        pitch_rad: float,
        feet_world: np.ndarray,
    ) -> dict:
        """把四足目标反解为关节角并写入 qpos，然后 mj_forward。"""
        rotation = _rot_y(pitch_rad)
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = base_xyz
        self.data.qpos[3:7] = _quat_pitch(pitch_rad)
        angles = np.zeros(12)
        for leg in range(4):
            local = rotation.T @ (feet_world[leg] - base_xyz)
            hip, thigh, calf = fsm.leg_inverse_kinematics(
                float(local[0]),
                float(local[1]),
                float(local[2]),
                HIP_X[leg],
                HIP_Y[leg],
                THIGH_Y_OFFSET[leg],
                hip_low_rad=float(self.joint_low[3 * leg]),
                hip_high_rad=float(self.joint_high[3 * leg]),
            )
            angles[3 * leg : 3 * leg + 3] = (hip, thigh, calf)
        self.data.qpos[self.qpos_adr] = angles
        mujoco.mj_forward(self.model, self.data)

        joint_margin = np.minimum(
            self.data.qpos[self.qpos_adr] - self.joint_low,
            self.joint_high - self.data.qpos[self.qpos_adr],
        )
        actual_feet = np.asarray(self.data.geom_xpos[self.foot_ids], dtype=np.float64)
        thigh_pos = np.asarray(self.data.xpos[self.thigh_bodies], dtype=np.float64)
        reach = np.linalg.norm(feet_world - thigh_pos, axis=1)

        collisions = []
        foot_contacts = 0
        for index in range(int(self.data.ncon)):
            contact = self.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if g1 in self.robot_geoms and g2 in self.stair_geoms:
                robot_geom, stair_geom = g1, g2
            elif g2 in self.robot_geoms and g1 in self.stair_geoms:
                robot_geom, stair_geom = g2, g1
            else:
                continue
            if robot_geom in self.foot_id_set:
                foot_contacts += 1
                continue
            # 只把真实穿透（>2 mm）算作碰撞；擦边接触不计。
            if float(contact.dist) > -0.002:
                continue
            collisions.append(
                (self.geom_label(robot_geom), self.geom_label(stair_geom))
            )

        binding = int(np.argmin(joint_margin))
        return {
            "angles": angles,
            "joint_margin_min": float(np.min(joint_margin)),
            "binding_joint": JOINT_NAMES[binding],
            "joint_margin_per_joint": joint_margin.tolist(),
            "reach": reach.tolist(),
            "reach_ok": bool(
                np.all(reach <= MAX_REACH + 1e-9)
                and np.all(reach >= CONTROLLER_MIN_RADIUS_M - 1e-9)
            ),
            "reach_max": float(np.max(reach)),
            "reach_min": float(np.min(reach)),
            "ik_error": float(np.max(np.abs(actual_feet - feet_world))),
            "collisions": collisions,
            "foot_contacts": foot_contacts,
            "com": np.asarray(self.data.subtree_com[self.base_body]).tolist(),
        }

    def gravity_torque(self) -> np.ndarray:
        """返回当前姿态下各关节的重力/科氏项（rad/s=0 时即重力力矩）。"""
        return np.asarray(self.data.qfrc_bias[self.dof_adr], dtype=np.float64)


def _support_margin(com_xy: np.ndarray, feet_xy: np.ndarray) -> float:
    """返回重心投影到四足支撑多边形边界的最短距离；在外部返回负数。

    足端顺序为 FL/FR/RL/RR（世界 +X 前、+Y 左）。这里先按 FL->RL->RR->FR
    采样，再按有符号面积强制成逆时针，避免绕向写反导致“内部点被当成外部”。
    """
    poly = np.asarray(feet_xy[[0, 2, 3, 1]], dtype=np.float64)
    area = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        area += x1 * y2 - x2 * y1
    if area < 0.0:
        poly = poly[::-1]
    inside = True
    min_dist = 1e9
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        edge = b - a
        length = float(np.linalg.norm(edge))
        if length < 1e-9:
            continue
        # 逆时针多边形：左法线指向内部
        normal = np.array((-edge[1], edge[0])) / length
        side = float(np.dot(com_xy - a, normal))
        if side < 0.0:
            inside = False
        min_dist = min(min_dist, abs(side))
    return float(min_dist if inside else -min_dist)


def _evaluate_family(
    rig: Rig,
    front_tread: dict,
    rear_tread: dict | None,
    front_x_range: tuple[float, float],
    rear_x_range: tuple[float, float],
    base_x_range: tuple[float, float],
    base_z_range: tuple[float, float],
    pitch_range_deg: tuple[float, float],
    label: str,
) -> dict:
    """在给定地形族内网格搜索：可达 + 关节在限位内 + 无碰撞 + 重心在支撑多边形内。"""
    front_top = front_tread["top_z"] + FOOT_R
    rear_top = rear_tread["top_z"] + FOOT_R if rear_tread is not None else FOOT_R
    kinematic = 0
    stable = 0
    total = 0
    reason = {"reach": 0, "joint": 0, "collision": 0, "unstable": 0}
    collision_counter: Counter = Counter()
    best_stable = None
    best_kinematic = None
    for base_x in np.arange(*base_x_range, 0.02):
        for base_z in np.arange(*base_z_range, 0.02):
            for pitch_deg in np.arange(
                pitch_range_deg[0], pitch_range_deg[1] + 1e-9, 3.0
            ):
                for fx in np.linspace(*front_x_range, 3):
                    for rx in np.linspace(*rear_x_range, 3):
                        feet = np.array(
                            [
                                [fx, 0.468 + 0.142, front_top],
                                [fx, 0.468 - 0.142, front_top],
                                [rx, 0.468 + 0.142, rear_top],
                                [rx, 0.468 - 0.142, rear_top],
                            ]
                        )
                        report = rig.set_pose(
                            np.array([base_x, 0.468, base_z]),
                            math.radians(pitch_deg),
                            feet,
                        )
                        total += 1
                        if not report["reach_ok"] or report["ik_error"] > 0.02:
                            reason["reach"] += 1
                            continue
                        if report["joint_margin_min"] < 0.0:
                            reason["joint"] += 1
                            continue
                        kinematic += 1
                        entry = {
                            "base_x": float(base_x),
                            "base_z": float(base_z),
                            "pitch_deg": float(pitch_deg),
                            "front_x": float(fx),
                            "rear_x": float(rx),
                            "joint_margin_min": float(report["joint_margin_min"]),
                            "binding_joint": report["binding_joint"],
                            "reach_max": float(report["reach_max"]),
                            "reach_min": float(report["reach_min"]),
                            "ik_error": float(report["ik_error"]),
                            "angles": report["angles"].tolist(),
                            "feet": feet.tolist(),
                            "com": report["com"],
                        }
                        if report["collisions"]:
                            reason["collision"] += 1
                            for pair in report["collisions"]:
                                collision_counter[pair] += 1
                            continue
                        margin = _support_margin(
                            np.asarray(report["com"][:2]), feet[:, :2]
                        )
                        entry["com_margin"] = float(margin)
                        if best_kinematic is None or margin > best_kinematic["com_margin"]:
                            best_kinematic = entry
                        if margin <= 0.005:
                            reason["unstable"] += 1
                            continue
                        stable += 1
                        if best_stable is None or margin > best_stable["com_margin"]:
                            best_stable = entry
    return {
        "label": label,
        "total_poses": total,
        "kinematic_poses": kinematic,
        "stable_poses": stable,
        "rejected": reason,
        "collision_pairs": collision_counter.most_common(6),
        "best_stable": best_stable,
        "best_kinematic": best_kinematic,
    }


def _describe_family(say, family: dict) -> None:
    say("%s" % family["label"])
    say(
        "  采样=%d  运动学可行=%d  静态可站(重心余量>5mm)=%d  被拒=%s"
        % (
            family["total_poses"],
            family["kinematic_poses"],
            family["stable_poses"],
            family["rejected"],
        )
    )
    for pair, count in family["collision_pairs"]:
        say("    碰撞热点 %s x %s: %d 次" % (pair[0], pair[1], count))
    best = family["best_stable"] or family["best_kinematic"]
    if best:
        say(
            "  最优(重心余量 %.4f m)：base=(%.3f, %.3f) pitch=%.1f° "
            "前足x=%.3f 后足x=%.3f"
            % (
                best["com_margin"],
                best["base_x"],
                best["base_z"],
                best["pitch_deg"],
                best["front_x"],
                best["rear_x"],
            )
        )
        say(
            "     最小关节余量=%.4f rad（受限关节=%s）  可达距离=%.4f~%.4f/%.4f m  IK残差=%.4f m"
            % (
                best["joint_margin_min"],
                best.get("binding_joint", "?"),
                best.get("reach_min", 0.0),
                best["reach_max"],
                MAX_REACH,
                best["ik_error"],
            )
        )
    else:
        say("  最优：无可行姿态")
    say("")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rig = Rig(SCENE)
    treads = rig.treads()
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)

    say("=" * 78)
    say("Go2 在 0.195 m / 0.18 m 钢梯上的物理可行性测试")
    say("=" * 78)
    say(
        "机器人质量: %.3f kg（仅 base 子树）  刚体数: %d  nq=%d nv=%d nu=%d"
        % (rig.robot_mass, rig.model.nbody, rig.model.nq, rig.model.nv, rig.model.nu)
    )
    say(
        "腿长(大腿=小腿)=%.3f m  最大可达半径=%.3f m  足端半径=%.3f m"
        % (LEG_L, MAX_REACH, FOOT_R)
    )
    say(
        "前腿限位 hip=%s thigh=%s calf=%s"
        % (
            (rig.joint_low[0], rig.joint_high[0]),
            (rig.joint_low[1], rig.joint_high[1]),
            (rig.joint_low[2], rig.joint_high[2]),
        )
    )
    say(
        "后腿限位 hip=%s thigh=%s calf=%s"
        % (
            (rig.joint_low[6], rig.joint_high[6]),
            (rig.joint_low[7], rig.joint_high[7]),
            (rig.joint_low[8], rig.joint_high[8]),
        )
    )
    say(
        "力矩上限 hip/thigh/calf = %s N·m"
        % np.round([rig.torque_high[0], rig.torque_high[1], rig.torque_high[2]], 2).tolist()
    )
    say()
    say("--- 钢梯踏板几何（从已编译模型读取，非硬编码）---")
    for tread in treads:
        say(
            "  %-16s 顶面 z=%.4f  x=[%.4f, %.4f]  深度=%.4f  y=[%.4f, %.4f]"
            % (
                tread["name"],
                tread["top_z"],
                tread["x_min"],
                tread["x_max"],
                tread["x_max"] - tread["x_min"],
                tread["y_min"],
                tread["y_max"],
            )
        )
    up_steps = [t for t in treads if t["name"].startswith("up_step")]
    up_steps.sort(key=lambda t: t["x_min"])
    up_rises = [up_steps[i + 1]["top_z"] - up_steps[i]["top_z"] for i in range(len(up_steps) - 1)]
    up_runs = [up_steps[i + 1]["x_min"] - up_steps[i]["x_min"] for i in range(len(up_steps) - 1)]
    up_gaps = [up_steps[i + 1]["x_min"] - up_steps[i]["x_max"] for i in range(len(up_steps) - 1)]
    say("  上楼梯相邻高差: %s m" % np.round(up_rises, 4).tolist())
    say("  上楼梯相邻前缘间距(踏面+间隙): %s m" % np.round(up_runs, 4).tolist())
    say("  踏板之间的水平间隙: %s m" % np.round(up_gaps, 4).tolist())
    say()

    up1, up2 = up_steps[0], up_steps[1]

    family_f1 = _evaluate_family(
        rig,
        front_tread=up1,
        rear_tread=None,
        front_x_range=(up1["x_min"] + 0.05, up1["x_max"] - 0.03),
        rear_x_range=(-0.10, 0.10),
        base_x_range=(up1["x_min"] - 0.02, up1["x_max"] - 0.02),
        base_z_range=(0.26, 0.40),
        pitch_range_deg=(-6.0, 6.0),
        label="F1 前足搭第1级、后足在地面（跨步瞬间）",
    )
    family_f2 = _evaluate_family(
        rig,
        front_tread=up2,
        rear_tread=up1,
        front_x_range=(up2["x_min"] + 0.05, up2["x_max"] - 0.04),
        rear_x_range=(up1["x_min"] + 0.04, up1["x_max"] - 0.05),
        base_x_range=(up1["x_max"] - 0.04, up2["x_max"] - 0.06),
        base_z_range=(0.30, 0.46),
        pitch_range_deg=(-6.0, 6.0),
        label="F2 前足在第2级、后足在第1级（真实爬梯的对角姿态）",
    )
    family_f3 = _evaluate_family(
        rig,
        front_tread=up1,
        rear_tread=up1,
        front_x_range=(up1["x_min"] + 0.05, up1["x_max"] - 0.03),
        rear_x_range=(up1["x_min"] + 0.03, up1["x_max"] - 0.05),
        base_x_range=(up1["x_min"] + 0.02, up1["x_max"] + 0.01),
        base_z_range=(0.28, 0.42),
        pitch_range_deg=(-6.0, 6.0),
        label="F3 四足都在第1级踏板（环境成功判定所要求的姿态）",
    )

    say("--- 姿态可行性网格搜索（判据见文件头）---")
    for family in (family_f1, family_f2, family_f3):
        _describe_family(say, family)

    say("--- 悬空折叠极限（前足能否抬过 0.195 m 踏板边缘）---")
    calf_min = float(rig.joint_low[2])
    radius_min = 2.0 * LEG_L * math.cos(abs(calf_min) / 2.0)
    need_foot_center = up1["top_z"] + FOOT_R
    body_needed = need_foot_center + radius_min
    flat_body = 0.0093 + STANCE_BASE_ABOVE_FOOT_M
    say("  膝限位=%.4f rad -> 髋-足最小距离=%.4f m" % (calf_min, radius_min))
    say("  抬过边缘所需足端球心高度=%.4f m" % need_foot_center)
    say("  对应所需机身(大腿关节)高度=%.4f m；平地站立=%.4f m" % (body_needed, flat_body))
    say("  需要抬升机身=%.4f m（直腿极限可抬 0.168 m）" % (body_needed - flat_body))
    say(
        "  控制器 IK 自限: 髋-足距离 >= %.3f m；机构极限 >= %.4f m（死区 %.4f m）"
        % (CONTROLLER_MIN_RADIUS_M, radius_min, CONTROLLER_MIN_RADIUS_M - radius_min)
    )
    say()

    best = family_f2["best_stable"] or family_f1["best_stable"]
    if best:
        report = rig.set_pose(
            np.array([best["base_x"], 0.468, best["base_z"]]),
            math.radians(best["pitch_deg"]),
            np.asarray(best["feet"]),
        )
        gravity = rig.gravity_torque()
        ratio = np.abs(gravity) / np.maximum(np.abs(rig.torque_high), 1e-6)
        say("--- 静态保持需求（最优稳定姿态）---")
        say("  姿态: base=(%.3f, %.3f) pitch=%.1f°" % (
            best["base_x"], best["base_z"], best["pitch_deg"]))
        say("  重力/科氏关节力矩: %s N·m" % np.round(gravity, 2).tolist())
        say("  力矩占用率: 最大 %.1f%%  平均 %.1f%%" % (
            100.0 * float(np.max(ratio)), 100.0 * float(np.mean(ratio))))
        say("  足-踏板接触数: %d  非足碰撞: %s" % (
            report["foot_contacts"], report["collisions"] or "无"))
        stand = np.asarray(best["angles"], dtype=np.float64)
        kp = np.tile(np.array((130.0, 144.0, 144.0)), 4)
        kd = np.tile(np.array((2.7, 3.5, 3.5)), 4)
        start_base = np.asarray(rig.data.qpos[:3], dtype=np.float64).copy()
        start_feet = np.asarray(rig.data.geom_xpos[rig.foot_ids], dtype=np.float64).copy()
        max_friction = 0.0
        max_tau_ratio = 0.0
        for _ in range(int(0.3 * 50)):
            for _sub in range(10):
                q = np.asarray(rig.data.qpos[rig.qpos_adr])
                qd = np.asarray(rig.data.qvel[rig.dof_adr])
                tau = kp * (stand - q) - kd * qd + np.asarray(
                    rig.data.qfrc_bias[rig.dof_adr]
                )
                rig.data.ctrl[:] = np.clip(tau, rig.torque_low, rig.torque_high)
                mujoco.mj_step(rig.model, rig.data)
            max_tau_ratio = max(
                max_tau_ratio,
                float(np.max(np.abs(rig.data.actuator_force) / np.abs(rig.torque_high))),
            )
        # 摩擦比取“静置结束帧”，避免把落地瞬态当成稳态需求。
        final_friction = 0.0
        for index in range(int(rig.data.ncon)):
            contact = rig.data.contact[index]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if not (
                (g1 in rig.foot_id_set and g2 in rig.stair_geoms)
                or (g2 in rig.foot_id_set and g1 in rig.stair_geoms)
            ):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(rig.model, rig.data, index, force)
            normal = abs(float(force[0]))
            tangential = float(np.hypot(force[1], force[2]))
            if normal > 1e-6:
                final_friction = max(final_friction, tangential / normal)
        end_base = np.asarray(rig.data.qpos[:3], dtype=np.float64)
        end_feet = np.asarray(rig.data.geom_xpos[rig.foot_ids], dtype=np.float64)
        slip = np.linalg.norm(end_feet[:, :2] - start_feet[:, :2], axis=1)
        say("  静置0.3s: 机身下沉 %.4f m  最大足端滑移 %.4f m" % (
            float(start_base[2] - end_base[2]), float(np.max(slip))))
        say("  实测最大力矩占用率 %.1f%%" % (100.0 * max_tau_ratio))
        say("  静置末帧足底 切向/法向 比 %.3f（摩擦系数上限 0.8）" % final_friction)

    say()
    say("--- 单级台阶高度上限（解析估计，不是仿真）---")
    say("  判据：抬脚折叠余量 + 站姿可达 + 机身底面越过下一级踏板。")
    body_half_height = 0.057
    body_half_length = 0.188
    for leg_len in (0.213, 0.25, 0.28, 0.30, 0.32):
        stand_h = 1.21 * leg_len
        rise_limit = stand_h - body_half_height - 0.005
        foot_lift = stand_h - 2.0 * leg_len * math.cos(abs(calf_min) / 2.0)
        overlap = body_half_length - 0.5 * 0.18
        say(
            "  L=%.3f m: 站立高=%.3f  机身净空允许高差<=%.3f m  "
            "该腿长可抬足高度=%.3f m  机身与下级踏板重叠 %.3f m"
            % (leg_len, stand_h, rise_limit, foot_lift, overlap)
        )

    text = "\n".join(lines)
    (OUT_DIR / "report_195mm.txt").write_text(text, encoding="utf-8")
    summary = {
        "robot_mass_kg": rig.robot_mass,
        "leg_length_m": LEG_L,
        "max_reach_m": MAX_REACH,
        "treads": treads,
        "families": {
            f["label"]: {k: v for k, v in f.items() if k != "collision_pairs"}
            for f in (family_f1, family_f2, family_f3)
        },
        "collision_pairs": {
            f["label"]: f["collision_pairs"] for f in (family_f1, family_f2, family_f3)
        },
        "swing": {
            "calf_limit_rad": calf_min,
            "radius_min_m": radius_min,
            "needed_foot_center_m": need_foot_center,
            "needed_body_height_m": body_needed,
        },
    }
    (OUT_DIR / "report_195mm.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("WROTE", (OUT_DIR / "report_195mm.txt"))
    for family in (family_f1, family_f2, family_f3):
        print(
            "%s: kinematic=%d stable=%d / %d"
            % (
                family["label"].split(" ")[0],
                family["kinematic_poses"],
                family["stable_poses"],
                family["total_poses"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
