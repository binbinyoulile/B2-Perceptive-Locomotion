"""B2 跨级姿态（前足踩 0.195 m 台阶、后足在地面）的静态搜索 + 2 s 动力学验证。

文件目的：
    解决"跨级姿态失稳"：机头要抬头让开上一级踏板，前腿护板不能磕台阶，
    kp=800 下关节不能顶限位、机身不能塌。脚本做三件事：
    1) 在 (pitch, 机身 x, 机身 z, 后足 x) 四维上做静态搜索，用 MuJoCo 的
       ``mj_geomDistance`` 直接量"机身/机头/前腿护板 vs 各级踏板"的距离；
    2) 对 A（只蹲低、pitch=0）与 B（抬头翘尾）两个最优候选各跑 4 s PD 站立，
       量前足法向力、后足滑移、机身高度稳定性；
    3) 把胜出的姿态打印成可直接抄进 ``configs/robot_config.B2_CROSS_STEP`` 的数值。

判据约定：
    - 净空用几何距离（m），不看"穿没穿"：>0 即不接触，>0.05 才算"让开"。
    - 前足踩第一级踏面中心（x=0.30657, z_top=0.195），后足踩地面（z=0）。
    - 机身绕自身原点俯仰（pitch up 为正），脚的世界坐标固定，靠逆解求关节角。

依赖：mujoco 3.16、numpy 2.x、configs.robot_config。
运行：D:\\Python313\\python.exe .\\experiments\\b2\\b2_step_pose.py
输出：终端 + experiments/b2/b2_step_pose.txt / .json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_ROBOT  # noqa: E402

import mujoco  # noqa: E402

SCENE = ROOT / "simulations" / "mujoco" / "b2_stairs_static" / "scene.xml"
OUT = ROOT / "experiments" / "b2" / "b2_step_pose.txt"

LEGS = ("FL", "FR", "RL", "RR")
HIP_X = np.array(B2_ROBOT.hip_x_m, dtype=np.float64)
FOOT_Y = np.array(B2_ROBOT.nominal_foot_y_m, dtype=np.float64)
L = B2_ROBOT.thigh_length_m
FOOT_R = B2_ROBOT.foot_radius_m
BODY_Y = 0.468  # 场景里钢梯宽度方向的中心

# 场景几何（从 up_step_*.xml 编译结果读出，见 docs/B2_VS_M1_REPORT.md）
TREAD_X = 0.306570
STEP1_TOP = 0.195
STEP2_TOP = 0.3703
STEP3_TOP = 0.5555


def load_scene():
    previous = Path.cwd()
    os.chdir(SCENE.parent)
    try:
        model = mujoco.MjModel.from_xml_path(SCENE.name)
    finally:
        os.chdir(previous)
    return model, mujoco.MjData(model)


def leg_ik(dx: float, dz: float) -> tuple[float, float] | None:
    """两连杆逆解（机身坐标系，矢状面）。dz 向下为负。"""
    reach = math.hypot(dx, dz)
    if reach > 2.0 * L - 0.02:
        return None
    psi = math.acos(max(-1.0, min(1.0, reach * reach / (2.0 * L * L) - 1.0)))
    return psi / 2.0 - math.atan2(dx, -dz), -psi


def classify_base_geoms(model) -> dict[str, list[int]]:
    """把机身碰撞体按位置分成 body / head / tail / lidar 四组。"""
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, B2_ROBOT.base_body_name)
    groups: dict[str, list[int]] = {"body": [], "head": [], "tail": [], "lidar": []}
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != base or model.geom_contype[g] <= 0:
            continue
        x, z = float(model.geom_pos[g, 0]), float(model.geom_pos[g, 2])
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER:
            groups["lidar"].append(g)
        elif x > 0.3:
            groups["head"].append(g)
        elif x < -0.3:
            groups["tail"].append(g)
        else:
            groups["body"].append(g)
    return groups


def classify_leg_geoms(model) -> list[int]:
    """四条小腿上的碰撞体（护板/小腿盒），用于"护板磕台阶"判据。"""
    ids = []
    for leg in LEGS:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) == b and model.geom_contype[g] > 0:
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    continue  # 足端球单独算（它本来就踩在踏面上）
                ids.append(g)
    return ids


@dataclass
class Pose:
    pitch_deg: float
    base_x: float
    base_z: float
    rear_x: float
    joints: list[float] = field(default_factory=list)
    thigh_front: float = 0.0
    calf_front: float = 0.0
    thigh_rear: float = 0.0
    calf_rear: float = 0.0
    reach_front: float = 0.0
    reach_rear: float = 0.0
    d_head: float = 0.0
    d_body: float = 0.0
    d_shin: float = 0.0
    joint_margin: float = 0.0

    @property
    def valid(self) -> bool:
        return self.d_head > 0.05 and self.d_body > 0.02 and self.d_shin > 0.005


class CrossStepSolver:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.base_geoms = classify_base_geoms(model)
        self.shin_geoms = classify_leg_geoms(model)
        self.stairs = {
            name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ("up_step_1", "up_step_2", "up_step_3")
        }
        self.fromto = np.zeros(6)

    def dist(self, geoms: list[int], stairs: tuple[str, ...]) -> float:
        best = 1.0
        for g in geoms:
            for name in stairs:
                sg = self.stairs[name]
                if sg < 0:
                    continue
                best = min(
                    best, mujoco.mj_geomDistance(self.model, self.data, g, sg, 0.5, self.fromto)
                )
        return best

    def build(self, pitch_deg: float, base_x: float, base_z: float, rear_x: float) -> Pose | None:
        """按目标脚位反解 12 个关节角，并用 mujoco 量净空。"""
        pose = Pose(pitch_deg, base_x, base_z, rear_x)
        a = math.radians(-pitch_deg)  # 绕 Y 轴负角 = 抬头
        c, s = math.cos(a), math.sin(a)
        qpos = np.zeros(self.model.nq)
        qpos[0], qpos[1], qpos[2] = base_x, BODY_Y, base_z
        qpos[3] = math.cos(a / 2.0)
        qpos[5] = math.sin(a / 2.0)
        joints: list[float] = []
        for i, (leg, hip_x) in enumerate(zip(LEGS, HIP_X)):
            on_step = hip_x > 0
            foot_x = TREAD_X if on_step else rear_x
            foot_z = (STEP1_TOP if on_step else 0.0) + FOOT_R
            dxw, dzw = foot_x - base_x, foot_z - base_z
            # 世界 → 机身：左乘 R(-a)
            lx = dxw * c - dzw * s
            lz = dxw * s + dzw * c
            sol = leg_ik(lx - hip_x, lz)
            if sol is None:
                return None
            thigh, calf = sol
            if not (-2.82 <= calf <= -0.43 and -0.94 <= thigh <= 4.69):
                return None
            if on_step:
                pose.thigh_front, pose.calf_front = thigh, calf
                pose.reach_front = math.hypot(lx - hip_x, lz)
            else:
                pose.thigh_rear, pose.calf_rear = thigh, calf
                pose.reach_rear = math.hypot(lx - hip_x, lz)
            joints.extend([0.0, thigh, calf])
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
            qpos[self.model.jnt_qposadr[j]] = 0.0
            for jn, val in (("thigh", thigh), ("calf", calf)):
                jj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{jn}_joint")
                qpos[self.model.jnt_qposadr[jj]] = val
        pose.joints = joints
        pose.joint_margin = min(
            pose.calf_front + 2.82, -0.43 - pose.calf_front,
            pose.calf_rear + 2.82, -0.43 - pose.calf_rear,
            pose.thigh_front + 0.94, 4.69 - pose.thigh_front,
            pose.thigh_rear + 0.94, 4.69 - pose.thigh_rear,
        )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        pose.d_head = self.dist(
            self.base_geoms["head"] + self.base_geoms["lidar"],
            ("up_step_1", "up_step_2", "up_step_3"),
        )
        pose.d_body = self.dist(self.base_geoms["body"], ("up_step_1", "up_step_2"))
        pose.d_shin = self.dist(self.shin_geoms, ("up_step_1",))
        return pose

    def measure(self) -> tuple[float, float, float]:
        """按当前 data 状态量 (机头, 机身, 前腿护板) 的净空。"""
        return (
            self.dist(self.base_geoms["head"] + self.base_geoms["lidar"],
                      ("up_step_1", "up_step_2", "up_step_3")),
            self.dist(self.base_geoms["body"], ("up_step_1", "up_step_2")),
            self.dist(self.shin_geoms, ("up_step_1",)),
        )

    def static_margins(self, pose: Pose) -> tuple[float, float]:
        """重心的前后余量（到后足/前足支撑边界），负值=重心在支撑外、必翻。"""
        com = pose.base_x + B2_ROBOT.com_offset_x_m
        return com - pose.rear_x, TREAD_X - com

    def search(self, pitch_values) -> list[Pose]:
        """粗网格 + 细网格。硬约束：净空、关节余量、重心得在支撑多边形里。"""
        found: list[Pose] = []

        def keep(pose: Pose | None) -> bool:
            if pose is None or not pose.valid or pose.joint_margin <= 0.08:
                return False
            rear, front = self.static_margins(pose)
            if not (rear > 0.05 and front > 0.05):
                return False
            # 腿的倾斜方向要自然：后足在后髋之后、前足在前髋之前不远处
            dx_rear = pose.rear_x - (pose.base_x - abs(HIP_X[0]))
            dx_front = TREAD_X - (pose.base_x + abs(HIP_X[0]))
            return -0.45 <= dx_rear <= -0.05 and -0.15 <= dx_front <= 0.25

        for pitch in pitch_values:
            for bx in np.arange(-0.05, 0.21, 0.025):
                for rx in np.arange(-0.50, -0.14, 0.05):
                    for h in np.arange(0.34, 0.561, 0.02):
                        pose = self.build(pitch, bx, h, rx)
                        if keep(pose):
                            found.append(pose)
        if not found:
            return []
        found.sort(key=lambda p: (p.d_head + 0.5 * p.joint_margin), reverse=True)
        refined: list[Pose] = []
        for base in found[:10]:
            for pitch in np.arange(max(0.0, base.pitch_deg - 2.0), base.pitch_deg + 2.1, 0.5):
                for bx in np.arange(base.base_x - 0.02, base.base_x + 0.021, 0.01):
                    for rx in np.arange(base.rear_x - 0.02, base.rear_x + 0.021, 0.01):
                        for h in np.arange(base.base_z - 0.02, base.base_z + 0.021, 0.005):
                            pose = self.build(pitch, bx, h, rx)
                            if keep(pose):
                                refined.append(pose)
        return refined or found


def pd_hold(solver: CrossStepSolver, joints: list[float], seconds: float):
    """按 cfg 的 kp/kd + 重力前馈保持目标角，返回逐控制步的统计。"""
    model, data = solver.model, solver.data
    nu = model.nu
    target = np.empty(nu)
    for a in range(nu):
        j = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        leg = name.split("_")[0]
        part = name.split("_")[1]
        idx = LEGS.index(leg) * 3 + ("hip", "thigh", "calf").index(part)
        target[a] = joints[idx]
    feet = [
        g
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        and abs(float(model.geom_size[g, 0]) - FOOT_R) < 1e-9
    ]
    foot0 = np.array([data.geom_xpos[g].copy() for g in feet])
    foot_mid = foot0.copy()
    order = np.argsort(-foot0[:, 0])  # 前足在 +x 侧，取最靠前的两只
    front_ids = [feet[i] for i in order[:2]]
    rear_ids = [feet[i] for i in order[2:]]
    control_hz = B2_ROBOT.fsm.control_hz
    substeps = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    zs, pitches = [], []
    mid_set = False
    t_start = data.time  # data 会在 A/B 之间复用，稳态窗口必须相对本次调用
    front_force_max = 0.0
    force = np.zeros(6)
    for _ in range(int(seconds * control_hz)):
        for a in range(nu):
            j = int(model.actuator_trnid[a, 0])
            dof = int(model.jnt_dofadr[j])
            q = data.qpos[model.jnt_qposadr[j]]
            tau = (
                B2_ROBOT.kp * (target[a] - q)
                - B2_ROBOT.kd * data.qvel[dof]
                + B2_ROBOT.fsm.gravity_feedforward_scale * data.qfrc_bias[dof]
            )
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = min(hi, max(lo, tau))
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        # 前足法向力（接触力的第 1 分量沿接触法向）
        for ci in range(data.ncon):
            c = data.contact[ci]
            if c.geom1 in front_ids or c.geom2 in front_ids:
                mujoco.mj_contactForce(model, data, ci, force)
                front_force_max = max(front_force_max, float(force[0]))
        if data.time - t_start >= seconds - 0.5 and not mid_set:
            foot_mid = np.array([data.geom_xpos[g].copy() for g in feet])
            mid_set = True
        zs.append(float(data.qpos[2]))
        q = data.qpos[3:7]
        pitches.append(
            -math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (q[0] * q[2] - q[3] * q[1])))))
        )
    foot1 = np.array([data.geom_xpos[g].copy() for g in feet])
    slip = np.linalg.norm(foot1[:, :2] - foot0[:, :2], axis=1) * 1000.0
    steady = np.linalg.norm(foot1[:, :2] - foot_mid[:, :2], axis=1) * 1000.0
    tail = np.array(zs[len(zs) // 2:])
    return {
        "z_mean": float(tail.mean()),
        "z_ptp_mm": float((tail.max() - tail.min()) * 1000.0),
        "pitch_end_deg": float(pitches[-1]),
        "front_force_max_n": front_force_max,
        "front_force_per_foot_n": front_force_max / 2.0,
        "rear_slip_mm": float(max(slip[i] for i in order[2:])),
        "rear_slip_steady_mm": float(max(steady[i] for i in order[2:])),
        "front_slip_mm": float(max(slip[i] for i in order[:2])),
        "settled_qpos": data.qpos.copy(),
    }


def render_pose(model, data, out_dir: Path) -> None:
    """把当前状态渲染成侧视 + 3/4 视两张 PNG（用于肉眼核对净空）。"""
    from PIL import Image

    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    for name, azim, elev in (("side", 90.0, -6.0), ("iso", 130.0, -18.0)):
        cam.lookat[:] = (0.15, 0.468, 0.32)
        cam.distance = 2.2
        cam.azimuth = azim
        cam.elevation = elev
        renderer.update_scene(data, camera=cam)
        Image.fromarray(renderer.render()).save(out_dir / f"b2_cross_step_{name}.png")
    renderer.close()


def main() -> int:
    model, data = load_scene()
    solver = CrossStepSolver(model, data)
    flat_candidates = [p for p in solver.search([0.0]) if abs(p.pitch_deg) < 1e-9]
    all_candidates = solver.search(np.arange(0.0, 25.1, 2.5))
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)

    say("=" * 88)
    say("B2 跨级姿态搜索：前足踩 0.195 m 第一级踏面，后足在地面")
    say("=" * 88)
    say(f"机器人：{B2_ROBOT.nominal_mass_kg:.3f} kg，腿长 {L:.2f}+{L:.2f} m，"
        f"足端 r={FOOT_R:.3f}，kp/kd={B2_ROBOT.kp}/{B2_ROBOT.kd}")
    say(f"场景：踏面中心 x={TREAD_X:.3f}，一级顶面 {STEP1_TOP:.3f} m，"
        f"二级顶面 {STEP2_TOP:.4f} m，三级顶面 {STEP3_TOP:.4f} m")
    say(f"候选（含 pitch=0 的 {len(flat_candidates)} 个）：机头净空>5cm、机身>2cm、"
        f"护板>5mm、关节余量>0.08rad、重心在支撑内且前后各留 5cm")
    if not all_candidates:
        print("\n".join(lines))
        return 1

    # A（只蹲低）：pitch=0 里机身最低的那一个
    flat_pool = [p for p in flat_candidates]
    best_a = min(flat_pool, key=lambda p: p.base_z) if flat_pool else None
    # B（抬头翘尾）：pitch>=8 里机头净空最大的一个
    pitch_pool = [p for p in all_candidates if p.pitch_deg >= 5.0
                  and p.d_head >= 0.15 and p.joint_margin >= 0.6]
    best_b = min(pitch_pool, key=lambda p: (p.pitch_deg, -p.joint_margin)) if pitch_pool else None

    pairs = (("A 只蹲低（pitch=0）", best_a), ("B 抬头翘尾", best_b))
    results: dict[str, dict] = {}
    for tag, pose in pairs:
        if pose is None:
            say()
            say(f"[{tag}] 无可行解")
            continue
        solver.build(pose.pitch_deg, pose.base_x, pose.base_z, pose.rear_x)
        dyn = pd_hold(solver, pose.joints, 4.0)
        results[tag] = {"pose": asdict(pose),
                        "dyn": {k: v for k, v in dyn.items() if k != "settled_qpos"}}
        say()
        say(f"[{tag}] pitch {pose.pitch_deg:.1f}°，机身 ({pose.base_x:+.3f}, "
            f"{BODY_Y:.3f}, {pose.base_z:.3f})，后足 x {pose.rear_x:+.3f}")
        say(f"    净空：机头 {pose.d_head*100:.1f} cm，机身 {pose.d_body*100:.1f} cm，"
            f"前腿护板 {pose.d_shin*100:.1f} cm；关节最小余量 {pose.joint_margin:.2f} rad")
        say(f"    前腿 thigh {pose.thigh_front:+.3f} / calf {pose.calf_front:+.3f}"
            f"（伸展 {pose.reach_front/0.7*100:.0f}%）；"
            f"后腿 thigh {pose.thigh_rear:+.3f} / calf {pose.calf_rear:+.3f}"
            f"（伸展 {pose.reach_rear/0.7*100:.0f}%）")
        say(f"    4 s PD：机身 z {dyn['z_mean']:.4f} m（下沉 {pose.base_z-dyn['z_mean']:+.3f}，"
            f"峰峰 {dyn['z_ptp_mm']:.2f} mm），末态 pitch {dyn['pitch_end_deg']:+.2f}°，"
            f"前足最大法向力 {dyn['front_force_max_n']:.0f} N（每足 "
            f"{dyn['front_force_per_foot_n']:.0f} N），后足滑移 {dyn['rear_slip_mm']:.2f} mm"
            f"（最后 0.5 s {dyn['rear_slip_steady_mm']:.2f} mm）")

    best = pairs[1][1] or pairs[0][1]
    solver.build(best.pitch_deg, best.base_x, best.base_z, best.rear_x)
    settled_dyn = pd_hold(solver, best.joints, 4.0)
    solver.data.qpos[:] = settled_dyn["settled_qpos"]
    solver.data.qvel[:] = 0.0
    mujoco.mj_forward(solver.model, solver.data)
    s_head, s_body, s_shin = solver.measure()
    say()
    say(f"落地后复测（机身 z {settled_dyn['z_mean']:.4f} m）：机头 {s_head*100:.1f} cm，"
        f"机身 {s_body*100:.1f} cm，前腿护板 {s_shin*100:.1f} cm")

    if "--render" in sys.argv:
        render_pose(solver.model, solver.data, ROOT / "experiments" / "b2")
        say("已渲染 b2_cross_step_side.png / b2_cross_step_iso.png")

    say()
    say("=" * 88)
    say(f"建议固化：pitch {best.pitch_deg:.1f}°，机身 ({best.base_x:+.4f}, "
        f"{BODY_Y:.3f}, {best.base_z:.4f})，后足 x {best.rear_x:+.4f}")
    say("12 维关节目标（FL/FR/RL/RR × hip/thigh/calf）：")
    say("    " + ", ".join(f"{v:+.4f}" for v in best.joints))

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
