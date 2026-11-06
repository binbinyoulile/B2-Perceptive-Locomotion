"""AS2 能否吃下 0.195 m 高 / 0.18 m 深钢梯：关键站姿 IK + PD 实测 + 重心窗口。

文件目的：
    B2 在 0.18 m 窄踏面上被"四足同踩一级"卡住（支撑深 0.116 m、重心窗口
    ±3.7 cm）。换平台前先回答一个可量化的问题：**这台狗的腿够不够长、
    髋距够不够短，能不能在 0.18 m 踏面上摆出三个关键站姿并保持住**：
      S1 平地站（基线，标定 PD）；
      S2 四足同踩一级（0.18 m 踏面站四只脚 —— 任何四足都绕不开的姿态）；
      S3 跨级（前足踩上一级、后足留本级）。
    同时算两个指标，用来横向比较机型：
      可挪动量：四足钉死时，机身前后还能挪多少（腿的可达范围余量）；
      需要量：抬起一只前足时，重心投影到"另三足三角形"边界的距离。
    可挪动量 > 需要量 才谈得上"先挪重心再抬脚"。

输入：
    simulations/mujoco/as2_stairs_static/scene.xml（官方 AS2 MJCF + 共用钢梯）
输出：
    experiments/as2/as2_stance_probe.txt / .json，可选 --render 出 PNG
依赖：
    mujoco 3.16、numpy、configs.robot_config
运行：
    D:\\Python313\\python.exe .\\experiments\\as2\\as2_stance_probe.py [--render]

关键假设：
    - 腿是矢状面两连杆（髋 roll 保持 0），足端当球体；逆解用 MuJoCo 实测
      校核过的 FK 约定（见 leg_fk / leg_ik，IK 残差 0.000 mm）。
    - 官方 MJCF 质量 17.64 kg（不含电池的连杆质量），厂商 20 kg 含电池；
      力矩上限用 MJCF 的 60/60/90 N·m。

已知限制：
    - 只做静态/准静态站姿验证，不涉及腾空相、不涉及真机延迟。
    - 踏面可行区间按"足心离踏面边缘 ≥ 足半径"的保守约定算（与 B2 报告一致）。
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
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

SCENE = ROOT / "simulations" / "mujoco" / "as2_stairs_static" / "scene.xml"
OUT = ROOT / "experiments" / "as2" / "as2_stance_probe.txt"

LEGS = ("FL", "FR", "RL", "RR")
BODY_Y = 0.468  # 钢梯宽度中心

# 钢梯几何（从 stair_bodies.xml 编译结果读出）
TREAD_DEPTH = 0.18
STEP_PITCH = 0.185262
STEP1_CENTER = 0.306570
STEP1_TOP = 0.195
STEP2_CENTER = 0.491832
STEP2_TOP = 0.3803
STEP3_TOP = 0.5655  # pos 0.5555 + 半厚 0.01（旧 B2 脚本写的 0.5555 是中心，差 1 cm）
STAIR_GEOMS = ("up_step_1", "up_step_2", "up_step_3", "platform")


def load_scene():
    previous = Path.cwd()
    os.chdir(SCENE.parent)
    try:
        model = mujoco.MjModel.from_xml_path(SCENE.name)
    finally:
        os.chdir(previous)
    return model, mujoco.MjData(model)


@dataclass
class Robot:
    """从 MJCF 读出来的、算站姿只需要的那几个数。"""

    name: str
    hip_x: float
    hip_y: float
    thigh: float
    calf: float
    foot_r: float
    foot_y: float
    r_max: float
    r_min: float
    mass: float
    calf_range: tuple[float, float]
    thigh_range: tuple[float, float]
    torque: tuple[float, float]


def read_robot(model, name: str = "AS2") -> Robot:
    base = model.body("base_link").id
    thigh_body = model.body("FL_thigh")
    calf_body = model.body("FL_calf")
    l1 = float(np.linalg.norm(calf_body.pos))  # FL_thigh → FL_calf
    foot_geom = [
        g
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == calf_body.id
        and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]
    g = foot_geom[0]
    l2 = float(-model.geom_pos[g, 2])
    foot_r = float(model.geom_size[g, 0])
    hip = model.body("FL_hip")
    thigh_j = model.joint("FL_thigh_joint")
    calf_j = model.joint("FL_calf_joint")
    c_lo, c_hi = model.jnt_range[calf_j.id]
    # 小腿限位 c_hi（-0.8133）最接近伸直 → 最大可达；c_lo 最蜷 → 最小可达
    r_max = math.sqrt(l1 * l1 + l2 * l2 + 2 * l1 * l2 * math.cos(c_hi))
    r_min = math.sqrt(max(0.0, l1 * l1 + l2 * l2 + 2 * l1 * l2 * math.cos(c_lo)))
    acts = {model.actuator(a).name: model.actuator_ctrlrange[a] for a in range(model.nu)}
    return Robot(
        name=name,
        hip_x=float(hip.pos[0]),
        hip_y=float(hip.pos[1]),
        thigh=l1,
        calf=l2,
        foot_r=foot_r,
        foot_y=float(hip.pos[1] + thigh_body.pos[1]),
        r_max=r_max,
        r_min=r_min,
        mass=float(model.body_subtreemass[base]),
        calf_range=(float(c_lo), float(c_hi)),
        thigh_range=tuple(float(v) for v in model.jnt_range[thigh_j.id]),
        torque=(float(acts["FL_thigh"][1]), float(acts["FL_calf"][1])),
    )


def leg_fk(robot: Robot, q1: float, q2: float) -> tuple[float, float]:
    """(thigh, calf) → 足心相对髋的前后/竖向偏移。约定 u(θ)=(-sinθ,-cosθ)。"""
    l1, l2 = robot.thigh, robot.calf
    dx = -l1 * math.sin(q1) - l2 * math.sin(q1 + q2)
    dz = -l1 * math.cos(q1) - l2 * math.cos(q1 + q2)
    return dx, dz


def leg_ik(robot: Robot, dx: float, dz: float) -> tuple[float, float] | None:
    """足心相对髋的 (前后, 竖向) → (thigh, calf)；不可达/超限位返回 None。"""
    l1, l2 = robot.thigh, robot.calf
    r = math.hypot(dx, dz)
    if r > robot.r_max - 1e-4 or r < robot.r_min + 1e-4:
        return None
    cos_q2 = (r * r - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    q2 = -math.acos(max(-1.0, min(1.0, cos_q2)))
    delta = math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    q1 = math.atan2(-dx, -dz) - delta
    lo, hi = robot.thigh_range
    if not (lo + 1e-3 < q1 < hi - 1e-3):
        return None
    lo, hi = robot.calf_range
    if not (lo + 1e-3 < q2 < hi - 1e-3):
        return None
    return q1, q2


class Stance:
    """把机身与四足摆到指定位姿，并用 MuJoCo 自己的 FK 复核 IK。"""

    def __init__(self, model, data, robot: Robot):
        self.model, self.data, self.robot = model, data, robot
        self.joints = np.zeros(12)
        self.base = np.zeros(3)

    def build(self, base_xz, feet, pitch_deg: float = 0.0) -> bool:
        x, z = base_xz
        a = math.radians(-pitch_deg)
        c, s = math.cos(a), math.sin(a)
        qpos = np.zeros(self.model.nq)
        qpos[0], qpos[1], qpos[2] = x, BODY_Y, z
        qpos[3], qpos[5] = math.cos(a / 2.0), math.sin(a / 2.0)
        ok = True
        for i, leg in enumerate(LEGS):
            fx, _fy, fz = feet[leg]
            dxw, dzw = fx - x, fz - z
            lx = dxw * c - dzw * s
            lz = dxw * s + dzw * c
            hip_x = self.robot.hip_x * (1 if leg in ("FL", "FR") else -1)
            sol = leg_ik(self.robot, lx - hip_x, lz)
            if sol is None:
                ok = False
                sol = (0.8, -1.5)
            self.joints[3 * i:3 * i + 3] = (0.0, sol[0], sol[1])
            for jn, val in (("hip", 0.0), ("thigh", sol[0]), ("calf", sol[1])):
                j = self.model.joint(f"{leg}_{jn}_joint")
                qpos[self.model.jnt_qposadr[j.id]] = val
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.base = np.array([x, z, pitch_deg])
        return ok

    def _foot_geom(self, leg: str) -> int:
        calf = self.model.body(f"{leg}_calf").id
        for g in range(self.model.ngeom):
            if (self.model.geom_bodyid[g] == calf
                    and self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE):
                return g
        raise RuntimeError(leg)

    def check_ik(self, feet) -> float:
        return max(float(np.linalg.norm(self.data.geom_xpos[self._foot_geom(leg)]
                                        - np.array(feet[leg][:3]))) for leg in LEGS)

    def clearance(self) -> tuple[float, float]:
        """(机身/机头, 大腿小腿) 到踏板的最小几何距离；足端球不算。"""
        model, data = self.model, self.data
        base = model.body("base_link").id
        feet = {self._foot_geom(leg) for leg in LEGS}
        body_geoms, leg_geoms = [], []
        for g in range(model.ngeom):
            if model.geom_contype[g] <= 0 or g in feet:
                continue
            if model.body_rootid[model.geom_bodyid[g]] != base:
                continue
            if model.geom_bodyid[g] == base:
                body_geoms.append(g)
            else:
                leg_geoms.append(g)
        stairs = [model.geom(n).id for n in STAIR_GEOMS]
        fromto = np.zeros(6)

        def best(geoms):
            value = 1.0
            for g in geoms:
                for s in stairs:
                    value = min(value, mujoco.mj_geomDistance(model, data, g, s, 0.6, fromto))
            return value

        return best(body_geoms), best(leg_geoms)

    def ctrl_target(self) -> np.ndarray:
        target = np.empty(self.model.nu)
        for a in range(self.model.nu):
            name = self.model.actuator(a).name
            leg, part = name.split("_")[0], name.split("_")[1]
            target[a] = self.joints[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]
        return target

    def hold(self, seconds: float, kp: float, kd: float) -> dict:
        model, data = self.model, self.data
        target = self.ctrl_target()
        control_hz = 50.0
        substeps = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
        feet = [self._foot_geom(leg) for leg in LEGS]
        foot0 = np.array([data.geom_xpos[g].copy() for g in feet])
        foot_mid = foot0.copy()
        zs, pitches, rolls = [], [], []
        t_start = data.time
        mid_set = False
        for _ in range(int(seconds * control_hz)):
            for a in range(model.nu):
                j = int(model.actuator_trnid[a, 0])
                dof = int(model.jnt_dofadr[j])
                q = data.qpos[model.jnt_qposadr[j]]
                tau = kp * (target[a] - q) - kd * data.qvel[dof] + 1.0 * data.qfrc_bias[dof]
                lo, hi = model.actuator_ctrlrange[a]
                data.ctrl[a] = min(hi, max(lo, tau))
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            if data.time - t_start >= seconds - 0.5 and not mid_set:
                foot_mid = np.array([data.geom_xpos[g].copy() for g in feet])
                mid_set = True
            zs.append(float(data.qpos[2]))
            q = data.qpos[3:7]
            pitches.append(math.degrees(math.asin(max(-1.0, min(1.0,
                                                          2.0 * (q[0] * q[2] - q[3] * q[1]))))))
            rolls.append(math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                                 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))))
        foot1 = np.array([data.geom_xpos[g].copy() for g in feet])
        tail = np.array(zs[len(zs) // 2:])
        return {
            "z": float(np.mean(tail)),
            "z_ptp_mm": float((tail.max() - tail.min()) * 1000.0),
            "pitch_deg": float(np.mean(pitches[len(pitches) // 2:])),
            "roll_deg": float(np.mean(rolls[len(rolls) // 2:])),
            "slip_mm": float(np.max(np.linalg.norm(foot1[:, :2] - foot_mid[:, :2], axis=1)) * 1000.0),
            "fell": bool(data.qpos[2] < 0.12),
            "tilt_max_deg": float(max(abs(p) for p in pitches + rolls)),
            "final": {"x": float(data.qpos[0]), "z": float(data.qpos[2])},
        }


def point_in_poly(pt, poly) -> tuple[bool, float]:
    """返回 (是否在内, 到最近边的有符号距离；内为正)。

    先把多边形统一成逆时针（按有符号面积判断），否则顺/逆时针会让内外判断整体翻转。
    """
    pts = list(poly)
    area2 = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area2 += x1 * y2 - x2 * y1
    if area2 < 0:
        pts.reverse()
    best = 1e9
    for i in range(len(pts)):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % len(pts)]
        ex, ey = bx - ax, by - ay
        norm = math.hypot(ex, ey)
        if norm < 1e-12:
            continue
        dist = (ex * (pt[1] - ay) - ey * (pt[0] - ax)) / norm  # 左正右负
        best = min(best, dist)
    return best > 0.0, best


def support_metrics(stance: Stance, feet) -> dict:
    model, data, robot = stance.model, stance.data, stance.robot
    com = data.subtree_com[model.body("base_link").id].copy()
    pts = {leg: (feet[leg][0], feet[leg][1]) for leg in LEGS}
    out = {"com": (float(com[0]), float(com[1]))}
    for leg in LEGS:
        others = [pts[k] for k in ("FL", "RL", "RR", "FR") if k != leg]
        inside, margin = point_in_poly((com[0], com[1]), others)
        out[leg] = {"inside": bool(inside), "margin_mm": float(margin * 1000.0)}
    shift = {}
    for sign, tag in ((1.0, "shift_fwd_m"), (-1.0, "shift_back_m")):
        reach = 0.0
        for step in range(1, 60):
            cand = step * 0.005
            ok = True
            for leg in LEGS:
                hip_x = robot.hip_x * (1 if leg in ("FL", "FR") else -1)
                dx = feet[leg][0] - (stance.base[0] + sign * cand) - hip_x
                dz = feet[leg][2] - stance.base[1]
                if leg_ik(robot, dx, dz) is None:
                    ok = False
                    break
            if not ok:
                break
            reach = cand
        shift[tag] = reach
    out.update(shift)
    return out


def search_pose(stance: Stance, feet, pitch_values, x_values, z_values) -> dict | None:
    """在 (pitch, 机身 x, 髋高) 上找净空最好、关节余量够的站姿。"""
    best = None
    for pitch in pitch_values:
        for bx in x_values:
            for h in z_values:
                if not stance.build((float(bx), float(h)), feet, float(pitch)):
                    continue
                if stance.check_ik(feet) > 1e-3:
                    continue
                d_body, d_leg = stance.clearance()
                if d_body < 0.02 or d_leg < 0.01:
                    continue
                margin = min(
                    min(stance.joints[3 * i + 2] - stance.robot.calf_range[0],
                        stance.robot.calf_range[1] - stance.joints[3 * i + 2],
                        stance.joints[3 * i + 1] - stance.robot.thigh_range[0],
                        stance.robot.thigh_range[1] - stance.joints[3 * i + 1])
                    for i in range(4)
                )
                if margin < 0.08:
                    continue
                score = (min(d_body, d_leg * 2.0), margin, -h)
                if best is None or score > best["score"]:
                    best = {"score": score, "pitch": float(pitch), "x": float(bx),
                            "z": float(h), "d_body": d_body, "d_leg": d_leg,
                            "margin": margin}
    return best


def render(stance: Stance, out_dir: Path) -> list[str]:
    from PIL import Image

    renderer = mujoco.Renderer(stance.model, 720, 1280)
    cam = mujoco.MjvCamera()
    files = []
    for name, azim, elev in (("side", 28.0, -16.0), ("iso", 335.0, -20.0)):
        cam.lookat[:] = (0.32, BODY_Y, 0.34)
        cam.distance = 2.0
        cam.azimuth = azim
        cam.elevation = elev
        renderer.update_scene(stance.data, camera=cam)
        path = out_dir / f"as2_stance_{name}.png"
        Image.fromarray(renderer.render()).save(path)
        files.append(str(path))
    renderer.close()
    return files


def main() -> int:
    model, data = load_scene()
    robot = read_robot(model)
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 96)
    say("AS2 站姿探针：0.195 m 高 / 0.18 m 深钢梯，三个关键姿态 + 重心窗口")
    say("=" * 96)
    say(f"官方 MJCF：质量 {robot.mass:.3f} kg，髋距 {2*robot.hip_x:.4f} m，"
        f"大腿 {robot.thigh:.4f} / 小腿 {robot.calf:.4f} m")
    say(f"足端球 r={robot.foot_r:.3f} m，小腿限位 {robot.calf_range[0]:.4f}..{robot.calf_range[1]:.4f} rad "
        f"→ 可达 {robot.r_min:.4f}..{robot.r_max:.4f} m（几何极限 0.4254）")
    say(f"执行器上限：thigh {robot.torque[0]:.0f} N·m / calf {robot.torque[1]:.0f} N·m；"
        f"踏面可用深度 {TREAD_DEPTH-2*robot.foot_r:.4f} m")

    results: dict[str, dict] = {}
    stance = Stance(model, data, robot)

    # ---------- S1：平地 PD 标定 ----------
    flat_feet = {
        leg: (robot.hip_x * (1 if leg in ("FL", "FR") else -1),
              BODY_Y + (1 if leg in ("FL", "RL") else -1) * robot.foot_y,
              robot.foot_r)
        for leg in LEGS
    }
    say()
    say("[S1] 平地 PD 标定（每只脚在自己的髋正下方，机身 0.33 m）")
    gains = {}
    for kp, kd in ((150.0, 5.0), (250.0, 8.0), (400.0, 12.0)):
        stance.build((0.0, 0.33), flat_feet)
        res = stance.hold(2.0, kp, kd)
        gains[(kp, kd)] = res
        say(f"    kp {kp:3.0f} / kd {kd:4.1f}：z {res['z']:.4f} m，峰峰 {res['z_ptp_mm']:.2f} mm，"
            f"倾角 {res['tilt_max_deg']:.1f}°，{'跌落' if res['fell'] else '站住'}")
    best_kp, best_kd = min(
        (k for k in gains if not gains[k]["fell"]),
        key=lambda k: gains[k]["z_ptp_mm"],
        default=(250.0, 8.0),
    )
    results["flat"] = gains[(best_kp, best_kd)]
    say(f"    → 采用 kp {best_kp:.0f} / kd {best_kd:.1f}（下面所有站姿都用这一组）")

    def hold_at(tag: str, feet, pitch: float, x: float, z: float, seconds=2.5):
        ok = stance.build((x, z), feet, pitch)
        err = stance.check_ik(feet)
        res = stance.hold(seconds, best_kp, best_kd)
        res.update({"ik_ok": ok, "ik_err_mm": err * 1000.0, "cmd": (x, z, pitch)})
        results[tag] = res
        return res

    # ---------- S2：四足同踩一级 ----------
    half = TREAD_DEPTH / 2.0 - robot.foot_r
    tread1 = {
        "FL": (STEP1_CENTER + half, BODY_Y + robot.foot_y, STEP1_TOP + robot.foot_r),
        "FR": (STEP1_CENTER + half, BODY_Y - robot.foot_y, STEP1_TOP + robot.foot_r),
        "RL": (STEP1_CENTER - half, BODY_Y + robot.foot_y, STEP1_TOP + robot.foot_r),
        "RR": (STEP1_CENTER - half, BODY_Y - robot.foot_y, STEP1_TOP + robot.foot_r),
    }
    say()
    say(f"[S2] 四足同踩一级：足心前后各让 1 个足半径，支撑深 {2*half:.4f} m")
    pick2 = search_pose(
        stance, tread1,
        pitch_values=np.arange(0.0, 21.0, 5.0),
        x_values=np.arange(STEP1_CENTER - 0.06, STEP1_CENTER + 0.061, 0.01),
        z_values=np.arange(STEP1_TOP + 0.32, STEP1_TOP + 0.401, 0.01),
    )
    if pick2 is None:
        say("    搜索无解：0.18 m 踏面 + 机头/护板净空 + 关节余量三条约束不能同时满足")
    else:
        say(f"    搜索：pitch {pick2['pitch']:.0f}°，机身 x {pick2['x']:.3f}，"
            f"髋高 {pick2['z']-STEP1_TOP:.3f} m（机身净空 {pick2['d_body']*100:.1f} cm，"
            f"腿净空 {pick2['d_leg']*100:.1f} cm，关节余量 {pick2['margin']:.2f} rad）")
        # 同一 (pitch, 髋高) 下扫机身 x，找"鼻头不碰上一级 + 重心在支撑内"的可站窗口
        window = []
        for bx in np.arange(STEP1_CENTER - 0.12, STEP1_CENTER + 0.121, 0.005):
            if not stance.build((float(bx), pick2["z"]), tread1, pick2["pitch"]):
                continue
            d_body, d_leg = stance.clearance()
            if d_body < 0.01 or d_leg < 0.01:
                continue
            com = stance.data.subtree_com[stance.model.body("base_link").id]
            half_span = half - 0.01  # 重心离踏板边缘至少 1 cm
            if not (STEP1_CENTER - half_span < com[0] < STEP1_CENTER + half_span):
                continue
            window.append((float(bx), float(d_body), float(com[0])))
        if window:
            xs = [w[0] for w in window]
            say(f"    可站机身窗口：x {min(xs):.3f}..{max(xs):.3f}（宽 {(max(xs)-min(xs))*100:.1f} cm，"
                f"鼻头让开上一级 + 重心在 0.124 m 支撑内）；最宽处净空 "
                f"{max(w[1] for w in window)*100:.1f} cm")
            results["window_x"] = [min(xs), max(xs)]
            pick2["x"] = float(np.mean(xs))
        res2 = hold_at("all_four_tread1", tread1, pick2["pitch"], pick2["x"], pick2["z"])
        say(f"    IK 残差 {res2['ik_err_mm']:.3f} mm；2.5 s PD：z {res2['z']:.4f}，"
            f"峰峰 {res2['z_ptp_mm']:.2f} mm，俯仰 {res2['pitch_deg']:+.2f}°，"
            f"横滚 {res2['roll_deg']:+.2f}°，足端滑移 {res2['slip_mm']:.2f} mm，"
            f"{'跌落' if res2['fell'] else '站住'}")
        stance.build((pick2["x"], pick2["z"]), tread1, pick2["pitch"])
        res2["clearance"] = stance.clearance()
        say(f"    落地后净空：机身 {res2['clearance'][0]*100:.1f} cm，腿 {res2['clearance'][1]*100:.1f} cm")
        sup2 = support_metrics(stance, tread1)
        res2["support"] = sup2
        say(f"    重心投影 x={sup2['com'][0]:.3f}；抬起单足后：")
        for leg in LEGS:
            s = sup2[leg]
            say(f"      {leg}：{'在内' if s['inside'] else '出界'}，到边界 {s['margin_mm']:+.1f} mm")
        say(f"    机身可前后挪动：前 {sup2['shift_fwd_m']*1000:.0f} mm / 后 {sup2['shift_back_m']*1000:.0f} mm")

    # ---------- S3：跨级 ----------
    straddle = dict(tread1)
    straddle["FL"] = (STEP2_CENTER, BODY_Y + robot.foot_y, STEP2_TOP + robot.foot_r)
    straddle["FR"] = (STEP2_CENTER, BODY_Y - robot.foot_y, STEP2_TOP + robot.foot_r)
    mid = (STEP1_CENTER + STEP2_CENTER) / 2.0
    say()
    say("[S3] 跨级：前足踩二级、后足留一级（机身水平 → 再允许抬头）")
    pick3 = search_pose(
        stance, straddle,
        pitch_values=np.arange(0.0, 16.0, 4.0),
        x_values=np.arange(mid - 0.03, mid + 0.031, 0.01),
        z_values=np.arange(STEP1_TOP + 0.30, STEP1_TOP + 0.381, 0.01),
    )
    if pick3 is None:
        say("    搜索无解")
    else:
        say(f"    搜索：pitch {pick3['pitch']:.0f}°，机身 x {pick3['x']:.3f}，"
            f"髋高 {pick3['z']-STEP1_TOP:.3f} m（机身净空 {pick3['d_body']*100:.1f} cm，"
            f"腿净空 {pick3['d_leg']*100:.1f} cm，关节余量 {pick3['margin']:.2f} rad）")
        res3 = hold_at("straddle", straddle, pick3["pitch"], pick3["x"], pick3["z"])
        say(f"    IK 残差 {res3['ik_err_mm']:.3f} mm；2.5 s PD：z {res3['z']:.4f}，"
            f"峰峰 {res3['z_ptp_mm']:.2f} mm，俯仰 {res3['pitch_deg']:+.2f}°，"
            f"足端滑移 {res3['slip_mm']:.2f} mm，{'跌落' if res3['fell'] else '站住'}")
        stance.build((pick3["x"], pick3["z"]), straddle, pick3["pitch"])
        res3["clearance"] = stance.clearance()
        sup3 = support_metrics(stance, straddle)
        res3["support"] = sup3
        say(f"    落地后净空：机身 {res3['clearance'][0]*100:.1f} cm，腿 {res3['clearance'][1]*100:.1f} cm")
        for i, leg in enumerate(LEGS):
            q1, q2 = stance.joints[3 * i + 1], stance.joints[3 * i + 2]
            dx, dz = leg_fk(robot, q1, q2)
            say(f"    {leg}: thigh {q1:+.3f} / calf {q2:+.3f}，伸展 "
                f"{math.hypot(dx, dz)/robot.r_max*100:.0f}%")
        say(f"    机身可前后挪动：前 {sup3['shift_fwd_m']*1000:.0f} mm / 后 {sup3['shift_back_m']*1000:.0f} mm")

    # ---------- S4：抬前足到上一级 ----------
    say()
    say("[S4] 从 S2 抬起 FL 到二级（三足支撑）—— IK 可达性")
    lift = dict(tread1)
    lift["FL"] = (STEP2_CENTER, BODY_Y + robot.foot_y, STEP2_TOP + robot.foot_r)
    if pick2 is not None:
        ok4 = stance.build((pick2["x"], pick2["z"]), lift, pick2["pitch"])
        err4 = stance.check_ik(lift)
        q1, q2 = stance.joints[4], stance.joints[5]
        dx, dz = leg_fk(robot, q1, q2)
        results["lift_front"] = {"ok": ok4, "err_mm": err4 * 1000.0, "thigh": q1, "calf": q2}
        say(f"    IK {'成功' if ok4 else '失败'}，残差 {err4*1000:.3f} mm；FL thigh {q1:+.3f} / "
            f"calf {q2:+.3f}，伸展 {math.hypot(dx, dz)/robot.r_max*100:.0f}%")

    # ---------- 渲染 ----------
    if "--render" in sys.argv and pick3 is not None:
        stance.build((pick3["x"], pick3["z"]), straddle, pick3["pitch"])
        files = render(stance, ROOT / "experiments" / "as2")
        say()
        say("已渲染：" + ", ".join(Path(f).name for f in files))

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
