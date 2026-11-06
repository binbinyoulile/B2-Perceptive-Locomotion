"""B2 在 0.195 m / 0.18 m 窄踏面钢梯上的步态实验：脚本化动作序列 + MuJoCo 实测。

核心问题（先算清楚再动手）：
    这是 45° 楼梯：每级上升 0.1853 m、前进 0.1853 m，踏面只有 0.18 m 深。
    B2 髋距 0.657 m、单腿最大伸展 0.70 m，于是：
      1) 前足不能比后足高两个台阶（差 0.371 m）：后腿要伸 0.37+0.55 ≈ 0.92 m > 0.70；
      2) 所以"前足高一阶、后足低一阶"是唯一能维持的跨级姿态；
      3) 而要让后足追上来，后足必须先落到前足所在的那一级 → 必然出现"四足同踩一级"；
      4) 四足同踩时支撑深度只有 0.18−2×0.032 = 0.116 m，重心余量 3.7 cm。

    另一个可能绕开 (3) 的办法是"交错站立"：左前+右后踩上一级，右前+左后留在下一级，
    每级各两只脚。本脚本就把这两种思路都跑一遍。

测的是什么：
    A. 四足同踩一级能否站住（静态 + PD 保持）；
    B. 从"四足同踩一级"把前足成对抬到上一级（动态前扑）能否接住；
    C. 交错对角步态（2 只脚/级）走 2 个循环能否前进。

判据（每个实验都打印）：
    机身高度、机身 x 前进量、最大倾角、是否跌落、机身/护板与钢梯的接触次数。

依赖：mujoco 3.16、numpy 2.x、configs.robot_config。
运行：D:\\Python313\\python.exe .\\experiments\\b2\\b2_stair_gait.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_ROBOT  # noqa: E402

import mujoco  # noqa: E402

SCENE = ROOT / "simulations" / "mujoco" / "b2_stairs_static" / "scene.xml"
OUT = ROOT / "experiments" / "b2" / "b2_stair_gait.txt"

LEGS = ("FL", "FR", "RL", "RR")
HIP_X = np.array(B2_ROBOT.hip_x_m, dtype=np.float64)
FOOT_Y = np.array(B2_ROBOT.nominal_foot_y_m, dtype=np.float64)
L = B2_ROBOT.thigh_length_m
FOOT_R = B2_ROBOT.foot_radius_m
BODY_Y = 0.468

# 钢梯几何（compiled 实测）：每级 x 前进与 z 上升都是 0.1853
STEP_PITCH = 0.1853
TREAD1_CENTER = 0.306570
TREAD1_TOP = 0.195
TREAD2_CENTER = 0.491832
TREAD2_TOP = 0.3803


def load_scene():
    previous = Path.cwd()
    os.chdir(SCENE.parent)
    try:
        model = mujoco.MjModel.from_xml_path(SCENE.name)
    finally:
        os.chdir(previous)
    return model, mujoco.MjData(model)


def leg_ik(dx: float, dz: float):
    """机身坐标系下两连杆逆解；dz 向下为负。"""
    reach = math.hypot(dx, dz)
    if reach > 2.0 * L - 0.015:
        return None
    psi = math.acos(max(-1.0, min(1.0, reach * reach / (2.0 * L * L) - 1.0)))
    thigh = psi / 2.0 - math.atan2(dx, -dz)
    calf = -psi
    if not (-2.80 <= calf <= -0.45):
        return None
    return thigh, calf


@dataclass
class Segment:
    """一段动作：duration 秒内把机身与足端从当前值插值到目标值。"""

    duration: float
    body: tuple[float, float, float] | None = None      # (x, z, pitch_deg)
    feet: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    lift_m: float = 0.05


class Walker:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.joints = np.zeros(12)
        self.feet = {leg: np.zeros(3) for leg in LEGS}
        self.body = np.zeros(3)

    def place(self, body, feet) -> bool:
        """把机身和四足直接摆到目标位姿（无插值），返回 IK 是否全部成功。"""
        x, z, pitch = body
        a = math.radians(-pitch)
        c, s = math.cos(a), math.sin(a)
        qpos = np.zeros(self.model.nq)
        qpos[0], qpos[1], qpos[2] = x, BODY_Y, z
        qpos[3], qpos[5] = math.cos(a / 2.0), math.sin(a / 2.0)
        ok = True
        for i, (leg, hip_x) in enumerate(zip(LEGS, HIP_X)):
            fx, fy, fz = feet[leg]
            dxw, dzw = fx - x, fz - z
            lx = dxw * c - dzw * s
            lz = dxw * s + dzw * c
            sol = leg_ik(lx - hip_x, lz)
            if sol is None:
                ok = False
                sol = (0.8, -1.6)
            self.joints[3 * i:3 * i + 3] = (0.0, sol[0], sol[1])
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_hip_joint")
            qpos[self.model.jnt_qposadr[j]] = 0.0
            for k, jn in enumerate(("thigh", "calf")):
                jj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{jn}_joint")
                qpos[self.model.jnt_qposadr[jj]] = self.joints[3 * i + 1 + k]
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        self.body = np.array(body, dtype=float)
        self.feet = {k: np.array(v, dtype=float) for k, v in feet.items()}
        mujoco.mj_forward(self.model, self.data)
        return ok

    def ctrl_from_joints(self) -> np.ndarray:
        target = np.empty(self.model.nu)
        for a in range(self.model.nu):
            j = int(self.model.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            leg, part = name.split("_")[0], name.split("_")[1]
            target[a] = self.joints[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]
        return target

    def joint_targets(self, body, feet) -> np.ndarray:
        """按计划位姿做 IK，返回 12 维目标角（不写 qpos，交给 PD 跟踪）。"""
        x, z, pitch = body
        a = math.radians(-pitch)
        c, s = math.cos(a), math.sin(a)
        joints = np.zeros(12)
        for i, (leg, hip_x) in enumerate(zip(LEGS, HIP_X)):
            fx, fy, fz = feet[leg]
            dxw, dzw = fx - x, fz - z
            lx = dxw * c - dzw * s
            lz = dxw * s + dzw * c
            sol = leg_ik(lx - hip_x, lz)
            if sol is None:
                sol = (0.8, -1.6)  # 目标不可达时退化到中位，交给 PD 拉住
            joints[3 * i:3 * i + 3] = (0.0, sol[0], sol[1])
        return joints

    def run(self, segments: list[Segment], trace: bool = False,
            sync=None, realtime: bool = False) -> dict:
        """按计划推进：每控制步算 IK 目标角，再用 PD 跟踪（不直接改 qpos）。"""
        model, data = self.model, self.data
        control_hz = B2_ROBOT.fsm.control_hz
        substeps = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
        x0, z0 = float(data.qpos[0]), float(data.qpos[2])
        max_tilt = 0.0
        trace_pts = []
        for seg in segments:
            body_src = self.body.copy()
            body_dst = np.array(seg.body if seg.body else tuple(self.body), dtype=float)
            feet_src = {k: v.copy() for k, v in self.feet.items()}
            feet_dst = {k: np.array(seg.feet.get(k, tuple(feet_src[k])), dtype=float)
                        for k in LEGS}
            steps = max(1, int(round(seg.duration * control_hz)))
            for n in range(steps):
                u = (n + 1) / steps
                smooth = u * u * (3.0 - 2.0 * u)
                body = body_src + (body_dst - body_src) * smooth
                feet = {}
                for leg in LEGS:
                    src, dst = feet_src[leg], feet_dst[leg]
                    pos = src + (dst - src) * smooth
                    if np.linalg.norm(dst[:2] - src[:2]) > 1e-4:
                        pos[2] += seg.lift_m * math.sin(math.pi * u)
                    feet[leg] = pos
                joints = self.joint_targets(tuple(body), feet)
                target = self._ctrl_map(joints)
                t_step = time.perf_counter()
                for _ in range(substeps):
                    for a in range(model.nu):
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
                    mujoco.mj_step(model, data)
                if sync is not None:
                    sync()
                if realtime:
                    rest = 1.0 / control_hz - (time.perf_counter() - t_step)
                    if rest > 0:
                        time.sleep(rest)
                q = data.qpos[3:7]
                pitch = math.degrees(math.asin(max(-1.0, min(1.0,
                                                         2.0 * (q[0] * q[2] - q[3] * q[1])))))
                roll = math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                               1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])))
                max_tilt = max(max_tilt, abs(pitch), abs(roll))
                if trace:
                    trace_pts.append((data.time, float(data.qpos[0]), float(data.qpos[2])))
            self.body = np.array(tuple(body_dst), dtype=float)
            self.feet = {k: np.array(v, dtype=float) for k, v in feet_dst.items()}
        return {
            "x": float(data.qpos[0]),
            "z": float(data.qpos[2]),
            "advance_m": float(data.qpos[0] - x0),
            "rise_m": float(data.qpos[2] - z0),
            "max_tilt_deg": max_tilt,
            "fallen": bool(data.qpos[2] < 0.28 or max_tilt > 55.0),
            "trace": trace_pts,
        }

    def _ctrl_map(self, joints: np.ndarray) -> np.ndarray:
        target = np.empty(self.model.nu)
        for a in range(self.model.nu):
            j = int(self.model.actuator_trnid[a, 0])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            leg, part = name.split("_")[0], name.split("_")[1]
            target[a] = joints[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]
        return target


def feet_on_tread(center_x: float, top_z: float, depth: float = 0.058):
    """四足压在一级踏面上：前足靠前、后足靠后，各离踏面中心 depth。"""
    z = top_z + FOOT_R
    return {
        "FL": (center_x + depth, BODY_Y + 0.1917, z),
        "FR": (center_x + depth, BODY_Y - 0.1917, z),
        "RL": (center_x - depth, BODY_Y + 0.1917, z),
        "RR": (center_x - depth, BODY_Y - 0.1917, z),
    }


def main() -> int:
    model, data = load_scene()
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)

    say("=" * 90)
    say("B2 窄踏面钢梯步态实验（0.195 m 高 / 0.18 m 深，45° 楼梯）")
    say("=" * 90)
    say(f"B2：髋距 {2*abs(HIP_X[0]):.3f} m，单腿最大伸展 {2*L:.3f} m，"
        f"踏面深 0.18 m → 支撑深度仅 {0.18-2*FOOT_R:.3f} m")
    say("几何死结：前足最多比后足高一阶（差 0.371 m 时后腿要伸 0.92 m > 0.70 m），")
    say("          所以后足必须落到前足那一级 → 必然出现四足同踩一级。")
    say()

    results: dict[str, dict] = {}

    def foot_heights() -> dict[str, float]:
        ids = {leg: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
               for leg in LEGS}
        return {leg: float(data.xpos[b][2]) for leg, b in ids.items()}

    # ---------- A：四足同踩一级（0.18 m 踏面站四只脚）----------
    w = Walker(model, data)
    body_a = (TREAD1_CENTER, 0.50, 0.0)
    ok = w.place(body_a, feet_on_tread(TREAD1_CENTER, TREAD1_TOP))
    res = w.run([Segment(2.5, body=body_a)])
    results["A 四足同踩一级"] = res
    say(f"[A] 四足同踩一级（前足 +5.8 cm / 后足 −5.8 cm，IK {'成功' if ok else '部分失败'}）")
    say(f"    2.5 s：机身 x {res['x']:+.3f}，z {res['z']:.4f}（相对 0.50 下沉 {0.50-res['z']:+.3f}），"
        f"最大倾角 {res['max_tilt_deg']:.1f}°，{'跌落' if res['fallen'] else '站住'}")
    say(f"    结论：{'这个瓶颈姿态本身是稳的' if not res['fallen'] else '连站都站不住'}")

    # ---------- B：从同踩姿态把左前足单独抬到上一级（3 足支撑）----------
    w = Walker(model, data)
    w.place(body_a, feet_on_tread(TREAD1_CENTER, TREAD1_TOP))
    feet_b = {k: np.array(v, dtype=float) for k, v in feet_on_tread(TREAD1_CENTER, TREAD1_TOP).items()}
    feet_b["FL"] = np.array((TREAD2_CENTER + 0.02, BODY_Y + 0.1917, TREAD2_TOP + FOOT_R))
    body_b = (TREAD1_CENTER + 0.02, 0.53, 6.0)
    res = w.run([
        Segment(1.6, body=body_a, feet={"FL": tuple(feet_b["FL"])}, lift_m=0.085),
        Segment(1.2, body=body_b),
    ])
    fh = foot_heights()
    results["B 单足上一级"] = res
    say(f"[B] 左前足单独抬到二级（3 足支撑）：x {res['advance_m']:+.3f}，"
        f"z {res['rise_m']:+.3f}，最大倾角 {res['max_tilt_deg']:.1f}°，"
        f"{'跌落' if res['fallen'] else '未跌落'}")
    say(f"    末态小腿高度 FL {fh['FL']:.3f} / FR {fh['FR']:.3f} / "
        f"RL {fh['RL']:.3f} / RR {fh['RR']:.3f}（踏面1顶 0.195、踏面2顶 0.380）")

    # ---------- C：四拍爬一级（每拍先挪重心再抬脚）----------
    w = Walker(model, data)
    w.place(body_a, feet_on_tread(TREAD1_CENTER, TREAD1_TOP))
    feet_c = {k: np.array(v, dtype=float) for k, v in feet_on_tread(TREAD1_CENTER, TREAD1_TOP).items()}
    body_c = np.array(body_a, dtype=float)
    segs = []
    order = ("FL", "RR", "FR", "RL")
    for leg in order:
        # 1) 把重心挪到"剩下三只脚"的形心方向（这是静态爬行最关键的一步）
        others = [k for k in LEGS if k != leg]
        cx = float(np.mean([feet_c[k][0] for k in others]))
        body_shift = body_c.copy()
        body_shift[0] += 0.7 * (cx - body_c[0])
        segs.append(Segment(0.8, body=tuple(body_shift)))
        # 2) 抬脚 → 上一级
        feet_c[leg][0] += STEP_PITCH
        feet_c[leg][2] += STEP_PITCH
        body_shift[0] += STEP_PITCH / 4.0
        body_shift[1] += 0.02
        body_shift[2] += 2.0
        segs.append(Segment(1.2, body=tuple(body_shift),
                            feet={leg: tuple(feet_c[leg])}, lift_m=0.09))
        body_c = body_shift
        segs.append(Segment(0.3, body=tuple(body_c)))
    res = w.run(segs, trace=True)
    results["C 四拍爬一级（带重心前移）"] = res
    say(f"[C] 四拍爬一级（每拍先挪重心，再抬脚）：x {res['advance_m']:+.3f}，"
        f"z {res['rise_m']:+.3f}，最大倾角 {res['max_tilt_deg']:.1f}°，"
        f"{'跌落' if res['fallen'] else '未跌落'}")
    fh = foot_heights()
    say(f"    末态小腿高度 FL {fh['FL']:.3f} / FR {fh['FR']:.3f} / "
        f"RL {fh['RL']:.3f} / RR {fh['RR']:.3f}")

    say()
    say("-" * 90)
    say("逐段轨迹（C）：")
    for t, x, z in results["C 四拍爬一级（带重心前移）"]["trace"][::40]:
        say(f"    t={t:5.2f}s  x={x:+.3f}  z={z:.3f}")

    # ---------- D：重心 / 支撑多边形分析（解释为什么抬脚就走）----------
    def point_in_poly(pt, poly):
        """多边形内外 + 到最近边的有符号距离（外为正）。"""
        inside = True
        best = -1e9
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            cross = (bx - ax) * (pt[1] - ay) - (by - ay) * (pt[0] - ax)
            if cross < 0:
                inside = False
            best = max(best, cross)
        return inside, best

    stance = feet_on_tread(TREAD1_CENTER, TREAD1_TOP)
    com = np.array((TREAD1_CENTER + B2_ROBOT.com_offset_x_m, BODY_Y))
    poly_all = [(stance[k][0], stance[k][1]) for k in ("FL", "RL", "RR", "FR")]
    ins_all, _ = point_in_poly(com, poly_all)
    say()
    say("[D] 四足同踩一级时的重心分析（COM 投影 vs 支撑多边形）")
    say(f"    COM 投影 ({com[0]:.3f}, {com[1]:.3f})，四足全落时 {'在' if ins_all else '不在'}多边形内")
    for leg in LEGS:
        poly = [(stance[k][0], stance[k][1]) for k in ("FL", "RL", "RR", "FR") if k != leg]
        ins, margin = point_in_poly(com, poly)
        say(f"    抬起 {leg} 后：COM {'仍在' if ins else '跑到'}多边形{'内' if ins else '外'}"
            f"（到最近边 {margin*1000:+.1f} mm）")
    say("    → 每抬一只脚，重心必须先在 0.116 m 的支撑面上挪 4 cm 左右，"
        "剩给控制误差的余量不到 1.5 cm。")

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
