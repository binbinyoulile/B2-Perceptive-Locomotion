"""AS2 在 0.195 m / 0.18 m 钢梯上的爬梯动作序列：跨级姿态起手，把后腿收上来，再换级。

文件目的：
    回答三个问题（全部用 MuJoCo 实测，不用静力学猜）：
      1) 从"前足二级、后足一级"的跨级站姿出发，**后腿能不能收到二级**（这就是收腿那一步）；
      2) 收上来以后"四足同踩一级"能不能撑住，撑住多久；
      3) 用纯规则（脚本化 FSM，无学习）能不能爬完一级。
    每一步都记录：机身位姿、四足落点、倾角、足端在不在目标踏面上。

输入：
    simulations/mujoco/as2_stairs_static/scene.xml
输出：
    experiments/as2/as2_stair_gait.txt / .json（逐段结果 + 接触诊断）
依赖：
    mujoco 3.16、numpy、同目录的 as2_stance_probe（复用 IK / 重心工具）
运行：
    D:\\Python313\\python.exe .\\experiments\\as2\\as2_stair_gait.py [--render]

关键假设：
    - 腿是矢状面两连杆、roll=0；IK 残差用 MuJoCo 正运动学复核（失败会打印）。
    - 机身轨迹由动作序列给出（这是"脚本化 FSM"），不是 MPC/RL。
    - 50 Hz PD，kp/kd = 400/12（探针里标定过的值）。

已知限制：
    - 这是"人写的动作序列"，不是优化出来的步态；失败只能说明这条序列不行，
      不能直接推断"AS2 一定爬不上去"。
    - 没有腾空相（四拍都是三足支撑），所以和真实四足的 trot 爬梯不是一回事。
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

import as2_stance_probe as probe  # noqa: E402

OUT = ROOT / "experiments" / "as2" / "as2_stair_gait.txt"
LEGS = ("FL", "FR", "RL", "RR")
BODY_Y = probe.BODY_Y
KP, KD = 400.0, 12.0

T1 = probe.STEP1_CENTER      # 0.30657
T2 = probe.STEP2_CENTER      # 0.49183
T3 = 0.677094
Z1, Z2, Z3 = probe.STEP1_TOP, probe.STEP2_TOP, probe.STEP3_TOP  # 0.195 / 0.3803 / 0.5655


def foot_y(leg: str, robot) -> float:
    return BODY_Y + (1 if leg in ("FL", "RL") else -1) * robot.foot_y


@dataclass
class Segment:
    """一段动作：duration 秒内把机身与指定足端插值到目标。"""

    name: str
    duration: float
    body: tuple[float, float, float]          # (x, z, pitch_deg)
    feet: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    lift_m: float = 0.08


class Walker:
    def __init__(self, model, data, robot, kp=KP, kd=KD):
        self.model, self.data, self.robot = model, data, robot
        self.kp, self.kd = kp, kd
        self.body = np.zeros(3)
        self.feet: dict[str, np.ndarray] = {}
        self.joints = np.zeros(12)
        self.ctrl_hz = 50.0
        self.substeps = max(1, int(round(1.0 / (self.ctrl_hz * model.opt.timestep))))

    # ---------- 摆放与 IK ----------
    def place(self, body, feet) -> bool:
        x, z, pitch = body
        a = math.radians(-pitch)
        c, s = math.cos(a), math.sin(a)
        qpos = np.zeros(self.model.nq)
        qpos[0], qpos[1], qpos[2] = x, BODY_Y, z
        qpos[3], qpos[5] = math.cos(a / 2.0), math.sin(a / 2.0)
        ok = True
        for i, leg in enumerate(LEGS):
            fx, _fy, fz = feet[leg]
            dxw, dzw = fx - x, fz - z
            lx, lz = dxw * c - dzw * s, dxw * s + dzw * c
            hip_x = self.robot.hip_x * (1 if leg in ("FL", "FR") else -1)
            sol = probe.leg_ik(self.robot, lx - hip_x, lz)
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
        self.body = np.array(body, dtype=float)
        self.feet = {k: np.array(v, dtype=float) for k, v in feet.items()}
        return ok

    def targets(self, body, feet) -> np.ndarray:
        x, z, pitch = body
        a = math.radians(-pitch)
        c, s = math.cos(a), math.sin(a)
        joints = np.zeros(12)
        for i, leg in enumerate(LEGS):
            fx, _fy, fz = feet[leg]
            dxw, dzw = fx - x, fz - z
            lx, lz = dxw * c - dzw * s, dxw * s + dzw * c
            hip_x = self.robot.hip_x * (1 if leg in ("FL", "FR") else -1)
            sol = probe.leg_ik(self.robot, lx - hip_x, lz)
            if sol is None:
                sol = (0.8, -1.5)  # 不可达：退化到中位，让 PD 拉住（会体现在结果里）
            joints[3 * i:3 * i + 3] = (0.0, sol[0], sol[1])
        return joints

    def ctrl_map(self, joints) -> np.ndarray:
        target = np.empty(self.model.nu)
        for a in range(self.model.nu):
            name = self.model.actuator(a).name
            leg, part = name.split("_")[0], name.split("_")[1]
            target[a] = joints[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]
        return target

    def step(self, joints) -> None:
        target = self.ctrl_map(joints)
        for _ in range(self.substeps):
            for a in range(self.model.nu):
                j = int(self.model.actuator_trnid[a, 0])
                dof = int(self.model.jnt_dofadr[j])
                q = self.data.qpos[self.model.jnt_qposadr[j]]
                tau = (self.kp * (target[a] - q) - self.kd * self.data.qvel[dof]
                       + self.data.qfrc_bias[dof])
                lo, hi = self.model.actuator_ctrlrange[a]
                self.data.ctrl[a] = min(hi, max(lo, tau))
            mujoco.mj_step(self.model, self.data)

    # ---------- 执行 ----------
    def run(self, segments: list[Segment], sync=None, realtime: bool = False) -> dict:
        import time

        model, data = self.model, self.data
        x0, z0 = float(data.qpos[0]), float(data.qpos[2])
        trace, per_seg = [], []
        max_tilt = 0.0
        for seg in segments:
            body_src = self.body.copy()
            body_dst = np.array(seg.body, dtype=float)
            feet_src = {k: v.copy() for k, v in self.feet.items()}
            feet_dst = {k: np.array(seg.feet.get(k, tuple(feet_src[k])), dtype=float)
                        for k in LEGS}
            steps = max(1, int(round(seg.duration * self.ctrl_hz)))
            for n in range(steps):
                u = (n + 1) / steps
                smooth = u * u * (3.0 - 2.0 * u)
                body = body_src + (body_dst - body_src) * smooth
                feet = {}
                for leg in LEGS:
                    src, dst = feet_src[leg], feet_dst[leg]
                    pos = src + (dst - src) * smooth
                    if np.linalg.norm(dst[:2] - src[:2]) > 1e-4 or abs(dst[2] - src[2]) > 1e-4:
                        pos[2] += seg.lift_m * math.sin(math.pi * u)
                    feet[leg] = pos
                self.step(self.targets(tuple(body), feet))
                if sync is not None:
                    sync()
                if realtime:
                    time.sleep(max(0.0, 1.0 / self.ctrl_hz - 1.0 / self.ctrl_hz))
                q = data.qpos[3:7]
                pitch = math.degrees(math.asin(max(-1.0, min(1.0,
                                                          2.0 * (q[0] * q[2] - q[3] * q[1])))))
                roll = math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                               1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])))
                max_tilt = max(max_tilt, abs(pitch), abs(roll))
                trace.append((float(data.time), float(data.qpos[0]), float(data.qpos[2])))
            self.body = body_dst
            self.feet = {k: v.copy() for k, v in feet_dst.items()}
            per_seg.append(self.snapshot(seg.name, feet_dst))
            if per_seg[-1]["fallen"]:
                break
        return {
            "advance_m": float(data.qpos[0] - x0),
            "rise_m": float(data.qpos[2] - z0),
            "max_tilt_deg": max_tilt,
            "fallen": bool(data.qpos[2] < 0.15 or max_tilt > 60.0),
            "segments": per_seg,
            "trace": trace,
        }

    def snapshot(self, name: str, feet_dst) -> dict:
        model, data = self.model, self.data
        q = data.qpos[3:7]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (q[0] * q[2] - q[3] * q[1])))))
        roll = math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                       1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])))
        feet_now = {}
        for leg in LEGS:
            g = None
            calf = model.body(f"{leg}_calf").id
            for gi in range(model.ngeom):
                if (model.geom_bodyid[gi] == calf
                        and model.geom_type[gi] == mujoco.mjtGeom.mjGEOM_SPHERE):
                    g = gi
                    break
            feet_now[leg] = [float(v) for v in data.geom_xpos[g]]
        err = max(float(np.linalg.norm(np.array(feet_now[leg]) - feet_dst[leg])) for leg in LEGS)
        return {
            "segment": name,
            "x": float(data.qpos[0]), "z": float(data.qpos[2]),
            "pitch_deg": pitch, "roll_deg": roll,
            "feet_err_mm": err * 1000.0,
            "feet_z": {k: round(v[2], 3) for k, v in feet_now.items()},
            "fallen": bool(data.qpos[2] < 0.15 or abs(roll) > 60.0 or abs(pitch) > 60.0),
        }


def straddle_feet(robot):
    return {
        "FL": (T2, foot_y("FL", robot), Z2 + robot.foot_r),
        "FR": (T2, foot_y("FR", robot), Z2 + robot.foot_r),
        "RL": (T1 - 0.062, foot_y("RL", robot), Z1 + robot.foot_r),
        "RR": (T1 - 0.062, foot_y("RR", robot), Z1 + robot.foot_r),
    }


def body_for(feet, pitch=10.0, stand=0.375):
    """朴素的 FSM 机身位姿：x 取四足形心，z 以**最低**那只脚所在的踏面为基准加站高。

    机身高度由低处的那对脚决定（跨级时后腿在下），所以用 min 而不是 mean。
    """
    xs = [feet[k][0] for k in LEGS]
    tops = [feet[k][2] - 0.028 for k in LEGS]
    return (float(np.mean(xs)), float(min(tops)) + stand, float(pitch))


def plan(robot) -> tuple[dict, list[Segment]]:
    """四拍收腿 + 换级：先挪重心、再抬一只脚（同侧两脚同时动必翻）。

    顺序：RR 上二级 → RL 上二级 → FL 前移 → FR 前移（此时四足同踩二级）
          → FL 上三级 → FR 上三级 → RL 上三级 → RR 上三级。
    """
    feet0 = {k: np.array(v, dtype=float) for k, v in straddle_feet(robot).items()}
    segs: list[Segment] = []
    feet = {k: v.copy() for k, v in feet0.items()}
    shift = 0.045  # 抬脚前先把机身（重心）往前进方向挪一点，盖住支撑三角形的斜边

    def body(extra_shift=0.0, pitch=10.0):
        x, z, p = body_for(feet, pitch)
        return (x + extra_shift, z, p)

    def add(name, leg, target, duration=1.4, lift=0.09, pitch=10.0):
        # 1) 先挪重心（三足仍全落地）
        segs.append(Segment(f"  挪重心（{name}）", 0.9, body(shift, pitch)))
        # 2) 再抬这只脚
        feet[leg] = np.array(target, dtype=float)
        segs.append(Segment(name, duration, body(shift, pitch), {leg: tuple(feet[leg])},
                            lift_m=lift))
        # 3) 落稳
        segs.append(Segment("  落稳", 0.5, body(0.0, pitch)))

    def add_hold(name, duration=0.8, pitch=10.0):
        segs.append(Segment(name, duration, body(0.0, pitch)))

    # 后腿收到二级（落点在前足后方 0.124 m 处）
    add("1 右后腿上二级", "RR", (T2 - 0.062, feet["RR"][1], Z2 + robot.foot_r), 1.5, 0.10)
    add("2 左后腿上二级（四足同踩）", "RL", (T2 - 0.062, feet["RL"][1], Z2 + robot.foot_r), 1.6, 0.16)
    add_hold("  四足同踩二级保持", 1.2)
    # 前足前移到二级前缘
    add("3 左前足前移", "FL", (T2 + 0.062, feet["FL"][1], Z2 + robot.foot_r), 1.2, 0.07)
    add("4 右前足前移", "FR", (T2 + 0.062, feet["FR"][1], Z2 + robot.foot_r), 1.2, 0.07)
    add_hold("  四足同踩二级站定", 0.8)
    # 前足上三级
    add("5 左前足上三级", "FL", (T3 - 0.062, feet["FL"][1], Z3 + robot.foot_r), 1.5, 0.11, 12.0)
    add("6 右前足上三级", "FR", (T3 - 0.062, feet["FR"][1], Z3 + robot.foot_r), 1.5, 0.11, 12.0)
    # 后腿跟上三级
    add("7 左后腿上三级", "RL", (T3 - 0.062, feet["RL"][1], Z3 + robot.foot_r), 1.5, 0.11, 10.0)
    add("8 右后腿上三级", "RR", (T3 - 0.062, feet["RR"][1], Z3 + robot.foot_r), 1.5, 0.11, 10.0)
    add_hold("  三级站定", 1.0, 8.0)
    return feet0, segs


def main() -> int:
    model, data = probe.load_scene()
    robot = probe.read_robot(model)
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    feet0, segs = plan(robot)
    body0 = body_for(feet0, 12.0)
    w = Walker(model, data, robot)
    # 预压 3 mm 起手：足心正好落在踏面时 MuJoCo 不产生接触（dist=0 不算相交），
    # 先压进去 3 mm 才能从第一帧就有四足支撑。
    feet_pre = {k: np.array(v, dtype=float) for k, v in feet0.items()}
    for k in feet_pre:
        feet_pre[k][2] -= 0.003
    ok = w.place(body0, feet_pre)
    say("=" * 96)
    say("AS2 爬梯动作序列（跨级起手 + 四拍收腿/换级，纯脚本 FSM）")
    say("=" * 96)
    say(f"起手：跨级姿态，pitch 12°，机身 ({body0[0]:.3f}, {BODY_Y:.3f}, {body0[1]:.3f})，"
        f"IK {'成功' if ok else '失败'}；kp/kd {KP:.0f}/{KD:.0f}")
    say(f"踏面：一级 {T1:.3f} 顶 {Z1:.3f}；二级 {T2:.3f} 顶 {Z2:.3f}；三级 {T3:.3f} 顶 {Z3:.3f}")

    settle = w.run([Segment("settle", 0.8, body0, {k: tuple(v) for k, v in feet0.items()})])
    say(f"落地 0.8 s：机身 z {settle['segments'][-1]['z']:.4f}，"
        f"倾角 {settle['max_tilt_deg']:.1f}°，{'跌落' if settle['fallen'] else '稳定'}，"
        f"接触数 {w.data.ncon}")

    # 抬哪只脚时，重心离"另三足三角形"边界还有多远（静态余量）
    com = w.data.subtree_com[model.body("base_link").id]
    say()
    say(f"抬腿前的重心投影 ({com[0]:.4f}, {com[1]:.4f})，抬起单足后的静态余量：")
    margins = {}
    for leg in LEGS:
        poly = [(w.feet[k][0], w.feet[k][1]) for k in ("FL", "RL", "RR", "FR") if k != leg]
        inside, margin = probe.point_in_poly((com[0], com[1]), poly)
        margins[leg] = (inside, margin)
        say(f"  抬 {leg}：{'在内' if inside else '已出界'}，到最近边界 "
            f"{margin*100:+.1f} cm")

    segs_run = []
    fallen = False
    res_x0, res_z0 = float(w.data.qpos[0]), float(w.data.qpos[2])
    for seg in segs:
        if seg.name[0].isdigit() and seg.feet:
            leg = next(iter(seg.feet))
            com = w.data.subtree_com[model.body("base_link").id]
            poly = [(w.feet[k][0], w.feet[k][1]) for k in ("FL", "RL", "RR", "FR") if k != leg]
            inside, margin = probe.point_in_poly((com[0], com[1]), poly)
            say(f"    抬 {leg} 前：重心到三足边界 {margin*100:+.1f} cm（"
                f"{'在内' if inside else '已出界'}）")
        r = w.run([seg])
        segs_run.extend(r["segments"])
        snap = r["segments"][-1]
        say(f"  {seg.name:<26} x {snap['x']:+.3f}  z {snap['z']:.3f}  "
            f"pitch {snap['pitch_deg']:+6.1f}°  roll {snap['roll_deg']:+6.1f}°  "
            f"足端误差 {snap['feet_err_mm']:6.1f} mm  {'跌落' if snap['fallen'] else ''}")
        if snap["fallen"]:
            fallen = True
            break
    res = {
        "segments": segs_run,
        "fallen": fallen,
        "advance_m": float(w.data.qpos[0] - res_x0),
        "rise_m": float(w.data.qpos[2] - res_z0),
        "max_tilt_deg": max(max(abs(s["pitch_deg"]), abs(s["roll_deg"])) for s in segs_run),
    }

    say()
    say(f"整条序列：前进 {res['advance_m']:+.3f} m，上升 {res['rise_m']:+.3f} m，"
        f"最大倾角 {res['max_tilt_deg']:.1f}°，{'跌落' if res['fallen'] else '未跌落'}")

    # 末态四足高度 → 判断"后腿有没有收上来"
    last = res["segments"][-1]
    say()
    say("末态四足足心高度（一级顶 0.195 / 二级顶 0.380 / 三级顶 0.566）：")
    for leg in LEGS:
        z = last["feet_z"][leg]
        level = "地面" if z < 0.1 else ("一级" if abs(z - Z1 - robot.foot_r) < 0.03
                                       else ("二级" if abs(z - Z2 - robot.foot_r) < 0.03
                                             else ("三级" if abs(z - Z3 - robot.foot_r) < 0.03 else "悬空/其他")))
        say(f"  {leg}: {z:.3f} → {level}")

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

