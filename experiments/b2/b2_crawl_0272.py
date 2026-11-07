"""B2 在实测钢梯（0.197 级高 / 0.272 踏面）上的错位对角姿态 + 四拍爬升。

文件目的：
    踏面从 0.18 放宽到 0.272 后，验证用户提的"阶梯错位对角"：左前踩二级、右前与右后
    分踩一级的前后、左后在地面；再按"一次一脚 + 先挪重心"的四拍往上换级。
    纯脚本 FSM，无训练。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（新尺寸钢梯）
输出：
    experiments/b2/b2_crawl_0272.txt / .json
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_crawl_0272.py [--render]

关键假设：
    - 落点：FL 二级中心、FR/RR 一级前/后、RL 地面 0.12；足心 = 踏面顶 + 足半径。
    - 机身只绕 y 俯仰；kp/kd = 800/20；50 Hz。
已知限制：
    - 动作序列是人写的；失败只说明这条序列不行。
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

import b2_step_pose as base  # noqa: E402

OUT = ROOT / "experiments" / "b2" / "b2_crawl_0272.txt"
LEGS = ("FL", "FR", "RL", "RR")
BODY_Y = 0.468
HIP_X = np.array([0.3285, 0.3285, -0.3285, -0.3285])
FOOT_R = 0.032
L = 0.35
KP, KD = 800.0, 20.0
Y = 0.1916

T1, T2, T3 = 0.197, 0.394, 0.591
X1, X2, X3 = 0.3526, 0.6246, 0.8966


def stance_feet():
    """用户提的错位对角：FL 二级；FR/RR 一级前后；RL 地面。"""
    return {
        "FL": (X2, BODY_Y + Y, T2 + FOOT_R),
        "FR": (0.440, BODY_Y - Y, T1 + FOOT_R),
        "RR": (0.260, BODY_Y - Y, T1 + FOOT_R),
        "RL": (0.120, BODY_Y + Y, 0.0 + FOOT_R),
    }


@dataclass
class Seg:
    name: str
    duration: float
    body: tuple[float, float, float]
    feet: dict = field(default_factory=dict)
    lift: float = 0.10


class Robot:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self.joints = np.zeros(12)
        self.body = np.zeros(3)
        self.feet = {}

    def build(self, body, feet, check=True) -> bool:
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
            sol = base.leg_ik(lx - HIP_X[i], lz)
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
            sol = base.leg_ik(lx - HIP_X[i], lz)
            joints[3 * i:3 * i + 3] = (0.0, *(sol if sol else (0.8, -1.5)))
        target = np.empty(self.model.nu)
        for adr in range(self.model.nu):
            name = self.model.actuator(adr).name
            leg, part = name.split("_")[0], name.split("_")[1]
            target[adr] = joints[3 * LEGS.index(leg) + ("hip", "thigh", "calf").index(part)]
        return target

    def hold(self, target, seconds: float, sync=None):
        for _ in range(int(seconds * 50)):
            for a in range(self.model.nu):
                j = int(self.model.actuator_trnid[a, 0])
                dof = int(self.model.jnt_dofadr[j])
                q = self.data.qpos[self.model.jnt_qposadr[j]]
                tau = KP * (target[a] - q) - KD * self.data.qvel[dof] + self.data.qfrc_bias[dof]
                lo, hi = self.model.actuator_ctrlrange[a]
                self.data.ctrl[a] = min(hi, max(lo, tau))
            for _ in range(10):
                mujoco.mj_step(self.model, self.data)
            if sync is not None:
                sync()


def com_and_margin(rb: Robot, feet) -> tuple[float, float]:
    com = rb.data.subtree_com[rb.model.body("base_link").id]
    pts = [(feet[k][0], feet[k][1]) for k in LEGS]
    cx, cy = np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    area = sum(pts[i][0] * pts[(i + 1) % 4][1] - pts[(i + 1) % 4][0] * pts[i][1] for i in range(4))
    if area < 0:
        pts.reverse()
    best = 1e9
    for i in range(4):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % 4]
        n = math.hypot(bx - ax, by - ay)
        best = min(best, ((bx - ax) * (com[1] - ay) - (by - ay) * (com[0] - ax)) / n)
    return float(com[0]), float(best)


PLAN = [
    ("1 RR 上二级", "RR", (0.530, BODY_Y - Y, T2 + FOOT_R), 0.09),
    ("2 RL 上一级", "RL", (0.150, BODY_Y + Y, T1 + FOOT_R), 0.09),
    ("3 FR 上二级", "FR", (0.700, BODY_Y - Y, T2 + FOOT_R), 0.09),
    ("4 FL 上三级", "FL", (X3, BODY_Y + Y, T3 + FOOT_R), 0.10),
]


def main() -> int:
    model, data = base.load_scene()
    rb = Robot(model, data)
    feet0 = stance_feet()
    lines = []

    def say(t=""):
        lines.append(t)

    say("=" * 90)
    say("B2 错位对角（0.197 / 0.272 实测钢梯）：FL 二级 / FR 一级前 / RR 一级后 / RL 地面")
    say("=" * 90)

    best = None
    for pitch in np.arange(0.0, 31.0, 2.0):
        for bx in np.arange(0.28, 0.56, 0.02):
            for bz in np.arange(0.50, 0.81, 0.02):
                if not rb.build((float(bx), float(bz), float(pitch)), feet0):
                    continue
                if min(rb.joints[2], rb.joints[5], rb.joints[8], rb.joints[11]) < -2.70:
                    continue
                com, margin = com_and_margin(rb, feet0)
                if margin < 0.02:
                    continue
                score = margin - 0.002 * abs(pitch)
                if best is None or score > best[0]:
                    best = (score, float(bx), float(bz), float(pitch), margin, com)
    if best is None:
        say("无可行静态姿态")
        OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1
    _, bx, bz, pitch, margin, com = best
    rb.build((bx, bz, pitch), feet0)
    say(f"静态解：pitch {pitch:.0f}°，机身 ({bx:.3f}, {BODY_Y:.3f}, {bz:.3f})，"
        f"重心 ({com:.3f}, {BODY_Y:.3f})，余量 {margin*100:+.1f} cm")

    # 2.5 s PD
    rb.build((bx, bz, pitch), feet0)
    rb.hold(rb.targets((bx, bz, pitch), feet0), 2.5)
    z = float(data.qpos[2])
    q = data.qpos[3:7]
    tilt = math.degrees(math.asin(max(-1, min(1, 2 * (q[0] * q[2] - q[3] * q[1])))))
    say(f"2.5 s PD：机身 z {z:.4f}（指令 {bz:.4f}），俯仰 {tilt:+.1f}°，"
        f"{'站住' if z > 0.35 else '翻倒'}")

    # 四拍：RR→二级后、RL→一级、FR→二级前、FL→三级
    plan = PLAN
    feet = {k: np.array(v, dtype=float) for k, v in feet0.items()}
    body = np.array([bx, bz, pitch])
    results = []
    for name, leg, dst, lift in plan:
        # 先挪重心：向"剩下三足"的形心方向偏 4 cm
        others = [k for k in LEGS if k != leg]
        cx = float(np.mean([feet[k][0] for k in others]))
        shift = body.copy()
        shift[0] += 0.4 * (cx - body[0])
        rb.hold(rb.targets(tuple(shift), feet), 1.0)
        body = shift
        # 抬脚
        src = feet[leg].copy()
        dst_v = np.array(dst, dtype=float)
        for n in range(75):
            u = (n + 1) / 75
            sm = u * u * (3 - 2 * u)
            pos = src + (dst_v - src) * sm
            pos[2] += lift * math.sin(math.pi * u)
            f = {k: v for k, v in feet.items()}
            f[leg] = pos
            rb.hold(rb.targets(tuple(body), f), 0.02)
        feet[leg] = dst_v
        rb.hold(rb.targets(tuple(body), feet), 0.6)
        zz = float(data.qpos[2])
        results.append((name, zz, float(data.qpos[0])))
        say(f"  {name}: 末态 z {zz:.3f}，x {data.qpos[0]:+.3f}"
            f"{'  ← 跌落' if zz < 0.35 else ''}")

    say(f"整条：x {data.qpos[0]-bx:+.3f}，z {data.qpos[2]-bz:+.3f}")
    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    OUT.with_suffix(".json").write_text(json.dumps(
        {"best": [bx, bz, pitch, margin], "steps": results}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
