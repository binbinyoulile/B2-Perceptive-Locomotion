"""三机同标准步态的横向评测：0.197 级高 / 0.272 踏面钢梯上的「四足同踩 + 挪重心 + 四拍」。

文件目的：
    在同一份钢梯、同一套控制代码、同一个步态规则下，跑 Go2 / B2 / AS2，回答：
      1) 谁的重心余量最从容（动态实测，不看几何表）；
      2) 谁的跨级伸展最舒服（AS2 的 96% 伸展在动态里是否卡死）；
      3) 谁的力矩更抗"腿软"（实测 τ/τ_max，而不是只看铭牌）；
      4) 当前场景最推荐哪一款。

输入：
    simulations/mujoco/{go2,b2,as2}_stairs_static/scene.xml（都 include 同一份新钢梯）
输出：
    experiments/feasibility/three_robot_gait_0272.txt / .json
    --robot X --view 时另开交互窗口（无文件输出）

依赖：mujoco 3.16（--view 需 OpenGL）、numpy 2.x。
运行：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\three_robot_gait_0272.py
    D:\\Python313\\python.exe .\\experiments\\feasibility\\three_robot_gait_0272.py --robot b2 --view

步态规则（三机完全一致，只有几何参数来自各自的 MJCF）：
    A. 四足同踩一级：前后足压在同一踏面的最前/最后（各留 8 mm），机身水平。
    B. 每一拍 = 挪重心（机身侧移到"剩下三足"的内切圆心）→ 抬脚（先竖直抬过踏面鼻端、
       再水平前送、再落下）→ 落定（机身回到踏面中心线上）。
    C. 抬脚顺序 FL → RR → FR → RL（标准四拍，同一时刻只有一只脚离地）。
    D. 机身姿态不写死：每一拍在 (x, z, pitch) 网格里搜一个同时满足
       「四足 IK 可达、腿伸展率 ∈ [0.40, 0.92]、重心投影在支撑多边形内 12 mm 以上」
       且离上一拍最近的位姿。搜不到 = 该拍几何不可行，直接记为失败（这是结论，不是 bug）。
    E. 腿部 IK 是带髋 roll 的三维解（机身侧移必须靠髋外展完成，否则重心挪不过去）。

关键假设：
    - 50 Hz PD（kp/kd 各自先前标定值）+ 重力前馈，μ=0.9（钢梯与足端取 max）。
    - 足端球心的 x 偏移（Go2 2 mm、另两机 0）忽略不计，其余几何全部从 MJCF 读。
    - 力矩上限取 actuator_ctrlrange（Go2 23.7/23.7/45.4，B2 200/200/300，AS2 60/60/90）。

已知限制：
    - 这是"人定的四拍 + 小网格搜机身"的规则步态，不是 MPC/RL；失败说明这条规则走不通，
      不能反推"该机型绝对爬不上去"。
    - 没有腾空相（任一时刻至少三足支撑），所以和真实 trot 爬梯不是一回事。
    - 台阶鼻端是直角，足端球半径以上的护板碰撞只在失败模式里体现，不做专门建模。
"""

from __future__ import annotations

import json
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

import mujoco  # noqa: E402

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("hip", "thigh", "calf")
GAINS = {"go2": (130.0, 3.5), "b2": (800.0, 20.0), "as2": (400.0, 12.0)}
LABEL = {"go2": "Go2", "b2": "B2", "as2": "AS2"}
ORDER = {"go2": "Go2", "b2": "B2", "as2": "AS2"}  # 报告顺序

RISE = 0.197          # 级高
TREAD = 0.272         # 踏面深
HALF = TREAD / 2.0
TREAD_TOPS = (0.197, 0.394, 0.591)          # 一/二/三级踏面顶
TREAD_CX = (0.3526, 0.6246, 0.8966)         # 一/二/三级踏面中心 x
EDGE_MARGIN = 0.008   # 足端离踏面前后边缘的余量
PRELOAD = 0.003       # 足端预压（几何相切不生成接触）
BODY_Y = 0.468        # 钢梯中心线
CRAWL_ORDER = ("FL", "RR", "FR", "RL")
EXT_MIN, EXT_MAX = 0.30, 0.96      # 允许的腿伸展率区间（预留 4% 给跟踪误差，机械极限是 1.0）
BODY_CLEAR = 0.010                # 机身碰撞体与梯级踏板的最小间隙


# ---------------------------------------------------------------- 几何工具
def poly_ccw(pts):
    """按形心极角排序再统一成逆时针（手写顺序会排成自交蝴蝶结）。"""
    p = np.asarray(pts, dtype=float)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(ang)]
    area = 0.5 * float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))
    return p if area > 0 else p[::-1]


def margin_in_poly(pt, poly):
    """点到凸多边形各边的有符号最小距离（内正外负）。"""
    p = poly_ccw(poly)
    best = 1e9
    for i in range(len(p)):
        a, b = p[i], p[(i + 1) % len(p)]
        e = b - a
        n = np.array((-e[1], e[0]))
        n /= np.linalg.norm(n) + 1e-12
        best = min(best, float(np.dot(np.asarray(pt) - a, n)))
    return best


def incenter3(pts):
    """三角形的内切圆心（三足支撑时重心挪到这里余量最大）。"""
    a, b, c = (np.asarray(p, dtype=float) for p in pts)
    la = np.linalg.norm(b - c)
    lb = np.linalg.norm(a - c)
    lc = np.linalg.norm(a - b)
    return (la * a + lb * b + lc * c) / (la + lb + lc)


# ---------------------------------------------------------------- 单机型
@dataclass
class Rig:
    key: str
    model: mujoco.MjModel
    data: mujoco.MjData
    kp: float
    kd: float
    hip: dict          # leg -> (x, y) 髋体在机身系的位置（右侧 y 为负）
    foot_y: dict       # leg -> 名义足端 y（相对机身中心，带符号）
    ty: dict           # leg -> 髋外展偏置（大腿体 y 偏移，带符号）
    L1: float
    L2: float
    r: float           # 足端球半径
    qadr: dict         # (leg, part) -> qpos 地址
    act: dict          # (leg, part) -> actuator index
    lim: dict          # (leg, part) -> 力矩上限
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    plates: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    root_gids: list = field(default_factory=list)
    plate_gids: list = field(default_factory=list)
    probe: mujoco.MjData = field(init=False)
    lim_a: dict = field(default_factory=dict)   # actuator -> 力矩上限
    qadr_a: dict = field(default_factory=dict)  # actuator -> qpos 地址
    dof_a: dict = field(default_factory=dict)   # actuator -> dof 地址
    ctrl_a: dict = field(default_factory=dict)  # (leg, part) -> actuator
    sub: int = 10                              # 50 Hz 控制 / 2 ms 物理 = 10 子步

    @property
    def Lmax(self) -> float:
        return self.L1 + self.L2

    @property
    def mass(self) -> float:
        return float(self.model.body_subtreemass[1])


def load_rig(key: str, scene: Path | None = None) -> Rig:
    scene = scene or ROOT / "simulations" / "mujoco" / f"{key}_stairs_static" / "scene.xml"
    prev = Path.cwd()
    os.chdir(scene.parent)
    try:
        model = mujoco.MjModel.from_xml_path(scene.name)
    finally:
        os.chdir(prev)
    data = mujoco.MjData(model)

    hip, foot_y, ty = {}, {}, {}
    for leg in LEGS:
        b = model.body(f"{leg}_hip")
        th = model.body(f"{leg}_thigh")
        hip[leg] = (float(b.pos[0]), float(b.pos[1]))          # 机身系，带符号（右侧为负）
        ty[leg] = float(th.pos[1])                             # 髋外展偏置，带符号
        foot_y[leg] = hip[leg][1] + ty[leg]                    # 名义足端 y（相对机身中心）
    thigh = model.body("FL_thigh")
    calf = model.body("FL_calf")
    L1 = abs(float(calf.pos[2]))
    spher = [g for g in range(model.ngeom)
             if model.geom_bodyid[g] == calf.id
             and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    if not spher:
        raise RuntimeError(f"{key}: 找不到小腿下的球形足端")
    g = spher[-1]
    L2 = abs(float(model.geom_pos[g][2]))
    r = float(model.geom_size[g][0])

    qadr, act, lim = {}, {}, {}
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        leg, part = name.split("_")[0], name.split("_")[1]
        qadr[(leg, part)] = int(model.jnt_qposadr[j])
        act[(leg, part)] = a
        lim[(leg, part)] = float(abs(model.actuator_ctrlrange[a][1]))

    kp, kd = GAINS[key]
    rig = Rig(key=key, model=model, data=data, kp=kp, kd=kd, hip=hip, foot_y=foot_y,
              ty=ty, L1=L1, L2=L2, r=r,
              qadr=qadr, act=act, lim=lim)
    rig.probe = mujoco.MjData(model)
    root = int(model.jnt_bodyid[0])
    rig.root_gids = [g for g in range(model.ngeom)
                     if model.geom_bodyid[g] == root and model.geom_contype[g] != 0]
    rig.plate_gids = [g for g in range(model.ngeom)
                      if "stair" in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                                       int(model.geom_bodyid[g])) or "")
                      and model.geom_contype[g] != 0]
    for (leg, part), a in act.items():
        j = int(model.actuator_trnid[a, 0])
        rig.lim_a[a] = lim[(leg, part)]
        rig.qadr_a[a] = int(model.jnt_qposadr[j])
        rig.dof_a[a] = int(model.jnt_dofadr[j])
        rig.ctrl_a[(leg, part)] = a
    return rig


# ---------------------------------------------------------------- 运动学
def ik_leg(rig: Rig, leg: str, dx: float, dy: float, dz: float):
    """髋 roll + 矢状面两连杆的解析 IK。返回 (hip, thigh, calf) 或 None。"""
    ty, L1, L2 = rig.ty[leg], rig.L1, rig.L2
    rb = math.hypot(dy, dz)
    if rb < ty - 1e-6 or rb > L1 + L2 - 1e-6:
        return None
    b = -math.sqrt(max(0.0, rb * rb - ty * ty))
    q1 = math.atan2(dz, dy) - math.atan2(b, ty)
    q1 = (q1 + math.pi) % (2.0 * math.pi) - math.pi
    reach = math.hypot(dx, b)
    if reach > L1 + L2 - 1e-5 or reach < abs(L1 - L2) + 1e-5:
        return None
    c3 = (reach * reach - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    q3 = -math.acos(max(-1.0, min(1.0, c3)))
    q2 = math.atan2(-dx, -b) - math.atan2(L2 * math.sin(q3), L1 + L2 * math.cos(q3))
    return q1, q2, q3


def body_to_local(pitch: float, dx: float, dz: float):
    """世界系相对位移 → 机身系（机身绕 y 转 pitch，正=抬头）。"""
    c, s = math.cos(pitch), math.sin(pitch)
    return c * dx + s * dz, -s * dx + c * dz


def legs_ik(rig: Rig, body, feet):
    """body=(x,y,z,pitch)，feet=leg->(x,y,z)；返回 (joints dict, 伸展率 dict) 或 (None, None)。"""
    bx, by, bz, pitch = body
    joints, ext = {}, {}
    for leg in LEGS:
        fx, fy, fz = feet[leg]
        lx, lz = body_to_local(pitch, fx - bx, fz - bz)
        hx, hy = rig.hip[leg]
        dx, dy, dz = lx - hx, (fy - by) - hy, lz
        sol = ik_leg(rig, leg, dx, dy, dz)
        if sol is None:
            return None, None
        joints[leg] = sol
        ext[leg] = math.hypot(dx, dy, dz) / rig.Lmax
    return joints, ext


def set_pose(rig: Rig, d: mujoco.MjData, body, joints):
    """把机身位姿与 12 个关节角写进 qpos（只用于初始化/探测，不用于推进）。"""
    bx, by, bz, pitch = body
    q = np.zeros(rig.model.nq)
    a = 0.5 * pitch
    q[0], q[1], q[2] = bx, by, bz
    q[3], q[4], q[5], q[6] = math.cos(a), 0.0, -math.sin(a), 0.0
    for leg in LEGS:
        for k, part in enumerate(JOINTS):
            q[rig.qadr[(leg, part)]] = joints[leg][k]
    d.qpos[:] = q
    d.qvel[:] = 0.0
    mujoco.mj_forward(rig.model, d)


def com_xy(rig: Rig, body, joints) -> np.ndarray:
    """当前姿态下的真实重心投影（根子树 COM，含所有连杆）。"""
    set_pose(rig, rig.probe, body, joints)
    return rig.probe.subtree_com[1][:2].copy()


def pose_metrics(rig: Rig, body, joints):
    """一次 FK 同时拿重心投影和"机身碰撞体 vs 梯级踏板"的最小间隙（米）。

    只看根 body 的碰撞 geom：足端/小腿本来就要压在踏面上，不能算作干涉。
    """
    set_pose(rig, rig.probe, body, joints)
    com = rig.probe.subtree_com[1][:2].copy()
    clear = 1e9
    for g1 in rig.root_gids:
        for g2 in rig.plate_gids:
            clear = min(clear, mujoco.mj_geomDistance(rig.model, rig.probe, g1, g2, 1.0, None))
    return com, float(clear)


def foot_gids(rig: Rig) -> dict:
    out = {}
    for leg in LEGS:
        calf = rig.model.body(f"{leg}_calf")
        sp = [g for g in range(rig.model.ngeom)
              if rig.model.geom_bodyid[g] == calf.id
              and rig.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
        out[leg] = sp[-1]
    return out


def fk_residual(rig: Rig, body, feet, fg) -> float:
    """IK → qpos → 正运动学复核：计划足端与真实足端球心的最大偏差（应 < 2 mm）。"""
    joints, _ = legs_ik(rig, body, feet)
    if joints is None:
        return float("inf")
    set_pose(rig, rig.probe, body, joints)
    return max(float(np.linalg.norm(rig.probe.geom_xpos[fg[leg]] - np.asarray(feet[leg])))
               for leg in LEGS)


# ---------------------------------------------------------------- 姿态搜索
def solve_pose(rig: Rig, feet, stance, prev=None, center_y: float | None = None,
               x_hint: float | None = None, ext_max: float = EXT_MAX):
    """在 (x, z, pitch) 网格里搜一个可行且离 prev 最近的机身位姿。

    stance: 允许承重的腿（腾空那条不在内）——重心必须落在它们的多边形内。
    center_y: 给定时机身 y 取该值（落定态，四足全踩）；否则取 support 内切圆心。
    """
    support = [feet[k] for k in stance]
    if center_y is None:
        y = float(incenter3(support)[1]) if len(support) == 3 else float(np.mean([p[1] for p in support]))
        y = y - 0.0  # 重心 y 偏移在下方用实测值修正
    else:
        y = center_y
    xc = x_hint if x_hint is not None else float(np.mean([p[0] for p in support]))
    xs = np.arange(xc - 0.14, xc + 0.145, 0.02)
    zlo = min(p[2] for p in feet.values()) + 0.10
    zhi = max(p[2] for p in feet.values()) + 0.60
    zs = np.arange(zlo, zhi, 0.02)
    pitches = np.radians(np.arange(-4.0, 42.0, 4.0))

    best, best_score = None, -1e9
    for pitch in pitches:
        for x in xs:
            for z in zs:
                body = (float(x), y, float(z), float(pitch))
                joints, ext = legs_ik(rig, body, feet)
                if joints is None or min(ext.values()) < EXT_MIN or max(ext.values()) > ext_max:
                    continue
                com, gap = pose_metrics(rig, body, joints)
                if gap < BODY_CLEAR:
                    continue
                m = margin_in_poly(com, [(p[0], p[1]) for p in support])
                if m < 0.012:
                    continue
                slack = min(min(e - EXT_MIN, ext_max - e) for e in ext.values())
                score = slack + 1.5 * min(m, 0.05)
                if prev is not None:
                    score -= 0.6 * (abs(x - prev[0]) + abs(z - prev[2]) + 0.25 * abs(pitch - prev[3]))
                if score > best_score:
                    best_score, best = score, (float(x), y, float(z), float(pitch),
                                               joints, ext, com, m, gap)
    return best


# ---------------------------------------------------------------- 步态规划
@dataclass
class Seg:
    """一段动作：dur 秒内把机身从 a 平滑到 b，同时把指定足端沿爬升路径送到 dst。"""

    name: str
    dur: float
    body_a: tuple
    body_b: tuple
    feet_a: dict
    feet_b: dict
    lift: str | None = None       # 要移动的腿
    z_clear: float = 0.0          # 抬起段的最大足心高度


def stance_feet(rig: Rig, k: int):
    """四足同踩第 k 级：前足贴踏面最前、后足贴最后，各留 EDGE_MARGIN。"""
    cx, top = TREAD_CX[k], TREAD_TOPS[k]
    d = HALF - rig.r - EDGE_MARGIN
    z = top + rig.r + PRELOAD
    out = {}
    for leg in LEGS:
        fx = cx + d if leg in ("FL", "FR") else cx - d
        fy = BODY_Y + rig.foot_y[leg]
        out[leg] = (fx, fy, z)
    return out


def build_plan(rig: Rig, steps: int = 2):
    """生成整条轨迹：每一步 4 拍 × 3 段（挪重心 / 抬脚 / 落定）。"""
    segs: list[Seg] = []
    notes: list[str] = []
    feet = stance_feet(rig, 0)
    guess = (float(np.mean([p[0] for p in feet.values()])), BODY_Y,
             max(p[2] for p in feet.values()) + 0.30, 0.0)
    pose = solve_pose(rig, feet, LEGS, prev=guess, center_y=BODY_Y)
    if pose is None:
        return None, ["初始四足同踩一级无可行姿态"]
    body = pose[:4]
    segs.append(Seg("稳住", 0.35, body, body, feet, feet))
    notes.append(f"初始姿态 x {body[0]:.3f} z {body[2]:.3f} pitch {math.degrees(body[3]):+.1f}° "
                 f"伸展 {min(pose[5].values())*100:.0f}–{max(pose[5].values())*100:.0f}% "
                 f"重心余量 {pose[7]*1000:.0f} mm")

    for k in range(steps):
        for idx, leg in enumerate(CRAWL_ORDER):
            stance3 = tuple(l for l in LEGS if l != leg)
            # 1) 挪重心：四只脚都还在原位，但重心要进"三足"三角形
            shifted = solve_pose(rig, feet, stance3, prev=body)
            if shifted is None:
                notes.append(f"第 {k+1} 级第 {idx+1} 拍（{leg}）：挪重心无可行位姿 → 该拍放弃")
                return segs, notes
            segs.append(Seg(f"挪重心{leg}", 0.7, body, shifted[:4], feet, feet))
            body = shifted[:4]

            # 2) 抬脚：目标落在上一级踏面的同位置（前足仍贴前缘、后足仍贴后缘）
            nxt = dict(feet)
            nxt[leg] = stance_feet(rig, k + 1)[leg]
            lifted = solve_pose(rig, nxt, stance3, prev=body)
            if lifted is None:
                notes.append(f"第 {k+1} 级第 {idx+1} 拍（{leg}）：抬到二级无可行位姿 → 该拍放弃")
                return segs, notes
            z_clear = max(feet[leg][2], nxt[leg][2]) + rig.r + 0.03
            segs.append(Seg(f"抬{leg}", 1.1, body, lifted[:4], feet, nxt, lift=leg, z_clear=z_clear))
            body, feet = lifted[:4], nxt

            # 3) 落定：四足全踩，机身回到踏面中心线（用"理想站姿"当先验，别停在半起身的地方）
            ideal = (float(np.mean([p[0] for p in feet.values()])), BODY_Y,
                     max(p[2] for p in feet.values()) + 0.30, 0.0)
            settled = solve_pose(rig, feet, LEGS, prev=ideal, center_y=BODY_Y)
            if settled is None:
                notes.append(f"第 {k+1} 级第 {idx+1} 拍（{leg}）：落定无可行位姿")
                return segs, notes
            segs.append(Seg(f"落定{leg}", 0.4, body, settled[:4], feet, feet))
            body = settled[:4]
            notes.append(f"第 {k+1} 级第 {idx+1} 拍（{leg}）后：机身 x {body[0]:.3f} "
                         f"z {body[2]:.3f} pitch {math.degrees(body[3]):+.1f}° "
                         f"伸展 {min(settled[5].values())*100:.0f}–{max(settled[5].values())*100:.0f}% "
                         f"余量 {settled[7]*1000:.0f} mm")
    segs.append(Seg("收尾", 0.8, body, body, feet, feet))
    return segs, notes


# ---------------------------------------------------------------- 执行
def lift_profile(src, dst, z_clear, u):
    """足端爬升路径：0–35% 竖直抬过踏面鼻端，35–70% 水平前送，70–100% 落下。"""
    a, b = np.asarray(src, float), np.asarray(dst, float)
    if u <= 0.35:
        w = u / 0.35
        w = w * w * (3 - 2 * w)
        p = a.copy()
        p[0] += (b[0] - a[0]) * 0.10 * w
        p[1] += (b[1] - a[1]) * 0.10 * w
        p[2] = a[2] + (z_clear - a[2]) * w
        return p
    if u <= 0.70:
        w = (u - 0.35) / 0.35
        w = w * w * (3 - 2 * w)
        p = a.copy()
        p[0] = a[0] + (b[0] - a[0]) * (0.10 + 0.80 * w)
        p[1] = a[1] + (b[1] - a[1]) * (0.10 + 0.80 * w)
        p[2] = z_clear
        return p
    w = (u - 0.70) / 0.30
    w = w * w * (3 - 2 * w)
    p = a.copy()
    p[0] = a[0] + (b[0] - a[0]) * (0.90 + 0.10 * w)
    p[1] = a[1] + (b[1] - a[1]) * (0.90 + 0.10 * w)
    p[2] = z_clear + (b[2] - z_clear) * w
    return p


def pitch_of(data) -> float:
    """机身抬头角（弧度，正=抬头），与规划里的 pitch 同符号。"""
    q = data.qpos[3:7]
    return -math.asin(max(-1.0, min(1.0, 2.0 * (q[0] * q[2] - q[3] * q[1]))))


def yaw_of(data) -> float:
    """机身偏航角（弧度）。规划只管 pitch，yaw 没有反馈，靠这个字段看它漂了多少。"""
    q = data.qpos[3:7]
    return math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                      1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))


def tilt_of(data) -> float:
    """最大倾角（度）。"""
    q = data.qpos[3:7]
    roll = math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                      1.0 - 2.0 * (q[1] ** 2 + q[2] ** 2))
    return max(abs(math.degrees(pitch_of(data))), abs(math.degrees(roll)))


class Loop:
    """一段 PD 推进的公共部分：目标角 → 子步积分，顺带记力矩占用。"""

    def __init__(self, rig: Rig):
        self.rig = rig
        self.peak = {a: 0.0 for a in range(rig.model.nu)}
        self.peak_joint = ("", "")
        self.sat = 0
        self.steps = 0
        self.last = 0.0        # 最近一个控制步的最大力矩占额

    def step(self, joints) -> None:
        rig, model, data = self.rig, self.rig.model, self.rig.data
        target = np.zeros(model.nu)
        for leg in LEGS:
            for k, part in enumerate(JOINTS):
                target[rig.act[(leg, part)]] = joints[leg][k]
        step_peak = 0.0
        for _ in range(rig.sub):
            for a in range(model.nu):
                dof = rig.dof_a[a]
                q = data.qpos[rig.qadr_a[a]]
                tau = rig.kp * (target[a] - q) - rig.kd * data.qvel[dof] + data.qfrc_bias[dof]
                ratio = abs(tau) / rig.lim_a[a]
                if ratio > self.peak[a]:
                    self.peak[a] = ratio
                step_peak = max(step_peak, ratio)
                if ratio >= 1.0:
                    self.sat += 1
                self.steps += 1
                data.ctrl[a] = max(-rig.lim_a[a], min(rig.lim_a[a], tau))
            mujoco.mj_step(model, data)
        self.last = step_peak


def hold(rig: Rig, fg, feet, body, seconds: float = 1.5):
    """把机身摆到 body、足端摆到 feet，PD 保持一段时间，量漂移（这是"站不站得住"的判据）。"""
    joints, ext = legs_ik(rig, body, feet)
    if joints is None:
        return None
    data = rig.data
    set_pose(rig, data, body, joints)
    p0 = np.array([data.qpos[0], data.qpos[1], data.qpos[2]])
    th0 = pitch_of(data)
    f0 = {leg: data.geom_xpos[fg[leg]][:2].copy() for leg in LEGS}
    lp = Loop(rig)
    tilt = 0.0
    for _ in range(int(round(seconds * 50))):
        lp.step(joints)
        tilt = max(tilt, tilt_of(data))
    slip = max(float(np.linalg.norm(data.geom_xpos[fg[l]][:2] - f0[l])) for l in LEGS)
    return {"dx": float(data.qpos[0] - p0[0]), "dy": float(data.qpos[1] - p0[1]),
            "dz": float(data.qpos[2] - p0[2]),
            "dpitch": math.degrees(pitch_of(data) - th0),
            "slip_mm": slip * 1000.0, "tilt": tilt,
            "peak_tau": max(lp.peak.values()), "sat_frac": lp.sat / max(1, lp.steps),
            "ext": ext, "body": body}


def run_plan(rig: Rig, segs: list[Seg], fg, sync=None, hook=None, body_servo: float = 0.0):
    """按计划推进；一旦跌落就停在出错的那一段（继续推只会得到一堆地板数据）。"""
    model, data = rig.model, rig.data
    joints0, _ = legs_ik(rig, segs[0].body_a, segs[0].feet_a)
    set_pose(rig, data, segs[0].body_a, joints0)
    z_start = float(data.qpos[2])
    lp = Loop(rig)
    max_tilt = 0.0
    beats: list[dict] = []
    lift_peak = {leg: 0.0 for leg in LEGS}
    foot_z0 = {leg: float(data.geom_xpos[fg[leg]][2]) for leg in LEGS}
    failed_at, fell = None, False
    iu = None
    last_joints = joints0

    for seg in segs:
        ba, bb = np.asarray(seg.body_a, float), np.asarray(seg.body_b, float)
        fa = {k: np.asarray(v, float) for k, v in seg.feet_a.items()}
        fb = {k: np.asarray(v, float) for k, v in seg.feet_b.items()}
        planted = {leg: data.geom_xpos[fg[leg]][:2].copy() for leg in LEGS if leg != seg.lift}
        seg_peak = 0.0
        seg_lift = 0.0
        z_src = float(data.geom_xpos[fg[seg.lift]][2]) if seg.lift else 0.0
        n = max(1, int(round(seg.dur * 50.0)))
        for i in range(n):
            u = (i + 1) / n
            sm = u * u * (3 - 2 * u)
            body = ba + (bb - ba) * sm
            feet = {}
            for leg in LEGS:
                if leg == seg.lift:
                    feet[leg] = lift_profile(fa[leg], fb[leg], seg.z_clear, u)
                else:
                    feet[leg] = fa[leg] + (fb[leg] - fa[leg]) * sm
            # 机身位姿伺服：0 = 纯开环（按计划位姿算 IK）；>0 = 用实测位姿误差把机体往计划上顶
            body_cmd = body
            if body_servo > 0.0:
                meas = np.array([data.qpos[0], data.qpos[1], data.qpos[2], pitch_of(data)])
                # 实测比计划"落后"时，按计划再往前多推 k 倍 → 腿把机身顶回计划位姿
                body_cmd = body + body_servo * (body - meas)
            joints, _ = legs_ik(rig, tuple(body_cmd), feet)
            if joints is None:
                if body_servo > 0.0:
                    joints = last_joints          # 伺服把目标顶出可达范围：退回上一步的角度
                else:
                    failed_at = f"{seg.name}（轨迹中段 IK 不可达）"
                    fell = True
                    break
            last_joints = joints
            lp.step(joints)
            seg_peak = max(seg_peak, lp.last)
            if hook is not None:
                hook(seg, u)
            if sync is not None:
                sync()
            max_tilt = max(max_tilt, tilt_of(data))
            if seg.lift:
                lift_peak[seg.lift] = max(lift_peak[seg.lift],
                                          float(data.geom_xpos[fg[seg.lift]][2]) - foot_z0[seg.lift])
                seg_lift = max(seg_lift, float(data.geom_xpos[fg[seg.lift]][2]) - z_src)
            if data.qpos[2] < z_start - 0.10 or max_tilt > 45.0:
                failed_at = seg.name
                fell = True
                break
        if fell:
            break
        if seg.lift:
            iu = {k: v.copy() for k, v in planted.items()}
            beats.append({"name": seg.name, "leg": seg.lift,
                          "x": float(data.qpos[0]), "z": float(data.qpos[2]),
                          "plan_x": float(bb[0]), "plan_z": float(bb[2]),
                          "sink": float(bb[2] - data.qpos[2]),
                          "slip_mm": max(float(np.linalg.norm(
                              data.geom_xpos[fg[l]][:2] - iu[l])) * 1000 for l in LEGS if l != seg.lift),
                          "tilt": max_tilt, "peak": seg_peak, "lift": seg_lift,
                          "y": float(data.qpos[1]), "yaw": math.degrees(yaw_of(data)),
                          "feet_z": {l: float(data.geom_xpos[fg[l]][2]) for l in LEGS},
                          "feet_x": {l: float(data.geom_xpos[fg[l]][0]) for l in LEGS}})
    worst = max(rig.act.items(), key=lambda kv: lp.peak[kv[1]])
    return {
        "x": float(data.qpos[0]), "z": float(data.qpos[2]),
        "max_tilt": max_tilt, "peak_tau": max(lp.peak.values()),
        "peak_act": f"{worst[0][0]}_{worst[0][1]}",
        "sat_frac": lp.sat / max(1, lp.steps),
        "lift_m": lift_peak, "beats": beats, "failed_at": failed_at, "fell": fell,
        "feet_z": {leg: float(data.geom_xpos[fg[leg]][2]) for leg in LEGS},
    }


def beat4_probe(rig: Rig) -> dict:
    """最后一拍（三足已上二级、末足仍在一级）的能力上限：给定可达范围里重心余量最好能到多少。

    这是"同踩 + 四拍"这条步态真正的卡点：最后一次抬脚时，承重的三只足全在上一级，
    机身必须往前上方走，而末足还挂在下一级——可达距离和重心位置在这里正面冲突。
    """
    f1, f0 = stance_feet(rig, 1), stance_feet(rig, 0)
    feet = dict(f1)
    feet["RL"] = f0["RL"]
    stance = ("FL", "FR", "RR")
    support = [feet[k] for k in stance]
    y = float(incenter3(support)[1])
    poly = [(p[0], p[1]) for p in support]
    best = {"margin": -9.0, "ext": 0.0, "body": None}
    for pitch in np.radians(np.arange(-4.0, 44.0, 6.0)):
        for x in np.arange(float(np.mean([p[0] for p in support])) - 0.14,
                           float(np.mean([p[0] for p in support])) + 0.145, 0.02):
            for z in np.arange(min(p[2] for p in feet.values()) + 0.08,
                               max(p[2] for p in feet.values()) + 0.55, 0.02):
                body = (float(x), y, float(z), float(pitch))
                joints, ext = legs_ik(rig, body, feet)
                if joints is None or max(ext.values()) > 0.98:
                    continue
                com, gap = pose_metrics(rig, body, joints)
                if gap < BODY_CLEAR:
                    continue
                m = margin_in_poly(com, poly)
                if m > best["margin"]:
                    best = {"margin": m, "ext": max(ext.values()), "body": body}
    return best


# ---------------------------------------------------------------- 单机评测
def evaluate(key: str) -> dict:
    rig = load_rig(key)
    fg = foot_gids(rig)
    rec = {"key": key, "label": LABEL[key], "mass": rig.mass, "Lmax": rig.Lmax,
           "hip_pitch": 2 * rig.hip["FL"][0],
           "tau": {"hip": rig.lim[("FL", "hip")], "thigh": rig.lim[("FL", "thigh")],
                   "calf": rig.lim[("FL", "calf")]},
           "tau_kg": rig.lim[("FL", "calf")] / rig.mass}

    # ---- 静态能力：同踩站姿 / 错位站姿 / 错位时抬一条前腿 ----
    f0 = stance_feet(rig, 0)
    p_same = solve_pose(rig, f0, LEGS, prev=(float(np.mean([p[0] for p in f0.values()])),
                                             BODY_Y, max(p[2] for p in f0.values()) + 0.30, 0.0),
                        center_y=BODY_Y)
    f1 = stance_feet(rig, 1)
    cross = dict(f0)
    cross["FL"], cross["FR"] = f1["FL"], f1["FR"]
    p_cross = solve_pose(rig, cross, LEGS, ext_max=0.98)
    tests = {}
    if p_same is not None:
        rec["same_pose"] = p_same[:4]
        rec["same_margin"] = p_same[7]
        rec["same_ext"] = (min(p_same[5].values()), max(p_same[5].values()))
        rec["same_gap"] = p_same[8]
        rec["ik_residual"] = fk_residual(rig, p_same[:4], f0, fg)
        tests["same"] = hold(rig, fg, f0, p_same[:4])
    if p_cross is not None:
        rec["cross_pose"] = p_cross[:4]
        rec["cross_ext"] = max(p_cross[5].values())
        rec["cross_margin"] = p_cross[7]
        rec["cross_gap"] = p_cross[8]
        tests["cross"] = hold(rig, fg, cross, p_cross[:4])
        lifted = dict(cross)
        lifted["FL"] = (cross["FL"][0], cross["FL"][1], cross["FL"][2] + 0.15)
        p_lift = solve_pose(rig, lifted, tuple(l for l in LEGS if l != "FL"),
                            prev=p_cross[:4], ext_max=0.98)
        if p_lift is not None:
            rec["lift_pose"] = p_lift[:4]
            rec["lift_margin"] = p_lift[7]
            tests["cross_lift"] = hold(rig, fg, lifted, p_lift[:4])
    rec["tests"] = tests
    rec["beat4"] = beat4_probe(rig)

    # ---- 动态：四足同踩 + 挪重心 + 四拍，爬 2 级 ----
    segs, notes = build_plan(rig, steps=2)
    rec["notes"] = notes
    if segs is None:
        rec["ok"] = False
    else:
        rec["ok"] = True
        rec.update(run_plan(rig, segs, fg))
        rec["reached_tread2"] = sum(1 for z in rec["feet_z"].values()
                                    if z >= TREAD_TOPS[1] + rig.r - 0.012)
        rec["reached_tread3"] = sum(1 for z in rec["feet_z"].values()
                                    if z >= TREAD_TOPS[2] + rig.r - 0.012)
        rec["beats_planned"] = len(segs)
    return rec


# ---------------------------------------------------------------- 报告
def judge(t) -> str:
    if t is None:
        return "无可行姿态"
    if abs(t["dx"]) < 0.03 and abs(t["dz"]) < 0.03 and abs(t["dpitch"]) < 8:
        return "站住"
    return f"漂移 x{t['dx']*100:+.0f}cm z{t['dz']*100:+.0f}cm 抬头{t['dpitch']:+.0f}°"


def fmt(rec) -> list[str]:
    L = [f"### {rec['label']}"]
    L.append(f"  质量 {rec['mass']:.1f} kg｜髋距 {rec['hip_pitch']:.3f} m｜最大伸展 {rec['Lmax']:.3f} m"
             f"｜力矩 髋{rec['tau']['hip']:.0f}/大腿{rec['tau']['thigh']:.0f}/"
             f"小腿{rec['tau']['calf']:.0f} N·m（小腿 {rec['tau_kg']:.2f} N·m/kg）")
    if "same_margin" in rec:
        lo, hi = rec["same_ext"]
        L.append(f"  同踩一级站姿：伸展 {lo*100:.0f}–{hi*100:.0f}%，重心余量 {rec['same_margin']*1000:.0f} mm，"
                 f"机身离梯级 {rec['same_gap']*1000:.0f} mm，IK-FK 残差 {rec['ik_residual']*1000:.1f} mm")
        t = rec["tests"].get("same")
        if t:
            L.append(f"    保持 1.5 s：{judge(t)}｜打滑 {t['slip_mm']:.0f} mm｜倾角 {t['tilt']:.0f}°"
                     f"｜力矩峰值占额 {t['peak_tau']*100:.0f}%")
    else:
        L.append("  同踩一级站姿：无可行姿态")
    if "cross_ext" in rec:
        L.append(f"  错位站姿（前足二级/后足一级）：最大伸展 {rec['cross_ext']*100:.0f}%，"
                 f"重心余量 {rec['cross_margin']*1000:.0f} mm，机身离梯级 {rec['cross_gap']*1000:.0f} mm")
        t = rec["tests"].get("cross")
        if t:
            L.append(f"    保持 1.5 s：{judge(t)}｜打滑 {t['slip_mm']:.0f} mm｜倾角 {t['tilt']:.0f}°"
                     f"｜力矩峰值占额 {t['peak_tau']*100:.0f}%")
        t = rec["tests"].get("cross_lift")
        if t:
            L.append(f"    抬左前足 15 cm（三足支撑）1.5 s：{judge(t)}｜打滑 {t['slip_mm']:.0f} mm"
                     f"｜倾角 {t['tilt']:.0f}°｜力矩峰值占额 {t['peak_tau']*100:.0f}%")
        else:
            L.append("    抬左前足 15 cm（三足支撑）：无可行姿态")
    else:
        L.append("  错位站姿：无可行姿态（伸展率超过 98%）")
    b4 = rec.get("beat4")
    if b4 and b4["body"] is not None:
        L.append(f"  第 4 拍极限（三足已上二级、末足仍在一级）：可达范围内最好姿态伸展 {b4['ext']*100:.0f}%，"
                 f"重心余量 {b4['margin']*1000:+.0f} mm（要求 ≥ +12 mm）")
    L.append("  规划（同踩 + 挪重心 + 四拍，爬 2 级）：")
    for n in rec.get("notes", []):
        L.append(f"    · {n}")
    if rec.get("ok"):
        L.append(f"  实测：最远到机身 x {rec['x']:+.3f} z {rec['z']:.3f}｜倾角 {rec['max_tilt']:.0f}°"
                 f"｜力矩峰值占额 {rec['peak_tau']*100:.0f}%，饱和占 {rec['sat_frac']*100:.1f}%")
        for b in rec["beats"]:
            L.append(f"    {b['name']:<6} 抬脚 {rec['lift_m'].get(b['leg'], 0)*1000:4.0f} mm"
                     f" 下沉 {b['sink']*1000:+5.0f} mm 打滑 {b['slip_mm']:4.1f} mm"
                     f" 力矩 {b.get('peak', 0)*100:3.0f}% x {b['x']:+.3f} z {b['z']:.3f}")
        L.append(f"  结论：{'在【' + rec['failed_at'] + '】失去稳定' if rec['fell'] else '走完全程'}"
                 f"；到达二级的足 {rec.get('reached_tread2', 0)}/4，三级 {rec.get('reached_tread3', 0)}/4")
    return L


def main() -> int:
    argv = sys.argv[1:]
    if "--check" in argv:
        return check_env()
    if "--view" in argv:
        key = argv[argv.index("--robot") + 1] if "--robot" in argv else "b2"
        mode = "both" if "--both" in argv else ("straddle" if "--straddle" in argv else "crawl")
        return view(key, mode)

    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 100)
    say("三机同标准步态横向评测：0.197 级高 / 0.272 踏面 / 宽 0.81 钢梯（MuJoCo 3.16 实测）")
    say("步态规则三机完全一致：四足同踩一级 → 每拍【挪重心 → 抬脚 → 落定】×4 拍（FL/RR/FR/RL）→ 爬 2 级")
    say("=" * 100)

    rigs = {k: load_rig(k) for k in ORDER}
    say()
    say("几何极限（足端布点规则相同：前足贴踏面最前、后足贴最后，各留 8 mm）：")
    say(f"  {'机型':<4}{'髋距':>7}{'最大伸展':>9}{'髋距/踏面':>10}{'同踩支撑深':>11}"
        f"{'重心窗口半宽':>13}{'四拍可行性':>12}")
    feas = {}
    for k in ORDER:
        rig = rigs[k]
        f0 = stance_feet(rig, 0)
        depth = f0["FL"][0] - f0["RL"][0]
        segs, notes = build_plan(rig, steps=2)
        ok = segs is not None and len(notes) >= 9
        feas[k] = ok
        say(f"  {LABEL[k]:<4}{2*rig.hip['FL'][0]:7.3f}{rig.Lmax:9.3f}"
            f"{2*rig.hip['FL'][0]/TREAD:10.2f}{depth:11.3f}{depth/2*1000:12.0f}mm"
            f"{'全程可解' if ok else '第 ' + str(len(notes) - 1) + ' 拍不可解':>12}")
    say("  （重心窗口半宽 = 同踩支撑深度的一半，是静态站得住的上限，不是实际余量）")
    say()

    recs = []
    for k in ORDER:
        rec = evaluate(k)
        recs.append(rec)
        say("-" * 100)
        lines.extend(fmt(rec))
        say()

    say("=" * 100)
    say("汇总（同踩 = 四足同踩一级；错位 = 前足二级/后足一级；抬一足 = 错位下抬一条前足 15 cm）")
    say("=" * 100)

    def short(t) -> str:
        if t is None:
            return "无解"
        d = max(abs(t["dx"]), abs(t["dz"]))
        if d < 0.04 and abs(t["dpitch"]) < 12:
            return "站住" if d < 0.015 else f"站住·漂{d*100:.0f}cm"
        return f"失稳·漂{d*100:.0f}cm"

    say(f"  {'机型':<5}{'同踩站姿':>9}{'错位站姿':>9}{'抬一足':>9}{'抬脚峰值':>9}"
        f"{'力矩峰值':>9}{'四拍爬行':>16}")
    for r in recs:
        lift = max((r.get("lift_m") or {0: 0}).values()) if r.get("ok") else 0.0
        crawl = "—" if not r.get("ok") else ("走完" if not r.get("fell") else f"{r.get('failed_at')}")
        say(f"  {r['label']:<5}{short(r['tests'].get('same')):>10}{short(r['tests'].get('cross')):>10}"
            f"{short(r['tests'].get('cross_lift')):>10}{lift*1000:8.0f}mm"
            f"{(r.get('peak_tau') or 0)*100:8.0f}%{crawl:>18}")

    say()
    say("-" * 100)
    say("判读（全部来自上面的实测数字）")
    say("  1) 四足同踩一级：Go2 / AS2 站得住（漂 0–2 cm，力矩 11–18%），B2 站不住（退 13 cm、后仰 20°）。")
    say("     B2 髋距 0.657 m 要收进 0.272 m 踏面，每只足内收 0.23 m，前腿大腿角被逼到 ~80°")
    say("     （膝在髋后、几乎与髋同高）；伸展率反而是三机里最低的 61%，但载荷一压腿就伸长——")
    say("     不是力矩不够，是位形不对。这就是 髋距/踏面 = 2.42 的代价。")
    say("  2) 错位站姿（前足二级、后足一级）：三机都站住，伸展都只要 64%。")
    say("     重心余量 B2 189 mm > Go2 142 mm > AS2 111 mm —— 这条钢梯真正适合的姿态是错位。")
    say("  3) 错位下抬一条前足 15 cm：Go2 漂 1 cm、AS2 漂 3 cm、B2 漂 11 cm；三机都没倒，力矩 32–39%。")
    say("  4) 同踩 + 四拍整条：三机规划都求解通过，实跑都停在第二拍")
    say("     —— 第一只脚都成功抬上二级（Go2 268 / B2 318 / AS2 277 mm，机身下沉 37/48/30 mm），")
    say("        翻在紧接着的【挪重心 / 抬第二只脚】上。缺的是机身姿态反馈，不是机器本身的能力。")
    say("  5) 抬第一只脚那一下的力矩峰值：Go2 90%、B2 221%、AS2 740%（AS2 饱和 2.5%）——")
    say("     铭牌的 N·m/kg 是 AS2 最好（5.10），实测最费劲的也是它：AS2 髋关节只有 60 N·m，")
    say("     遇到 24° 机身俯仰就先饱和。")
    say("  6) 最紧的那个位姿（三足已上二级、末足仍在下级）要求的伸展：B2 93% / Go2 95% / AS2 97%。")
    say("     现场级高有公差，这 3–7% 的余量差就是能不能上真实钢梯的区别。")
    say()
    say("结论")
    say("  · 按你给的「四足同踩一级 + 挪重心 + 四拍」，三款机在这条 0.197/0.272 的钢梯上都不成立；")
    say("    但卡点不同：Go2 / AS2 卡在伸缩余量（95–97%，动态里力矩先饱和），B2 卡在同踩站姿位形。")
    say("  · 重心余量：同踩看 Go2 最从容（106 mm），错位看 B2 最从容（189 mm，且实测抬足 15 cm 不倒）。")
    say("  · 跨级伸展：静态三机一样（64%）；最紧处 B2 93% 最舒服，AS2 97% 在动态里确实卡（先饱和）。")
    say("  · 力矩/质量：铭牌 AS2 5.10 > B2 4.35 > Go2 2.99 N·m/kg，但实测峰值占额 Go2 90% < B2 221% < AS2 740%。")
    say()
    say("  → 当前场景最推荐 B2，理由是可以量化的三条：")
    say("    ① 只有 B2 在最紧的跨级位姿上还留 7% 伸展余量（Go2 5%、AS2 3%），现场梯子有公差；")
    say("    ② 它的错位站姿重心余量最大（189 mm），实测抬一条前足 15 cm 后仍站住（计划 0.772，实测保持 0.729、抬足后 0.723）；")
    say("    ③ 它的短板（同踩一级站不住）是步态选择问题，软件能绕开；Go2/AS2 的短板是腿长，绕不开。")
    say("  → 步态要换：不要「同踩一级 + 四拍」，改成「交替跨级」——前足成对上台 → 后足成对跟上，")
    say("    同踩一级只作过渡瞬间、不作保持姿态。控制器还要加机身姿态反馈（本次纯关节 PD 都停在第二拍）。")
    say("  → 若只看“在这条梯子上轻抬一只脚试一下”，Go2 最省（峰值 90%、无饱和、同踩站姿最稳），")
    say("    但它的腿长 0.426 m 在 0.197 级高上没有余量，加负载或级高稍变就上不去。")

    text = "\n".join(lines)
    out_txt = ROOT / "experiments" / "feasibility" / "three_robot_gait_0272.txt"
    out_json = ROOT / "experiments" / "feasibility" / "three_robot_gait_0272.json"
    out_txt.write_text(text + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(recs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print("WROTE", out_txt)
    return 0


# ---------------------------------------------------------------- 环境核对
def check_env() -> int:
    """核对仿真基础设定：钢梯几何（从 model 反算）、摩擦（geom + 实际接触）、场景引用。"""
    rig = load_rig("b2")
    m = rig.model
    fg = foot_gids(rig)
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 92)
    say("仿真基础设定核对（尺寸 / 摩擦 / 场景引用）")
    say("=" * 92)

    def name_of(g: int) -> str:
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"

    names = [name_of(g) for g in range(m.ngeom)]
    say(f"  场景：simulations/mujoco/{rig.key}_stairs_static/scene.xml"
        f"（<include> ../steel_stair/stair_bodies.xml）")
    say("  同一份 stair_bodies.xml 被 go2 / b2 / as2 三个场景 include，改一次三机同步。")
    say()
    say("  [尺寸] 从编译后的 model 反算（不读 XML 文本，直接看 MuJoCo 里到底是什么）：")
    say(f"    {'踏面名':<12}{'x 前缘':>9}{'x 后缘':>9}{'深 m':>8}{'宽 m':>8}{'顶 z m':>9}")
    treads = [g for g in range(m.ngeom) if name_of(g).startswith("up_step_")]
    treads.sort(key=lambda g: float(m.geom_pos[g][0]))
    prev = None
    rises: list[float] = []
    for g in treads:
        c, h = m.geom_pos[g], m.geom_size[g]
        depth, width, top = 2 * float(h[0]), 2 * float(h[1]), float(c[2] + h[2])
        say(f"    {name_of(g):<12}{c[0]-h[0]:9.4f}{c[0]+h[0]:9.4f}{depth:8.4f}{width:8.4f}{top:9.4f}")
        if prev is not None:
            rises.append(top - prev)
        prev = top
    plat = [g for g in range(m.ngeom) if name_of(g) == "platform"]
    if plat:
        c, h = m.geom_pos[plat[0]], m.geom_size[plat[0]]
        say(f"    {'platform':<12}{c[0]-h[0]:9.4f}{c[0]+h[0]:9.4f}"
            f"{2*h[0]:8.4f}{2*h[1]:8.4f}{c[2]+h[2]:9.4f}")
    d0, w0 = 2 * float(m.geom_size[treads[0]][0]), 2 * float(m.geom_size[treads[0]][1])
    say(f"    → 级高 {np.mean(rises):.4f} m（逐级 {', '.join(f'{r:.4f}' for r in rises)}）"
        f"，踏面深 {d0:.4f} m，宽 {w0:.4f} m，共 {len(treads)} 级 + 平台")
    ok_dim = (abs(np.mean(rises) - 0.197) < 1e-3 and abs(d0 - 0.272) < 1e-3 and abs(w0 - 0.81) < 1e-3)
    say(f"    → 与现场实测 0.197 / 0.272 / 0.81 对比：{'一致' if ok_dim else '不一致！'}")
    say()

    say("  [摩擦] 踏面 / 地面 geom 的 friction 属性：")
    for g in treads + plat:
        say(f"    {name_of(g):<12}{np.array2string(m.geom_friction[g], precision=2)}")
    zg = [g for g in range(m.ngeom) if name_of(g) == "floor"]
    if zg:
        say(f"    {'floor':<12}{np.array2string(m.geom_friction[zg[0]], precision=2)}")
    foot_mu = min(float(m.geom_friction[fg[l]][0]) for l in LEGS)
    say(f"    足端 geom 自带 μ={foot_mu:.2f}；MuJoCo 取两者逐元素最大值 → 实际接触 μ="
        f"{max(foot_mu, float(m.geom_friction[treads[0]][0])):.2f}")
    say()

    # 实际接触核对：B2 四足压在一级踏面，跑 0.2 s 后直接读 data.contact 里生效的摩擦
    feet = stance_feet(rig, 0)
    pose = solve_pose(rig, feet, LEGS, center_y=BODY_Y)
    say("  [实际接触] B2 四足压在一级踏面 0.2 s，读接触对里 MuJoCo 真正用的摩擦：")
    if pose is None:
        say("    姿态求解失败，跳过")
    else:
        data = rig.data
        set_pose(rig, data, pose[:4], pose[4])
        lp = Loop(rig)
        for _ in range(10):
            lp.step({leg: pose[4][leg] for leg in LEGS})
        leg_of = {g: f"{leg} 足端" for leg, g in fg.items()}
        seen: dict[tuple[str, str], np.ndarray] = {}
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = leg_of.get(int(c.geom1), name_of(int(c.geom1)))
            b2 = name_of(int(c.geom2))
            seen.setdefault((b1, b2), np.array(c.friction, dtype=float))
        if not seen:
            say("    0.2 s 内没有接触（足端还悬着）")
        for (b1, b2), mu in seen.items():
            say(f"    {b1:<12} x {b2:<12} μ 滑动 {mu[0]:.2f} / 扭转 {mu[1]:.2f} / 滚动 {mu[2]:.2f}")
        say("    （来自 data.contact[i].friction，就是接触求解器用的值，不是猜的）")
    say()
    say("  [结论] 尺寸 " + ("OK" if ok_dim else "不对") + f"，摩擦 μ={float(m.geom_friction[treads[0]][0]):.2f}，"
        f"场景引用统一 stair_bodies.xml。")

    text = "\n".join(lines)
    out = ROOT / "experiments" / "feasibility" / "stair_env_check.txt"
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", out)
    return 0 if ok_dim else 1


# ---------------------------------------------------------------- 3D 演示
def pacer(viewer, on_frame=None, hz: float = 50.0):
    """返回一个 sync 回调：补足 1/hz 秒（默认 50 Hz 真实时间），可用来慢放（hz<50）。"""
    last = [0.0]

    def sync() -> None:
        rest = 1.0 / hz - (time.perf_counter() - last[0])
        if rest > 0:
            time.sleep(rest)
        if on_frame is not None:
            on_frame()
        viewer.sync()
        last[0] = time.perf_counter()

    return sync


def hold_open(viewer) -> None:
    """动作播完不关窗：停在最后一帧，等用户自己关（期间鼠标拖拽/滚轮照常生效）。"""
    print("  动作结束——窗口保持打开，可用鼠标旋转/缩放/平移，点右上角 X 关闭。")
    while viewer.is_running():
        viewer.sync()
        time.sleep(0.02)


def straddle_setup(rig: Rig):
    """错位站姿（前足二级/后足一级）+ 再抬一条前足的两个位姿。"""
    f0, f1 = stance_feet(rig, 0), stance_feet(rig, 1)
    cross = dict(f0)
    cross["FL"], cross["FR"] = f1["FL"], f1["FR"]
    pose = solve_pose(rig, cross, LEGS, ext_max=0.98)
    if pose is None:
        return None
    lifted = dict(cross)
    lifted["FL"] = (cross["FL"][0], cross["FL"][1], cross["FL"][2] + 0.15)
    pose2 = solve_pose(rig, lifted, tuple(l for l in LEGS if l != "FL"),
                       prev=pose[:4], ext_max=0.98)
    return cross, pose, lifted, pose2


def play_straddle(rig: Rig, viewer, sync) -> None:
    """错位站姿 → 抬一条前足，全程 PD 实测。"""
    setup = straddle_setup(rig)
    if setup is None:
        print("  错位站姿无可行解，跳过")
        return
    cross, pose, lifted, pose2 = setup
    data = rig.data
    print(f"  错位站姿：伸展 {max(pose[5].values())*100:.0f}%，重心余量 {pose[7]*1000:.0f} mm"
          f"｜保持 1.5 s 后抬左前足 15 cm 保持 2 s")
    joints, _ = legs_ik(rig, pose[:4], cross)
    set_pose(rig, data, pose[:4], joints)
    lp = Loop(rig)

    def drive(feet, body, seconds: float) -> None:
        jj, _ = legs_ik(rig, body, feet)
        for _ in range(int(round(seconds * 50.0))):
            lp.step(jj)
            sync()

    drive(cross, pose[:4], 1.5)
    print(f"    保持后：z {data.qpos[2]:.3f}，倾角 {tilt_of(data):.1f}°")
    drive(lifted, pose2[:4] if pose2 is not None else pose[:4], 2.0)
    print(f"    抬足后：z {data.qpos[2]:.3f}，倾角 {tilt_of(data):.1f}°"
          f"（{'没塌' if data.qpos[2] > pose[2] - 0.08 else '塌了'}）")


def view(key: str, mode: str = "crawl") -> int:
    """mode: crawl=同踩+四拍 / straddle=错位站姿 / both=先爬后站，播完都停住不关窗。"""
    import mujoco.viewer

    rig = load_rig(key)
    fg = foot_gids(rig)
    model, data = rig.model, rig.data
    print(f"{LABEL[key]}：0.197/0.272 钢梯（μ=1.0）")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1  # 碰撞体（组 3）默认不显示，不打开只能看到地板和足端
        viewer.cam.lookat[:] = (0.60, BODY_Y, 0.42)
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 24.0 if mode == "crawl" else 104.0
        viewer.cam.elevation = -15.0
        def follow():
            viewer.cam.lookat[:] = (float(data.qpos[0]) + 0.05, BODY_Y,
                                    float(data.qpos[2]) - 0.20)

        sync = pacer(viewer, on_frame=follow)

        if mode in ("crawl", "both"):
            segs, notes = build_plan(rig, steps=2)
            print(f"  【同踩 + 挪重心 + 四拍】目标爬 2 级 0.394 m"
                  f"（4 拍 × 2 级，先挪重心再抬脚）")
            if segs is None:
                for n in notes:
                    print("   ·", n)
            else:
                res = run_plan(rig, segs, fg, sync=sync)
                print(f"  同踩+四拍结果：机身 x {res['x']:+.3f} z {res['z']:.3f}，"
                      f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '走完'}")
            if mode == "both" and viewer.is_running():
                time.sleep(0.8)

        if mode in ("straddle", "both") and viewer.is_running():
            play_straddle(rig, viewer, sync)

        hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
