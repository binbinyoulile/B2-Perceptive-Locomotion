"""B2 「交替跟进」步态：从错位姿态起步，一次只动一条腿，一路爬到平台（第 4 级）。

文件目的：
    把用户提的「交替跟进步态」在 0.197/0.272 钢梯上跑出来，并检验它能不能**登上平台**：
        起步（错位）：FL 第2级 / FR 第1级 / RR 第1级 / RL 地面
        每个循环四步：FR +1（与 FL 汇合）/ RL +1（与 RR 汇合）/ FL +1 / RR +1（与 FR 汇合）
    每循环四步做完，四足所在级各 +1，正好是起步姿态整体上移一级 —— 所以能一直循环。
    跑到平台后 FL 已无级可上（平台后沿 x=1.0326、顶 0.788、走面深 2 m），
    循环只会剩下"后足补最后一级"，四个循环共 12 步：
        FR 1→2 / RL 0→1 / FL 2→3 / RR 1→2      循环1
        FR 2→3 / RL 1→2 / FL 3→4（上平台）/ RR 2→3
        FR 3→4 / RL 2→3 /（FL 已到平台，跳过）/ RR 3→4
        （FR/RR/FL 已到平台，跳过）/ RL 3→4
    关键点：这条步态**任何时刻都不会出现"四足同踩一级"**（B2 那个姿态站不住），
    始终是"前足成对级 + 后足成对级"错开一级。

每一步怎么走（三条硬约束，全靠搜索而不是手摆）：
    1) 挪重心：四足都还在原位，把机身移到"抬起那条腿以后剩下三足"的内侧；
    2) 抬腿：这条腿沿"竖直抬过鼻端 → 水平前送 → 落下"的路径上台；
    3) 落定：四足都在新位置，机身回到支撑面中间。
    每个位姿都在 (x, y, z, pitch 0–30°) 里搜：四腿可达、伸展 <95%、
    非足端碰撞体离梯级 ≥30 mm、重心在支撑多边形内；并按"离上一位姿最近"做平滑。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（0.197/0.272/0.81，μ=1.0）
输出：
    experiments/b2/b2_follow_gait.txt / .json
    experiments/b2/b2_follow_gait.png —— 起步 / 第4步 / 第8步 / 第12步 四联图（--png）
依赖：mujoco 3.16、numpy；复用
      three_robot_gait_0272（rig / IK / PD 循环 / run_plan / pacer）
      b2_diag_stance（候选枚举 / 非足端避碰 / PD 保持）
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py          （默认 4 循环 = 12 步上平台）
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --cycles 2
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --view   （实时播放；--slow = 0.5 倍速）
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --png

关键假设：
    - 级高 0.197、踏面深 0.272；第 4 级是 2 m 深平台（后沿 x=1.0326、顶 0.788），
      所以上平台的落点仍按"前足贴该级前沿、后足贴后沿"算，等价于再走一级同样的踏步。
    - 50 Hz PD（kp/kd 800/20）+ 重力前馈；摩擦 μ=1.0。
    - 每步时序：挪重心 1.0 s → 抬腿 1.5 s → 落定 0.5 s（慢，是为了让 PD 跟得上）。
已知限制：
    - 仍是"脚本化轨迹 + 纯关节 PD"，机身姿态只靠 body_servo 这个比例项，没有真正的姿态闭环。
    - 机身只按 pitch 参数化（不滚转）。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "b2"))
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

import three_robot_gait_0272 as G  # noqa: E402
import b2_diag_stance as D  # noqa: E402

OUT_TXT = ROOT / "experiments" / "b2" / "b2_follow_gait.txt"
OUT_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait.json"
OUT_PNG = ROOT / "experiments" / "b2" / "b2_follow_gait.png"
OUT_TRACE = ROOT / "experiments" / "b2" / "b2_follow_gait_trace.txt"

TREAD_DEPTH = 0.272
TREAD_REAR0 = 0.2166                     # 第 1 级后沿
LEVEL_TOP = {0: 0.0, 1: 0.197, 2: 0.394, 3: 0.591, 4: 0.788}
TOP_LEVEL = 4                            # 平台：再往上没有级了
EXT_CAP = 0.95
PITCH_HI = 30.0          # 抬头上限。⚠ 不要为了"多省点余量"调小这个：轨迹是按这套参数逐步推出来的，
#   改任何一个参数都会让前 12 步的位姿整体平移 2–4 cm，实测会把已经跑通的爬梯段重新跑坏。
EDGE = 0.010
PRELOAD = 0.003
SWING_CLEAR = 0.060      # 抬脚时足心比目标高多少（含足端半径另加）。
#   原来是 0.03：第 2 循环抬 FR（2→3）时足底离三级踏面只剩 3.3 cm，
#   PD 静置要沉 2.7 cm，实测足端扫到三级前沿、39–308 N 的接触，跟着侧滑失稳。加到 6 cm。

SHIFT_S, LIFT_S, SETTLE_S = 1.0, 1.5, 0.5
WIN_X, WIN_Z = 0.28, 0.24    # 位姿搜索窗口（绕上一位姿）。⚠ 也别调：前 12 步的位姿依赖这个窗口

PLAT_REAR = 0.120        # 平台上：后足落在后沿以东多远。不能贴边：实测贴到 0.042 时，
#   落地后被机身拖动 1.5 cm 就滑出后沿、变成悬空（z 0.766、受力 0 N），整机只剩对角两足支撑然后侧翻。
PLAT_FRONT = 0.230       # 平台上：前足刚上来时先踩在这里（= 再走一级一样的踏步）
PLAT_ADJ = 0.520         # 两只前足都上平台后，在平地上各向前迈一步到这里：
#   上平台那几拍的支撑三角形是"前面两只足 + 后面一只足"，跨距只有 0.19 m 时内切圆仅 72 mm
#   （实测第 12 步就是绕 FL–RR 对角线翻下去）。前足在平地上前移把跨距撑到 0.40，
#   最后一拍的余量从 72 抬到 114 mm —— 而且这一步完全在平台上，不影响已经跑通的爬梯段。

# ---- 平地起步 ----
GROUND_X, GROUND_Z = -0.115, 0.550   # 平地站姿的机身位置（离楼梯口 0.33 m，正常站高）
GROUND_FRONT, GROUND_REAR = 0.30, -0.185   # 平地站姿的前/后足离机身中心多远的 x
#   前足落在 0.185：正好在楼梯鼻端（0.2166）之外，足端半径 0.032 还留 3 cm。
#   后足不放在"对称"的 -0.357：那样后足到第 1 级要走 0.7 m，得拆成 3 步；
#   收一点（后髋前 0.16 m）看着仍是正常站姿，后足两步就能上到位。


# ---- 完整作业流（--demo）的参数：平地走近 → 入口 → 爬梯 → 登顶站直 + 平台前行 ----
APPROACH_X = -0.715          # 平地起步的机身 x：比入口起点 GROUND_X(-0.115) 后退 0.6 m
APPROACH_CYCLES = 2          # 走近段 = 2 个循环 = 8 步
APPROACH_STRIDE = 0.30       # 步幅 0.30 m → 2 循环正好前进 0.60 m，落点回到名义站姿
PLAT_CYCLES = 2              # 平台段 = 2 个循环 = 8 步
PLAT_STRIDE = 0.26
PLAT_STAND_OFF = 0.470       # 站直后机身离平台顶多高。平地站姿是 0.550，这里取 0.470：
#   实测站到全高（1.338 m）时腿伸展 84%，而爬梯末态在平台上带 5–10 cm 横向/偏航漂移，
#   三足支撑那一拍没有修正余量 → 前倾 45° 摔倒；低 8 cm 后 8/8 步全通（倾角 19°、峰值 92%）。
PLAT_PITCH_HI = 8.0          # 平台段允许的抬头（越小越像正常走路，但三足支撑时余量会小一点）
STANDUP_S = 4.0              # 登顶后「站直」（仰 9° → 0°、z 1.208 → 1.338）用这么久；太短会残留 15° 倾角


def approach_stance(rig: G.Rig):
    """完整作业流的起点：和入口站姿同一个正常站姿，只是整体后退 APPROACH 距离。"""
    feet = {}
    for leg in G.LEGS:
        dx = GROUND_FRONT if leg in ("FL", "FR") else GROUND_REAR
        feet[leg] = (APPROACH_X + dx, G.BODY_Y + rig.foot_y[leg], rig.r + PRELOAD)
    return feet, (APPROACH_X, G.BODY_Y, GROUND_Z, 0.0)


def ground_stance(rig: G.Rig):
    """平地正常站姿：四足都在地面、俯仰 0、机身 0.55 m。（这是"狗先正常站在那儿"的那一帧。）"""
    feet = {}
    for leg in G.LEGS:
        dx = GROUND_FRONT if leg in ("FL", "FR") else GROUND_REAR
        feet[leg] = (GROUND_X + dx, G.BODY_Y + rig.foot_y[leg], rig.r + PRELOAD)
    return feet, (GROUND_X, G.BODY_Y, GROUND_Z, 0.0)


def entry_spec(rig: G.Rig) -> list[tuple[str, tuple, str]]:
    """平地站姿 → 标准状态 S0（FL2/FR1/RR1/RL0）的 7 步。

    按用户提的顺序：先把两只前足送上第 1 级（再把左前足抬到第 2 级），
    然后把两只后足在平地上往前挪、右后足上第 1 级，最后左后足挪到楼梯口那个位置。
    走完 7 步四足落点与 S0 逐足相同，后面直接接已经验证过的 14 步爬梯计划。
    """
    def g(leg: str, x: float):
        return (x, G.BODY_Y + rig.foot_y[leg], rig.r + PRELOAD)

    return [
        # 顺序按两条硬规矩排：
        # ① 任何时刻同一级上最多两只脚（三只脚挤在 0.272 m 的踏面上，B2 髋距 0.657 撑不住）；
        # ② 抬起某条腿时剩下三足的三角形要够宽 —— 两只前足停在第 1 级同一 x 时是薄三角，所以
        #    左后足在"两前足刚上第 1 级、机身还在后面"时挪到地面位置，那时三角形最宽。
        ("FR", foot_target(rig, "FR", 1), "FR 上第1级"),
        ("FL", foot_target(rig, "FL", 1), "FL 上第1级"),
        ("RL", g("RL", TREAD_REAR0 - 0.15), "RL 平地到位"),
        ("RR", g("RR", 0.000), "RR 平地前移"),
        ("FL", foot_target(rig, "FL", 2), "FL 上第2级"),
        ("RR", foot_target(rig, "RR", 1), "RR 上第1级"),
    ]


def edge_x(level: int) -> float:
    return TREAD_REAR0 + TREAD_DEPTH * (level - 1)


def foot_target(rig: G.Rig, leg: str, level: int):
    """某条腿在第 level 级上的落点。

    level ≤ 0 都算地面：地面就是一块"往后退的假踏面"，每退一级减 0.272 m
    （这样"从平地起步"可以按同一条循环往上数，不需要单独写一套落地逻辑）。
    第 4 级是 2 m 深平台：后足不能贴边（贴边落地会被拖动滑出去），前足往里踩深一点。
    """
    y = G.BODY_Y + rig.foot_y[leg]
    if level <= 0:
        return (TREAD_REAR0 - 0.15 + TREAD_DEPTH * level, y, rig.r + PRELOAD)
    if level == TOP_LEVEL:
        x = edge_x(level) + (PLAT_FRONT if leg in ("FL", "FR") else PLAT_REAR)
        return (x, y, LEVEL_TOP[level] + rig.r + PRELOAD)
    cx = edge_x(level) + TREAD_DEPTH / 2.0
    d = TREAD_DEPTH / 2.0 - rig.r - EDGE
    x = cx + d if leg in ("FL", "FR") else cx - d
    return (x, y, LEVEL_TOP[level] + rig.r + PRELOAD)


def start_levels(base: int = 0) -> dict:
    """标准状态的基准：FL = base+2 / FR、RR = base+1 / RL = base。
    base=0 就是验证过的爬梯起点（FL2/FR1/RR1/RL0）；base=-1 是"平地起步"那一态
    （FL 已在第 1 级、其余三足在平地），走完一个循环正好回到 base=0。"""
    return {"FL": 2 + base, "FR": 1 + base, "RR": 1 + base, "RL": 0 + base}


def steps_of_cycle(c: int, base: int = 0) -> list[tuple[str, int, int]]:
    """第 c 个循环的四步：(腿, 从第几级, 到第几级)。"""
    return [("FR", 1 + base + c, 2 + base + c), ("RL", 0 + base + c, 1 + base + c),
            ("FL", 2 + base + c, 3 + base + c), ("RR", 1 + base + c, 2 + base + c)]


def pick(cands, prev=None, tol: float = 0.030):
    """在候选里挑位姿：余量不低于最优值 tol 的前提下，离上一位姿最近。

    为什么不一味取"余量最大"：纯关节 PD 没有俯仰闭环，机身抬头越大越容易自己往上翻
    （实测第 3 步按最大余量选出的 27° 抬头，静置时自己后仰到 39°、只剩一条后腿着地）。
    但 tol=0.03 在临界的一拍会白丢 2–3 cm 余量：所以"抬腿/挪重心"这两段用 tol=0.005
    （几乎取最大余量，只是挑其中离上一位姿最近的），只有"落定"仍用 0.03 防漂。
    """
    if prev is None:
        return max(cands, key=lambda c: c["margin"])
    top = max(c["margin"] for c in cands)
    sub = [c for c in cands if c["margin"] >= top - tol]
    return min(sub, key=lambda c: (abs(c["body"][0] - prev[0]) + abs(c["body"][2] - prev[2])
                                   + 0.35 * abs(c["body"][3] - prev[3])))


def solve(rig: G.Rig, fg, feet, support, prev=None, tol: float = 0.030,
          pitch_hi: float = PITCH_HI, win: float | None = None):
    """细网格搜索（2 cm / 3°），窗口绕上一位姿。
    粗网格（5 cm）快 5 倍，但搜出的位姿动力学上走不通，别换；窗口只缩"搜索范围"，不改分辨率。
    pitch_hi=0 把入口段压成"平地走路"：不压的话规划器会为了多要点余量把机身一路抬到 30°。
    win 是显式给的小窗口（入口段用）：让机身只能在小范围里挑（尤其横向偏重心），
    整体前进量由给定的锚点定，规划器不会自己乱跑。
    """
    wx = WIN_X if win is None else win
    wz = WIN_Z if win is None else win
    if prev is None:
        xs = np.arange(-0.12, 0.96, 0.02)      # 平地起步时机身会退到楼梯口之前（x<0.2）
        zs = np.arange(0.56, 1.12, 0.02)       # 也不要从 0.46 起：太低了腿是折着的，
        #   B2 折着腿撑不住 69 kg，PD 会沉 4–7 cm、机身直接蹭到梯级（实测 2017 N 接触）。
    else:
        # 注意别在这里夹一个"下界 -0.20"：入口段锚点会退到 -0.35，一夹就变成空区间（xs 为空 → 静默 0 候选）
        xs = np.arange(prev[0] - wx, prev[0] + wx + 1e-9, 0.02)
        zs = np.arange(max(0.40, prev[2] - wz), prev[2] + wz + 1e-9, 0.02)
    cands = D.enumerate_poses(rig, feet, pitch_lo=0.0, pitch_hi=pitch_hi, ext_max=EXT_CAP,
                              xs=xs, ys=np.arange(G.BODY_Y - 0.10, G.BODY_Y + 0.101, 0.02),
                              zs=zs, support=support)
    cands = D.filter_clear(rig, cands, fg)
    if not cands:
        if len(xs) == 0 or len(zs) == 0:      # 空搜索区间不是"不可行"，是参数写错了，必须吵出来
            raise ValueError(f"搜索区间为空：xs {len(xs)} 个、zs {len(zs)} 个（prev={prev}）")
        return None
    best = pick(cands, prev, tol)
    # 命中窗口边界 → 说明真正的解可能在窗口外（下一步会因此解不出来），必须报警而不是静默
    best["win_hit"] = bool(prev is not None and (
        abs(best["body"][0] - prev[0]) > wx - 0.021 or
        abs(best["body"][2] - prev[2]) > wz - 0.021))
    return best


def read_state(rig: G.Rig, fg) -> dict:
    """实测状态：机身 (x, y, z, pitch) + 四足实际位置。两段式流程用它当第二段的起点。"""
    d = rig.data
    return {"body": (float(d.qpos[0]), float(d.qpos[1]), float(d.qpos[2]), G.pitch_of(d)),
            "feet": {leg: tuple(float(v) for v in d.geom_xpos[g]) for leg, g in fg.items()}}


def merge_res(a, b):
    """合并两段的运行结果（入口段 + 重新规划后的爬梯段）。"""
    if a is None:
        return b
    if b is None:
        return a
    out = dict(b)
    out["beats"] = a["beats"] + b["beats"]
    out["max_tilt"] = max(a["max_tilt"], b["max_tilt"])
    out["sat_frac"] = max(a["sat_frac"], b["sat_frac"])
    out["lift_m"] = {k: max(a["lift_m"][k], b["lift_m"][k]) for k in a["lift_m"]}
    if a["peak_tau"] >= b["peak_tau"]:
        out["peak_tau"], out["peak_act"] = a["peak_tau"], a["peak_act"]
    out["fell"] = a["fell"] or b["fell"]
    out["failed_at"] = a["failed_at"] or b["failed_at"]
    return out


def build(rig: G.Rig, fg, cycles: int, base: int = 0, entry: bool = False,
          only_entry: bool = False, state: dict | None = None,
          approach: bool = False, only_approach: bool = False,
          platform: bool = False, only_platform: bool = False):
    """生成某一段轨迹 + 每步的计划信息。返回 (segs, steps, notes, 末态 feet)。

    entry=True：从平地正常站姿起步，走 6 步进入标准状态 S0，再接 base=0 的爬梯计划。
    entry=False：直接从 S0（楼梯上的错位姿态）起步 —— 就是之前验证过的那 14 步。
    only_entry=True：只生成入口那一段（两段式流程的第一段）。
    state=实测状态：按实测机身/足端位姿生成（爬梯段 / 入口段 / 平台段都可以用）。
    approach=True（配合 only_approach）：平地起步段 —— 从楼梯口后方的正常站姿走到入口起点。
    platform=True（配合 only_platform）：登顶后段 —— 先站直，再在平台上往前走几步。
    """
    notes: list[str] = []
    if state is not None and not (entry or approach or platform):
        # 两段式流程的第二段：入口已经跑完，按**实测**机身/足端位姿生成爬梯段。
        # 不插"起步稳住"段（机器人已经在那儿了，不需要再摆一次）。
        feet = {leg: tuple(state["feet"][leg]) for leg in G.LEGS}
        levels0 = start_levels(base)
        dev = max(max(abs(feet[leg][k] - foot_target(rig, leg, levels0[leg])[k]) for k in (0, 2))
                  for leg in G.LEGS)
        notes.append(f"按实测重规划爬梯段：实测落点与 S0 目标最大偏差 {dev*1000:.0f} mm")
        body = {"body": tuple(state["body"])}
        levels = dict(levels0)
        segs, steps = [], []
        cur = body
    elif approach:
        feet, gbody = approach_stance(rig)
        joints, ext = G.legs_ik(rig, gbody, feet)
        if joints is None:
            return None, [], ["平地站姿 IK 不可达"], None
        com, gap = G.pose_metrics(rig, gbody, joints)
        poly = [(feet[k][0], feet[k][1]) for k in G.LEGS]
        body = {"body": gbody, "joints": joints, "ext": ext, "max_ext": max(ext.values()),
                "com": com, "margin": G.margin_in_poly(com, poly), "gap": gap}
        levels = start_levels(base)
        segs = [G.Seg("起步站稳", 0.6, gbody, gbody, dict(feet), dict(feet))]
        steps = []
        cur = body
        notes.append(f"平地起步：机身 ({gbody[0]:+.3f}, {gbody[2]:.3f}) pitch 0°、四足在地面站姿；"
                     f"距入口起点 {GROUND_X - gbody[0]:.2f} m")
    elif platform:
        if state is None:
            return None, [], ["平台段需要实测状态"], None
        feet = {leg: tuple(float(v) for v in state["feet"][leg]) for leg in G.LEGS}
        body0 = tuple(float(v) for v in state["body"])
        stand = (body0[0], G.BODY_Y, LEVEL_TOP[TOP_LEVEL] + PLAT_STAND_OFF, 0.0)
        joints, _ = G.legs_ik(rig, stand, feet)
        if joints is None:
            return None, [], ["平台站直位姿 IK 不可达"], None
        # 登顶后先停一拍再站起来：爬梯最后一步（后足上平台）落地时还在晃，
        # 立刻抬腿会让支撑腿打滑（实测第一拍打滑 248 mm、跟着侧翻 45°）。
        segs = [G.Seg("登顶停顿", 1.5, body0, body0, dict(feet), dict(feet)),
                G.Seg("登顶站稳", STANDUP_S, body0, stand, dict(feet), dict(feet)),
                G.Seg("站定", 2.0, stand, stand, dict(feet), dict(feet))]
        steps = []
        body = {"body": stand, "max_ext": 0.0, "margin": 0.0, "ext": {}}
        cur = body
        levels = start_levels(base)
        notes.append(f"登顶站直：机身 z {body0[2]:.3f} pitch {math.degrees(body0[3]):+.1f}° → "
                     f"z {stand[2]:.3f} pitch 0°（平台顶 {LEVEL_TOP[TOP_LEVEL]:.3f} + 站高 {PLAT_STAND_OFF:.3f}，"
                     f"比平地站高 {GROUND_Z:.3f} 低 {(GROUND_Z - PLAT_STAND_OFF)*100:.0f} cm 以留横向修正余量）")
    elif entry:
        if state is not None:
            # 完整作业流的第一段之后：机器狗是自己走过来的，落点不可能正好等于名义站姿，
            # 所以入口段也按实测重规划（和爬梯段同一个思路，见 docs/B2_PLATFORM_CLIMB.md §6）。
            feet = {leg: tuple(float(v) for v in state["feet"][leg]) for leg in G.LEGS}
            gbody = tuple(float(v) for v in state["body"])
            dev = max(max(abs(feet[leg][k] - ground_stance(rig)[0][leg][k]) for k in (0, 2))
                      for leg in G.LEGS)
            notes.append(f"按实测重规划入口段：实测落点与平地站姿最大偏差 {dev*1000:.0f} mm、"
                         f"机身 ({gbody[0]:+.3f}, {gbody[2]:.3f}) pitch {math.degrees(gbody[3]):+.1f}°")
        else:
            feet, gbody = ground_stance(rig)
        joints, ext = G.legs_ik(rig, gbody, feet)
        if joints is None:
            return None, [], ["平地站姿 IK 不可达"], None
        com, gap = G.pose_metrics(rig, gbody, joints)
        poly = [(feet[k][0], feet[k][1]) for k in G.LEGS]
        body = {"body": gbody, "joints": joints, "ext": ext, "max_ext": max(ext.values()),
                "com": com, "margin": G.margin_in_poly(com, poly), "gap": gap}
        levels = start_levels(base)
        segs, steps = [], []
        cur = body
    else:
        levels0 = start_levels(base)
        feet = {leg: foot_target(rig, leg, levels0[leg]) for leg in G.LEGS}
        body = solve(rig, fg, feet, G.LEGS)
    if body is None:
        return None, [], ["起步姿态无可行解"], None
    if state is None and not entry:
        levels = start_levels(base)
        segs = [G.Seg("起步稳住", 0.6, body["body"], body["body"], dict(feet), dict(feet))]
        steps = []
        cur = body

    def solve_pair(new_feet: dict, support3: tuple, pitch_hi: float = PITCH_HI):
        """解"抬腿位姿"和"挪重心位姿"这一对。

        抬腿那 1.5 s 是最危险的（只有三足撑着），所以取余量尽量大的位姿（tol=0.005）。
        但如果这么选导致抬腿段要挪 3 cm 以上机身，就退回 tol=0.03 的"贴着上一位姿"解：
        实测只有 RL 从地面上一级那种（脚还在地上时腿已经到极限）才必须"边抬边升机身"，
        其余各步都能把机身运动放在四足着地的挪重心/落定两段完成。
        """
        lifted = solve(rig, fg, new_feet, support3, prev=cur["body"], tol=0.005,
                       pitch_hi=pitch_hi)
        shifted = (solve(rig, fg, feet, support3, prev=lifted["body"], tol=0.005,
                         pitch_hi=pitch_hi) if lifted else None)
        if lifted is not None and shifted is not None and (
                abs(lifted["body"][2] - shifted["body"][2]) > 0.030
                or abs(lifted["body"][0] - shifted["body"][0]) > 0.040):
            lifted = solve(rig, fg, new_feet, support3, prev=cur["body"], tol=0.030,
                           pitch_hi=pitch_hi)
            shifted = (solve(rig, fg, feet, support3, prev=lifted["body"], tol=0.030,
                             pitch_hi=pitch_hi) if lifted else None)
        return lifted, shifted

    def do_step(leg: str, new_feet: dict, label: str, cyc: int,
                aggressive: bool = False, pitch_hi: float = PITCH_HI,
                anchor=None, win: float | None = None, slow: float = 1.0) -> bool:
        """一次单腿移动：挪重心 → 抬腿 → 落定。失败时写 notes 并返回 False。

        aggressive=False 是爬梯段的既定做法（tol=0.03，贴着上一位姿）。
        aggressive=True 只用在"平台段"（前足上平台后前移、后足最后上平台）：
        那里支撑三角形整个落在同一个水平面上，余量上限只有 70–120 mm，
        必须真取最大余量的位姿（含 8 cm 量级的横向偏重心），否则绕对角线一推就翻。
        """
        nonlocal cur
        support3 = tuple(l for l in G.LEGS if l != leg)
        ref = cur["body"] if anchor is None else anchor      # 锚点：入口段用它指定"这一步机身该到哪"
        if aggressive:
            lifted, shifted = solve_pair(new_feet, support3, pitch_hi)
            tol_settle = 0.010
        else:
            lifted = solve(rig, fg, new_feet, support3, prev=ref, tol=0.030,
                           pitch_hi=pitch_hi, win=win)
            shifted = (solve(rig, fg, feet, support3, prev=ref, tol=0.030,
                             pitch_hi=pitch_hi, win=win) if lifted else None)
            tol_settle = 0.030
        if lifted is None:
            notes.append(f"{label}（{leg}）：抬腿后无可行位姿")
            return False
        if shifted is None:
            notes.append(f"{label}（{leg}）：挪重心无可行位姿")
            return False
        for tag, cand in (("抬腿", lifted), ("挪重心", shifted)):
            if cand.get("win_hit"):
                notes.append(f"{label}（{leg}）：{tag}位姿落在搜索窗口边界")
        z_clear = max(feet[leg][2], new_feet[leg][2]) + rig.r + SWING_CLEAR
        segs.append(G.Seg(f"挪重心{leg}", SHIFT_S * slow, cur["body"], shifted["body"],
                          dict(feet), dict(feet)))
        segs.append(G.Seg(f"抬{leg}", LIFT_S * slow, shifted["body"], lifted["body"],
                          dict(feet), dict(new_feet), lift=leg, z_clear=z_clear))
        settled = solve(rig, fg, new_feet, G.LEGS, prev=lifted["body"], tol=tol_settle,
                        pitch_hi=pitch_hi, win=win)
        if settled is None:
            notes.append(f"{label}（{leg}）：落定无可行位姿")
            return False
        segs.append(G.Seg(f"落定{leg}", SETTLE_S * slow, lifted["body"], settled["body"],
                          dict(new_feet), dict(new_feet)))
        steps.append({"cycle": cyc, "leg": leg, "label": label,
                      "d_shift": [shifted["body"][2] - cur["body"][2],
                                  math.degrees(shifted["body"][3] - cur["body"][3])],
                      "d_lift": [lifted["body"][2] - shifted["body"][2],
                                 math.degrees(lifted["body"][3] - shifted["body"][3])],
                      "margin_shift": shifted["margin"], "margin_lift": lifted["margin"],
                      "margin_settle": settled["margin"],
                      "max_ext": settled["max_ext"],
                      "ext": settled["ext"], "pitch": math.degrees(settled["body"][3]),
                      "body": settled["body"], "leg_clear": settled.get("leg_clear")})
        feet.update(new_feet)
        cur = settled
        return True

    # 平地四拍（走近段 / 平台段共用）：一次只动一条腿，腿落点 = 自己原位 + 步幅，
    # 机身锚点 = 上一位姿往前 1/4 步幅（平滑前进），位姿仍由 solve 在窗口内搜：
    # 实测"直接按几何位置摆机身"会在最后一条腿抬起那拍把重心丢到支撑三角形外（−36 mm、侧翻 48°），
    # 所以必须让 solve 在 ±12 cm 窗口里挑一个余量够的位姿；俯仰压到 10° 以内，看起来是平地走路。
    def flat_walk(n_cycles: int, stride: float, z_stand: float, tag: str,
                  pitch_hi: float = 10.0, slow: float = 1.15) -> bool:
        nonlocal cur
        for c in range(n_cycles):
            for leg in G.LEGS:
                anchor = (cur["body"][0] + stride / 4.0, G.BODY_Y, z_stand, 0.0)
                nf = dict(feet)
                nf[leg] = (feet[leg][0] + stride, feet[leg][1], feet[leg][2])
                if not do_step(leg, nf, f"{tag}{c+1}", c + 1, pitch_hi=pitch_hi,
                               anchor=anchor, win=0.12, slow=slow):
                    return False
        return True

    # ---- 完整作业流的首段：平地走近（从后方站姿走到入口起点）----
    if approach:
        if not flat_walk(APPROACH_CYCLES, APPROACH_STRIDE, GROUND_Z, "走近"):
            return segs, steps, notes, feet
        gfeet, gbody = ground_stance(rig)
        dev = max(abs(feet[leg][0] - gfeet[leg][0]) for leg in G.LEGS)
        segs.append(G.Seg("走近后对齐站姿", 1.5, cur["body"], gbody, dict(feet), dict(feet)))
        cur = {"body": gbody}
        notes.append(f"走近完成：机身 x {gbody[0]:+.3f}，四足落点与入口站姿偏差 {dev*1000:.0f} mm")
        if only_approach:
            return segs, steps, notes, feet

    # ---- 完整作业流的末段：登顶站直 + 平台前行 ----
    if platform:
        if not flat_walk(PLAT_CYCLES, PLAT_STRIDE, LEVEL_TOP[TOP_LEVEL] + PLAT_STAND_OFF, "平台", pitch_hi=PLAT_PITCH_HI):
            return segs, steps, notes, feet
        segs.append(G.Seg("站定", 0.8, cur["body"], cur["body"], dict(feet), dict(feet)))
        notes.append(f"平台行走完成：机身 x {cur['body'][0]:+.3f}（平台后沿 "
                     f"{edge_x(TOP_LEVEL):.3f}，顶 {LEVEL_TOP[TOP_LEVEL]:.3f}）")
        if only_platform:
            return segs, steps, notes, feet

    # 平地起步：先走 7 步进入标准状态 S0（前两足先上第 1 级，再把后足挪到位）。
    # 这一段把俯仰压到 0°：不压的话规划器会为了"多要点余量"把机身一路抬到 30°，
    # 看起来是原地越抬越高，不是走路。
    if entry:
        spec = entry_spec(rig)
        z0, z1 = gbody[2], 0.680
        for i, (leg, target, label) in enumerate(spec, start=1):
            t = i / len(spec)
            # 锚点 = "抬起这条腿以后剩下三足"的三角形内切圆心（重心余量最大的那个点），
            # h 按 0.55→0.68 线性升。手工试过线性推 x：FR 抬起时重心被留在支撑线外侧，
            # 余量从 131 mm 掉到 45 mm；按内切圆心走才对。
            ic = G.incenter3([(feet[l][0], feet[l][1]) for l in G.LEGS if l != leg])
            anchor = (float(ic[0]), float(ic[1]), z0 + (z1 - z0) * t, 0.0)
            nf = dict(feet)
            nf[leg] = target
            # 入口段比爬梯段慢 1.4 倍：机身低、腿折着，PD 跟踪误差比爬梯时大（实测下沉 20 mm vs 12 mm）
            if not do_step(leg, nf, label, 0, pitch_hi=18.0, anchor=anchor, win=0.06, slow=1.4):
                return segs, steps, notes, feet
        # 进入 S0 后先把机身对齐到"验证过的那个位姿"，再站定：
        # 入口最后一段是搜出来的 (0.367, 0.660, 9°)，与 S0 的 (0.360, 0.680, 3°) 差 2 cm/6°，
        # 这个残差会在爬梯段被放大（实测第 16 步滑下台阶、RL 掉到地面，峰值力矩 141%）。
        s0_body = (0.360, G.BODY_Y, 0.680, math.radians(3.0))
        segs.append(G.Seg("进入后对齐S0", 3.0, cur["body"], s0_body, dict(feet), dict(feet)))
        cur = {"body": s0_body}
        # 自检：走完这 7 步，四足落点必须与 S0 逐足一致（否则后面的爬梯计划前提不成立）
        dev = max(abs(feet[leg][0] - foot_target(rig, leg, levels[leg])[0]) for leg in G.LEGS)
        notes.append(f"进入标准状态：四足落点与 S0 的最大偏差 {dev*1000:.1f} mm"
                     + ("（已对齐，后续直接接爬梯计划）" if dev < 0.005 else "（偏差偏大，检查）"))
        if only_entry:
            return segs, steps, notes, feet

    adjusted = False
    for c in range(cycles):
        for leg, lv_from, lv_to in steps_of_cycle(c, base):
            if lv_to > TOP_LEVEL:
                notes.append(f"第 {c+1} 循环 {leg}：{lv_from}→{lv_to} 越过平台，跳过")
                continue
            if levels[leg] != lv_from:
                notes.append(f"第 {c+1} 循环 {leg}：期望从 {lv_from} 级起，实际在 {levels[leg]} 级，跳过")
                continue
            new_feet = dict(feet)
            new_feet[leg] = foot_target(rig, leg, lv_to)
            # 后足踏上平台那两步是整条轨迹余量最低的（三足全在同一水平面），用平台段的激进挑选
            rear_onto_plat = lv_to == TOP_LEVEL and leg in ("RL", "RR")
            if not do_step(leg, new_feet, f"{lv_from}→{lv_to}", c + 1,
                           aggressive=rear_onto_plat):
                return segs, steps, notes, feet
            levels[leg] = lv_to
            # 两只前足都上了平台、两只后足都在第 3 级 —— 此时先让前足在平地上各迈一步，
            # 把"前两足 + 后一足"的三角形撑开，再让后足上平台（用户提的办法）
            if (not adjusted and levels["FL"] == TOP_LEVEL and levels["FR"] == TOP_LEVEL
                    and levels["RL"] == TOP_LEVEL - 1 and levels["RR"] == TOP_LEVEL - 1):
                adjusted = True
                for l2 in ("FL", "FR"):
                    nf = dict(feet)
                    x, y, z = feet[l2]
                    nf[l2] = (edge_x(TOP_LEVEL) + PLAT_ADJ, y, z)
                    if not do_step(l2, nf, "平台上前移", c + 1, aggressive=True):
                        return segs, steps, notes, feet
    segs.append(G.Seg("收尾", 0.5, cur["body"], cur["body"], dict(feet), dict(feet)))
    return segs, steps, notes, feet


PLAN_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait_plan.json"
CLIMB_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait_plan_climb.json"
ENTRY_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait_plan_entry.json"
APPROACH_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait_plan_approach.json"
PLATFORM_JSON = ROOT / "experiments" / "b2" / "b2_follow_gait_plan_platform.json"
# 规划参数指纹：改了任何一个规划参数，爬梯段缓存自动失效（否则会悄悄复用手改参数之前的计划）
PLAN_FINGERPRINT = (PITCH_HI, EDGE, PRELOAD, SWING_CLEAR, EXT_CAP, WIN_X, WIN_Z,
                    SHIFT_S, LIFT_S, SETTLE_S, PLAT_FRONT, PLAT_REAR, PLAT_ADJ,
                    GROUND_X, GROUND_Z, GROUND_FRONT, GROUND_REAR)


def segs_dump(segs) -> list[dict]:
    return [{"name": s.name, "dur": s.dur, "body_a": list(s.body_a), "body_b": list(s.body_b),
             "feet_a": {k: list(v) for k, v in s.feet_a.items()},
             "feet_b": {k: list(v) for k, v in s.feet_b.items()},
             "lift": s.lift, "z_clear": s.z_clear} for s in segs]


def segs_load(items) -> list:
    return [G.Seg(s["name"], s["dur"], tuple(s["body_a"]), tuple(s["body_b"]),
                  {k: tuple(v) for k, v in s["feet_a"].items()},
                  {k: tuple(v) for k, v in s["feet_b"].items()},
                  lift=s["lift"], z_clear=s["z_clear"]) for s in items]


def save_plan(segs, steps, cycles: int, base: int = 0, entry: bool = False) -> None:
    PLAN_JSON.write_text(json.dumps({"cycles": cycles, "base": base, "entry": entry,
                                     "params": list(PLAN_FINGERPRINT),
                                     "segments": segs_dump(segs)}, ensure_ascii=False),
                         encoding="utf-8")


def load_plan(cycles: int, base: int = 0, entry: bool = False):
    """读缓存的计划；没有、循环数/基准/是否平地起步对不上、或规划参数改过就返回 None。"""
    if not PLAN_JSON.exists():
        return None
    data = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    if (data.get("cycles") != cycles or data.get("base", 0) != base
            or bool(data.get("entry", False)) != entry
            or data.get("params") != list(PLAN_FINGERPRINT)):
        return None
    return segs_load(data["segments"])


def entry_part(rig: G.Rig, fg, cycles: int, base: int = 0, segs=None):
    """入口段（平地起步那 7 段）：能从完整计划里截出来就别重算 —— 重算要十几秒，
    而且入口计划和爬梯计划里的入口部分本来就是同一段代码生成的。"""
    for i, s in enumerate(segs or []):
        if s.name == "进入后对齐S0":
            return list(segs[:i + 1])
    return build(rig, fg, cycles, base, entry=True, only_entry=True)[0]


def state_key(st: dict) -> list:
    """实测状态的缓存键：机身 4 个量 + 四足各 3 个坐标，取到毫米/毫弧度。"""
    return [round(float(v), 3) for v in st["body"]] + [
        round(float(c), 3) for leg in G.LEGS for c in st["feet"][leg]]


def cached_plan(path, rig: G.Rig, fg, cycles: int, base: int, st: dict, label: str,
                build_kw: dict, use_cache: bool = True):
    """按实测状态规划某一段，并把结果按「实测状态 + 规划参数指纹」缓存下来。

    为什么要缓存：爬梯段（14 步的细网格位姿搜索）一次要 6 分钟，以前是入口走完才现场算，
    于是 --view 里机器狗在窗前站着不动好几分钟 —— 看着像卡死，其实在规划。
    现在规划全部放在开窗口之前，而且同样的实测状态第二次跑就是读缓存、几乎不等。
    """
    key = state_key(st)
    hit = None
    if use_cache and path.exists():
        try:
            hit = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            hit = None
    if (hit and hit.get("cycles") == cycles and hit.get("base") == base
            and hit.get("params") == list(PLAN_FINGERPRINT) and hit.get("state") == key):
        print(f"  {label}读缓存（{len(hit['segments'])} 段，同一实测状态，不重算）", flush=True)
        return segs_load(hit["segments"]), [], [f"（{label}读自缓存）"]
    print(f"  按实测位姿重规划{label}……（跑完会缓存，下次就不用等）", flush=True)
    t0 = time.perf_counter()
    segs, steps, notes, _ = build(rig, fg, cycles, base, state=st, **build_kw)
    print(f"  {label}规划完成：{len(segs) if segs else 0} 段，用时 {time.perf_counter() - t0:.0f} s",
          flush=True)
    if segs is not None:
        path.write_text(json.dumps({"cycles": cycles, "base": base,
                                    "params": list(PLAN_FINGERPRINT), "state": key,
                                    "segments": segs_dump(segs)}, ensure_ascii=False),
                        encoding="utf-8")
    return segs, steps, notes


def plan_climb(rig: G.Rig, fg, cycles: int, base: int, st: dict, use_cache: bool = True):
    """爬梯段（14 步）：按实测状态规划 + 缓存。"""
    return cached_plan(CLIMB_JSON, rig, fg, cycles, base, st, "爬梯段", {}, use_cache)


def plan_entry(rig: G.Rig, fg, cycles: int, base: int, st: dict, use_cache: bool = True):
    """入口段（6 步）：按实测状态规划 + 缓存（走过来的落点和名义站姿差几厘米时用）。"""
    return cached_plan(ENTRY_JSON, rig, fg, cycles, base, st, "入口段",
                       {"entry": True, "only_entry": True}, use_cache)


def level_of(rig: G.Rig, z: float, x: float) -> str:
    """把足心位置映射回"在哪一级"（容差 3 cm，否则返回 ?）。
    地面上的级别只能靠 x 分辨（level 0 与负级的高度一样），所以按落点规则反算。"""
    if abs(z - (rig.r + PRELOAD)) < 0.030:
        return str(int(round((x - (TREAD_REAR0 - 0.15)) / TREAD_DEPTH)))
    for lv, top in sorted(LEVEL_TOP.items()):
        if lv == 0:
            continue
        if abs(z - (top + rig.r + PRELOAD)) < 0.030:
            return str(lv)
    return "?"


class ContactLog:
    """按段记录：支撑腿最小法向力、非足端碰撞体撞梯级的接触。

    为什么要记：第 2 循环抬 FR 失稳时力矩只有 68%，说明不是关节不够 ——
    真正的证据只能是 contact 列表（哪两个 geom、多少牛、哪只脚卸载）。
    """

    def __init__(self, rig: G.Rig, fg):
        self.rig = rig
        self.fg = fg
        self.g2leg = {g: leg for leg, g in fg.items()}
        self.feet = set(fg.values())
        self.plates = set(rig.plate_gids)
        m = rig.model
        self.robot = {g for g in range(m.ngeom)
                      if int(m.body_rootid[int(m.geom_bodyid[g])]) == 1}
        self.by_id: dict[int, dict] = {}     # id(seg) → 该段记录（同名段跨循环不会互相覆盖）
        self.trace: list[dict] = []          # 每个控制步一行，失败后拿来看时间历程
        self.cur = None
        self.last = None

    def __call__(self, seg, u) -> None:
        if self.cur is None or self.cur["seg"] is not seg:
            self.cur = {"seg": seg, "foot_min": {leg: 1e9 for leg in G.LEGS}, "foreign": {}}
            self.by_id[id(seg)] = self.cur
            self.last = self.cur
        m, data = self.rig.model, self.rig.data
        f = {leg: 0.0 for leg in G.LEGS}
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            fvec = np.zeros(6)
            mujoco.mj_contactForce(m, data, i, fvec)
            n = abs(float(fvec[0]))
            for g in (g1, g2):
                if g in self.g2leg:
                    f[self.g2leg[g]] += n
            if not ((g1 in self.plates and g2 in self.robot) or
                    (g2 in self.plates and g1 in self.robot)):
                continue
            for g, other in ((g1, g2), (g2, g1)):
                if g in self.robot and g not in self.feet and n > 5.0:
                    # geom 名可能为空（MJCF 里很多碰撞体没取名），补上所属 body 才看得懂是哪个部件
                    def nm(x: int) -> str:
                        gname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, x)
                        bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                                  int(m.geom_bodyid[x]))
                        return gname or f"{bname or '?'}#{x}"
                    key = (nm(g), nm(other))
                    self.cur["foreign"][key] = max(self.cur["foreign"].get(key, 0.0), n)
        for leg in G.LEGS:
            self.cur["foot_min"][leg] = min(self.cur["foot_min"][leg], f[leg])
        self.trace.append({"seg": seg.name, "u": float(u),
                           "x": float(data.qpos[0]), "z": float(data.qpos[2]),
                           "pitch": math.degrees(G.pitch_of(data)), "tilt": G.tilt_of(data),
                           "fx": {leg: float(data.geom_xpos[g][0]) for leg, g in self.fg.items()},
                           "fz": {leg: float(data.geom_xpos[g][2]) for leg, g in self.fg.items()},
                           "f": f, "ncon": int(data.ncon)})

    def of(self, seg) -> dict | None:
        return self.by_id.get(id(seg))

    def dump(self, path, seg_name: str, every: int = 5) -> None:
        """把失败那一段的时间历程写出来：机身 x/z/pitch/tilt + 各足高度/受力 + 各足 x 区间。"""
        rows = [t for t in self.trace if t["seg"] == seg_name]
        out = [f"# {seg_name}：每 {every} 个控制步一行（50 Hz）",
               f"{'u':>5}{'body x':>8}{'z':>7}{'pitch':>7}{'tilt':>6}"
               + "".join(f"{leg + ' z':>8}{leg + ' F':>7}" for leg in G.LEGS) + f"{'ncon':>6}"]
        for t in rows[::every] + rows[-1:]:
            out.append(f"{t['u']:5.2f}{t['x']:8.3f}{t['z']:7.3f}{t['pitch']:7.1f}{t['tilt']:6.1f}"
                       + "".join(f"{t['fz'][leg]:8.3f}{t['f'][leg]:7.0f}" for leg in G.LEGS)
                       + f"{t['ncon']:6d}")
        out.append("")
        out.append("# 各足在这一段里的 x 范围 / z 范围 / 最小法向力（滑出踏面就会体现成 x 倒退 + 力到 0）")
        for leg in G.LEGS:
            xs = [t["fx"][leg] for t in rows]
            zs = [t["fz"][leg] for t in rows]
            fs = [t["f"][leg] for t in rows]
            out.append(f"  {leg}: x {min(xs):.3f}..{max(xs):.3f}  z {min(zs):.3f}..{max(zs):.3f}"
                       f"  最小力 {min(fs):5.0f} N")
        path.write_text("\n".join(out) + "\n", encoding="utf-8")


def contact_map(rig: G.Rig, fg) -> dict:
    """末态：每个足端压在哪个 geom 上、法向力多少（这才是"站在平台上"的证据）。"""
    m, data = rig.model, rig.data
    out: dict[str, list] = {}
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        fvec = np.zeros(6)
        mujoco.mj_contactForce(m, data, i, fvec)
        n = abs(float(fvec[0]))
        for a, b in ((g1, g2), (g2, g1)):
            leg = next((lg for lg, g in fg.items() if g == a), None)
            if leg is not None:
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, b) or f"geom{b}"
                out.setdefault(leg, []).append((name, n))
    return out


def plan_table(steps, notes, total: int, cycles: int) -> list[str]:
    """规划表：每步机身位姿 + 三个余量（挪重心/抬腿/落定）+ 伸展，外加所有跳过的说明。"""
    out = [f"  规划：{len(steps)}/{total} 步可解（{cycles} 个循环）"]
    out.append(f"    {'循环':<4}{'腿':<5}{'级':<7}{'机身 x':>7}{'z':>7}{'pitch':>7}"
               f"{'余量 挪重心/抬腿/落定':>21}{'伸展':>6}{'挪重心段':>13}{'抬腿段':>13}")
    for st in steps:
        b = st["body"]
        out.append(f"    {st['cycle']:<4}{st['leg']:<5}{st['label']:<7}"
                   f"{b[0]:7.3f}{b[2]:7.3f}{st['pitch']:7.1f}"
                   f"{st['margin_shift']*1000:7.0f}/{st['margin_lift']*1000:.0f}/"
                   f"{st['margin_settle']*1000:.0f}mm"
                   f"{st['max_ext']*100:5.0f}%"
                   f"{st['d_shift'][0]*100:+7.1f}cm/{st['d_shift'][1]:+5.1f}°"
                   f"{st['d_lift'][0]*100:+7.1f}cm/{st['d_lift'][1]:+5.1f}°")
    for n in notes:
        out.append(f"    · {n}")
    return out


def run_two_phase(rig: G.Rig, fg, cycles: int, base: int, servo: float,
                  hook=None, sync=None, plans=None, segs=None, steps=None,
                  rebuild: bool = False):
    """两段式跑法：入口段 → 读实测位姿 → 按实测重规划爬梯段 → 不重置地接着跑。

    返回 (合并结果, 全部步骤, 全部说明, 实际跑的段, 入口步数)；失败时返回已完成的那部分。
    这是 20/20 能走通的关键：入口结束时足端与标准状态目标差 46 mm，
    不分段的话爬梯段每一拍的位姿都和现实不自洽（实测只有 17/20）。

    plans=(入口段, 爬梯段, 规划用的实测状态)：两段都预先规划好了（--view 在开窗口前先做），
    这时不再现场规划 —— 中途规划会让窗口停住好几分钟。
    segs=完整计划：入口段直接从里面截，省一次重算。
    """
    n_entry = len(entry_spec(rig))
    e_steps = list(steps[:n_entry]) if steps else []
    e_notes = ["（入口段读自缓存/完整计划）"] if not e_steps else []
    c_steps, c_notes = [], []
    if plans is not None:
        e_segs, c_segs, st = plans
    else:
        e_segs = entry_part(rig, fg, cycles, base, segs)
    res1 = G.run_plan(rig, e_segs, fg, hook=hook, body_servo=servo, sync=sync)
    if res1["fell"]:
        return res1, e_steps, e_notes, e_segs, n_entry
    if plans is not None:
        # 预演与实播是同一条确定性轨迹，理论上一致；差得多说明计划的前提不成立，按实测重规划
        st2 = read_state(rig, fg)
        a = list(st["body"]) + [c for leg in G.LEGS for c in st["feet"][leg]]
        b = list(st2["body"]) + [c for leg in G.LEGS for c in st2["feet"][leg]]
        dev = max(abs(x - y) for x, y in zip(a, b))
        if dev > 0.02:
            print(f"  ⚠ 入口落点与预演差 {dev * 1000:.0f} mm，按实测重新规划爬梯段", flush=True)
            c_segs, c_steps, c_notes = plan_climb(rig, fg, cycles, base, st2, use_cache=not rebuild)
    else:
        c_segs, c_steps, c_notes = plan_climb(rig, fg, cycles, base, read_state(rig, fg), use_cache=not rebuild)
    if c_segs is None:
        print("按实测重规划爬梯段失败：", c_notes)
        return res1, e_steps, e_notes + c_notes, e_segs, n_entry
    res2 = G.run_plan(rig, c_segs, fg, hook=hook, body_servo=servo, sync=sync, reset=False)
    return (merge_res(res1, res2), e_steps + c_steps, e_notes + c_notes,
            e_segs + c_segs, n_entry)


# ================================================================ 完整作业流（--demo）
# 平地起步 → 自主走到楼梯口 → 按实测重规划入场 → 20 步爬梯 → 登顶站直 → 平台上前走几步。
# 四段都走 build()：走近段 / 平台段是"平地上的四拍走"（腿落点 = 自己原位 + 步幅，
# 机身锚点取剩下三足的内切圆心、俯仰压到 10° 以内）；入口段 / 爬梯段仍是细网格位姿搜索 +
# 按实测位姿重规划 + 缓存。参数见文件开头「完整作业流的参数」。
OUT_COURSE = ROOT / "experiments" / "b2" / "b2_full_course.txt"
OUT_COURSE_JSON = ROOT / "experiments" / "b2" / "b2_full_course.json"


def approach_plan(rig: G.Rig, fg, use_cache: bool = True):
    """前置阶段：从楼梯口后方的正常站姿，走到入口起点（名义平地站姿，机身 x = GROUND_X）。

    起点是固定的，所以「实测状态」就是名义起点站姿；缓存键因此也是固定的（改了参数会自动失效）。
    """
    feet, body = approach_stance(rig)
    st0 = {"body": body, "feet": {leg: tuple(v) for leg, v in feet.items()}}
    return cached_plan(APPROACH_JSON, rig, fg, APPROACH_CYCLES, 0, st0, "走近段",
                       {"approach": True, "only_approach": True}, use_cache)


def platform_plan(rig: G.Rig, fg, st: dict, use_cache: bool = True):
    """后置阶段：登顶后先站直（恢复平地站姿），再用同一套四拍在平台上往前走。

    起点是登顶后的实测状态，所以缓存键跟着实测状态走（和入口段/爬梯段同一个机制）。
    """
    return cached_plan(PLATFORM_JSON, rig, fg, PLAT_CYCLES, 0, st, "平台段",
                       {"platform": True, "only_platform": True}, use_cache)

def demo_plans(rig: G.Rig, fg, cycles: int, base: int, use_cache: bool = True):
    """把整条作业流的四段计划预先算好（不开窗口、也不播画面）。

    为什么要先"空跑"一遍：入口段和爬梯段都要按**实测**位姿重规划，而实测位姿只有真的跑过
    才有。这里先在无窗口状态下把整条流程跑一遍拿到各阶段切换时的真实状态，规划好四段；
    播画面时从起点重跑，物理是确定性的，所以这些计划照样成立（每段切换处还会核对偏差）。
    顺序：走近(解析) → 实测量 → 入口(搜) → 实测量 → 爬梯(搜+缓存) → 实测量 → 站直+平台走(解析)。
    """
    servo = 1.2
    a_segs, a_steps, a_notes = approach_plan(rig, fg)
    G.run_plan(rig, a_segs, fg, body_servo=servo)
    st1 = read_state(rig, fg)
    e_segs, e_steps, e_notes = plan_entry(rig, fg, cycles, base, st1, use_cache)
    if e_segs is None:
        return None, ["入口段按实测重规划失败：" + str(e_notes)], [st1]
    G.run_plan(rig, e_segs, fg, body_servo=servo, reset=False)
    st2 = read_state(rig, fg)
    c_segs, c_steps, c_notes = plan_climb(rig, fg, cycles, base, st2, use_cache)
    if c_segs is None:
        return None, ["爬梯段按实测重规划失败：" + str(c_notes)], [st1, st2]
    G.run_plan(rig, c_segs, fg, body_servo=servo, reset=False)
    st3 = read_state(rig, fg)
    p_segs, p_steps, p_notes = platform_plan(rig, fg, st3)
    phases = [("平地走近", a_segs, a_steps, a_notes),
              ("按实测重规划入口", e_segs, e_steps, e_notes),
              ("爬梯（20 步核心）", c_segs, c_steps, c_notes),
              ("登顶站直 + 平台前行", p_segs, p_steps, p_notes)]
    return phases, [st1, st2, st3], None


def run_demo_flow(rig: G.Rig, fg, phases, servo: float, hook=None, sync=None,
                  expect=None):
    """按顺序跑四段（第一段重置到起点，后面的段不重置接着跑），返回合并结果。

    expect：规划时用到的三个实测状态；每段跑完都核对一次偏差，>2 cm 就明说
    （播画面时不再现场重规划，避免窗口卡住，但会提示哪一段的前提不成立了）。
    """
    res = None
    segs_all: list = []
    steps_all: list = []
    notes_all: list = []
    state_after: list = []
    for i, (name, segs, steps, notes) in enumerate(phases):
        r = G.run_plan(rig, segs, fg, hook=hook, body_servo=servo, sync=sync, reset=(i == 0))
        state_after.append(read_state(rig, fg))
        res = r if res is None else merge_res(res, r)
        segs_all += list(segs)
        steps_all += list(steps)
        notes_all += list(notes)
        print(f"  {name}：{len(r['beats'])} 步，机身 x {r['x']:+.3f} z {r['z']:.3f}"
              + ("（失稳：" + str(r["failed_at"]) + "）" if r["fell"] else ""), flush=True)
        if r["fell"]:
            break
    if expect:
        for k, (a, b) in enumerate(zip(expect, state_after)):
            if k >= len(state_after) - 1:
                break
            dev = max(abs(x - y) for x, y in zip(state_key(a), state_key(b)))
            if dev > 0.02:
                print(f"  ⚠ 第 {k+1} 段结束时与预演差 {dev*1000:.0f} mm（该段之后的计划前提已变）",
                      flush=True)
    return res, steps_all, notes_all, segs_all


def view_demo(rig: G.Rig, fg, phases, cycles: int, base: int) -> int:
    """3D 播放完整作业流：先规划（不开窗口），再一路不停地播完四段。"""
    import mujoco.viewer

    hz = 25.0 if "--slow" in sys.argv[1:] else 50.0
    model, data = rig.model, rig.data
    counts = [len([x for x in segs if x.lift]) for _, segs, _, _ in phases]
    n_lift = sum(counts)
    print(f"  播放（{hz/50:.1f} 倍速）：{n_lift} 步 = 走近 {counts[0]} + 入口 {counts[1]} + "
          f"爬梯 {counts[2]} + 平台 {counts[3]}，四段之间不停", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = (0.55, G.BODY_Y, 0.40)
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 72.0
        viewer.cam.elevation = -8.0

        def follow():
            viewer.cam.lookat[:] = (float(data.qpos[0]) + 0.05, G.BODY_Y,
                                    float(data.qpos[2]) - 0.12)

        sync = G.pacer(viewer, on_frame=follow, hz=hz)
        res, _, _, _ = run_demo_flow(rig, fg, phases, 1.2, sync=sync)
        print(f"  结果：走通 {len(res['beats'])}/{n_lift} 步，"
              f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '全程未失稳'}"
              f"，末态机身 x {res['x']:+.3f} z {res['z']:.3f}"
              + ("，四足都在平台上" if not res["fell"] else ""), flush=True)
        G.hold_open(viewer)
    return 0


def demo_main(rig: G.Rig, fg, cycles: int, base: int, argv) -> int:
    """--demo：完整作业流。数据模式写 b2_full_course.txt/.json；带 --view 则只播放。"""
    phases, states, err = demo_plans(rig, fg, cycles, base)
    if phases is None:
        print(err)
        return 1
    if "--view" in argv:
        return view_demo(rig, fg, phases, cycles, base)
    servo = float(argv[argv.index("--servo") + 1]) if "--servo" in argv else 1.2
    log = ContactLog(rig, fg)
    res, steps, notes, run_segs = run_demo_flow(rig, fg, phases, servo, hook=log, expect=states)
    total = len([x for x in run_segs if x.lift])
    counts = [len([x for x in s if x.lift]) for _, s, _, _ in phases]
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 108)
    say("B2 完整作业流：平地起步 → 自主走到楼梯口 → 按实测重规划入场 → 20 步爬梯 → 登顶站直 → 平台前行")
    say("=" * 108)
    for name, segs, steps, notes_ in phases:
        lift = [x for x in segs if x.lift]
        b0 = segs[0].body_a
        b1 = segs[-1].body_b
        say(f"  {name:<18} {len(lift):>2} 步   机身 x {b0[0]:+.3f} → {b1[0]:+.3f}   "
            f"z {b0[2]:.3f} → {b1[2]:.3f}   pitch {math.degrees(b0[3]):+.1f}° → {math.degrees(b1[3]):+.1f}°")
    say()
    for n in notes:
        say(f"  · {n}")
    say()
    say(f"  实测（PD 800/20 + 机身位姿伺服 k={servo}；每步 挪重心 {SHIFT_S} s → 抬腿 {LIFT_S} s → "
        f"落定 {SETTLE_S} s；抬脚离目标 {SWING_CLEAR*100:.0f} cm）：")
    say(f"    {'步骤':<8}{'抬脚高度':>9}{'机身下沉':>9}{'打滑':>8}{'力矩峰值':>9}{'倾角':>6}   四足所在级（实测）")
    lift_segs = [x for x in run_segs if x.lift]
    for k, b in enumerate(res["beats"]):
        lvs = " ".join(f"{l}{level_of(rig, b['feet_z'][l], b['feet_x'][l])}" for l in G.LEGS)
        say(f"    {b['name']:<8}{b.get('lift', 0)*1000:8.0f}mm{b['sink']*1000:+8.0f}mm"
            f"{b['slip_mm']:7.1f}mm{b.get('peak', 0)*100:8.0f}%{b['tilt']:5.0f}°   {lvs}")
    say(f"    走通 {len(res['beats'])}/{total} 步；"
        f"{'在【' + str(res['failed_at']) + '】失去稳定' if res['fell'] else '全程未失稳'}")
    say(f"    峰值：{res['peak_act']} 占额 {res['peak_tau']*100:.0f}%；最大倾角 {res['max_tilt']:.0f}°")
    final = None
    if not res["fell"]:
        body_f = tuple(run_segs[-1].body_b)
        feet_f = dict(run_segs[-1].feet_b)
        h = G.hold(rig, fg, feet_f, body_f, 2.0)
        cmap = contact_map(rig, fg)
        on_plat = {leg: any(nm == "platform" for nm, _ in cmap.get(leg, [])) for leg in G.LEGS}
        final = {"hold": h, "contact": {k: [[nm, round(n, 1)] for nm, n in v] for k, v in cmap.items()},
                 "on_platform": on_plat}
        say()
        say("  末态核验（原地静置 2 s，读 contact 里真实生效的 geom 与法向力）：")
        say(f"    机身 ({body_f[0]:.3f}, {body_f[1]:.3f}, {body_f[2]:.3f})  pitch {math.degrees(body_f[3]):+.1f}°"
            f"（平台顶 {LEVEL_TOP[TOP_LEVEL]:.3f} + 站高 {PLAT_STAND_OFF:.3f}；比平地站高 {GROUND_Z:.3f} 低 "
                f"{(GROUND_Z - PLAT_STAND_OFF)*100:.0f} cm —— 为了在漂移后的站姿上还有横向修正余量）")
        if h is not None:
            say(f"    静置：Δx {h['dx']*1000:+.0f} / Δy {h['dy']*1000:+.0f} / Δz {h['dz']*1000:+.0f} mm，"
                f"倾角 {h['tilt']:.1f}°，打滑 {h['slip_mm']:.1f} mm，力矩峰值 {h['peak_tau']*100:.0f}%，"
                f"{'未倾覆' if h['tilt'] < 20 else '有倾覆风险'}")
        say("    四足接触：" + "  ".join(
            f"{leg}→{'/'.join(f'{nm} {n:.0f}N' for nm, n in cmap.get(leg, [])) or '无'}" for leg in G.LEGS))
        say("    四足都在平台（platform geom）上：" + ("是" if all(on_plat.values()) else "否"))
    say()
    ok = (not res["fell"]) and len(res["beats"]) == total
    n_plat = total - 28
    say(f"  结论：{'四段全通、' if ok else ''}{total} 步走通"
        f"（走近 {counts[0]} + 入口 {counts[1]} + 爬梯 {counts[2]} + 平台 {n_plat}），"
        f"机身 x {res['x']:+.3f}、z {res['z']:.3f}；峰值力矩 {res['peak_tau']*100:.0f}%（{res['peak_act']}）")
    text = "\n".join(lines)
    OUT_COURSE.write_text(text + "\n", encoding="utf-8")
    OUT_COURSE_JSON.write_text(json.dumps(
        {"phases": [{"name": n, "steps": len([x for x in s if x.lift]),
                     "body_a": list(s[0].body_a), "body_b": list(s[-1].body_b)} for n, s, _, _ in phases],
         "steps": steps, "notes": notes, "final": final,
         "result": {k: v for k, v in res.items() if k != "beats"}, "beats": res["beats"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print("WROTE", OUT_COURSE)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    entry = "--no-entry" not in argv            # 默认从平地正常站姿起步（多 7 步进入标准状态）
    base = 0
    cycles = int(argv[argv.index("--cycles") + 1]) if "--cycles" in argv else 4
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    if "--demo" in argv:
        # 完整作业流：走近 → 入口 → 爬梯 → 平台前行（自己规划、自己的报告，不复用下面的名义计划）
        return demo_main(rig, fg, cycles, base, argv)
    cached = None if ("--rebuild" in argv or "--plan" in argv) else load_plan(cycles, base, entry)
    if cached is not None:
        print(f"读缓存计划（{len(cached)} 段）；要重算加 --rebuild")
        segs, steps, notes = cached, [], ["（读自缓存，规划表见上一次输出）"]
    else:
        segs, steps, notes, _ = build(rig, fg, cycles, base, entry=entry)
        if segs is not None:
            save_plan(segs, steps, cycles, base, entry)
    if segs is None:
        print("无法起步：", notes)
        return 1
    total = len([x for x in segs if x.lift])
    if "--view" in argv:
        return view(rig, fg, segs, steps, cycles, base, entry, "--one-phase" in argv)
    if "--plan" in argv:        # 只规划（几十秒），先确认每步都可解再花几分钟跑动力学
        print("\n".join(plan_table(steps, notes, total, cycles)))
        return 0

    servo = float(argv[argv.index("--servo") + 1]) if "--servo" in argv else 1.2
    log = ContactLog(rig, fg)
    res = None
    run_segs, n_entry = segs, 0
    if entry and "--one-phase" not in argv:
        # 两段式：入口跑完 → 读实测位姿 → 按实测重规划爬梯段 → 不重置接着跑（--one-phase 做对照）
        res, steps, notes, run_segs, n_entry = run_two_phase(
            rig, fg, cycles, base, servo, hook=log, segs=segs, steps=steps,
            rebuild="--rebuild" in argv)
        if res is None:
            return 1
    if res is None:
        res = G.run_plan(rig, segs, fg, hook=log, body_servo=servo)
    res0 = None
    if "--quick" not in argv:      # 纯开环对照（同样一条轨迹），翻倍耗时，赶时间时关掉
        rig0 = G.load_rig("b2")
        res0 = G.run_plan(rig0, segs, G.foot_gids(rig0), body_servo=0.0)
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    done = len(res["beats"])
    say("=" * 104)
    say("B2「交替跟进」步态：" + ("平地正常站姿起步 → 7 步进入标准状态 → " if entry else "")
        + "一次只动一条腿，一路爬到平台（第 4 级，顶 0.788 m）")
    say("=" * 104)
    lv = start_levels(base)

    def lname(v: int) -> str:
        return "地面" if v == 0 else (f"地面{v}" if v < 0 else f"第{v}级")

    if entry:
        say(f"  起步：平地站姿 —— 四足都在地面（前足 x {GROUND_X + GROUND_FRONT:+.3f}、"
            f"后足 x {GROUND_X + GROUND_REAR:+.3f}），机身 ({GROUND_X:+.3f}, {GROUND_Z:.3f}) pitch 0°；"
            f"楼梯鼻端 x {TREAD_REAR0:+.3f}")
        say("  进入（7 步）：FR 上第1级 / FL 上第1级 / FL 上第2级 / RL、RR 平地前移 / RR 上第1级 / RL 到位")
        say("  进入后 = 标准状态 " + "  ".join(f"{l}{lname(lv[l])}" for l in G.LEGS)
            + "（就是之前验证过的爬梯起点），后面是同一套 14 步")
    else:
        say("  起步：" + "  ".join(f"{l} {lname(lv[l])}" for l in G.LEGS))
    say("  每循环四步：FR +1（与 FL 汇合）/ RL +1（与 RR 汇合）/ FL +1 / RR +1（与 FR 汇合）")
    say()
    if steps:
        for line in plan_table(steps, notes, total, cycles):
            say(line)
    else:
        say(f"  规划：{total} 步计划读自缓存（规划表见上一次输出）；要重排加 --rebuild")
        for n in notes:
            say(f"    · {n}")
    say()
    say(f"  实测（PD 800/20 + 机身位姿伺服 k={servo}，每步：挪重心 {SHIFT_S} s → "
        f"抬腿 {LIFT_S} s → 落定 {SETTLE_S} s；抬脚离目标 {SWING_CLEAR*100:.0f} cm）：")
    say(f"    {'步骤':<8}{'抬脚高度':>9}{'机身下沉':>9}{'打滑':>8}{'力矩峰值':>9}{'倾角':>6}"
        f"{'支撑腿最小力':>13}   {'四足所在级（实测）'}")
    lift_segs = [x for x in run_segs if x.lift]  # beats 与抬腿段一一对应（两段式时用实际跑的段）
    for k, b in enumerate(res["beats"]):
        lvs = " ".join(f"{l}{level_of(rig, b['feet_z'][l], b['feet_x'][l])}" for l in G.LEGS)
        sl = log.of(lift_segs[k]) if k < len(lift_segs) else None
        supp = (min(sl["foot_min"][l] for l in G.LEGS if l != b["leg"])
                if sl else float("nan"))
        say(f"    {b['name']:<8}{b.get('lift', 0)*1000:8.0f}mm"
            f"{b['sink']*1000:+8.0f}mm{b['slip_mm']:7.1f}mm{b.get('peak', 0)*100:8.0f}%"
            f"{b['tilt']:5.0f}°{supp:11.0f}N   {lvs}")
    say(f"    最吃力的关节：{res['peak_act']}（峰值占额 {res['peak_tau']*100:.0f}%）")
    if res["beats"]:
        say(f"    机身横向/朝向：y {res['beats'][-1]['y']:.3f}"
            f"（这一步的命令值本身含「挪重心」的横向偏置，不能直接和楼梯中线 {G.BODY_Y:.3f} 比）、"
            f"yaw {res['beats'][-1]['yaw']:+.1f}°（规划只管俯仰，yaw 没有闭环）")
    say(f"    走通 {done}/{total} 步；"
        f"{'在【' + str(res['failed_at']) + '】失去稳定' if res['fell'] else '全程未失稳'}")
    say(f"    末态机身 x {res['x']:+.3f} z {res['z']:.3f}，最大倾角 {res['max_tilt']:.0f}°")
    lost = log.last if res["fell"] else None
    if lost is not None:
        say("    失稳那一段的接触证据（非足端撞梯级，>5 N）：")
        for (a, b_), n in sorted(lost["foreign"].items(), key=lambda kv: -kv[1])[:6]:
            say(f"      · {a} × {b_}：{n:.0f} N")
        say("      支撑腿最小力：" + "  ".join(
            f"{l} {lost['foot_min'][l]:.0f}N" for l in G.LEGS))
        log.dump(OUT_TRACE, lost["seg"].name)
        say(f"      时间历程已写入 {OUT_TRACE.name}（机身 x/z/pitch/tilt、各足高度与受力）")
    say(f"    足端高度（实测）：" + "  ".join(f"{l} {res['feet_z'][l]:.3f}" for l in G.LEGS)
        + "（一级顶 0.197 / 二级 0.394 / 三级 0.591 / 平台 0.788，足心在顶面 + 0.035）")
    say()
    if res0 is not None:
        say(f"  对照（同一条轨迹，只改控制）：无机身伺服走通 {len(res0['beats'])}/{total} 步"
            f"（在 {res0['failed_at'] or '—'} 停），加伺服 k={servo} 走通 {done}/{total} 步。")

    final = None
    if not res["fell"] and run_segs:
        # 末态取"实际跑完的最后一段"：计划从缓存读时 steps 是空的，不能靠 steps[-1]
        body_f = tuple(run_segs[-1].body_b)
        final_feet = dict(run_segs[-1].feet_b)
        h = G.hold(rig, fg, final_feet, body_f, 2.0)          # 末态原地静置 2 s
        cmap = contact_map(rig, fg)
        on_plat = {leg: any(nm == "platform" for nm, _ in cmap.get(leg, [])) for leg in G.LEGS}
        final = {"hold": h, "contact": {k: [[nm, round(n, 1)] for nm, n in v]
                                       for k, v in cmap.items()},
                 "on_platform": on_plat}
        say()
        say("  末态核验（原地静置 2 s，读 contact 里真实生效的 geom 与法向力）：")
        say(f"    机身 ({body_f[0]:.3f}, {body_f[1]:.3f}, {body_f[2]:.3f})  "
            f"pitch {math.degrees(body_f[3]):+.1f}°"
            + (f"  伸展 {steps[-1]['max_ext']*100:.0f}%  "
               f"重心余量 {steps[-1]['margin_settle']*1000:.0f} mm"
               if len(steps) == total else ""))
        if h is None:
            say("    静置：末态 IK 不可达，未做静置核验")
        else:
            say(f"    静置：漂移 Δx {h['dx']*1000:+.0f} / Δy {h['dy']*1000:+.0f} / Δz {h['dz']*1000:+.0f} mm，"
                f"倾角 {h['tilt']:.1f}°，打滑 {h['slip_mm']:.1f} mm，力矩峰值 {h['peak_tau']*100:.0f}%，"
                f"{'未倾覆' if h['tilt'] < 20 else '有倾覆风险'}")
        say("    四足接触：" + "  ".join(
            f"{leg}→{'/'.join(f'{nm} {n:.0f}N' for nm, n in cmap.get(leg, [])) or '无'}"
            for leg in G.LEGS))
        say("    四足都在平台（platform geom）上：" + ("是" if all(on_plat.values()) else
                                                    "否（" + ",".join(l for l, v in on_plat.items() if not v) + " 不在）"))

    if not res["fell"] and done == total:
        say()
        say(f"  结论：{total} 步全部走通，四足全部落到平台（第 4 级，顶 0.788 m）："
            f"机身 {0.680:.3f} → {res['z']:.3f} m。"
            f"全程峰值力矩 {res['peak_tau']*100:.0f}%（{res['peak_act']}）—— 成功登上平台。"
            + ("起步在平地上（%d 步入场 + %d 步爬梯）。" % (n_entry, total - n_entry) if entry and n_entry else ""))
    elif done >= 4:
        say()
        say(f"  结论：走到第 {done}/{total} 步（{res['failed_at']}）失稳，"
            f"机身到 {res['z']:.3f} m、四足在 "
            + " ".join(f"{l}{level_of(rig, res['feet_z'][l], res.get('feet_x', {}).get(l, 0.0))}"
                       for l in G.LEGS)
            + f"；峰值力矩 {res['peak_tau']*100:.0f}%（{res['peak_act']}）。")
    else:
        say()
        say(f"  结论：第 {done} 步之后失稳（{res['failed_at']}）；峰值力矩 "
            f"{res['peak_tau']*100:.0f}%（{res['peak_act']}）")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"cycles": cycles, "base": base, "steps": steps, "notes": notes,
                                    "final": final,
                                    "result": {k: v for k, v in res.items() if k != "beats"},
                                    "beats": res["beats"]}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(text)
    print("WROTE", OUT_TXT)
    if "--png" in argv:
        render(rig, fg, run_segs, steps, res, servo)   # 实际跑的那套段（两段式），不是名义计划
    return 0


def render(rig: G.Rig, fg, segs, steps, res, servo: float = 1.2, picks=None) -> None:
    """抽几帧拼成联图：起步 + 每个循环结束（第 4/8/12 步）+ 最后一步。"""
    from PIL import Image, ImageDraw, ImageFont

    model, data = rig.model, rig.data
    if picks is None:
        n_lift = len([x for x in segs if x.lift])
        picks = [n for n in (4, 8, 12, n_lift) if n <= n_lift] if n_lift >= 8 \
            else list(range(1, n_lift + 1))
        picks = sorted(set(picks))
    want = set(picks)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    shots, tags = [], []
    state = {"n": 0}

    def snap(tag: str, lookat) -> None:
        r = mujoco.Renderer(model, 560, 800)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 96.0, -6.0, 2.8
        cam.lookat[:] = lookat
        r.update_scene(data, camera=cam, scene_option=opt)
        shots.append(Image.fromarray(r.render()))
        r.close()
        tags.append(tag)

    # 起步那一帧要先把机器人摆到起步姿态（否则拍到的是 MJCF 默认位，不是这条轨迹的起点）
    j0, _ = G.legs_ik(rig, segs[0].body_a, segs[0].feet_a)
    G.set_pose(rig, data, segs[0].body_a, j0)
    flat = max(v[2] for v in segs[0].feet_a.values()) < rig.r + PRELOAD + 0.05
    snap("起步：四足在地面（平地站姿）" if flat else "起步：FL2 / FR1 / RR1 / RL0",
         (0.42, G.BODY_Y, 0.60))

    def hook(seg, u):
        if seg.name.startswith("落定") and u > 0.97:
            state["n"] += 1
            if state["n"] in want:
                snap(f"第 {state['n']} 步：{seg.name[2:]}",
                     (max(0.55, float(data.qpos[0])), G.BODY_Y, float(data.qpos[2]) - 0.12))

    G.run_plan(rig, segs, fg, hook=hook, body_servo=servo)
    if not shots:
        return
    W, H, bar = 800, 560, 34
    canvas = Image.new("RGB", (W * len(shots) + 8 * (len(shots) + 1), H + bar + 16), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    font = None
    for name in ("msyh.ttc", "simhei.ttf", "simsun.ttc"):      # PIL 默认字体没有中文字形
        try:
            font = ImageFont.truetype(f"C:/Windows/Fonts/{name}", 19)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()
    for i, img in enumerate(shots):
        x = 8 + i * (W + 8)
        canvas.paste(img, (x, bar))
        draw.text((x + 4, 8), tags[i], fill=(235, 235, 235), font=font)
    canvas.save(OUT_PNG)
    print("WROTE", OUT_PNG)


def view(rig: G.Rig, fg, segs, steps, cycles: int = 4, base: int = 0,
         entry: bool = True, one_phase: bool = False) -> int:
    """3D 播放整条步态（与跑数据同一条流程），播完停在窗口里（--slow = 0.5 倍速）。

    规划全部在开窗口之前做完：两段式的爬梯段要按入口结束的实测位姿重规划，
    以前是在入口走完之后才算的 —— 窗口里机器狗站着不动好几分钟，看着像卡死。
    现在先在无窗口状态下把入口段快跑一遍、拿到实测状态，把爬梯段规划好（有缓存就直读），
    开窗口以后就是一路不停地走完 20 步。
    """
    import mujoco.viewer

    hz = 25.0 if "--slow" in sys.argv[1:] else 50.0
    model, data = rig.model, rig.data
    n_lift = len([x for x in segs if x.lift])
    two = entry and not one_phase
    plans = None
    if two:
        e_segs = entry_part(rig, fg, cycles, base, segs)
        G.run_plan(rig, e_segs, fg, body_servo=1.2)      # 无窗口快跑入口段（约 1 s），拿实测状态
        st = read_state(rig, fg)
        c_segs, _, c_notes = plan_climb(rig, fg, cycles, base, st, "--rebuild" not in sys.argv[1:])
        if c_segs is None:
            print("  按实测重规划爬梯段失败：", c_notes)
            return 1
        plans = (e_segs, c_segs, st)
    print(f"  播放（{hz/50:.1f} 倍速）：{n_lift} 步"
          f"{'（两段式：入口 → 实测重规划 → 爬梯，中间不停）' if two else ''}，"
          f"每步 挪重心→抬腿→落定", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = (0.55, G.BODY_Y, 0.40)
        viewer.cam.distance = 2.6
        viewer.cam.azimuth = 72.0
        viewer.cam.elevation = -8.0

        def follow():
            viewer.cam.lookat[:] = (float(data.qpos[0]) + 0.05, G.BODY_Y,
                                    float(data.qpos[2]) - 0.12)

        sync = G.pacer(viewer, on_frame=follow, hz=hz)
        if two:
            res, _, _, _, _ = run_two_phase(rig, fg, cycles, base, 1.2, sync=sync, plans=plans)
        else:
            res = G.run_plan(rig, segs, fg, sync=sync, body_servo=1.2)
        print(f"  结果：走通 {len(res['beats'])}/{n_lift} 步，"
              f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '全程未失稳'}"
              f"，末态机身 x {res['x']:+.3f} z {res['z']:.3f}"
              + ("，四足都在平台上了" if not res["fell"] else ""), flush=True)
        G.hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
