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
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --view   （读缓存，0.5 倍速）
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


def edge_x(level: int) -> float:
    return TREAD_REAR0 + TREAD_DEPTH * (level - 1)


def foot_target(rig: G.Rig, leg: str, level: int):
    """某条腿在第 level 级上的落点。第 4 级是 2 m 深平台：前足往里踩深一点，
    把"前后足跨距"撑开（只有 0.19 m 时最后一拍的支撑三角形内切圆只有 72 mm，实测就是这一步失稳）。"""
    y = G.BODY_Y + rig.foot_y[leg]
    if level == 0:
        return (TREAD_REAR0 - 0.15, y, rig.r + PRELOAD)
    if level == TOP_LEVEL:
        x = edge_x(level) + (PLAT_FRONT if leg in ("FL", "FR") else PLAT_REAR)
        return (x, y, LEVEL_TOP[level] + rig.r + PRELOAD)
    cx = edge_x(level) + TREAD_DEPTH / 2.0
    d = TREAD_DEPTH / 2.0 - rig.r - EDGE
    x = cx + d if leg in ("FL", "FR") else cx - d
    return (x, y, LEVEL_TOP[level] + rig.r + PRELOAD)


def start_levels() -> dict:
    return {"FL": 2, "FR": 1, "RR": 1, "RL": 0}


def steps_of_cycle(c: int) -> list[tuple[str, int, int]]:
    """第 c 个循环的四步：(腿, 从第几级, 到第几级)。"""
    return [("FR", 1 + c, 2 + c), ("RL", 0 + c, 1 + c),
            ("FL", 2 + c, 3 + c), ("RR", 1 + c, 2 + c)]


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


def solve(rig: G.Rig, fg, feet, support, prev=None, tol: float = 0.030):
    """细网格搜索（2 cm / 3°），窗口绕上一位姿。
    粗网格（5 cm）快 5 倍，但搜出的位姿动力学上走不通，别换；窗口只缩"搜索范围"，不改分辨率。
    """
    if prev is None:
        xs = np.arange(0.20, 0.96, 0.02)
        zs = np.arange(0.46, 1.12, 0.02)
    else:
        xs = np.arange(max(0.10, prev[0] - WIN_X), prev[0] + WIN_X + 1e-9, 0.02)
        zs = np.arange(max(0.40, prev[2] - WIN_Z), prev[2] + WIN_Z + 1e-9, 0.02)
    cands = D.enumerate_poses(rig, feet, pitch_lo=0.0, pitch_hi=PITCH_HI, ext_max=EXT_CAP,
                              xs=xs, ys=np.arange(G.BODY_Y - 0.10, G.BODY_Y + 0.101, 0.02),
                              zs=zs, support=support)
    cands = D.filter_clear(rig, cands, fg)
    if not cands:
        return None
    best = pick(cands, prev, tol)
    # 命中窗口边界 → 说明真正的解可能在窗口外（下一步会因此解不出来），必须报警而不是静默
    best["win_hit"] = bool(prev is not None and (
        abs(best["body"][0] - prev[0]) > WIN_X - 0.021 or
        abs(best["body"][2] - prev[2]) > WIN_Z - 0.021))
    return best


def build(rig: G.Rig, fg, cycles: int):
    """生成整条轨迹 + 每步的计划信息。返回 (segs, steps, notes, 末态 feet)。"""
    levels = start_levels()
    feet = {leg: foot_target(rig, leg, levels[leg]) for leg in G.LEGS}
    body = solve(rig, fg, feet, G.LEGS)
    if body is None:
        return None, [], ["起步姿态无可行解"], None
    segs = [G.Seg("起步稳住", 0.6, body["body"], body["body"], dict(feet), dict(feet))]
    steps, notes = [], []
    cur = body

    def solve_pair(new_feet: dict, support3: tuple):
        """解"抬腿位姿"和"挪重心位姿"这一对。

        抬腿那 1.5 s 是最危险的（只有三足撑着），所以取余量尽量大的位姿（tol=0.005）。
        但如果这么选导致抬腿段要挪 3 cm 以上机身，就退回 tol=0.03 的"贴着上一位姿"解：
        实测只有 RL 从地面上一级那种（脚还在地上时腿已经到极限）才必须"边抬边升机身"，
        其余各步都能把机身运动放在四足着地的挪重心/落定两段完成。
        """
        lifted = solve(rig, fg, new_feet, support3, prev=cur["body"], tol=0.005)
        shifted = solve(rig, fg, feet, support3, prev=lifted["body"], tol=0.005) if lifted else None
        if lifted is not None and shifted is not None and (
                abs(lifted["body"][2] - shifted["body"][2]) > 0.030
                or abs(lifted["body"][0] - shifted["body"][0]) > 0.040):
            lifted = solve(rig, fg, new_feet, support3, prev=cur["body"], tol=0.030)
            shifted = solve(rig, fg, feet, support3, prev=lifted["body"], tol=0.030) if lifted else None
        return lifted, shifted

    def do_step(leg: str, new_feet: dict, label: str, cyc: int,
                aggressive: bool = False) -> bool:
        """一次单腿移动：挪重心 → 抬腿 → 落定。失败时写 notes 并返回 False。

        aggressive=False 是爬梯段的既定做法（tol=0.03，贴着上一位姿）。
        aggressive=True 只用在"平台段"（前足上平台后前移、后足最后上平台）：
        那里支撑三角形整个落在同一个水平面上，余量上限只有 70–120 mm，
        必须真取最大余量的位姿（含 8 cm 量级的横向偏重心），否则绕对角线一推就翻。
        """
        nonlocal cur
        support3 = tuple(l for l in G.LEGS if l != leg)
        if aggressive:
            lifted, shifted = solve_pair(new_feet, support3)
            tol_settle = 0.010
        else:
            lifted = solve(rig, fg, new_feet, support3, prev=cur["body"], tol=0.030)
            shifted = (solve(rig, fg, feet, support3, prev=lifted["body"], tol=0.030)
                       if lifted else None)
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
        segs.append(G.Seg(f"挪重心{leg}", SHIFT_S, cur["body"], shifted["body"],
                          dict(feet), dict(feet)))
        segs.append(G.Seg(f"抬{leg}", LIFT_S, shifted["body"], lifted["body"],
                          dict(feet), dict(new_feet), lift=leg, z_clear=z_clear))
        settled = solve(rig, fg, new_feet, G.LEGS, prev=lifted["body"], tol=tol_settle)
        if settled is None:
            notes.append(f"{label}（{leg}）：落定无可行位姿")
            return False
        segs.append(G.Seg(f"落定{leg}", SETTLE_S, lifted["body"], settled["body"],
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

    adjusted = False
    for c in range(cycles):
        for leg, lv_from, lv_to in steps_of_cycle(c):
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


def save_plan(segs, steps, cycles: int) -> None:
    data = {"cycles": cycles, "segments": [
        {"name": s.name, "dur": s.dur, "body_a": list(s.body_a), "body_b": list(s.body_b),
         "feet_a": {k: list(v) for k, v in s.feet_a.items()},
         "feet_b": {k: list(v) for k, v in s.feet_b.items()},
         "lift": s.lift, "z_clear": s.z_clear} for s in segs]}
    PLAN_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def load_plan(cycles: int):
    """读缓存的计划；没有或循环数对不上就返回 None。"""
    if not PLAN_JSON.exists():
        return None
    data = json.loads(PLAN_JSON.read_text(encoding="utf-8"))
    if data.get("cycles") != cycles:
        return None
    segs = [G.Seg(s["name"], s["dur"], tuple(s["body_a"]), tuple(s["body_b"]),
                  {k: tuple(v) for k, v in s["feet_a"].items()},
                  {k: tuple(v) for k, v in s["feet_b"].items()},
                  lift=s["lift"], z_clear=s["z_clear"]) for s in data["segments"]]
    return segs


def level_of(rig: G.Rig, z: float) -> str:
    """把足心高度映射回"在哪一级"（容差 3 cm，否则返回 ?）。"""
    for lv, top in sorted(LEVEL_TOP.items()):
        want = (rig.r + PRELOAD) if lv == 0 else (top + rig.r + PRELOAD)
        if abs(z - want) < 0.030:
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
                    key = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                           mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, other))
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


def main() -> int:
    argv = sys.argv[1:]
    cycles = int(argv[argv.index("--cycles") + 1]) if "--cycles" in argv else 4
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    cached = (None if ("--rebuild" in argv or "--plan" in argv or "--view" not in argv)
              else load_plan(cycles))
    if cached is not None:
        print(f"读缓存计划（{len(cached)} 段）；要重算加 --rebuild（12 步约 3–5 分钟）")
        segs, steps, notes, final_feet = cached, [], ["（读自缓存，规划表见上一次输出）"], None
    else:
        segs, steps, notes, final_feet = build(rig, fg, cycles)
        if segs is not None:
            save_plan(segs, steps, cycles)
    if segs is None:
        print("无法起步：", notes)
        return 1
    total = len([x for x in segs if x.lift])
    if "--view" in argv:
        return view(rig, fg, segs, steps)
    if "--plan" in argv:        # 只规划（几十秒），先确认每步都可解再花几分钟跑动力学
        print("\n".join(plan_table(steps, notes, total, cycles)))
        return 0

    servo = float(argv[argv.index("--servo") + 1]) if "--servo" in argv else 1.2
    log = ContactLog(rig, fg)
    res = G.run_plan(rig, segs, fg, hook=log, body_servo=servo)
    rig0 = G.load_rig("b2")
    res0 = G.run_plan(rig0, segs, G.foot_gids(rig0), body_servo=0.0)   # 同一条轨迹、纯开环
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    done = len(res["beats"])
    say("=" * 104)
    say("B2「交替跟进」步态：错位姿态起步，一次只动一条腿，一路爬到平台（第 4 级，顶 0.788 m）")
    say("=" * 104)
    lv = start_levels()
    say("  起步：" + "  ".join(f"{l} 第{lv[l]}级" if lv[l] else f"{l} 地面" for l in G.LEGS))
    say("  每循环四步：FR +1（与 FL 汇合）/ RL +1（与 RR 汇合）/ FL +1 / RR +1（与 FR 汇合）")
    say()
    for line in plan_table(steps, notes, total, cycles):
        say(line)
    say()
    say(f"  实测（PD 800/20 + 机身位姿伺服 k={servo}，每步：挪重心 {SHIFT_S} s → "
        f"抬腿 {LIFT_S} s → 落定 {SETTLE_S} s；抬脚离目标 {SWING_CLEAR*100:.0f} cm）：")
    say(f"    {'步骤':<8}{'抬脚高度':>9}{'机身下沉':>9}{'打滑':>8}{'力矩峰值':>9}{'倾角':>6}"
        f"{'支撑腿最小力':>13}   {'四足所在级（实测）'}")
    lift_segs = [x for x in segs if x.lift]     # beats 与抬腿段一一对应（都按顺序）
    for k, b in enumerate(res["beats"]):
        lvs = " ".join(f"{l}{level_of(rig, b['feet_z'][l])}" for l in G.LEGS)
        sl = log.of(lift_segs[k]) if k < len(lift_segs) else None
        supp = (min(sl["foot_min"][l] for l in G.LEGS if l != b["leg"])
                if sl else float("nan"))
        say(f"    {b['name']:<8}{b.get('lift', 0)*1000:8.0f}mm"
            f"{b['sink']*1000:+8.0f}mm{b['slip_mm']:7.1f}mm{b.get('peak', 0)*100:8.0f}%"
            f"{b['tilt']:5.0f}°{supp:11.0f}N   {lvs}")
    say(f"    最吃力的关节：{res['peak_act']}（峰值占额 {res['peak_tau']*100:.0f}%）")
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
    say(f"  对照（同一条轨迹，只改控制）：无机身伺服走通 {len(res0['beats'])}/{total} 步"
        f"（在 {res0['failed_at'] or '—'} 停），加伺服 k={servo} 走通 {done}/{total} 步。")

    final = None
    if not res["fell"] and steps:
        body_f = tuple(steps[-1]["body"])
        h = G.hold(rig, fg, final_feet, body_f, 2.0)          # 末态原地静置 2 s
        cmap = contact_map(rig, fg)
        on_plat = {leg: any(nm == "platform" for nm, _ in cmap.get(leg, [])) for leg in G.LEGS}
        final = {"hold": h, "contact": {k: [[nm, round(n, 1)] for nm, n in v]
                                       for k, v in cmap.items()},
                 "on_platform": on_plat}
        say()
        say("  末态核验（原地静置 2 s，读 contact 里真实生效的 geom 与法向力）：")
        say(f"    机身 ({body_f[0]:.3f}, {body_f[1]:.3f}, {body_f[2]:.3f})  "
            f"pitch {math.degrees(body_f[3]):+.1f}°  伸展 {steps[-1]['max_ext']*100:.0f}%  "
            f"重心余量 {steps[-1]['margin_settle']*1000:.0f} mm")
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
            f"机身 {0.680:.3f} → {res['z']:.3f} m（升 {(res['z']-0.680)*100:.0f} cm）。"
            f"全程峰值力矩 {res['peak_tau']*100:.0f}%（{res['peak_act']}）—— 成功登上平台。")
    elif done >= 4:
        say()
        say(f"  结论：走到第 {done}/{total} 步（{res['failed_at']}）失稳，"
            f"机身到 {res['z']:.3f} m、四足在 "
            + " ".join(f"{l}{level_of(rig, res['feet_z'][l])}" for l in G.LEGS)
            + f"；峰值力矩 {res['peak_tau']*100:.0f}%（{res['peak_act']}）。")
    else:
        say()
        say(f"  结论：第 {done} 步之后失稳（{res['failed_at']}）；峰值力矩 "
            f"{res['peak_tau']*100:.0f}%（{res['peak_act']}）")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"cycles": cycles, "steps": steps, "notes": notes,
                                    "final": final,
                                    "result": {k: v for k, v in res.items() if k != "beats"},
                                    "beats": res["beats"]}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(text)
    print("WROTE", OUT_TXT)
    if "--png" in argv:
        render(rig, fg, segs, steps, res, servo)
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
    snap("起步：FL2 / FR1 / RR1 / RL0", (0.42, G.BODY_Y, 0.60))

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


def view(rig: G.Rig, fg, segs, steps) -> int:
    """3D 慢动作：播整条步态，播完停在窗口里（--fast = 1 倍速）。"""
    import mujoco.viewer

    hz = 50.0 if "--fast" in sys.argv[1:] else 25.0
    model, data = rig.model, rig.data
    n_lift = len([x for x in segs if x.lift])
    print(f"  播放（{hz/50:.1f} 倍速）：{n_lift} 步（{'→'.join(x.lift for x in segs if x.lift)}），"
          f"每步 挪重心→抬腿→落定")
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
        res = G.run_plan(rig, segs, fg, sync=sync, body_servo=1.2)
        print(f"  结果：走通 {len(res['beats'])}/{n_lift} 步，"
              f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '全程未失稳'}")
        G.hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
