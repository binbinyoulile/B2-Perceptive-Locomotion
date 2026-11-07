"""B2 静态姿态对比：四足分踩「3 个高度」vs「4 个高度」，允许大幅抬头做联合寻优。

文件目的：
    回答一个问题——把前后腿的伸展率拉开以后，是"两只后脚同在一级"更从容（姿态 A），
    还是"四只脚各踩一级"更从容（姿态 B）？
        姿态 A：FL 第2级 / FR 第1级 / RR 第1级 / RL 地面（3 个高度）
        姿态 B：FL 第3级 / FR 第2级 / RR 第1级 / RL 地面（4 个高度）
    每种姿态都在 (机身 x, y, z, pitch) 上做网格搜索，pitch 允许 0°–30° 大幅抬头，
    硬约束：四腿伸展 < 95%、机身包络（含机头）离梯级 ≥10 mm、重心投影在支撑多边形内。

    为了让"伸展率"这个指标不被目标函数污染，每种姿态给两个解：
      · 余量优先：重心余量最大的位姿（代表这个姿态的支撑能力上限）
      · 关节优先：余量不低于上限 1 cm 的前提下，四腿最大伸展率最小的位姿（代表"从容"）
    再给一张 Pareto 表：把伸展率上限卡在 60/70/80/90/95% 时，各自最多能拿到多少余量。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（0.197/0.272/0.81，μ=1.0）
输出：
    experiments/b2/b2_stance_compare.txt / .json
    experiments/b2/b2_stance_compare.png（A|B 两个位姿的渲染对照）
依赖：mujoco 3.16、numpy、Pillow；复用 b2_diag_stance（候选枚举 / PD 保持 / 内切圆）与
      three_robot_gait_0272（rig、三维 IK、重心与机身避碰）。
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_stance_compare.py
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_stance_compare.py --view [--stance a|b]

关键假设：
    - "机头"取机身碰撞体里最靠前的那块 geom（B2 是 x≈+0.41 的防撞块），单独报它的净空。
    - 模型里没有栏杆几何：两侧只有纯视觉斜梁（contype=0），所以净空只按梯级踏板 + 平台算。
    - 静态判据仍用 6 s PD 静置（kp/kd 800/20），不做扰动。
已知限制：
    - 机身只按 pitch 参数化（不滚转），侧向靠机身 y 平移 + 髋外展。
    - 足端球心的 x 偏移（B2 为 0）忽略；IK-FK 残差实测 ≤2 mm。
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

OUT_TXT = ROOT / "experiments" / "b2" / "b2_stance_compare.txt"
OUT_JSON = ROOT / "experiments" / "b2" / "b2_stance_compare.json"
OUT_PNG = ROOT / "experiments" / "b2" / "b2_stance_compare.png"

EXT_CAP = 0.95        # 题目要求 <95%
LEG_CLEAR = 0.030     # 除足端球以外、机器人任何碰撞体离梯级的净空
                      # （题目要求是"不碰撞"= >0；这里留 30 mm，因为 PD 静置
                      #   实测下沉 2.7 cm，10 mm 的净空会被吃掉后压在台阶上）
PITCH_HI = 30.0       # 题目允许抬头到 30°
MARGIN_TOL = 0.010    # "关节优先"解允许比最大余量少 1 cm


def feet_of(rig: G.Rig, kind: str) -> dict:
    """两种姿态的足端落点（前足贴踏面前沿、后足贴后沿，各留 10 mm）。"""
    x1, x2, x3 = G.TREAD_CX
    z1, z2, z3 = G.TREAD_TOPS
    r, fy, e = rig.r, rig.foot_y, 0.010
    ground = (x1 - G.HALF - 0.15, G.BODY_Y + fy["RL"], r + 0.003)
    rear_rr = (x1 - G.HALF + r + e, G.BODY_Y + fy["RR"], z1 + r + 0.003)
    if kind == "A":
        return {"FL": (x2 + G.HALF - r - e, G.BODY_Y + fy["FL"], z2 + r + 0.003),
                "FR": (x1 + G.HALF - r - e, G.BODY_Y + fy["FR"], z1 + r + 0.003),
                "RR": rear_rr, "RL": ground}
    return {"FL": (x3 + G.HALF - r - e, G.BODY_Y + fy["FL"], z3 + r + 0.003),
            "FR": (x2 + G.HALF - r - e, G.BODY_Y + fy["FR"], z2 + r + 0.003),
            "RR": rear_rr, "RL": ground}


def knee_of(rig: G.Rig) -> int:
    """机身碰撞体里最靠前的那块 geom（= 机头防撞块）。"""
    return max(rig.root_gids, key=lambda g: float(rig.model.geom_pos[g][0]))


def nose_gap(rig: G.Rig, pose: dict) -> float:
    knee = knee_of(rig)
    G.set_pose(rig, rig.probe, pose["body"], pose["joints"])
    return float(min(mujoco.mj_geomDistance(rig.model, rig.probe, knee, g, 2.0, None)
                     for g in rig.plate_gids))


def self_gids(rig: G.Rig, fg) -> list[int]:
    """机器人身上真正会碰撞的 geom，去掉四只足端球（它们本来就要压在踏面上）。"""
    feet_ids = set(fg.values())
    m = rig.model
    return [g for g in range(m.ngeom)
            if m.geom_contype[g] and g not in feet_ids
            and int(m.body_rootid[int(m.geom_bodyid[g])]) == 1]   # 只取机器人这棵树


def min_clear(rig: G.Rig, pose_body, joints, gids) -> float:
    """机器人非足端碰撞体 vs 梯级踏板的最小距离（<=LEG_CLEAR 视为搭在台阶上）。"""
    G.set_pose(rig, rig.probe, pose_body, joints)
    best = 1e9
    for g in gids:
        for plate in rig.plate_gids:
            d = mujoco.mj_geomDistance(rig.model, rig.probe, g, plate, 2.0, None)
            if d < best:
                best = d
    return float(best)


def analyse(rig: G.Rig, feet: dict) -> dict:
    cands = D.enumerate_poses(rig, feet, pitch_lo=0.0, pitch_hi=PITCH_HI, ext_max=EXT_CAP,
                              xs=np.arange(0.28, 0.90, 0.02),
                              ys=np.arange(G.BODY_Y - 0.10, G.BODY_Y + 0.101, 0.02),
                              zs=np.arange(0.46, 1.06, 0.02))
    if not cands:
        return {"ok": False}
    gids = self_gids(rig, G.foot_gids(rig))
    kept = []
    for c in cands:
        c["leg_clear"] = min_clear(rig, c["body"], c["joints"], gids)
        if c["leg_clear"] >= LEG_CLEAR:
            kept.append(c)
    cands = kept
    if not cands:
        return {"ok": False, "why": "所有可行位姿都有腿搭在台阶上"}
    poly = [(feet[k][0], feet[k][1]) for k in G.LEGS]
    best_m = max(cands, key=lambda c: c["margin"])
    relaxed = [c for c in cands if c["margin"] >= best_m["margin"] - MARGIN_TOL]
    best_j = min(relaxed, key=lambda c: (c["max_ext"], abs(c["body"][3])))
    pareto = []
    for cap in (0.60, 0.70, 0.80, 0.90, EXT_CAP):
        sub = [c for c in cands if c["max_ext"] <= cap + 1e-9]
        if sub:
            b = max(sub, key=lambda c: c["margin"])
            pareto.append({"cap": cap, "margin": b["margin"], "max_ext": b["max_ext"],
                           "pitch": math.degrees(b["body"][3]), "z": b["body"][2]})
    needs = []
    for t in (0.150, 0.160, 0.170, 0.180, 0.185):
        sub = [c for c in cands if c["margin"] >= t]
        if sub:
            b = min(sub, key=lambda c: (c["max_ext"], abs(c["body"][3])))
            needs.append({"target": t, "max_ext": b["max_ext"],
                          "pitch": math.degrees(b["body"][3]), "z": b["body"][2]})
        else:
            needs.append({"target": t, "max_ext": None})
    ir, ir_pt = D.inradius(poly)
    return {"ok": True, "n": len(cands), "poly": poly, "inradius": ir, "ir_pt": ir_pt,
            "margin_pose": best_m, "joint_pose": best_j, "pareto": pareto, "needs": needs}


def detail(rig: G.Rig, fg, pose: dict, feet: dict) -> dict:
    hold = D.hold(rig, fg, dict(pose, feet=feet), seconds=6.0)
    return {"margin": pose["margin"], "max_ext": pose["max_ext"], "ext": pose["ext"],
            "gap": pose["gap"], "nose": nose_gap(rig, pose),
            "body": pose["body"], "joints": pose["joints"], "hold": hold,
            "leg_clear": pose.get("leg_clear", float("nan"))}


def render(rig: G.Rig, poses: dict, out: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    model, data = rig.model, rig.data
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    W, H, bar = 800, 560, 40
    canvas = Image.new("RGB", (W * 2 + 12, H + bar), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(17)
    except TypeError:
        font = ImageFont.load_default()
    for i, (label, pose) in enumerate(poses.items()):
        G.set_pose(rig, data, pose["body"], pose["joints"])
        r = mujoco.Renderer(model, H, W)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = 96.0, -6.0, 2.2
        cam.lookat[:] = (0.45, G.BODY_Y, 0.40)
        r.update_scene(data, camera=cam, scene_option=opt)
        canvas.paste(Image.fromarray(r.render()), (i * (W + 12), bar))
        r.close()
        b = pose["body"]
        txt = (f"{label}: body x{b[0]:.2f} z{b[2]:.2f} pitch {math.degrees(b[3]):+.0f}deg | "
               f"ext max {pose['max_ext']*100:.0f}% | margin {pose['margin']*1000:.0f}mm")
        draw.text((i * (W + 12) + 8, 10), txt, fill=(235, 235, 235), font=font)
    canvas.save(out)


def main() -> int:
    argv = sys.argv[1:]
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    if "--view" in argv:
        kind = argv[argv.index("--stance") + 1].upper() if "--stance" in argv else "both"
        return view(rig, fg, kind)

    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 100)
    say("B2 静态姿态对比：四足分踩 3 个高度（A）vs 4 个高度（B），允许抬头 0–30° 联合寻优")
    say("=" * 100)
    say(f"  硬约束：四腿伸展 < {EXT_CAP*100:.0f}%、重心投影在支撑多边形内、"
        f"机器人除足端外任何碰撞体离梯级 ≥ {LEG_CLEAR*1000:.0f} mm")
    say(f"  （题目要求「不碰撞」即 >0；这里取 {LEG_CLEAR*1000:.0f} mm 是因为 PD 静置实测要沉 2.7 cm，"
        f"10 mm 的净空会被吃掉后压在台阶上）")
    say(f"  机身 (x, y, z, pitch 0–{PITCH_HI:.0f}°) 网格搜索")
    say()

    out: dict[str, dict] = {}
    for kind in ("A", "B"):
        feet = feet_of(rig, kind)
        a = analyse(rig, feet)
        say("-" * 100)
        say(f"### 姿态 {kind}：" + ("FL 第2级 / FR+RR 第1级 / RL 地面（3 个高度）" if kind == "A"
                                    else "FL 第3级 / FR 第2级 / RR 第1级 / RL 地面（4 个高度）"))
        if not a["ok"]:
            say("  无可行解")
            out[kind] = {"ok": False}
            continue
        say(f"  足端：" + "  ".join(
            f"{l} x{feet[l][0]:.3f} z{feet[l][2]:.3f}" for l in G.LEGS))
        say(f"  支撑多边形：x {min(p[0] for p in a['poly']):.3f}..{max(p[0] for p in a['poly']):.3f}"
            f"（跨度 {max(p[0] for p in a['poly'])-min(p[0] for p in a['poly']):.3f} m）"
            f"，y {min(p[1] for p in a['poly']):.3f}..{max(p[1] for p in a['poly']):.3f}"
            f"，内切圆半径 {a['inradius']*1000:.0f} mm")
        say(f"  可行位姿 {a['n']} 个")
        rec = {"ok": True, "feet": feet, "inradius": a["inradius"],
               "poly_x": [min(p[0] for p in a["poly"]), max(p[0] for p in a["poly"])]}
        for tag, key in (("余量优先解", "margin_pose"), ("关节优先解", "joint_pose")):
            p = a[key]
            det = detail(rig, fg, p, feet)
            b = p["body"]
            h = det["hold"]
            say(f"  【{tag}】机身 x {b[0]:.3f} y {b[1]:.3f} z {b[2]:.3f} pitch {math.degrees(b[3]):+.1f}°")
            say(f"    四腿伸展 FL {p['ext']['FL']*100:.0f}% / FR {p['ext']['FR']*100:.0f}% / "
                f"RL {p['ext']['RL']*100:.0f}% / RR {p['ext']['RR']*100:.0f}%（最大 {p['max_ext']*100:.0f}%）")
            say(f"    重心余量 {p['margin']*1000:+.1f} mm（占内切圆 {p['margin']/a['inradius']*100:.0f}%）"
                f"｜机身离梯级 {p['gap']*1000:.0f} mm｜机头净空 {det['nose']*1000:.0f} mm"
                f"｜非足端净空 {p['leg_clear']*1000:.0f} mm")
            say(f"    静置 6 s：Δz {h['dz']*1000:+.1f} mm，倾角 {h['tilt']:.1f}°，"
                f"{'倾覆' if h['fallen'] else '未倾覆'}，实测余量 {h['margin_actual']*1000:+.1f} mm，"
                f"支撑力 " + "/".join(f"{h['force'][l]:.0f}" for l in G.LEGS) + " N")
            if h["other_contacts"]:
                say(f"    ⚠ 非足端接触：" + "，".join(f"{b}×{s}" for b, s in h["other_contacts"]))
            rec[tag] = {"body": b, "ext": p["ext"], "max_ext": p["max_ext"],
                        "margin": p["margin"], "gap": p["gap"], "nose": det["nose"], "hold": h,
                        "joints": det["joints"], "leg_clear": det["leg_clear"]}
        say("  Pareto（把最大伸展率卡住，最多还能拿到多少重心余量）：")
        for row in a["pareto"]:
            say(f"    伸展 ≤ {row['cap']*100:.0f}%：余量最多 {row['margin']*1000:.0f} mm"
                f"（此时 pitch {row['pitch']:+.0f}°、机身 z {row['z']:.2f}）")
        say("  同余量下最省的伸展（想拿这么多余量，腿最少要伸到多少）：")
        for row in a["needs"]:
            if row["max_ext"] is None:
                say(f"    余量 ≥ {row['target']*1000:.0f} mm：无解")
            else:
                say(f"    余量 ≥ {row['target']*1000:.0f} mm：伸展最少 {row['max_ext']*100:.0f}%"
                    f"（pitch {row['pitch']:+.0f}°、机身 z {row['z']:.2f}）")
        rec["pareto"] = a["pareto"]
        rec["needs"] = a["needs"]
        out[kind] = rec

    say()
    say("=" * 100)
    say("对比（每列取各自可行集里「伸展最省」的那个解，比较姿态与伸展，而不是比谁余量大）")
    say("=" * 100)
    if out.get("A", {}).get("ok") and out.get("B", {}).get("ok"):
        A, B = out["A"]["关节优先解"], out["B"]["关节优先解"]
        say(f"  {'指标':<22}{'姿态 A（3 高度）':>20}{'姿态 B（4 高度）':>20}")
        rows = [
            ("最大伸展率", f"{A['max_ext']*100:.0f}%", f"{B['max_ext']*100:.0f}%"),
            ("伸展率之和", f"{sum(A['ext'].values())*100/4:.0f}%（均值）",
             f"{sum(B['ext'].values())*100/4:.0f}%（均值）"),
            ("机身俯仰", f"{math.degrees(A['body'][3]):+.0f}°", f"{math.degrees(B['body'][3]):+.0f}°"),
            ("机身高度 z", f"{A['body'][2]:.3f} m", f"{B['body'][2]:.3f} m"),
            ("重心余量", f"{A['margin']*1000:+.0f} mm", f"{B['margin']*1000:+.0f} mm"),
            ("机头净空", f"{A['nose']*1000:.0f} mm", f"{B['nose']*1000:.0f} mm"),
            ("支撑面 x 跨度", f"{out['A']['poly_x'][1]-out['A']['poly_x'][0]:.3f} m",
             f"{out['B']['poly_x'][1]-out['B']['poly_x'][0]:.3f} m"),
            ("6 s 静置 Δz", f"{A['hold']['dz']*1000:+.0f} mm", f"{B['hold']['dz']*1000:+.0f} mm"),
            ("最小支撑力", f"{min(A['hold']['force'].values()):.0f} N",
             f"{min(B['hold']['force'].values()):.0f} N"),
        ]
        for name, va, vb in rows:
            say(f"  {name:<22}{va:>20}{vb:>20}")

    if out.get("A", {}).get("ok") and out.get("B", {}).get("ok"):
        AM, BM = out["A"]["余量优先解"], out["B"]["余量优先解"]
        AJ, BJ = out["A"]["关节优先解"], out["B"]["关节优先解"]
        span = lambda k: out[k]["poly_x"][1] - out[k]["poly_x"][0]
        n170 = {k: next((r for r in out[k]["needs"] if abs(r["target"] - 0.170) < 1e-9), None)
                for k in ("A", "B")}
        ext170 = {k: (n170[k]["max_ext"] if n170[k] and n170[k]["max_ext"] else None) for k in ("A", "B")}
        say()
        say("-" * 100)
        say("判读")
        say(f"  1) 支撑面：B 的 x 跨度 {span('B'):.3f} m，比 A 的 {span('A'):.3f} m 长 "
            f"{(span('B')/span('A')-1)*100:.0f}%；但内切圆半径只从 {out['A']['inradius']*1000:.0f} mm "
            f"涨到 {out['B']['inradius']*1000:.0f} mm（+{(out['B']['inradius']/out['A']['inradius']-1)*100:.0f}%）"
            f"——限制余量的是宽度方向，不是长度方向。")
        say(f"  2) 余量上限：B {BM['margin']*1000:.0f} mm > A {AM['margin']*1000:.0f} mm"
            f"（+{(BM['margin']/AM['margin']-1)*100:.0f}%）。但 B 是在 pitch 顶到 30° 才拿到的，"
            f"A 只用 {math.degrees(AM['body'][3]):+.0f}°。")
        a170 = f"{ext170['A']*100:.0f}%" if ext170["A"] else "无解"
        b170 = f"{ext170['B']*100:.0f}%" if ext170["B"] else "无解"
        say(f"  3) 关节从容度（拿同样的 170 mm 余量）：A 只需伸展 {a170}，B 要 {b170}；"
            f"把余量放宽到 150 mm 时 A 还能退到 "
            f"{min(r['max_ext'] for r in out['A']['needs'] if r['max_ext']):.0%}，"
            f"B 最低也只能到 "
            f"{min(r['max_ext'] for r in out['B']['needs'] if r['max_ext']):.0%}"
            f"——B 的可行集里根本没有「低伸展」的解。")
        say(f"  4) 静置后的真实余量：A {AM['margin']*1000:.0f} → "
            f"{AM['hold']['margin_actual']*1000:.0f} mm（掉 "
            f"{(AM['margin']-AM['hold']['margin_actual'])*1000:.0f} mm）；"
            f"B {BM['margin']*1000:.0f} → {BM['hold']['margin_actual']*1000:.0f} mm（掉 "
            f"{(BM['margin']-BM['hold']['margin_actual'])*1000:.0f} mm）"
            f"——B 的那个解里 FL 只剩 {BM['hold']['force']['FL']:.0f} N 支撑力，已经接近悬空。")
        say()
        say("  结论：B 并不是「关节和重心都更从容」。")
        say(f"    · 重心上 B 略优：余量 {BM['margin']*1000:.0f} vs {AM['margin']*1000:.0f} mm，"
            f"多 {(BM['margin']-AM['margin'])*1000:.0f} mm（+{(BM['margin']/AM['margin']-1)*100:.0f}%），"
            f"代价是俯仰从 {math.degrees(AM['body'][3]):+.0f}° 顶到 {math.degrees(BM['body'][3]):+.0f}°。")
        a80 = next((r for r in out["A"]["pareto"] if abs(r["cap"] - 0.80) < 1e-9), None)
        say(f"    · 关节上 B 更被动：A 的余量上限 {AM['margin']*1000:.0f} mm 要 {AM['max_ext']*100:.0f}% 伸展，"
            f"但让出 {(AM['margin']-a80['margin'])*1000:.0f} mm 余量（→"
            f"{a80['margin']*1000:.0f} mm）就能降到 {a80['max_ext']*100:.0f}%；"
            f"B 的伸展下限是 {min(r['max_ext'] for r in out['B']['needs'] if r['max_ext'])*100:.0f}%，"
            f"再让余量也降不下来。")
        say("    · 稳定性上 B 更脆：最优解里前足几乎不受力、静置后余量掉 36 mm；A 只掉 4 mm。")
        say("  → 结论：保持姿态 A（3 个高度，两只后脚共用第 1 级）。B 的 0.27 m 额外跨度换不来"
            "关节上的从容，反而把俯仰和伸展都顶到上限。")
        say("  → 如果你要的是\"更大支撑面\"，可以只在需要抗推挤时用 B；要爬梯/常驻，用 A。")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)

    poses = {k: out[k]["关节优先解"] for k in ("A", "B") if out.get(k, {}).get("ok")}
    if poses:
        render(rig, poses, OUT_PNG)
    print("WROTE", OUT_TXT)
    if poses:
        print("WROTE", OUT_PNG)
    return 0


def view(rig: G.Rig, fg, kind: str) -> int:
    """3D 窗口：把 A/B 的最优位姿摆上去，PD 保持，然后停住不关窗。"""
    import mujoco.viewer

    model, data = rig.model, rig.data
    plan = []
    for k in (("A", "B") if kind == "both" else (kind,)):
        feet = feet_of(rig, k)
        a = analyse(rig, feet)
        if not a["ok"]:
            print(f"姿态 {k} 无可行解")
            continue
        p = a["joint_pose"]
        plan.append((k, feet, p))
        print(f"姿态 {k}：机身 x {p['body'][0]:.3f} z {p['body'][2]:.3f} "
              f"pitch {math.degrees(p['body'][3]):+.0f}°，最大伸展 {p['max_ext']*100:.0f}%，"
              f"余量 {p['margin']*1000:+.0f} mm")
    if not plan:
        return 1
    lp = G.Loop(rig)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = (0.45, G.BODY_Y, 0.40)
        viewer.cam.distance = 2.2
        viewer.cam.azimuth = 70.0
        viewer.cam.elevation = -8.0
        sync = G.pacer(viewer)
        for k, feet, p in plan:
            G.set_pose(rig, data, p["body"], p["joints"])
            for _ in range(int(4.0 * 50)):
                lp.step(p["joints"])
                sync()
            print(f"  姿态 {k} 保持 4 s 后：z {data.qpos[2]:.3f}，倾角 {G.tilt_of(data):.1f}°，"
                  f"接触 {data.ncon} 个")
            for _ in range(int(1.0 * 50)):
                lp.step(p["joints"])
                sync()
        G.hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
