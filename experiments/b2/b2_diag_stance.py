"""B2 极限错位姿态：四只脚踩在四个不同高度上，能不能稳住（纯静态，不开步态）。

文件目的：
    只回答一个问题——把 B2 摆成下面这个又不对称又扭曲的姿态，它会不会自己倒：
        左前足 FL → 第 2 级踏面（顶 0.394）
        右前足 FR → 第 1 级踏面的前沿（顶 0.197）
        右后足 RR → 第 1 级踏面的后沿（顶 0.197）
        左后足 RL → 地面（第 0 级，z=0）
    量三件事：
      1) 重心投影到"四足支撑多边形"的最小距离（重心余量）——几何判据；
      2) PD 保持若干秒后的机身漂移、足端打滑、倾角、是否倾覆——动力学判据；
      3) 每只脚的实际法向支撑力——证明它是真的四点支撑，而不是有一只脚早就悬空了。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（0.197/0.272/0.81 钢梯，μ=1.0）
输出：
    experiments/b2/b2_diag_stance.txt / .json —— 实测数据
    experiments/b2/b2_diag_stance.png —— 侧视 + 斜视两张渲染
依赖：mujoco 3.16、numpy、Pillow；复用 experiments/feasibility/three_robot_gait_0272.py 的
      rig 加载 / 三维腿部 IK / 重心与机身避碰 / PD 循环。
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_diag_stance.py
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_diag_stance.py --view   （3D 窗口，播完不关）

关键假设：
    - 机身位姿不是我手选的一个点：在 (x, y, z, pitch) 网格里搜"重心余量最大"的那一个，
      同时要求四足可达（伸展 35–94%）、机身包络离梯级 ≥10 mm。这样得到的是这个姿态的
      上限余量，而不是我随手摆的位置能拿到多少。
    - 脚点按"把支撑面铺到最大"来放：FR 贴第 1 级前沿、RR 贴第 1 级后沿、FL 贴第 2 级前沿。
    - 50 Hz PD，kp/kd 800/20，重力前馈；摩擦 μ=1.0（足端 0.4 与踏面 1.0 取逐元素最大值）。
已知限制：
    - 没有扰动/推挤测试，纯静置；真实机器人还要考虑脚底柔性、格栅咬合、电池重心分布差异。
    - 机身只按 pitch 参数化（不滚转），侧向重心的调整靠机身 y 平移 + 髋外展完成。
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
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402

import three_robot_gait_0272 as G  # noqa: E402

OUT_TXT = ROOT / "experiments" / "b2" / "b2_diag_stance.txt"
OUT_JSON = ROOT / "experiments" / "b2" / "b2_diag_stance.json"
OUT_PNG = ROOT / "experiments" / "b2" / "b2_diag_stance.png"

EDGE = 0.010          # 足端离踏面边缘的余量
PRELOAD = 0.003       # 足端预压
EXT_MIN, EXT_MAX = 0.35, 0.94


def stance(rig: G.Rig) -> dict:
    """四足按题目放到四个不同高度。FL 第2级 / FR 第1级前沿 / RR 第1级后沿 / RL 地面。"""
    r = rig.r
    x1, z1 = G.TREAD_CX[0], G.TREAD_TOPS[0]        # 第 1 级中心 x / 顶面
    x2, z2 = G.TREAD_CX[1], G.TREAD_TOPS[1]        # 第 2 级
    zf = r + PRELOAD                                # 地面上的足心高度
    feet = {
        "FL": (x2 + 0.136 - r - EDGE, G.BODY_Y + rig.foot_y["FL"], z2 + r + PRELOAD),
        "FR": (x1 + 0.136 - r - EDGE, G.BODY_Y + rig.foot_y["FR"], z1 + r + PRELOAD),
        "RR": (x1 - 0.136 + r + EDGE, G.BODY_Y + rig.foot_y["RR"], z1 + r + PRELOAD),
        "RL": (x1 - 0.136 - 0.15, G.BODY_Y + rig.foot_y["RL"], zf),
    }
    return feet


def poly_margin(pt, poly) -> float:
    return G.margin_in_poly(pt, poly)


def inradius(poly) -> float:
    """凸多边形内切圆半径（网格 + 局部细化）：这是"任何机型都能拿到"的余量上限。"""
    p = G.poly_ccw(poly)
    lo = np.array([p[:, 0].min(), p[:, 1].min()])
    hi = np.array([p[:, 0].max(), p[:, 1].max()])
    best, best_pt = -1e9, None
    for x in np.arange(lo[0], hi[0], 0.01):
        for y in np.arange(lo[1], hi[1], 0.01):
            m = poly_margin((x, y), p)
            if m > best:
                best, best_pt = m, np.array([x, y])
    for step in (0.004, 0.001, 0.0002):
        for dx in np.arange(-2, 2.01, 1):
            for dy in np.arange(-2, 2.01, 1):
                q = best_pt + np.array([dx, dy]) * step
                m = poly_margin(q, p)
                if m > best:
                    best, best_pt = m, q
    return best, best_pt


def solve(rig: G.Rig, feet) -> dict | None:
    """搜重心余量最大的机身位姿（四足可达 + 机身不碰梯级）。"""
    poly = [(feet[k][0], feet[k][1]) for k in G.LEGS]
    xs = np.arange(0.26, 0.54, 0.02)
    ys = np.arange(G.BODY_Y - 0.12, G.BODY_Y + 0.121, 0.02)
    zs = np.arange(0.46, 0.82, 0.02)
    pitches = np.radians(np.arange(-6.0, 25.0, 3.0))
    best = None
    for x in xs:
        for y in ys:
            for z in zs:
                for pitch in pitches:
                    body = (float(x), float(y), float(z), float(pitch))
                    joints, ext = G.legs_ik(rig, body, feet)
                    if joints is None or min(ext.values()) < EXT_MIN or max(ext.values()) > EXT_MAX:
                        continue
                    com, gap = G.pose_metrics(rig, body, joints)
                    if gap < G.BODY_CLEAR:
                        continue
                    m = poly_margin(com, poly)
                    if m < 0.0:
                        continue
                    score = m - 0.02 * abs(pitch)      # 同等余量优先小俯仰
                    if best is None or score > best["score"]:
                        best = {"score": score, "body": body, "joints": joints, "ext": ext,
                                "com": com, "margin": m, "gap": gap}
    return best


def hold(rig: G.Rig, feet_gid: dict, pose: dict, seconds: float) -> dict:
    """PD 保持：机身从零速度开始，只给关节目标角，让物理自己走。"""
    data = rig.data
    G.set_pose(rig, data, pose["body"], pose["joints"])
    p0 = np.array([data.qpos[0], data.qpos[1], data.qpos[2]])
    th0 = G.pitch_of(data)
    f0 = {leg: data.geom_xpos[feet_gid[leg]].copy() for leg in G.LEGS}
    lp = G.Loop(rig)
    tilt = 0.0
    n = int(round(seconds * 50.0))
    for _ in range(n):
        lp.step(pose["joints"])
        tilt = max(tilt, G.tilt_of(data))
    # 末态：每一只脚的实际法向支撑力（证明确实是四点受力）
    force = {leg: 0.0 for leg in G.LEGS}
    touch: dict[str, dict[str, float]] = {leg: {} for leg in G.LEGS}

    def surf(g: int) -> str:
        n = mujoco.mj_id2name(rig.model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        return n

    for i in range(data.ncon):
        c = data.contact[i]
        f = np.zeros(6)
        mujoco.mj_contactForce(rig.model, data, i, f)
        for leg, g in feet_gid.items():
            other = int(c.geom2) if int(c.geom1) == g else (int(c.geom1) if int(c.geom2) == g else None)
            if other is not None:
                force[leg] += abs(float(f[0]))
                k = surf(other)
                touch[leg][k] = touch[leg].get(k, 0.0) + abs(float(f[0]))
    slip = {leg: float(np.linalg.norm(data.geom_xpos[feet_gid[leg]] - f0[leg])) * 1000
            for leg in G.LEGS}
    poly = [(pose["feet"][k][0], pose["feet"][k][1]) for k in G.LEGS]
    com = data.subtree_com[1][:2].copy()
    return {
        "dx": float(data.qpos[0] - p0[0]), "dy": float(data.qpos[1] - p0[1]),
        "dz": float(data.qpos[2] - p0[2]),
        "dpitch": math.degrees(G.pitch_of(data) - th0),
        "tilt": tilt, "slip_mm": slip, "force": force,
        "ncon": int(data.ncon), "touch": touch,
        "margin_actual": poly_margin(com, poly),
        "com": com.tolist(), "feet_z": {l: float(data.geom_xpos[feet_gid[l]][2]) for l in G.LEGS},
        "peak_tau": max(lp.peak.values()), "sat_frac": lp.sat / max(1, lp.steps),
        "fallen": bool(data.qpos[2] < p0[2] - 0.10 or tilt > 45.0),
        "z_final": float(data.qpos[2]),
    }


def render(rig: G.Rig, pose: dict, out: Path) -> None:
    from PIL import Image

    model, data = rig.model, rig.data
    G.set_pose(rig, data, pose["body"], pose["joints"])
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    shots = []
    for azim in (100.0, 34.0):
        r = mujoco.Renderer(model, 560, 800)
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = azim, -10.0, 2.0
        cam.lookat[:] = (0.40, G.BODY_Y, 0.36)
        r.update_scene(data, camera=cam, scene_option=opt)
        shots.append(Image.fromarray(r.render()))
        r.close()
    from PIL import ImageDraw, ImageFont
    bar = 34
    canvas = Image.new("RGB", (800 * 2 + 12, 560 + bar), (18, 18, 20))
    for i, img in enumerate(shots):
        canvas.paste(img, (i * (800 + 12), bar))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(18)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((8, 8), "B2 staggered stance: FL=step2 | FR=step1 | RR=step1 | RL=ground"
                       "   (left: side view, right: 3/4 view)",
              fill=(235, 235, 235), font=font)
    canvas.save(out)


def main() -> int:
    argv = sys.argv[1:]
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    feet = stance(rig)
    poly = [(feet[k][0], feet[k][1]) for k in G.LEGS]
    pose = solve(rig, feet)
    if pose is None:
        print("这个姿态下找不到可行机身位姿")
        return 1
    pose["feet"] = feet

    if "--view" in argv:
        return view(rig, fg, pose)

    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    ir, ir_pt = inradius(poly)
    res = hold(rig, fg, pose, seconds=6.0)
    bx, by, bz, bp = pose["body"]

    say("=" * 92)
    say("B2 极限错位姿态：四足分踩三个高度 —— FL 第2级 / FR+RR 同在第1级 / RL 地面")
    say("=" * 92)
    say("  足端落点（世界坐标，z 是足心，接触点 = z − r）：")
    for leg in G.LEGS:
        fx, fy, fz = feet[leg]
        surface = {0: "地面", 1: "第1级", 2: "第2级"}[
            {"FL": 2, "FR": 1, "RR": 1, "RL": 0}[leg]]
        say(f"    {leg}  {surface:<5} x {fx:+.3f}  y {fy:+.3f}  z {fz:.3f}")
    r = rig.r
    t1_lo, t1_hi = G.TREAD_CX[0] - G.HALF, G.TREAD_CX[0] + G.HALF
    say(f"  第 1 级踏面 x {t1_lo:.4f}..{t1_hi:.4f}（深 {t1_hi-t1_lo:.3f} m）；"
        f"FR 足心 {feet['FR'][0]:.3f}（足印 {feet['FR'][0]-r:.3f}..{feet['FR'][0]+r:.3f}），"
        f"RR 足心 {feet['RR'][0]:.3f}（足印 {feet['RR'][0]-r:.3f}..{feet['RR'][0]+r:.3f}）")
    say(f"    → 两只脚心相距 {feet['FR'][0]-feet['RR'][0]:.3f} m，两枚足印之间净空 "
        f"{feet['FR'][0]-feet['RR'][0]-2*r:.3f} m，都在同一级内 "
        f"({'能' if feet['FR'][0]+r < t1_hi and feet['RR'][0]-r > t1_lo else '不能'}同时放下)")
    say(f"  支撑多边形（水平投影）x {poly_min(poly, 0):.3f}..{poly_max(poly, 0):.3f}，"
        f"y {poly_min(poly, 1):.3f}..{poly_max(poly, 1):.3f}")
    say(f"  该四边形的内切圆半径 {ir*1000:.1f} mm（内切圆心 x {ir_pt[0]:.3f} y {ir_pt[1]:.3f}）"
        f"—— 这是任何摆法都拿不到更多的余量上限")
    say()
    say("  最优机身位姿（网格搜索最大化余量）：")
    say(f"    x {bx:.3f}  y {by:.3f}  z {bz:.3f}  pitch {math.degrees(bp):+.1f}°"
        f"  （脚心形心 x {np.mean([p[0] for p in poly]):.3f} y {np.mean([p[1] for p in poly]):.3f}）")
    say(f"    四腿伸展 {min(pose['ext'].values())*100:.0f}–{max(pose['ext'].values())*100:.0f}%"
        f"｜机身离梯级 {pose['gap']*1000:.0f} mm｜IK-FK 残差 {G.fk_residual(rig, pose['body'], feet, fg)*1000:.1f} mm")
    say(f"    重心投影 ({pose['com'][0]:.3f}, {pose['com'][1]:.3f})，"
        f"到支撑多边形边界最小距离 = 重心余量 {pose['margin']*1000:+.1f} mm")
    say()
    say("  静置 6 s 实测（PD 800/20，零初速）：")
    say(f"    机身漂移 Δx {res['dx']*1000:+.1f} mm  Δy {res['dy']*1000:+.1f} mm  Δz {res['dz']*1000:+.1f} mm"
        f"  Δpitch {res['dpitch']:+.2f}°")
    say(f"    最大倾角 {res['tilt']:.1f}°，末态机身 z {res['z_final']:.3f}"
        f"｜{'倾覆' if res['fallen'] else '未倾覆'}")
    slips = "  ".join(f"{l} {res['slip_mm'][l]:.1f}" for l in G.LEGS)
    force = "  ".join(f"{l} {res['force'][l]:.0f}N" for l in G.LEGS)
    say(f"    足端总位移（含足心高度变化）{slips} mm")
    say(f"    末态接触对 {res['ncon']} 个，各足法向支撑力：{force}")
    touch = "  ".join(
        f"{l}→" + ("/".join(res["touch"][l].keys()) if res["touch"][l] else "悬空")
        for l in G.LEGS)
    say(f"    各足实际压在（直接读接触对，不是按坐标推的）：{touch}")
    say(f"    末态重心余量（用实测重心重算）{res['margin_actual']*1000:+.1f} mm")
    say(f"    力矩峰值占额 {res['peak_tau']*100:.0f}%，饱和步数占比 {res['sat_frac']*100:.1f}%")
    say()
    ok = (not res["fallen"]) and res["margin_actual"] > 0.02 and min(res["force"].values()) > 20.0
    say(f"  判定：{'这个姿态在物理上站得住' if ok else '站不住/已经临界'}"
        f"（余量 {res['margin_actual']*1000:.0f} mm，"
        f"最小支撑力 {min(res['force'].values()):.0f} N）")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"feet": feet, "body": list(pose["body"]),
                                    "margin": pose["margin"], "inradius": ir,
                                    "hold": res}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    render(rig, pose, OUT_PNG)
    print("WROTE", OUT_TXT)
    print("WROTE", OUT_PNG)
    return 0


def poly_min(poly, axis) -> float:
    return float(min(p[axis] for p in poly))


def poly_max(poly, axis) -> float:
    return float(max(p[axis] for p in poly))


def view(rig: G.Rig, fg, pose) -> int:
    """3D 窗口：摆上姿态、PD 保持到稳定，然后停在窗口里让你自己拖拽。"""
    import mujoco.viewer

    model, data = rig.model, rig.data
    print("B2 极限错位姿态：FL 第2级 / FR 第1级前沿 / RR 第1级后沿 / RL 地面")
    print(f"  重心余量 {pose['margin']*1000:+.0f} mm，伸展 "
          f"{min(pose['ext'].values())*100:.0f}–{max(pose['ext'].values())*100:.0f}%")
    G.set_pose(rig, data, pose["body"], pose["joints"])
    lp = G.Loop(rig)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = (0.40, G.BODY_Y, 0.36)
        viewer.cam.distance = 2.0
        viewer.cam.azimuth = 70.0
        viewer.cam.elevation = -8.0
        sync = G.pacer(viewer)
        for _ in range(int(6.0 * 50)):
            lp.step(pose["joints"])
            sync()
        print(f"  保持 6 s 后：机身 z {data.qpos[2]:.3f}，倾角 {G.tilt_of(data):.1f}°，"
              f"接触 {data.ncon} 个")
        G.hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
