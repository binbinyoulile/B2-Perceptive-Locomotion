"""B2 「交替跟进」步态：从错位姿态起步，四步缓慢上攀一级（一次只动一条腿）。

文件目的：
    把用户提的「交替跟进步态」在 0.197/0.272 钢梯上跑出来并检验：
        起步（错位）：FL 第2级 / FR 第1级 / RR 第1级 / RL 地面
        第1步 FR：第1级 → 第2级（与 FL 汇合）
        第2步 RL：地面   → 第1级（与 RR 汇合）
        第3步 FL：第2级 → 第3级
        第4步 RR：第1级 → 第2级（与 FR 汇合）
    四步做完：FL 第3级 / FR+RR 第2级 / RL 第1级 —— 正好是起步姿态整体上移一级，
    所以理论上可以循环爬第 4 级（平台）。
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
    experiments/b2/b2_follow_gait.png —— 四步逐帧拼图（可选，--png）
依赖：mujoco 3.16、numpy；复用
      three_robot_gait_0272（rig / IK / PD 循环 / run_plan / pacer）
      b2_diag_stance（候选枚举 / 非足端避碰 / PD 保持）
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --view        （默认 2 个循环）
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_follow_gait.py --png

关键假设：
    - 级高 0.197、踏面深 0.272；第 4 级按"平台"处理（后沿 x=1.0326，顶 0.788）。
    - 50 Hz PD（kp/kd 800/20）+ 重力前馈；摩擦 μ=1.0。
    - 每步时序：挪重心 0.7 s → 抬腿 1.0 s → 落定 0.4 s（慢，是为了让 PD 跟得上）。
已知限制：
    - 仍是"脚本化轨迹 + 纯关节 PD"，没有机身姿态闭环；跑不通就是这条轨迹在开环下不行。
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

TREAD_DEPTH = 0.272
TREAD_REAR0 = 0.2166                     # 第 1 级后沿
LEVEL_TOP = {0: 0.0, 1: 0.197, 2: 0.394, 3: 0.591, 4: 0.788}
EXT_CAP = 0.95
PITCH_HI = 30.0
EDGE = 0.010
PRELOAD = 0.003

SHIFT_S, LIFT_S, SETTLE_S = 1.0, 1.5, 0.5


def edge_x(level: int) -> float:
    return TREAD_REAR0 + TREAD_DEPTH * (level - 1)


def foot_target(rig: G.Rig, leg: str, level: int):
    """某条腿在第 level 级上的落点（前足贴该级前沿、后足贴后沿）。"""
    y = G.BODY_Y + rig.foot_y[leg]
    if level == 0:
        return (TREAD_REAR0 - 0.15, y, rig.r + PRELOAD)
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
    """在候选里挑位姿：余量不低于最优值 3 cm 的前提下，离上一位姿最近。

    为什么不用"余量最大"：纯关节 PD 没有俯仰闭环，机身抬头越大越容易自己往上翻
    （实测第 3 步按最大余量选出的 27° 抬头，静置时自己后仰到 39°、只剩一条后腿着地）。
    所以宁可用 3 cm 余量换"少动一点"。
    """
    if prev is None:
        return max(cands, key=lambda c: c["margin"])
    top = max(c["margin"] for c in cands)
    sub = [c for c in cands if c["margin"] >= top - tol]
    return min(sub, key=lambda c: (abs(c["body"][0] - prev[0]) + abs(c["body"][2] - prev[2])
                                   + 0.35 * abs(c["body"][3] - prev[3])))


def solve(rig: G.Rig, fg, feet, support, prev=None):
    """细网格搜索（2 cm / 3°）。粗网格快 5 倍，但搜出的位姿在动力学上走不通，别换。"""
    cands = D.enumerate_poses(rig, feet, pitch_lo=0.0, pitch_hi=PITCH_HI, ext_max=EXT_CAP,
                              xs=np.arange(0.20, 0.95, 0.02),
                              ys=np.arange(G.BODY_Y - 0.10, G.BODY_Y + 0.101, 0.02),
                              zs=np.arange(0.46, 1.10, 0.02),
                              support=support)
    cands = D.filter_clear(rig, cands, fg)
    if not cands:
        return None
    return pick(cands, prev)


def build(rig: G.Rig, fg, cycles: int):
    """生成整条轨迹 + 每步的计划信息。返回 (segs, steps, notes)。"""
    levels = start_levels()
    feet = {leg: foot_target(rig, leg, levels[leg]) for leg in G.LEGS}
    body = solve(rig, fg, feet, G.LEGS)
    if body is None:
        return None, [], ["起步姿态无可行解"]
    segs = [G.Seg("起步稳住", 0.6, body["body"], body["body"], dict(feet), dict(feet))]
    steps, notes = [], []
    cur = body
    for c in range(cycles):
        for leg, lv_from, lv_to in steps_of_cycle(c):
            if levels[leg] != lv_from:
                notes.append(f"第 {c+1} 循环 {leg}：期望从 {lv_from} 级起，实际在 {levels[leg]} 级，跳过")
                continue
            support3 = tuple(l for l in G.LEGS if l != leg)
            new_feet = dict(feet)
            new_feet[leg] = foot_target(rig, leg, lv_to)
            # 先解"这条腿已上台"的位姿，再解"挪重心"让机身尽量先到位 ——
            # 把俯仰/升高的需求放在四足着地时完成，抬腿那一段只负责搬脚（否则三足撑不住翻转量）
            lifted = solve(rig, fg, new_feet, support3, prev=cur["body"])
            if lifted is None:
                notes.append(f"第 {c+1} 循环 {leg}：抬腿后无可行位姿（{lv_from}→{lv_to} 级）")
                return segs, steps, notes
            shifted = solve(rig, fg, feet, support3, prev=lifted["body"])
            if shifted is None:
                notes.append(f"第 {c+1} 循环 {leg}：挪重心无可行位姿")
                return segs, steps, notes
            z_clear = max(feet[leg][2], new_feet[leg][2]) + rig.r + 0.03
            segs.append(G.Seg(f"挪重心{leg}", SHIFT_S, cur["body"], shifted["body"],
                              dict(feet), dict(feet)))
            segs.append(G.Seg(f"抬{leg}", LIFT_S, shifted["body"], lifted["body"],
                              dict(feet), dict(new_feet), lift=leg, z_clear=z_clear))
            # 3) 落定
            settled = solve(rig, fg, new_feet, G.LEGS, prev=lifted["body"])
            if settled is None:
                notes.append(f"第 {c+1} 循环 {leg}：落定无可行位姿")
                return segs, steps, notes
            segs.append(G.Seg(f"落定{leg}", SETTLE_S, lifted["body"], settled["body"],
                              dict(new_feet), dict(new_feet)))
            steps.append({"cycle": c + 1, "leg": leg, "from": lv_from, "to": lv_to,
                          "d_shift": [shifted["body"][2] - cur["body"][2],
                                      math.degrees(shifted["body"][3] - cur["body"][3])],
                          "d_lift": [lifted["body"][2] - shifted["body"][2],
                                     math.degrees(lifted["body"][3] - shifted["body"][3])],
                          "margin_shift": shifted["margin"], "margin_lift": lifted["margin"],
                          "margin_settle": settled["margin"],
                          "max_ext": settled["max_ext"],
                          "ext": settled["ext"], "pitch": math.degrees(settled["body"][3]),
                          "body": settled["body"], "leg_clear": settled.get("leg_clear"),
                          "front": settled["body"][2]})
            feet, cur = new_feet, settled
            levels[leg] = lv_to
    segs.append(G.Seg("收尾", 0.5, cur["body"], cur["body"], dict(feet), dict(feet)))
    return segs, steps, notes


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


def main() -> int:
    argv = sys.argv[1:]
    cycles = 1 if ("--cycles" in argv and argv[argv.index("--cycles") + 1] == "1") else 2
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    cached = None if ("--rebuild" in argv or "--view" not in argv) else load_plan(cycles)
    if cached is not None:
        print(f"读缓存计划（{len(cached)} 段）；要重算加 --rebuild（约 1–3 分钟）")
        segs, steps, notes = cached, [], ["（读自缓存，规划表见上一次输出）"]
    else:
        segs, steps, notes = build(rig, fg, cycles)
        if segs is not None:
            save_plan(segs, steps, cycles)
    if segs is None:
        print("无法起步：", notes)
        return 1
    if "--view" in argv:
        return view(rig, fg, segs, steps)

    servo = float(argv[argv.index("--servo") + 1]) if "--servo" in argv else 1.2
    res = G.run_plan(rig, segs, fg, body_servo=servo)
    rig0 = G.load_rig("b2")
    res0 = G.run_plan(rig0, segs, G.foot_gids(rig0), body_servo=0.0)   # 同一条轨迹、纯开环
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 100)
    say("B2「交替跟进」步态：错位姿态起步，四步上一级（一次只动一条腿，前后腿成对汇合）")
    say("=" * 100)
    lv = start_levels()
    say("  起步：" + "  ".join(f"{l} 第{lv[l]}级" if lv[l] else f"{l} 地面" for l in G.LEGS))
    say("  第 c 循环的四步：FR 1→2 / RL 0→1 / FL 2→3 / RR 1→2（第 2 循环整体 +1 级，FL 上平台）")
    say()
    say(f"  规划：{len(steps)} 步可解" + ("，全部 8 步（2 个循环）" if cycles == 2 and len(steps) == 8 else ""))
    say(f"    {'循环':<4}{'腿':<5}{'级':<7}{'机身 x':>7}{'z':>7}{'pitch':>7}"
        f"{'余量':>7}{'伸展':>6}{'挪重心段':>13}{'抬腿段':>13}")
    for st in steps:
        b = st["body"]
        say(f"    {st['cycle']:<4}{st['leg']:<5}{str(st['from']) + '→' + str(st['to']):<7}"
            f"{b[0]:7.3f}{b[2]:7.3f}{st['pitch']:7.1f}{st['margin_settle']*1000:6.0f}mm"
            f"{st['max_ext']*100:5.0f}%"
            f"{st['d_shift'][0]*100:+7.1f}cm/{st['d_shift'][1]:+5.1f}°"
            f"{st['d_lift'][0]*100:+7.1f}cm/{st['d_lift'][1]:+5.1f}°")
    for n in notes:
        say(f"    · {n}")
    say()
    say(f"  实测（PD 800/20 + 机身位姿伺服 k={servo}，每步：挪重心 {SHIFT_S} s → "
        f"抬腿 {LIFT_S} s → 落定 {SETTLE_S} s）：")
    say(f"    {'步骤':<8}{'抬脚高度':>9}{'机身下沉':>9}{'打滑':>8}{'力矩峰值':>9}{'倾角':>6}"
        f"   {'四足所在级（实测）'}")
    for b in res["beats"]:
        lv = " ".join(f"{l}{level_of(rig, b['feet_z'][l])}" for l in G.LEGS)
        say(f"    {b['name']:<8}{b.get('lift', 0)*1000:8.0f}mm"
            f"{b['sink']*1000:+8.0f}mm{b['slip_mm']:7.1f}mm{b.get('peak', 0)*100:8.0f}%"
            f"{b['tilt']:5.0f}°   {lv}")
    say(f"    最吃力的关节：{res['peak_act']}（峰值占额 {res['peak_tau']*100:.0f}%）")
    done = len(res["beats"])
    say(f"    走通 {done}/{len(steps)} 步；"
        f"{'在【' + str(res['failed_at']) + '】失去稳定' if res['fell'] else '全程未失稳'}")
    say(f"    末态机身 x {res['x']:+.3f} z {res['z']:.3f}，最大倾角 {res['max_tilt']:.0f}°")
    say(f"    失稳后的末态足端高度（仅供参考，不是走完四步的状态）"
        + "  ".join(f"{l} {res['feet_z'][l]:.3f}" for l in G.LEGS)
        + f"（一级顶 0.197 / 二级顶 0.394 / 三级顶 0.591 / 平台 0.788）")
    say()
    total = len(segs) and len([x for x in segs if x.lift])
    say(f"  对照（同一条轨迹，只改控制）：无机身伺服走通 {len(res0['beats'])}/{total} 步"
        f"（在 {res0['failed_at'] or '—'} 停），加伺服 k={servo} 走通 {done}/{total} 步。")
    if not res["fell"]:
        say(f"  结论：全部 {total} 步走通，姿态整体上移 {total // 4} 级；下一循环同构，可以一直循环爬。")
    elif done >= 4:
        say(f"  结论：第一循环四步全走通（机身 0.680 → {res['z']:.3f} m，升 {(res['z']-0.680)*100:.0f} cm），"
            f"第 {done+1} 步（第 2 循环 {res['failed_at']}）失稳 —— "
            f"但峰值力矩只有 {res['peak_tau']*100:.0f}%（{res['peak_act']}），不是关节撑不住，是支撑切换时失稳。")
    else:
        say(f"  结论：第 {done} 步之后失稳（{res['failed_at']}）；峰值力矩 {res['peak_tau']*100:.0f}%"
            f"（{res['peak_act']}）")

    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps({"steps": steps, "notes": notes,
                                    "result": {k: v for k, v in res.items() if k != "beats"},
                                    "beats": res["beats"]}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(text)
    print("WROTE", OUT_TXT)
    if "--png" in argv:
        render(rig, fg, segs, steps, res, servo)
    return 0


def render(rig: G.Rig, fg, segs, steps, res, servo: float = 1.2) -> None:
    """四步各抽一帧（每步落定后）拼成 4 联图。"""
    from PIL import Image, ImageDraw, ImageFont

    model, data = rig.model, rig.data
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1
    shots, tags = [], []
    state = {"n": 0}

    def hook(seg, u):
        if seg.name.startswith("落定") and u > 0.97 and state["n"] < len(steps):
            r = mujoco.Renderer(model, 520, 780)
            cam = mujoco.MjvCamera()
            cam.azimuth, cam.elevation, cam.distance = 96.0, -6.0, 2.6
            cam.lookat[:] = (0.55, G.BODY_Y, 0.42)
            r.update_scene(data, camera=cam, scene_option=opt)
            shots.append(Image.fromarray(r.render()))
            r.close()
            tags.append(f"step {state['n']+1}: {seg.name[2:]}")
            state["n"] += 1

    G.run_plan(rig, segs, fg, hook=hook, body_servo=servo)
    if not shots:
        return
    W, H, bar = 780, 520, 34
    canvas = Image.new("RGB", (W * len(shots) + 8 * (len(shots) + 1), H + bar + 16), (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(19)
    except TypeError:
        font = ImageFont.load_default()
    for i, img in enumerate(shots):
        x = 8 + i * (W + 8)
        canvas.paste(img, (x, bar))
        draw.text((x + 4, 8), tags[i], fill=(235, 235, 235), font=font)
    canvas.save(OUT_PNG)
    print("WROTE", OUT_PNG)


def view(rig: G.Rig, fg, segs, steps) -> int:
    """3D 慢动作：0.5 倍速播整条步态，播完停在窗口里。"""
    import mujoco.viewer

    model, data = rig.model, rig.data
    n_lift = len([x for x in segs if x.lift])
    print(f"  慢动作播放（0.5 倍速）：{n_lift} 步（{'→'.join(x.lift for x in segs if x.lift)}），"
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

        sync = G.pacer(viewer, on_frame=follow, hz=25.0)
        res = G.run_plan(rig, segs, fg, sync=sync, body_servo=1.2)
        print(f"  结果：走通 {len(res['beats'])}/{len(steps)} 步，"
              f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '全程未失稳'}")
        G.hold_open(viewer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
