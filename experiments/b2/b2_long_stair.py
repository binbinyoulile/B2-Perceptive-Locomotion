"""B2 在**任意级数**钢梯上爬：每级一个 CLIMB_ONE_STEP，配一次 REALIGN，循环到"再往上就是平台"。

和 b2_follow_gait.py 的关系：
    b2_follow_gait.py 是"3 级楼梯"的完整作业流（平地走近 → 入口 → 14 步爬梯 → 平台前行），
    里面那套已经验证过的机械全都保留不动：位姿搜索 solve / 单腿动作 / 两段式按实测重规划 /
    按「实测状态 + 参数指纹」缓存。这里**不重写**它们，只是把它按"一级"切开循环调用：
        build(cycles=1, base=当前级号, state=实测状态)      ← 一个循环 = 一级 = 4 步
    所以每一级的动作序列（FR+1 → RL+1 → FL+1 → RR+1）、搜索参数、验收口径都和 3 级那条一样。

对应需求：
    1. 场景：StairSpec(n_steps=N) → stair_bodies_xml / b2_stairs_scene_xml 生成 MuJoCo 场景，
       级高/踏面/宽仍是实测的 0.197/0.272/0.81。加载后再用 StairSpec.from_model() 从**编译后的
       模型**反算几何并核对，对不上就直接报错（不靠"我记得场景里是 8 级"）。
    2. CLIMB_ONE_STEP：一个循环四步，四足各上一级、机身升一级。
       主循环每轮从**实测足端高度**反算当前级号（不信计数器），再按这个级号规划；
       直到"再上一级就是 2 m 平台"才退出循环，交给登顶段。
       本项目没有激光雷达，这里用场景几何查询代替 —— 上真机时把这个查询换成高度图即可。
    3. 落点：foot_target() 本来就是几何公式（级号 → x/y/z），没有写死坐标。
    4. REALIGN：每级走完按**规划站位**站定一拍，然后量"实际走的"离"计划的"多远（超限就重新锚定）。
       原本想按需求做"把脚收拢回标准站姿"，但实测两次都是硬掰更危险：
       把俯仰掰回 0° → 下一级支撑足滑移 17→109 mm、峰值 240%，第 5 级翻倒；逐个重摆偏 4–5 cm 的脚
       → 等于在楼梯上多做 1–3 次单腿支撑。真正防累积的是落点本身：它按几何算成**绝对坐标**
       （foot_target(级号)），不是"上一级再加一格"，所以误差不会被带到后面去。

运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_long_stair.py --view            （默认 8 级，3D 播放）
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_long_stair.py                   （数据模式，写 txt/json）
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_long_stair.py --steps 6 --view
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_long_stair.py --no-entry        （从标准状态起步，不走近/入场）

已知限制：
    - 和 3 级那条一样是"脚本轨迹 + 关节 PD + 机身位姿伺服"，没有真正的姿态/偏航闭环；
      偏航只能观察、不能修正（报告里给每级末的 yaw）。
    - "登顶检测"是几何查询（场景由本程序生成，所以知道平台在哪），不是感知。
    - 第一次跑要现规划每一级（约 2 分钟/级），之后按实测状态读缓存（b2_follow_gait_plan_long_*.json）。
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "b2"))
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))
sys.path.insert(0, str(ROOT))

import three_robot_gait_0272 as G  # noqa: E402
import b2_follow_gait as B  # noqa: E402

SCENE_DIR = ROOT / "simulations" / "mujoco" / "b2_stairs_static"
OUT_TXT = ROOT / "experiments" / "b2" / "b2_long_stair.txt"
OUT_JSON = ROOT / "experiments" / "b2" / "b2_long_stair.json"
SERVO = 1.2                    # 机身位姿伺服增益（和 3 级那条一样）
# 每级末尾核对"实际走的"与"计划的"：偏差超过这个值就说明中途打滑/被顶偏了，
# 下一级改用实测状态重新锚定（和入口→爬梯那道边界同一个机制）。实测正常级别是 15–45 mm。
REANCHOR_DEV = 0.060


def cache_path(tag: str) -> Path:
    return ROOT / "experiments" / "b2" / f"b2_follow_gait_plan_long_{tag}.json"


# ---------------------------------------------------------------- 场景
def make_rig(steps: int) -> tuple[G.Rig, G.StairSpec]:
    """按级数生成场景并加载；再用编译后的模型反算几何核对一遍。"""
    spec = G.StairSpec(n_steps=steps)
    rig = G.load_rig_xml("b2", G.b2_stairs_scene_xml(spec), SCENE_DIR)
    got = G.StairSpec.from_model(rig.model)
    if got.key() != spec.key():
        raise SystemExit(f"场景几何与 spec 不一致：\n  spec {spec}\n  模型 {got}")
    B.S = spec                    # 落点/级号/高度：b2_follow_gait 从这里取数
    return rig, spec


# ---------------------------------------------------------------- 实测状态 → 级号
def stance_levels(rig: G.Rig, st: dict) -> dict:
    """把四足足心位置映射回"在哪一级"（容差 3.5 cm）。地面靠 x 反算（假踏面往前数）。"""
    out = {}
    for leg in G.LEGS:
        x, _y, z = st["feet"][leg]
        if abs(z - (rig.r + B.PRELOAD)) < 0.035:
            out[leg] = int(round((x - (B.S.rear0 - 0.15)) / B.S.tread))
            continue
        out[leg] = 99
        for lv in range(1, B.S.top_level + 1):
            if abs(z - (B.S.top(lv) + rig.r + B.PRELOAD)) < 0.035:
                out[leg] = lv
                break
    return out


def base_of(rig: G.Rig, st: dict) -> tuple[int, dict, bool]:
    """当前级号：标准站位是 FL=base+2 / FR=RR=base+1 / RL=base，返回 (base, 各级号, 是否吻合)。

    base 取实测的**最小**级号（也就是左后足那一级）—— 这是"读当前脚下台阶的序号"的正解：
    从实测反算而不是数自己走了几级，中间掉了一级也能看出来。
    """
    lv = stance_levels(rig, st)
    base = min(lv.values())
    want = {leg: base + d for leg, d in (("FL", 2), ("FR", 1), ("RL", 0), ("RR", 1))}
    ok = all(lv[leg] == want[leg] for leg in G.LEGS)
    return base, lv, ok


def top_ahead(spec: G.StairSpec, base: int) -> bool:
    """登顶检测（雷达替身）：再往上一级就是 2 m 平台了就返回 True。"""
    return base + 3 >= spec.top_level


# ---------------------------------------------------------------- CLIMB_ONE_STEP
def plan_entry_long(rig: G.Rig, fg, spec: G.StairSpec, st: dict, use_cache: bool = True):
    """入口段（平地站姿 → 标准状态 S0）：几何只涉及第 0~2 级，所以按局部几何缓存。

    单独一个缓存文件：作业流（--demo）的入口段用的是另一份，共用会互相擦掉、两边都要重算。
    """
    return B.cached_plan(cache_path("entry"), rig, fg, spec.top_level, 0, st, "入口段",
                         {"entry": True, "only_entry": True}, use_cache,
                         stairs=spec.local_key())


def plan_platform_long(rig: G.Rig, fg, spec: G.StairSpec, st: dict, use_cache: bool = True):
    """登顶自然恢复 + 平台前行：平台高度随级数变，所以缓存键要带完整几何与逻辑版本。"""
    return B.cached_plan(cache_path(f"platform_v{B.PLATFORM_VERSION}_h{int(B.PLAT_STAND_OFF * 1000)}"),
                         rig, fg,
                         B.PLAT_CYCLES, 0, st, "平台段",
                         {"platform": True, "only_platform": True}, use_cache,
                         stairs=spec.key())


def climb_one_step(rig: G.Rig, fg, spec: G.StairSpec, base: int, st: dict,
                   use_cache: bool = True):
    """CLIMB_ONE_STEP：走完一级（4 步：FR→RL→FL→RR 各上一级，机身升一级）。

    build(cycles=1, base=base, state=st) 就是这个动作：一个循环正好一级，
    落点是 foot_target(级号) 算出来的**绝对坐标**，位姿由 solve 在给定位姿附近搜。
    st 由调用方决定用哪份状态（见 plan_all：默认用上一级的规划末态接链，
    实测偏离超限时才改用实测状态重新锚定）。
    """
    return B.cached_plan(cache_path(f"l{base:02d}"), rig, fg, 1, base, st,
                         f"第 {base + 1} 级", {}, use_cache, stairs=spec.local_key())


def chain_state(segs) -> dict:
    """一级的规划末态（机身 + 四足），下一级从这里接着规划 —— 和一整条长计划里 `cur` 的接力等价。"""
    last = segs[-1]
    return {"body": tuple(float(v) for v in last.body_b),
            "feet": {k: tuple(float(v) for v in v_) for k, v_ in last.feet_b.items()}}


# ---------------------------------------------------------------- REALIGN
def realign_phase(plan_st: dict, st: dict) -> list:
    """REALIGN：按**规划站位**站定一拍（比例伺服本来就在往那儿顶），同时量实测离多远。

    为什么不做"把脚收拢回标准站姿"的两个硬版本 —— 都实测过，都会把腿别住：
      ① 把俯仰从 12–18° 掰回 0°：下一级支撑足滑移 17 → 109 mm、峰值 240%，第 5 级翻倒；
      ② 把每条偏 4–5 cm 的脚逐一重摆回名义点：等于在楼梯上多做 1–3 次单腿支撑，风险更大。
    落点本来就是按几何算的绝对坐标（foot_target(级号)），下一级会把它拉回来；
    这里只把机身按规划站位站住，再看偏离有多大 —— 超限就说明真打滑了，交给主循环重新锚定。
    """
    body = tuple(float(v) for v in plan_st["body"])
    feet = {k: tuple(float(v) for v in st["feet"][k]) for k in G.LEGS}
    return [G.Seg("站定（REALIGN）", 0.6, body, body, feet, feet)]


def state_dev(a: dict, b: dict) -> float:
    """两份状态之间机身 + 四足的最大线性偏差（米）。用来判断"实际走的"有没有离开"计划的"。"""
    d = max(abs(float(x) - float(y)) for x, y in zip(a["body"], b["body"]))
    for leg in G.LEGS:
        d = max(d, max(abs(float(x) - float(y))
                       for x, y in zip(a["feet"][leg], b["feet"][leg])))
    return d


def foot_residuals(rig: G.Rig, st: dict, levels: dict) -> dict:
    """四足离"该级名义落点"多远（米）。平台上的脚不比（平台落点本来就是前/后沿的约定值）。"""
    out = {}
    for leg in G.LEGS:
        lv = levels[leg]
        if lv > B.S.n_steps:
            out[leg] = None
            continue
        tgt = B.foot_target(rig, leg, lv)
        out[leg] = max(abs(st["feet"][leg][0] - tgt[0]), abs(st["feet"][leg][1] - tgt[1]))
    return out


# ---------------------------------------------------------------- 预演 + 逐段规划
def _run(rig, fg, segs, reset: bool):
    return G.run_plan(rig, segs, fg, body_servo=SERVO, reset=reset)


class _Fell(Exception):
    """预演中途失稳：立刻停下（带着摔倒的机器人继续规划只会得到一堆地板数据）。"""


def plan_all(rig: G.Rig, fg, spec: G.StairSpec, entry: bool = True, use_cache: bool = True,
             phases: list | None = None, ends: list | None = None,
             level_log: list | None = None) -> list[str]:
    """无窗口把整条流程预演一遍：每段都按上一段的**实测**状态规划，返回错误清单（空 = 全通）。

    播画面/跑数据时从起点重跑同一条流程（物理是确定性的），所以这些计划照样成立。
    phases / ends / level_log 由调用方传入并原地填充：中途失稳抛 _Fell 时，已经跑完的段不丢。
    """
    phases = [] if phases is None else phases
    ends = [] if ends is None else ends
    level_log = [] if level_log is None else level_log

    def add(name, segs, steps, notes, reset=False):
        r = _run(rig, fg, segs, reset)
        st = B.read_state(rig, fg)
        phases.append((name, list(segs), list(steps or []), list(notes or [])))
        ends.append(st)
        if r["fell"]:
            raise _Fell(f"{name}：失稳在【{r['failed_at']}】、机身 z {r['z']:.3f}")
        return r, st

    notes0: list[str] = []
    if entry:
        a_segs, a_steps, a_notes = B.approach_plan(rig, fg, use_cache)
        if a_segs is None:
            return ["走近段规划失败"]
        _, st = add("平地走近 8 步", a_segs, a_steps, a_notes, reset=True)
        e_segs, e_steps, e_notes = plan_entry_long(rig, fg, spec, st, use_cache)
        if e_segs is None:
            return ["入口段按实测重规划失败：" + str(e_notes)]
        add("入口 6 步（进入标准状态）", e_segs, e_steps, e_notes)
        notes0.append(f"走近+入口完成：机身 x {B.read_state(rig, fg)['body'][0]:+.3f}")
    else:
        # 直接从标准状态起步：把机器人摆到 S0，再按实测重规划
        feet = {leg: B.foot_target(rig, leg, B.start_levels(0)[leg]) for leg in G.LEGS}
        body = B.solve(rig, fg, feet, G.LEGS)
        if body is None:
            return ["标准状态无可行解"]
        joints, _ = G.legs_ik(rig, body["body"], feet)
        G.set_pose(rig, rig.data, body["body"], joints)
        segs = [G.Seg("起步站稳", 0.6, body["body"], body["body"], dict(feet), dict(feet))]
        add("直接起步（标准状态）", segs, [], [], reset=False)

    st = B.read_state(rig, fg)
    # 规划链：第一级从实测状态起步（= --demo 里"入口结束 → 按实测重规划爬梯"那一步，已验过 20/20）；
    # 之后每一级接上一级的**规划**末态，等价于"一整条长计划"里 cur 的接力 —— 这样位姿序列不会
    # 因为每级都从带下垂/带俯仰的实测位姿重新起头而跳档（实测那样每级都会多一次 FL_thigh 250% 的尖峰）。
    plan_st = {"body": tuple(st["body"]), "feet": {k: tuple(st["feet"][k]) for k in G.LEGS}}
    guard = 0
    while True:
        base, levels, ok = base_of(rig, st)
        if not ok:
            return [
                f"实测站位与标准站位不符：{ {l: levels[l] for l in G.LEGS} }，"
                "不敢继续往上规划（先看上一级是不是有脚打滑掉级）"]
        if top_ahead(spec, base):
            break
        guard += 1
        if guard > spec.n_steps + 1:
            return ["级号循环没有收敛（实测级号一直不到平台前），停下"]
        t0 = time.perf_counter()
        c_segs, c_steps, c_notes = climb_one_step(rig, fg, spec, base, plan_st, use_cache)
        if c_segs is None:
            return [f"第 {base + 1} 级规划失败：" + str(c_notes)]
        x0, z0 = st["body"][0], st["body"][2]
        r, st = add(f"第 {base + 1} 级（CLIMB_ONE_STEP）", c_segs, c_steps, c_notes)
        if not r["beats"]:
            return [f"第 {base + 1} 级规划出来 0 步（四步全被跳过），停下"]
        # ---- 规划链接力 + REALIGN（按规划站位站定一拍）----
        plan_st = chain_state(c_segs)
        add(f"第 {base + 1} 级 REALIGN", realign_phase(plan_st, st), [], [])
        st = B.read_state(rig, fg)
        levels = stance_levels(rig, st)
        res = foot_residuals(rig, st, levels)
        dev = state_dev(st, plan_st)
        reanchored = dev > REANCHOR_DEV
        if reanchored:                          # 实际离开计划太远：下一级改用实测状态，别硬接旧链
            plan_st = {"body": tuple(st["body"]),
                       "feet": {k: tuple(st["feet"][k]) for k in G.LEGS}}
        level_log.append({
            "level": base + 1, "steps": len(r["beats"]), "fell": r["fell"],
            "body_x": [x0, st["body"][0]], "body_z": [z0, st["body"][2]],
            "y": st["body"][1], "yaw": r["beats"][-1]["yaw"] if r["beats"] else 0.0,
            "tilt": r["max_tilt"], "peak": r["peak_tau"], "peak_act": r["peak_act"],
            "dev_mm": round(dev * 1000, 1), "reanchored": reanchored,
            "residual_mm": {l: (None if res[l] is None else round(res[l] * 1000, 1))
                            for l in G.LEGS},
            "levels": {l: levels[l] for l in G.LEGS},
            "plan_s": round(time.perf_counter() - t0, 1)})

    # ---- 登顶尾段：从"平台就在上面"那一级起 3 个循环（含前足在平台上先撑开、后足最后上平台）----
    base2, levels2, ok2 = base_of(rig, st)
    if not ok2:
        return ["登顶前实测站位不符，停下"]
    t_segs, t_steps, t_notes = B.cached_plan(cache_path(f"tail_s{spec.n_steps}"), rig, fg,
                                             spec.top_level - base2, base2, plan_st, "登顶段",
                                             {}, use_cache, stairs=spec.key())
    if t_segs is None:
        return ["登顶段规划失败：" + str(t_notes)]
    tx0, tz0 = st["body"][0], st["body"][2]
    plan_st = chain_state(t_segs)
    r, st = add("登顶：后足补位 + 前足撑开", t_segs, t_steps, t_notes)
    level_log.append({"level": "登顶", "steps": len(r["beats"]), "fell": r["fell"],
                      "body_x": [tx0, st["body"][0]], "body_z": [tz0, st["body"][2]],
                      "y": st["body"][1], "yaw": r["beats"][-1]["yaw"] if r["beats"] else 0.0,
                      "tilt": r["max_tilt"], "peak": r["peak_tau"], "peak_act": r["peak_act"],
                      "dev_mm": round(state_dev(st, plan_st) * 1000, 1), "reanchored": False,
                      "residual_mm": {}, "levels": stance_levels(rig, st),
                      "plan_s": 0.0})

    # ---- 登顶自然恢复（边走边站直）+ 平台前行 ----
    p_segs, p_steps, p_notes = plan_platform_long(rig, fg, spec, st, use_cache)
    if p_segs is None:
        return ["平台段规划失败：" + str(p_notes)]
    add("登顶自然恢复 + 平台前行", p_segs, p_steps, p_notes)
    return []


def view(rig: G.Rig, fg, phases, spec: G.StairSpec) -> int:
    """3D 播放：规划全部在开窗口之前做完，窗口里一路不停地爬完。"""
    import mujoco.viewer

    hz = 25.0 if "--slow" in sys.argv[1:] else 50.0
    model, data = rig.model, rig.data
    n_lift = sum(len([x for x in segs if x.lift]) for _, segs, _, _ in phases)
    print(f"  播放（{hz / 50:.1f} 倍速）：{n_lift} 步、{len(phases)} 段，段与段之间不停；"
          f"目标平台顶 {spec.top(spec.top_level):.3f} m", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                      show_right_ui=False) as viewer:
        viewer.opt.geomgroup[3] = 1
        viewer.cam.lookat[:] = (0.55, G.BODY_Y, 0.40)
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 72.0
        viewer.cam.elevation = -10.0

        def follow():
            viewer.cam.lookat[:] = (float(data.qpos[0]) + 0.05, G.BODY_Y,
                                    float(data.qpos[2]) - 0.12)

        sync = G.pacer(viewer, on_frame=follow, hz=hz)
        res, _, _, _ = B.run_demo_flow(rig, fg, phases, SERVO, sync=sync)
        print(f"  结果：{len(res['beats'])}/{n_lift} 步，"
              f"{'在 ' + str(res['failed_at']) + ' 失稳' if res['fell'] else '全程未失稳'}，"
              f"末态机身 ({res['x']:+.3f}, {res['z']:.3f})"
              + ("，四足都在平台上" if not res["fell"] else ""), flush=True)
        G.hold_open(viewer)
    return 0


def main() -> int:
    argv = sys.argv[1:]
    steps = int(argv[argv.index("--steps") + 1]) if "--steps" in argv else 8
    entry = "--no-entry" not in argv
    rig, spec = make_rig(steps)
    fg = G.foot_gids(rig)
    print(f"场景：{spec}", flush=True)
    print(f"  编译后模型：{rig.model.ngeom} 个 geom、{rig.mass:.3f} kg、楼梯踏板 "
          f"{len(rig.plate_gids)} 块（{spec.n_steps} 级 + 平台）", flush=True)
    phases: list = []
    ends: list = []
    level_log: list = []
    try:
        err = plan_all(rig, fg, spec, entry, "--rebuild" not in argv, phases, ends, level_log)
    except _Fell as e:
        err = [str(e)]
    if not phases:
        print("规划失败：" + str(err))
        return 1
    if "--view" in argv:
        return view(rig, fg, phases, spec)

    log = B.ContactLog(rig, fg)
    total = sum(len([x for x in segs if x.lift]) for _, segs, _, _ in phases)
    res, steps_all, notes_all, run_segs = B.run_demo_flow(
        rig, fg, phases, SERVO, hook=log, expect=ends)
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 104)
    say(f"B2 任意级数钢梯：{spec.n_steps} 级（级高 {spec.rise:.3f}、踏面 {spec.tread:.3f}、"
        f"宽 {spec.width:.3f}），平台顶 {spec.top(spec.top_level):.3f} m")
    say("=" * 104)
    say("  每级 = CLIMB_ONE_STEP（FR+1 → RL+1 → FL+1 → RR+1，四个循环步上升一级）+ REALIGN"
        "（机身回中线 / 量足端残余）")
    say("  每级都从上一级的**实测**状态重新规划（落点是按几何算的绝对坐标，所以误差不累积）")
    say()
    for name, segs, _steps, _notes in phases:
        if not segs:
            continue
        b0, b1 = segs[0].body_a, segs[-1].body_b
        say(f"  {name}：{len([x for x in segs if x.lift])} 步，"
            f"机身 x {b0[0]:+.3f} → {b1[0]:+.3f}、z {b0[2]:.3f} → {b1[2]:.3f}")
    say()
    say(f"  逐级实测（{spec.n_steps} 级 + 登顶；yaw 只是观察值，规划不管偏航）：")
    say(f"    {'级':<5}{'步':>3}{'机身 z':>17}{'横向 y':>8}{'yaw':>7}{'倾角':>6}{'峰值':>7}"
        f"{'耗时':>6}{'偏差':>7}   足端残余 mm（REALIGN 后）")
    for lg in level_log:
        res_mm = lg["residual_mm"]
        rtxt = " ".join(f"{l}{res_mm[l]:.0f}" for l in G.LEGS if res_mm.get(l) is not None)
        say(f"    {str(lg['level']):<5}{lg['steps']:>3}"
            f"{lg['body_z'][0]:8.3f}→{lg['body_z'][1]:.3f}{lg['y']:8.3f}"
            f"{lg['yaw']:6.1f}°{lg['tilt']:5.0f}°{lg['peak'] * 100:6.0f}%{lg['plan_s']:6.1f}s"
            f"{lg['dev_mm']:7.0f}" + ("（已重锚）" if lg["reanchored"] else "")
            + f"   {rtxt}")
    say()
    for n in notes_all:
        say(f"  · {n}")
    for e in err:
        say(f"  · {e}")
    done = len(res["beats"])
    say(f"  走通 {done}/{total} 步；"
        f"{'在【' + str(res['failed_at']) + '】失去稳定' if res['fell'] else '全程未失稳'}；"
        f"峰值力矩 {res['peak_tau'] * 100:.0f}%（{res['peak_act']}）、最大倾角 {res['max_tilt']:.0f}°")
    final = None
    if not res["fell"] and run_segs:
        body_f = tuple(run_segs[-1].body_b)
        feet_f = dict(run_segs[-1].feet_b)
        h = G.hold(rig, fg, feet_f, body_f, 2.0)
        cmap = B.contact_map(rig, fg)
        on_plat = {leg: any(nm == "platform" for nm, _ in cmap.get(leg, [])) for leg in G.LEGS}
        final = {"hold": h, "contact": {k: [[nm, round(n, 1)] for nm, n in v]
                                        for k, v in cmap.items()},
                 "on_platform": on_plat, "body": list(body_f)}
        say()
        say("  末态核验（原地静置 2 s，读 contact 里真实生效的 geom 与法向力）：")
        say(f"    机身 ({body_f[0]:.3f}, {body_f[1]:.3f}, {body_f[2]:.3f})  "
            f"pitch {math.degrees(body_f[3]):+.1f}°")
        if h is not None:
            say(f"    静置：Δx {h['dx'] * 1000:+.0f} / Δy {h['dy'] * 1000:+.0f} / "
                f"Δz {h['dz'] * 1000:+.0f} mm，倾角 {h['tilt']:.1f}°，"
                f"打滑 {h['slip_mm']:.1f} mm，{'未倾覆' if h['tilt'] < 20 else '有倾覆风险'}")
        say("    四足接触：" + "  ".join(
            f"{leg}→{'/'.join(f'{nm} {n:.0f}N' for nm, n in cmap.get(leg, [])) or '无'}"
            for leg in G.LEGS))
        say("    四足都在平台上：" + ("是" if all(on_plat.values()) else
                                     "否（" + ",".join(l for l, v in on_plat.items() if not v) + "）"))
        say()
        arrived = all(on_plat.values()) and done == total
        say(f"  结论：{'爬到第 ' if arrived else ''}{spec.n_steps} 级楼梯顶"
            f"（{spec.top(spec.top_level):.3f} m，{total} 步）"
            + ("，四足都在平台上。" if arrived else f"；只走通 {done}/{total} 步。"))
    text = "\n".join(lines)
    OUT_TXT.write_text(text + "\n", encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {"stairs": {"steps": spec.n_steps, "rise": spec.rise, "tread": spec.tread,
                    "width": spec.width, "rear0": spec.rear0, "plat_len": spec.plat_len,
                    "top": spec.top(spec.top_level)},
         "levels": level_log, "phases": [{"name": n, "steps": len([x for x in s if x.lift]),
                                          "body_a": list(s[0].body_a) if s else None,
                                          "body_b": list(s[-1].body_b) if s else None}
                                         for n, s, _, _ in phases],
         "notes": notes_all + err, "final": final,
         "result": {k: v for k, v in res.items() if k != "beats"}, "beats": res["beats"]},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print("WROTE", OUT_TXT)
    return 0 if (not err and not res["fell"] and done == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
