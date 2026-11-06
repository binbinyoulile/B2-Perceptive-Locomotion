"""B2 错位对角姿态：四足分别落在第 3/2/1/0 级踏面上，静态搜索 + PD 实测 + 交叉步试探。

文件目的：
    验证一个"错位对角"站位——FL 在第三级、FR 在第二级、RR 在第一级、RL 在地面，
    四足落差 0.5655 m，机身被迫大角度倾斜。要回答：
      1) 这个姿态存在吗（12 个关节都在限位内、机身与腿不穿模）；
      2) 放下去 2.5 s 站得住吗（kp=800/kd=20，与 B2 既有测试同口径）；
      3) 重心投影在四足四边形内吗、余量多少；
      4) 若站得住，对角两组（FL+RR / FR+RL）交替抬腿能不能往上挪。

输入：
    simulations/mujoco/b2_stairs_static/scene.xml（B2 官方 MJCF + 共用钢梯）
输出：
    experiments/b2/b2_staggered_stance.txt / .json
    experiments/b2/b2_staggered_*.png（3D 渲染截图）
依赖：
    mujoco 3.16、numpy、configs.robot_config、同目录 b2_step_pose（复用 IK 与碰撞体分类）
运行：
    D:\\Python313\\python.exe .\\experiments\\b2\\b2_staggered_stance.py [--render]

关键假设：
    - 四级落点取各级踏面中心（第 0 级取楼梯前地面 x=0.12），足心高度 = 踏面顶 + 足半径。
    - 机身只做绕 y 的俯仰（roll/yaw = 0），左右对称落点使 roll 不必要。
    - 重心用 MuJoCo 的 subtree_com（含全部连杆质量），不是手算近似。

已知限制：
    - 只测静态与"脚本化对角换步"，不是优化/学习出来的步态。
    - 第 0 级落点取 0.12 m 是人为选择，离楼梯越远越省腿长、但重心越往前越难站。
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

from configs.robot_config import B2_ROBOT  # noqa: E402

import mujoco  # noqa: E402

import b2_step_pose as base  # noqa: E402

OUT = ROOT / "experiments" / "b2" / "b2_staggered_stance.txt"
LEGS = ("FL", "FR", "RL", "RR")
BODY_Y = base.BODY_Y
HIP_X = np.array(B2_ROBOT.hip_x_m, dtype=np.float64)
FOOT_R = B2_ROBOT.foot_radius_m
L = B2_ROBOT.thigh_length_m
KP, KD = 800.0, 20.0

# 钢梯（stair_bodies.xml 编译结果）：三级踏面 + 中间平台 + 地面
TREAD = {
    "s1": (0.306570, 0.1950),
    "s2": (0.491832, 0.3803),
    "s3": (0.677094, 0.5655),
    "platform": (0.7934 + 0.10, 0.7900),
}


def feet_target(robot=None):
    """FL 三级 / FR 二级 / RR 一级 / RL 地面。"""
    y = B2_ROBOT.nominal_foot_y_m
    return {
        "FL": (TREAD["s3"][0], BODY_Y + y[0], TREAD["s3"][1] + FOOT_R),
        "FR": (TREAD["s2"][0], BODY_Y + y[1], TREAD["s2"][1] + FOOT_R),
        "RR": (TREAD["s1"][0], BODY_Y + y[3], TREAD["s1"][1] + FOOT_R),
        "RL": (0.120, BODY_Y + y[2], 0.0 + FOOT_R),
    }


class Staggered:
    """按 (pitch, 机身 x, 机身 z) 反解 12 个关节角，并量净空/重心。"""

    def __init__(self, model, data, feet):
        self.model, self.data = model, data
        self.feet = feet
        self.base_geoms = base.classify_base_geoms(model)
        self.shin_geoms = base.classify_leg_geoms(model)
        self.stairs = [
            g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name == "steel_stair_collision"
        ]
        self.fromto = np.zeros(6)

    def dist(self, geoms) -> float:
        best = 1.0
        for g in geoms:
            for s in self.stairs:
                best = min(best, mujoco.mj_geomDistance(self.model, self.data, g, s, 0.8,
                                                        self.fromto))
        return best

    def build(self, pitch_deg: float, bx: float, bz: float) -> dict | None:
        a = math.radians(-pitch_deg)
        c, s = math.cos(a), math.sin(a)
        qpos = np.zeros(self.model.nq)
        qpos[0], qpos[1], qpos[2] = bx, BODY_Y, bz
        qpos[3], qpos[5] = math.cos(a / 2.0), math.sin(a / 2.0)
        joints, reach = {}, {}
        for leg in LEGS:
            fx, _fy, fz = self.feet[leg]
            dxw, dzw = fx - bx, fz - bz
            lx, lz = dxw * c - dzw * s, dxw * s + dzw * c
            hip_x = HIP_X[LEGS.index(leg)]
            sol = base.leg_ik(lx - hip_x, lz)
            if sol is None:
                return None
            thigh, calf = sol
            if not (-2.80 <= calf <= -0.45 and -0.92 <= thigh <= 4.67):
                return None
            joints[leg] = (0.0, thigh, calf)
            reach[leg] = math.hypot(lx - hip_x, lz)
            for jn, val in (("hip", 0.0), ("thigh", thigh), ("calf", calf)):
                j = self.model.joint(f"{leg}_{jn}_joint")
                qpos[self.model.jnt_qposadr[j.id]] = val
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        margin = min(
            min(c[2] + 2.82, -0.43 - c[2], c[1] + 0.94, 4.69 - c[1])
            for c in joints.values()
        )
        return {
            "pitch": pitch_deg, "x": bx, "z": bz, "joints": joints, "reach": reach,
            "margin": margin,
            "d_body": self.dist(self.base_geoms["body"]),
            "d_head": self.dist(self.base_geoms["head"] + self.base_geoms["lidar"]),
            "d_shin": self.dist(self.shin_geoms),
            "com": np.array(self.data.subtree_com[self.model.body("base_link").id]),
        }

    def support(self, com) -> tuple[bool, float, list]:
        """重心到四足四边形最近边的有符号距离（内为正）。

        四足按"绕形心的极角"排序，避免手写顺序把四边形排成自交的蝴蝶结。
        """
        pts = [(self.feet[k][0], self.feet[k][1]) for k in LEGS]
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
        area2 = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            area2 += x1 * y2 - x2 * y1
        if area2 < 0:
            pts.reverse()
        best = 1e9
        for i in range(len(pts)):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            ex, ey = bx - ax, by - ay
            n = math.hypot(ex, ey)
            if n < 1e-9:
                continue
            best = min(best, (ex * (com[1] - ay) - ey * (com[0] - ax)) / n)
        return best > 0, best, pts


def search(st: Staggered) -> list[dict]:
    found = []
    for pitch in np.arange(8.0, 46.0, 2.0):
        for bx in np.arange(0.36, 0.56, 0.02):
            for bz in np.arange(0.46, 0.81, 0.02):
                pose = st.build(float(pitch), float(bx), float(bz))
                if pose is None:
                    continue
                if pose["d_body"] < 0.02 or pose["d_head"] < 0.02 or pose["d_shin"] < 0.005:
                    continue
                if pose["margin"] < 0.10:
                    continue
                inside, m, _ = st.support(pose["com"])
                if not inside or m < 0.015:
                    continue
                pose["com_margin"] = m
                found.append(pose)
    found.sort(key=lambda p: (p["com_margin"], p["margin"]), reverse=True)
    return found


def pd_hold(st: Staggered, pose, seconds=2.5) -> dict:
    model, data = st.model, st.data
    st.build(pose["pitch"], pose["x"], pose["z"])
    target = np.empty(model.nu)
    for a in range(model.nu):
        name = model.actuator(a).name
        leg, part = name.split("_")[0], name.split("_")[1]
        target[a] = pose["joints"][leg][("hip", "thigh", "calf").index(part)]
    feet = [base.foot_geom_id(model, leg) if hasattr(base, "foot_geom_id") else None
            for leg in LEGS]
    # 足端 geom：B2 的足端球在 calf body 下
    feet = []
    for leg in LEGS:
        calf = model.body(f"{leg}_calf").id
        g = next(i for i in range(model.ngeom)
                 if model.geom_bodyid[i] == calf
                 and model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE)
        feet.append(g)
    foot0 = np.array([data.geom_xpos[g].copy() for g in feet])
    zs, tilt = [], []
    t_start = data.time
    for _ in range(int(seconds * 50)):
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            dof = int(model.jnt_dofadr[j])
            q = data.qpos[model.jnt_qposadr[j]]
            tau = KP * (target[a] - q) - KD * data.qvel[dof] + data.qfrc_bias[dof]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = min(hi, max(lo, tau))
        for _ in range(10):
            mujoco.mj_step(model, data)
        zs.append(float(data.qpos[2]))
        q = data.qpos[3:7]
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (q[0] * q[2] - q[3] * q[1])))))
        roll = math.degrees(math.atan2(2.0 * (q[0] * q[1] + q[2] * q[3]),
                                       1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])))
        tilt.append(max(abs(pitch), abs(roll)))
    foot1 = np.array([data.geom_xpos[g].copy() for g in feet])
    tail = np.array(zs[len(zs) // 2:])
    return {
        "z_mean": float(tail.mean()),
        "z_ptp_mm": float((tail.max() - tail.min()) * 1000.0),
        "tilt_max_deg": float(max(tilt)),
        "slip_max_mm": float(np.max(np.linalg.norm(foot1[:, :2] - foot0[:, :2], axis=1)) * 1000.0),
        "feet_z": [round(float(v), 3) for v in foot1[:, 2]],
        "fallen": bool(data.qpos[2] < 0.20 or max(tilt) > 60.0),
        "seconds": seconds,
    }


def _prep_visual(st: Staggered, mesh_alpha: float = 0.0) -> None:
    """统一的渲染处理：楼梯 STL（含栏杆）隐藏、斜梁碰撞体隐藏、踏面染成亮色并强制可见。

    MuJoCo 的渲染器默认只显示 geomgroup 0/1/2，碰撞体在组 3；所以要显式打开组 3。
    另外 geom 一旦绑定了 material，geom_rgba 会被忽略，必须先把 geom_matid 断开。
    """
    model = st.model
    for g in range(model.ngeom):
        name = model.geom(g).name
        body = model.body(model.geom_bodyid[g]).name
        if body == "steel_stair_collision" and name not in (
                "up_step_1", "up_step_2", "up_step_3", "platform"):
            model.geom_group[g] = 0
        if name in ("up_step_1", "up_step_2", "up_step_3"):
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [0.25, 0.70, 1.00, 1.0]
        elif name == "platform":
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [0.35, 0.45, 0.55, 1.0]
        elif name == "floor":
            model.geom_group[g] = 1


def render(st: Staggered, pose, out_dir: Path) -> list[str]:
    from PIL import Image

    st.build(pose["pitch"], pose["x"], pose["z"])
    _prep_visual(st)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0   # 隐藏楼梯视觉 STL（含栏杆）
    opt.geomgroup[3] = 1   # 打开碰撞踏面
    renderer = mujoco.Renderer(st.model, 720, 1280)
    cam = mujoco.MjvCamera()
    files = []
    views = (("side", 90.0, -6.0, 2.35), ("iso", 138.0, -20.0, 2.4),
             ("front", 203.0, -14.0, 2.2))
    for name, azim, elev, dist in views:
        cam.lookat[:] = (0.40, BODY_Y, 0.38)
        cam.distance = dist
        cam.azimuth = azim
        cam.elevation = elev
        renderer.update_scene(st.data, camera=cam, scene_option=opt)
        path = out_dir / f"b2_staggered_{name}.png"
        Image.fromarray(renderer.render()).save(path)
        files.append(str(path))
    renderer.close()
    return files


def joint_target_from(st: Staggered, pose, feet) -> tuple[np.ndarray, dict] | tuple[None, dict]:
    """在固定机身位姿下，为给定四足落点求 12 维关节目标；不可达返回 None。"""
    a = math.radians(-pose["pitch"])
    c, s = math.cos(a), math.sin(a)
    bx, bz = pose["x"], pose["z"]
    joints = {}
    for leg in LEGS:
        fx, _fy, fz = feet[leg]
        dxw, dzw = fx - bx, fz - bz
        lx, lz = dxw * c - dzw * s, dxw * s + dzw * c
        hip_x = HIP_X[LEGS.index(leg)]
        sol = base.leg_ik(lx - hip_x, lz)
        if sol is None:
            return None, joints
        joints[leg] = (0.0, sol[0], sol[1])
    return joints, joints


def cross_step(st: Staggered, pose, seconds_per_phase=1.6) -> dict:
    """对角两组交替抬腿：A=FL+RR，B=FR+RL。每组各上一级。"""
    model, data = st.model, st.data
    st.build(pose["pitch"], pose["x"], pose["z"])
    # 当前关节目标
    cur = {leg: (0.0, *pose["joints"][leg][1:]) for leg in LEGS}
    feet = {k: np.array(v, dtype=float) for k, v in st.feet.items()}
    target = np.empty(model.nu)

    def to_ctrl(joints_by_leg) -> np.ndarray:
        for a in range(model.nu):
            name = model.actuator(a).name
            leg, part = name.split("_")[0], name.split("_")[1]
            target[a] = joints_by_leg[leg][("hip", "thigh", "calf").index(part)]
        return target

    def hold(joints_by_leg, seconds):
        tgt = to_ctrl(joints_by_leg)
        for _ in range(int(seconds * 50)):
            for a in range(model.nu):
                j = int(model.actuator_trnid[a, 0])
                dof = int(model.jnt_dofadr[j])
                q = data.qpos[model.jnt_qposadr[j]]
                tau = KP * (tgt[a] - q) - KD * data.qvel[dof] + data.qfrc_bias[dof]
                lo, hi = model.actuator_ctrlrange[a]
                data.ctrl[a] = min(hi, max(lo, tau))
            for _ in range(10):
                mujoco.mj_step(model, data)
        return float(data.qpos[2])

    def swing(legs, target_feet, seconds, lift=0.14):
        """把 legs 的足端从当前位置沿弧线搬到 target_feet，其余腿保持。"""
        src = {leg: feet[leg].copy() for leg in legs}
        dst = {leg: np.array(target_feet[leg], dtype=float) for leg in legs}
        steps = int(seconds * 50)
        for n in range(steps):
            u = (n + 1) / steps
            sm = u * u * (3.0 - 2.0 * u)
            probe_feet = {k: v for k, v in feet.items()}
            for leg in legs:
                pos = src[leg] + (dst[leg] - src[leg]) * sm
                pos[2] += lift * math.sin(math.pi * u)
                probe_feet[leg] = pos
            joints, _ = joint_target_from(st, pose, probe_feet)
            if joints is None:
                return False
            hold(joints, 0.02)
        for leg in legs:
            feet[leg] = dst[leg].copy()
        return True

    log = []
    # A 组：FL → 平台，RR → 二级
    a_feet = {"FL": (0.88, feet["FL"][1], TREAD["platform"][1] + FOOT_R),
              "RR": (TREAD["s2"][0], feet["RR"][1], TREAD["s2"][1] + FOOT_R)}
    ok_a = swing(("FL", "RR"), a_feet, seconds_per_phase)
    log.append(("A 组 FL→平台 / RR→二级", ok_a, float(data.qpos[2]),
                float(data.qpos[0])))
    hold(cur, 0.6)

    # B 组：FR → 三级，RL → 一级
    b_feet = {"FR": (TREAD["s3"][0], feet["FR"][1], TREAD["s3"][1] + FOOT_R),
              "RL": (TREAD["s1"][0], feet["RL"][1], TREAD["s1"][1] + FOOT_R)}
    ok_b = swing(("FR", "RL"), b_feet, seconds_per_phase)
    log.append(("B 组 FR→三级 / RL→一级", ok_b, float(data.qpos[2]),
                float(data.qpos[0])))
    hold(cur, 0.6)
    return {"phases": [{"name": n, "ik_ok": ok, "z": z, "x": x} for n, ok, z, x in log],
            "final_z": float(data.qpos[2]), "final_x": float(data.qpos[0])}


def render_diag(st: Staggered, pose, out_dir: Path) -> list[str]:
    """示意图版：隐藏楼梯视觉 STL（含栏杆），把四级踏面染成亮色、地面保留，四足落点一目了然。"""
    from PIL import Image

    st.build(pose["pitch"], pose["x"], pose["z"])
    model = st.model
    _prep_visual(st)
    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0   # 隐藏楼梯视觉 STL（含栏杆）
    opt.geomgroup[3] = 1   # 碰撞体默认是关的（默认只有 0/1/2 开），这里打开 3 才看得到踏面
    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    files = []
    for name, azim, elev, dist in (("diag", 118.0, -38.0, 2.05), ("diag_iso", 152.0, -32.0, 2.2)):
        cam.lookat[:] = (0.38, BODY_Y, 0.33)
        cam.distance = dist
        cam.azimuth = azim
        cam.elevation = elev
        renderer.update_scene(st.data, camera=cam, scene_option=opt)
        path = out_dir / f"b2_staggered_{name}.png"
        Image.fromarray(renderer.render()).save(path)
        files.append(str(path))
    renderer.close()
    return files


def main() -> int:
    model, data = base.load_scene()
    st = Staggered(model, data, feet_target())
    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    say("=" * 96)
    say("B2 错位对角姿态：FL 三级 / FR 二级 / RR 一级 / RL 地面")
    say("=" * 96)
    say(f"落点：FL x {st.feet['FL'][0]:.3f} z {st.feet['FL'][2]:.3f}；"
        f"FR x {st.feet['FR'][0]:.3f} z {st.feet['FR'][2]:.3f}；"
        f"RR x {st.feet['RR'][0]:.3f} z {st.feet['RR'][2]:.3f}；"
        f"RL x {st.feet['RL'][0]:.3f} z {st.feet['RL'][2]:.3f}")
    say(f"高差：FL−RL = {st.feet['FL'][2]-st.feet['RL'][2]:.3f} m；"
        f"单腿最大伸展 {2*L:.3f} m；kp/kd {KP:.0f}/{KD:.0f}")

    found = search(st)
    say()
    if not found:
        say("搜索无解：三/二/一/零级 + 净空 + 重心余量 + 关节余量四条约束不能同时满足。")
    else:
        best = found[0]
        say(f"搜索：可行姿态 {len(found)} 个；最优 pitch {best['pitch']:.0f}°，"
            f"机身 ({best['x']:.3f}, {BODY_Y:.3f}, {best['z']:.3f})")
        say(f"  关节最小余量 {best['margin']:.2f} rad；净空 机身 {best['d_body']*100:.1f} cm / "
            f"机头 {best['d_head']*100:.1f} cm / 小腿 {best['d_shin']*100:.1f} cm")
        say(f"  重心投影 ({best['com'][0]:.3f}, {best['com'][1]:.3f})，"
            f"到支撑四边形边界 {best['com_margin']*100:+.1f} cm")
        for leg in LEGS:
            reach = best["reach"][leg]
            say(f"  {leg}: thigh {best['joints'][leg][1]:+.3f} / calf {best['joints'][leg][2]:+.3f}，"
                f"伸展 {reach/(2*L)*100:.0f}%")

        hold = pd_hold(st, best)
        say()
        say(f"2.5 s PD 静置：机身 z {hold['z_mean']:.4f}（指令 {best['z']:.4f}，"
            f"下沉 {best['z']-hold['z_mean']:+.3f}），峰峰 {hold['z_ptp_mm']:.2f} mm，"
            f"最大倾角 {hold['tilt_max_deg']:.1f}°，足端滑移 {hold['slip_max_mm']:.2f} mm，"
            f"{'翻倒' if hold['fallen'] else '站住'}")
        say(f"  末态四足足心高度 {hold['feet_z']}（目标 "
            f"{[round(st.feet[k][2],3) for k in LEGS]}）")

        if "--step" in sys.argv:
            say()
            say("对角交叉步试探（A=FL+RR，B=FR+RL，各上一级；机身位姿保持不变）")
            step = cross_step(st, best)
            for ph in step["phases"]:
                say(f"  {ph['name']:<24} IK {'成功' if ph['ik_ok'] else '失败'}，"
                    f"末态 z {ph['z']:.3f}，x {ph['x']:+.3f}")
            say(f"  交叉步结束：z {step['final_z']:.3f}，x {step['final_x']:+.3f}"
                f"（起手 z {best['z']:.3f}，x {best['x']:.3f}）")

        if "--render" in sys.argv:
            files = render(st, best, ROOT / "experiments" / "b2")
            files += render_diag(st, best, ROOT / "experiments" / "b2")
            say("已渲染：" + ", ".join(Path(f).name for f in files))

        OUT.with_suffix(".json").write_text(json.dumps(
            {"best": {k: v for k, v in best.items() if k not in ("joints", "reach", "com")},
             "joints": {k: list(v) for k, v in best["joints"].items()},
             "hold": hold, "n_feasible": len(found)}, ensure_ascii=False, indent=2),
            encoding="utf-8")

    text = "\n".join(lines)
    OUT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
