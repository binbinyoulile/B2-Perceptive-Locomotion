"""B2 平地 PD 站立 + 高摩擦钢梯场景加载测试（不训练、不走路）。

本脚本回答两个问题：
    1) 按 configs/robot_config.B2_ROBOT 固化的参数，B2 在平地上 PD 站立稳不稳
       （机身高度、波动、俯仰/横滚）；
    2) 把站好的 B2 平移进 0.195 m/0.18 m 钢梯场景，不做动作，
       只看会不会打滑、会不会穿模。

判据约定：
    - 站立高度用“机身中心离地”。官网站立尺寸 645 mm 是**总高**（含雷达顶 0.2385 m），
      所以机身中心应在 0.4065 m，总高才是 0.645 m。
    - 打滑 = 足端/机身水平位移；穿模 = 接触穿透深度（contact.dist 为负即互相嵌入）。
    - 场景踏面摩擦 0.9，足端 0.4；MuJoCo 取两者逐元素最大值，所以实际接触 μ=0.9
      （脚本会把 contact.friction 打出来验证）。

依赖：mujoco 3.16、numpy 2.x、configs.robot_config。
运行：D:\\Python313\\python.exe .\\experiments\\b2\\b2_stand_and_load.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from configs.robot_config import B2_ROBOT  # noqa: E402

import os  # noqa: E402

import mujoco  # noqa: E402

FLAT_SCENE = ROOT / "models" / "unitree_b2_mujoco" / "xml" / "scene.xml"
STAIR_SCENE = ROOT / "simulations" / "mujoco" / "b2_stairs_static" / "scene.xml"
REPORT = ROOT / "experiments" / "b2" / "b2_stand_and_load.txt"

LEGS = ("FL", "FR", "RL", "RR")
THIGH = B2_ROBOT.thigh_length_m
CALF = B2_ROBOT.calf_length_m
HIP_X = np.array(B2_ROBOT.hip_x_m, dtype=np.float64)
FOOT_Y = np.array(B2_ROBOT.nominal_foot_y_m, dtype=np.float64)



def load(path: Path):
    """加载模型。MuJoCo 打不开含中文的绝对路径，必须 chdir + 相对名。"""
    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        model = mujoco.MjModel.from_xml_path(path.name)
    finally:
        os.chdir(previous)
    return model, mujoco.MjData(model)


def foot_geom_ids(model) -> list[int]:
    """B2 的 4 个足端球没有名字，按“球 + 半径 0.032”找出来。"""
    ids = [
        g
        for g in range(model.ngeom)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
        and abs(float(model.geom_size[g, 0]) - B2_ROBOT.foot_radius_m) < 1e-9
    ]
    if len(ids) != 4:
        raise RuntimeError(f"足端球数量不是 4 个：{ids}")
    return ids


def set_pose(model, data, base_xyz, joint_rad_per_leg) -> None:
    """把基座位姿 + 每条腿的 (thigh, calf) 写进 qpos（按关节名，不按索引假设）。"""
    data.qpos[:] = 0.0
    data.qpos[0:3] = base_xyz
    data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    for leg in LEGS:
        for joint, angle in zip(("hip", "thigh", "calf"), joint_rad_per_leg[leg]):
            j = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{joint}_joint"
            )
            data.qpos[model.jnt_qposadr[j]] = angle
    data.qvel[:] = 0.0


def ctrl_targets(model, data) -> np.ndarray:
    """返回 12 维关节目标（= 当前 qpos 的关节角，等价于“保持这个姿态、不做动作”）。"""
    target = np.empty(model.nu, dtype=np.float64)
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        target[a] = data.qpos[model.jnt_qposadr[j]]
    return target


def pd_torque(model, data, target: np.ndarray) -> None:
    """位置 PD + 重力/科氏前馈：τ = kp·(target − q) − kd·q̇ + k_ff·qfrc_bias。

    与 fsm_stairs_controller.write_motor_command 同一套算法。前馈是必需的：
    只有 PD 时 kp=160 托不住 69 kg 的深蹲站姿，机身会沉到 0.267 m（实测）。
    """
    kp, kd = B2_ROBOT.kp, B2_ROBOT.kd
    kff = B2_ROBOT.fsm.gravity_feedforward_scale
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        dof = model.jnt_dofadr[j]
        q = data.qpos[model.jnt_qposadr[j]]
        tau = (
            kp * (target[a] - q)
            - kd * data.qvel[dof]
            + kff * data.qfrc_bias[dof]
        )
        lo, hi = model.actuator_ctrlrange[a]
        data.ctrl[a] = min(hi, max(lo, tau))


def run(model, data, seconds: float, target: np.ndarray):
    """按 50 Hz 控制、0.002 s 物理步长跑固定时长，逐控制步记录状态。"""
    control_hz = B2_ROBOT.fsm.control_hz
    substeps = max(1, int(round(1.0 / (control_hz * model.opt.timestep))))
    log = []
    for _ in range(int(seconds * control_hz)):
        pd_torque(model, data, target)
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        w, x, y, z = data.qpos[3:7]
        log.append(
            {
                "t": data.time,
                "base": data.qpos[0:3].copy(),
                # 四元数 (w,x,y,z) → 欧拉角；机身姿态角用度数更直观
                "pitch": float(np.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x)))))),
                "roll": float(np.degrees(math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))),
                "ncon": int(data.ncon),
                "min_dist": float(min(c.dist for c in data.contact[: data.ncon]))
                if data.ncon
                else 0.0,
            }
        )
    return log


def report_flat() -> dict:
    model, data = load(FLAT_SCENE)
    base_z0 = 0.4306  # 悬空站姿 D=0.3986 + 足半径，落地后下沉约 2.5 cm 到 0.4065
    stand = {}
    for i, leg in enumerate(LEGS):
        stand[leg] = (
            float(B2_ROBOT.stand_joint_rad[3 * i]),
            float(B2_ROBOT.stand_joint_rad[3 * i + 1]),
            float(B2_ROBOT.stand_joint_rad[3 * i + 2]),
        )
    set_pose(model, data, (0.0, 0.0, base_z0), stand)
    mujoco.mj_forward(model, data)
    target = ctrl_targets(model, data)
    log = run(model, data, seconds=6.0, target=target)
    tail = [r for r in log if r["t"] >= 4.0]
    steady = [r for r in log if r["t"] >= 5.0]
    z = np.array([r["base"][2] for r in tail])
    return {
        "mass_model": float(model.body_subtreemass[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, B2_ROBOT.base_body_name)
        ]),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_ptp": float(z.max() - z.min()),
        "z_min": float(z.min()),
        "z_max": float(z.max()),
        "pitch_max_deg": float(max(abs(r["pitch"]) for r in tail)),
        "roll_max_deg": float(max(abs(r["roll"]) for r in tail)),
        "x_drift": float(abs(tail[-1]["base"][0] - tail[0]["base"][0])),
        "y_drift": float(abs(tail[-1]["base"][1] - tail[0]["base"][1])),
        # 稳态滑移率：最后 1 s 内的水平位移（mm），用来区分“落地沉降”和“一直滑”
        "x_creep_mm": float(abs(steady[-1]["base"][0] - steady[0]["base"][0]) * 1000.0),
        "ncon": tail[-1]["ncon"],
        "min_dist": min(r["min_dist"] for r in log),
        "settled_qpos": data.qpos.copy(),
        "settled_qvel": data.qvel.copy(),
        "target": target,
    }


def report_loaded(settled_qpos: np.ndarray, target: np.ndarray) -> dict:
    """把“平地站好后”的整机状态搬进钢梯场景，2 s 不做动作，看滑不滑/穿不穿模。

    直接用 keyframe 投放会带落地冲击，那样测到的是瞬态而不是摩擦；所以先平地站稳，
    再连人带指令一起平移过来。目标角必须沿用平地那一套（不能拿当前关节角当目标，
    否则 PD 会去找新的平衡点、再沉 2 cm）。
    """
    model, data = load(STAIR_SCENE)
    feet = foot_geom_ids(model)
    data.qpos[:] = settled_qpos
    data.qpos[0:2] = (-0.90, 0.468)  # 只换水平位置，高度/关节沿用平地站稳后的状态
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    z0 = float(data.qpos[2])
    foot0 = np.array([data.geom_xpos[g].copy() for g in feet])
    log = run(model, data, seconds=2.0, target=target)
    foot1 = np.array([data.geom_xpos[g].copy() for g in feet])
    stair_contacts = sorted(
        {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
            for c in data.contact[: data.ncon]
            for gid in (c.geom1, c.geom2)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").startswith(
                ("up_", "down_", "platform", "steel")
            )
        }
    )
    friction = sorted({tuple(np.round(c.friction[:2], 2)) for c in data.contact[: data.ncon]})
    return {
        "z0": z0,
        "z_end": float(data.qpos[2]),
        "z_mean": float(np.mean([r["base"][2] for r in log])),
        "x_drift_mm": abs(float(data.qpos[0]) + 0.90) * 1000.0,
        "y_drift_mm": abs(float(data.qpos[1]) - 0.468) * 1000.0,
        "slip_mm": (np.linalg.norm(foot1[:, :2] - foot0[:, :2], axis=1) * 1000.0).round(2).tolist(),
        "min_dist": min(r["min_dist"] for r in log),
        "ncon_end": int(data.ncon),
        "stair_contacts": stair_contacts,
        "friction": friction,
        "foot_z": (foot1[:, 2] * 1000.0).round(1).tolist(),
        "pitch_max_deg": float(max(abs(r["pitch"]) for r in log)),
        "roll_max_deg": float(max(abs(r["roll"]) for r in log)),
    }


def main() -> int:
    lines: list[str] = []

    def say(text: str = "") -> None:
        lines.append(text)

    flat = report_flat()
    loaded = report_loaded(flat["settled_qpos"], flat["target"])

    base_z = B2_ROBOT.nominal_stand_height_m + B2_ROBOT.foot_radius_m
    say("=" * 84)
    say("B2（标准版）平地 PD 站立 + 高摩擦钢梯场景加载")
    say("=" * 84)
    say(f"配置来源 configs/robot_config.B2_ROBOT：质量 {B2_ROBOT.nominal_mass_kg:.3f} kg，"
        f"力矩 {B2_ROBOT.joint_torque_nm} N·m，足端摩擦 {B2_ROBOT.foot_friction}，"
        f"kp/kd {B2_ROBOT.kp}/{B2_ROBOT.kd}")
    say(f"模型实测质量 {flat['mass_model']:.4f} kg（应等于配置值）")
    say(f"目标机身高度 {base_z:.4f} m → 总高 {base_z + 0.2385:.4f} m（官网 0.645 m）")
    say()
    say("[1] 平地站立（6 s，最后 3 s 统计）")
    say(f"    机身高度  均值 {flat['z_mean']:.4f} m  标准差 {flat['z_std']*1000:.2f} mm  "
        f"峰峰 {flat['z_ptp']*1000:.2f} mm  [{flat['z_min']:.4f}, {flat['z_max']:.4f}]")
    say(f"    总高      均值 {flat['z_mean']+0.2385:.4f} m")
    say(f"    姿态      |pitch| ≤ {flat['pitch_max_deg']:.2f}°，|roll| ≤ {flat['roll_max_deg']:.2f}°")
    say(f"    水平漂移  Δx {flat['x_drift']*1000:.2f} mm，Δy {flat['y_drift']*1000:.2f} mm"
        f"（最后 1 s 残余 {flat['x_creep_mm']:.2f} mm）")
    say(f"    支撑      接触 {flat['ncon']} 个，最深穿透 {abs(flat['min_dist'])*1000:.2f} mm")
    say(f"    相对目标  高度偏差 {(flat['z_mean']-base_z)*1000:+.1f} mm")
    say()
    say("[2] 钢梯场景加载（先平地站稳，再平移放楼梯前地面，2 s 不做动作）")
    say(f"    高度      机身 z {loaded['z_end']:.4f} m（统计均值 {loaded['z_mean']:.4f}），"
        f"足端 z {loaded['foot_z']} mm")
    say(f"    水平漂移  Δx {loaded['x_drift_mm']:.2f} mm，Δy {loaded['y_drift_mm']:.2f} mm")
    say(f"    足端滑移  " + "，".join(
        f"{leg} {v:.2f} mm" for leg, v in zip(LEGS, loaded['slip_mm'])))
    say(f"    穿透      最深 {abs(loaded['min_dist'])*1000:.2f} mm，末态接触 {loaded['ncon_end']} 个，"
        f"μ={loaded['friction']}")
    say(f"    与钢梯接触 {loaded['stair_contacts'] or '无（站在楼梯前地面）'}")
    say(f"    姿态      |pitch| ≤ {loaded['pitch_max_deg']:.2f}°，|roll| ≤ {loaded['roll_max_deg']:.2f}°")

    text = "\n".join(lines)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
