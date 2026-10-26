"""在平地上用关节空间 PD 控制维持 Go2 站立，并输出指标和曲线。

文件目的：
    为后续运动控制和强化学习提供一个可复现的“平地站立”基线。控制器直接
    读取 12 个关节的位置/速度，计算 PD 力矩后写入 MuJoCo motor 的 ctrl。
整体输入：
    ``scene.xml`` 中的官方 Go2 模型和地面，以及 ``stand_home`` 关键帧。
整体输出：
    ``stand_metrics.csv``、``stand_summary.json``、``stand_metrics.png``，
    同时在终端打印高度、姿态、关节误差和跌倒判定。
依赖与版本要求：
    Python 3.13、MuJoCo 3.16、NumPy 2.x、matplotlib 3.11。
运行方式：
    ``D:\\Python313\\python.exe D:\\桌面\\机器狗\\simulations\\mujoco\\go2_stand_flat\\stand_pd.py``
关键假设：
    1. 12 个 motor 的顺序与 ``JOINT_NAMES`` 中的关节顺序一致。
    2. 目标姿态使用官方 home 关节角，不包含机身位置/姿态反馈。
    3. 0.5 s 前视为接触建立瞬态，不参与稳定性指标统计。
    4. 仿真步长 0.002 s，每 5 步记录一次，即记录频率 100 Hz。
已知限制：
    这是一个固定关节目标、由接触和重力决定机身高度的保守 PD 基线，不是平衡控制器，
    也没有步态、地形适应或真机执行器延迟/噪声建模。
重要性等级：A（后续站立、步态和强化学习基线入口）。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent
XML_PATH = HERE / "scene.xml"
CSV_PATH = HERE / "stand_metrics.csv"
SUMMARY_PATH = HERE / "stand_summary.json"
FIGURE_PATH = HERE / "stand_metrics.png"

# 关节顺序必须与 go2.xml 中 actuator 的顺序一致。若官方模型更新，先运行脚本
# 的映射断言，不要凭名称猜测后直接写 ctrl。
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
LEG_NAMES = ("FL", "FR", "RL", "RR")
JOINT_LABELS = ("hip", "thigh", "calf")
Q_TARGET = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float64)

# 保守基线增益，只用于建立可复现的站立对照，不作为最终步态参数：
# - kp 太小：腿软、机身下沉、关节误差大；kp 太大：接触冲击和振荡增强。
# - kd 太小：落地后持续振荡；kd 太大：响应迟滞、关节显得“发粘”。
# 若后续要调参，应先固定随机条件和评价指标，再小步扫描，不要根据一次失败大改。
KP = 60.0
KD = 2.0

TIMESTEP = 0.002
DURATION_S = 5.0
SETTLE_S = 0.5
SAMPLE_INTERVAL_STEPS = 5
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def configure_chinese_plot_font() -> None:
    """选择可用的中文字体并修正负号显示。

    功能：
        避免 matplotlib 图表中的中文标题和标签出现方框。
    参数：
        无。
    返回：
        无。
    副作用：
        修改 matplotlib 的全局 rcParams。
    边界：
        如果系统没有中文字体，中文可能仍无法显示，但数值曲线不受影响。
    重要性等级：C。
    """
    matplotlib.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def quaternion_to_euler(quaternion_wxyz: np.ndarray) -> tuple[float, float, float, float]:
    """把 MuJoCo 的 ``w x y z`` 四元数转换为 roll/pitch/yaw 和倾角。

    参数：
        quaternion_wxyz: shape=(4,)，单位四元数，顺序为 w/x/y/z。
    返回：
        ``(roll_deg, pitch_deg, yaw_deg, tilt_deg)``。角度单位为度；
        tilt 是机身 Z 轴相对世界 Z 轴的夹角，不受 yaw 影响。
    可能异常：
        输入 shape 不为 4 或四元数长度接近 0 时抛出 ValueError。
    数值边界：
        转为欧拉角前先归一化，并对 asin/acos 的自变量裁剪到 [-1, 1]，
        避免浮点误差导致 NaN。
    重要性等级：A（姿态指标依赖该坐标约定）。
    """
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"四元数 shape 应为 (4,)，实际 {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("四元数长度接近 0")
    w, x, y, z = quaternion / norm

    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    up_world_z = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    tilt = np.arccos(up_world_z)
    return tuple(float(np.rad2deg(value)) for value in (roll, pitch, yaw, tilt))


def foot_floor_support_count(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    """统计当前与地面接触的 Go2 足端数量。

    参数：
        model: 已编译模型。
        data: 当前仿真状态，需在 mj_step/mj_forward 后读取 contact。
    返回：
        0..4 的整数。
    可能异常：
        geom 名称不存在时不会抛错，只是不计入支撑数。
    副作用：
        无。
    重要性等级：B。
    """
    supported: set[str] = set()
    for contact in data.contact[: data.ncon]:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if geom1 in LEG_NAMES and geom2 == "floor":
            supported.add(geom1)
        elif geom2 in LEG_NAMES and geom1 == "floor":
            supported.add(geom2)
    return len(supported)


def simulate_pd_stand() -> dict[str, Any]:
    """执行固定 5 秒 PD 站立仿真并按 100 Hz 采样。

    返回：
        字典，包含时间、base 位置、欧拉姿态、倾角、12 个关节位置/误差、
        支撑足数量和当步跌倒标志。数组时间轴 shape=(N,)，关节数组 shape=(N, 12)。
    可能异常：
        XML/关键帧缺失、关节或 actuator 映射不符时抛出 AssertionError/RuntimeError。
    副作用：
        只修改内存中的 MuJoCo model/data，不写文件。
    控制公式：
        ``tau_i = clip(kp * (q_target_i - q_i) - kd * qdot_i, ctrlrange_i)``。
        q 单位为 rad，qdot 单位为 rad/s，tau 单位为 N·m。
    跌倒判据：
        base 高度低于 0.15 m，或机身倾角超过 60°。
    重要性等级：A。
    """
    previous_cwd = Path.cwd()
    try:
        os.chdir(HERE)
        model = mujoco.MjModel.from_xml_path(XML_PATH.name)
    finally:
        os.chdir(previous_cwd)

    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand_home")
    if key_id < 0:
        raise RuntimeError("scene.xml 缺少 stand_home 关键帧")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
        dtype=np.int32,
    )
    assert np.all(joint_ids >= 0), f"缺少关节：{JOINT_NAMES}"
    qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])
    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    assert np.array_equal(actuator_joint_ids, joint_ids), (
        "actuator 顺序与 JOINT_NAMES 不一致，不能直接写 data.ctrl"
    )
    assert model.nu == len(JOINT_NAMES) == 12

    mujoco.mj_forward(model, data)
    samples: dict[str, list[Any]] = {
        "time_s": [],
        "base_x_m": [],
        "base_y_m": [],
        "base_z_m": [],
        "roll_deg": [],
        "pitch_deg": [],
        "yaw_deg": [],
        "tilt_deg": [],
        "joint_position_rad": [],
        "joint_error_rad": [],
        "max_abs_joint_error_rad": [],
        "support_feet": [],
        "fallen": [],
    }

    def record() -> None:
        """记录当前仿真状态，供后续 CSV、指标和曲线使用。

        参数：
            无；读取外层函数的 ``model``、``data``、``qpos_adr`` 等局部变量。
        返回：
            无。
        可能异常：
            ``qpos`` 维度或关节映射异常时由 NumPy 索引抛错。
        副作用：
            向 ``samples`` 的各列表追加一帧数据；不直接写文件。
        数据约定：
            时间单位 s、位置单位 m、姿态单位 deg、关节角和误差单位 rad。
        重要性等级：A（所有评测指标都依赖该采样口径）。
        """
        quaternion = np.asarray(data.qpos[3:7], dtype=np.float64)
        roll, pitch, yaw, tilt = quaternion_to_euler(quaternion)
        joint_position = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
        joint_error = Q_TARGET - joint_position
        fallen = bool(data.qpos[2] < FALL_HEIGHT_M or tilt > FALL_TILT_DEG)
        samples["time_s"].append(float(data.time))
        samples["base_x_m"].append(float(data.qpos[0]))
        samples["base_y_m"].append(float(data.qpos[1]))
        samples["base_z_m"].append(float(data.qpos[2]))
        samples["roll_deg"].append(roll)
        samples["pitch_deg"].append(pitch)
        samples["yaw_deg"].append(yaw)
        samples["tilt_deg"].append(tilt)
        samples["joint_position_rad"].append(joint_position.tolist())
        samples["joint_error_rad"].append(joint_error.tolist())
        samples["max_abs_joint_error_rad"].append(float(np.max(np.abs(joint_error))))
        samples["support_feet"].append(foot_floor_support_count(model, data))
        samples["fallen"].append(fallen)

    record()
    total_steps = int(round(DURATION_S / TIMESTEP))
    for step in range(total_steps):
        joint_position = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
        joint_velocity = np.asarray(data.qvel[dof_adr], dtype=np.float64)
        torque = KP * (Q_TARGET - joint_position) - KD * joint_velocity
        data.ctrl[:] = np.clip(
            torque,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(model, data)
        if (step + 1) % SAMPLE_INTERVAL_STEPS == 0:
            record()

    output: dict[str, Any] = {}
    for key, value in samples.items():
        if key in ("joint_position_rad", "joint_error_rad"):
            output[key] = np.asarray(value, dtype=np.float64)
        else:
            output[key] = np.asarray(value, dtype=np.float64)
    output["fallen"] = output["fallen"].astype(bool)
    output["model"] = model
    return output


def compute_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """从完整时间序列计算稳态段指标和跌倒结论。

    参数：
        result: ``simulate_pd_stand`` 返回的时间序列，第二维关节数据 shape=(N,12)。
    返回：
        摘要字典；高度单位为米，姿态单位为度，关节误差单位为弧度。
    可能异常：
        记录样本不足以覆盖 SETTLE_S 时抛出 RuntimeError。
    关键判据：
        指标只统计 ``time_s >= SETTLE_S`` 的样本；最大姿态偏差使用机身倾角。
    重要性等级：A。
    """
    time_s = result["time_s"]
    steady = time_s >= SETTLE_S
    if steady.sum() < 2:
        raise RuntimeError("稳态采样不足，无法计算指标")
    height = result["base_z_m"][steady]
    attitude = result["joint_error_rad"][steady]
    max_joint_error = float(np.max(np.abs(attitude)))
    final_joint_error = float(np.max(np.abs(result["joint_error_rad"][-1])))
    final_rms_joint_error = float(np.sqrt(np.mean(result["joint_error_rad"][-1] ** 2)))
    fallen = bool(np.any(result["fallen"]))
    max_tilt = float(np.max(result["tilt_deg"][steady]))
    max_roll = float(np.max(np.abs(result["roll_deg"][steady])))
    max_pitch = float(np.max(np.abs(result["pitch_deg"][steady])))
    max_yaw = float(np.max(np.abs(result["yaw_deg"][steady])))
    height_fluctuation = float(np.max(height) - np.min(height))
    stable = bool(
        not fallen
        and height_fluctuation < 0.03
        and max_tilt < 10.0
        and result["support_feet"][-1] >= 4
    )
    return {
        "kp_nm_per_rad": KP,
        "kd_nms_per_rad": KD,
        "duration_s": DURATION_S,
        "settle_s": SETTLE_S,
        "record_hz": 1.0 / (TIMESTEP * SAMPLE_INTERVAL_STEPS),
        "initial_height_m": float(result["base_z_m"][0]),
        "final_height_m": float(result["base_z_m"][-1]),
        "mean_height_m": float(np.mean(height)),
        "max_height_fluctuation_m": height_fluctuation,
        "max_attitude_deviation_deg": max_tilt,
        "max_abs_roll_deg": max_roll,
        "max_abs_pitch_deg": max_pitch,
        "max_abs_yaw_deg": max_yaw,
        "max_abs_joint_error_rad": max_joint_error,
        "final_max_abs_joint_error_rad": final_joint_error,
        "final_rms_joint_error_rad": final_rms_joint_error,
        "final_support_feet": int(result["support_feet"][-1]),
        "fallen": fallen,
        "stable": stable,
    }


def save_outputs(result: dict[str, Any], metrics: dict[str, Any]) -> None:
    """保存 CSV、摘要 JSON 和曲线 PNG。

    参数：
        result: 时间序列，shape=(N, ...)。
        metrics: ``compute_metrics`` 得到的摘要。
    返回：
        无。
    可能异常：
        目录无写权限或 matplotlib 字体/后端异常时抛出。
    副作用：
        覆盖写入本目录下的三个固定输出文件。
    重要性等级：A（后续报告和回归比较依赖固定输出路径）。
    """
    fieldnames = [
        "time_s", "base_x_m", "base_y_m", "base_z_m",
        "roll_deg", "pitch_deg", "yaw_deg", "tilt_deg",
        *[f"q_{name}_rad" for name in JOINT_NAMES],
        *[f"error_{name}_rad" for name in JOINT_NAMES],
        "max_abs_joint_error_rad", "support_feet", "fallen",
    ]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, time_value in enumerate(result["time_s"]):
            row: dict[str, Any] = {
                "time_s": float(time_value),
                "base_x_m": float(result["base_x_m"][index]),
                "base_y_m": float(result["base_y_m"][index]),
                "base_z_m": float(result["base_z_m"][index]),
                "roll_deg": float(result["roll_deg"][index]),
                "pitch_deg": float(result["pitch_deg"][index]),
                "yaw_deg": float(result["yaw_deg"][index]),
                "tilt_deg": float(result["tilt_deg"][index]),
                "max_abs_joint_error_rad": float(result["max_abs_joint_error_rad"][index]),
                "support_feet": int(result["support_feet"][index]),
                "fallen": bool(result["fallen"][index]),
            }
            for joint_index, joint_name in enumerate(JOINT_NAMES):
                row[f"q_{joint_name}_rad"] = float(result["joint_position_rad"][index, joint_index])
                row[f"error_{joint_name}_rad"] = float(result["joint_error_rad"][index, joint_index])
            writer.writerow(row)

    SUMMARY_PATH.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    configure_chinese_plot_font()
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True)
    figure.suptitle("Go2 平地 PD 站立基线", fontsize=16)

    axes[0].plot(result["time_s"], result["base_z_m"], color="#1f77b4", linewidth=2)
    axes[0].axvline(SETTLE_S, color="#777777", linestyle="--", label=f"统计起点 {SETTLE_S}s")
    axes[0].set_ylabel("机身高度 (m)")
    axes[0].set_title(
        f"高度波动 {metrics['max_height_fluctuation_m']:.4f} m，最终高度 {metrics['final_height_m']:.4f} m"
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for column, label in (("roll_deg", "roll"), ("pitch_deg", "pitch"), ("tilt_deg", "tilt")):
        axes[1].plot(result["time_s"], result[column], linewidth=1.8, label=label)
    axes[1].axvline(SETTLE_S, color="#777777", linestyle="--")
    axes[1].set_ylabel("姿态角 (deg)")
    axes[1].set_title(f"最大倾角 {metrics['max_attitude_deviation_deg']:.3f} deg")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=3)

    colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    linestyles = ("-", "--", ":")
    for leg_index, leg_name in enumerate(LEG_NAMES):
        for joint_index, joint_label in enumerate(JOINT_LABELS):
            global_index = leg_index * 3 + joint_index
            axes[2].plot(
                result["time_s"],
                result["joint_error_rad"][:, global_index],
                color=colors[leg_index],
                linestyle=linestyles[joint_index],
                linewidth=1.2,
                label=f"{leg_name}-{joint_label}",
            )
    axes[2].axvline(SETTLE_S, color="#777777", linestyle="--")
    axes[2].set_xlabel("时间 (s)")
    axes[2].set_ylabel("关节误差 (rad)")
    axes[2].set_title(f"12 关节误差，稳态最大绝对值 {metrics['max_abs_joint_error_rad']:.4f} rad")
    axes[2].grid(alpha=0.25)
    axes[2].legend(ncol=4, fontsize=8)

    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(FIGURE_PATH, dpi=160)
    plt.close(figure)


def main() -> int:
    """运行 PD 站立仿真、保存结果并打印终端摘要。

    返回：
        0 表示脚本成功完成。即使 ``fallen=True`` 也返回 0，因为脚本工程上
        完成了测量；稳定性由摘要字段表达，不能把“测量完成”误当成“站立成功”。
    副作用：
        写入 CSV、JSON 和 PNG。
    重要性等级：A。
    """
    result = simulate_pd_stand()
    metrics = compute_metrics(result)
    save_outputs(result, metrics)

    print("=" * 72)
    print("B2 Go2 平地站立基线")
    print("=" * 72)
    print(f"PD 增益: kp={KP:.1f} N·m/rad, kd={KD:.1f} N·m·s/rad")
    print(f"时长/统计起点/记录频率: {DURATION_S:.1f}s / {SETTLE_S:.1f}s / 100Hz")
    print(f"初始高度: {metrics['initial_height_m']:.4f} m")
    print(f"最终高度: {metrics['final_height_m']:.4f} m")
    print(f"稳态平均高度: {metrics['mean_height_m']:.4f} m")
    print(f"最大高度波动（稳态）: {metrics['max_height_fluctuation_m']:.4f} m")
    print(
        "最大姿态偏差（稳态）: "
        f"tilt={metrics['max_attitude_deviation_deg']:.3f}°, "
        f"roll={metrics['max_abs_roll_deg']:.3f}°, "
        f"pitch={metrics['max_abs_pitch_deg']:.3f}°, "
        f"yaw={metrics['max_abs_yaw_deg']:.3f}°"
    )
    print(f"稳态最大关节误差: {metrics['max_abs_joint_error_rad']:.4f} rad")
    print(f"最终最大关节误差: {metrics['final_max_abs_joint_error_rad']:.4f} rad")
    print(f"最终足端支撑数: {metrics['final_support_feet']}/4")
    print(f"是否跌倒: {'是' if metrics['fallen'] else '否'}")
    print(f"是否满足站立基线判据: {'是' if metrics['stable'] else '否'}")
    print(f"曲线: {FIGURE_PATH}")
    print(f"原始数据: {CSV_PATH}")
    print(f"摘要: {SUMMARY_PATH}")

    if not metrics["stable"]:
        print("\n可能原因（本次没有自动调参）：")
        print("  - PD 增益或初始高度不合适，接触建立阶段产生了较大下沉/振荡")
        print("  - 足端与地面摩擦、接触软化参数或模型质量参数与真机不一致")
        print("  - 固定关节目标缺少机身姿态反馈，无法抵抗持续倾斜")
        print("  - 仿真步长、执行器限幅或关节阻尼导致跟踪不足")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
