r"""Go2 楼梯阶段 0 单文件实时演示：平地站立、Trot 接近钢梯并停稳保持。

文件目的：
    用一条 Python 命令打开 MuJoCo 3D 窗口，展示官方 Unitree Go2 在完整钢梯前
    完成“稳定站立 -> Trot 向楼梯低端接近 -> 原地停稳站立”的一次性过程。
    当前阶段只演示机器狗和完整钢梯，不执行上楼梯或强化学习。
整体输入：
    simulations/mujoco/go2_stairs_static/scene.xml。
    该场景只显示官方 Go2 和完整钢梯；机器狗初始位置由本文件显式设置。
整体输出：
    实时 MuJoCo 3D 窗口和终端阶段日志。默认不写 CSV、图片或模型文件。
依赖与版本要求：
    Python 3.13、MuJoCo 3.16、NumPy 2.x、Windows 图形桌面及可用 OpenGL。
运行方式：
    python .\run_demo.py
    开发验证可追加 --headless --duration 11；不传参数时窗口在接近完成后保持打开。
关键假设：
    1. 12 个关节与 actuator 的顺序沿用仓库中已验证的站立/Trot 基线。
    2. scene.xml 中包含 Go2 的 stairs_home 关键帧和同名钢梯场景。
    3. 站立和 Trot 都只输出 12 个关节的 PD 力矩，不施加机身外力辅助。
    4. 机器狗在接触楼梯前停稳，演示结束后不重置、不评价上楼能力。
已知限制：
    纯开环 Trot 没有姿态、速度和落脚反馈，可能逐渐跑偏或提前倾倒。
    当前脚本用于展示和测量这一真实限制，不代表已经具备稳定上楼能力。
重要性等级：A（当前阶段的唯一演示入口）。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE_DIR = ROOT / "simulations" / "mujoco" / "go2_stairs_static"
SCENE_PATH = SCENE_DIR / "scene.xml"

# 一次性演示：先站立，再 Trot 接近钢梯，最后停稳并保持窗口，不再重播。
STAND_DURATION_S = 1.5
TROT_DURATION_S = 7.0
TROT_PHASE_START_S = STAND_DURATION_S
APPROACH_FINISH_TIME_S = STAND_DURATION_S + TROT_DURATION_S
STOP_SETTLE_DURATION_S = 1.5

# 起点与钢梯中心线。起点沿用已验证关键帧的高度，X 坐标后移以保留完整接近过程。
DEMO_START_XYZ_M = np.array((-1.600, 0.468, 0.300), dtype=np.float64)

# 若机身在接触冲击中出现明显倾倒，只记录警告并继续保持站立控制，不自动重播。
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0

# 仿真与窗口刷新频率。物理步长保持 0.002 s；窗口约 60 Hz 刷新，并按墙钟
# 限制为约 1 倍速，避免不同电脑上运行快慢完全不可控。
TIMESTEP_EXPECTED_S = 0.002
VIEWER_SYNC_HZ = 60.0
CONSOLE_PRINT_INTERVAL_S = 1.0

# 初始相机同时覆盖起点、机器狗和完整钢梯。若后续场景尺寸变化，应同步调整。
CAMERA_LOOKAT = (1.10, 0.468, 0.62)
CAMERA_DISTANCE_M = 7.0
CAMERA_AZIMUTH_DEG = 108.0
CAMERA_ELEVATION_DEG = -20.0

PHASE_STAND = "STAND"
PHASE_TROT = "TROT"
PHASE_STOPPED = "STOPPED"
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
LEG_NAMES = ("FL", "FR", "RL", "RR")

# 本入口独立保留一份最小站立/开环节律参数，避免依赖已从项目中移除的旧测试脚本。
# Q_STAND 单位 rad；KP/KD 分别为关节位置刚度 N·m/rad 与速度阻尼 N·m·s/rad。
Q_STAND = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float64)
KP = 60.0
KD = 2.0

# 开环 Trot 参数与历史基线一致。频率单位为 Hz，振幅单位为 rad；
# 频率过高会增加接触冲击，振幅过大则可能造成机身横滚和足端横向漂移。
TROT_FREQUENCY_HZ = 2.0
HIP_ABDUCTION_AMPLITUDE_RAD = 0.02
THIGH_AMPLITUDE_RAD = 0.18
KNEE_AMPLITUDE_RAD = 0.14
SWING_HEIGHT_M = 0.045
SWING_KNEE_GAIN_RAD_PER_M = np.array(
    (4.4, 4.4, 11.0, 11.0), dtype=np.float64
)
GAIT_RAMP_S = 0.8
PAIR_PHASE_CYCLES = np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
HIP_SIGNS = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float64)


def quaternion_to_euler(
    quaternion_wxyz: np.ndarray,
) -> tuple[float, float, float, float]:
    """把 MuJoCo ``w x y z`` 四元数转换为 roll/pitch/yaw 和倾角。

    输入 shape=(4,)，顺序 w/x/y/z；输出角度单位均为度。倾角表示机体 Z 轴与
    世界 Z 轴夹角，不受 yaw 影响。输入长度接近 0 或 shape 错误时抛出 ValueError。
    重要性等级：A（姿态日志和倾倒诊断依赖该约定）。
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
    tilt = np.arccos(np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0))
    return tuple(float(np.rad2deg(value)) for value in (roll, pitch, yaw, tilt))


def _smoothstep01(value: float) -> float:
    """把比例裁剪到 0..1，并返回两端导数为零的平滑过渡值。重要性等级：C。"""
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def generate_joint_targets(
    trot_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成指定 Trot 时刻的 12 关节目标、摆动掩码和腿部相位。

    输入 ``trot_time_s`` 为步态开始后的时间，单位 s，必须为非负有限值。
    返回目标 shape=(12,)、掩码 shape=(4,)、相位 shape=(4,)。算法使用
    对角腿相差半周期的正弦目标，并在摆动相额外屈膝抬脚。该函数是开环的，
    不使用机身姿态、速度或接触反馈；输入非法时抛出 ValueError。
    重要性等级：A（实时演示的全部步态目标由此生成）。
    """
    if not np.isfinite(trot_time_s) or trot_time_s < 0.0:
        raise ValueError(f"trot_time_s 必须是非负有限值，实际 {trot_time_s}")
    leg_phase = 2.0 * np.pi * (
        TROT_FREQUENCY_HZ * trot_time_s + PAIR_PHASE_CYCLES
    )
    swing_weight = np.maximum(0.0, -np.sin(leg_phase))
    ramp = _smoothstep01(trot_time_s / GAIT_RAMP_S)
    target = Q_STAND.copy()
    for leg_index in range(len(LEG_NAMES)):
        sine = float(np.sin(leg_phase[leg_index]))
        target[3 * leg_index] = (
            Q_STAND[3 * leg_index]
            + HIP_ABDUCTION_AMPLITUDE_RAD
            * HIP_SIGNS[leg_index]
            * sine
            * ramp
        )
        target[3 * leg_index + 1] = (
            Q_STAND[3 * leg_index + 1]
            + THIGH_AMPLITUDE_RAD * sine * ramp
        )
        target[3 * leg_index + 2] = (
            Q_STAND[3 * leg_index + 2]
            + KNEE_AMPLITUDE_RAD * sine * ramp
            - SWING_KNEE_GAIN_RAD_PER_M[leg_index]
            * SWING_HEIGHT_M
            * swing_weight[leg_index]
            * ramp
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("关节目标包含 NaN 或 Inf")
    return target, swing_weight > 1e-12, leg_phase

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def foot_contact_flags(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[bool, ...]:
    """读取四个足端当前是否接触任意物体。

    参数：
        model：已编译的 Go2 场景。
        data：当前 MuJoCo 状态，须含本步接触结果。
    返回：
        shape=(4,)，顺序 FL/FR/RL/RR 的布尔元组。
    重要性等级：C（终端日志用它显示当前支撑足数量）。
    """
    flags = np.zeros(4, dtype=bool)
    for contact in data.contact[: data.ncon]:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        if geom1 in LEG_NAMES:
            flags[LEG_NAMES.index(geom1)] = True
        if geom2 in LEG_NAMES:
            flags[LEG_NAMES.index(geom2)] = True
    return tuple(bool(value) for value in flags)


class Go2DemoController:
    """管理 Go2 站立与持续 Trot 演示。

    功能：
        本次演示从固定起点开始，先保持站立 1.5 s，再生成开环 Trot 关节目标。
        控制流程只有站立、Trot 和停稳站立三种状态，不向机身穿插外力。
    数据约定：
        data.qpos 前 7 项为机身位置与 w/x/y/z 四元数，后续为 12 个关节角；
        qpos 关节角单位为 rad，data.qvel 对应速度单位为 rad/s 或 m/s。
    生命周期：
        模型加载后创建一次；每次重置只覆盖同一 MjData，不重建模型。
    重要性等级：A（所有关节命令的唯一控制入口）。
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        """校验模型结构并缓存关节、执行器和自由基座索引。

        参数：
            model：scene.xml 编译后的 MuJoCo 模型。
        可能异常：
            物理步长、模型维度、关节、执行器或关键帧不符合约定时抛出 ValueError。
        重要性等级：A（错误模型若继续运行会产生错误关节映射）。
        """
        if abs(float(model.opt.timestep) - TIMESTEP_EXPECTED_S) > 1e-9:
            raise ValueError(
                f"物理步长应为 {TIMESTEP_EXPECTED_S}s，实际 {model.opt.timestep}s"
            )
        if model.nu != 12 or model.nq != 19 or model.nv != 18:
            raise ValueError(
                f"Go2 模型维度异常：nq={model.nq}, nv={model.nv}, nu={model.nu}"
            )

        self.model = model
        self.qpos_adr = np.zeros(12, dtype=np.int32)
        self.dof_adr = np.zeros(12, dtype=np.int32)
        for index, joint_name in enumerate(JOINT_NAMES):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name.removesuffix("_joint")
            )
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"缺少关节或执行器：{joint_name}")
            if int(model.actuator_trnid[actuator_id, 0]) != joint_id:
                raise ValueError(f"执行器 {actuator_id} 未驱动 {joint_name}")
            self.qpos_adr[index] = int(model.jnt_qposadr[joint_id])
            self.dof_adr[index] = int(model.jnt_dofadr[joint_id])

        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
        if base_body_id < 0:
            raise ValueError("缺少名为 base 的 Go2 机身")
        base_joint_id = int(model.body_jntadr[base_body_id])
        if (base_joint_id < 0 or
                int(model.jnt_type[base_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise ValueError("base body 必须由自由关节驱动")
        self.base_linear_dof_adr = int(model.jnt_dofadr[base_joint_id])

        self.keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home"
        )
        if self.keyframe_id < 0:
            raise ValueError("scene.xml 缺少 stairs_home 关键帧")

        self.last_forward_speed_mps = 0.0
        self.last_contact_lift_count = 0

    def reset(self, data: mujoco.MjData) -> None:
        """恢复站立关键帧并把机器狗放到本演示的固定起点。

        参数：
            data：MuJoCo 数据对象，会被原地覆盖。
        副作用：
            清空速度、控制量、外力与接触状态，并调用 mj_forward。
        重要性等级：A（一次性演示必须从固定初始条件启动）。
        """
        mujoco.mj_resetDataKeyframe(self.model, data, self.keyframe_id)
        data.qpos[:3] = DEMO_START_XYZ_M
        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, data)
        self.last_forward_speed_mps = 0.0
        self.last_contact_lift_count = 0

    def phase(self, data: mujoco.MjData) -> str:
        """按本次演示时间返回 STAND、TROT 或 STOPPED。

        参数：
            data：当前状态；data.time 在每次 reset 后从 0 开始。
        返回：
            PHASE_STAND、PHASE_TROT 或 PHASE_STOPPED。
        重要性等级：C。
        """
        elapsed_s = float(data.time)
        if elapsed_s < TROT_PHASE_START_S:
            return PHASE_STAND
        if elapsed_s < APPROACH_FINISH_TIME_S:
            return PHASE_TROT
        return PHASE_STOPPED

    def body_forward_speed_mps(self, data: mujoco.MjData, yaw_rad: float) -> float:
        """计算机身沿当前朝向的前向速度。

        参数：
            data：当前状态。
            yaw_rad：机身偏航角，单位 rad。
        返回：
            世界速度在机身 X 轴上的投影，单位 m/s。
        重要性等级：C（速度反馈和终端日志使用）。
        """
        dof = self.base_linear_dof_adr
        velocity_xyz = np.asarray(data.qvel[dof:dof + 3], dtype=np.float64)
        forward = np.array((math.cos(yaw_rad), math.sin(yaw_rad), 0.0))
        return float(np.dot(velocity_xyz, forward))

    def write_motor_command(self, data: mujoco.MjData, target_rad: np.ndarray) -> None:
        """用关节 PD 计算 12 个 motor 的力矩命令。

        参数：
            data：当前状态。
            target_rad：shape=(12,)，期望关节角，单位 rad。
        算法：
            tau = clip(KP * (q_target - q) - KD * qdot, actuator_ctrlrange)。
        重要性等级：A（所有关节控制的唯一输出函数）。
        """
        q = np.asarray(data.qpos[self.qpos_adr], dtype=np.float64)
        qd = np.asarray(data.qvel[self.dof_adr], dtype=np.float64)
        torque = KP * (target_rad - q) - KD * qd
        low = np.asarray(self.model.actuator_ctrlrange[:, 0], dtype=np.float64)
        high = np.asarray(self.model.actuator_ctrlrange[:, 1], dtype=np.float64)
        data.ctrl[:] = np.clip(torque, low, high)

    def update(self, data: mujoco.MjData) -> tuple[np.ndarray, str]:
        """生成当前物理步的关节目标和阶段。

        参数：
            data：当前状态，须在 mj_step 之前调用。
        返回：
            joint_target_rad shape=(12,)，单位 rad；phase 为控制阶段。
        副作用：
            写入 data.ctrl；不向基座写入 qfrc_applied，也不推进物理时间。
        重要性等级：A（主循环每步调用的控制入口）。
        """
        phase = self.phase(data)
        if phase in (PHASE_STAND, PHASE_STOPPED):
            # STOPPED 仍使用站立关节目标完成缓冲，使 Trot 迈步姿态落回四足支撑；
            # 缓冲结束后仍持续执行 mj_step，让机器狗在物理世界中稳定站立。
            joint_target = Q_STAND.copy()
            self.last_contact_lift_count = 0
            self.last_forward_speed_mps = 0.0
            data.qfrc_applied[:] = 0.0
            self.write_motor_command(data, joint_target)
            return joint_target, phase

        gait_time_s = max(0.0, float(data.time) - TROT_PHASE_START_S)
        joint_target, swing_mask, _leg_phase = generate_joint_targets(gait_time_s)

        # 若计划摆动的足端仍在地面，稍微增加膝关节屈曲，帮助足端离地。
        contacts = foot_contact_flags(self.model, data)
        lift_count = 0
        for leg_index in range(len(LEG_NAMES)):
            if swing_mask[leg_index] and contacts[leg_index]:
                joint_target[3 * leg_index + 2] -= 0.06
                lift_count += 1
        self.last_contact_lift_count = lift_count

        # 只记录实际前向速度供日志观察；该值不参与 Trot 控制，保证开环测试无反馈。
        _roll, _pitch, yaw_deg, _tilt = quaternion_to_euler(data.qpos[3:7])
        self.last_forward_speed_mps = self.body_forward_speed_mps(
            data, math.radians(yaw_deg)
        )
        self.write_motor_command(data, joint_target)
        return joint_target, phase

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析开发验证参数。

    参数：
        argv：None 时读取 sys.argv，否则使用给定参数列表。
    返回：
        argparse.Namespace，包含 headless 和 duration。
    说明：
        最终演示不传参数时始终打开 3D 窗口并在停稳后持续仿真；--headless 仅用于自动验证，
        --duration 用于开发验证时让程序在指定时长后自动关闭。
    重要性等级：C。
    """
    parser = argparse.ArgumentParser(description="Go2 平地站立与接近钢梯实时演示")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开窗口，仅用于自动验证",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        metavar="SECONDS",
        help="开发验证累计仿真时长；默认在停稳后持续运行窗口",
    )
    args = parser.parse_args(argv)
    if args.duration is not None and args.duration <= 0.0:
        parser.error("--duration 必须大于 0")
    return args


def load_model() -> mujoco.MjModel:
    """从场景目录加载 scene.xml。

    返回：
        已编译的 MuJoCo 模型。
    可能异常：
        文件缺失或 XML 编译失败时向 main 传播；中文绝对路径通过切换工作目录规避。
    重要性等级：A。
    """
    if not SCENE_PATH.is_file():
        raise FileNotFoundError(f"场景不存在：{SCENE_PATH}")
    previous_cwd = Path.cwd()
    try:
        os.chdir(SCENE_DIR)
        return mujoco.MjModel.from_xml_path(SCENE_PATH.name)
    finally:
        os.chdir(previous_cwd)


def run_demo(args: argparse.Namespace) -> int:
    """运行站立、Trot 接近钢梯、停稳保持的一次性主循环。

    参数：
        args：parse_args 返回的运行选项。
    返回：
        0 表示窗口正常关闭、用户按 Ctrl+C 或开发验证结束。
    可能异常：
        场景、关节映射、OpenGL 或窗口初始化失败时由 main 打印中文错误并返回 1。
    重要性等级：A（单文件演示的实际主循环）。
    """
    model = load_model()
    data = mujoco.MjData(model)
    controller = Go2DemoController(model)
    controller.reset(data)

    print("=" * 72)
    print("Go2 阶段 0：平地站立 -> Trot 接近钢梯 -> 停稳")
    print("=" * 72)
    print("当前只演示机器狗和完整钢梯：平地站立后进入 Trot 接近。")
    print(f"场景：{SCENE_PATH}")
    print(
        f"模型：nq={model.nq}, nv={model.nv}, nu={model.nu}, "
        f"dt={model.opt.timestep:.4f}s"
    )
    print(
        f"起点：({DEMO_START_XYZ_M[0]:+.2f}, {DEMO_START_XYZ_M[1]:+.2f}, "
        f"{DEMO_START_XYZ_M[2]:.2f})"
    )
    print("模式：3D 实时窗口（停稳后保持打开，关闭窗口或按 Ctrl+C 结束）\n"
          if not args.headless else "模式：headless 自动验证\n")

    phase = controller.phase(data)
    last_phase = phase
    last_console_s = -1.0
    total_sim_time_s = 0.0
    last_sync_wall_s = 0.0
    demo_wall_start_s = time.perf_counter()
    stable_announced = False
    fall_announced = False
    print("[开始] Go2 先站立稳定，再进入 Trot 接近钢梯。")

    viewer_context = (
        contextlib.nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )

    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = CAMERA_LOOKAT
                viewer.cam.distance = CAMERA_DISTANCE_M
                viewer.cam.azimuth = CAMERA_AZIMUTH_DEG
                viewer.cam.elevation = CAMERA_ELEVATION_DEG
                viewer.sync()

            while True:
                if viewer is not None and not viewer.is_running():
                    break

                # 停稳后仍持续推进物理和刷新窗口；此时控制目标是站立姿态，
                # 机器狗会在重力、足端接触和关节 PD 共同作用下稳定不动。
                _joint_target, phase = controller.update(data)
                mujoco.mj_step(model, data)
                total_sim_time_s += float(model.opt.timestep)

                if phase != last_phase:
                    if phase == PHASE_STOPPED:
                        print("[完成] 已抵达钢梯前，切换为原地站立保持。")
                    elif phase == PHASE_TROT:
                        print("[阶段] TROT：开始向钢梯低端接近。")
                    last_phase = phase

                # 倾倒时只提示一次，不重置场景，也不停止物理仿真。
                _roll, _pitch, _yaw, tilt_deg = quaternion_to_euler(data.qpos[3:7])
                fallen = (
                    float(data.qpos[2]) < FALL_HEIGHT_M
                    or float(tilt_deg) > FALL_TILT_DEG
                )
                if fallen and not fall_announced:
                    print("[警告] 检测到机身倾倒；继续保持物理仿真，不自动重播。")
                    fall_announced = True

                if float(data.time) - last_console_s >= CONSOLE_PRINT_INTERVAL_S:
                    last_console_s = float(data.time)
                    _roll, _pitch, yaw_deg, tilt_deg = quaternion_to_euler(
                        data.qpos[3:7]
                    )
                    support_count = sum(foot_contact_flags(model, data))
                    print(
                        f"[t={data.time:4.1f}s | {phase:<7}] "
                        f"位置=({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f}, "
                        f"{data.qpos[2]:.2f}) yaw={yaw_deg:+4.1f}deg "
                        f"倾角={tilt_deg:4.1f}deg 支撑={support_count}/4 "
                        f"v={controller.last_forward_speed_mps:+.3f}m/s"
                    )

                if viewer is not None:
                    now_wall_s = time.perf_counter()
                    if now_wall_s - last_sync_wall_s >= 1.0 / VIEWER_SYNC_HZ:
                        viewer.sync()
                        last_sync_wall_s = now_wall_s

                    # 以墙面时间为锚保持约 1 倍速，让物理时间与窗口播放同步。
                    target_wall_s = demo_wall_start_s + float(data.time)
                    delay_s = target_wall_s - time.perf_counter()
                    if delay_s > 0.0:
                        time.sleep(delay_s)

                if (
                    phase == PHASE_STOPPED
                    and not stable_announced
                    and float(data.time) >= APPROACH_FINISH_TIME_S + STOP_SETTLE_DURATION_S
                ):
                    stable_announced = True
                    print("[保持] Go2 已停稳站立；物理仿真和窗口继续实时运行。")

                # headless 模式在停稳后自动退出；如果显式给出更长 duration，
                # 则继续验证站立保持是否稳定。
                if args.headless and stable_announced:
                    if args.duration is None or float(data.time) >= args.duration:
                        break
                if args.duration is not None and float(data.time) >= args.duration:
                    break
    except KeyboardInterrupt:
        print("\n用户按 Ctrl+C，结束演示。")

    return 0

def main() -> int:
    """统一处理参数、路径和顶层异常。

    返回：
        0 表示正常结束；1 表示场景、控制或图形环境错误。
    重要性等级：A。
    """
    args = parse_args()
    try:
        return run_demo(args)
    except Exception as error:
        print(f"\n[错误] 仿真启动失败：{type(error).__name__}: {error}", file=sys.stderr)
        print(
            "排查：确认当前 Python 环境已安装 mujoco 和 numpy，"
            "并确认 scene.xml 与根目录 models/ 资产完整。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




