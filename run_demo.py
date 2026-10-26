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
    D:\Python313\python.exe D:\桌面\机器狗\run_demo.py
    开发验证可追加 --headless --duration 11；不传参数时窗口在接近完成后保持打开。
关键假设：
    1. 12 个关节与 actuator 的顺序沿用仓库中已验证的站立/Trot 基线。
    2. scene.xml 中包含 Go2 的 stairs_home 关键帧和同名钢梯场景。
    3. Trot 是开环步态，机身辅助力和姿态反馈只用于让平地演示可重复。
    4. 机器狗在接触楼梯前停稳，演示结束后不重置、不评价上楼能力。
已知限制：
    开环 Trot 仍有轻微机身晃动；接近楼梯后会用站立目标缓冲停稳。
    当前机身辅助不能直接迁移到真机，后续阶段需要独立的速度、姿态和落脚控制。
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
GO2_STAND_DIR = ROOT / "simulations" / "mujoco" / "go2_stand_flat"
SCENE_DIR = ROOT / "simulations" / "mujoco" / "go2_stairs_static"
SCENE_PATH = SCENE_DIR / "scene.xml"

# 复用已经验证过的站立角、PD 增益和 Trot 目标生成器，确保本入口与其他基线
# 使用同一套关节顺序，避免复制多份关节约定后逐渐不一致。
sys.path.insert(0, str(GO2_STAND_DIR))
from stand_pd import (  # noqa: E402
    KD,
    KP,
    LEG_NAMES,
    Q_TARGET as Q_STAND,
    quaternion_to_euler,
)
from trot_open_loop import generate_joint_targets  # noqa: E402

# 一次性演示：先站立，再 Trot 接近钢梯，最后停稳并保持窗口，不再重播。
STAND_DURATION_S = 1.5
TROT_DURATION_S = 7.0
TROT_PHASE_START_S = STAND_DURATION_S
APPROACH_FINISH_TIME_S = STAND_DURATION_S + TROT_DURATION_S
STOP_SETTLE_DURATION_S = 1.5

# 7 s 的平地 Trot 会让机器狗停在楼梯低端前方。巡航速度过大时会提前碰到
# 第一级踏步，过小时看不到明显接近过程；0.145 m/s 为当前安全值。
CRUISE_FORWARD_SPEED_MPS = 0.145

# 起点与钢梯中心线。起点沿用已验证关键帧的高度，X 坐标后移以保留完整接近过程。
DEMO_START_XYZ_M = np.array((-1.600, 0.468, 0.300), dtype=np.float64)
DEMO_CENTER_Y_M = 0.468

# 机身辅助只负责小范围平衡、保持朝向和稳定前进速度。全部输出硬限幅，
# 避免单帧数值误差把机器狗推倒；这些参数不是强化学习策略或真机控制器。
YAW_KP = 7.5
YAW_KD = 2.2
YAW_TORQUE_LIMIT_NM = 9.0
LATERAL_KP = 4.0
LATERAL_KD = 1.8
LATERAL_FORCE_LIMIT_N = 5.0
ROLL_KP = 1.5
ROLL_KD = 0.12
ROLL_TORQUE_LIMIT_NM = 5.0
PITCH_KP = 1.5
PITCH_KD = 0.12
PITCH_TORQUE_LIMIT_NM = 5.0
FORWARD_SPEED_KP = 8.0
FORWARD_FORCE_LIMIT_N = 4.0

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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
def wrap_angle_rad(angle_rad: float) -> float:
    """把角度归一化到 [-pi, pi)。

    参数：
        angle_rad：任意有限角度，单位 rad。
    返回：
        与输入等价、位于 [-pi, pi) 的角度，单位 rad。
    重要性等级：B（偏航误差必须走最短转向方向）。
    """
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


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
        本次演示从固定起点开始，先保持站立 1.5 s，再生成 Trot 关节目标并提供
        朝向、横向、姿态和前进速度辅助，控制流程只有站立和 Trot 两种状态。
    数据约定：
        data.qpos 前 7 项为机身位置与 w/x/y/z 四元数，后续为 12 个关节角；
        qpos 关节角单位为 rad，data.qvel 对应速度单位为 rad/s 或 m/s。
    生命周期：
        模型加载后创建一次；每次重置只覆盖同一 MjData，不重建模型。
    重要性等级：A（所有关节命令和机身辅助的唯一控制入口）。
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
        self.base_angular_dof_adr = self.base_linear_dof_adr + 3

        self.keyframe_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home"
        )
        if self.keyframe_id < 0:
            raise ValueError("scene.xml 缺少 stairs_home 关键帧")

        self.last_forward_speed_mps = 0.0
        self.last_target_forward_speed_mps = 0.0
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
        self.last_target_forward_speed_mps = 0.0
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

    def apply_body_assist(
        self,
        data: mujoco.MjData,
        target_forward_speed_mps: float,
    ) -> None:
        """施加限幅平衡、朝向和前进辅助力。

        参数：
            data：当前状态。
            target_forward_speed_mps：期望前向速度，单位 m/s。
        算法：
            1. 偏航 PD 保持机身朝向世界 +X，横向 PD 保持钢梯中心线 Y=0.468 m。
            2. roll/pitch PD 抑制 Trot 接触冲击带来的机身摇晃。
            3. 前向速度 P 控制把实际速度拉向巡航目标。
        数值边界：
            所有力矩和力均限幅；增益过大可能振荡，过小则跟不上目标。
        重要性等级：A（当前开环 Trot 能稳定接近楼梯的关键辅助）。
        """
        roll_deg, pitch_deg, yaw_deg, _tilt_deg = quaternion_to_euler(data.qpos[3:7])
        yaw_rad = math.radians(yaw_deg)
        angular_dof = self.base_angular_dof_adr

        yaw_rate = float(data.qvel[angular_dof + 2])
        yaw_error = wrap_angle_rad(-yaw_rad)
        yaw_torque = float(np.clip(
            YAW_KP * yaw_error - YAW_KD * yaw_rate,
            -YAW_TORQUE_LIMIT_NM,
            YAW_TORQUE_LIMIT_NM,
        ))

        y_error = DEMO_CENTER_Y_M - float(data.qpos[1])
        y_velocity = float(data.qvel[self.base_linear_dof_adr + 1])
        lateral_force = float(np.clip(
            LATERAL_KP * y_error - LATERAL_KD * y_velocity,
            -LATERAL_FORCE_LIMIT_N,
            LATERAL_FORCE_LIMIT_N,
        ))

        roll_rate_deg_s = math.degrees(float(data.qvel[angular_dof]))
        pitch_rate_deg_s = math.degrees(float(data.qvel[angular_dof + 1]))
        roll_torque = float(np.clip(
            -ROLL_KP * roll_deg - ROLL_KD * roll_rate_deg_s,
            -ROLL_TORQUE_LIMIT_NM,
            ROLL_TORQUE_LIMIT_NM,
        ))
        pitch_torque = float(np.clip(
            -PITCH_KP * pitch_deg - PITCH_KD * pitch_rate_deg_s,
            -PITCH_TORQUE_LIMIT_NM,
            PITCH_TORQUE_LIMIT_NM,
        ))

        self.last_forward_speed_mps = self.body_forward_speed_mps(data, yaw_rad)
        speed_error = target_forward_speed_mps - self.last_forward_speed_mps
        forward_force = float(np.clip(
            FORWARD_SPEED_KP * speed_error,
            -FORWARD_FORCE_LIMIT_N,
            FORWARD_FORCE_LIMIT_N,
        ))
        forward = np.array((math.cos(yaw_rad), math.sin(yaw_rad), 0.0))

        dof = self.base_linear_dof_adr
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = float(forward_force * forward[0])
        data.qfrc_applied[dof + 1] = float(forward_force * forward[1] + lateral_force)
        data.qfrc_applied[angular_dof] = roll_torque
        data.qfrc_applied[angular_dof + 1] = pitch_torque
        data.qfrc_applied[angular_dof + 2] = yaw_torque
        self.last_target_forward_speed_mps = float(target_forward_speed_mps)

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
            写入 data.ctrl 和 data.qfrc_applied；不推进物理时间。
        重要性等级：A（主循环每步调用的控制入口）。
        """
        phase = self.phase(data)
        if phase in (PHASE_STAND, PHASE_STOPPED):
            # STOPPED 仍使用站立关节目标完成缓冲，使 Trot 迈步姿态落回四足支撑；
            # 缓冲结束后仍持续执行 mj_step，让机器狗在物理世界中稳定站立。
            joint_target = Q_STAND.copy()
            self.last_contact_lift_count = 0
            self.last_forward_speed_mps = 0.0
            self.last_target_forward_speed_mps = 0.0
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

        # 起步 0.8 s 内把前向目标速度从 0 平滑拉到巡航速度，避免第一帧就施加
        # 最大前向力。这里只平滑前进速度，不增加额外等待状态。
        ramp = float(np.clip(gait_time_s / 0.8, 0.0, 1.0))
        target_speed = CRUISE_FORWARD_SPEED_MPS * ramp
        self.apply_body_assist(data, target_speed)
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
            "排查：确认 D:\\Python313\\python.exe 已安装 mujoco 和 numpy，"
            "并确认 scene.xml 与根目录 models/ 资产完整。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())




