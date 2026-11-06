"""机器人参数配置：把 Go2 专属常量从环境/步态/观测代码中抽离。

文件目的：
    集中保存“换狗时要改的硬件相关参数”（关节顺序、站立姿态、腿长、质量、
    控制增益、Trot 振幅、雷达安装位、模型维度），让环境、步态、观测、规则
    教师都只读这一份配置，而不是各自硬编码。
整体输入：
    无运行时输入；模块加载时构造只读配置对象。
整体输出：
    ``GO2_ROBOT``（已注册、本仓库在跑）、``B2_ROBOT``（官方数据已填、未注册）、
    ``ROBOT_REGISTRY``；``get_robot(name)``。
依赖与版本要求：
    Python 3.10+、NumPy 2.x。不依赖 MuJoCo（B2/M1 的几何计算脚本也可复用）。
运行方式：
    ``from configs.robot_config import GO2_ROBOT, get_robot``。
关键假设：
    1. 12 个驱动关节按 FL/FR/RL/RR × hip/thigh/calf 排列，与官方 MJCF 执行器
       顺序一致（go2_env 启动时会再校验一次）。
    2. ``stand_joint_rad`` 是“悬空出生姿态”；落地承重后机身会低约 3.3 cm，
       成功判定必须用实测稳定高度，不能直接用这里的数值。
    3. 雷达安装位是 XML 的事实记录，供文档/脚本核对；运行时不改模型。
已知限制：
    Go2 是“已在本仓库验证过”的实例；B2 的几何/质量/力矩/摩擦已按官方 MJCF 固化，
    但步态、FSM 增益、跌倒阈值与足端 geom 名仍是未标定值（接训练前必须重标）。
    M1 没有公开模型，暂不建实例。
重要性等级：A（改错会同时破坏训练、回放与规则教师）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class TrotConfig:
    """Trot 先验与残差参数（步态形状，与具体地形无关）。"""

    frequency_hz: float
    ramp_time_s: float
    warmup_time_s: float
    pair_phase_cycles: Tuple[float, float, float, float]
    hip_signs: Tuple[float, float, float, float]
    hip_amplitude_rad: float
    thigh_amplitude_rad: float
    calf_amplitude_rad: float
    swing_height_m: float
    swing_knee_gain_rad_per_m: Tuple[float, float, float, float]
    residual_scale: float


@dataclass(frozen=True)
class FsmConfig:
    """规则教师（FSM Teacher）专用的控制参数。"""

    control_hz: float
    pd_kp: Tuple[float, ...]
    pd_kd: Tuple[float, ...]
    gravity_feedforward_scale: float
    residual_scale: float


@dataclass(frozen=True)
class RobotConfig:
    """一台四足机器人的完整硬件/控制契约。

    字段分组：
        命名：``name``、关节名、足端 geom 名、机身 body 名、雷达 site 名。
        几何：腿长、足端半径、髋位、雷达安装位、体长/体高、标称站高、
        机头下沿、重心 X 偏移（后四项只用于楼梯静态可行性计算）。
        动力学：标称质量、PD 增益、力矩限位（脚本从 MJCF 读，配置只做记录）。
        契约：``expected_nq/nv/nu``，加载 MJCF 后必须完全一致。
    """

    name: str
    joint_names: Tuple[str, ...]
    foot_geom_names: Tuple[str, ...]
    base_body_name: str
    lidar_site_name: str
    stand_joint_rad: np.ndarray
    thigh_length_m: float
    calf_length_m: float
    foot_radius_m: float
    hip_x_m: Tuple[float, float, float, float]
    hip_y_m: Tuple[float, float, float, float]
    thigh_y_offset_m: Tuple[float, float, float, float]
    nominal_foot_y_m: Tuple[float, float, float, float]
    lidar_mount_pos_m: Tuple[float, float, float]
    body_half_length_m: float
    body_half_height_m: float
    nominal_stand_height_m: float
    head_bottom_m: float
    com_offset_x_m: float
    nominal_mass_kg: float
    joint_torque_nm: Tuple[float, float, float]
    foot_friction: float
    expected_nq: int
    expected_nv: int
    expected_nu: int
    kp: float
    kd: float
    joint_limit_margin_rad: float
    fall_height_m: float
    fall_tilt_rad: float
    trot: TrotConfig
    fsm: FsmConfig

    @property
    def joint_count(self) -> int:
        return len(self.joint_names)

    @property
    def max_reach_m(self) -> float:
        """大腿+小腿完全伸直时的髋—足最大距离，单位 m。"""
        return self.thigh_length_m + self.calf_length_m

    def stand_pose(self) -> np.ndarray:
        """返回 12 维站立关节角（独立副本），单位 rad。"""
        return np.asarray(self.stand_joint_rad, dtype=np.float64).copy()


# Go2 的数值全部来自本仓库已验证过的 MJCF 与评测日志：
#   - 关节顺序/限位：models/unitree_go2/go2.xml
#   - 腿长、足端半径、髋位：同上（thigh/calf 各 0.213 m，足端球半径 0.022 m）
#   - 标称质量：MJCF base 子树合计 15.206 kg
#   - 站立姿态 [0.0, 0.9, -1.8]：出生关键帧与全部已评测实验共用
#   - 雷达安装位：models/unitree_go2/go2_lidar.xml 中 lidar_site pos
GO2_ROBOT = RobotConfig(
    name="go2",
    joint_names=(
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ),
    foot_geom_names=("FL", "FR", "RL", "RR"),
    base_body_name="base",
    lidar_site_name="lidar_site",
    stand_joint_rad=np.tile(np.array((0.0, 0.9, -1.8), dtype=np.float64), 4),
    thigh_length_m=0.213,
    calf_length_m=0.213,
    foot_radius_m=0.022,
    hip_x_m=(0.1934, 0.1934, -0.1934, -0.1934),
    hip_y_m=(0.0465, -0.0465, 0.0465, -0.0465),
    thigh_y_offset_m=(0.0955, -0.0955, 0.0955, -0.0955),
    nominal_foot_y_m=(0.1420, -0.1420, 0.1420, -0.1420),
    lidar_mount_pos_m=(0.32, 0.0, 0.07),
    body_half_length_m=0.1881,
    body_half_height_m=0.057,
    nominal_stand_height_m=0.2577,
    head_bottom_m=-0.107,
    com_offset_x_m=0.0211,
    nominal_mass_kg=15.206,
    # hip/thigh/calf 的力矩上限（Go2 MJCF 执行器 ctrlrange 实测）
    joint_torque_nm=(23.7, 23.7, 45.43),
    foot_friction=0.8,
    expected_nq=19,
    expected_nv=18,
    expected_nu=12,
    kp=60.0,
    kd=2.0,
    joint_limit_margin_rad=0.05,
    fall_height_m=0.16,
    fall_tilt_rad=np.radians(60.0),
    trot=TrotConfig(
        frequency_hz=2.0,
        ramp_time_s=0.8,
        warmup_time_s=0.5,
        pair_phase_cycles=(0.0, 0.5, 0.5, 0.0),
        hip_signs=(1.0, -1.0, 1.0, -1.0),
        hip_amplitude_rad=0.02,
        thigh_amplitude_rad=0.18,
        calf_amplitude_rad=0.14,
        swing_height_m=0.045,
        swing_knee_gain_rad_per_m=(4.4, 4.4, 11.0, 11.0),
        residual_scale=0.08,
    ),
    fsm=FsmConfig(
        control_hz=50.0,
        pd_kp=(130.0, 144.0, 144.0) * 4,
        pd_kd=(2.7, 3.5, 3.5) * 4,
        gravity_feedforward_scale=1.0,
        residual_scale=0.10,
    ),
)

# ---------------------------------------------------------------------------
# Unitree B2（标准纯四足版）
# ---------------------------------------------------------------------------
# 型号核对结论（2026-11-06，详见 docs/B2_VS_M1_REPORT.md 第 1 节）：
#   - 之前引用的 unitree_mujoco/unitree_robots/b2/b2.xml 合计 83.5 kg，与官网
#     B2-W（轮足版）标称 85 kg 吻合、与 B2 标称 60 kg 相差 39%；该文件与同仓库
#     b2w/b2w.xml 的连杆惯量逐项相同，只少了 4 个 wheel body（少 3.67 kg）。
#     即：它其实是“B2-W 去掉轮子”的派生模型，质量/站高都是轮足版口径。
#   - 标准版 B2 的官方模型在 unitree_ros/robots/b2_description_mujoco（URDF 与
#     MuJoCo 两套，连杆惯量一致），本仓库已取到 models/unitree_b2_mujoco/，
#     实测 69.003 kg（不含头/尾；加头 3.69 + 尾 0.795 后为 73.5 kg）。
#   下面的数值全部来自该官方模型 + 官网参数，逐项标了出处。
#
# 未标定项（接入训练前必须重标，不要直接跑）：
#   - 步态形状（trot 振幅/相位）、FSM 的 PD 增益、跌倒阈值、足端 geom 名。
B2_ROBOT = RobotConfig(
    name="b2",
    joint_names=GO2_ROBOT.joint_names,
    # B2 官方 MJCF 的 4 个足端球没有名字；接入时必须先在场景里 name 出来，
    # 否则接触判定只能按 geom id 定位。这里先写 body 名占位并保持顺序一致。
    foot_geom_names=("FL", "FR", "RL", "RR"),
    base_body_name="base_link",
    # 官方模型只有 imu 一个 site，雷达没有 site；坐标同 lidar_mount_pos_m。
    lidar_site_name="imu",
    # 站姿：对称深蹲 (thigh 0.9649, calf −1.9298)，对应悬空 D=0.3986 m。
    # 这是“指令角”（悬空）；落地后 PD 下沉约 2.5 cm，稳定在机身 0.4064 m，
    # 加雷达顶 0.2385 m = 0.6449 m ≈ 官网 645 mm。实测峰峰抖动 0.001 mm。
    # 注：unitree_rl_lab 里 UNITREE_B2_CFG 的 (0.8/1.0, −1.5) 是它家 RL 的默认站姿，
    # 落地后机身 0.526 m（总高 0.765 m），不是官网 645 mm 的那个姿态，故不采用。
    stand_joint_rad=np.tile(
        np.array((0.0, 0.9649, -1.9298), dtype=np.float64), 4
    ),
    thigh_length_m=0.35,
    calf_length_m=0.35,
    foot_radius_m=0.032,
    hip_x_m=(0.3285, 0.3285, -0.3285, -0.3285),
    hip_y_m=(0.072, -0.072, 0.072, -0.072),
    thigh_y_offset_m=(0.11973, -0.11973, 0.11973, -0.11973),
    nominal_foot_y_m=(0.1916, -0.1916, 0.1916, -0.1916),
    lidar_mount_pos_m=(0.34218, 0.0, 0.17851),
    body_half_length_m=0.25,
    body_half_height_m=0.075,
    # 机身中心离地：官网 645 mm 是总高（含雷达顶 0.2385 m），故 0.645 − 0.2385 =
    # 0.4065 m；减足端半径 0.032 得 D=0.3745 m。本仓库标定后的 PD 站姿实测
    # 0.4064 m（D=0.3744），差 1 mm。
    nominal_stand_height_m=0.3745,
    # 机头碰撞盒 pos z=0.005、半高 0.07 → 下沿 −0.065；机身盒下沿 −0.075 更低。
    head_bottom_m=-0.065,
    com_offset_x_m=0.0213,
    # 官方 CAD 69.003 kg；官网标称 60 kg（含电池）。已按用户决定：不再等厂商核对，
    # 直接用 MJCF 的保守值（比标称重 9 kg，动力学更悲观）。
    nominal_mass_kg=69.003,
    # 关节额定力矩：hip/thigh 200、calf 320 N·m（官方 URDF <limit effort> 与
    # unitree_rl_lab 的 UNITREE_B2_CFG 一致）。注意 MJCF 执行器 ctrlrange 只写
    # 300 N·m，仿真里实际限幅是 300；选型与热分析按 320 记录。
    joint_torque_nm=(200.0, 200.0, 320.0),
    # 足端橡胶摩擦。场景踏步按防滑格栅取 0.9，MuJoCo 取两者逐元素最大值，
    # 因此接触等效 μ=0.9（足端 0.4 会被地面覆盖，见 docs 里的一节说明）。
    foot_friction=0.4,
    expected_nq=19,
    expected_nv=18,
    expected_nu=12,
    # 被动站立标定（本仓库实测，2026-11-06）：官方 Isaac 的 160/5 在没有上层
    # 控制器时托不住 69 kg——髋关节会缓慢外张（0.39 rad）并最终塌到机身 0.24 m；
    # 800/20 在 50 Hz 控制下稳定（峰峰 0.001 mm），1600 以上在 50 Hz 反而发散。
    kp=800.0,
    kd=20.0,
    joint_limit_margin_rad=0.05,
    fall_height_m=0.20,  # 未标定：按站高 0.4065 m 的比例粗设，接入时必须重标
    fall_tilt_rad=np.radians(60.0),
    # 步态形状沿用 Go2 占位，未在 B2 上标定。
    trot=GO2_ROBOT.trot,
    fsm=FsmConfig(
        control_hz=50.0,
        pd_kp=(800.0, 800.0, 800.0) * 4,  # 被动站立标定值，见 kp/kd 注释
        pd_kd=(20.0, 20.0, 20.0) * 4,
        gravity_feedforward_scale=1.0,
        residual_scale=0.10,
    ),
)

# B2 已注册：硬件参数按官方 MJCF 固化（不再等厂商校准）。步态与 FSM 增益仍是
# 未标定值，接训练前必须重标；但几何/质量/力矩/摩擦已经是“可复现的定值”。
ROBOT_REGISTRY = {GO2_ROBOT.name: GO2_ROBOT, B2_ROBOT.name: B2_ROBOT}


def get_robot(name: str = "go2") -> RobotConfig:
    """按名字取机器人配置。

    参数：
        name：注册名（当前只有 ``"go2"``）。
    返回：
        ``RobotConfig``。
    异常：
        名字未注册时抛出 KeyError，并列出可用名字，避免静默回退到 Go2。
    重要性等级：A。
    """
    try:
        return ROBOT_REGISTRY[name]
    except KeyError as error:
        raise KeyError(
            f"未注册的机器人 {name!r}；可用: {sorted(ROBOT_REGISTRY)}"
        ) from error
