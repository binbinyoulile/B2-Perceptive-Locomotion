"""机器人参数配置：把 Go2 专属常量从环境/步态/观测代码中抽离。

文件目的：
    集中保存“换狗时要改的硬件相关参数”（关节顺序、站立姿态、腿长、质量、
    控制增益、Trot 振幅、雷达安装位、模型维度），让环境、步态、观测、规则
    教师都只读这一份配置，而不是各自硬编码。
整体输入：
    无运行时输入；模块加载时构造只读配置对象。
整体输出：
    ``GO2_ROBOT`（当前唯一实例）与 ``ROBOT_REGISTRY``；``get_robot(name)``。
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
    只有 Go2 是“已在本仓库验证过”的实例。B2 / M1 的实例需要拿到官方 MJCF
    后再用 ``build_robot_config_from_mjcf`` 填充，不能凭宣传参数猜。
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
        几何：腿长、足端半径、髋位、雷达安装位、体长/体高（用于可行性计算）。
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
    nominal_mass_kg: float
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
    nominal_mass_kg=15.206,
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

ROBOT_REGISTRY = {GO2_ROBOT.name: GO2_ROBOT}


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
