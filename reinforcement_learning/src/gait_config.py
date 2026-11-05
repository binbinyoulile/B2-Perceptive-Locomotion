"""Go2 Trot 先验配置的兼容层（真正的数值已迁到 configs/robot_config.py）。

文件目的：
    历史上 ``go2_env.py``、``gait_controller.py`` 都从这里导入 Trot 常量。
    参数解耦后，唯一事实来源是 ``configs.robot_config.GO2_ROBOT.trot``；
    本文件只做“旧名字 → 新配置”的转发，保证既有导入路径不被打断。

整体输入输出：
    无运行时输入；导入时立即从配置对象取值。
依赖与版本要求：
    Python 3.10+、NumPy 2.x。
运行方式：
    由 ``go2_env.py`` / ``gait_controller.py`` 导入。
重要说明：
    新增参数请直接加到 ``RobotConfig``，不要在这里重新写一份字面量，
    否则会重新制造“两处常量漂移”的老问题。
重要性等级：B（兼容层；数值本身在 robot_config 中是 A 级）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 本文件可能被 ``src.gait_controller`` 在 PROJECT_ROOT 尚未进入 sys.path 时导入，
# 这里显式补齐仓库根目录，保证 ``configs`` 包可解析。
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from configs.robot_config import GO2_ROBOT  # noqa: E402

_TROT = GO2_ROBOT.trot

TROT_RESIDUAL_SCALE = _TROT.residual_scale
TROT_FREQUENCY_HZ = _TROT.frequency_hz
TROT_RAMP_TIME_S = _TROT.ramp_time_s
TROT_WARMUP_TIME_S = _TROT.warmup_time_s
TROT_PAIR_PHASE_CYCLES = np.array(_TROT.pair_phase_cycles, dtype=np.float64)
TROT_HIP_SIGNS = np.array(_TROT.hip_signs, dtype=np.float64)
TROT_HIP_AMPLITUDE_RAD = _TROT.hip_amplitude_rad
TROT_THIGH_AMPLITUDE_RAD = _TROT.thigh_amplitude_rad
TROT_CALF_AMPLITUDE_RAD = _TROT.calf_amplitude_rad
TROT_SWING_HEIGHT_M = _TROT.swing_height_m
TROT_SWING_KNEE_GAIN_RAD_PER_M = np.array(
    _TROT.swing_knee_gain_rad_per_m, dtype=np.float64
)
JOINT_LIMIT_MARGIN_RAD = GO2_ROBOT.joint_limit_margin_rad
STAND_JOINT_RAD = GO2_ROBOT.stand_pose()
