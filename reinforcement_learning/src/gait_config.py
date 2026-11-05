"""Go2 Trot 先验与关节残差配置。

文件目的：
    集中保存开环 Trot 频率、相位、振幅和动作残差尺度，供环境骨架与步态
    控制器共享，避免两处常量漂移。
整体输入输出：
    无运行时输入；输出为只读 NumPy 标量/数组配置。
依赖与版本要求：
    Python 3.10+、NumPy 2.x。
运行方式：
    由 ``go2_env.py`` 与 ``gait_controller.py`` 导入。
关键假设：
    足端顺序固定为 FL/FR/RL/RR，控制频率为 50 Hz。
已知限制：
    参数针对当前 Go2 质量、PD 增益和 0.04 m 单级台阶课程整定。
重要性等级：B（决定先验步态形状和残差权限）。
"""

from __future__ import annotations

import numpy as np


TROT_RESIDUAL_SCALE = 0.08
TROT_FREQUENCY_HZ = 2.0
TROT_RAMP_TIME_S = 0.8
TROT_WARMUP_TIME_S = 0.5
TROT_PAIR_PHASE_CYCLES = np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
TROT_HIP_SIGNS = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float64)
TROT_HIP_AMPLITUDE_RAD = 0.02
TROT_THIGH_AMPLITUDE_RAD = 0.18
TROT_CALF_AMPLITUDE_RAD = 0.14
TROT_SWING_HEIGHT_M = 0.045
TROT_SWING_KNEE_GAIN_RAD_PER_M = np.array(
    (4.4, 4.4, 11.0, 11.0), dtype=np.float64
)
JOINT_LIMIT_MARGIN_RAD = 0.05
STAND_JOINT_RAD = np.tile(
    np.array((0.0, 0.9, -1.8), dtype=np.float64), 4
)
