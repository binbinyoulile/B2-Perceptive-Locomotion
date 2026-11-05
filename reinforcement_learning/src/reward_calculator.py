"""Go2 强化学习奖励计算模块。

文件目的：
    集中保存奖励权重并计算全部奖励项，把奖励塑形从 Gym 环境骨架中解耦。
    修改奖励时只需检查本文件，不必阅读环境重置、物理推进和终止逻辑。
整体输入：
    当前 ``Go2Env`` 的速度、姿态、动作、力矩，以及地形/足端模块产生的分项。
整体输出：
    ``(total_reward, terms)``；``terms`` 保留所有正负分项供 TensorBoard 和诊断。
依赖与版本要求：
    Python 3.10+、NumPy 2.x。
运行方式：
    由 ``Go2Env._compute_reward()`` 每个 50 Hz 控制步调用。
关键假设：
    速度单位为 m/s、姿态单位为 rad、力矩单位为 N·m；奖励为无量纲标量。
已知限制：
    奖励仍是手工塑形，调到相同总报酬下可能改变探索行为；本轮不保证策略收敛。
重要性等级：A（奖励权重和分项定义直接决定策略行为）。
"""

from __future__ import annotations

import math

import numpy as np


# 前进、速度跟踪与稳定性基础项。
FORWARD_SPEED_REWARD_WEIGHT = 3.0
FORWARD_SPEED_REWARD_CLIP = 1.0
SPEED_TRACKING_REWARD_WEIGHT = 2.0
SPEED_TRACKING_SIGMA_MPS = 0.15
STABILITY_REWARD_WEIGHT = 0.5
STABILITY_TILT_SCALE_RAD = 0.20
STANDSTILL_PENALTY_WEIGHT = 2.0
STANDSTILL_SPEED_SIGMA_MPS = 0.12
TARGET_FORWARD_SPEED_MPS = 0.5
# 以下姿态约束只负责稳定机身，不应压过前进目标；数值单位均为奖励标量。
LATERAL_VELOCITY_PENALTY_WEIGHT = 0.10
ORIENTATION_PENALTY_WEIGHT = 0.125
LATERAL_OFFSET_PENALTY_WEIGHT = 0.5
YAW_RATE_PENALTY_WEIGHT = 0.05

# 动作平滑和能耗项保留，用于抑制高频抖动；本轮去掉动作均值正则和
# 三足/同侧支撑惩罚，让策略能探索足够大的动作。
ACTION_SMOOTHNESS_PENALTY_WEIGHT = 0.20
ACTION_MAGNITUDE_PENALTY_WEIGHT = 0.60
ACTION_SATURATION_PENALTY_WEIGHT = 1.0
ACTION_SATURATION_THRESHOLD = 0.95
ACTION_RATE_LIMIT_THRESHOLD = 0.35
ACTION_RATE_LIMIT_PENALTY_WEIGHT = 2.0
ENERGY_PENALTY_WEIGHT = 5e-5
ACTION_MEAN_PENALTY_WEIGHT = 0.0
# 关节限位惩罚：使用量定义为 |q - 中位| / 半行程，超过 90% 后平方惩罚。
# 该阈值迫使残差策略通过调整重心和步态完成动作，而不是把腿伸到机械极限。
JOINT_LIMIT_USAGE_THRESHOLD = 0.90
JOINT_LIMIT_PENALTY_WEIGHT = 2.0
SAME_SIDE_SUPPORT_PENALTY_PER_STEP = 0.0
TRIPOD_SUPPORT_PENALTY_PER_STEP = 0.0

# 存活奖励降到 0.1，避免原地站立刷分；跌倒仍保留一次性大额惩罚。
ALIVE_BONUS_WEIGHT = 0.1
FALL_TERMINATION_PENALTY = 100.0

# 单级台阶的当前帧奖励契约：前足搭台 0.5、后足落台 3.0、四足当前登顶 15.0。
STEP_FRONT_FEET_REWARD = 0.5
STEP_ALL_FEET_REWARD = 15.0
STEP_STABLE_REWARD_PER_STEP = 0.0
STEP_STABLE_SPEED_MPS = 0.10
STEP_EDGE_TILT_THRESHOLD_RAD = math.radians(35.0)
STEP_EDGE_DISTANCE_M = 0.5
STEP_EDGE_TILT_PENALTY = 10.0
STEP_FOOT_TOP_TOLERANCE_M = 0.01
STEP_FOOT_TOP_MAX_TOLERANCE_M = 0.02
STEP_FOOT_TOP_XY_INSET_M = 0.01
STEP_STABLE_FOOT_Z_TOLERANCE_M = 0.02
STEP_CROSS_TOLERANCE_M = 0.03

# 悬挂契约：前足首次真实搭台后锁存计时，50 个控制步内机身 Z 未达到
# “初始机身高度 + 台阶高度 - 容差”则扣 10 分并终止。
HANGING_WINDOW_STEPS = 50
HANGING_TERMINATION_PENALTY = 10.0
HANGING_BASE_Z_TOLERANCE_M = 0.02
HANGING_FRONT_CONTACT_MEMORY_STEPS = 1
HANGING_BASE_ADVANCE_REWARD_PER_M = 0.0
STEP_BASE_CROSSING_REWARD = 0.0

# 台前低速卡住判定与碰撞惩罚。窗口和力阈值沿用已调过的物理参数。
STEP_STUCK_SPEED_MPS = 0.08
STEP_STUCK_ZONE_M = 0.35
STEP_CONTACT_FORCE_THRESHOLD_N = 20.0
STEP_CONTACT_PENALTY_WEIGHT = 0.01
STEP_CONTACT_PENALTY_MAX = 2.0
STEP_FRONT_CONTACT_BAND_M = 0.05
STEP_TOP_CONTACT_MARGIN_M = 0.005

# 足端抬升和落足塑形。后足落顶改为本轮明确要求的 +3.0；前足普通落顶
# 不再重复叠加 +1，前足阶段奖励由 STEP_FRONT_FEET_REWARD 统一控制。
FOOT_CLEARANCE_THRESHOLD_M = 0.04
FOOT_CLEARANCE_REWARD_PER_FOOT = 0.15
FOOT_CLEARANCE_MIN_SUPPORT_FEET = 2
FOOT_CLEARANCE_MIN_FORWARD_SPEED_MPS = 0.05
ALL_FEET_AIRBORNE_PENALTY = 1.0
CONTINUOUS_AIRBORNE_GRACE_STEPS = 2
CONTINUOUS_AIRBORNE_ESCALATION_BASE = 2.0
CONTINUOUS_AIRBORNE_ESCALATION_CAP = 64.0
DIAGONAL_SUPPORT_REWARD_PER_STEP = 0.05
DIAGONAL_ALTERNATION_REWARD = 1.5
TROT_PHASE_CONTACT_REWARD_PER_STEP = 0.0
TROT_PHASE_SIN_THRESHOLD = 0.35
DEAD_LEG_WINDOW_STEPS = 12
DEAD_LEG_PENALTY_PER_FOOT = 0.2
DEAD_LEG_FORWARD_SPEED_MPS = 0.08
FOOT_PLACEMENT_SWING_REWARD_PER_FOOT = 0.20
FOOT_PLACEMENT_TOP_LANDING_REWARD = 0.0
FOOT_PLACEMENT_PRE_EDGE_PENALTY = 0.05
FOOT_PLACEMENT_SWING_MAX_REWARDED_STEPS = 3
FOOT_PLACEMENT_NEAR_EDGE_DISTANCE_M = 0.6
REAR_FOOT_TOP_LANDING_REWARD = 3.0

# 机身历史最远 X 的密集前移奖励；只奖励新前沿，回退再前进不会重复领奖。
BASE_PROGRESS_REWARD_PER_STEP = 0.10
BASE_PROGRESS_DELTA_FOR_MAX_REWARD_M = 0.01


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    """计算机身局部 +Z 轴与世界 +Z 轴的夹角。"""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"四元数 shape 应为 (4,)，实际 {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("四元数长度接近 0")
    w, x, y, z = quaternion / norm
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(float(np.clip(body_z_world_z, -1.0, 1.0))))


def compute_reward(
    env,
    action: np.ndarray,
    previous_action: np.ndarray,
    torque: np.ndarray,
    foot_contact_count: int = 4,
    step_front_feet_reward: float = 0.0,
    step_all_feet_reward: float = 0.0,
    step_stable_reward: float = 0.0,
    step_edge_tilt_penalty: float = 0.0,
    step_contact_penalty: float = 0.0,
    foot_clearance_reward: float = 0.0,
    diagonal_support_reward: float = 0.0,
    diagonal_alternation_reward: float = 0.0,
    same_side_support_penalty: float = 0.0,
    tripod_support_penalty: float = 0.0,
    all_feet_airborne_penalty: float = 0.0,
    dead_leg_penalty: float = 0.0,
    trot_phase_contact_reward: float = 0.0,
    foot_placement_swing_reward: float = 0.0,
    foot_placement_top_reward: float = 0.0,
    foot_placement_pre_edge_penalty: float = 0.0,
    rear_foot_top_landing_reward: float = 0.0,
    base_progress_reward: float = 0.0,
    step_base_crossing_reward: float = 0.0,
    hanging_base_advance_reward: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """计算一个控制步的总奖励与分项。

    参数：
        env：当前 Go2 环境，只读取速度和姿态。
        action：shape=(12,)，送入物理仿真的裁剪动作，范围 [-1,1]。
        previous_action：shape=(12,)，上一控制步动作。
        torque：shape=(12,)，本控制步末次 PD 力矩，单位 N·m。
        其余参数为地形、足端和阶段模块传入的已计算分项；单位均为奖励标量。
    返回：
        ``(total_reward, terms)``；terms 中同时保留动作均值惩罚字段，但其权重
        为 0，便于日志兼容。``joint_limit_penalty`` 对超过 90% 关节行程的
        状态施加平方惩罚，避免策略用机械极限换取前进。
    可能异常：
        动作 shape 异常由上游 Gymnasium 校验；此函数只做数值计算。
    副作用：
        无，不修改环境状态。
    重要性等级：A。
    """
    local_velocity = env._get_base_linear_velocity_local()
    forward_velocity_mps = float(local_velocity[0])
    lateral_velocity_mps = float(local_velocity[1])
    normalized_forward_speed = float(
        np.clip(
            forward_velocity_mps / TARGET_FORWARD_SPEED_MPS,
            -FORWARD_SPEED_REWARD_CLIP,
            FORWARD_SPEED_REWARD_CLIP,
        )
    )
    contact_count = int(np.clip(foot_contact_count, 0, 4))
    all_feet_airborne = contact_count == 0

    if all_feet_airborne:
        forward_speed_reward = 0.0
        speed_tracking_reward = 0.0
        stability_reward = 0.0
    else:
        forward_speed_reward = FORWARD_SPEED_REWARD_WEIGHT * normalized_forward_speed
        speed_error_mps = forward_velocity_mps - TARGET_FORWARD_SPEED_MPS
        speed_tracking_reward = SPEED_TRACKING_REWARD_WEIGHT * float(
            np.exp(-(speed_error_mps * speed_error_mps) / (2.0 * SPEED_TRACKING_SIGMA_MPS**2))
        )
        tilt_for_stability = _tilt_rad(np.asarray(env.data.qpos[3:7], dtype=np.float64))
        tilt_score = float(np.exp(-((tilt_for_stability / STABILITY_TILT_SCALE_RAD) ** 2)))
        support_score = min(contact_count, 2) / 2.0
        stability_reward = STABILITY_REWARD_WEIGHT * tilt_score * support_score

    standstill_penalty = STANDSTILL_PENALTY_WEIGHT * float(
        1.0
        - np.exp(
            -(forward_velocity_mps * forward_velocity_mps)
            / (2.0 * STANDSTILL_SPEED_SIGMA_MPS * STANDSTILL_SPEED_SIGMA_MPS)
        )
    )
    tilt_rad = _tilt_rad(np.asarray(env.data.qpos[3:7], dtype=np.float64))
    lateral_offset_m = float(env.data.qpos[1] - env.initial_base_y_m)
    yaw_rate_rad_s = float(env.data.qvel[5])
    action_values = np.asarray(action, dtype=np.float64)
    action_delta = action_values - np.asarray(previous_action, dtype=np.float64)

    orientation_penalty = ORIENTATION_PENALTY_WEIGHT * tilt_rad * tilt_rad
    lateral_velocity_penalty = LATERAL_VELOCITY_PENALTY_WEIGHT * lateral_velocity_mps**2
    lateral_offset_penalty = LATERAL_OFFSET_PENALTY_WEIGHT * lateral_offset_m**2
    yaw_rate_penalty = YAW_RATE_PENALTY_WEIGHT * yaw_rate_rad_s**2
    action_smoothness_penalty = ACTION_SMOOTHNESS_PENALTY_WEIGHT * float(
        np.mean(np.square(action_delta))
    )
    action_rate_excess = np.maximum(
        np.abs(action_delta) - ACTION_RATE_LIMIT_THRESHOLD,
        0.0,
    )
    action_rate_limit_penalty = ACTION_RATE_LIMIT_PENALTY_WEIGHT * float(
        np.mean(np.square(action_rate_excess))
    )
    action_magnitude_penalty = ACTION_MAGNITUDE_PENALTY_WEIGHT * float(
        np.mean(np.square(action_values))
    )
    action_saturation_excess = np.maximum(
        np.abs(action_values) - ACTION_SATURATION_THRESHOLD,
        0.0,
    )
    action_saturation_penalty = ACTION_SATURATION_PENALTY_WEIGHT * float(
        np.mean(
            np.square(
                action_saturation_excess / (1.0 - ACTION_SATURATION_THRESHOLD)
            )
        )
    )
    # 本轮明确取消动作均值正则；字段保留为 0 便于旧日志读取。
    action_mean_penalty = ACTION_MEAN_PENALTY_WEIGHT * float(
        np.mean(np.abs(action_values)) ** 2
    )
    energy_penalty = ENERGY_PENALTY_WEIGHT * float(
        np.sum(np.square(np.asarray(torque, dtype=np.float64)))
    )
    # 关节使用率只依据可测关节角和模型限位，不使用全局位置真值。
    joint_position = np.asarray(
        env.data.qpos[env.qpos_addresses], dtype=np.float64
    )
    joint_center = (
        np.asarray(env.joint_low, dtype=np.float64)
        + np.asarray(env.joint_high, dtype=np.float64)
    ) / 2.0
    joint_half_range = (
        np.asarray(env.joint_high, dtype=np.float64)
        - np.asarray(env.joint_low, dtype=np.float64)
    ) / 2.0
    joint_usage_ratio = np.abs((joint_position - joint_center) / joint_half_range)
    joint_limit_excess = np.maximum(
        joint_usage_ratio - JOINT_LIMIT_USAGE_THRESHOLD,
        0.0,
    )
    joint_limit_penalty = JOINT_LIMIT_PENALTY_WEIGHT * float(
        np.mean(np.square(joint_limit_excess))
    )
    alive_bonus = ALIVE_BONUS_WEIGHT

    total_reward = (
        forward_speed_reward
        + speed_tracking_reward
        + stability_reward
        + alive_bonus
        - standstill_penalty
        - lateral_velocity_penalty
        - orientation_penalty
        - lateral_offset_penalty
        - yaw_rate_penalty
        - action_smoothness_penalty
        - action_rate_limit_penalty
        - action_magnitude_penalty
        - action_saturation_penalty
        - action_mean_penalty
        - energy_penalty
        - joint_limit_penalty
        + step_front_feet_reward
        + step_all_feet_reward
        + step_stable_reward
        + foot_clearance_reward
        + diagonal_support_reward
        + diagonal_alternation_reward
        + trot_phase_contact_reward
        - same_side_support_penalty
        - tripod_support_penalty
        - all_feet_airborne_penalty
        - dead_leg_penalty
        - step_edge_tilt_penalty
        - step_contact_penalty
        + foot_placement_swing_reward
        + foot_placement_top_reward
        - foot_placement_pre_edge_penalty
        + rear_foot_top_landing_reward
        + base_progress_reward
        + step_base_crossing_reward
        + hanging_base_advance_reward
    )
    terms = {
        "forward_speed_reward": float(forward_speed_reward),
        "speed_tracking_reward": float(speed_tracking_reward),
        "stability_reward": float(stability_reward),
        "alive_bonus": float(alive_bonus),
        "standstill_penalty": standstill_penalty,
        "lateral_velocity_penalty": lateral_velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "lateral_offset_penalty": lateral_offset_penalty,
        "yaw_rate_penalty": yaw_rate_penalty,
        "action_smoothness_penalty": action_smoothness_penalty,
        "action_rate_limit_penalty": action_rate_limit_penalty,
        "action_magnitude_penalty": action_magnitude_penalty,
        "action_saturation_penalty": action_saturation_penalty,
        "action_mean_penalty": action_mean_penalty,
        "energy_penalty": energy_penalty,
        "joint_limit_penalty": joint_limit_penalty,
        "joint_limit_max_usage_ratio": float(np.max(joint_usage_ratio)),
        "joint_limit_excess_count": int(np.count_nonzero(joint_limit_excess > 0.0)),
        "step_front_feet_reward": float(step_front_feet_reward),
        "step_all_feet_reward": float(step_all_feet_reward),
        "step_stable_reward": float(step_stable_reward),
        "foot_clearance_reward": float(foot_clearance_reward),
        "diagonal_support_reward": float(diagonal_support_reward),
        "diagonal_alternation_reward": float(diagonal_alternation_reward),
        "trot_phase_contact_reward": float(trot_phase_contact_reward),
        "same_side_support_penalty": float(same_side_support_penalty),
        "tripod_support_penalty": float(tripod_support_penalty),
        "all_feet_airborne_penalty": float(all_feet_airborne_penalty),
        "dead_leg_penalty": float(dead_leg_penalty),
        "step_edge_tilt_penalty": float(step_edge_tilt_penalty),
        "step_contact_penalty": float(step_contact_penalty),
        "foot_placement_swing_reward": float(foot_placement_swing_reward),
        "foot_placement_top_reward": float(foot_placement_top_reward),
        "foot_placement_pre_edge_penalty": float(foot_placement_pre_edge_penalty),
        "rear_foot_top_landing_reward": float(rear_foot_top_landing_reward),
        "base_progress_reward": float(base_progress_reward),
        "step_base_crossing_reward": float(step_base_crossing_reward),
        "hanging_base_advance_reward": float(hanging_base_advance_reward),
        "total_reward": float(total_reward),
    }
    return float(total_reward), terms
