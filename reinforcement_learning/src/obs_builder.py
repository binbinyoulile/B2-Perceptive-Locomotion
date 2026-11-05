"""Go2 策略观测构建器。

文件目的：
    集中定义观测布局，并从当前 MuJoCo 物理状态构建 175 维 float32 观测。
    环境骨架只负责调用本模块，不再包含观测拼接细节。
整体输入：
    ``Go2Env`` 实例，且已完成 ``reset()``；所需字段包括 ``data``、关节地址、
    足端 geom、机身 body、上一动作、LiDAR 高度图和 Trot 相位。
整体输出：
    shape=(175,)、dtype=float32、无 NaN/Inf 的观测向量；另外提供与观测空间
    对应的上下界。
依赖与版本要求：
    Python 3.10+、NumPy 2.x、MuJoCo 3.x、gymnasium 1.x。
运行方式：
    由 ``Go2Env._get_observation()`` 在每个控制边界调用。
关键假设：
    足端顺序固定为 FL/FR/RL/RR；机身坐标系 +X 向前、+Y 向左、+Z 向上。
已知限制：
    观测包含仿真 LiDAR 高度图，但未来真机必须通过 ROS2 PointCloud2 生成同格式
    高度图后复用同一布局；旧 150/163 维策略权重与 175 维输入不兼容。
重要性等级：A（观测顺序改变会直接破坏训练与回放兼容性）。
"""

from __future__ import annotations

import numpy as np

try:
    from .terrain_perception import (
        HEIGHTMAP_COLS,
        HEIGHTMAP_RESOLUTION_M,
        HEIGHTMAP_ROWS,
        HEIGHTMAP_X_MIN_M,
        HEIGHTMAP_Y_MIN_M,
        heightmap_to_nearest_step_distance,
    )
except ImportError:
    from terrain_perception import (  # type: ignore[no-redef]
        HEIGHTMAP_COLS,
        HEIGHTMAP_RESOLUTION_M,
        HEIGHTMAP_ROWS,
        HEIGHTMAP_X_MIN_M,
        HEIGHTMAP_Y_MIN_M,
        heightmap_to_nearest_step_distance,
    )


# 新观测前缀严格沿用旧 150 维语义，末段追加 12 维足端局部坐标和 1 维最近台阶距离。
OBSERVATION_LAYOUT = {
    "base_height_relative_m": slice(0, 1),
    "base_quaternion_wxyz": slice(1, 5),
    "base_linear_velocity_local_mps": slice(5, 8),
    "joint_position_rad": slice(8, 20),
    "joint_velocity_rad_s": slice(20, 32),
    "foot_contacts": slice(32, 36),
    "last_action": slice(36, 48),
    "terrain_heightmap": slice(48, 148),
    "trot_phase_sin_cos": slice(148, 150),
    "foot_local_positions_m": slice(150, 162),
    "nearest_step_distance_m": slice(162, 163),
    # 当前关节角到最近物理限位的距离，单位 rad；12 个值均非负。该观测
    # 不依赖真值地形，真机可由关节编码器和 URDF 限位直接计算。
    "joint_limit_distance_rad": slice(163, 175),
}
TERRAIN_HEIGHTMAP_DIM = HEIGHTMAP_ROWS * HEIGHTMAP_COLS
OBSERVATION_DIM = 175


def get_foot_local_positions(env) -> np.ndarray:
    """返回四只脚相对机身中心的局部三维坐标。

    参数：
        env：当前 ``Go2Env``；``data.geom_xpos`` 和 ``data.xmat`` 必须为最新值。
    返回：
        shape=(12,)、float64、单位 m，顺序为 FL/FR/RL/RR 的 x/y/z。
    算法：
        世界偏移先减去 base 世界坐标，再用 ``base_rotation.T`` 转到机身局部坐标。
    异常与边界：
        MuJoCo 位姿出现 NaN/Inf 时由调用方通过 ``np.isfinite`` 拦截。
    副作用：
        只读 MuJoCo 状态，不推进物理。
    重要性等级：A（真机可由关节正运动学提供同语义量）。
    """
    base_position = np.asarray(env.data.xpos[env.base_body_id], dtype=np.float64)
    base_rotation = np.asarray(
        env.data.xmat[env.base_body_id], dtype=np.float64
    ).reshape(3, 3)
    foot_world = np.asarray(env.data.geom_xpos[env.foot_geom_ids], dtype=np.float64)
    offsets_world = foot_world - base_position[None, :]
    local_offsets = offsets_world @ base_rotation
    return local_offsets.reshape(-1)


def get_nearest_step_distance(env) -> float:
    """从当前局部高度图估计机身到最近台阶的水平距离。

    参数：
        env：当前 ``Go2Env``。无 LiDAR 或尚无扫描时返回量程上限 1.0 m。
    返回：
        float，单位 m，范围 [0, 1]。无台阶时返回 1.0。
    重要假设：
        高度图行沿机身 +X、列沿 +Y，格子高度为相对地面的局部高程。
    副作用：
        纯读取，不修改高度图。
    重要性等级：A（新增的本体/雷达相对观测）。
    """
    scan = getattr(env, "last_terrain_scan", None)
    if scan is None:
        return 1.0
    return float(
        heightmap_to_nearest_step_distance(
            scan.heightmap,
            x_min_m=HEIGHTMAP_X_MIN_M,
            y_min_m=HEIGHTMAP_Y_MIN_M,
            resolution_m=HEIGHTMAP_RESOLUTION_M,
            height_threshold_m=0.015,
            max_distance_m=1.0,
        )
    )


def build_observation(env) -> np.ndarray:
    """拼接完整的 175 维策略观测。

    参数：
        env：当前 ``Go2Env``，调用前应已执行 reset 或本控制步的物理推进。
    返回：
        shape=(175,)、float32；顺序见 ``OBSERVATION_LAYOUT``。
    异常：
        若观测中出现 NaN/Inf 会抛出 RuntimeError，避免损坏 PPO 数值。
    副作用：
        只读状态；高度图返回独立副本，不共享给外部修改。
    重要性等级：A。
    """
    base_height_relative_m = float(env.data.qpos[2]) - float(env.initial_base_z_m)
    quaternion_wxyz = np.asarray(env.data.qpos[3:7], dtype=np.float64)
    base_linear_velocity_local = env._get_base_linear_velocity_local()
    joint_position_rad = np.asarray(
        env.data.qpos[env.qpos_addresses], dtype=np.float64
    )
    joint_velocity_rad_s = np.asarray(
        env.data.qvel[env.dof_addresses], dtype=np.float64
    )
    foot_contacts = env._get_foot_contacts()
    if env.last_terrain_scan is None:
        terrain_heightmap = np.zeros(
            (HEIGHTMAP_ROWS, HEIGHTMAP_COLS), dtype=np.float32
        )
    else:
        terrain_heightmap = env.last_terrain_scan.heightmap.astype(
            np.float32, copy=True
        )
    phase_sin, phase_cos = env._get_trot_phase_sin_cos()
    foot_local_positions_m = get_foot_local_positions(env)
    nearest_step_distance_m = get_nearest_step_distance(env)
    # 同时包含“距下限”和“距上限”的较小值；越接近 0 表示越接近机械限位。
    joint_position = np.asarray(
        env.data.qpos[env.qpos_addresses], dtype=np.float64
    )
    joint_limit_distance_rad = np.maximum(
        0.0,
        np.minimum(
            joint_position - np.asarray(env.joint_low, dtype=np.float64),
            np.asarray(env.joint_high, dtype=np.float64) - joint_position,
        ),
    )

    observation = np.concatenate(
        (
            np.array((base_height_relative_m,), dtype=np.float64),
            quaternion_wxyz,
            base_linear_velocity_local,
            joint_position_rad,
            joint_velocity_rad_s,
            foot_contacts,
            env.last_action,
            terrain_heightmap.reshape(-1),
            np.array((phase_sin, phase_cos), dtype=np.float64),
            foot_local_positions_m,
            np.array((nearest_step_distance_m,), dtype=np.float64),
            joint_limit_distance_rad,
        )
    ).astype(np.float32)
    if observation.shape != (OBSERVATION_DIM,):
        raise RuntimeError(
            f"观测 shape 应为 ({OBSERVATION_DIM},)，实际 {observation.shape}"
        )
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("观测包含 NaN/Inf，请检查 MuJoCo 位姿和 LiDAR 高度图")
    return observation


def build_observation_space_bounds(env) -> tuple[np.ndarray, np.ndarray]:
    """构造与 175 维观测顺序一致的 Box 上下界。

    参数：
        env：已完成模型加载的 ``Go2Env``，需要 ``joint_low``/``joint_high``。
    返回：
        ``(low, high)``，均为 shape=(175,)、dtype=float64。
    调参说明：
        足端局部坐标和最近台阶距离使用保守边界；边界过窄会让 Gymnasium
        检查失败，过宽会降低观测归一化质量。
    重要性等级：B（支持策略网络输入层与 Gymnasium 接口校验）。
    """
    low_parts = (
        np.array((-2.0,), dtype=np.float64),
        -np.ones(4, dtype=np.float64),
        -20.0 * np.ones(3, dtype=np.float64),
        np.asarray(env.joint_low, dtype=np.float64),
        -50.0 * np.ones(12, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        -np.ones(12, dtype=np.float64),
        -2.0 * np.ones(TERRAIN_HEIGHTMAP_DIM, dtype=np.float64),
        -np.ones(2, dtype=np.float64),
        -np.ones(12, dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        np.zeros(12, dtype=np.float64),
    )
    high_parts = (
        np.array((2.0,), dtype=np.float64),
        np.ones(4, dtype=np.float64),
        20.0 * np.ones(3, dtype=np.float64),
        np.asarray(env.joint_high, dtype=np.float64),
        50.0 * np.ones(12, dtype=np.float64),
        np.ones(4, dtype=np.float64),
        np.ones(12, dtype=np.float64),
        2.0 * np.ones(TERRAIN_HEIGHTMAP_DIM, dtype=np.float64),
        np.ones(2, dtype=np.float64),
        np.ones(12, dtype=np.float64),
        np.ones(1, dtype=np.float64),
        np.asarray(env.joint_high, dtype=np.float64)
        - np.asarray(env.joint_low, dtype=np.float64),
    )
    return np.concatenate(low_parts), np.concatenate(high_parts)
