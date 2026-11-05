"""Go2 单级台阶几何、随机化与进度判定模块。

文件目的：
    把台阶随机化、足端台面支撑、当前帧登顶判定、悬挂计时和雷达刷新从
    Gym 环境骨架中拆出，避免地形规则散落在 ``step()`` 各处。
整体输入：
    ``Go2Env`` 实例的最新 MuJoCo 状态、局部高度图和台阶几何参数。
整体输出：
    地形进度字典、雷达扫描副本和终止所需状态。
依赖与版本要求：
    Python 3.10+、NumPy 2.x、MuJoCo 3.x。
运行方式：
    由 ``Go2Env`` 继承，并在 reset/step 中调用。
关键假设：
    单级台阶 geom 名称为 ``curriculum_step_01``，地形模式为 ``single_step``。
已知限制：
    只实现单级台阶，不等同于多级楼梯课程。
重要性等级：A（登顶与悬挂判定直接决定训练目标）。
"""

from __future__ import annotations

import numpy as np
import mujoco

try:
    from .terrain_perception import TerrainScan
    from .reward_calculator import *  # noqa: F401,F403
    from .reward_calculator import HANGING_BASE_Z_TOLERANCE_M
    from .reward_calculator import HANGING_WINDOW_STEPS
    from .reward_calculator import STEP_ALL_FEET_REWARD
    from .reward_calculator import STEP_CONTACT_FORCE_THRESHOLD_N
    from .reward_calculator import STEP_CONTACT_PENALTY_MAX
    from .reward_calculator import STEP_CONTACT_PENALTY_WEIGHT
    from .reward_calculator import STEP_CROSS_TOLERANCE_M
    from .reward_calculator import STEP_EDGE_DISTANCE_M
    from .reward_calculator import STEP_EDGE_TILT_PENALTY
    from .reward_calculator import STEP_EDGE_TILT_THRESHOLD_RAD
    from .reward_calculator import STEP_FOOT_TOP_MAX_TOLERANCE_M
    from .reward_calculator import STEP_FOOT_TOP_TOLERANCE_M
    from .reward_calculator import STEP_FOOT_TOP_XY_INSET_M
    from .reward_calculator import STEP_FRONT_CONTACT_BAND_M
    from .reward_calculator import STEP_FRONT_FEET_REWARD
    from .reward_calculator import STEP_STUCK_SPEED_MPS
    from .reward_calculator import STEP_STUCK_ZONE_M
    from .reward_calculator import STEP_TOP_CONTACT_MARGIN_M
except ImportError:
    from terrain_perception import TerrainScan  # type: ignore[no-redef]
    from reward_calculator import *  # type: ignore[no-redef]  # noqa: F401,F403
    from reward_calculator import (  # type: ignore[no-redef]
        HANGING_BASE_Z_TOLERANCE_M, HANGING_WINDOW_STEPS, STEP_ALL_FEET_REWARD,
        STEP_CONTACT_FORCE_THRESHOLD_N, STEP_CONTACT_PENALTY_MAX,
        STEP_CONTACT_PENALTY_WEIGHT, STEP_CROSS_TOLERANCE_M,
        STEP_EDGE_DISTANCE_M, STEP_EDGE_TILT_PENALTY,
        STEP_EDGE_TILT_THRESHOLD_RAD, STEP_FOOT_TOP_MAX_TOLERANCE_M,
        STEP_FOOT_TOP_TOLERANCE_M, STEP_FOOT_TOP_XY_INSET_M,
        STEP_FRONT_CONTACT_BAND_M, STEP_FRONT_FEET_REWARD,
        STEP_STUCK_SPEED_MPS, STEP_STUCK_ZONE_M, STEP_TOP_CONTACT_MARGIN_M,
    )

STEP_STUCK_STEPS = 100
TERRAIN_SINGLE_STEP = "single_step"


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    """计算机身局部 +Z 轴与世界 +Z 轴的夹角，单位 rad。"""
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"四元数 shape 应为 (4,)，实际 {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("四元数不可为零或包含 NaN/Inf")
    w, x, y, z = quaternion / norm
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return float(np.arccos(np.clip(up_z, -1.0, 1.0)))


class TerrainTaskMixin:
    """提供单级台阶随机化、接触判定和任务进度方法。

    该 mixin 只读宿主模型/数据并更新宿主任务状态，不负责 Gym 返回值组装。
    重要性等级：A。
    """

    def _apply_terrain_randomization(self) -> None:
        """恢复或采样本 episode 的课程台阶几何。

        功能：
            每次 reset 都从 XML 编译后的原始几何重新计算台阶 X 和高度，避免在
            上一回合随机值上重复累加。关闭域随机化时恢复原始 0.025 m 台阶；开启
            时采样 X 中心 ±``step_x_randomization_m``、总高度
            ``0.025 ± step_height_randomization_m`` 的均匀扰动。
        返回：
            无。
        副作用：
            修改 ``model.geom_pos``/``geom_size``，调用 ``mj_setConst`` 刷新几何
            常量；随后由 ``reset()`` 调用 ``mj_forward`` 更新世界位姿和碰撞宽相位。
        关键参数：
            所有距离单位均为 m。X 扰动过大时台阶可能离开 LiDAR 高度图覆盖范围；
            高度扰动过大时会改变课程难度，甚至造成关键帧足底位置语义不一致。
        重要性等级：
            A（训练域随机化的唯一真值几何入口）。
        """
        self.applied_step_x_offset_m = 0.0
        self.applied_step_height_m = self.step_base_height_m
        if self.terrain != "single_step":
            return

        x_offset_m = 0.0
        total_height_m = self.step_base_height_m
        if self.domain_randomization:
            x_offset_m = float(
                self.np_random.uniform(
                    -self.step_x_randomization_m,
                    self.step_x_randomization_m,
                )
            )
            total_height_m = float(
                self.step_base_height_m
                + self.np_random.uniform(
                    -self.step_height_randomization_m,
                    self.step_height_randomization_m,
                )
            )

        bottom_z_m = float(
            self._step_geom_base_pos[2] - self._step_geom_base_size[2]
        )
        self.model.geom_pos[self.step_geom_id, 0] = (
            self._step_geom_base_pos[0] + x_offset_m
        )
        self.model.geom_pos[self.step_geom_id, 2] = bottom_z_m + total_height_m / 2.0
        geom_size = np.asarray(
            self.model.geom_size[self.step_geom_id], dtype=np.float64
        ).copy()
        geom_size[2] = total_height_m / 2.0
        self.model.geom_size[self.step_geom_id] = geom_size
        self.applied_step_x_offset_m = x_offset_m
        self.applied_step_height_m = total_height_m

        # geom_size 改变后必须刷新碰撞包围球等常量；否则宽相位可能沿用旧半径。
        mujoco.mj_setConst(self.model, self.data)

    def _reset_terrain_state(self) -> None:
        """根据当前 MuJoCo 几何重置台阶进度状态。

        返回：
            无。
        副作用：
            写入 ``step_crossed``、悬挂截断计数、越线奖励锁存、
            ``step_stuck_steps`` 和台阶世界坐标包围盒。
            单级台阶模式从 ``curriculum_step_01`` 的世界位姿计算前缘 X、顶面 Z
            及 X/Y 边界，避免在 Python 里重复写死 XML 尺寸。
        重要性等级：A（台阶成功与卡住判定依赖该基准位置）。
        """
        self.step_crossed = False
        self.step_stuck_steps = 0
        self.hanging_steps = 0
        self.hanging_triggered = False
        self.hanging_front_memory_steps = 0
        self.hanging_front_contact_latched = False
        self.step_base_crossing_rewarded = False
        self.last_step_base_crossing_reward = 0.0
        self.step_top_contact_latched = False
        self.step_top_visited_mask = np.zeros(self.foot_geom_ids.size, dtype=bool)
        self.front_feet_rewarded = False
        self.all_feet_rewarded = False
        self.last_step_front_feet_reward = 0.0
        self.last_step_all_feet_reward = 0.0
        self.last_step_stable_reward = 0.0
        self.last_step_edge_tilt_penalty = 0.0
        self.last_step_contact_penalty = 0.0
        self.foot_dead_leg_steps = np.zeros(self.foot_geom_ids.size, dtype=np.int32)
        self.continuous_airborne_steps = 0
        self.previous_foot_contacts = np.zeros(self.foot_geom_ids.size, dtype=bool)
        self.foot_placement_swing_steps = np.zeros(self.foot_geom_ids.size, dtype=np.int32)
        self.foot_placement_top_rewarded = np.zeros(self.foot_geom_ids.size, dtype=bool)
        self.rear_foot_top_rewarded = np.zeros(2, dtype=bool)
        if self.terrain != "single_step":
            self.step_front_x_m = None
            self.step_top_z_m = None
            self.step_x_min_m = None
            self.step_x_max_m = None
            self.step_y_min_m = None
            self.step_y_max_m = None
            return

        rotation = np.asarray(
            self.data.geom_xmat[self.step_geom_id], dtype=np.float64
        ).reshape(3, 3)
        half_size = np.asarray(
            self.model.geom_size[self.step_geom_id], dtype=np.float64
        )
        x_half_extent_m = float(np.sum(np.abs(rotation[0, :]) * half_size))
        y_half_extent_m = float(np.sum(np.abs(rotation[1, :]) * half_size))
        z_half_extent_m = float(np.sum(np.abs(rotation[2, :]) * half_size))
        step_center = np.asarray(
            self.data.geom_xpos[self.step_geom_id], dtype=np.float64
        )
        self.step_x_min_m = float(step_center[0] - x_half_extent_m)
        self.step_x_max_m = float(step_center[0] + x_half_extent_m)
        self.step_y_min_m = float(step_center[1] - y_half_extent_m)
        self.step_y_max_m = float(step_center[1] + y_half_extent_m)
        self.step_front_x_m = self.step_x_min_m
        self.step_top_z_m = float(step_center[2] + z_half_extent_m)

    def _get_step_contact_penalty(self) -> float:
        """计算机身或腿撞击台阶立面的单步接触力惩罚。

        返回：
            非负惩罚值，单位无量纲，范围 [0, STEP_CONTACT_PENALTY_MAX]。
        算法：
            遍历当前 MuJoCo 接触，只处理包含 ``curriculum_step_01`` 的接触对。
            足端接触台阶顶面属于正常攀爬，不罚；其他 Go2 碰撞 geom 的法向力
            超过阈值后按超出量累加，最后截断上限。
        数值边界：
            ``mujoco.mj_contactForce`` 只在已有接触上调用；无接触或平地模式
            返回 0。接触力可能在单步内出现尖峰，因此上限防止奖励爆炸。
        重要性等级：B（台阶碰撞塑形；错误地把足端算入会严重阻碍正常踩踏）。
        """
        if self.terrain != "single_step":
            return 0.0

        foot_geom_ids = {int(geom_id) for geom_id in self.foot_geom_ids}
        raw_penalty = 0.0
        force = np.zeros(6, dtype=np.float64)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.step_geom_id:
                other_geom = geom2
            elif geom2 == self.step_geom_id:
                other_geom = geom1
            else:
                continue
            if other_geom == self.step_geom_id or other_geom in foot_geom_ids:
                continue

            # 当前课程台阶轴对齐且来车方向为 -X。只惩罚前缘立面附近的接触，
            # 避免机器狗正常踩上顶面时被持续扣分。
            contact_position = np.asarray(contact.pos, dtype=np.float64)
            near_front_face = (
                abs(float(contact_position[0]) - float(self.step_front_x_m))
                <= STEP_FRONT_CONTACT_BAND_M
            )
            below_step_top = float(contact_position[2]) < (
                float(self.step_top_z_m) - STEP_TOP_CONTACT_MARGIN_M
            )
            if not (near_front_face and below_step_top):
                continue

            force.fill(0.0)
            mujoco.mj_contactForce(self.model, self.data, contact_index, force)
            normal_force_n = float(force[0])
            raw_penalty += max(0.0, normal_force_n - STEP_CONTACT_FORCE_THRESHOLD_N)
        return float(
            min(STEP_CONTACT_PENALTY_MAX, STEP_CONTACT_PENALTY_WEIGHT * raw_penalty)
        )

    def _get_step_foot_support(self) -> np.ndarray:
        """读取四足是否实际踩在课程台阶顶面。

        返回：
            shape=(4,)、dtype=bool，顺序为 FL/FR/RL/RR。True 表示该足端与
            ``curriculum_step_01`` 存在接触，且接触点同时落在台阶顶面的
            X/Y 内缩边界和 Z 容差范围内。
        算法：
            只检查“足端 geom—课程台阶 geom”的接触对，避免把地面接触或
            机身碰撞误判为登顶。接触点使用 MuJoCo ``contact.pos`` 的世界坐标，
            先要求 X/Y 位于顶面内缩 0.01 m 的范围内，再要求 Z 位于顶面
            -0.01～+0.02 m；X/Y 内缩可排除足端仅擦到竖直侧面的提前奖励。
        副作用：
            只读取 ``data.contact`` 和台阶几何，不推进物理。
        重要性等级：A（前足搭阶、四足登顶和站稳奖励的共同判据）。
        """
        support = np.zeros(self.foot_geom_ids.size, dtype=bool)
        if (
            self.terrain != "single_step"
            or self.step_top_z_m is None
            or self.step_x_min_m is None
            or self.step_x_max_m is None
            or self.step_y_min_m is None
            or self.step_y_max_m is None
        ):
            return support

        step_top_z_m = float(self.step_top_z_m)
        x_min_m = float(self.step_x_min_m) + STEP_FOOT_TOP_XY_INSET_M
        x_max_m = float(self.step_x_max_m) - STEP_FOOT_TOP_XY_INSET_M
        y_min_m = float(self.step_y_min_m) + STEP_FOOT_TOP_XY_INSET_M
        y_max_m = float(self.step_y_max_m) - STEP_FOOT_TOP_XY_INSET_M
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 == self.step_geom_id:
                other_geom = geom2
            elif geom2 == self.step_geom_id:
                other_geom = geom1
            else:
                continue
            foot_index = self._foot_geom_to_index.get(other_geom)
            if foot_index is None:
                continue
            contact_x_m = float(contact.pos[0])
            contact_y_m = float(contact.pos[1])
            contact_z_m = float(contact.pos[2])
            xy_on_top = (
                x_min_m <= contact_x_m <= x_max_m
                and y_min_m <= contact_y_m <= y_max_m
            )
            z_on_top = (
                step_top_z_m - STEP_FOOT_TOP_TOLERANCE_M
                <= contact_z_m
                <= step_top_z_m + STEP_FOOT_TOP_MAX_TOLERANCE_M
            )
            if xy_on_top and z_on_top:
                support[foot_index] = True
        return support

    def _update_terrain_progress(
        self, local_velocity_mps: np.ndarray
    ) -> dict[str, float | int | bool | tuple[bool, ...]]:
        """更新当前帧台阶支撑、悬挂和阶段奖励。

        参数：
            local_velocity_mps：shape=(3,)，机身局部线速度，单位 m/s。
        返回：
            当前帧地形判定与奖励分项；单级台阶模式额外包含
            ``currently_on_platform`` 和 ``hanging_triggered``。
        判定契约：
            1. 成功只由当前帧四足是否都实际踩在台面、且机身 Z 已达到台面高度
               决定，不读取 ``step_crossed`` 或历史触顶掩码。
            2. 任一只前足首次触顶后启动悬挂锁存；连续 50 个控制步内机身 Z
               未达到“初始机身 Z + 台阶高度 - 0.02 m”则终止。
            3. ``step_crossed`` 仅作为诊断字段保留，不参与成功判定。
        副作用：
            更新悬挂计数、前足奖励锁存和卡住计时，不推进物理。
        重要性等级：A（成功和悬挂终止的核心契约）。
        """
        empty_result: dict[str, float | int | bool | tuple[bool, ...]] = {
            "step_front_feet_on_top": False,
            "step_all_feet_on_top": False,
            "step_all_feet_visited_top": False,
            "step_all_feet_over_top": False,
            "step_base_on_top": False,
            "step_stable_on_top": False,
            "step_top_visited_mask": (False, False, False, False),
            "step_top_support_count": 0,
            "step_front_feet_reward": 0.0,
            "step_all_feet_reward": 0.0,
            "step_stable_reward": 0.0,
            "step_edge_tilt_penalty": 0.0,
            "step_contact_penalty": 0.0,
            "step_stuck_steps": 0,
            "step_stuck_paused_by_step_contact": False,
            "stuck_at_step": False,
            "step_base_crossing_reward": 0.0,
            "currently_on_platform": False,
            "hanging_steps": 0,
            "hanging_front_memory_steps": 0,
            "hanging_triggered": False,
        }
        if self.terrain != "single_step":
            self.hanging_steps = 0
            self.hanging_triggered = False
            self.hanging_front_memory_steps = 0
            self.hanging_front_contact_latched = False
            self.last_step_base_crossing_reward = 0.0
            self.last_step_front_feet_reward = 0.0
            self.last_step_all_feet_reward = 0.0
            self.last_step_stable_reward = 0.0
            self.last_step_edge_tilt_penalty = 0.0
            self.last_step_contact_penalty = 0.0
            return empty_result
        if self.step_front_x_m is None or self.step_top_z_m is None:
            raise RuntimeError("单级台阶模式缺少前缘或顶面几何")

        base_x_m = float(self.data.qpos[0])
        base_z_m = float(self.data.qpos[2])
        # 历史越线只保留给日志和兼容字段，任何奖励和终止均不依赖该状态。
        self.step_crossed = bool(
            base_x_m > self.step_front_x_m + STEP_CROSS_TOLERANCE_M
        )
        self.last_step_base_crossing_reward = 0.0

        foot_support = self._get_step_foot_support()
        front_feet_support = foot_support[:2]
        rear_feet_support = foot_support[2:]
        any_front_foot_on_top = bool(np.any(front_feet_support))
        self.step_top_visited_mask |= foot_support
        if np.any(foot_support):
            self.step_top_contact_latched = True

        self.last_step_front_feet_reward = 0.0
        if any_front_foot_on_top and not self.front_feet_rewarded:
            self.last_step_front_feet_reward = STEP_FRONT_FEET_REWARD
            self.front_feet_rewarded = True

        platform_z_threshold_m = (
            float(self.initial_base_z_m)
            + float(self.applied_step_height_m)
            - HANGING_BASE_Z_TOLERANCE_M
        )
        base_height_reached_platform = base_z_m >= platform_z_threshold_m
        currently_on_platform = bool(
            np.all(foot_support) and base_height_reached_platform
        )
        self.last_step_all_feet_reward = 0.0
        if currently_on_platform and not self.all_feet_rewarded:
            self.last_step_all_feet_reward = STEP_ALL_FEET_REWARD
            self.all_feet_rewarded = True

        if any_front_foot_on_top:
            self.hanging_front_contact_latched = True
        if self.hanging_front_contact_latched and not base_height_reached_platform:
            self.hanging_steps += 1
        else:
            self.hanging_steps = 0
            if base_height_reached_platform:
                self.hanging_front_contact_latched = False
        self.hanging_front_memory_steps = int(self.hanging_front_contact_latched)
        self.hanging_triggered = self.hanging_steps >= HANGING_WINDOW_STEPS

        tilt_rad = _tilt_rad(np.asarray(self.data.qpos[3:7], dtype=np.float64))
        horizontal_speed_mps = float(np.linalg.norm(local_velocity_mps[:2]))
        near_step_edge = abs(base_x_m - self.step_front_x_m) <= STEP_EDGE_DISTANCE_M
        self.last_step_edge_tilt_penalty = 0.0
        if near_step_edge and tilt_rad > STEP_EDGE_TILT_THRESHOLD_RAD:
            self.last_step_edge_tilt_penalty = STEP_EDGE_TILT_PENALTY

        distance_to_front_m = self.step_front_x_m - base_x_m
        near_uncrossed_step = 0.0 < distance_to_front_m <= STEP_STUCK_ZONE_M
        # 足端一旦真实触顶，本回合后续短暂抬脚也应继续豁免卡住计时，
        # 否则前足刚搭台就会因重心转移阶段的速度较低被误截断。
        step_stuck_paused_by_step_contact = bool(self.step_top_contact_latched)
        if step_stuck_paused_by_step_contact:
            self.step_stuck_steps = 0
        elif near_uncrossed_step and horizontal_speed_mps < STEP_STUCK_SPEED_MPS:
            self.step_stuck_steps += 1
        else:
            self.step_stuck_steps = 0

        self.last_step_contact_penalty = self._get_step_contact_penalty()
        self.last_step_stable_reward = 0.0
        return {
            "step_front_feet_on_top": bool(np.all(front_feet_support)),
            "step_all_feet_on_top": bool(np.all(foot_support)),
            "step_all_feet_visited_top": bool(np.all(self.step_top_visited_mask)),
            "step_all_feet_over_top": bool(np.all(foot_support)),
            "step_base_on_top": currently_on_platform,
            "step_stable_on_top": currently_on_platform,
            "step_top_visited_mask": tuple(bool(value) for value in self.step_top_visited_mask),
            "step_top_support_count": int(np.count_nonzero(foot_support)),
            "step_front_feet_reward": self.last_step_front_feet_reward,
            "step_all_feet_reward": self.last_step_all_feet_reward,
            "step_stable_reward": 0.0,
            "step_edge_tilt_penalty": self.last_step_edge_tilt_penalty,
            "step_contact_penalty": self.last_step_contact_penalty,
            "step_stuck_steps": self.step_stuck_steps,
            "step_stuck_paused_by_step_contact": step_stuck_paused_by_step_contact,
            "stuck_at_step": self.step_stuck_steps >= STEP_STUCK_STEPS,
            "step_base_crossing_reward": 0.0,
            "currently_on_platform": currently_on_platform,
            "hanging_steps": self.hanging_steps,
            "hanging_front_memory_steps": self.hanging_front_memory_steps,
            "hanging_triggered": self.hanging_triggered,
        }

    def _refresh_terrain_scan(self) -> None:
        """刷新一次当前地形感知结果。

        返回：
            无。单级台阶模式把最新 ``TerrainScan`` 写入
            ``self.last_terrain_scan``；平地模式保持为 None。
        可能异常：
            LiDAR site 缺失、``mj_ray`` 扫描或点云 shape 校验失败时直接抛出，
            不静默伪造零高度图。这样训练日志能立即暴露传感器故障。
        副作用：
            更新 LiDAR 包装器内部缓存，不推进 MuJoCo 物理。
        重要性等级：A（高度图必须来自真实渲染场景的射线结果）。
        """
        if self.terrain_perception is None:
            self.last_terrain_scan = None
            return
        raw_scan = self.terrain_perception.scan(self.data)
        if self.heightmap_noise_std_m <= 0.0:
            self.last_terrain_scan = raw_scan
            return
        # 先把噪声写入唯一的 TerrainScan，再让 observation 和 info 同时读取它。
        # 不能在两个读取位置分别采样，否则策略看到的图与日志诊断图不是同一帧。
        noise = self.np_random.normal(
            loc=0.0,
            scale=self.heightmap_noise_std_m,
            size=raw_scan.heightmap.shape,
        ).astype(np.float32)
        noisy_heightmap = np.clip(
            raw_scan.heightmap + noise,
            -2.0,
            2.0,
        ).astype(np.float32, copy=False)
        self.last_terrain_scan = TerrainScan(
            heightmap=noisy_heightmap,
            points_m=raw_scan.points_m.copy(),
            hit_count=int(raw_scan.hit_count),
            ray_count=int(raw_scan.ray_count),
            scan_ms=float(raw_scan.scan_ms),
        )

    def _get_terrain_info(self) -> dict:
        """返回可放进 Gymnasium info 的地形感知字段。

        返回：
            字典包含最新高度图、有效命中数和射线数。扫描耗时作为非确定性的性能
            遥测保留在 ``last_terrain_scan.scan_ms``，不放入 info，避免
            Gymnasium 的同种子确定性检查因微小计时抖动失败。平地模式中的
            高度图字段为 None；单级台阶模式中的高度图为 shape=(10,10)、
            dtype=float32 的独立副本。
        重要性等级：B（供回放日志、训练监控和地形感知闭环诊断）。
        """
        scan = self.last_terrain_scan
        if scan is None:
            return {
                "terrain_heightmap": None,
                "terrain_heightmap_hit_count": 0,
                "terrain_heightmap_ray_count": 0,
            }
        return {
            "terrain_heightmap": scan.heightmap.copy(),
            "terrain_heightmap_hit_count": int(scan.hit_count),
            "terrain_heightmap_ray_count": int(scan.ray_count),
        }
