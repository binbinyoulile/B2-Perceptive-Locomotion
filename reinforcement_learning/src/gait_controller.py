"""Go2 步态、关节目标与足端接触状态模块。

文件目的：
    把 Trot 先验、残差动作映射、足底高度、接触、抬脚和落足塑形从 Gym
    环境骨架中拆出。环境仍拥有 MuJoCo 状态，本模块通过 mixin 只操作该状态。
整体输入：
    ``Go2Env`` 实例及其中最新的 ``data``、关节地址和足端 geom 索引。
整体输出：
    12 维关节目标、4 维接触/足底高度及完整足端事件 info。
依赖与版本要求：
    Python 3.10+、NumPy 2.x、MuJoCo 3.x。
运行方式：
    由 ``Go2Env`` 继承并在 ``step()`` 与 ``obs_builder.py`` 中调用。
关键假设：
    足端顺序为 FL/FR/RL/RR，机身坐标系 +X 向前、+Z 向上。
已知限制：
    开环先验仍为固定频率和固定振幅，不随真实地形自适应。
重要性等级：A（动作映射或接触语义错误会直接破坏策略学习）。
"""

from __future__ import annotations

import math

import numpy as np

try:
    from .gait_config import (
        JOINT_LIMIT_MARGIN_RAD,
        STAND_JOINT_RAD,
        TROT_CALF_AMPLITUDE_RAD,
        TROT_FREQUENCY_HZ,
        TROT_HIP_AMPLITUDE_RAD,
        TROT_HIP_SIGNS,
        TROT_PAIR_PHASE_CYCLES,
        TROT_RAMP_TIME_S,
        TROT_RESIDUAL_SCALE,
        TROT_SWING_HEIGHT_M,
        TROT_SWING_KNEE_GAIN_RAD_PER_M,
        TROT_THIGH_AMPLITUDE_RAD,
        TROT_WARMUP_TIME_S,
    )
    from .reward_calculator import *  # noqa: F401,F403
    from .reward_calculator import ALL_FEET_AIRBORNE_PENALTY
    from .reward_calculator import CONTINUOUS_AIRBORNE_ESCALATION_BASE
    from .reward_calculator import CONTINUOUS_AIRBORNE_ESCALATION_CAP
    from .reward_calculator import CONTINUOUS_AIRBORNE_GRACE_STEPS
    from .reward_calculator import DEAD_LEG_FORWARD_SPEED_MPS
    from .reward_calculator import DEAD_LEG_PENALTY_PER_FOOT
    from .reward_calculator import DEAD_LEG_WINDOW_STEPS
    from .reward_calculator import DIAGONAL_ALTERNATION_REWARD
    from .reward_calculator import DIAGONAL_SUPPORT_REWARD_PER_STEP
    from .reward_calculator import FOOT_CLEARANCE_MIN_FORWARD_SPEED_MPS
    from .reward_calculator import FOOT_CLEARANCE_MIN_SUPPORT_FEET
    from .reward_calculator import FOOT_CLEARANCE_REWARD_PER_FOOT
    from .reward_calculator import FOOT_CLEARANCE_THRESHOLD_M
    from .reward_calculator import FOOT_PLACEMENT_NEAR_EDGE_DISTANCE_M
    from .reward_calculator import FOOT_PLACEMENT_PRE_EDGE_PENALTY
    from .reward_calculator import FOOT_PLACEMENT_SWING_MAX_REWARDED_STEPS
    from .reward_calculator import FOOT_PLACEMENT_SWING_REWARD_PER_FOOT
    from .reward_calculator import FOOT_PLACEMENT_TOP_LANDING_REWARD
    from .reward_calculator import REAR_FOOT_TOP_LANDING_REWARD
    from .reward_calculator import SAME_SIDE_SUPPORT_PENALTY_PER_STEP
    from .reward_calculator import TRIPOD_SUPPORT_PENALTY_PER_STEP
    from .reward_calculator import TROT_PHASE_CONTACT_REWARD_PER_STEP
    from .reward_calculator import TROT_PHASE_SIN_THRESHOLD
except ImportError:
    from gait_config import (  # type: ignore[no-redef]
        JOINT_LIMIT_MARGIN_RAD,
        STAND_JOINT_RAD,
        TROT_CALF_AMPLITUDE_RAD,
        TROT_FREQUENCY_HZ,
        TROT_HIP_AMPLITUDE_RAD,
        TROT_HIP_SIGNS,
        TROT_PAIR_PHASE_CYCLES,
        TROT_RAMP_TIME_S,
        TROT_RESIDUAL_SCALE,
        TROT_SWING_HEIGHT_M,
        TROT_SWING_KNEE_GAIN_RAD_PER_M,
        TROT_THIGH_AMPLITUDE_RAD,
        TROT_WARMUP_TIME_S,
    )
    from reward_calculator import *  # type: ignore[no-redef]  # noqa: F401,F403
    from reward_calculator import (  # type: ignore[no-redef]
        ALL_FEET_AIRBORNE_PENALTY, CONTINUOUS_AIRBORNE_ESCALATION_BASE,
        CONTINUOUS_AIRBORNE_ESCALATION_CAP, CONTINUOUS_AIRBORNE_GRACE_STEPS,
        DEAD_LEG_FORWARD_SPEED_MPS, DEAD_LEG_PENALTY_PER_FOOT,
        DEAD_LEG_WINDOW_STEPS, DIAGONAL_ALTERNATION_REWARD,
        DIAGONAL_SUPPORT_REWARD_PER_STEP, FOOT_CLEARANCE_MIN_FORWARD_SPEED_MPS,
        FOOT_CLEARANCE_MIN_SUPPORT_FEET, FOOT_CLEARANCE_REWARD_PER_FOOT,
        FOOT_CLEARANCE_THRESHOLD_M, FOOT_PLACEMENT_NEAR_EDGE_DISTANCE_M,
        FOOT_PLACEMENT_PRE_EDGE_PENALTY, FOOT_PLACEMENT_SWING_MAX_REWARDED_STEPS,
        FOOT_PLACEMENT_SWING_REWARD_PER_FOOT, FOOT_PLACEMENT_TOP_LANDING_REWARD,
        REAR_FOOT_TOP_LANDING_REWARD, SAME_SIDE_SUPPORT_PENALTY_PER_STEP,
        TRIPOD_SUPPORT_PENALTY_PER_STEP, TROT_PHASE_CONTACT_REWARD_PER_STEP,
        TROT_PHASE_SIN_THRESHOLD,
    )


class GaitMixin:
    """提供 Trot 先验、动作映射和足端状态更新方法。

    该 mixin 不拥有模型或生命周期；它假定宿主 ``Go2Env`` 已完成 reset 并
    保持 ``data`` 为最新物理状态。所有方法均为宿主内部接口。
    重要性等级：A。
    """

    def _get_base_linear_velocity_local(self) -> np.ndarray:
        """读取机身坐标系下的三轴线速度。

        返回：
            shape=(3,)，float64，单位 m/s；顺序为机身局部 X/Y/Z，X 指向机器人
            前进方向。MuJoCo 自由关节线速度先以世界坐标读取，再用机身旋转
            矩阵转置投影到局部坐标。
        重要性等级：A（观测和速度跟踪奖励共用该量）。
        """
        rotation = np.asarray(
            self.data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        world_linear_velocity = np.asarray(self.data.qvel[:3], dtype=np.float64)
        return rotation.T @ world_linear_velocity

    def _get_foot_contacts(self) -> np.ndarray:
        """读取 FL/FR/RL/RR 四足当前接触状态。

        返回：
            shape=(4,)，float32，值为 0.0 表示腾空、1.0 表示存在任何 MuJoCo
            接触对。该定义也适用于后续楼梯、平台和障碍物地形。
        数值边界：
            只读取 ``data.contact[:data.ncon]``；无接触时返回全零。该函数不会
            推进物理，也不修改 data。
        重要性等级：A（楼梯落脚判断和奖励扩展的关键观测）。
        """
        contacts = np.zeros(self.foot_geom_ids.size, dtype=np.float32)
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1_index = self._foot_geom_to_index.get(int(contact.geom1))
            geom2_index = self._foot_geom_to_index.get(int(contact.geom2))
            if geom1_index is not None:
                contacts[geom1_index] = 1.0
            if geom2_index is not None:
                contacts[geom2_index] = 1.0
        return contacts

    def _get_trot_phase_sin_cos(self) -> tuple[float, float]:
        """返回当前 Trot 全局相位的 sin/cos 观测。

        返回：
            ``(sin_phase, cos_phase)``，均为 float，范围 [-1, 1]，对应
            ``2*pi*TROT_FREQUENCY_HZ*trot_time_s``。FL+RR 组以该相位振荡，
            FR+RL 组由先验内部额外偏移半个周期。

        关键参数：
            观测使用全局相位而不是某条腿的局部相位，避免策略把两个相位组
            混为同一信号；``sin`` 的正负直接对应本环境新增的相位接触奖励。

        副作用：
            只读取 ``trot_time_s``，不推进时间、不修改物理状态。训练和回放
            必须保持同一时间基准，否则观测中的相位会和先验动作错位。

        重要性等级：A（175 维观测的第 148～149 维，决定策略能否踩准 Trot 节拍）。
        """
        gait_time_s = max(self.trot_time_s - TROT_WARMUP_TIME_S, 0.0)
        phase_rad = 2.0 * math.pi * TROT_FREQUENCY_HZ * gait_time_s
        return float(math.sin(phase_rad)), float(math.cos(phase_rad))

    def _get_trot_prior_joint_targets(self, time_s: float | None = None) -> np.ndarray:
        """计算当前时刻的开环 Trot 先验关节目标角。

        参数：
            time_s：可选浮点数，表示 Trot 时钟时刻，单位 s。传 None 时读取
                当前 ``self.trot_time_s``；物理子步循环传入子步时刻，避免一整个
                20 ms 控制周期使用冻结的关节目标角。

        返回：
            shape=(12,)，float64，单位 rad；顺序为 FL/FR/RL/RR 各自的
            hip/thigh/calf，与 ``JOINT_NAMES`` 完全一致。

        算法：
            每组对角腿共享相位，FR/RL 相对 FL/RR 偏移半周期。髋关节做小幅
            左右摆正，thigh 做前后摆动，calf 在摆动相按
            ``-sin(phase)`` 增加屈膝量。先站立 0.5 s，再用 smoothstep 在
            随后 0.8 s 内从站立姿平滑起振，使 reset 后第一步严格等于
            ``STAND_JOINT_RAD``，并与开环基线时序一致。

        关键参数：
            ``TROT_FREQUENCY_HZ=2.0`` 对应 2.0 个步态周期/秒；调高会增加
            单位时间步数但更容易失稳，调低则步幅/推进能力不足。
            注意：规则 FSM 教师内部的 FLAT_TROT 另用 1.8 Hz，两者不是同一
            时钟，比较节拍时必须先确认调用方。
            ``TROT_SWING_HEIGHT_M=0.045`` 是设计抬脚高度，不是实测足底高度。
            ``TROT_SWING_KNEE_GAIN_RAD_PER_M`` 把目标抬脚高度转为膝关节屈曲。

        副作用：
            无。时间推进由 ``step()`` 在 PHYSICS 目标计算后统一完成。

        重要性等级：A（决定基础节律和所有残差策略的动作语义）。
        """
        prior_time_s = float(self.trot_time_s if time_s is None else time_s)
        gait_time_s = max(prior_time_s - TROT_WARMUP_TIME_S, 0.0)
        phase_rad = (
            2.0
            * math.pi
            * TROT_FREQUENCY_HZ
            * gait_time_s
            + 2.0
            * math.pi
            * TROT_PAIR_PHASE_CYCLES
        )
        ramp_fraction = float(
            np.clip(gait_time_s / TROT_RAMP_TIME_S, 0.0, 1.0)
        )
        # smoothstep 的一阶导数在起点和终点为 0，避免起振时速度突变。
        ramp = ramp_fraction * ramp_fraction * (3.0 - 2.0 * ramp_fraction)
        oscillation = np.sin(phase_rad)
        swing = np.maximum(0.0, -oscillation)

        hip_rad = (
            TROT_HIP_AMPLITUDE_RAD * TROT_HIP_SIGNS * oscillation * ramp
        )
        # 站立基准取自 configs/robot_config.py 的 stand_joint_rad（hip/thigh/calf），
        # 不再硬编码 0.9/-1.8，换机器人时无需改步态公式。
        stand_thigh_rad = float(STAND_JOINT_RAD[1])
        stand_calf_rad = float(STAND_JOINT_RAD[2])
        thigh_rad = stand_thigh_rad + TROT_THIGH_AMPLITUDE_RAD * oscillation * ramp
        calf_rad = (
            stand_calf_rad
            + TROT_CALF_AMPLITUDE_RAD * oscillation * ramp
            - TROT_SWING_KNEE_GAIN_RAD_PER_M * TROT_SWING_HEIGHT_M * swing * ramp
        )
        return np.stack((hip_rad, thigh_rad, calf_rad), axis=1).reshape(-1)

    def _action_to_joint_targets(
        self,
        action: np.ndarray,
        prior_joint_rad: np.ndarray | None = None,
    ) -> np.ndarray:
        """把归一化残差动作叠加到 Trot 先验并裁剪到安全关节范围。

        参数：
            action：shape=(12,)，float64 数组，范围 [-1, 1]；0 表示完全信任
                Trot 先验。顺序为 FL/FR/RL/RR 各自的 hip/thigh/calf。
            prior_joint_rad：可选 shape=(12,)，单位 rad；由
                ``_get_trot_prior_joint_targets()`` 返回。传入可避免同一控制步
                重复计算相位，不传时使用当前 ``trot_time_s`` 现算。

        返回：
            shape=(12,)，float64，单位 rad；已裁剪到机械限位内侧的
            ``JOINT_LIMIT_MARGIN_RAD`` 安全边界。

        算法：
            ``target = trot_prior_rad + TROT_RESIDUAL_SCALE * action``。
            ``TROT_RESIDUAL_SCALE=0.08`` 是残差上限尺度；调大后网络可更快
            改写基础步态但更容易抖动，调小后更依赖固定先验。

        副作用：
            无，只做纯数值映射，不读写 MuJoCo 状态，也不推进 Trot 时间。

        重要性等级：A（动作语义变化会使旧绝对动作模型不兼容）。
        """
        values = np.asarray(action, dtype=np.float64)
        prior = (
            self._get_trot_prior_joint_targets()
            if prior_joint_rad is None
            else np.asarray(prior_joint_rad, dtype=np.float64)
        )
        if prior.shape != (self.joint_ids.size,):
            raise ValueError(f"Trot 先验 shape 应为 ({self.joint_ids.size},)，实际 {prior.shape}")
        target_rad = prior + TROT_RESIDUAL_SCALE * values
        safe_low = self.joint_low + JOINT_LIMIT_MARGIN_RAD
        safe_high = self.joint_high - JOINT_LIMIT_MARGIN_RAD
        return np.clip(target_rad, safe_low, safe_high)

    def _get_foot_bottom_heights(self) -> np.ndarray:
        """读取 FL/FR/RL/RR 足端球体最低点的世界高度。

        返回：
            shape=(4,)，float64，单位 m，顺序 FL/FR/RL/RR；
            ``bottom_z = geom_xpos.z - geom_size.radius``。
        可能异常：
            当前足端 geom 不是球体时由半径语义不匹配导致高度误差；本项目的
            Go2 官方模型使用球体足端，测试会检查数值有效性。
        副作用：
            只读取 MuJoCo 几何世界位姿，不推进物理。
        重要性等级：A（足端抬升奖励和回放诊断的共同测量入口）。
        """
        centers_z = np.asarray(
            self.data.geom_xpos[self.foot_geom_ids, 2], dtype=np.float64
        )
        radii = np.asarray(
            self.model.geom_size[self.foot_geom_ids, 0], dtype=np.float64
        )
        heights = centers_z - radii
        if not np.all(np.isfinite(heights)):
            raise RuntimeError("足底高度包含 NaN/Inf")
        return heights

    def _get_foot_support_heights(self) -> np.ndarray:
        """返回每只脚当前脚下的局部支撑面绝对高度。

        返回：
            shape=(4,)，float64，单位 m。足端 X/Y 位于课程台阶顶面范围内时，
            支撑面为当前随机化后的台阶顶面 Z；否则为平地 Z=0。
        算法：
            以足端球心世界 X/Y 判断是否位于台阶顶面矩形内。端到端抬脚奖励
            比较的是“足底高度 - 该支撑高度”，避免站上台阶后仍因绝对高度
            增加而持续刷抬脚奖励。
        副作用：
            无，只读取足端位姿和当前台阶几何。
        重要性等级：A（防止台阶站立状态产生虚假抬脚奖励）。
        """
        support = np.zeros(self.foot_geom_ids.size, dtype=np.float64)
        if (
            self.terrain != "single_step"
            or self.step_top_z_m is None
            or self.step_x_min_m is None
            or self.step_x_max_m is None
            or self.step_y_min_m is None
            or self.step_y_max_m is None
        ):
            return support
        foot_xy = np.asarray(
            self.data.geom_xpos[self.foot_geom_ids, :2], dtype=np.float64
        )
        over_step = (
            (foot_xy[:, 0] >= float(self.step_x_min_m))
            & (foot_xy[:, 0] <= float(self.step_x_max_m))
            & (foot_xy[:, 1] >= float(self.step_y_min_m))
            & (foot_xy[:, 1] <= float(self.step_y_max_m))
        )
        support[over_step] = float(self.step_top_z_m)
        return support

    def _update_foot_clearance_state(
        self,
        local_velocity_mps: np.ndarray,
        trot_phase_rad: float,
    ) -> dict[str, np.ndarray | float | int | tuple[int, ...]]:
        """更新足端离地、落足点奖励、Trot 支撑和触地/腾空计数。

        参数：
            local_velocity_mps：shape=(3,)，机身局部线速度，单位 m/s；只有
                X 方向速度超过 ``DEAD_LEG_FORWARD_SPEED_MPS`` 时才累计死腿。
            trot_phase_rad：float，当前控制步的 Trot 全局相位，单位 rad。该值
                来自动作执行前保存的相位，用来检查物理推进后的实际接触是否
                与预期半周期一致；传入错误相位会让节拍奖励失去意义。
        返回：
            字典包含足底高度 (4,)、支撑面高度 (4,)、离地余量 (4,)、接触
            (4,)、本步有效抬脚数、抬腿奖励、对角/同侧支撑奖励罚分、相位
            接触奖励、连续全腾空计数、全腾空总罚分、死腿计数数组和死腿惩罚。
        算法：
            1. 足底余量 = 足底高度 - 足下支撑面高度。
            2. 只有当至少 ``FOOT_CLEARANCE_MIN_SUPPORT_FEET`` 只脚接触地面，
               且机身局部前向速度超过 ``FOOT_CLEARANCE_MIN_FORWARD_SPEED_MPS``
               时，才允许统计无接触且余量超过 0.04 m 的摆动脚，每只脚奖励
               0.15；这同时封堵“两足站立、两足空摆”和原地摆腿刷分。
            3. 四足同时离地时，本步抬腿奖励强制为 0，并累计连续腾空步数。
               前 2 步每步返回 1.0 基础罚分；第 3 步起额外增加
               ``2 * 2**(n-3)``，并截断到 64.0，防止数值爆炸。
            4. 精确两足对角支撑 FL+RR 或 FR+RL 每步奖励 0.05；上一控制步
               为另一种精确对角且本步完成切换时，额外奖励 1.5。
            5. 当前相位 sin>0.35 且精确 FL+RR 支撑，或 sin<-0.35 且精确
               FR+RL 支撑时，再奖励 0.10；与第 4 条对角支撑奖励叠加。
            6. 精确同侧两足支撑每步罚 0.10，三足支撑每步罚 0.05，避免
               长期依赖同侧跳和拖步过渡。
            7. 仅前足在摆动相越过台阶前缘、至少两足支撑且机身确有前向
               速度时，每只前足最多奖励 3 步，每步 0.2；该奖励鼓励向前
               伸脚而不是原地高抬腿。后足不再获得任何离地或越缘持续奖励。
            8. 前足由腾空转为接触且实际踩在顶面时，每只前足一次性奖励 1.0。
            9. 机身 X 已越过前缘后，后足由腾空转为实际接触顶面时，每只后足
               只记录诊断事件；本轮不再发放一次性奖励。
            10. 新落足发生在台阶前缘之前且机身距前缘不足 0.6 m 时，每只脚
                罚 0.05，提示策略把落点送到更前方。
            11. 足端接触且机身向前运动时累计拖地步数；任一条件消失即清零。
            12. 连续拖地达到 12 个控制步后，每只脚每步惩罚 0.2。
        数值边界：
            接触来自 MuJoCo 实际 contact；平地、台阶顶面和腾空状态均使用同一
            判据。奖励只取决于当前物理状态，不使用物体真值或隐藏标签。
        重要性等级：A（端到端策略形成抬腿与交替支撑的核心塑形信号）。
        """
        contacts = self._get_foot_contacts()
        foot_bottom_z_m = self._get_foot_bottom_heights()
        support_z_m = self._get_foot_support_heights()
        clearance_m = foot_bottom_z_m - support_z_m
        contact_mask = contacts >= 0.5
        contact_count = int(np.count_nonzero(contact_mask))
        previous_contacts = self.previous_foot_contacts.copy()
        new_contact_mask = contact_mask & ~previous_contacts
        foot_x_m = np.asarray(
            self.data.geom_xpos[self.foot_geom_ids, 0], dtype=np.float64
        )
        foot_on_step_top = self._get_step_foot_support()
        support_ok = contact_count >= FOOT_CLEARANCE_MIN_SUPPORT_FEET
        clearance_motion_ok = (
            float(local_velocity_mps[0]) > FOOT_CLEARANCE_MIN_FORWARD_SPEED_MPS
        )
        airborne = ~contact_mask

        clear = airborne & (clearance_m > FOOT_CLEARANCE_THRESHOLD_M)
        clearance_count = (
            int(np.count_nonzero(clear))
            if support_ok and clearance_motion_ok
            else 0
        )
        clearance_reward = FOOT_CLEARANCE_REWARD_PER_FOOT * clearance_count

        # 摆动足越缘奖励只认可有支撑且机身前行的真实迈步，避免四足腾空或在
        # 原地抖动时刷分。计数只在实际满足“腾空且 X 已越过前缘”时累计：
        # 若脚尚未到前缘，则冻结计数而不提前消耗 3 步窗口；一旦触地立即清零。
        # 这样较长的 Trot 摆动相也不会在脚真正伸到台阶上方前把奖励窗口耗尽。
        placement_swing_steps_before = self.foot_placement_swing_steps.copy()
        front_feet_mask = np.array((True, True, False, False), dtype=bool)
        placement_swing_eligible = (
            airborne
            & front_feet_mask
            & (foot_x_m > float(self.step_front_x_m))
            & support_ok
            & clearance_motion_ok
        ) if self.step_front_x_m is not None else np.zeros(self.foot_geom_ids.size, dtype=bool)
        placement_swing_mask = placement_swing_eligible & (
            placement_swing_steps_before < FOOT_PLACEMENT_SWING_MAX_REWARDED_STEPS
        )
        foot_placement_swing_reward = (
            FOOT_PLACEMENT_SWING_REWARD_PER_FOOT
            * int(np.count_nonzero(placement_swing_mask))
        )
        self.foot_placement_swing_steps = np.where(
            contact_mask,
            0,
            np.where(
                placement_swing_eligible,
                placement_swing_steps_before + 1,
                placement_swing_steps_before,
            ),
        ).astype(np.int32, copy=False)

        # “新着地”定义为上一控制步无接触、当前控制步有接触。前足搭台由
        # 当前帧阶段奖励负责；后足只有机身重心已越过台阶前缘后首次落顶才
        # 发放 +3，防止后腿原地摆动刷分。
        foot_top_landing_flags = np.zeros(self.foot_geom_ids.size, dtype=bool)
        rear_foot_top_landing_flags = np.zeros(2, dtype=bool)
        foot_pre_edge_landing_flags = np.zeros(self.foot_geom_ids.size, dtype=bool)
        rear_base_crossed = False
        if self.terrain == "single_step" and self.step_front_x_m is not None:
            base_x_m = float(self.data.qpos[0])
            rear_base_crossed = base_x_m > float(self.step_front_x_m)
            near_front_edge = (
                base_x_m < float(self.step_front_x_m)
                and float(self.step_front_x_m) - base_x_m
                <= FOOT_PLACEMENT_NEAR_EDGE_DISTANCE_M
            )
            for foot_index in range(self.foot_geom_ids.size):
                if not bool(new_contact_mask[foot_index]):
                    continue
                if bool(foot_on_step_top[foot_index]):
                    if foot_index < 2:
                        if not bool(self.foot_placement_top_rewarded[foot_index]):
                            foot_top_landing_flags[foot_index] = True
                            self.foot_placement_top_rewarded[foot_index] = True
                    elif rear_base_crossed:
                        rear_index = foot_index - 2
                        if not bool(self.rear_foot_top_rewarded[rear_index]):
                            foot_top_landing_flags[foot_index] = True
                            rear_foot_top_landing_flags[rear_index] = True
                            self.rear_foot_top_rewarded[rear_index] = True
                elif near_front_edge and foot_x_m[foot_index] < float(self.step_front_x_m):
                    foot_pre_edge_landing_flags[foot_index] = True
            # 前足落顶奖励可在再次腾空后刷新；后足维持越线后的回合锁存。
            self.foot_placement_top_rewarded[:2][airborne[:2]] = False
        foot_placement_top_reward = (
            FOOT_PLACEMENT_TOP_LANDING_REWARD
            * int(np.count_nonzero(foot_top_landing_flags[:2]))
        )
        rear_foot_top_landing_reward = (
            REAR_FOOT_TOP_LANDING_REWARD
            * int(np.count_nonzero(rear_foot_top_landing_flags))
        )
        foot_placement_pre_edge_penalty = (
            FOOT_PLACEMENT_PRE_EDGE_PENALTY
            * int(np.count_nonzero(foot_pre_edge_landing_flags))
        )

        all_feet_airborne = contact_count == 0
        if all_feet_airborne:
            self.continuous_airborne_steps += 1
        else:
            self.continuous_airborne_steps = 0
        continuous_airborne_escalation_penalty = 0.0
        if (
            all_feet_airborne
            and self.continuous_airborne_steps > CONTINUOUS_AIRBORNE_GRACE_STEPS
        ):
            escalation_exponent = (
                self.continuous_airborne_steps - CONTINUOUS_AIRBORNE_GRACE_STEPS - 1
            )
            continuous_airborne_escalation_penalty = min(
                CONTINUOUS_AIRBORNE_ESCALATION_BASE * 2.0**escalation_exponent,
                CONTINUOUS_AIRBORNE_ESCALATION_CAP,
            )
        all_feet_airborne_penalty = (
            ALL_FEET_AIRBORNE_PENALTY if all_feet_airborne else 0.0
        ) + continuous_airborne_escalation_penalty

        exact_left_diagonal = (
            contact_count == 2 and bool(contact_mask[0]) and bool(contact_mask[3])
        )
        exact_right_diagonal = (
            contact_count == 2 and bool(contact_mask[1]) and bool(contact_mask[2])
        )
        exact_left_same_side = (
            contact_count == 2 and bool(contact_mask[0]) and bool(contact_mask[2])
        )
        exact_right_same_side = (
            contact_count == 2 and bool(contact_mask[1]) and bool(contact_mask[3])
        )
        diagonal_support = exact_left_diagonal or exact_right_diagonal
        same_side_support = exact_left_same_side or exact_right_same_side
        tripod_support = contact_count == 3
        current_diagonal_pattern = (
            1 if exact_left_diagonal else (2 if exact_right_diagonal else 0)
        )
        # 三足过渡是 Trot 摆腿时常见的中间态，不能因为相邻帧不是纯双足就
        # 丢掉完整对角切换。这里保留“最近一次精确对角模式”，允许中间经过
        # 三足或短暂腾空，只要下一次精确对角与上次相反就发放一次切换奖励。
        diagonal_alternation = bool(
            current_diagonal_pattern != 0
            and self.previous_diagonal_pattern != 0
            and current_diagonal_pattern != self.previous_diagonal_pattern
        )
        if current_diagonal_pattern != 0:
            self.previous_diagonal_pattern = current_diagonal_pattern
        diagonal_support_reward = (
            DIAGONAL_SUPPORT_REWARD_PER_STEP if diagonal_support else 0.0
        )
        diagonal_alternation_reward = (
            DIAGONAL_ALTERNATION_REWARD if diagonal_alternation else 0.0
        )
        phase_sin = float(math.sin(float(trot_phase_rad)))
        phase_left_support = (
            phase_sin > TROT_PHASE_SIN_THRESHOLD and exact_left_diagonal
        )
        phase_right_support = (
            phase_sin < -TROT_PHASE_SIN_THRESHOLD and exact_right_diagonal
        )
        trot_phase_contact_reward = (
            TROT_PHASE_CONTACT_REWARD_PER_STEP
            if phase_left_support or phase_right_support
            else 0.0
        )
        same_side_support_penalty = (
            SAME_SIDE_SUPPORT_PENALTY_PER_STEP if same_side_support else 0.0
        )
        tripod_support_penalty = (
            TRIPOD_SUPPORT_PENALTY_PER_STEP if tripod_support else 0.0
        )

        moving_forward = float(local_velocity_mps[0]) > DEAD_LEG_FORWARD_SPEED_MPS
        dragging = contact_mask & moving_forward
        self.foot_dead_leg_steps = np.where(
            dragging,
            self.foot_dead_leg_steps + 1,
            0,
        ).astype(np.int32, copy=False)
        dead_leg_mask = self.foot_dead_leg_steps >= DEAD_LEG_WINDOW_STEPS
        dead_leg_penalty = (
            DEAD_LEG_PENALTY_PER_FOOT * int(np.count_nonzero(dead_leg_mask))
        )
        self.previous_foot_contacts = contact_mask.copy()
        return {
            "foot_bottom_z_m": foot_bottom_z_m,
            "foot_support_z_m": support_z_m,
            "foot_clearance_m": clearance_m,
            "foot_contacts": contacts,
            "foot_contact_count": contact_count,
            "foot_support_ok": support_ok,
            "foot_clearance_motion_ok": clearance_motion_ok,
            "all_feet_airborne": all_feet_airborne,
            "foot_clearance_count": clearance_count,
            "foot_clearance_reward": float(clearance_reward),
            "foot_x_m": foot_x_m.copy(),
            "foot_on_step_top": foot_on_step_top.copy(),
            "foot_top_landing_flags": foot_top_landing_flags.copy(),
            "foot_pre_edge_landing_flags": foot_pre_edge_landing_flags.copy(),
            "foot_placement_swing_steps": self.foot_placement_swing_steps.copy(),
            "foot_placement_swing_reward": float(foot_placement_swing_reward),
            "foot_placement_top_reward": float(foot_placement_top_reward),
            "foot_placement_pre_edge_penalty": float(
                foot_placement_pre_edge_penalty
            ),
            "rear_base_crossed": bool(rear_base_crossed),
            "rear_foot_top_landing_flags": (
                rear_foot_top_landing_flags.copy()
            ),
            "rear_foot_top_landing_reward": float(rear_foot_top_landing_reward),
            "rear_foot_top_rewarded": self.rear_foot_top_rewarded.copy(),
            "diagonal_support": bool(diagonal_support),
            "same_side_support": bool(same_side_support),
            "tripod_support": bool(tripod_support),
            "diagonal_alternation": bool(diagonal_alternation),
            "diagonal_support_reward": float(diagonal_support_reward),
            "diagonal_alternation_reward": float(diagonal_alternation_reward),
            "trot_phase_sin": float(phase_sin),
            "trot_phase_contact_reward": float(trot_phase_contact_reward),
            "same_side_support_penalty": float(same_side_support_penalty),
            "tripod_support_penalty": float(tripod_support_penalty),
            "continuous_airborne_steps": int(self.continuous_airborne_steps),
            "continuous_airborne_escalation_penalty": float(
                continuous_airborne_escalation_penalty
            ),
            "all_feet_airborne_penalty": float(all_feet_airborne_penalty),
            "foot_dead_leg_steps": self.foot_dead_leg_steps.copy(),
            "foot_dead_leg_penalty": float(dead_leg_penalty),
        }
