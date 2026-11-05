"""Go2 平地与单级台阶“Trot 先验 + 残差动作”强化学习环境。

文件目的：
    把官方 Unitree Go2 MuJoCo 场景包装为 Gymnasium 环境。策略不再直接输出
    12 个关节的绝对目标角，而是在低速稳定 Trot 正弦节律上学习小幅残差微调：
    ``target_joint_rad = trot_prior_rad + TROT_RESIDUAL_SCALE * action``。
    平地模式训练前进、速度跟踪和交替支撑；单级台阶模式复用同一接口，并额外
    加入抬脚、搭阶、登顶和站稳奖励。

整体输入：
    ``terrain="single_step"`` 默认加载保留的 0.04 m 单级台阶场景；极简资产
    未包含平地场景，平地兼容模式必须显式传入可用 ``scene_path``。每个动作
    shape=(12,)、dtype=float32，范围 [-1.0, 1.0]，表示归一化残差。

整体输出：
    观测 shape=(175,)、dtype=float32；奖励为 float；``step()`` 返回 Gymnasium
    五元组 ``(observation, reward, terminated, truncated, info)``。观测拼接由
    ``obs_builder`` 负责，奖励由 ``reward_calculator`` 负责，本文件只保留环境
    生命周期、物理推进、当前帧判定与终止逻辑。

依赖与版本要求：
    Python 3.10+、gymnasium 1.2.1、mujoco 3.16.0、numpy 2.x。

运行方式：
    本文件不打开窗口；由平地/台阶训练脚本、评测脚本和测试脚本调用。

关键假设：
    1. Go2 自由基座名为 ``base``，机身局部 +X 为前进方向。
    2. 12 个电机顺序与 ``JOINT_NAMES`` 一致，按名称校验执行器映射。
    3. 控制频率为 50 Hz，每个控制步包含 10 个 0.002 s 物理步。
    4. Trot 先验频率为 2.0 Hz（唯一来源是 gait_config.TROT_FREQUENCY_HZ；
       历史注释中的 1.7 Hz 已废弃），FL+RR 与 FR+RL 相差半个周期；先验在
       ``TROT_RAMP_TIME_S`` 内平滑起振，避免 reset 后瞬间踢腿。
       注意：规则 FSM 教师自己的 FLAT_TROT 内部仍按 1.8 Hz 生成（
       fsm_stairs_controller._flat_trot_joint_targets），两者不是同一时钟，
       排查步态节拍问题时必须区分。
    5. 足端 geom 名称为 FL/FR/RL/RR；足底高度为球心 Z 减足端球半径。
    6. LiDAR 高度图 shape=(10,10)，展平后占观测第 48～147 维；第 148～149
       维为 Trot 相位，第 150～161 维为四足局部坐标，第 162 维为最近台阶距离，第 163～174 维为 12 个关节到物理限位的距离。

已知限制：
    先验是固定频率、固定振幅的开环节律，尚不随地形或速度自适应。0.025 m
    单级台阶不代表完整楼梯能力；地形随机化未覆盖摩擦、质量、载荷和传感器
    外参。175 维模型与旧 50/110/148/150/163 维模型不直接兼容，必须从头训练。

重要性等级：A（Go2 平地与台阶训练的公共环境入口）。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

try:
    from .gait_controller import GaitMixin
except ImportError:
    from gait_controller import GaitMixin  # type: ignore[no-redef]

# 训练脚本以 ``src`` 包导入，测试脚本则把 ``src`` 目录直接放进 sys.path；
# 两种入口都支持，避免测试与正式训练使用不同的模块解析路径。
try:
    from .terrain_perception import (
        HEIGHTMAP_COLS,
        HEIGHTMAP_ROWS,
        TerrainPerception,
        TerrainScan,
    )
except ImportError:
    from terrain_perception import (  # type: ignore[no-redef]
        HEIGHTMAP_COLS,
        HEIGHTMAP_ROWS,
        TerrainPerception,
        TerrainScan,
    )


try:
    from .obs_builder import (
        OBSERVATION_DIM,
        OBSERVATION_LAYOUT,
        TERRAIN_HEIGHTMAP_DIM,
        build_observation,
        build_observation_space_bounds,
        get_foot_local_positions,
        get_nearest_step_distance,
    )
    from .reward_calculator import *  # noqa: F401,F403 - 兼容旧入口的公开常量
    from .reward_calculator import compute_reward
except ImportError:
    from obs_builder import (  # type: ignore[no-redef]
        OBSERVATION_DIM,
        OBSERVATION_LAYOUT,
        TERRAIN_HEIGHTMAP_DIM,
        build_observation,
        build_observation_space_bounds,
        get_foot_local_positions,
        get_nearest_step_distance,
    )
    from reward_calculator import *  # type: ignore[no-redef]  # noqa: F401,F403
    from reward_calculator import compute_reward  # type: ignore[no-redef]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SINGLE_STEP_SCENE_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "mujoco"
    / "go2_stairs_static"
    / "scene_single_step_004.xml"
)

# 单级台阶训练的固定助跑距离。0.45 m 让四足在接触台阶前完成起振并保留
# 约一个步态周期的调整空间；过近会导致首步直接撞立面，过远则增加稀疏探索难度。
SINGLE_STEP_BIRTH_DISTANCE_M = 0.45

TERRAIN_FLAT = "flat"
TERRAIN_SINGLE_STEP = "single_step"
SUPPORTED_TERRAINS = (TERRAIN_FLAT, TERRAIN_SINGLE_STEP)
TERRAIN_KEYFRAMES = {
    TERRAIN_FLAT: "stand_home",
    TERRAIN_SINGLE_STEP: "single_step_approach",
}
STEP_GEOM_NAME = "curriculum_step_01"

# 训练时可按要求对台阶几何和高度图做轻量域随机化。默认值在显式开启后才
# 生效，诊断脚本可以保持完全确定的地形。X 偏移 5 cm 用来防止策略死记单一
# 台阶位置；本轮台阶基准高度为 4 cm，故高度只扰动 ±5 mm，避免随机上限
# 重新变成 3 cm 以上高台阶。5 mm 高度图噪声模拟真实雷达测距抖动。
DEFAULT_STEP_X_RANDOMIZATION_M = 0.05
DEFAULT_STEP_HEIGHT_RANDOMIZATION_M = 0.005
DEFAULT_HEIGHTMAP_NOISE_STD_M = 0.005

# 顺序与官方 go2.xml 的 12 个执行器一致；启动时按名称逐个校验。
JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")
ACTION_DIM = len(JOINT_NAMES)
LAST_ACTION_DIM = ACTION_DIM
BASE_LINEAR_VELOCITY_DIM = 3
FOOT_CONTACT_DIM = len(FOOT_GEOM_NAMES)
# Teacher-Student 融合模式的 PPO 残差缩放，单位 rad/动作单位。动作范围
# [-1, 1]，因此学生最多修正教师关节目标 ±0.10 rad（约 ±5.7°）。该值过大
# 会破坏 FSM 的稳定接触序列，过小则学生无法补偿 IK 裁剪和地形误差。
FSM_RESIDUAL_SCALE = 0.10

# 12 维动作表示 Trot 先验附近的无量纲残差。共享配置集中在 gait_config.py，
# 便于单独调参而不改 Gym 生命周期。
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


# PD 增益沿用平地站立基线。kp 过小会重力塌腿，过大易接触振荡；kd 过小会
# 落地抖动，过大则关节响应迟滞。单位分别为 N·m/rad 和 N·m·s/rad。
KP = 60.0
KD = 2.0

# 控制周期 0.02 s（50 Hz）。若连续 20 个控制步（0.4 s）机身世界 X 坐标
# 相对当前死区锚点净移动不足 0.02 m，就直接截断回合并要求策略重来，防止
# “原地交替踏步”靠存活奖励长期刷分。只要净位移达到阈值，锚点就前移。
CONTROL_FREQUENCY_HZ = 50.0
MAX_EPISODE_STEPS = 1000
DEAD_ZONE_WINDOW_STEPS = 20
DEAD_ZONE_MIN_X_DISPLACEMENT_M = 0.02
# Trot 先验有 0.8 s 的平滑启动斜坡。若从 reset 后第 1 步就开始累计 0.4 s
# 位移死区，固定步态尚未来得及建立推进力就会被误判为原地不动。这里
# 保留 50 个控制步（1.0 s）启动宽限，其中前 40 步覆盖 Trot 斜坡，后 10 步
# 用于观察初步位移；宽限之后仍严格要求每个 0.4 s 窗口至少前进 0.02 m。
DEAD_ZONE_GRACE_STEPS = 50
# 台前低速计时：50 Hz 下连续 100 个控制步（2 s）仍无有效进展才截断，
# 避免单次抬腿或接触调整被误判为卡死。
try:
    from .terrain_task import STEP_STUCK_STEPS, TerrainTaskMixin
except ImportError:
    from terrain_task import STEP_STUCK_STEPS, TerrainTaskMixin  # type: ignore[no-redef]
BASE_FALL_HEIGHT_M = 0.16
BASE_FALL_TILT_RAD = math.radians(60.0)

def _load_model(scene_path: Path) -> mujoco.MjModel:
    """编译指定 XML，并在加载期间临时切换到场景目录。

    参数：
        scene_path：场景 XML 的绝对路径。
    返回：
        编译后的 ``mujoco.MjModel``。
    可能异常：
        文件不存在时抛出 FileNotFoundError；XML、mesh 或 include 缺失时由
        MuJoCo 抛出编译异常。
    副作用：
        临时修改进程当前工作目录，函数结束前恢复；不要在其他线程同时加载模型。
    重要性等级：A（路径解析错误会导致所有环境初始化失败）。
    """
    scene_path = scene_path.resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"找不到 Go2 场景 XML：{scene_path}")

    previous_cwd = Path.cwd()
    try:
        os.chdir(scene_path.parent)
        return mujoco.MjModel.from_xml_path(scene_path.name)
    finally:
        os.chdir(previous_cwd)


def _tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    """计算机身局部 Z 轴与世界 Z 轴的夹角。

    参数：
        quaternion_wxyz：shape=(4,)，MuJoCo 顺序 w/x/y/z 的单位四元数。
    返回：
        倾角，单位 rad，范围 [0, pi]。
    可能异常：
        shape 不是 (4,) 或四元数长度接近 0 时抛出 ValueError。
    数值边界：
        计算 ``acos`` 前把余弦裁剪到 [-1, 1]，避免浮点误差产生 NaN。
    重要性等级：B（终止条件和姿态惩罚依赖该角度）。
    """
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,):
        raise ValueError(f"四元数 shape 应为 (4,)，实际 {quaternion.shape}")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-12:
        raise ValueError("四元数长度接近 0")
    w, x, y, z = quaternion / norm
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(float(np.clip(body_z_world_z, -1.0, 1.0))))


class Go2Env(GaitMixin, TerrainTaskMixin, gym.Env):
    """Go2 平地/单级台阶 Gymnasium 环境。

    功能：
        每个控制步接收 12 维归一化残差动作，先叠加 Trot 正弦先验得到
        PD 目标角，再施加力矩并推进 10 个 MuJoCo 物理步，返回 175 维观测、
        奖励和终止标志。台阶模式额外执行 143 条前向 LiDAR 射线，把 10×10
        局部高度图写入观测第 48～147 维；环境同时统计足端抬升、连续拖地、
        登阶、卡住、边缘倾角和立面碰撞。
    输入：
        scene_path：可选场景 XML 路径；为 ``None`` 时按 ``terrain`` 选择默认场景。
        max_episode_steps：最大控制步数，默认 1000，即 20 s。
        terrain：``"flat"`` 或 ``"single_step"``，默认平地，保持旧调用兼容。
        domain_randomization：是否在每个 episode 随机平移课程台阶并扰动高度；
            训练应开启，固定地形诊断和回放默认关闭。
        step_x_randomization_m：台阶 X 中心单边扰动幅度，单位 m，默认 0.05。
        step_height_randomization_m：台阶总高度单边扰动幅度，单位 m，默认 0.005。
        heightmap_noise_std_m：写入策略观测的高斯高度噪声标准差，单位 m，
            默认 0.0；训练入口显式设为 0.005。
    输出：
        observation shape=(175,)，float32；reward 为 float；info 包含奖励
        分项、足端高度/离地余量/拖地计数、台阶进度、终止原因及高度图。
    生命周期：
        ``reset()`` 按地形恢复 ``stand_home`` 或 ``single_step_approach`` 关键帧；
        ``close()`` 不持有外部资源。
    重要性等级：A（平地 PPO 与单级台阶课程共用的环境入口）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene_path: str | Path | None = None,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        terrain: str = TERRAIN_FLAT,
        domain_randomization: bool = False,
        step_x_randomization_m: float = DEFAULT_STEP_X_RANDOMIZATION_M,
        step_height_randomization_m: float = DEFAULT_STEP_HEIGHT_RANDOMIZATION_M,
        heightmap_noise_std_m: float = 0.0,
        controller_mode: str = "trot",
    ) -> None:
        """加载模型、校验关节映射并定义观测与动作空间。

        参数：
            scene_path：场景 XML 路径。``None`` 时由地形选择默认文件；显式传入
                自定义路径时优先使用该路径，并按当前目录规则解析相对路径。
            max_episode_steps：一个 episode 的最大控制步数，必须为正整数。
            terrain：地形模式，``"flat"`` 或 ``"single_step"``；决定默认场景、
                出生关键帧和台阶奖励分支。传错值会在加载模型前抛出 ValueError。
            domain_randomization：启用后每次 ``reset()`` 用环境随机数生成器采样
                台阶 X 偏移和总高度；固定种子可复现，关闭时恢复 XML 原始几何。
            step_x_randomization_m：台阶 X 中心均匀分布的单边扰动，单位 m；
                必须是有限非负数。调大后需确认台阶仍在雷达高度图覆盖范围内。
            step_height_randomization_m：台阶总高度均匀分布的单边扰动，单位 m；
                必须是有限非负数，且不能把基础台阶高度减到非正数。
            heightmap_noise_std_m：高斯测距噪声标准差，单位 m；0 表示关闭。
                噪声只污染策略观测/info，不修改 MuJoCo 真值几何。
            controller_mode：``"trot"`` 保持原 Trot 先验；``"fsm_residual"``
                使用规则 FSM 教师目标叠加 0.10 倍 PPO 残差。混合模式仅用于
                本轮 Teacher-Student 短测，两种模式的观测均为 175 维。
        可能异常：
            模型维度不是 nq=19、nv=18、nu=12，缺少关节/关键帧/足端/台阶 geom、
            LiDAR site，或控制周期不能由整数物理子步精确组成时抛出 ValueError；
            LiDAR site 缺失、mj_ray 扫描失败或模型维度异常时会让底层异常直接向上传播。
        副作用：
            创建并保存一份 MjModel/MjData；不打开窗口。
        重要性等级：A。
        """
        super().__init__()
        if not isinstance(max_episode_steps, int) or max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps 必须为正整数，实际 {max_episode_steps}")
        if terrain not in SUPPORTED_TERRAINS:
            raise ValueError(f"terrain 必须是 {SUPPORTED_TERRAINS}，实际 {terrain!r}")
        for name, value in (
            ("step_x_randomization_m", step_x_randomization_m),
            ("step_height_randomization_m", step_height_randomization_m),
            ("heightmap_noise_std_m", heightmap_noise_std_m),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} 必须为有限非负数，实际 {value!r}")

        self.terrain = terrain
        self.domain_randomization = bool(domain_randomization)
        self.step_x_randomization_m = float(step_x_randomization_m)
        self.step_height_randomization_m = float(step_height_randomization_m)
        self.heightmap_noise_std_m = float(heightmap_noise_std_m)
        if controller_mode not in ("trot", "fsm_residual"):
            raise ValueError(
                "controller_mode 必须是 'trot' 或 'fsm_residual'，"
                f"实际 {controller_mode!r}"
            )
        self.controller_mode = controller_mode
        if scene_path is None:
            if terrain == TERRAIN_FLAT:
                raise FileNotFoundError(
                    "极简资产未保留平地场景；请显式传入 scene_path，"
                    "或使用 terrain='single_step' 加载 0.04 m 台阶。"
                )
            scene_path = SINGLE_STEP_SCENE_PATH
        self.scene_path = Path(scene_path)
        self.model = _load_model(self.scene_path)
        self.data = mujoco.MjData(self.model)
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self._has_reset = False

        if (self.model.nq, self.model.nv, self.model.nu) != (19, 18, 12):
            raise ValueError(
                "Go2 模型维度应为 nq/nv/nu=19/18/12，"
                f"实际 {self.model.nq}/{self.model.nv}/{self.model.nu}"
            )

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                for joint_name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            raise ValueError(f"场景缺少 Go2 关节：{JOINT_NAMES}")

        self.qpos_addresses = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.dof_addresses = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        actuator_joint_ids = np.asarray(
            self.model.actuator_trnid[:, 0], dtype=np.int32
        )
        if not np.array_equal(actuator_joint_ids, self.joint_ids):
            raise ValueError("12 个电机顺序与 JOINT_NAMES 不一致")

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if self.base_body_id < 0:
            raise ValueError("场景缺少名为 base 的 Go2 机身")

        self.foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, foot_name
                )
                for foot_name in FOOT_GEOM_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.foot_geom_ids < 0):
            raise ValueError(f"场景缺少 Go2 足端 geom：{FOOT_GEOM_NAMES}")
        self._foot_geom_to_index = {
            int(geom_id): index for index, geom_id in enumerate(self.foot_geom_ids)
        }

        self.keyframe_name = TERRAIN_KEYFRAMES[self.terrain]
        self.keyframe_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, self.keyframe_name
        )
        if self.keyframe_id < 0:
            raise ValueError(
                f"{self.terrain} 场景缺少 {self.keyframe_name} 关键帧"
            )

        self.step_geom_id = -1
        # None 明确表示当前地形没有台阶；若使用 NaN，Gymnasium check_env 会因
        # NaN != NaN 误判 step() 的 info 非确定性。
        self.step_front_x_m: float | None = None
        self.step_top_z_m: float | None = None
        self.step_x_min_m: float | None = None
        self.step_x_max_m: float | None = None
        self.step_y_min_m: float | None = None
        self.step_y_max_m: float | None = None
        if self.terrain == TERRAIN_SINGLE_STEP:
            self.step_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, STEP_GEOM_NAME
            )
            if self.step_geom_id < 0:
                raise ValueError(f"单级台阶场景缺少 {STEP_GEOM_NAME} geom")
            self._step_geom_base_pos = np.asarray(
                self.model.geom_pos[self.step_geom_id], dtype=np.float64
            ).copy()
            self._step_geom_base_size = np.asarray(
                self.model.geom_size[self.step_geom_id], dtype=np.float64
            ).copy()
            self.step_base_height_m = float(2.0 * self._step_geom_base_size[2])
            if self.step_base_height_m <= self.step_height_randomization_m:
                raise ValueError(
                    "台阶高度随机化会把总高度降到非正数："
                    f"base={self.step_base_height_m}, "
                    f"range=±{self.step_height_randomization_m}"
                )
        else:
            self._step_geom_base_pos = np.zeros(3, dtype=np.float64)
            self._step_geom_base_size = np.zeros(3, dtype=np.float64)
            self.step_base_height_m = 0.0
        self.applied_step_x_offset_m = 0.0
        self.applied_step_height_m = self.step_base_height_m

        # 只有台阶模式需要地形感知。创建失败时直接报错，避免训练在“看似有
        # 高度图、实际全是默认值”的静默降级状态下继续消耗大量时间。
        self.terrain_perception: TerrainPerception | None = None
        self.last_terrain_scan: TerrainScan | None = None
        if self.terrain == TERRAIN_SINGLE_STEP:
            self.terrain_perception = TerrainPerception(self.model)

        control_dt_s = 1.0 / CONTROL_FREQUENCY_HZ
        substeps_float = control_dt_s / float(self.model.opt.timestep)
        self.n_substeps = int(round(substeps_float))
        if self.n_substeps <= 0 or not math.isclose(
            substeps_float, self.n_substeps, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "控制周期必须由整数个物理步组成："
                f"control_dt={control_dt_s}, timestep={self.model.opt.timestep}"
            )

        # 动作 [-1, 1] 是 Trot 先验附近的无量纲残差；0 表示不修正先验。
        # 映射见 _action_to_joint_targets，力矩另行裁剪到执行器范围。
        self.joint_low = np.asarray(
            self.model.jnt_range[self.joint_ids, 0], dtype=np.float64
        )
        self.joint_high = np.asarray(
            self.model.jnt_range[self.joint_ids, 1], dtype=np.float64
        )
        self.actuator_torque_low = np.asarray(
            self.model.actuator_ctrlrange[:, 0], dtype=np.float64
        )
        self.actuator_torque_high = np.asarray(
            self.model.actuator_ctrlrange[:, 1], dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(ACTION_DIM,),
            dtype=np.float32,
        )

        observation_low, observation_high = build_observation_space_bounds(self)
        self.observation_space = spaces.Box(
            low=observation_low.astype(np.float32),
            high=observation_high.astype(np.float32),
            shape=(OBSERVATION_DIM,),
            dtype=np.float32,
        )

        self.fsm_controller = None
        # 仅在 fsm_residual 模式启用；提前声明便于静态检查器识别生命周期，
        # Trot 分支不会读取这些字段。
        self.fsm_pd_kp: np.ndarray | None = None
        self.fsm_pd_kd: np.ndarray | None = None
        self.fsm_gravity_feedforward_scale: float = 0.0
        if self.controller_mode == "fsm_residual":
            if self.terrain != TERRAIN_SINGLE_STEP:
                raise ValueError("fsm_residual 当前只支持 single_step 地形")
            from fsm_stairs_controller import (
                GRAVITY_FEEDFORWARD_SCALE,
                PD_KD,
                PD_KP,
                RuleFsmController,
            )

            # Teacher 和混合控制器必须复用同一组 PD/重力前馈参数，否则同一
            # 教师目标在两套增益下会产生不同的接触力，无法比较纯 FSM 与融合效果。
            self.fsm_pd_kp = np.asarray(PD_KP, dtype=np.float64).copy()
            self.fsm_pd_kd = np.asarray(PD_KD, dtype=np.float64).copy()
            self.fsm_gravity_feedforward_scale = float(GRAVITY_FEEDFORWARD_SCALE)
            self.fsm_controller = RuleFsmController(
                self.model,
                keyframe_id=self.keyframe_id,
                scene_name="low",
            )

        self.initial_base_y_m = 0.0
        self.initial_base_x_m = 0.0
        self.initial_base_z_m = 0.0
        # 上一控制步机身 X 仅用于诊断；密集前移奖励使用历史最远机身 X，
        # 防止狗前后摆动时对同一段位移重复领奖。
        self.previous_base_x_m = 0.0
        self.best_base_x_m = 0.0
        # 悬挂截断只统计“至少一只前足触顶、但机身重心仍未越过台阶前缘”的
        # 连续控制步。触顶事件会短暂锁存，避免 Trot 抬脚造成计数闪断；
        # 锁存过期前机身越线则立即清零，防止跨事件累计或悬挂期继续领奖。
        self.hanging_steps = 0
        self.hanging_triggered = False
        self.hanging_front_memory_steps = 0
        self.hanging_front_contact_latched = False
        # 机身首次越线奖励是回合级一次性事件。锁存后即使前后摆动再次越线，
        # 也不会重复领取 +3.0；reset 时与悬挂状态一起清空。
        self.step_base_crossing_rewarded = False
        self.last_step_base_crossing_reward = 0.0
        self.last_action = np.zeros(LAST_ACTION_DIM, dtype=np.float32)
        # Trot 先验连续时间，只在控制步推进；reset() 必须显式归零，确保
        # 每个 episode 从站立姿平滑起振。
        self.trot_time_s = 0.0
        # 台阶状态每回合 reset；episode_steps 已由 Gymnasium/训练循环管理。
        self.step_crossed = False
        self.step_stuck_steps = 0
        # 任一脚首次接触台阶顶面后，本回合永久暂停低速截断。这里使用回合级
        # 锁存而不是逐帧判断，是因为收后腿时前足短暂腾空并不代表狗退回台阶
        # 前方；若立即恢复计时，恰好会在重心转移最需要时间的阶段误截断。
        self.step_top_contact_latched = False
        # 回合级历史掩码记录四只脚是否都曾真实接触台阶顶面。Trot 步态正常
        # 只有对角两足同时着地，因此不能再用“四足同一帧触顶”作为成功判据。
        self.step_top_visited_mask = np.zeros(FOOT_CONTACT_DIM, dtype=bool)
        self.front_feet_rewarded = False
        self.all_feet_rewarded = False
        self.last_step_front_feet_reward = 0.0
        self.last_step_all_feet_reward = 0.0
        self.last_step_stable_reward = 0.0
        self.last_step_edge_tilt_penalty = 0.0
        self.last_step_contact_penalty = 0.0
        self.foot_dead_leg_steps = np.zeros(FOOT_CONTACT_DIM, dtype=np.int32)
        # 连续全腾空计数跨控制步保存，接触地面后立即清零。第 1、2 步只收
        # 基础罚分，从第 3 步开始才叠加指数升级罚分。
        self.continuous_airborne_steps = 0
        # 死亡区锚点与连续计数只用于判断“原地摸鱼”，不进入策略观测，
        # 避免网络通过知道计时器状态来卡在截断前一帧。
        self.dead_zone_anchor_x_m = 0.0
        self.dead_zone_steps = 0
        # 0 表示尚未记录精确对角支撑，1 表示 FL+RR，2 表示 FR+RL。只保存
        # 最近一次精确对角模式，用于识别完整左右对角切换；中间允许三足过渡，
        # 但不构建逐帧强制的周期相位。
        self.previous_diagonal_pattern = 0
        # 落足点塑形需要跨控制步比较接触状态：上一帧接触用于识别“新着地”，
        # 摆动计数用于限制越缘悬空奖励。前足 top_rewarded 可在再次腾空后
        # 重新领取普通 +1；后足 rear_foot_top_rewarded 只在回合 reset 时清零，
        # 仅用于诊断并避免同一落顶事件重复计数，当前正式奖励为 0。
        self.previous_foot_contacts = np.zeros(FOOT_CONTACT_DIM, dtype=bool)
        self.foot_placement_swing_steps = np.zeros(FOOT_CONTACT_DIM, dtype=np.int32)
        self.foot_placement_top_rewarded = np.zeros(FOOT_CONTACT_DIM, dtype=bool)
        self.rear_foot_top_rewarded = np.zeros(2, dtype=bool)















    def _get_observation(self) -> np.ndarray:
        """构建 175 维观测，具体布局和拼接由 ``obs_builder`` 负责。"""
        return build_observation(self)

    def _compute_reward(
        self,
        action: np.ndarray,
        previous_action: np.ndarray,
        torque: np.ndarray,
        foot_contact_count: int = FOOT_CONTACT_DIM,
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
        """委托 ``reward_calculator`` 计算奖励，保持旧调用签名兼容。"""
        return compute_reward(
            self,
            action=action,
            previous_action=previous_action,
            torque=torque,
            foot_contact_count=foot_contact_count,
            step_front_feet_reward=step_front_feet_reward,
            step_all_feet_reward=step_all_feet_reward,
            step_stable_reward=step_stable_reward,
            step_edge_tilt_penalty=step_edge_tilt_penalty,
            step_contact_penalty=step_contact_penalty,
            foot_clearance_reward=foot_clearance_reward,
            diagonal_support_reward=diagonal_support_reward,
            diagonal_alternation_reward=diagonal_alternation_reward,
            same_side_support_penalty=same_side_support_penalty,
            tripod_support_penalty=tripod_support_penalty,
            all_feet_airborne_penalty=all_feet_airborne_penalty,
            dead_leg_penalty=dead_leg_penalty,
            trot_phase_contact_reward=trot_phase_contact_reward,
            foot_placement_swing_reward=foot_placement_swing_reward,
            foot_placement_top_reward=foot_placement_top_reward,
            foot_placement_pre_edge_penalty=foot_placement_pre_edge_penalty,
            rear_foot_top_landing_reward=rear_foot_top_landing_reward,
            base_progress_reward=base_progress_reward,
            step_base_crossing_reward=step_base_crossing_reward,
            hanging_base_advance_reward=hanging_base_advance_reward,
        )



    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """恢复当前地形的起跑关键帧并开始新 episode。

        参数：
            seed：Gymnasium 随机种子；启用域随机化时决定台阶 X/高度采样和
                高度图噪声序列，同 seed 可复现。
            options：Gymnasium 保留参数；当前未使用。
        返回：
            ``(observation, info)``；observation shape=(175,)，第 48～147 维是
            当前 LiDAR 高度图，第 148～149 维是 Trot 相位，第 150～161 维是
            足端局部坐标，第 162 维是最近台阶距离；info 记录场景、地形、
            关键帧、台阶几何、当前状态和高度图。
        副作用：
            覆盖 MjData、清零速度/控制/外力和上一动作，然后调用 mj_forward；
            同时清空台阶进度，不推进物理时间。
        重要性等级：A。
        """
        super().reset(seed=seed)
        del options
        # mj_setConst() 会重新初始化部分 MjData 状态；必须先改地形并刷新模型
        # 常量，再恢复关键帧，否则刚设置的起跑位姿会被清回 model.qpos0。
        self._apply_terrain_randomization()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.keyframe_id)
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # reset 顺序不能交换：先刷新随机化后的台阶世界几何，再根据实际前缘
        # 把机身放到固定助跑距离。这样 X 扰动不会改变“距前缘 0.45 m”的语义。
        self._reset_terrain_state()
        if self.terrain == TERRAIN_SINGLE_STEP and self.step_front_x_m is not None:
            self.data.qpos[0] = (
                float(self.step_front_x_m) - SINGLE_STEP_BIRTH_DISTANCE_M
            )
            mujoco.mj_forward(self.model, self.data)
        self.episode_steps = 0
        self.initial_base_x_m = float(self.data.qpos[0])
        self.initial_base_y_m = float(self.data.qpos[1])
        self.initial_base_z_m = float(self.data.qpos[2])
        self.previous_base_x_m = self.initial_base_x_m
        self.best_base_x_m = self.initial_base_x_m
        self.dead_zone_anchor_x_m = self.initial_base_x_m
        self.dead_zone_steps = 0
        self.previous_diagonal_pattern = 0
        self.last_action = np.zeros(LAST_ACTION_DIM, dtype=np.float32)
        self.trot_time_s = 0.0
        # mj_forward 后已有真实站立接触。用当前接触初始化上一帧，避免 reset 后
        # 第一步把四只已着地脚误判为“新落足”并领取顶面奖励。
        self.previous_foot_contacts = self._get_foot_contacts() >= 0.5
        self._has_reset = True
        if self.fsm_controller is not None:
            # 环境已完成域随机化和 0.45 m 助跑出生位姿，教师必须保留该状态。
            self.fsm_controller.reset(self.data, reset_keyframe=False)
        self._refresh_terrain_scan()
        reset_info = {
            "scene_path": str(self.scene_path),
            "terrain": self.terrain,
            "keyframe": self.keyframe_name,
            "step_front_x_m": self.step_front_x_m,
            "step_top_z_m": self.step_top_z_m,
            "domain_randomization": self.domain_randomization,
            "step_x_offset_m": self.applied_step_x_offset_m,
            "step_height_m": (
                self.applied_step_height_m
                if self.terrain == TERRAIN_SINGLE_STEP
                else None
            ),
            "heightmap_noise_std_m": self.heightmap_noise_std_m,
            "controller_mode": self.controller_mode,
            "initial_base_x_m": self.initial_base_x_m,
            "initial_birth_distance_to_step_m": (
                float(self.step_front_x_m) - self.initial_base_x_m
                if self.terrain == TERRAIN_SINGLE_STEP
                and self.step_front_x_m is not None
                else None
            ),
            "single_step_birth_distance_target_m": (
                SINGLE_STEP_BIRTH_DISTANCE_M
                if self.terrain == TERRAIN_SINGLE_STEP
                else None
            ),
            "dead_zone_window_steps": DEAD_ZONE_WINDOW_STEPS,
            "dead_zone_grace_steps": DEAD_ZONE_GRACE_STEPS,
            "dead_zone_min_x_displacement_m": DEAD_ZONE_MIN_X_DISPLACEMENT_M,
            "nearest_step_distance_m": get_nearest_step_distance(self),
            "foot_local_positions_m": tuple(
                float(value) for value in get_foot_local_positions(self)
            ),
        }
        reset_info.update(self._get_terrain_info())
        return self._get_observation(), reset_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """执行一个 50 Hz 控制步并返回 Gymnasium 五元组。

        参数：
            action：shape=(12,)，Trot 残差动作，范围 [-1, 1]，超出值会裁剪。
                实际 PD 目标按模式不同：
                ``trot`` 模式为 ``trot_prior_rad + 0.08 * action``（
                gait_config.TROT_RESIDUAL_SCALE）；
                ``fsm_residual`` 模式为 ``teacher_target_rad + 0.10 * action``
                （go2_env.FSM_RESIDUAL_SCALE）。动作 0 表示完全跟随当前先验/教师，
                不再表示绝对站立姿。
        返回：
            ``(observation, reward, terminated, truncated, info)``。
            terminated=True 表示当前帧四足均支撑在台面且机身 Z 达标
            （reason=success）、机身过低或倾角超过 60°（reason=fallen），或
            前足搭台后 50 个控制步内机身 Z 未上升（reason=hanging，扣 10 分）。
            truncated=True 表示连续 20 步净 X 位移不足 0.02 m、达到
            ``max_episode_steps``，或连续 2 s 低速卡在台阶前。info 含奖励分项、
            当前帧登顶、悬挂诊断、足端局部坐标、最近台阶距离、地形和终止原因。
        可能异常：
            reset 前调用或动作 shape 不是 (12,) 时抛出 RuntimeError/ValueError。
        副作用：
            每个物理子步都根据最新 Trot 相位、关节位置和速度重算 PD 目标与
            力矩并写入 data.ctrl，共推进 ``n_substeps`` 个物理步；不写
            qfrc_applied，因此没有仿真外力辅助。神经网络残差动作仍只在
            50 Hz 控制边界更新一次。
        重要性等级：A（训练和随机动作自检都依赖该接口）。
        """
        if not self._has_reset:
            raise RuntimeError("调用 step() 前必须先调用 reset()")

        action_array = np.asarray(action, dtype=np.float64)
        if action_array.shape != self.action_space.shape:
            raise ValueError(
                f"动作 shape 应为 {self.action_space.shape}，实际 {action_array.shape}"
            )
        action_clipped = np.clip(action_array, -1.0, 1.0)
        previous_action = self.last_action.copy()
        fsm_state_text = "DISABLED"
        fsm_event_text = ""
        teacher_target_rad = None
        if self.fsm_controller is not None:
            # 每 50 Hz 只更新一次教师；PPO 只提供小幅残差微调，不能接管
            # 完整步态。该调用只读物理状态并在内部做 LiDAR 扫描，不推进时间。
            self.fsm_controller.update_control_state(self.data)
            teacher_target_rad = self.fsm_controller.get_target_joint_rad()
            fsm_state_text = self.fsm_controller.state.value
            fsm_event_text = str(
                getattr(self.fsm_controller, "_quasi_event_name", "")
            )
        # 控制步起点用于策略观测和日志；真正下发给 PD 的 Trot 目标角会在
        # 每个 2 ms 物理子步按该子步时刻重新计算。这样既保留 50 Hz 策略接口，
        # 又复现已验证开环步态的高频目标跟踪，不再让 20 ms 内的目标角冻结。
        control_start_time_s = float(self.trot_time_s)
        gait_time_s = max(control_start_time_s - TROT_WARMUP_TIME_S, 0.0)
        trot_phase_rad = 2.0 * math.pi * TROT_FREQUENCY_HZ * gait_time_s
        trot_prior_rad = self._get_trot_prior_joint_targets(control_start_time_s)
        trot_residual_rad = TROT_RESIDUAL_SCALE * action_clipped
        self.trot_time_s += 1.0 / CONTROL_FREQUENCY_HZ

        # PD 力矩必须与 MuJoCo 物理子步同频计算。若只在 50 Hz 控制步起点
        # 计算一次，随后 10 个 0.002 s 子步保持同一个力矩，关节在实际接触
        # 负载下会明显跟踪滞后，Trot 蹬地推进也会衰减。这里保持“策略动作和
        # 目标角 50 Hz 更新一次”的接口不变，只在每个子步根据最新关节状态
        # 重算 PD 力矩；这与已验证的开环 Trot 基线一致，不属于外力辅助。
        torque = np.zeros(ACTION_DIM, dtype=np.float64)
        target_rad = (
            teacher_target_rad.copy()
            if teacher_target_rad is not None
            else trot_prior_rad.copy()
        )
        for substep_index in range(self.n_substeps):
            # 最后一个子步仍落在当前 20 ms 控制窗口内，因此分母使用
            # CONTROL_FREQUENCY_HZ * n_substeps，而不是物理步长常量。
            substep_time_s = control_start_time_s + substep_index / (
                CONTROL_FREQUENCY_HZ * self.n_substeps
            )
            joint_position_rad = np.asarray(
                self.data.qpos[self.qpos_addresses], dtype=np.float64
            )
            joint_velocity_rad_s = np.asarray(
                self.data.qvel[self.dof_addresses], dtype=np.float64
            )
            if teacher_target_rad is not None:
                # 固定教师目标，由 PPO 做最多 ±0.10 rad 的局部修正；随后
                # 保留 0.03 rad 安全余量，避免残差把关节推入机械限位。
                target_rad = np.clip(
                    teacher_target_rad + FSM_RESIDUAL_SCALE * action_clipped,
                    self.joint_low + 0.03,
                    self.joint_high - 0.03,
                )
                torque = (
                    self.fsm_pd_kp * (target_rad - joint_position_rad)
                    - self.fsm_pd_kd * joint_velocity_rad_s
                    + self.fsm_gravity_feedforward_scale
                    * np.asarray(
                        self.data.qfrc_bias[self.dof_addresses],
                        dtype=np.float64,
                    )
                )
            else:
                substep_prior_rad = self._get_trot_prior_joint_targets(
                    substep_time_s
                )
                target_rad = self._action_to_joint_targets(
                    action_clipped,
                    prior_joint_rad=substep_prior_rad,
                )
                torque = (
                    KP * (target_rad - joint_position_rad)
                    - KD * joint_velocity_rad_s
                )
            torque = np.clip(
                torque,
                self.actuator_torque_low,
                self.actuator_torque_high,
            )
            self.data.ctrl[:] = torque
            mujoco.mj_step(self.model, self.data)

        self.episode_steps += 1
        self.last_action = action_clipped.astype(np.float32)
        local_velocity = self._get_base_linear_velocity_local()
        base_x_m = float(self.data.qpos[0])
        # 只奖励“历史最远前沿”的增量。若使用相邻帧差值，狗前后摆动就能
        # 对同一段物理位移反复领奖，形成与任务目标无关的奖励漏洞。
        new_frontier_dx_m = max(base_x_m - self.best_base_x_m, 0.0)
        base_progress_reward = 0.0
        if self.terrain == TERRAIN_SINGLE_STEP:
            base_progress_reward = BASE_PROGRESS_REWARD_PER_STEP * float(
                np.clip(
                    new_frontier_dx_m / BASE_PROGRESS_DELTA_FOR_MAX_REWARD_M,
                    0.0,
                    1.0,
                )
            )
        self.best_base_x_m = max(self.best_base_x_m, base_x_m)
        self.previous_base_x_m = base_x_m
        terrain_progress = self._update_terrain_progress(local_velocity)
        # 前足触顶后的短暂锁存期内，按“历史最远机身 X”的增量追加推进奖励。
        # 该奖励只在重心尚未越线时生效，并与普通 base_progress 使用同一
        # frontier 增量，避免前后摆动对同一段位移重复刷分。
        hanging_base_advance_reward = 0.0
        if (
            self.terrain == TERRAIN_SINGLE_STEP
            and int(terrain_progress["hanging_front_memory_steps"]) > 0
            and not bool(terrain_progress["step_base_on_top"])
        ):
            hanging_base_advance_reward = (
                HANGING_BASE_ADVANCE_REWARD_PER_M * new_frontier_dx_m
            )
        foot_progress = self._update_foot_clearance_state(
            local_velocity,
            trot_phase_rad,
        )
        self._refresh_terrain_scan()
        observation = self._get_observation()
        reward, reward_terms = self._compute_reward(
            action=action_clipped,
            previous_action=previous_action,
            torque=torque,
            foot_contact_count=int(foot_progress["foot_contact_count"]),
            step_front_feet_reward=float(
                terrain_progress["step_front_feet_reward"]
            ),
            step_all_feet_reward=float(
                terrain_progress["step_all_feet_reward"]
            ),
            step_stable_reward=float(terrain_progress["step_stable_reward"]),
            step_edge_tilt_penalty=float(
                terrain_progress["step_edge_tilt_penalty"]
            ),
            step_contact_penalty=float(terrain_progress["step_contact_penalty"]),
            foot_clearance_reward=float(foot_progress["foot_clearance_reward"]),
            diagonal_support_reward=float(
                foot_progress["diagonal_support_reward"]
            ),
            same_side_support_penalty=float(
                foot_progress["same_side_support_penalty"]
            ),
            all_feet_airborne_penalty=float(
                foot_progress["all_feet_airborne_penalty"]
            ),
            diagonal_alternation_reward=float(
                foot_progress["diagonal_alternation_reward"]
            ),
            trot_phase_contact_reward=float(
                foot_progress["trot_phase_contact_reward"]
            ),
            tripod_support_penalty=float(
                foot_progress["tripod_support_penalty"]
            ),
            dead_leg_penalty=float(foot_progress["foot_dead_leg_penalty"]),
            foot_placement_swing_reward=float(
                foot_progress["foot_placement_swing_reward"]
            ),
            foot_placement_top_reward=float(
                foot_progress["foot_placement_top_reward"]
            ),
            foot_placement_pre_edge_penalty=float(
                foot_progress["foot_placement_pre_edge_penalty"]
            ),
            rear_foot_top_landing_reward=float(
                foot_progress["rear_foot_top_landing_reward"]
            ),
            base_progress_reward=base_progress_reward,
            step_base_crossing_reward=float(
                terrain_progress["step_base_crossing_reward"]
            ),
            hanging_base_advance_reward=float(
                hanging_base_advance_reward
            ),
        )
        dead_zone_displacement_m = base_x_m - self.dead_zone_anchor_x_m
        # 只认可向前净位移。若使用绝对值，策略可通过前后小幅振荡不断重置
        # 计时器，继续规避“原地摸鱼”截断，因此这里只累计 +X 方向进展。
        # 回合内首次触顶后持续暂停死区计时。不能用“本帧任一脚仍触顶”作为
        # 条件，因为策略抬脚收后腿的瞬间会解除暂停，反而截断正在完成的攀爬。
        any_foot_on_step_top = bool(np.any(foot_progress["foot_on_step_top"]))
        if any_foot_on_step_top:
            self.step_top_contact_latched = True
        dead_zone_paused_by_step_contact = self.step_top_contact_latched

        # 第二类豁免面向“已看见台阶且正在进行抬脚迈步”的短时低速状态：
        # 雷达高度图中各前向距离行的中位高度出现足够突变，说明台阶已进入
        # 视野；同时任一足端余量超过 2 cm，说明狗不是完全静止拖地。满足
        # 两项条件时暂停 dead_zone 和台前 stuck 计时，避免正常抬腿时误杀。
        stepping_in_progress = bool(
            float(np.max(foot_progress["foot_clearance_m"])) > 0.02
        )
        radar_step_visible = False
        if self.last_terrain_scan is not None:
            scan_heightmap = np.nan_to_num(
                self.last_terrain_scan.heightmap,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            row_median_heights = np.median(scan_heightmap, axis=1)
            radar_step_visible = bool(
                float(np.ptp(row_median_heights)) > 0.015
            )
        dead_zone_paused_by_radar_stepping = bool(
            self.terrain == TERRAIN_SINGLE_STEP
            and stepping_in_progress
            and radar_step_visible
        )
        if dead_zone_paused_by_radar_stepping:
            # stuck_at_step 与 dead_zone 共用“低速但可能在调整落足”的风险，
            # 因此同一豁免也清零台前计时，避免雷达已看见台阶仍被另一计时器截断。
            self.step_stuck_steps = 0
            terrain_progress["step_stuck_steps"] = 0
            terrain_progress["stuck_at_step"] = False

        # FSM 教师的前半段是原地逐足调整和重心转移，机身 X 可能连续数十步
        # 不产生净位移。这不是“原地摸鱼”，若套用面向端到端策略的死区计时，
        # 会在 HIGH_STEP 的准备段误杀，导致教师无法开始真正的摆腿。
        dead_zone_paused_by_fsm_teacher = (
            self.controller_mode == "fsm_residual"
        )
        dead_zone_paused = bool(
            dead_zone_paused_by_step_contact
            or dead_zone_paused_by_radar_stepping
            or dead_zone_paused_by_fsm_teacher
        )
        if dead_zone_paused:
            self.dead_zone_anchor_x_m = base_x_m
            self.dead_zone_steps = 0
            dead_zone_displacement_m = 0.0
        elif self.episode_steps <= DEAD_ZONE_GRACE_STEPS:
            # 启动宽限期内只移动锚点，不累计失败窗口；否则 Trot 尚未起速的
            # 前 20 步会错误触发 dead_zone，导致任何策略都只能存活 0.4 s。
            self.dead_zone_anchor_x_m = base_x_m
            self.dead_zone_steps = 0
            dead_zone_displacement_m = 0.0
        elif dead_zone_displacement_m >= DEAD_ZONE_MIN_X_DISPLACEMENT_M:
            self.dead_zone_anchor_x_m = base_x_m
            self.dead_zone_steps = 0
            dead_zone_displacement_m = 0.0
        else:
            self.dead_zone_steps += 1
        dead_zone_triggered = self.dead_zone_steps >= DEAD_ZONE_WINDOW_STEPS
        height_m = float(self.data.qpos[2])
        tilt_rad = _tilt_rad(np.asarray(self.data.qpos[3:7], dtype=np.float64))
        # 成功必须由当前帧物理状态直接证明：每只脚最近一个步态周期内都真实
        # 支撑在台面，且连续 SUCCESS_DWELL_STEPS 步保持台面高度（B1 口径，
        # 见 terrain_task.PlatformDwellTracker）。History/step_crossed 只保留
        # 诊断，不参与 success。
        success = bool(terrain_progress["success"])
        fallen = (
            height_m < BASE_FALL_HEIGHT_M
            or tilt_rad > BASE_FALL_TILT_RAD
        ) and not success
        # FSM 教师采用准静态逐足换支撑，1 s 的端到端悬挂阈值会在一只前足
        # 短暂卸载时误杀尚在执行的教师轨迹；教师模式只保留摔倒/成功终止。
        # PPO 残差仍通过关节限位、动作平滑和摔倒条件受到约束。
        hanging_triggered = bool(
            terrain_progress["hanging_triggered"]
        ) and not success and self.controller_mode != "fsm_residual"
        # 仅靠逐步负奖励时，策略可能故意快速摔倒或悬挂以缩短回合；终止瞬间
        # 追加一次性惩罚，使“继续尝试登顶”严格优于提前结束 episode。
        fall_termination_penalty = FALL_TERMINATION_PENALTY if fallen else 0.0
        hanging_termination_penalty = (
            HANGING_TERMINATION_PENALTY if hanging_triggered else 0.0
        )
        reward -= fall_termination_penalty + hanging_termination_penalty
        reward_terms["fall_termination_penalty"] = float(
            fall_termination_penalty
        )
        reward_terms["hanging_termination_penalty"] = float(
            hanging_termination_penalty
        )
        reward_terms["total_reward"] = float(reward)
        stuck_at_step = bool(terrain_progress["stuck_at_step"]) and not success
        terminated = bool(success or fallen or hanging_triggered)
        truncated = bool(
            not terminated
            and (
                dead_zone_triggered
                or stuck_at_step
                or self.episode_steps >= self.max_episode_steps
            )
        )
        if success:
            termination_reason = "success"
        elif fallen:
            termination_reason = "fallen"
        elif hanging_triggered:
            termination_reason = "hanging"
        elif dead_zone_triggered:
            termination_reason = "dead_zone"
        elif stuck_at_step:
            termination_reason = "stuck_at_step"
        elif self.episode_steps >= self.max_episode_steps:
            termination_reason = "time_limit"
        else:
            termination_reason = "running"
        info = {
            "step": self.episode_steps,
            "terrain": self.terrain,
            "height_m": height_m,
            "base_x_m": base_x_m,
            "initial_base_x_m": self.initial_base_x_m,
            "base_progress_delta_m": float(new_frontier_dx_m),
            "base_progress_reward": float(base_progress_reward),
            "best_base_x_m": float(self.best_base_x_m),
            "distance_to_step_front_m": (
                float(self.step_front_x_m) - base_x_m
                if self.terrain == TERRAIN_SINGLE_STEP
                and self.step_front_x_m is not None
                else None
            ),
            "dead_zone_steps": int(self.dead_zone_steps),
            "dead_zone_displacement_m": float(dead_zone_displacement_m),
            "dead_zone_paused_by_step_contact": bool(
                dead_zone_paused_by_step_contact
            ),
            "dead_zone_paused_by_radar_stepping": bool(
                dead_zone_paused_by_radar_stepping
            ),
            "dead_zone_paused_by_fsm_teacher": bool(
                dead_zone_paused_by_fsm_teacher
            ),
            "radar_step_visible": bool(radar_step_visible),
            "stepping_in_progress": bool(stepping_in_progress),
            "step_top_contact_latched": bool(self.step_top_contact_latched),
            "dead_zone_triggered": bool(dead_zone_triggered),
            "tilt_rad": tilt_rad,
            "fallen": fallen,
            "stuck_at_step": stuck_at_step,
            "hanging_steps": int(terrain_progress["hanging_steps"]),
            "hanging_front_memory_steps": int(
                terrain_progress["hanging_front_memory_steps"]
            ),
            "hanging_base_advance_reward": float(
                hanging_base_advance_reward
            ),
            "hanging_triggered": hanging_triggered,
            "hanging_window_steps": HANGING_WINDOW_STEPS,
            "step_base_crossing_reward": float(
                terrain_progress["step_base_crossing_reward"]
            ),
            "step_base_crossing_rewarded": bool(
                self.step_base_crossing_rewarded
            ),
            "step_front_x_m": self.step_front_x_m,
            "step_top_z_m": self.step_top_z_m,
            "step_x_offset_m": self.applied_step_x_offset_m,
            "step_height_m": (
                self.applied_step_height_m
                if self.terrain == TERRAIN_SINGLE_STEP
                else None
            ),
            "domain_randomization": self.domain_randomization,
            "heightmap_noise_std_m": self.heightmap_noise_std_m,
            "step_crossed": self.step_crossed,
            "currently_on_platform": bool(
                terrain_progress["currently_on_platform"]
            ),
            "success": success,
            "platform_dwell_steps": int(
                terrain_progress["platform_dwell_steps"]
            ),
            "platform_foot_age_steps": tuple(
                int(value)
                for value in terrain_progress["platform_foot_age_steps"]
            ),
            "nearest_step_distance_m": get_nearest_step_distance(self),
            "foot_local_positions_m": tuple(
                float(value) for value in get_foot_local_positions(self)
            ),
            "step_front_feet_on_top": bool(
                terrain_progress["step_front_feet_on_top"]
            ),
            "step_all_feet_on_top": bool(
                terrain_progress["step_all_feet_on_top"]
            ),
            "step_all_feet_visited_top": bool(
                terrain_progress["step_all_feet_visited_top"]
            ),
            "step_all_feet_over_top": bool(
                terrain_progress["step_all_feet_over_top"]
            ),
            "step_base_on_top": bool(terrain_progress["step_base_on_top"]),
            "step_stable_on_top": bool(terrain_progress["step_stable_on_top"]),
            "step_top_visited_mask": tuple(
                bool(value) for value in terrain_progress["step_top_visited_mask"]
            ),
            "step_top_support_count": int(terrain_progress["step_top_support_count"]),
            "step_stuck_steps": int(terrain_progress["step_stuck_steps"]),
            "step_stuck_paused_by_step_contact": bool(
                terrain_progress["step_stuck_paused_by_step_contact"]
            ),
            "termination_reason": termination_reason,
            "target_forward_speed_mps": TARGET_FORWARD_SPEED_MPS,
            # 记录环境实际执行的裁剪后动作，供训练日志计算饱和率。PPO 原始
            # 高斯样本可能超出 [-1, 1]，但真正送入 PD/物理的是 action_clipped；
            # 若统计原值会产生 100% 饱和的假象。
            "last_action": tuple(float(value) for value in action_clipped),
            "trot_phase_rad": float(trot_phase_rad),
            # info 面向本控制步的奖励诊断，必须使用计算先验和接触奖励时的
            # 当前相位；返回 observation 则已进入下一决策时刻，二者故意相差
            # 一个控制周期。
            "trot_phase_sin_cos": (
                float(math.sin(trot_phase_rad)),
                float(math.cos(trot_phase_rad)),
            ),
            "trot_phase_contact_reward": float(
                foot_progress["trot_phase_contact_reward"]
            ),
            "trot_frequency_hz": TROT_FREQUENCY_HZ,
            "trot_prior_joint_rad": tuple(
                float(value)
                for value in (
                    teacher_target_rad
                    if teacher_target_rad is not None
                    else trot_prior_rad
                )
            ),
            "trot_residual_rad": tuple(
                float(value)
                for value in (
                    FSM_RESIDUAL_SCALE * action_clipped
                    if teacher_target_rad is not None
                    else trot_residual_rad
                )
            ),
            "controller_mode": self.controller_mode,
            "fsm_state": fsm_state_text,
            "fsm_event": fsm_event_text,
            "teacher_target_joint_rad": (
                tuple(float(value) for value in teacher_target_rad)
                if teacher_target_rad is not None
                else None
            ),
            "residual_joint_rad": (
                tuple(float(value) for value in FSM_RESIDUAL_SCALE * action_clipped)
                if teacher_target_rad is not None
                else None
            ),
            "joint_limit_penalty": float(
                reward_terms.get("joint_limit_penalty", 0.0)
            ),
            "joint_limit_max_usage_ratio": float(
                reward_terms.get("joint_limit_max_usage_ratio", 0.0)
            ),
            "joint_limit_excess_count": int(
                reward_terms.get("joint_limit_excess_count", 0)
            ),
            "fsm_ik_clip_count": int(
                self.fsm_controller.metrics.ik_clip_count
                if self.fsm_controller is not None
                else 0
            ),
            "fsm_ik_clip_max_rad": float(
                self.fsm_controller.metrics.ik_clip_max_rad
                if self.fsm_controller is not None
                else 0.0
            ),
            "base_linear_velocity_local_mps": tuple(
                float(value) for value in local_velocity
            ),
            "foot_contacts": tuple(
                float(value) for value in foot_progress["foot_contacts"]
            ),
            "foot_bottom_z_m": tuple(
                float(value) for value in foot_progress["foot_bottom_z_m"]
            ),
            "foot_support_z_m": tuple(
                float(value) for value in foot_progress["foot_support_z_m"]
            ),
            "foot_bottom_clearance_m": tuple(
                float(value) for value in foot_progress["foot_clearance_m"]
            ),
            "foot_contact_count": int(foot_progress["foot_contact_count"]),
            "foot_support_ok": bool(foot_progress["foot_support_ok"]),
            "all_feet_airborne": bool(foot_progress["all_feet_airborne"]),
            "continuous_airborne_steps": int(
                foot_progress["continuous_airborne_steps"]
            ),
            "continuous_airborne_escalation_penalty": float(
                foot_progress["continuous_airborne_escalation_penalty"]
            ),
            "diagonal_support": bool(foot_progress["diagonal_support"]),
            "diagonal_alternation": bool(
                foot_progress["diagonal_alternation"]
            ),
            "same_side_support": bool(foot_progress["same_side_support"]),
            "tripod_support": bool(foot_progress["tripod_support"]),
            "foot_clearance_count": int(foot_progress["foot_clearance_count"]),
            "foot_x_m": tuple(float(value) for value in foot_progress["foot_x_m"]),
            "foot_on_step_top": tuple(
                bool(value) for value in foot_progress["foot_on_step_top"]
            ),
            "foot_top_landing_flags": tuple(
                bool(value) for value in foot_progress["foot_top_landing_flags"]
            ),
            "foot_pre_edge_landing_flags": tuple(
                bool(value)
                for value in foot_progress["foot_pre_edge_landing_flags"]
            ),
            "foot_placement_swing_steps": tuple(
                int(value)
                for value in foot_progress["foot_placement_swing_steps"]
            ),
            "foot_placement_swing_reward": float(
                foot_progress["foot_placement_swing_reward"]
            ),
            "foot_placement_top_reward": float(
                foot_progress["foot_placement_top_reward"]
            ),
            "foot_placement_pre_edge_penalty": float(
                foot_progress["foot_placement_pre_edge_penalty"]
            ),
            "rear_base_crossed": bool(foot_progress["rear_base_crossed"]),
            "rear_foot_top_landing_flags": tuple(
                bool(value)
                for value in foot_progress["rear_foot_top_landing_flags"]
            ),
            "rear_foot_top_landing_reward": float(
                foot_progress["rear_foot_top_landing_reward"]
            ),
            "rear_foot_top_rewarded": tuple(
                bool(value) for value in foot_progress["rear_foot_top_rewarded"]
            ),
            "foot_dead_leg_steps": tuple(
                int(value) for value in foot_progress["foot_dead_leg_steps"]
            ),
            "target_joint_rad": tuple(float(value) for value in target_rad),
            "reward_terms": reward_terms,
        }
        info.update(self._get_terrain_info())
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """释放环境持有的 LiDAR 感知资源。

        返回：
            无。
        副作用：
            单级台阶模式调用 ``TerrainPerception.close()``；当前 mj_ray 实现
            不持有窗口或线程，但保留统一生命周期以便未来替换后端。
        重要性等级：C（不影响物理结果，但避免资源生命周期不明确）。
        """
        if self.terrain_perception is not None:
            self.terrain_perception.close()
        if self.fsm_controller is not None:
            self.fsm_controller.perception.close()
        return None




