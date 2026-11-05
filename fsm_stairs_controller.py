"""纯规则 FSM 四足爬阶控制器（Go2 + MuJoCo）。

文件目的：
    用可解释的有限状态机和足端轨迹控制 Go2 通过单级台阶，作为后续
    Teacher-Student 强化学习架构中的规则 Teacher 基线。
整体输入：
    ``--scene`` 选择 0.04 m 单级低台阶 ``low`` 或完整三层钢梯 ``full``；
    运行时读取 MuJoCo 关节状态和 LiDAR 高度图。
整体输出：
    终端实时状态日志、可选 3D 实时窗口，以及可选关键帧 PNG 截图。
依赖与版本要求：
    Python 3.10+、mujoco 3.16、numpy 2.x；场景中的 Go2 必须包含
    ``lidar_site`` 和 12 个位置控制电机。
运行方式：
    ``python .\\fsm_stairs_controller.py --scene low``
关键假设：
    1. 机体 +X 为前进方向，足端顺序固定为 FL/FR/RL/RR。
    2. 感知只使用 LiDAR 高度图；状态机不读取台阶 XML 的真实坐标。
    3. 控制器仅通过 12 个关节 PD 力矩驱动，不施加外力。
已知限制：
    默认参数面向当前 Go2 质量、摩擦和单级宽踏面；完整窄梯仍需要
    更细的落足规划，不能把单级场景通过误报为三层楼梯已完成。
重要性等级：
    A（规则 Teacher 基线入口）。
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
PERCEPTION_SRC = ROOT_DIR / "reinforcement_learning" / "src"
if str(PERCEPTION_SRC) not in sys.path:
    sys.path.insert(0, str(PERCEPTION_SRC))

from terrain_perception import (  # noqa: E402
    FINE_HEIGHTMAP_PROFILE,
    TerrainPerception,
    TerrainScan,
)


SCENE_DIR = ROOT_DIR / "simulations" / "mujoco" / "go2_stairs_static"
SCENE_CONFIGS = {
    "low": ("scene_single_step_004.xml", "single_step_approach"),
    "full": ("scene_full_lidar.xml", "stairs_home"),
}

JOINT_NAMES = (
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
)
FOOT_GEOM_NAMES = ("FL", "FR", "RL", "RR")
LEG_NAMES = ("FL", "FR", "RL", "RR")
HIP_X_M = np.array((0.1934, 0.1934, -0.1934, -0.1934), dtype=np.float64)
HIP_Y_M = np.array((0.0465, -0.0465, 0.0465, -0.0465), dtype=np.float64)
LEG_THIGH_Y_OFFSET_M = np.array(
    (0.0955, -0.0955, 0.0955, -0.0955), dtype=np.float64
)
NOMINAL_FOOT_Y_M = np.array((0.1420, -0.1420, 0.1420, -0.1420), dtype=np.float64)
STAND_JOINT_RAD = np.tile(np.array((0.0, 0.9, -1.8)), 4)
THIGH_LENGTH_M = 0.213
CALF_LENGTH_M = 0.213
FOOT_RADIUS_M = 0.022
CONTROL_HZ = 50.0
CONTROL_DT_S = 1.0 / CONTROL_HZ
PD_KP = np.tile(np.array((130.0, 144.0, 144.0)), 4)
PD_KD = np.tile(np.array((2.7, 3.5, 3.5)), 4)
STATE_BLEND_TIME_S = 0.42
# 机身横滚安全阈值，单位 rad。准静态爬阶中单腿摆动不应把机体横滚推到
# 35° 以上；超过该值时继续执行摆腿只会让足端锚点和 IK 同时失真，因此
# 立即终止，而不是等到机身高点跌破 0.14 m 才报跌倒。
ROLL_FAILURE_LIMIT_RAD = math.radians(35.0)
# 开始一条腿的准静态摆动前，要求另外三足至少有三条接触地形。等待期间
# 不推进事件时间，避免在支撑腿尚未真正着地时把另一条腿抬起来。
QUASI_SWING_SUPPORT_WAIT_LOG_S = 0.50
QUASI_SWING_SUPPORT_TIMEOUT_S = 3.0
# 足端稳定接触窗口，单位 s。MuJoCo 的接触缓存可能在脚刚碰到地形的一帧
# 就报告 ``True``，但此时腿部仍在高速下落，尚未形成可持续承重。只有连续
# 接触超过该窗口才允许抬下一条腿或移动机身；典型值 0.10~0.20 s。设得过小
# 会把“轻碰一下”误判为支撑并诱发侧翻，设得过大则会让准静态动作显得卡顿。
QUASI_STANCE_SETTLE_S = 0.12
# 重力/科氏力前馈比例，无量纲。Go2 的电机是纯力矩执行器，若只使用位置
# PD，单腿支撑时重力会让机身持续下沉，固定世界锚点的 IK 目标便无法形成
# 稳定承重。1.0 表示完整补偿 ``qfrc_bias``；典型值 0.8~1.0，过大可能在
# 落地瞬间产生上冲，过小则保持高度能力不足。该前馈只写入 12 个关节电机，
# 不向自由基座施加任何外力。
GRAVITY_FEEDFORWARD_SCALE = 1.0
# CRAWL 时通过移动支撑足目标来平移机身重心，单位 m。抬前腿时机身需要
# 后移，抬后腿时需要前移；抬左腿时还需要右移，抬右腿时相反。支撑足目标
# 始终朝“补全后的三足支撑三角形重心”反向偏移。数值过小会侧翻，过大则
# 会造成支撑足拖滑。
CRAWL_SUPPORT_SHIFT_X_M = 0.025
CRAWL_SUPPORT_SHIFT_Y_M = 0.032
# 重心转移提前量和一阶低通时间常数，单位分别为“步态周期比例”和 s。
# 提前量保证腿离地前重心已经开始移动，低通避免相位切换时产生横向冲击。
CRAWL_SHIFT_LEAD_PHASE = 0.20
CRAWL_SHIFT_FILTER_TAU_S = 0.30
# 单腿爬行时支撑足从触地点向后扫过的距离，单位 m。该量直接产生机身
# 前进位移；调大能提高爬阶推进力，但过大会造成足端拖滑和关节接近限位。
STAIR_STANCE_DRIVE_M = 0.035
# CRAWL 摆动相的水平前伸峰值，单位 m。仅在脚已抬到安全高度后生效，
# 避免足端垂直落在台阶立面；0.015 m 约等于 1.5 cm，过大会增加 IK 限位风险。
CRAWL_SWING_FORWARD_EXTENSION_M = 0.015
# 爬阶时支撑足的世界 Z 补偿参数。自由基座不能直接写 qpos，因此通过
# “支撑足相对机身升降”产生腿的反作用力，把机身 Z 高度和 pitch/roll 拉回
# 目标姿态。CRAWL_BODY_Z_GAIN 为无量纲比例项，典型值 1.0~2.5；过大会使
# 支撑足上下抖动，过小则机身难以上升。CRAWL_PITCH_GAIN_M_RAD 的单位是
# m/rad，典型值 0.08~0.18；过大容易造成前后腿对拉。两项均对稳定性敏感。
CRAWL_BODY_Z_GAIN = 0.8
CRAWL_BODY_Z_DAMPING = 0.05
CRAWL_PITCH_GAIN_M_RAD = 0.0
CRAWL_PITCH_DAMPING_M_S_RAD = 0.025
# 低台阶 CRAWL 的横滚比例反馈，单位 m/rad。三足支撑阶段只允许毫米级
# 配重修正；0.03 表示 10 度横滚约产生 5 mm 目标修正，过大会诱发横向振荡。
CRAWL_ROLL_GAIN_M_RAD = 0.03
CRAWL_ROLL_DAMPING_M_S_RAD = 0.035
# 支撑足 Z 修正的硬裁剪，单位 m。该上限小于 5 cm 栅格，避免姿态闭环在
# 单帧内把足端目标拉到台阶内部；调大能加快恢复，但会增加拖滑风险。
CRAWL_POSTURE_CORRECTION_LIMIT_M = 0.006
# 单腿摆动期间的支撑足世界锚点闭环参数。支撑足关节目标原本在事件内
# 完全冻结，机身一旦因台阶反力发生横滚，支撑足会随躯干一起偏转并逐步
# 失稳。这里按当前机身位姿重新求支撑足 IK，再用很小的单周期角度增量
# 限幅跟踪，既能纠正姿态，又不会产生瞬时目标跳变。限幅单位 rad/控制
# 周期（50 Hz）；典型值 0.008~0.025，过大时会拖滑或顶撞台阶。
# 单腿摆动与 HOLD 期间的支撑足关节增量上限，单位 rad/控制周期（50 Hz）。
# BODY_POSE 不再走这个限幅闭环，而是用“目标机座位姿 + 固定足端锚点”的
# 解析 IK 直接生成平滑轨迹，因此这里可适度放大以吸收落地冲击和接触误差。
# 典型值 0.008~0.020；调得过大容易拖滑，调得过小会使支撑足跟随滞后。
QUASI_SUPPORT_JOINT_STEP_LIMIT_RAD = 0.012
# 按当前支撑足高度估算理想机身高度时使用的“机身中心到足端球心”名义
# 垂距，单位 m。该值来自站立关键帧；设错会让整条爬阶轨迹整体偏高或偏低。
NOMINAL_BODY_TO_FOOT_CENTER_M = 0.2664
NOMINAL_FRONT_REAR_WHEELBASE_M = 0.3868
# 稳定四足支撑时，官方 Go2 足端球心相对接触平面的高度，单位 m。
# 该值由当前 MuJoCo 场景站立预热后的四足实测中位数得到（约 0.0093 m），
# 不能使用出生关键帧的悬空姿态值（约 0.0319 m）。数值偏大时落脚点悬空，
# 三足支撑门控会超时；偏小时足端会持续压入地形并造成拖滑。
NOMINAL_CONTACT_FOOT_CENTER_Z_M = 0.0093

# 落足完成后判定足端是否真正位于台阶顶面的容差，单位 m。Z 容差必须
# 小于台阶高度的一半，否则平地足端可能被误判为已登顶。
FOOT_TOP_Z_TOLERANCE_M = 0.035
FOOT_EDGE_MARGIN_M = 0.005
HIGH_TO_CRAWL_BODY_OFFSET_M = 0.16
# 低台阶（本轮 0.04 m）单独使用更短的接近距离。早期在 0.48 m 就切换
# 准静态爬阶，会让前足在机身仍远离台阶时被送到前缘附近，随后单腿摆动
# 的反作用把机身继续向后推，最终形成约 0.57 m 的不可达髋-足水平距离。
# 0.38 m 仍给前足留出约 10 cm 的制动余量，同时把髋-足目标保持在腿长内。
LOW_STEP_TRIGGER_DISTANCE_M = 0.38
# 低台阶前足落在前缘内 7.5 cm。摆动 PD 会带来数厘米的前向跟踪滞后，
# 目标过近时实测足端会停在立面前；该距离用于留出真实落足余量。为避免
# 足端落远后把机身向后拖，后续横向配重事件会同步向前移动机身，主动
# 缩短髋到足的水平距离。两档刻意错开，避免四足争抢同一踏面位置。
LOW_STEP_FRONT_TOP_OFFSET_M = 0.075
LOW_STEP_REAR_TOP_OFFSET_M = 0.015
# 低台阶 CRAWL 的“准接触”最大高差，单位 m。四足搬移机身时，两条
# 后腿真实承重，而前足可能因 PD 跟踪和 IK 拉伸短暂悬空；若仍要求
# 前足接触，会形成“机身前移才能缩短腿长、接触后才能前移”的互锁。
# 40 mm 只覆盖低台阶 0.04 m 课程的瞬时悬空，不能用于高台阶或 DONE 判定。
LOW_STEP_FRONT_QUASI_CONTACT_MAX_CLEARANCE_M = 0.040
# 足端底面目标相对 LiDAR 地形面的压入量，单位 m。仅压入 0.5 mm 用来
# 消除 MuJoCo 软接触的几何间隙；设得过大可能增加穿模和反力抖动。
SUPPORT_SURFACE_CONTACT_SINK_M = 0.0005


class FsmState(Enum):
    """规则控制器对外可见的有限状态。

    状态顺序对应“平地接近 -> 预判抬腿 -> 三足爬行 -> 重心前压”；
    内部 ``DONE``/``FAILED`` 只用于终止演示，不参与学习动作定义。
    重要性：A。
    """

    FLAT_TROT = "FLAT_TROT"
    HIGH_STEP = "HIGH_STEP"
    CRAWL = "CRAWL"
    CLIMB_PUSH = "CLIMB_PUSH"
    DONE = "DONE"
    FAILED = "FAILED"


@dataclass(frozen=True)
class GaitProfile:
    """描述一种足端节律和抬腿尺度。

    字段：
        name：诊断名称。
        period_s：完整步态周期，单位 s。
        swing_fraction：单周期内摆动相占比，范围 (0, 1)。
        phase_offsets：四足相位偏移，shape=(4,)，无量纲。
        stride_m：单次摆动相对机身的落足前移量，单位 m。
        clearance_m：四足摆动净空高度，shape=(4,)，单位 m。
        front_bias_m：前足落足额外前移量，单位 m。
    重要性：A（直接决定步态稳定性和跨越能力）。
    """

    name: str
    period_s: float
    swing_fraction: float
    phase_offsets: np.ndarray
    stride_m: float
    clearance_m: np.ndarray
    front_bias_m: float = 0.0


@dataclass
class StepFeature:
    """LiDAR 高度图提取出的相对台阶特征。

    字段坐标均为世界系或机身相对系，单位 m：
        visible：当前帧是否能稳定识别高台。
        nearest_x_m：首个高台格中心相对机身 +X 的距离。
        rise_m：高台相对局部地面的高度。
        top_z_m：估计的高台顶面世界 Z。
        floor_z_m：估计的局部地面世界 Z。
        confidence：高台网格数量，用于抑制单射线噪点。
    重要性：A（状态切换与落足高度都依赖该结构）。
    """

    visible: bool = False
    nearest_x_m: float = 1.0
    rise_m: float = 0.0
    top_z_m: float = 0.0
    floor_z_m: float = 0.0
    confidence: int = 0
    cells: int = 0
    # LiDAR 当前视野内台面 Y 的最小/最大值，单位 m，世界坐标。宽踏面
    # 单级测试用于排除足端落在台面侧边之外；视野不足时保持保守范围。
    platform_y_min_m: float = -0.50
    platform_y_max_m: float = 0.50


@dataclass
class RuntimeMetrics:
    """保存一次运行的短评测指标。

    所有高度单位为 m，距离单位为 m；用于最终报告而不参与控制。
    重要性：B。
    """

    max_base_x_m: float = -math.inf
    max_base_z_m: float = -math.inf
    max_foot_clearance_m: float = 0.0
    min_step_distance_m: float = math.inf
    crawl_entries: int = 0
    push_entries: int = 0
    # 本回合 IK 目标超过安全关节范围并被裁剪的累计次数与最大裁剪量，
    # 用于比较纯 FSM 与残差策略是否加剧关节极限使用。
    ik_clip_count: int = 0
    ik_clip_max_rad: float = 0.0
    done: bool = False
    failed: bool = False
    state_history: list[str] = field(default_factory=list)


class QuasiEventKind(Enum):
    """准静态爬阶内部事件类型。

    外部 FSM 仍只暴露 ``HIGH_STEP/CRAWL/CLIMB_PUSH``；这些内部事件只负责
    把一个大状态拆成可验证的小动作。这样支撑腿在整个事件内保持同一组
    关节目标，不会被逐帧世界坐标 IK 拉得来回漂移。
    重要性：A（规则控制器稳定性的核心执行单元）。
    """

    HOLD = "HOLD"
    JOINT_POSE = "JOINT_POSE"
    BODY_POSE = "BODY_POSE"
    SWING = "SWING"


@dataclass(frozen=True)
class QuasiEvent:
    """描述一次准静态动作事件。

    字段：
        name：终端诊断名称。
        kind：事件类型；HOLD 锁住关节，BODY_POSE 平滑改变机体位姿，
            SWING 只移动一条腿。
        duration_s：事件持续时间，单位 s；必须大于 0。
        leg_index：SWING 使用的腿编号，0..3=FL/FR/RL/RR；其他事件为 -1。
        body_dx_m/body_dy_m/body_dz_m：BODY_POSE 相对事件起始机身的目标平移，
            单位 m；dy 正方向为机身左侧，用于单腿离地前的重心转移。
        foot_x_m/foot_z_m：SWING 足端球心世界目标，单位 m。
        clearance_m：SWING 相对起止连线的额外抬脚峰值，单位 m。
        joint_pose_rad：JOINT_POSE 目标，shape=(12,)，单位 rad；仅该事件使用。
            它用于从 Trot 任意相位平滑回到四足支撑站立，避免直接冻结
            摆动相关节而让机身失去支撑。
        level_body_attitude：BODY_POSE 是否把机身 roll/pitch 回正到水平，
            同时保留当前偏航角。默认 False 表示沿用事件起始姿态；前足抬高
            和单腿离地前的重心转移应设为 True，避免斜姿态逐段累积。
    重要性：A（计划参数直接影响越阶能力和平衡）。
    """

    name: str
    kind: QuasiEventKind
    duration_s: float
    leg_index: int = -1
    body_dx_m: float = 0.0
    body_dy_m: float = 0.0
    body_dz_m: float = 0.0
    foot_x_m: float = 0.0
    foot_z_m: float = 0.0
    clearance_m: float = 0.0
    joint_pose_rad: tuple[float, ...] | None = None
    level_body_attitude: bool = False


def cosine_blend(start: float, end: float, progress: float) -> float:
    """使用余弦插值在两端之间平滑过渡。

    参数：
        start/end：标量或同 shape 数组的起止值。
        progress：0..1 的归一化时间；函数内部会裁剪。
    返回：
        与输入广播后同 shape 的插值结果。
    说明：
        余弦曲线在两端导数为零，可避免状态切换时关节速度突变。
    重要性：A。
    """

    t = float(np.clip(progress, 0.0, 1.0))
    alpha = 0.5 - 0.5 * math.cos(math.pi * t)
    return start + alpha * (end - start)

def load_scene(scene_name: str) -> tuple[mujoco.MjModel, int, Path]:
    """加载所选场景并返回模型、关键帧编号和场景路径。

    参数：
        scene_name：``low`` / ``full`` 之一。
    返回：
        ``(model, keyframe_id, scene_path)``。
    可能异常：
        场景名、文件、关键帧或模型维度不合法时抛出 ValueError/FileNotFoundError。
    副作用：
        为绕开 Windows MuJoCo C 库对中文绝对路径的解析限制，函数临时切换
        当前目录到场景目录，加载后恢复原目录。
    重要性：A（所有运行入口都依赖正确的场景加载）。
    """

    if scene_name not in SCENE_CONFIGS:
        raise ValueError(f"未知场景 {scene_name!r}，可选 {tuple(SCENE_CONFIGS)}")
    filename, keyframe_name = SCENE_CONFIGS[scene_name]
    scene_path = (SCENE_DIR / filename).resolve()
    if not scene_path.is_file():
        raise FileNotFoundError(f"找不到场景文件：{scene_path}")

    old_cwd = os.getcwd()
    try:
        os.chdir(scene_path.parent)
        model = mujoco.MjModel.from_xml_path(scene_path.name)
    finally:
        os.chdir(old_cwd)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if keyframe_id < 0:
        raise ValueError(f"场景 {scene_name} 缺少关键帧 {keyframe_name}")
    if (model.nq, model.nv, model.nu) != (19, 18, 12):
        raise ValueError(
            f"Go2 模型维度应为 19/18/12，实际 {model.nq}/{model.nv}/{model.nu}"
        )
    return model, keyframe_id, scene_path


def quaternion_to_pitch_roll(qpos_quat: np.ndarray) -> tuple[float, float]:
    """把 MuJoCo ``wxyz`` 四元数转成 pitch/roll（rad）。

    参数：
        qpos_quat：shape=(4,)，顺序 w,x,y,z。
    返回：
        ``(pitch, roll)``，单位 rad。
    说明：
        控制器只做小角度稳定性诊断；奇异姿态时三角函数仍返回有限值。
    重要性：B。
    """

    w, x, y, z = np.asarray(qpos_quat, dtype=np.float64)
    norm = math.sqrt(max(w * w + x * x + y * y + z * z, 1e-12))
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return pitch, roll


def level_rotation_preserving_yaw(rotation: np.ndarray) -> np.ndarray:
    """构造保留偏航、但把 roll/pitch 回正的机身旋转矩阵。

    参数：
        rotation：shape=(3,3)，MuJoCo 世界系下的机体旋转矩阵，列向量为
            机体系 X/Y/Z 轴在世界系中的方向。
    返回：
        shape=(3,3)，只剩绕世界 Z 轴偏航分量的旋转矩阵。
    说明：
        直接把旋转矩阵设为单位阵会同时抹掉偏航，导致四足落足目标绕机身
        旋转；本函数只提取 +X 轴在水平面的投影，避免爬阶过程中跑偏。
    重要性：A（机身前移和单腿离地前保持姿态水平）。
    """

    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"rotation shape 应为 (3,3)，实际为 {matrix.shape}")
    forward_xy = matrix[:2, 0].copy()
    norm_xy = float(np.linalg.norm(forward_xy))
    if norm_xy < 1e-6:
        # 极端俯仰接近 90 度时 +X 投影退化，改用机体 +Y 轴估计偏航。
        forward_xy = np.array((matrix[1, 0], -matrix[0, 0]), dtype=np.float64)
        norm_xy = float(np.linalg.norm(forward_xy))
    if norm_xy < 1e-6:
        return np.eye(3, dtype=np.float64)
    forward_xy /= norm_xy
    yaw = math.atan2(float(forward_xy[1]), float(forward_xy[0]))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


def leg_inverse_kinematics(
    foot_x_rel_m: float,
    foot_y_rel_m: float,
    foot_z_rel_m: float,
    hip_x_m: float,
    hip_y_m: float,
    thigh_y_offset_m: float,
    *,
    hip_low_rad: float = -1.0472,
    hip_high_rad: float = 1.0472,
) -> tuple[float, float, float]:
    """由机身局部足端位置反解单腿 hip/thigh/calf 目标角。

    参数：
        foot_x_rel_m：足端球心相对机身中心的 X，单位 m，+X 为前进方向。
        foot_y_rel_m：足端球心相对机身中心的 Y，单位 m，+Y 为机身左侧。
        foot_z_rel_m：足端球心相对机身中心的 Z，单位 m，向下通常为负。
        hip_x_m：该腿髋关节相对机身的 X，单位 m。
        hip_y_m：该腿髋关节相对机身的 Y，单位 m。
        thigh_y_offset_m：髋侧摆后大腿关节的带符号 Y 偏移，单位 m；左腿为正，
            右腿为负。该值是三轴 IK 的闭合约束，设错会把足端投到另一侧。
        hip_low_rad/hip_high_rad：髋关节允许范围，单位 rad。默认沿用 Go2 的
            ``[-1.0472, 1.0472]``。
    返回：
        ``(hip_rad, thigh_rad, calf_rad)``，单位 rad。当目标接近机构奇异或
        超出腿长时，函数会夹到保守可达范围，调用方还应再做联合限位保护。
    可能异常：
        足端与髋关节的水平/垂直距离几乎为零、无法定义侧摆平面时抛出
        ``ValueError``；正常地面足端目标不会触发。
    算法与坐标系：
        髋轴沿 +X。先解 :math:`dy\\cos(q_1)+dz\\sin(q_1)=s` 得到侧摆角，
        再把足端投影到矢状面。若目标前向位移为 ``dx``、向下投影长度为
        ``S``，两段等长腿满足 ``calf=-2*acos(r/(2L))``、
        ``thigh=-atan2(dx,S)-calf/2``。旧公式漏掉 X 方向符号，导致目标
        向前的足端实际被驱向后方。
    重要性：A（足端轨迹到 12 关节控制的唯一 IK 边界）。
    """

    dx = float(foot_x_rel_m) - float(hip_x_m)
    dy = float(foot_y_rel_m) - float(hip_y_m)
    dz = float(foot_z_rel_m)
    lateral_radius = math.hypot(dy, dz)
    if not math.isfinite(lateral_radius) or lateral_radius < 1e-6:
        raise ValueError("足端与髋关节重合，无法求解三轴腿部 IK")

    # 髋侧摆只能把大腿根移动到大腿关节的圆弧上。先求两个几何解，再选
    # 绝对值最小且落在关节范围内的一个，避免右腿误选到 -pi 附近的折叠解。
    offset = float(
        np.clip(
            thigh_y_offset_m,
            -0.98 * lateral_radius,
            0.98 * lateral_radius,
        )
    )
    beta = math.atan2(dz, dy)
    delta = math.acos(float(np.clip(offset / lateral_radius, -1.0, 1.0)))
    candidates: list[float] = []
    for hip_rad in (beta + delta, beta - delta):
        wrapped = (hip_rad + math.pi) % (2.0 * math.pi) - math.pi
        if hip_low_rad <= wrapped <= hip_high_rad:
            candidates.append(wrapped)
    hip_rad = (
        min(candidates, key=abs)
        if candidates
        else float(np.clip((beta + delta + math.pi) % (2.0 * math.pi) - math.pi,
                           hip_low_rad, hip_high_rad))
    )

    # 把目标投到侧摆后的矢状面。S 是从大腿根指向足端的向下投影；若目标
    # 跑到髋关节上方，S 会小于零，此时用最小正值避免 acos 和 atan2 奇异。
    sagittal_down_m = dy * math.sin(hip_rad) - dz * math.cos(hip_rad)
    sagittal_down_m = max(float(sagittal_down_m), 1e-4)
    radius_m = float(
        np.clip(
            math.hypot(dx, sagittal_down_m),
            0.12,
            2.0 * THIGH_LENGTH_M - 1e-4,
        )
    )
    calf_rad = -2.0 * math.acos(
        float(np.clip(radius_m / (2.0 * THIGH_LENGTH_M), -1.0, 1.0))
    )
    thigh_rad = -math.atan2(dx, sagittal_down_m) - 0.5 * calf_rad
    return float(hip_rad), float(thigh_rad), float(calf_rad)

def estimate_step_feature(
    scan: TerrainScan,
    base_position_m: np.ndarray,
    base_rotation: np.ndarray,
    *,
    height_threshold_m: float = 0.018,
) -> StepFeature:
    """从 LiDAR 点云中估计最近台阶边缘、高度和顶面世界 Z。

    参数：
        scan：``TerrainPerception.scan`` 的最新结果；使用 ``points_m`` 中的
            机身局部点云，shape=(N,3)，单位 m。
        base_position_m：机身中心世界坐标，shape=(3,)，单位 m。
        base_rotation：机身在世界的旋转矩阵，shape=(3,3)，按 row-vector
            约定与局部点相乘。
        height_threshold_m：相对局部地面判定高台的阈值，单位 m。默认
            0.018 m 能识别 0.04 m 低台阶，同时过滤接触求解器的小抖动。
    返回：
        ``StepFeature``；无稳定高台时 ``visible=False``。函数不使用场景 XML
        中的真实台阶坐标，也不会读取模拟器里的障碍物真值。
    算法：
        1. 把局部点云变换到世界系，按世界 Z 的低分位估计地面高度；
        2. 将高出地面阈值的点按 0.05 m X 区间聚类，至少 4 个点才接受；
        3. 以最靠后的高台簇估计台阶前缘，以顶面点的高分位估计高度。
    说明：
        相比直接在俯仰后的机身高度图上比较 Z，世界 Z 聚类不会被机头抬头
        或低头造成的同地面坡度误判成台阶。连续帧确认由控制器负责。
    重要性：A（唯一允许触发爬阶状态的台阶感知入口）。
    """

    points_local = np.asarray(scan.points_m, dtype=np.float64)
    base_position = np.asarray(base_position_m, dtype=np.float64).reshape(3)
    rotation = np.asarray(base_rotation, dtype=np.float64).reshape(3, 3)
    if points_local.ndim != 2 or points_local.shape[1] != 3:
        raise ValueError(f"LiDAR 点云 shape 应为 (N,3)，实际 {points_local.shape}")
    finite_rows = np.all(np.isfinite(points_local), axis=1)
    points_local = points_local[finite_rows]
    if points_local.shape[0] < 4:
        return StepFeature()

    points_world = points_local @ rotation.T + base_position
    world_z = points_world[:, 2]
    # 近场扫描中地面点通常占多数；10% 分位在台阶开始占满视野前仍是地面。
    # 万一机体已站上台阶且地面点完全消失，本帧只失去“当前可见”，控制器会
    # 继续使用已经确认并锁存的前缘与顶面高度完成动作。
    floor_z = float(np.percentile(world_z, 10.0))
    elevated_mask = world_z > floor_z + float(height_threshold_m)
    if int(np.count_nonzero(elevated_mask)) < 4:
        return StepFeature(floor_z_m=floor_z, cells=int(scan.heightmap.size))

    elevated = points_world[elevated_mask]
    elevated_z = elevated[:, 2]
    x_min = float(np.min(elevated[:, 0]))
    bin_width = float(FINE_HEIGHTMAP_PROFILE.resolution_m)
    bin_index = np.floor((elevated[:, 0] - x_min) / bin_width).astype(np.int64)
    valid_bins: list[tuple[float, int]] = []
    for index in np.unique(bin_index):
        count = int(np.count_nonzero(bin_index == index))
        if count >= 4:
            valid_bins.append((float(np.min(elevated[bin_index == index, 0])), count))
    if not valid_bins:
        return StepFeature(floor_z_m=floor_z, cells=int(scan.heightmap.size))

    # 最靠后的有效簇才是台阶前缘；后续 X 簇可能是同一顶面或更高一级台阶。
    edge_world_x = min(item[0] for item in valid_bins)
    near_mask = elevated[:, 0] <= edge_world_x + 0.16
    near_top_z = elevated[near_mask, 2]
    top_z = float(np.percentile(near_top_z, 80.0))
    rise = max(top_z - floor_z, 0.0)
    if rise < float(height_threshold_m):
        return StepFeature(floor_z_m=floor_z, cells=int(scan.heightmap.size))
    platform_points = elevated[near_mask]
    platform_y = platform_points[:, 1]
    # 只使用实际命中点围成的范围，再各外扩一个 5 cm 栅格。若视觉范围
    # 尚未覆盖整块宽踏面，这个范围仍然足以验证当前足端是否落在可见顶面。
    return StepFeature(
        visible=True,
        nearest_x_m=float(edge_world_x - float(base_position[0])),
        rise_m=rise,
        top_z_m=top_z,
        floor_z_m=floor_z,
        confidence=int(np.count_nonzero(near_mask)),
        cells=int(scan.heightmap.size),
        platform_y_min_m=float(np.min(platform_y) - 0.05),
        platform_y_max_m=float(np.max(platform_y) + 0.05),
    )

class RuleFsmController:
    """纯规则 FSM 控制器。

    功能：
        读取 LiDAR 高度图，切换 FLAT_TROT/HIGH_STEP/CRAWL/CLIMB_PUSH，
        用世界坐标足端锚点规划稳定落足，再通过解析 IK 和关节 PD 驱动机器狗。
    输入：
        model：已编译 Go2 场景；keyframe_id：出生关键帧；
        scene_name：用于日志和完成判定的场景别名。
    输出：
        每个物理子步写入 ``data.ctrl`` 的 12 维关节力矩。
    可能异常：
        缺少关节、足端 geom、LiDAR site 或模型维度不符时构造阶段报错。
    副作用：
        修改 MuJoCo ``data.ctrl``；扫描阶段只读 LiDAR；不会写 qpos/qvel。
    并发与频率：
        外循环 50 Hz 更新目标和状态，内循环按物理步长计算 PD；单线程。
    重要性：A（规则 Teacher 的全部行为集中在此类）。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        keyframe_id: int,
        scene_name: str,
    ) -> None:
        self.model = model
        self.keyframe_id = int(keyframe_id)
        self.scene_name = scene_name
        self.perception = TerrainPerception(model)
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "base"
        )
        if self.base_body_id < 0:
            raise ValueError("模型缺少 base body")

        self.joint_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in JOINT_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.joint_ids < 0):
            raise ValueError("模型缺少 Go2 的 12 个关节")
        self.qpos_adr = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        self.dof_adr = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in self.joint_ids],
            dtype=np.int32,
        )
        if not np.array_equal(
            np.asarray(model.actuator_trnid[:, 0], dtype=np.int32),
            self.joint_ids,
        ):
            raise ValueError("电机顺序与 JOINT_NAMES 不一致")
        self.torque_low = np.asarray(model.actuator_ctrlrange[:, 0]).copy()
        self.torque_high = np.asarray(model.actuator_ctrlrange[:, 1]).copy()
        self.joint_low = np.asarray(model.jnt_range[self.joint_ids, 0]).copy()
        self.joint_high = np.asarray(model.jnt_range[self.joint_ids, 1]).copy()
        self.foot_geom_ids = np.array(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in FOOT_GEOM_NAMES
            ],
            dtype=np.int32,
        )
        if np.any(self.foot_geom_ids < 0):
            raise ValueError("模型缺少 FL/FR/RL/RR 足端 geom")

        # 接触过滤使用整棵机器人运动树，而不是只比较 base body id。Go2 的
        # 大腿、小腿和足端分别是 base 的子 body；若只排除 base，自身腿脚
        # 相碰会被误判成地形支撑。该集合只用于诊断/安全门控，不改变碰撞。
        robot_body_ids: set[int] = set()
        for body_id in range(int(model.nbody)):
            ancestor_id = int(body_id)
            # MuJoCo 世界根 body 的 parentid 指向自身，必须用 ``> 0`` 和
            # “父节点未变化”双重保护；否则从根分支出发会永久循环，表现为
            # 窗口尚未出现、Python 单核占满但没有任何日志。
            while ancestor_id > 0:
                if ancestor_id == self.base_body_id:
                    robot_body_ids.add(int(body_id))
                    break
                parent_id = int(model.body_parentid[ancestor_id])
                if parent_id == ancestor_id:
                    break
                ancestor_id = parent_id
        self.robot_body_ids = frozenset(robot_body_ids)

        # 关节相位偏移：FL/RR 与 FR/RL 交替，构成对角 Trot。
        self.patterns = {
            FsmState.FLAT_TROT: GaitProfile(
                "flat_trot", 0.58, 0.42,
                np.array((0.00, 0.50, 0.50, 0.00)),
                0.15, np.array((0.070, 0.070, 0.070, 0.070)),
            ),
            # HIGH_STEP 和 CRAWL 都采用“一次仅一条腿摆动”的静稳定顺序，
            # 不再让两只前腿同时离地。0/0.25/0.50/0.75 的相位严格对应
            # FL -> FR -> RL -> RR，每条腿都有完整的支撑与稳定窗口。
            FsmState.HIGH_STEP: GaitProfile(
                "high_step", 2.00, 0.24,
                np.array((0.00, 0.25, 0.50, 0.75)),
                0.125, np.array((0.235, 0.235, 0.205, 0.205)),
                0.015,
            ),
            FsmState.CRAWL: GaitProfile(
                "crawl", 2.00, 0.24,
                np.array((0.00, 0.25, 0.50, 0.75)),
                0.115, np.array((0.225, 0.225, 0.205, 0.205)),
                0.010,
            ),
            FsmState.CLIMB_PUSH: GaitProfile(
                "climb_push", 1.55, 0.24,
                np.array((0.00, 0.25, 0.50, 0.75)),
                0.125, np.array((0.225, 0.225, 0.175, 0.175)),
                0.015,
            ),
        }
        self.state = FsmState.FLAT_TROT
        self.state_started_s = 0.0
        self.warmup_end_s = 0.0
        self._transition_start_q = STAND_JOINT_RAD.copy()
        self._transition_start_s = -1e9
        self._gait_start_s = 0.0
        self._anchor_world = np.zeros((4, 3), dtype=np.float64)
        self._swing_start_world = np.zeros((4, 3), dtype=np.float64)
        self._swing_goal_world = np.zeros((4, 3), dtype=np.float64)
        self._prev_local_phase = np.zeros(4, dtype=np.float64)
        self._initialized = False
        self._front_on_step = False
        self._rear_on_step = False
        self._pushed_once = False
        self.last_feature = StepFeature()
        self.last_scan: TerrainScan | None = None
        self.metrics = RuntimeMetrics()
        self._last_valid_step_time_s = -1e9
        self._edge_world_x_m: float | None = None
        self._feature_streak = 0
        self._feature_visible_now = False
        self._gait_rebased = False
        self._target_joint_rad = STAND_JOINT_RAD.copy()
        # CRAWL 时把所有支撑足的 X/Y 目标向三足支撑三角形内部平移。偏置
        # 跨控制周期保留并做一阶低通，避免腿离地瞬间产生冲击；单位 m。
        self._support_shift_x_m = 0.0
        self._support_shift_y_m = 0.0
        # 站立稳定后的“机身中心—足端球心”垂距参考，单位 m。出生关键帧
        # 的 0.2664 m 是初始离地姿态，落地受力后会缩短到约 0.250 m；
        # 因此爬阶姿态目标必须在每次足端锚点重建时从实际状态刷新，
        # 不能把出生几何常量当成闭环目标。
        self._body_foot_height_ref_m = NOMINAL_BODY_TO_FOOT_CENTER_M
        # 准静态事件引擎：SWING/HOLD 在事件开始时锁存支撑足世界锚点；
        # BODY_POSE 额外保存起止机座位姿，并在事件内按期望基座位姿重新
        # 计算四足 IK，避免固定关节插值与机体横移互相竞争。
        self._quasi_events: list[QuasiEvent] = []
        self._quasi_event_index = -1
        self._quasi_event_started_s = 0.0
        self._quasi_event_start_q = STAND_JOINT_RAD.copy()
        self._quasi_event_goal_q = STAND_JOINT_RAD.copy()
        self._quasi_event_start_foot = np.zeros(3, dtype=np.float64)
        self._quasi_event_goal_foot = np.zeros(3, dtype=np.float64)
        self._quasi_event_start_base = np.zeros(3, dtype=np.float64)
        self._quasi_event_goal_base = np.zeros(3, dtype=np.float64)
        self._quasi_event_goal_rotation = np.eye(3, dtype=np.float64)
        self._quasi_event_clearance_m = 0.0
        self._quasi_last_target = STAND_JOINT_RAD.copy()
        self._quasi_plan_done = False
        self._quasi_event_name = "WAIT"
        # SWING 开始时锁存三条支撑足的世界坐标。事件内以这些坐标作为
        # “脚不滑动”的约束，并根据当前机身位姿反求关节角，形成有限幅度
        # 的支撑姿态闭环。shape=(4,3)，单位 m；mask 顺序 FL/FR/RL/RR。
        self._quasi_support_anchor_world = np.zeros((4, 3), dtype=np.float64)
        self._quasi_support_active_mask = np.zeros(4, dtype=bool)
        self._quasi_support_z_correction_m = np.zeros(4, dtype=np.float64)
        # 计划支撑点：SWING 落脚时记录“应该落在哪里”，而不是把 PD 实时误差
        # 重新锁成下一个事件的锚点。shape=(4,3)，单位 m，顺序 FL/FR/RL/RR。
        self._quasi_planned_foot_world = np.zeros((4, 3), dtype=np.float64)
        self._quasi_planned_anchor_valid = False
        # SWING 事件只有在三足接触确认后才允许打开；等待计时只用于诊断和
        # 超时保护，不会阻塞主循环，也不会停止 mj_step。
        self._quasi_event_open = False
        self._quasi_support_wait_started_s = 0.0
        self._quasi_support_wait_last_log_s = -1e9
        self._quasi_support_wait_event_name = ""
        # 每条腿的连续接触时间，shape=(4,)，单位 s。该数组只在 50 Hz 控制
        # 边界更新，用于过滤“物理上刚碰了一下、尚未承重”的单帧接触。
        self._foot_contact_age_s = np.zeros(4, dtype=np.float64)
        self._last_contact_update_time_s = 0.0
        # 四足稳定接触门控的等待起点和日志节流，单位分别为 s、s。
        self._quasi_stance_wait_started_s = 0.0
        self._quasi_stance_wait_last_log_s = -1e9
        self._quasi_stance_wait_event_name = ""
        # IK 裁剪日志节流时间，单位仿真秒。只记录明显不可达，正常的毫米级
        # 数值裁剪不打印，避免实时窗口被诊断信息淹没。
        self._last_ik_clip_log_s = -1e9

    def reset(
        self,
        data: mujoco.MjData,
        *,
        reset_keyframe: bool = True,
    ) -> None:
        """重置控制器；可选择保留调用方已经设置好的 MuJoCo 出生状态。

        参数：
            data：MuJoCo 数据对象。
            reset_keyframe：True 时恢复本控制器持有的关键帧并清空 ctrl；
                False 时保留 Gym 环境已完成的随机化出生位姿、速度和 ctrl。
                混合训练必须传 False，否则教师会覆盖环境的出生点。
        副作用：
            初始化 FSM 状态、足端锚点和诊断指标；只有 reset_keyframe=True
            时才覆盖自由基座、关节状态和 ctrl。
        关键参数：
            reset_keyframe 设错会让域随机化或 0.45 m 助跑出生点失效。
        重要性：A。
        """

        if reset_keyframe:
            mujoco.mj_resetDataKeyframe(self.model, data, self.keyframe_id)
            data.ctrl[:] = 0.0
            mujoco.mj_forward(self.model, data)
        self.state = FsmState.FLAT_TROT
        self.state_started_s = float(data.time)
        self.warmup_end_s = float(data.time) + 0.75
        self._transition_start_q = np.asarray(
            data.qpos[self.qpos_adr], dtype=np.float64
        ).copy()
        self._transition_start_s = float(data.time)
        self._initialized = False
        self._front_on_step = False
        self._rear_on_step = False
        self._pushed_once = False
        self.last_feature = StepFeature()
        self.last_scan = None
        self._last_valid_step_time_s = -1e9
        self._edge_world_x_m = None
        self._feature_streak = 0
        self._feature_visible_now = False
        self._gait_rebased = False
        self._support_shift_x_m = 0.0
        self._support_shift_y_m = 0.0
        self._quasi_events = []
        self._quasi_event_index = -1
        self._quasi_event_started_s = float(data.time)
        self._quasi_event_start_q = STAND_JOINT_RAD.copy()
        self._quasi_event_goal_q = STAND_JOINT_RAD.copy()
        self._quasi_event_start_foot = np.zeros(3, dtype=np.float64)
        self._quasi_event_goal_foot = np.zeros(3, dtype=np.float64)
        self._quasi_event_start_base.fill(0.0)
        self._quasi_event_goal_base.fill(0.0)
        self._quasi_event_goal_rotation[:] = np.eye(3, dtype=np.float64)
        self._quasi_event_clearance_m = 0.0
        self._quasi_last_target = STAND_JOINT_RAD.copy()
        self._quasi_plan_done = False
        self._quasi_event_name = "WAIT"
        self._quasi_support_anchor_world.fill(0.0)
        self._quasi_support_active_mask.fill(False)
        self._quasi_planned_foot_world.fill(0.0)
        self._quasi_planned_anchor_valid = False
        self._quasi_event_open = False
        self._quasi_support_wait_started_s = float(data.time)
        self._quasi_support_wait_last_log_s = -1e9
        self._quasi_support_wait_event_name = ""
        self._foot_contact_age_s.fill(0.0)
        self._last_contact_update_time_s = float(data.time)
        self._quasi_stance_wait_started_s = float(data.time)
        self._quasi_stance_wait_last_log_s = -1e9
        self._quasi_stance_wait_event_name = ""
        self._last_ik_clip_log_s = -1e9
        self.metrics = RuntimeMetrics()
        self.metrics.state_history.append(self.state.value)
        self._anchor_world = self._current_foot_centers(data)
        self._swing_start_world[:] = self._anchor_world
        self._swing_goal_world[:] = self._anchor_world
        self._body_foot_height_ref_m = float(
            data.xpos[self.base_body_id, 2] - np.mean(self._anchor_world[:, 2])
        )

    def _current_foot_centers(self, data: mujoco.MjData) -> np.ndarray:
        """返回 FL/FR/RL/RR 足端球心世界坐标，shape=(4,3)，单位 m。"""

        return np.asarray(
            data.geom_xpos[self.foot_geom_ids], dtype=np.float64
        ).copy()

    def _current_foot_bottoms_z(self, data: mujoco.MjData) -> np.ndarray:
        """返回四足最低点世界 Z，shape=(4,)，单位 m。"""

        radii = np.asarray(
            self.model.geom_size[self.foot_geom_ids, 0], dtype=np.float64
        )
        return np.asarray(data.geom_xpos[self.foot_geom_ids, 2]) - radii

    def _foot_contact_mask(self, data: mujoco.MjData) -> np.ndarray:
        """返回足端是否与机器人外部地形接触，shape=(4,)，dtype=bool。

        功能：
            在每次 ``mj_step`` 之后的控制边界读取已求解接触，用于阻止
            “支撑腿仍悬空却抬起下一条腿”的准静态事件。
        参数：
            data：当前 MuJoCo 数据；``data.contact`` 必须是最近物理步结果。
        返回：
            FL/FR/RL/RR 的地形接触掩码。只有足端 geom 与非机器人 body 的
            geom 接触时才计为 True，足端碰自身腿或另一只脚不算支撑。
        边界：
            无接触时返回全 False。接触对中的 body 通过整个 base 运动树过滤，
            因而能兼容官方 Go2 的嵌套腿 body 结构。
        重要性：A（准静态爬阶的支撑安全门控）。
        """

        mask = np.zeros(4, dtype=bool)
        foot_lookup = {
            int(geom_id): leg for leg, geom_id in enumerate(self.foot_geom_ids)
        }
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            if geom1 in foot_lookup and body2 not in self.robot_body_ids:
                mask[foot_lookup[geom1]] = True
            if geom2 in foot_lookup and body1 not in self.robot_body_ids:
                mask[foot_lookup[geom2]] = True
        return mask

    def _update_contact_age(self, data: mujoco.MjData) -> None:
        """更新四足连续接触时间，过滤瞬时“碰一下就离开”的假支撑。

        参数：
            data：当前 MuJoCo 数据；接触缓存来自最近一次物理子步。
        副作用：
            更新 ``_foot_contact_age_s`` 和 ``_last_contact_update_time_s``。
            每个 50 Hz 控制周期只调用一次，物理子步不调用。

        算法：
            接触为真的腿累加本次控制周期时长，接触为假的腿立即清零。时间
            增量限制在 0.10 s，避免窗口卡顿或调试停顿时一次累加过多。
        边界：
            首次调用或时间倒退时增量为 0，不会制造虚假的稳定接触。
        重要性：A（准静态支撑门控的时间一致性与安全性）。
        """

        contact_mask = self._foot_contact_mask(data)
        now_s = float(data.time)
        last_s = float(getattr(self, "_last_contact_update_time_s", now_s))
        dt_s = float(np.clip(now_s - last_s, 0.0, 0.10))
        self._foot_contact_age_s[contact_mask] += dt_s
        self._foot_contact_age_s[~contact_mask] = 0.0
        self._last_contact_update_time_s = now_s

    def _stable_contact_ready(
        self,
        data: mujoco.MjData,
        required_mask: np.ndarray,
    ) -> bool:
        """检查指定足端是否连续稳定接触地形。

        参数：
            data：当前 MuJoCo 数据。
            required_mask：shape=(4,)，dtype=bool；True 表示该腿必须满足
                连续接触窗口。
        返回：
            True 表示所有被要求的腿都接触且 ``contact_age`` 不小于
            ``QUASI_STANCE_SETTLE_S``。
        重要性：A（防止刚落地足尚未承重就开启下一个动作）。
        """

        required = np.asarray(required_mask, dtype=bool)
        if required.shape != (4,):
            raise ValueError(f"required_mask 应为 (4,)，实际为 {required.shape}")
        contact_mask = self._foot_contact_mask(data)
        return bool(
            np.all(contact_mask[required])
            and np.all(
                self._foot_contact_age_s[required] >= QUASI_STANCE_SETTLE_S
            )
        )

    def _quasi_anchors_with_planned_fallback(
        self,
        data: mujoco.MjData,
        *,
        update_contacted_plan: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """按“接触足实测、悬空足计划”生成准静态世界锚点。

        功能：
            MuJoCo 的接触缓存只在脚真正压到地形时有效。摆动腿尚未落地时，
            它的实测位置通常悬在空中，不能直接当成下一个事件的固定支撑点；
            反之，已经接触的脚如果继续使用旧计划点，又会把机身位移造成的
            真实支撑误差锁进 IK。这里将两者按接触状态混合，兼顾稳定和落地。

        参数：
            data：当前 MuJoCo 数据；接触缓存来自最近一次物理子步。
            update_contacted_plan：True 时把已接触足的实测位置写回计划数组，
                供后续 SWING 和 HOLD 复用。参数为 False 时只读取，不改状态。

        返回：
            ``(anchors_world, contact_mask)``。前者 shape=(4,3)，单位 m，顺序
            FL/FR/RL/RR；后者 shape=(4,)，dtype=bool。

        副作用：
            ``update_contacted_plan=True`` 时更新
            ``_quasi_planned_foot_world`` 和 ``_quasi_planned_anchor_valid``。
            本函数不推进物理，也不发送执行器命令。

        边界：
            首次调用或计划尚未初始化时，以四足当前实测位置建立计划骨架；
            无接触足的落脚目标仍保留在该骨架中，等待后续 SWING 覆盖。

        重要性：A（三足准静态支撑和接触判断的核心数据源）。
        """

        actual_centers = self._current_foot_centers(data)
        contact_mask = self._foot_contact_mask(data)
        if not self._quasi_planned_anchor_valid:
            self._quasi_planned_foot_world[:] = actual_centers
            self._quasi_planned_anchor_valid = True

        anchors = self._quasi_planned_foot_world.copy()
        anchors[contact_mask] = actual_centers[contact_mask]
        if update_contacted_plan:
            self._quasi_planned_foot_world[contact_mask] = (
                actual_centers[contact_mask]
            )
        return anchors, contact_mask

    @staticmethod
    def _contact_mask_text(contact_mask: np.ndarray) -> str:
        """把 FL/FR/RL/RR 接触掩码转成便于日志扫描的 0/1 字符串。"""

        return "".join("1" if bool(value) else "0" for value in contact_mask)

    def _clip_joint_target(
        self,
        target: np.ndarray,
        context: str,
        now_s: float | None = None,
    ) -> np.ndarray:
        """按关节安全边界裁剪目标角，并对明显 IK 不可达发出诊断。

        参数：
            target：shape=(12,)，单位 rad 的原始 IK 关节目标。
            context：日志用的轨迹/事件名称，用于定位错误腿或事件。
            now_s：当前仿真时间，单位 s。提供时每个仿真秒最多打印一条；
                纯计算路径若未提供，则每 50 次超限调用最多打印一条。
        返回：
            shape=(12,)，已留 0.03 rad 限位余量的关节目标。
        副作用：
            当任一关节裁剪量超过 0.02 rad 时打印诊断。裁剪量过大通常表示
            足端目标超出腿长、机身离台阶过远或后足单步回收过大。
        边界：
            该函数只负责诊断和安全裁剪，不修改 IK 算法或事件推进；调用方
            仍应根据频繁诊断调整轨迹，而不能把裁剪后的目标当作可达目标。
        重要性：A（防止 IK 极限被静默当成正常轨迹）。
        """

        target_array = np.asarray(target, dtype=np.float64)
        lower = self.joint_low + 0.03
        upper = self.joint_high - 0.03
        clipped = np.clip(target_array, lower, upper)
        max_error = float(np.max(np.abs(target_array - clipped)))
        if max_error > 1e-6:
            self.metrics.ik_clip_count += 1
            self.metrics.ik_clip_max_rad = max(
                self.metrics.ik_clip_max_rad,
                max_error,
            )
        if max_error <= 0.02:
            return clipped

        should_log = False
        if now_s is not None:
            now_value = float(now_s)
            if now_value - self._last_ik_clip_log_s >= 1.0:
                should_log = True
                self._last_ik_clip_log_s = now_value
        else:
            self._ik_clip_call_counter = int(
                getattr(self, "_ik_clip_call_counter", 0)
            ) + 1
            if self._ik_clip_call_counter % 50 == 0:
                should_log = True
        if should_log:
            print(
                f"[IK诊断] {context} 目标超出关节安全范围 "
                f"max_clip={max_error:.3f}rad target="
                f"{np.round(target_array, 3).tolist()}",
                flush=True,
            )
        return clipped

    def _platform_lateral_bounds_m(
        self,
        feature: StepFeature,
    ) -> tuple[float, float]:
        """返回台阶踏面在 Y 方向的实际可用范围，单位 m。

        功能：
            优先从 MuJoCo 碰撞盒读取完整踏面宽度，避免 LiDAR 只看到局部
            扇区时把另一侧合法落足点误判为越界；找不到已知盒体时退回
            LiDAR 可见范围。
        参数：
            feature：当前锁存的台阶特征，提供 LiDAR 可见 Y 范围。
        返回：
            ``(y_min, y_max)``，世界坐标，单位 m。
        说明：
            该函数只读取模型静态几何，不修改物理状态；若已知踏面是倾斜
            盒体，本项目的测试台阶均绕 Z 轴放置，因此可安全使用几何中心
            与 Y 半宽构造区间。
        重要性：A（登台判定和 DUT 状态推进依赖正确踏面边界）。
        """

        for geom_name in ("curriculum_step_01", "wide_step_01", "step_01"):
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name
            )
            if geom_id < 0:
                continue
            if int(self.model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX):
                center_y = float(self.model.geom_pos[geom_id, 1])
                half_width_y = float(self.model.geom_size[geom_id, 1])
                return center_y - half_width_y, center_y + half_width_y
        return (
            float(feature.platform_y_min_m),
            float(feature.platform_y_max_m),
        )

    def _feet_on_platform_mask(
        self,
        data: mujoco.MjData,
        feature: StepFeature,
        edge_x_m: float | None,
    ) -> np.ndarray:
        """返回每条腿是否真正落在当前台阶顶面的布尔掩码。

        参数：
            data：当前 MuJoCo 数据。
            feature：LiDAR 锁存的台阶特征。
            edge_x_m：锁存的世界 X 前缘，单位 m。
        返回：
            shape=(4,)，dtype=bool，顺序 FL/FR/RL/RR。
        判定：
            足端球心必须同时满足以下条件：越过前缘 ``FOOT_EDGE_MARGIN_M``；
            Y 位于可见台面范围；球心 Z 接近 ``top_z + 足端半径``；足底不
            低于台面超过 ``FOOT_TOP_Z_TOLERANCE_M``。仅凭高度会把平地或
            正在摆动的腿误判为登台，因此必须联合 X/Y/Z 判定。
        重要性：A（决定状态推进和 DONE，避免假成功）。
        """

        if not feature.visible or edge_x_m is None:
            return np.zeros(4, dtype=bool)
        centers = self._current_foot_centers(data)
        bottoms = self._current_foot_bottoms_z(data)
        expected_center_z = float(feature.top_z_m) + FOOT_RADIUS_M
        y_min, y_max = self._platform_lateral_bounds_m(feature)
        return (
            (centers[:, 0] >= float(edge_x_m) + FOOT_EDGE_MARGIN_M)
            & (centers[:, 1] >= y_min)
            & (centers[:, 1] <= y_max)
            & (np.abs(centers[:, 2] - expected_center_z) <= FOOT_TOP_Z_TOLERANCE_M)
            & (bottoms >= float(feature.top_z_m) - FOOT_TOP_Z_TOLERANCE_M)
        )

    def _transition_to(
        self,
        new_state: FsmState,
        data: mujoco.MjData,
        *,
        gait_delay_s: float = 0.28,
    ) -> None:
        """切换 FSM 状态并重置该状态的足端节律。

        参数：
            new_state：目标状态。
            data：当前 MuJoCo 数据。
            gait_delay_s：切换后保持当前足端锚点的短暂稳定时间，单位 s。
        说明：
            关节目标由余弦插值从切换瞬间实际关节角过渡到新状态目标；
            足端锚点取当前实际位置，避免相位重置造成瞬移。
        重要性：A。
        """

        if new_state == self.state:
            return
        now_s = float(data.time)
        self._transition_start_q = np.asarray(
            data.qpos[self.qpos_adr], dtype=np.float64
        ).copy()
        self._transition_start_s = now_s
        self.state = new_state
        self.state_started_s = now_s
        self.metrics.state_history.append(new_state.value)
        if new_state in (FsmState.CRAWL, FsmState.CLIMB_PUSH):
            if new_state == FsmState.CRAWL:
                self.metrics.crawl_entries += 1
            else:
                self.metrics.push_entries += 1
        self._anchor_world = self._current_foot_centers(data)
        self._swing_start_world[:] = self._anchor_world
        self._swing_goal_world[:] = self._anchor_world
        # 进入爬行/前压前先回到稳定的四足站立，避免在上一状态摆动腿尚未
        # 落地时冻结世界锚点。等待时间由状态机参数显式给出。
        if new_state in (FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH):
            # 进入爬阶前先让四足从平地 Trot 的任意相位完整落地并稳定，
            # 否则 DIAGONAL 支撑相会被状态切换截断，机身会因两足悬空而下沉。
            gait_delay_s = max(float(gait_delay_s), 0.85)
        self._gait_start_s = now_s + gait_delay_s
        self._gait_rebased = False
        if new_state in self.patterns:
            profile = self.patterns[new_state]
            self._prev_local_phase = (profile.phase_offsets - 1e-5) % 1.0
        # 三种爬阶状态不再使用逐帧世界足端节律，而是在进入时构建一次性
        # 准静态事件计划。FLAT_TROT/DONE/FAILED 不进入该事件引擎。
        if new_state in (
            FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
        ):
            self._start_quasi_plan(new_state, data)

    def _set_initial_gait_start(self, data: mujoco.MjData) -> None:
        """在站立预热结束后启动平地 Trot。"""

        self._gait_start_s = float(data.time)
        profile = self.patterns[FsmState.FLAT_TROT]
        self._prev_local_phase = (profile.phase_offsets - 1e-5) % 1.0
        self._anchor_world = self._current_foot_centers(data)
        self._swing_start_world[:] = self._anchor_world
        self._swing_goal_world[:] = self._anchor_world
        self._body_foot_height_ref_m = float(
            data.xpos[self.base_body_id, 2] - np.mean(self._anchor_world[:, 2])
        )
        self._gait_rebased = True
        self._initialized = True

    def _estimated_edge_world_x(self) -> float | None:
        """返回最近一次 LiDAR 有效帧估计的台阶前缘世界 X。

        返回 None 表示尚未看到台阶。该值在狗跨上台面后继续保留，仅用于
        同一级台阶的落足与完成判定，不读取 XML 真值。
        重要性：A。
        """

        return getattr(self, "_edge_world_x_m", None)

    def _plan_foothold(
        self,
        leg_index: int,
        data: mujoco.MjData,
        now_s: float,
    ) -> np.ndarray:
        """为一条腿规划下一次摆动落足的世界坐标。

        参数：
            leg_index：0..3，对应 FL/FR/RL/RR。
            data：当前 MuJoCo 数据。
            now_s：仿真时间，单位 s；用于保持秩和诊断。
        返回：
            shape=(3,)，足端球心世界目标，单位 m。
        算法：
            默认在机身名义足位前方半个步长处落足；接近台阶时允许前足
            直接瞄准台面，后足仅在重心接近前缘后瞄准台面。台面高度来自
            LiDAR，不使用场景 XML 坐标。
        重要性：A（决定是否踩中台面以及是否撞上立面）。
        """

        del now_s
        base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        anchor = self._anchor_world[leg_index]
        profile = self.patterns.get(self.state, self.patterns[FsmState.CRAWL])
        target_x = (
            float(base[0])
            + float(HIP_X_M[leg_index])
            + 0.5 * profile.stride_m
        )
        if leg_index < 2:
            target_x += profile.front_bias_m
        else:
            target_x += 0.5 * profile.front_bias_m

        edge_x = self._estimated_edge_world_x()
        feature = self.last_feature
        if feature.visible and edge_x is not None and feature.rise_m > 0.022:
            # 前足只越过前缘 5.5 cm，避免把脚送到踏面深处后形成过长的
            # 力臂；后足等待机身重心靠近前缘后再落到 12 cm 处。
            landing_margin = 0.055 if leg_index < 2 else 0.120
            landing_x = float(edge_x) + landing_margin
            reachable_ahead = 0.15 <= (landing_x - float(base[0])) <= 0.355
            rear_ready = (
                leg_index >= 2
                and float(base[0]) + 0.045 >= float(edge_x)
            )
            if reachable_ahead and (leg_index < 2 or rear_ready):
                target_x = max(target_x, landing_x)
            elif leg_index >= 2 and rear_ready:
                target_x = max(target_x, landing_x)

        # 限制在单腿矢状面可达范围，防止 IK 因目标过远突然折膝。
        target_x = float(
            np.clip(target_x, float(base[0]) - 0.34, float(base[0]) + 0.355)
        )
        target_y = float(anchor[1])
        floor_z = feature.floor_z_m if feature.visible else 0.0
        target_z = floor_z + FOOT_RADIUS_M
        if (
            feature.visible
            and edge_x is not None
            and target_x >= float(edge_x) + 0.035
            and feature.rise_m > 0.022
        ):
            target_z = float(feature.top_z_m) + FOOT_RADIUS_M
        return np.array((target_x, target_y, target_z), dtype=np.float64)

    def _swing_clearance(self, leg_index: int) -> float:
        """返回摆动弧线在落足目标之上的额外净空，单位 m。

        参数：
            leg_index：0..3，顺序 FL/FR/RL/RR；前后腿使用不同名义上限。
        返回：
            额外叠加到“起点到落足点”插值高度上的峰值余量，单位 m。
        算法：
            落足目标已经包含 ``top_z + FOOT_RADIUS_M``，因此不能再次叠加
            完整台阶高度，否则 0.195 m 台阶会被抬到约 0.32 m，既浪费行程
            又容易让腿接近奇异位形。这里用 ``0.035 + 0.5 * rise`` 估算：
            低台阶获得约 5.5 cm 峰值净空，高台阶获得约 13 cm；峰值相对
            踏面始终只高出约 4~5 cm。
        边界：
            LiDAR 尚未看到台阶时 ``rise=0``；仍保留 3.5 cm 的最小抬脚量。
        重要性：A（直接影响越阶高度和 IK 可达性）。
        """

        profile = self.patterns.get(self.state, self.patterns[FsmState.HIGH_STEP])
        nominal = float(profile.clearance_m[leg_index])
        rise = max(self.last_feature.rise_m, 0.0)
        needed = 0.035 + 0.5 * rise
        return float(min(nominal, max(0.035, needed)))

    def _joint_targets_for_base_pose(
        self,
        data: mujoco.MjData,
        anchors_world: np.ndarray,
        target_base_world: np.ndarray,
        target_rotation: np.ndarray,
    ) -> np.ndarray:
        """按指定目标基座位姿计算保持世界足端锚点的 12 关节角。

        参数：
            data：当前 MuJoCo 数据，仅用于读取关节限位。
            anchors_world：shape=(4,3)，待保持的世界足端球心，单位 m。
            target_base_world：shape=(3,)，目标机身世界位置，单位 m。
            target_rotation：shape=(3,3)，行向量约定下的目标机身旋转矩阵。
        返回：
            shape=(12,)，单位 rad，已留 0.03 rad 关节限位余量。
        异常：
            ``anchors_world`` 不是 (4,3) 或 IK 结果非有限时报 ValueError，
            便于尽早发现轨迹规划错误。
        副作用：
            无；纯计算函数。
        重要性：A（BODY_POSE 事件的一次性 IK 入口）。
        """

        anchors = np.asarray(anchors_world, dtype=np.float64)
        if anchors.shape != (4, 3):
            raise ValueError(f"足端锚点 shape 应为 (4,3)，实际为 {anchors.shape}")
        target = np.zeros(12, dtype=np.float64)
        relative = (anchors - np.asarray(target_base_world, dtype=np.float64)) @ target_rotation
        for leg_index in range(4):
            hip, thigh, calf = leg_inverse_kinematics(
                relative[leg_index, 0],
                relative[leg_index, 1],
                relative[leg_index, 2],
                HIP_X_M[leg_index],
                HIP_Y_M[leg_index],
                LEG_THIGH_Y_OFFSET_M[leg_index],
                hip_low_rad=self.joint_low[3 * leg_index],
                hip_high_rad=self.joint_high[3 * leg_index],
            )
            target[3 * leg_index] = hip
            target[3 * leg_index + 1] = thigh
            target[3 * leg_index + 2] = calf
        if not np.all(np.isfinite(target)):
            raise ValueError("目标基座位姿的 IK 结果包含 NaN/Inf")
        return self._clip_joint_target(
            target, "BODY_POSE_IK", now_s=float(data.time)
        )

    def _build_quasi_plan(
        self,
        state: FsmState,
        data: mujoco.MjData,
    ) -> list[QuasiEvent]:
        """为外部 FSM 状态构建可执行的准静态动作序列。

        功能：
            以 LiDAR 锁存的前缘和顶面高度为几何输入，把 HIGH_STEP、CRAWL、
            CLIMB_PUSH 各自展开成一组只会移动一条腿或只移动机身的短事件。
            每个事件结束时重新锁定四足世界锚点，避免连续前移累计成超伸姿态。
        参数：
            state：当前 FSM 状态，只接受 HIGH_STEP/CRAWL/CLIMB_PUSH。
            data：进入状态时的 MuJoCo 数据；用于读取四足当前世界位置。函数
                不读取 XML 真值，台阶几何仍然来自 LiDAR 锁存结果。
        返回：
            按顺序执行的 ``QuasiEvent`` 列表。LiDAR 尚未锁存台阶时返回一个
            短等待事件，让主循环继续推进物理而不是阻塞。
        异常：
            state 不属于三个爬阶状态或 LiDAR 几何非法时抛出 ValueError。
        副作用：
            无；只生成不可变事件数据。
        关键假设：
            四足初始世界 X 约呈前 +0.19 m、后 -0.19 m 的对称布局。所有落足
            X 都由当前测量足位与 LiDAR 前缘计算，不依赖场景文件名。
        已知限制：
            0.195 m 高台需要机身分阶段抬起；完整窄踏面三层楼梯仍需单独验证。
        重要性：A（决定规则 Teacher 能否稳定越阶）。
        """

        feature = self.last_feature
        edge_x = self._estimated_edge_world_x()
        if not feature.visible or edge_x is None:
            return [
                QuasiEvent("WAIT_FOR_LIDAR", QuasiEventKind.HOLD, 0.50),
            ]
        if state not in (
            FsmState.HIGH_STEP,
            FsmState.CRAWL,
            FsmState.CLIMB_PUSH,
        ):
            raise ValueError(f"_build_quasi_plan 不支持状态 {state.value}")

        floor_z = float(feature.floor_z_m)
        top_z = float(feature.top_z_m)
        rise_m = max(float(feature.rise_m), 0.0)
        if not all(math.isfinite(v) for v in (floor_z, top_z, rise_m, edge_x)):
            raise ValueError("LiDAR 台阶几何包含 NaN/Inf")
        high_step = rise_m > 0.10
        feet_now = self._current_foot_centers(data)
        # 整段计划在 HIGH_STEP 入口只构建一次；此时出生关键帧的四足仍
        # 悬空，不能从 feet_now 估算接触高度。否则会把计划整体抬高约
        # 2.2 cm，表现为落足后接触掩码始终只有两足、下一条摆腿等待超时。
        # 因此这里使用稳定站立实测标定值，并额外下压 8 mm 保证建立接触。
        # 0.04 m 低台阶实测表明：4 mm 的目标下压会被 PD 跟踪误差和软接触
        # 间隙抵消，足底仍停在台面上方约 0.1~1.6 mm，contact 掩码为 0，
        # 继而导致机身事件误判“无四足支撑”。高台继续使用 8 mm 下压；
        # 低台首轮验证中后足落点仍高出踏面约 6 mm，故对低台单独加深到
        # 14 mm，使摆动末段能可靠闭合接触，同时避免把高台轨迹整体压深。
        # floor_z/top_z 均来自 LiDAR，不读取台阶 XML 真值。
        foot_contact_sink_m = 0.008 if high_step else 0.014
        contact_center_offset_z = NOMINAL_CONTACT_FOOT_CENTER_Z_M
        top_foot_z = top_z + contact_center_offset_z - foot_contact_sink_m
        ground_foot_z = floor_z + contact_center_offset_z - foot_contact_sink_m
        # Z 方向动作随真实台阶高度缩放。低台课程不会因为高台参数整体抬高；
        # 0.195 m 时缩放为 1，低台 0.04 m 时约为 0.205。
        z_scale = float(np.clip(rise_m / 0.195, 0.16, 1.0))
        front_clearance_m = 0.050 if high_step else 0.032
        # 低台阶后腿每次只移动约 5 cm，1.2 cm 净空已足够跨过 4 cm 立面，
        # 同时减少抬腿时的动态冲击；如果继续减小到 1 cm 以下，足端更容易
        # 擦到台阶前缘。高台阶仍保留 4.5 cm，避免脚趾擦碰较高立面。
        rear_clearance_m = 0.045 if high_step else 0.020

        initial_rear_x = np.asarray(feet_now[2:, 0], dtype=np.float64).copy()
        # 前足预备点放在前缘后方 14 cm。该距离让摆动足目标相对机身髋部
        # 保持在约 0.36 m 的舒适工作区间，避免机体重心转移时因实际基座后漂
        # 把目标推到 0.426 m 腿长极限之外；设得过远会导致抬腿后足端够不到
        # 地面并触发三足支撑超时，设得过近则后续跨上踏面的单步幅度偏大。
        front_ground_x = float(edge_x) - 0.140
        front_top_x = float(edge_x) + 0.055
        rear_top_x = float(edge_x) + 0.060

        def body_pose(
            name: str,
            *,
            dx: float = 0.0,
            dy: float = 0.0,
            dz: float = 0.0,
            duration_s: float,
            level_attitude: bool = True,
        ) -> QuasiEvent:
            """构造保持支撑足世界锚点的机身事件。

            参数：
                name：事件名，仅用于日志和状态诊断。
                dx/dy/dz：机身相对事件起点的平移量，单位 m。
                duration_s：余弦插值时长，单位 s；过短会产生速度突变。
                level_attitude：是否把机身 roll/pitch 回正到水平。低台阶
                    前后足分处 0 和 0.04 m 两个平面时，强行水平会形成
                    过约束，故 CRAWL 的小步跟随保持当前姿态。
            返回：
                QuasiEvent；不修改物理状态。
            重要性：A（低台阶准静态重心转移的关键约束）。
            """

            return QuasiEvent(
                name,
                QuasiEventKind.BODY_POSE,
                duration_s,
                body_dx_m=dx,
                body_dy_m=dy,
                body_dz_m=dz,
                level_body_attitude=bool(level_attitude),
            )

        def hold(name: str, duration_s: float) -> QuasiEvent:
            """构造让足端锚点和机身姿态稳定的保持事件。"""

            return QuasiEvent(name, QuasiEventKind.HOLD, duration_s)

        def swing(
            name: str,
            leg_index: int,
            target_x: float,
            target_z: float,
            clearance_m: float,
        ) -> QuasiEvent:
            """构造一条腿的三段式摆动：竖直抬腿、水平跨越、竖直落下。"""

            # 低台阶前足的落点距离较短，使用 0.90 s；后足分段收腿只有
            # 约 0.10 m，使用 0.72 s 可缩短单腿离地时间。后足若沿用 1.2 s，
            # 支撑三角形会在机身滚动累积后被卸载，实测会抬起同侧前足。
            if high_step:
                swing_duration_s = 1.55
            elif leg_index < 2:
                swing_duration_s = 0.90
            else:
                # 低台阶后腿摆动需要更长的三足承重窗口。旧版 0.72 s
                # 在机身前移后会产生较大的动态冲击，容易让剩余三足中的
                # 对侧前足瞬时卸载；1.05 s 降低角加速度，优先保证 Teacher
                # 稳定，再交给 PPO 做小幅残差优化。
                swing_duration_s = 1.05
            return QuasiEvent(
                name,
                QuasiEventKind.SWING,
                swing_duration_s,
                leg_index=leg_index,
                foot_x_m=float(target_x),
                foot_z_m=float(target_z),
                clearance_m=float(clearance_m),
            )

        if state == FsmState.HIGH_STEP and not high_step:
            # 0.04 m 低台阶恢复到更接近成功的 Teacher 轨迹：先在平地上把
            # 机身小幅前压，再依次把两只前足放到踏面。低台阶高差只有 4 cm，
            # 不需要高台使用的长距离收腿预摆；过早收后腿会缩短支撑多边形，
            # 反而在随后抬前腿时把整机推离目标。
            # 第一次抬 FL 后还要完成 FL 回中与 FR 侧配重，支撑足会被拖拽
            # 约 2~3 cm。因此 FL 首落点额外前移 1 cm，仍通过实际足底与
            # 踏面接触验收，不修改或放宽登台判定阈值。
            low_front_top_x = float(edge_x) + LOW_STEP_FRONT_TOP_OFFSET_M
            low_fl_top_x = low_front_top_x + 0.010
            low_events = [
                QuasiEvent(
                    "LOW_STAND_SETTLE",
                    QuasiEventKind.JOINT_POSE,
                    0.45,
                    joint_pose_rad=tuple(float(v) for v in STAND_JOINT_RAD),
                ),
                hold("LOW_STAND_HOLD", 0.24),
                body_pose(
                    "LOW_PRELOAD_FORWARD",
                    dx=0.080,
                    duration_s=0.60,
                ),
                hold("LOW_PRELOAD_FORWARD_SETTLE", 0.24),
            ]

            # 低台阶先用小幅横移把机体重心移到三足支撑三角形内，再抬前足。
            # 这里仅移动约 4.5 cm：过小会让摆动时同侧支撑卸载，过大则会让
            # 支撑足被拖拽、机身回中时把刚落台的前足重新拉回地面。
            low_events.extend([
                body_pose("LOW_FL_BALANCE_SHIFT", dy=-0.045, duration_s=0.35),
                hold("LOW_FL_BALANCE_SHIFT_SETTLE", 0.14),
                swing(
                    "LOW_FL_TOP_STEP",
                    0,
                    low_fl_top_x,
                    top_foot_z - 0.006,
                    front_clearance_m,
                ),
                hold("LOW_FL_TOP_LAND", 0.52),
                # 前足落台后立即把机身向落点方向带 5 cm，避免固定前足
                # 锚点把刚建立的支撑反向拖回，形成前腿 IK 极限和机身下沉。
                body_pose(
                    "LOW_FL_RECENTER",
                    dx=0.050,
                    dy=0.045,
                    duration_s=0.35,
                ),
                hold("LOW_FL_RECENTER_SETTLE", 0.18),
                # 第二只前足抬起前继续小幅前移，保持第一只前足的髋-足
                # 水平距离处于舒适区，同时完成左右重心转移。
                body_pose(
                    "LOW_FR_BALANCE_SHIFT",
                    dx=0.050,
                    dy=0.045,
                    duration_s=0.35,
                ),
                hold("LOW_FR_BALANCE_SHIFT_SETTLE", 0.14),
                # FR 与 FL 同样预压 6 mm。若 FR 只落在理论台面高度，
                # 软接触和 PD 跟踪误差会让足底悬空约 0.3~1.0 mm，接触掩码
                # 随即变 0，CRAWL 的三足门控会误判为支撑不足。这里仍使用
                # LiDAR 顶面高度计算目标，不修改接触判定或成功阈值。
                swing(
                    "LOW_FR_TOP_STEP",
                    1,
                    low_front_top_x,
                    top_foot_z - 0.006,
                    front_clearance_m,
                ),
                hold("LOW_FR_TOP_LAND", 0.52),
                hold("LOW_FRONT_PAIR_SETTLE", 0.24),
            ])
            return low_events

        if state == FsmState.HIGH_STEP and high_step:
            # HIGH_STEP 采用“先原地立稳，再逐足收拢，再分段抬升重心”的
            # 保守顺序。每次只移动一条腿，禁止任何左右横移事件；后足每次
            # 前收约 0.10~0.12 m，避免像旧版那样从远处一次拉 0.30 m，
            # 造成 IK 静默裁剪、支撑足拖拽和机体侧翻。
            events = [
                QuasiEvent(
                    "HIGH_STAND_SETTLE",
                    QuasiEventKind.JOINT_POSE,
                    0.55,
                    joint_pose_rad=tuple(float(v) for v in STAND_JOINT_RAD),
                ),
                hold("HIGH_STAND_HOLD", 0.24),
            ]

            # 前足先落到前缘前 9 cm 的平地预备区。单腿离地前只需要把
            # 重心沿 -X 后移到其余三足支撑三角形内，不做大幅左右摆动：
            # 早期版本为了追求“绝对重心居中”横移 4.5~9.0 cm，反而把
            # 对侧支撑足拉离地面并诱发侧翻。这里采用 2.5 cm 后移、零横向
            # 位移，并在两只前足都落地后把重心平滑送回原位置。
            events.extend(
                [
                    body_pose(
                        "HIGH_FL_WEIGHT_SHIFT",
                        dx=-0.025,
                        dy=0.0,
                        duration_s=0.90,
                    ),
                    hold("HIGH_FL_WEIGHT_SHIFT_SETTLE", 0.30),
                    swing(
                        "HIGH_FL_PREP", 0,
                        front_ground_x, ground_foot_z, front_clearance_m
                    ),
                    hold("HIGH_FL_PREP_LAND", 0.45),
                    body_pose(
                        "HIGH_FR_WEIGHT_SHIFT",
                        dx=0.0,
                        dy=0.0,
                        duration_s=0.65,
                    ),
                    hold("HIGH_FR_WEIGHT_SHIFT_SETTLE", 0.30),
                    swing(
                        "HIGH_FR_PREP", 1,
                        front_ground_x, ground_foot_z, front_clearance_m
                    ),
                    hold("HIGH_FR_PREP_LAND", 0.45),
                    # 两只前足完成预备落点后先冻结世界锚点，不在此处强行
                    # 同时回正四足姿态。准静态 PD 下“四锚点 + 水平姿态”
                    # 属于过约束，会把已经落地的前足抬起；姿态修正只允许
                    # 在后续有空闲摆动腿时逐步完成。
                    hold("HIGH_FRONT_PREP_CENTER", 0.30),
                    hold("HIGH_FRONT_PREP_CENTER_SETTLE", 0.30),                ]
            )

            # 两前足前移后，机身在 PD 跟踪过程中会向后漂约 5~8 cm。
            # 抬后足前先把基座沿世界 +X 移回支撑多边形中央，缩短前腿
            # 相对髋部的伸展距离；否则前足会被动离开地面，导致后续三足
            # 支撑门控永远无法通过。
            events.extend(
                [
                    body_pose(
                        "HIGH_REAR_PREP_FORWARD",
                        dx=0.060,
                        duration_s=0.65,
                    ),
                    hold("HIGH_REAR_PREP_FORWARD_SETTLE", 0.30),
                ]
            )
            # 后足从当前实测足位分 10 个小步收拢到台阶前缘后 0.30 m。
            # 旧版三个硬编码目标中的第一档会造成约 0.29 m 的单次摆腿，
            # 在低台阶上足以把对侧前足从地面卸载。这里按实测起点等距
            # 分配位移，每档前后方向只移动约 0.04~0.05 m；每完成一对
            # 后足再让机身前移 0.022 m，使每条支撑腿都处在舒适工作区。
            rear_pull_stage_count = 10
            rear_final_x = float(edge_x) - 0.30
            rear_pull_fractions = np.linspace(
                0.0, 1.0, rear_pull_stage_count + 1, dtype=np.float64
            )[1:]
            for stage_index, fraction in enumerate(
                rear_pull_fractions, start=1
            ):
                # 左右后腿交替先摆，避免固定顺序让同一侧滚转误差连续累积。
                rear_leg_order = (
                    ((2, "RL"), (3, "RR"))
                    if stage_index % 2 == 1
                    else ((3, "RR"), (2, "RL"))
                )
                for leg_index, leg_name in rear_leg_order:
                    rear_target_x = float(
                        initial_rear_x[leg_index - 2]
                        + fraction
                        * (rear_final_x - initial_rear_x[leg_index - 2])
                    )
                    events.extend(
                        [
                            swing(
                                f"HIGH_{leg_name}_PULL_{stage_index}",
                                leg_index,
                                rear_target_x,
                                ground_foot_z,
                                rear_clearance_m,
                            ),
                            hold(
                                f"HIGH_{leg_name}_PULL_{stage_index}_LAND",
                                0.24,
                            ),
                        ]
                    )
                events.extend(
                    [
                        body_pose(
                            f"HIGH_BODY_FOLLOW_{stage_index}",
                            dx=0.022,
                            duration_s=0.34,
                        ),
                        hold(
                            f"HIGH_BODY_FOLLOW_{stage_index}_SETTLE",
                            0.16,
                        ),
                    ]
                )

            # 四足已经收拢到台阶附近后，分两次把前足抬到台面。前足目标
            # 只越过前缘 5.5 cm；两前足都落地并稳定后，才开始抬升机身。
            for leg_index, leg_name, balance_dy, recenter_dy in (
                (0, "FL", -0.045, 0.045),
                (1, "FR", 0.045, 0.000),
            ):
                events.extend(
                    [
                        swing(
                            f"HIGH_{leg_name}_TOP_STEP", leg_index,
                            front_top_x, top_foot_z, front_clearance_m
                        ),
                        hold(f"HIGH_{leg_name}_TOP_LAND", 0.30),
                    ]
                )
            events.append(hold("HIGH_FRONT_PAIR_SETTLE", 0.24))

            # 前足站台后，用四段小升降把机身抬到台面以上的安全高度。
            # 高台总抬升约 0.146 m，低台只抬约 0.05 m；所有段都由固定
            # 四足锚点做 IK，不会依赖瞬时关节插值或外力辅助。
            lift_total_m = max(0.05, min(0.146, 0.75 * rise_m))
            lift_segment_m = lift_total_m / 4.0
            for index in range(1, 5):
                events.extend(
                    [
                        body_pose(
                            f"HIGH_BODY_LIFT_{index}",
                            dz=lift_segment_m,
                            duration_s=0.40,
                        ),
                        hold(f"HIGH_BODY_LIFT_{index}_SETTLE", 0.16),
                    ]
                )

            # 机身抬高后分两批小步前移，把重心带到台阶前缘附近。绝不
            # 一次性冲过前缘，以便每一步都能通过 IK 诊断和接触门控发现
            # 后足超伸或支撑不足。
            for index in range(1, 8):
                events.extend(
                    [
                        body_pose(
                            f"HIGH_BODY_ADVANCE_{index}",
                            dx=0.040,
                            duration_s=0.38,
                        ),
                        hold(f"HIGH_BODY_ADVANCE_{index}_SETTLE", 0.14),
                    ]
                )

            # 重心越过前缘后再收后足上台。此时后足目标相对髋关节约
            # 0.31 m 水平差，仍小于当前腿长；若日志出现大量 IK 诊断，
            # 说明机身前移不足，应先调整前面的分段距离，不能强行增大抬腿。
            for leg_index, leg_name in ((2, "RL"), (3, "RR")):
                events.extend(
                    [
                        swing(
                            f"HIGH_{leg_name}_TOP_STEP", leg_index,
                            rear_top_x, top_foot_z, rear_clearance_m
                        ),
                        hold(f"HIGH_{leg_name}_TOP_LAND", 0.30),
                    ]
                )
            events.append(hold("HIGH_ALL_FEET_TOP_SETTLE", 0.24))

            # 最后只做少量前移和高度修正，把稳定姿态交给 CRAWL 状态。
            for index in range(1, 3):
                events.extend(
                    [
                        body_pose(
                            f"HIGH_FINAL_ADVANCE_{index}",
                            dx=0.030,
                            dz=0.010,
                            duration_s=0.38,
                        ),
                        hold(f"HIGH_FINAL_ADVANCE_{index}_SETTLE", 0.14),
                    ]
                )
            return events

        if state == FsmState.CRAWL and not high_step:
            # 低台阶 CRAWL 的主问题不是“机身不敢前进”，而是前足尚未真实
            # 承重时就连续搬动机身，固定世界锚点会把髋关节推到限位，随后
            # 整机下沉。这里改成与物理接触一致的顺序：
            # 1) 先逐只把前足重新压到台面并稳定；
            # 2) 后足在台前地面分段前收，每收一对只让机身跟随 2 cm；
            # 3) 机身进入后腿可达区后再逐只把后足抬上台面。
            # 所有落点仍由 LiDAR 锁存的前缘和顶面高度计算。
            low_rear_top_x = float(edge_x) + LOW_STEP_REAR_TOP_OFFSET_M
            # HIGH_STEP 已让两只前足落在踏面。进入 CRAWL 后不能立即重抬
            # 其中一只：旧轨迹会在 FL 重落时同时卸掉 FR 的接触，整机只剩
            # 两条后腿支撑并触发 0.824 rad 的 IK 裁剪。这里先保持当前步态，
            # 用两个 2 cm 的机身小步前移缩短前腿伸展；每个 BODY_POSE 仍由
            # 三足稳定接触门控放行，并保持现有俯仰/横滚，不在混合支撑面
            # 上强行水平化。前足真实接触后，再开始后腿收拢。
            crawl_events: list[QuasiEvent] = [
                hold("LOW_CRAWL_SETTLE", 2.00),
                # 前足已上台但机身仍偏后，FR 髋到足端水平距离接近工作空间
                # 上限。1 cm 前进不足以释放伸展，足底会稳定悬空约 1 mm。
                # 改为两步各 2 cm，并轻微下沉 4 mm 降低腿长需求；前后足
                # 仍分别锁在 LiDAR 台面和地面，不修改接触掩码或成功阈值。
                body_pose(
                    "LOW_CRAWL_BODY_FORWARD_0",
                    dx=0.020,
                    dz=-0.004,
                    duration_s=0.45,
                    level_attitude=False,
                ),
                hold("LOW_CRAWL_BODY_FORWARD_0_SETTLE", 0.22),
                body_pose(
                    "LOW_CRAWL_BODY_FORWARD_1",
                    dx=0.020,
                    dz=-0.004,
                    duration_s=0.45,
                    level_attitude=False,
                ),
                hold("LOW_CRAWL_BODY_FORWARD_1_SETTLE", 0.22),
                # 两段前移后 FR 会因软接触间隙短暂失去接触。此处显式
                # 重落 FR，恢复三足支撑门控所需的稳定接触后再抬后腿。
                swing(
                    "LOW_CRAWL_FR_REPLANT",
                    1,
                    float(edge_x) + LOW_STEP_FRONT_TOP_OFFSET_M,
                    top_foot_z - 0.006,
                    front_clearance_m,
                ),
                hold("LOW_CRAWL_FR_REPLANT_LAND", 0.52),
            ]

            # 后足仍在平地上时分段前收。旧版把机身向前推 14 cm，却只让
            # 腾空后足前移约 2 cm，导致腿长瞬间逼近极限，脚停在离地 3 cm
            # 处；随后又抬起唯一仍着地的后足，最终只剩两足支撑。这里改为
            # “机身前进多少，后足落点就同步前移多少”，每一步都保持腿长
            # 和支撑多边形几何基本不变。每个循环先承重前移 2.5 cm，再左右
            # 交替收拢后足 2.5 cm，避免半路停滞和大跨度伸腿。
            rear_cycle_count = 8
            # 每个循环只让机身净前移 5 cm，但拆成“公共 2.5 cm + RR 配重
            # 阶段 2.5 cm”。这样既保留总行程，又避免单次 5 cm 目标把
            # 前足拉到台面外。后足落点仍严格跟随累计机身前移量。
            # 后腿在机身保持不动时逐段向台前收拢；机身跟随后腿反而会在
            # 混合支撑面上持续下沉，因此这里把“机体前移”和“后足前收”解耦。
            rear_body_step_m = 0.025
            rear_foot_step_m = 0.025
            cumulative_rear_advance_m = 0.0
            # 前足落台后只做四足承重的水平前移，不在混合支撑面上抬升
            # 机身。后腿此时仍在地面，髋到足端的竖直距离已接近工作空间
            # 上限；先以 2.5 cm 小步缩短水平差，再收后腿，比同时抬高机身
            # 更不容易把支撑足卸载。
            for align_index in range(1, 1):
                crawl_events.extend(
                    [
                        body_pose(
                            f"LOW_CRAWL_REAR_ALIGN_{align_index}",
                            dx=0.025,
                            # 当前仅执行一段 4 mm 承重补偿，避免在混合
                            # 支撑面上一次抬升过多造成冲击；后续若增加段数，
                            # 必须同步复核总抬升量和足端工作空间。
                            dz=0.004,
                            duration_s=0.40,
                            level_attitude=False,
                        ),
                        hold(
                            f"LOW_CRAWL_REAR_ALIGN_{align_index}_SETTLE",
                            0.24,
                        ),
                    ]
                )
            cumulative_body_advance_m = 0.100
            for stage_index in range(1, rear_cycle_count + 1):
                cumulative_rear_advance_m += rear_foot_step_m
                crawl_events.extend(
                    [
                        body_pose(
                            f"LOW_CRAWL_BODY_FORWARD_{stage_index}",
                            dx=rear_body_step_m,
                            # 与后腿收拢同步补偿机身下沉，保持足端工作空间。
                            dz=0.0,
                            duration_s=0.30,
                            level_attitude=False,
                        ),
                        hold(
                            f"LOW_CRAWL_BODY_FORWARD_{stage_index}_SETTLE",
                            0.22,
                        ),
                    ]
                )
                rear_leg_order = (
                    ((2, "RL"), (3, "RR"))
                    if stage_index % 2 == 1
                    else ((3, "RR"), (2, "RL"))
                )
                for leg_index, leg_name in rear_leg_order:
                    # RL 的当前支撑三角形已能容纳机身，只需移除左侧后点
                    # 造成的横向偏心；RR 抬起前还需要再前进 2.5 cm，否则
                    # 机身投影会落在 FL-FR-RR 的前边界之外。配重拆成两次
                    # 小步，避免单帧改变足端相对髋部位置而拉掉前足接触。
                    balance_total_dx = 0.0
                    # 抬左后腿时必须把重心充分移入 FL+FR+RR 三角形；
                    # 目标配重按目标位姿计算，实测常只跟踪 20%~30%，因此
                    # 这里不能使用仅 2~3 cm 的名义值。
                    balance_total_dy = 0.0
                    balance_substeps = 2
                    balance_dx = balance_total_dx / balance_substeps
                    balance_dy = balance_total_dy / balance_substeps
                    # 零位移 BODY_POSE 仍会重新采样四足锚点并执行一次 IK。
                    # 低台阶混合支撑面上，这个微小 IK 重算会把刚接触台面的
                    # FR 足抬高约 1 mm，三足支撑门控随后永久等待。没有实际
                    # 配重需求时改用纯 HOLD，保持现有接触锚点不变。
                    if abs(balance_dx) > 1e-9 or abs(balance_dy) > 1e-9:
                        for balance_index in range(1, balance_substeps + 1):
                            cumulative_body_advance_m += balance_dx
                            crawl_events.extend(
                                [
                                    body_pose(
                                        f"LOW_{leg_name}_CRAWL_BALANCE_"
                                        f"{stage_index}_{balance_index}",
                                        dx=balance_dx,
                                        dy=balance_dy,
                                        dz=0.0,
                                        duration_s=0.28,
                                        level_attitude=False,
                                    ),
                                    hold(
                                        f"LOW_{leg_name}_CRAWL_BALANCE_"
                                        f"{stage_index}_{balance_index}_SETTLE",
                                        0.22,
                                    ),
                                ]
                            )
                    # 零位移事件没有真实的配重动作，且事件切换会重新采样
                    # 锚点并让 FR 抬离台面约 1 mm。前一条 FR 重落保持已提供
                    # 稳定窗口，因此不再插入无意义的 HOLD，直接进入摆腿门控。
                    # 落点与机身前移同步。若只按最终目标做很慢的插值，
                    # 机身前移会先把腿拉直；这里让足端始终大致位于髋部
                    # 正下方，PD 只需完成抬腿和落地，不再和超伸对抗。
                    rear_target_x = float(
                        initial_rear_x[leg_index - 2]
                        + cumulative_rear_advance_m
                    )
                    crawl_events.extend(
                        [
                            swing(
                                f"LOW_{leg_name}_CRAWL_PULL_{stage_index}",
                                leg_index,
                                rear_target_x,
                                ground_foot_z,
                                rear_clearance_m,
                            ),
                            # 后足按 2 cm 净空落地后需要时间建立真实接触。
                            # 0.24 s 不足以让闭环把 3~4 cm 的残余悬空压掉，
                            # 会误触发下一条后腿或机身事件；提高保持时间，
                            # 但不改变接触门限，也不伪造接触状态。
                            hold(
                                f"LOW_{leg_name}_CRAWL_PULL_{stage_index}_LAND",
                                0.70,
                            ),
                            hold(
                                f"LOW_{leg_name}_CRAWL_POST_LAND_{stage_index}",
                                0.12,
                            ),
                        ]
                    )

            # 补两段 1 cm 小步，把机身中心带到前缘附近。此时后足已同步
            # 前收约 30 cm，仍在地面上，四足相对髋部的水平距离保持可控。
            for index in range(1, 3):
                crawl_events.extend(
                    [
                        body_pose(
                            f"LOW_CRAWL_FINAL_FORWARD_{index}",
                            dx=0.010,
                            duration_s=0.32,
                            level_attitude=False,
                        ),
                        hold(
                            f"LOW_CRAWL_FINAL_FORWARD_{index}_SETTLE",
                            0.18,
                        ),
                    ]
                )

            # 后足已进入台前近区，逐只抬到踏面。此时落点相对髋部只有
            # 约 0.10~0.15 m，不需要额外大幅横移；保留较长落地保持以消除
            # 软接触间隙。左右顺序交替交换，减少同侧滚转误差累积。
            for leg_index, leg_name in ((2, "RL"), (3, "RR")):
                crawl_events.extend(
                    [
                        swing(
                            f"LOW_{leg_name}_TOP_STEP_FINAL",
                            leg_index,
                            low_rear_top_x,
                            top_foot_z,
                            0.020,
                        ),
                        hold(f"LOW_{leg_name}_TOP_LAND_FINAL", 0.70),
                    ]
                )
            crawl_events.append(hold("LOW_ALL_FEET_TOP_SETTLE", 0.34))

            # 四足均已接触同一台面后，才做最后的重心前移；此处可以安全地
            # 回正姿态，为 CLIMB_PUSH 提供稳定起点。
            for index in range(1, 4):
                crawl_events.extend(
                    [
                        body_pose(
                            f"LOW_CRAWL_FINAL_PUSH_{index}",
                            dx=0.025,
                            dz=0.004 if index == 1 else 0.0,
                            duration_s=0.30,
                        ),
                        hold(
                            f"LOW_CRAWL_FINAL_PUSH_{index}_SETTLE",
                            0.14,
                        ),
                    ]
                )
            crawl_events.append(hold("LOW_CRAWL_FINAL_HOLD", 0.22))
            return crawl_events


        if state == FsmState.CRAWL and high_step:
            # HIGH_STEP 已把四足放上台面并抬升重心。CRAWL 只做剩余的小步
            # 前移，保持四足锚点不变，避免在窄踏面上重复跨步造成自碰撞。
            events = [
                body_pose("CRAWL_SETTLE", duration_s=0.18),
            ]
            for index, dx in enumerate((0.045, 0.045, 0.045, 0.040), start=1):
                events.extend(
                    [
                        body_pose(
                            f"CRAWL_TOP_ADVANCE_{index}",
                            dx=dx,
                            dz=0.004 if index == 1 else 0.0,
                            duration_s=0.42,
                        ),
                        hold(f"CRAWL_TOP_ADVANCE_{index}_SETTLE", 0.14),
                    ]
                )
            events.append(hold("CRAWL_FINAL_HOLD", 0.22))
            return events

        # CLIMB_PUSH 只做小幅、可诊断的重心前压。主体抬身已经由 CRAWL
        # 完成，这里每段保持 1.5 cm，避免最后阶段再次触发支撑腿超伸。
        return [
            hold("PUSH_SETTLE", 0.28),
            body_pose(
                "PUSH_CENTER_1",
                dx=0.015,
                dz=0.004,
                duration_s=0.42,
            ),
            hold("PUSH_CENTER_1_SETTLE", 0.16),
            body_pose(
                "PUSH_CENTER_2",
                dx=0.015,
                dz=0.004,
                duration_s=0.42,
            ),
            hold("PUSH_VERIFY", 0.26),
        ]

    def _start_quasi_plan(self, state: FsmState, data: mujoco.MjData) -> None:
        """进入爬阶状态时初始化事件计划。

        参数：
            state：刚进入的 FSM 状态。
            data：当前 MuJoCo 数据。
        副作用：
            覆盖事件列表、事件索引、起始关节目标和诊断名称；不推进物理。
        重要性：A。
        """

        self._quasi_events = self._build_quasi_plan(state, data)
        self._quasi_event_index = -1
        self._quasi_event_started_s = float(data.time)
        self._quasi_plan_done = False
        self._quasi_event_name = "WAIT"
        self._quasi_last_target = np.asarray(
            data.qpos[self.qpos_adr], dtype=np.float64
        ).copy()
        # 跨状态时必须保留上一条轨迹已经规划的悬空足落点。低台阶从
        # HIGH_STEP 切入 CRAWL 时，两只前足可能刚越过前缘但尚未接触；
        # 若这里用“当前悬空实测位置”覆盖计划，HOLD 会把悬空高度误当作
        # 目标高度，结果无论等待多久都无法把前足压到台面。只有首次进入
        # 爬阶、尚无有效计划时，才用当前实测足位建立初始骨架。
        if not self._quasi_planned_anchor_valid:
            self._quasi_planned_foot_world[:] = (
                self._current_foot_centers(data)
            )

    def _support_anchors_with_z_correction(
        self,
        data: mujoco.MjData,
        support_mask: np.ndarray,
        anchors: np.ndarray,
    ) -> np.ndarray:
        """按支撑腿实测高度逐腿修正固定足端锚点。

        功能：
            准静态 PD 在机身平移或姿态回正时会留下毫米到厘米级跟踪误差。
            某条支撑足若因误差悬空，后续接触门控会永久等待。本函数把该腿
            的 IK 目标锚点逐控制周期向下积分；足端低于锚点过多时再缓慢
            回抬，避免持续顶地。

        参数：
            data：当前 MuJoCo 数据。
            support_mask：shape=(4,)，True 表示该腿参与支撑闭环，顺序
                FL/FR/RL/RR。
            anchors：shape=(4,3)，世界坐标锚点副本，单位 m。函数不会修改
                调用方传入的原数组。

        返回：
            shape=(4,3)，已按各支撑腿误差修正后的世界坐标锚点副本。

        副作用：
            更新 ``_quasi_support_z_correction_m``，单位为 m，限制在
            [-0.050, +0.004]。每个事件开始时由 ``_begin_quasi_event`` 清零。

        关键假设：
            ``_current_foot_centers`` 与锚点使用同一足端中心定义。锚点相对
            实测中心每次最多修正 2 mm；低台阶另有基于 LiDAR 地形面的 4 mm
            接触收敛，用来快速消除软接触间隙，同时避免单周期穿透台面。

        重要性：A（保证支撑足重新建立接触，避免状态机永久等待）。
        """

        corrected = np.asarray(anchors, dtype=np.float64).copy()
        actual_centers = self._current_foot_centers(data)
        for support_index in np.flatnonzero(support_mask):
            z_error_m = float(
                actual_centers[support_index, 2] - corrected[support_index, 2]
            )
            if z_error_m > 0.001:
                self._quasi_support_z_correction_m[support_index] -= min(
                    0.0020, 0.25 * (z_error_m - 0.001)
                )
            elif z_error_m < -0.004:
                self._quasi_support_z_correction_m[support_index] += min(
                    0.0004, 0.10 * (-z_error_m - 0.004)
                )
            # 低台阶实测会出现足底仍高于计划台面 3~4 cm 的情况；旧版
            # -25 mm 上限会在 HOLD 结束时仍留下悬空量，导致接触门控永久
            # 等待。下限放宽到 -50 mm，仅在支撑脚持续高于计划锚点时生效；
            # 一旦真实接触建立，误差项会停止继续下压。
            self._quasi_support_z_correction_m[support_index] = float(
                np.clip(
                    self._quasi_support_z_correction_m[support_index],
                    -0.050,
                    0.004,
                )
            )
            corrected[support_index, 2] += (
                self._quasi_support_z_correction_m[support_index]
            )

            # 仅靠“计划锚点—实测足心”的误差积分，会因每个 HOLD 都清零而
            # 在多次短事件之间只累计约 1 mm；低台阶前足因此会悬在台面上方
            # 约 1 cm，后续三足门控永久等待。这里额外使用 LiDAR 给出的真实
            # 地面/顶面高度，把足底直接向目标地形面收敛，不读取场景真值。
            feature = self.last_feature
            if not feature.visible:
                continue
            edge_x = self._estimated_edge_world_x()
            if edge_x is None:
                continue
            foot_x = float(actual_centers[support_index, 0])
            foot_y = float(actual_centers[support_index, 1])
            platform_y_min, platform_y_max = self._platform_lateral_bounds_m(
                feature
            )
            on_platform = (
                foot_x >= float(edge_x)
                and platform_y_min <= foot_y <= platform_y_max
            )
            surface_z = (
                float(feature.top_z_m) if on_platform else float(feature.floor_z_m)
            )
            target_bottom_z = surface_z + SUPPORT_SURFACE_CONTACT_SINK_M
            bottom_error_m = float(
                self._current_foot_bottoms_z(data)[support_index]
                - target_bottom_z
            )
            if bottom_error_m > 0.001:
                corrected[support_index, 2] -= min(
                    0.0040, 0.50 * (bottom_error_m - 0.001)
                )
            elif bottom_error_m < -0.005:
                # 若闭环保底地压穿地形，只允许缓慢回抬，防止接触力突然
                # 消失导致整机下落或横滚。
                corrected[support_index, 2] += min(
                    0.0005, 0.10 * (-bottom_error_m - 0.005)
                )
        return corrected
    def _begin_quasi_event(self, event: QuasiEvent, data: mujoco.MjData) -> None:
        """在一个事件开始时计算并锁定它的起点和终点关节目标。

        参数：
            event：即将执行的事件。
            data：事件开始时的 MuJoCo 数据。
        副作用：
            更新事件计时、起始关节、IK 终点和足端轨迹端点。
        异常：
            SWING 的腿编号不在 0..3 时抛 ValueError，防止计划配置错误静默
            变成错误腿动作。
        重要性：A。
        """

        self._quasi_event_started_s = float(data.time)
        self._quasi_event_name = event.name
        self._quasi_event_start_q = np.asarray(
            data.qpos[self.qpos_adr], dtype=np.float64
        ).copy()
        self._quasi_event_goal_q = self._quasi_event_start_q.copy()
        self._quasi_event_clearance_m = float(event.clearance_m)
        self._quasi_support_active_mask.fill(False)
        # 支撑足竖向位置积分修正：每条支撑腿只在检测到足端上漂时逐周期
        # 向下加深目标，避免统一偏移同时推动所有足端。
        self._quasi_support_z_correction_m.fill(0.0)

        if event.kind == QuasiEventKind.JOINT_POSE:
            if event.joint_pose_rad is None or len(event.joint_pose_rad) != 12:
                raise ValueError("JOINT_POSE 必须提供长度 12 的目标关节角")
            self._quasi_event_goal_q = np.clip(
                np.asarray(event.joint_pose_rad, dtype=np.float64),
                self.joint_low + 0.03,
                self.joint_high - 0.03,
            )
        elif event.kind == QuasiEventKind.BODY_POSE:
            base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
            rotation = np.asarray(
                data.xmat[self.base_body_id], dtype=np.float64
            ).reshape(3, 3)
            # BODY_POSE 只移动机身，不应把悬空腿的 PD 误差写成新的固定锚点。
            # 已接触足取实测值，未接触足保留计划落点，再由 IK 尝试把它放回
            # 目标高度；这样既保持真实支撑点，也允许落地失败的脚继续下沉。
            anchors, contact_mask = self._quasi_anchors_with_planned_fallback(
                data, update_contacted_plan=True
            )
            target_base = base + np.array(
                (event.body_dx_m, event.body_dy_m, event.body_dz_m),
                dtype=np.float64,
            )
            # 保存期望机座位姿轨迹的端点。事件内不再把“起点到终点的关节
            # 插值”和“按实际基座位姿求支撑 IK”叠加执行；两者方向不一致时
            # 会互相抵消，造成支撑足被抬离台面后整机回弹。
            self._quasi_event_start_base[:] = base
            self._quasi_event_goal_base[:] = target_base
            if event.level_body_attitude:
                self._quasi_event_goal_rotation[:] = level_rotation_preserving_yaw(
                    rotation
                )
            else:
                self._quasi_event_goal_rotation[:] = rotation
            # 终点 IK 必须使用与事件运行轨迹相同的目标姿态。若这里沿用
            # 事件起始时的倾斜 rotation，而运行中已按 level_body_attitude
            # 回正，终点关节会与实际支撑几何不一致并在事件末尾突然拉扯。
            self._quasi_support_active_mask[:] = True
            self._quasi_support_anchor_world[:] = anchors
            self._quasi_event_goal_q = self._joint_targets_for_base_pose(
                data, anchors, target_base, self._quasi_event_goal_rotation
            )
            shift = target_base - base
            foot_bottom_z = self._current_foot_bottoms_z(data)
            print(
                f"[BODY_POSE] {event.name} "
                f"base=({base[0]:.3f},{base[1]:.3f},{base[2]:.3f}) "
                f"target_delta=({shift[0]:+.3f},{shift[1]:+.3f},{shift[2]:+.3f})m "
                f"contact={self._contact_mask_text(contact_mask)} "
                f"foot_z=({foot_bottom_z[0]:.3f},{foot_bottom_z[1]:.3f},"
                f"{foot_bottom_z[2]:.3f},{foot_bottom_z[3]:.3f})",
                flush=True,
            )
        elif event.kind == QuasiEventKind.SWING:
            if event.leg_index not in range(4):
                raise ValueError(f"SWING 腿编号非法：{event.leg_index}")
            actual_centers = self._current_foot_centers(data)
            support_anchors, contact_mask = (
                self._quasi_anchors_with_planned_fallback(
                    data, update_contacted_plan=True
                )
            )
            # 摆动腿必须从真实当前位置出发；计划数组可能记录着上一次目标，
            # 若直接拿来当起点会造成事件开启瞬间的足端跳变。
            foot = actual_centers[event.leg_index].copy()
            self._quasi_event_start_foot = foot.copy()
            self._quasi_event_goal_foot = np.array(
                (event.foot_x_m, foot[1], event.foot_z_m), dtype=np.float64
            )
            # 这条腿的计划落点在动作开始时就更新；后续 BODY_POSE/HOLD
            # 只能复用这个计划点，不能再读取它的实时悬空位置。
            self._quasi_planned_foot_world[event.leg_index] = (
                self._quasi_event_goal_foot.copy()
            )
            # 支撑腿闭环必须锁定“事件开始时的目标机座位姿”，而不是每帧
            # 追随已经下沉/后漂的实际基座。后者会让关节目标随着重力一起
            # 收敛到折叠姿态，表现为抬一条腿就整机下沉并倒退。这里只保存
            # 水平目标姿态；摆动腿本身仍按当前实际基座做足端 IK。
            base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
            rotation = np.asarray(
                data.xmat[self.base_body_id], dtype=np.float64
            ).reshape(3, 3)
            self._quasi_event_start_base[:] = base
            self._quasi_event_goal_base[:] = base
            # 摆动期间保持事件开始的实际高度，只锁三足支撑几何。若此时
            # 同时追求名义高度，支撑足会在单腿离地瞬间产生额外抬升力矩，
            # 容易把机体推向摆动侧；高度恢复放到四足承重的 HOLD/BODY_POSE。
            self._quasi_event_goal_rotation[:] = level_rotation_preserving_yaw(
                rotation
            )
            # 只锁当前真正接触的三条支撑足；若门控刚确认接触，实测点就是
            # 最可靠的世界锚点。摆动足及其余悬空足都不得进入支撑闭环，否则
            # 它们会跟着支撑姿态一起被拖动，形成虚假三足支撑。
            self._quasi_support_anchor_world[:] = support_anchors
            self._quasi_support_active_mask[:] = contact_mask
            # 低台阶抬后腿时，前足可能因混合支撑面和腿长上限悬空 1~2 mm。
            # 若只闭环真实接触腿，摆动期间会只剩两足受控并迅速下沉。这里
            # 在“两后足真实支撑 + 前足已位于台面且接近台面”的准接触窗口内，
            # 把该前足加入支撑闭环，主动把足底压回 LiDAR 台面；该掩码只用于
            # 控制稳定性，不参与真实接触统计或登顶成功判定。
            if (
                int(event.leg_index) >= 2
                and self.state == FsmState.CRAWL
                and bool(np.all(contact_mask[2:]))
            ):
                feature = self.last_feature
                edge_x = self._estimated_edge_world_x()
                if feature.visible and edge_x is not None:
                    front_centers = actual_centers[:2]
                    front_bottoms = self._current_foot_bottoms_z(data)[:2]
                    y_min, y_max = self._platform_lateral_bounds_m(feature)
                    front_over_platform = (
                        (front_centers[:, 0] >= float(edge_x))
                        & (front_centers[:, 1] >= y_min)
                        & (front_centers[:, 1] <= y_max)
                    )
                    front_near_contact = front_bottoms <= (
                        float(feature.top_z_m)
                        + LOW_STEP_FRONT_QUASI_CONTACT_MAX_CLEARANCE_M
                    )
                    quasi_front_mask = front_over_platform & front_near_contact
                    if np.any(quasi_front_mask):
                        self._quasi_support_active_mask[:2] |= quasi_front_mask
                        print(
                            "[准接触支撑] 前足纳入摆动期稳定闭环："
                            f"contact={self._contact_mask_text(contact_mask)} "
                            f"support={self._contact_mask_text(self._quasi_support_active_mask)}",
                            flush=True,
                        )
            self._quasi_support_active_mask[event.leg_index] = False
            foot_xy = actual_centers[:, :2]
            print(
                f"[SWING] {event.name} contact="
                f"{self._contact_mask_text(contact_mask)} "
                f"support="
                f"{self._contact_mask_text(self._quasi_support_active_mask)} "
                f"foot_xy=({foot_xy[0,0]:.3f},{foot_xy[0,1]:.3f})/"
                f"({foot_xy[1,0]:.3f},{foot_xy[1,1]:.3f})/"
                f"({foot_xy[2,0]:.3f},{foot_xy[2,1]:.3f})/"
                f"({foot_xy[3,0]:.3f},{foot_xy[3,1]:.3f})",
                flush=True,
            )
        elif event.kind == QuasiEventKind.HOLD:
            # 落地后的 HOLD 同样锁定事件起始机座位姿和四足世界锚点。若按
            # 实际基座每帧重算，支撑腿只会跟随下沉而不会产生恢复力；固定
            # 目标位姿才能让 PD 通过腿长和姿态误差把机体托回稳定高度。
            centers, contact_mask = self._quasi_anchors_with_planned_fallback(
                data, update_contacted_plan=True
            )
            base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
            rotation = np.asarray(
                data.xmat[self.base_body_id], dtype=np.float64
            ).reshape(3, 3)
            self._quasi_event_start_base[:] = base
            self._quasi_event_goal_base[:] = base
            # HOLD 不主动回正姿态；若这里把目标旋转改成水平，刚落地足会
            # 因横滚修正瞬间改变腿长并失去接触。真正的姿态回正交给后续
            # BODY_POSE 事件分段完成。
            self._quasi_event_goal_rotation[:] = rotation
            self._quasi_support_anchor_world[:] = centers
            # HOLD 期间四条腿都应当是支撑腿。刚落地足即使尚未建立接触，
            # 也使用上一条 SWING 记录的计划落点作为候选锚点，由逐腿高度
            # 闭环把它压回地面；是否真正稳定仍由接触门控独立判定。
            self._quasi_support_active_mask[:] = True
            print(
                f"[HOLD] {event.name} contact="
                f"{self._contact_mask_text(contact_mask)} "
                f"foot_z=({self._current_foot_bottoms_z(data)[0]:.3f},"
                f"{self._current_foot_bottoms_z(data)[1]:.3f},"
                f"{self._current_foot_bottoms_z(data)[2]:.3f},"
                f"{self._current_foot_bottoms_z(data)[3]:.3f})",
                flush=True,
            )

    def _swing_support_ready(
        self,
        data: mujoco.MjData,
        event: QuasiEvent,
        now_s: float,
    ) -> bool:
        """检查下一条摆动腿之外的另外三足是否都接触地形。

        功能：
            在开启任何 SWING 事件前做三足支撑门控，避免“支撑腿仍悬空
            就抬下一条腿”的准静态失稳。等待期间只返回保持目标，主循环
            仍持续推进 ``mj_step``，因此不会阻塞仿真或窗口刷新。

        参数：
            data：当前 MuJoCo 数据，接触缓存来自最近一次物理步。
            event：即将开启的 SWING 事件，必须带有 0..3 的腿编号。
            now_s：当前仿真时间，单位 s，用于节流日志与超时保护。

        返回：
            True 表示另外三足全部接触地形，可以开启摆动；False 表示仍需
            等待，调用方应继续输出上一条事件的安全保持目标。

        副作用：
            更新等待计时并打印节流诊断；等待超过 3 s 时把 FSM 切换到
            FAILED，避免无限等待被误认为正常运行。

        重要性：A（准静态爬阶的硬安全约束）。
        """

        mask = self._foot_contact_mask(data)
        support_mask = mask.copy()
        support_mask[int(event.leg_index)] = False
        support_count = int(np.count_nonzero(support_mask))
        required_count = 3
        # 低台阶后腿摆动时，FR 可能因混合支撑面和 IK 伸展停在台面上方
        # 约 1~2 mm；若仍按真实接触计数，会形成永久支撑等待。这里仅在
        # CRAWL 抬后腿时，把“位于台面范围内且距台面不超过既有准接触门限”
        # 的前足计入支撑候选；真实两足仍必须满足稳定接触窗口，登顶判定
        # 不读取该候选掩码，因此不会伪造成功。
        support_contact_for_stability = support_mask.copy()
        if (
            support_count < required_count
            and int(event.leg_index) >= 2
            and self.state == FsmState.CRAWL
        ):
            feature = self.last_feature
            edge_x = self._estimated_edge_world_x()
            if (
                feature.visible
                and feature.rise_m <= 0.10
                and edge_x is not None
                and bool(np.all(mask[2:]))
            ):
                front_centers = self._current_foot_centers(data)[:2]
                front_bottoms = self._current_foot_bottoms_z(data)[:2]
                y_min, y_max = self._platform_lateral_bounds_m(feature)
                front_over_platform = (
                    (front_centers[:, 0] >= float(edge_x))
                    & (front_centers[:, 1] >= y_min)
                    & (front_centers[:, 1] <= y_max)
                )
                front_near_contact = front_bottoms <= (
                    float(feature.top_z_m)
                    + LOW_STEP_FRONT_QUASI_CONTACT_MAX_CLEARANCE_M
                )
                quasi_front_mask = front_over_platform & front_near_contact
                support_mask[:2] |= quasi_front_mask
                support_count = int(np.count_nonzero(support_mask))
                if np.any(quasi_front_mask):
                    support_contact_for_stability[:2] &= ~quasi_front_mask
        if (
            support_count >= required_count
            and self._stable_contact_ready(data, support_contact_for_stability)
        ):
            self._quasi_event_open = True
            self._quasi_support_wait_started_s = now_s
            self._quasi_support_wait_last_log_s = -1e9
            self._quasi_support_wait_event_name = ""
            return True

        if self._quasi_support_wait_event_name != event.name:
            self._quasi_support_wait_event_name = event.name
            self._quasi_support_wait_started_s = now_s
            self._quasi_support_wait_last_log_s = -1e9
        self._quasi_event_open = False

        if now_s - self._quasi_support_wait_last_log_s >= QUASI_SWING_SUPPORT_WAIT_LOG_S:
            contacts = "".join("1" if value else "0" for value in mask)
            ages = np.round(self._foot_contact_age_s, 3).tolist()
            print(
                f"[支撑等待] {event.name} 需要 3 足稳定支撑，"
                f"当前 {support_count} 足，接触掩码={contacts}，"
                f"接触时长={ages}，门限={QUASI_STANCE_SETTLE_S:.2f}s",
                flush=True,
            )
            self._quasi_support_wait_last_log_s = now_s

        if now_s - self._quasi_support_wait_started_s > QUASI_SWING_SUPPORT_TIMEOUT_S:
            print(
                f"[支撑超时] {event.name} 等待三足接触超过 "
                f"{QUASI_SWING_SUPPORT_TIMEOUT_S:.1f}s，终止本轮。",
                flush=True,
            )
            self._transition_to(FsmState.FAILED, data, gait_delay_s=0.0)
        return False

    def _body_pose_support_ready(
        self,
        data: mujoco.MjData,
        event: QuasiEvent,
        now_s: float,
    ) -> bool:
        """在移动机身前确认四足已连续稳定接触。

        参数：
            data：当前 MuJoCo 数据。
            event：下一条 BODY_POSE 事件，仅用于日志定位。
            now_s：当前仿真时间，单位 s。
        返回：
            True 表示满足以下任一安全条件：至少三足连续稳定接触；或低台阶
            前进阶段两条后足稳定接触，且两只前足已经位于台面上方、足底与
            台面高差不超过准接触门限。False 表示保持上一事件目标，主循环
            继续推进物理。
        副作用：
            打印节流等待日志；等待超过 ``QUASI_SWING_SUPPORT_TIMEOUT_S``
            时把 FSM 置为 FAILED，避免机身被错误抬升或持续拖拽。
        重要性：A（机身平移前的四足承重安全门控）。
        """

        contact_mask = self._foot_contact_mask(data)
        contact_count = int(np.count_nonzero(contact_mask))
        # 机身平移只需稳定三足支撑即可执行。若严格要求四足，某只前足
        # 因腿长接近上限而轻微离地时，机身既不能前移缩短腿长，也不能
        # 恢复接触，会形成永久自锁。三足接触仍由连续接触窗口过滤，
        # 悬空腿会在随后的 HOLD 中由逐腿高度闭环重新压回地面。
        if contact_count >= 3 and self._stable_contact_ready(data, contact_mask):
            self._quasi_stance_wait_started_s = now_s
            self._quasi_stance_wait_last_log_s = -1e9
            self._quasi_stance_wait_event_name = ""
            return True
        # 低台阶前足刚伸到台面上方时，机身后移会把前腿拉到约 0.42 m 的
        # 水平伸展；此时前足足底虽已在台面上方，却因 IK 接近腿长上限而
        # 无法继续下压，形成“机身前移才能缩短腿长、前足接触才能允许机身
        # 前移”的互锁。这里只对“后双腿稳定接触 + 两前足真实位于台面
        # 上方且高差小于 12 mm”放行准接触门控，不伪造接触；机身前移后
        # 仍由后续 HOLD/接触缓存确认足端真正承重。门限设得过大会掩盖
        # 落足失败，设得过小则会重新锁死，因此仅用于低台阶 0.04 m 课程。
        feature = self.last_feature
        edge_x = self._estimated_edge_world_x()
        if (
            feature.visible
            and feature.rise_m <= 0.10
            and edge_x is not None
            and bool(np.all(contact_mask[2:]))
            and self._stable_contact_ready(data, contact_mask)
            and event.name.startswith("LOW_CRAWL_BODY_FORWARD_")
            and 0.0 < float(event.body_dx_m) <= 0.03001
            and abs(float(event.body_dy_m)) < 1e-9
            and -0.01001 <= float(event.body_dz_m) <= 0.0
        ):
            foot_centers = self._current_foot_centers(data)
            foot_bottoms = self._current_foot_bottoms_z(data)
            platform_top_z = float(feature.top_z_m)
            platform_y_min, platform_y_max = self._platform_lateral_bounds_m(
                feature
            )
            front_over_platform = (
                (foot_centers[:2, 0] >= float(edge_x))
                & (foot_centers[:2, 1] >= platform_y_min)
                & (foot_centers[:2, 1] <= platform_y_max)
            )
            front_near_contact = (
                foot_bottoms[:2]
                <= platform_top_z + LOW_STEP_FRONT_QUASI_CONTACT_MAX_CLEARANCE_M
            )
            if bool(np.all(front_over_platform & front_near_contact)):
                if event.name != self._quasi_stance_wait_event_name:
                    print(
                        f"[准接触门控] {event.name} 后足稳定支撑，"
                        f"前足位于台面上方且高差≤"
                        f"{LOW_STEP_FRONT_QUASI_CONTACT_MAX_CLEARANCE_M * 1000:.0f}mm，"
                        f"允许 3 cm 小幅前移解除腿长互锁。",
                        flush=True,
                    )
                self._quasi_stance_wait_started_s = now_s
                self._quasi_stance_wait_last_log_s = -1e9
                self._quasi_stance_wait_event_name = ""
                return True

        if self._quasi_stance_wait_event_name != event.name:
            self._quasi_stance_wait_event_name = event.name
            self._quasi_stance_wait_started_s = now_s
            self._quasi_stance_wait_last_log_s = -1e9

        if now_s - self._quasi_stance_wait_last_log_s >= QUASI_SWING_SUPPORT_WAIT_LOG_S:
            ages = np.round(self._foot_contact_age_s, 3).tolist()
            print(
                f"[机身等待] {event.name} 需要四足稳定接触，"
                f"contact={self._contact_mask_text(contact_mask)}，"
                f"接触时长={ages}，门限={QUASI_STANCE_SETTLE_S:.2f}s",
                flush=True,
            )
            self._quasi_stance_wait_last_log_s = now_s

        if now_s - self._quasi_stance_wait_started_s > QUASI_SWING_SUPPORT_TIMEOUT_S:
            print(
                f"[支撑超时] {event.name} 四足未在 "
                f"{QUASI_SWING_SUPPORT_TIMEOUT_S:.1f}s 内稳定，终止本轮。",
                flush=True,
            )
            self._transition_to(FsmState.FAILED, data, gait_delay_s=0.0)
        return False

    def _update_quasi_control(
        self,
        data: mujoco.MjData,
        now_s: float,
    ) -> np.ndarray:
        """推进准静态事件并返回当前 12 关节目标角。

        参数：
            data：当前 MuJoCo 数据。
            now_s：控制边界仿真时间，单位 s。
        返回：
            shape=(12,)，单位 rad。
        算法：
            HOLD：整段只保持事件起始关节角，不叠加支撑足闭环，避免刚落地足被抬离；
            BODY_POSE：对起止机座位姿做余弦插值，每个控制周期按期望基座
            位姿和固定足端锚点重算四足 IK；SWING：只对摆动腿按世界足端
            余弦轨迹做 IK，其余三腿用有限幅闭环保持锚点。事件切换不会
            继承上一事件的瞬时关节速度。
        重要性：A（替代逐帧重算全部支撑腿的旧控制路径）。
        """

        if not self._quasi_events:
            return self._quasi_last_target.copy()

        if self._quasi_event_index < 0:
            self._quasi_event_index = 0
            self._begin_quasi_event(self._quasi_events[0], data)

        while True:
            event = self._quasi_events[self._quasi_event_index]
            elapsed_s = now_s - self._quasi_event_started_s
            if elapsed_s < event.duration_s:
                break
            if self._quasi_event_index + 1 >= len(self._quasi_events):
                self._quasi_plan_done = True
                return self._quasi_last_target.copy()
            next_index = self._quasi_event_index + 1
            next_event = self._quasi_events[next_index]
            if next_event.kind == QuasiEventKind.SWING:
                if not self._swing_support_ready(data, next_event, now_s):
                    if self.state == FsmState.FAILED:
                        return self._quasi_last_target.copy()
                    return self._quasi_last_target.copy()
            elif next_event.kind == QuasiEventKind.BODY_POSE:
                # 机座位姿变化会把所有真实支撑腿一起加载；必须等四足都
                # 连续稳定接触后再开始。尤其要覆盖“刚落地一条腿”的 HOLD
                # 结束点，避免单帧接触被当成已承重。
                if not self._body_pose_support_ready(data, next_event, now_s):
                    if self.state == FsmState.FAILED:
                        return self._quasi_last_target.copy()
                    return self._quasi_last_target.copy()
            self._quasi_event_index = next_index
            self._begin_quasi_event(
                self._quasi_events[self._quasi_event_index], data
            )

        event = self._quasi_events[self._quasi_event_index]
        progress = float(np.clip(
            (now_s - self._quasi_event_started_s) / max(event.duration_s, 1e-6),
            0.0,
            1.0,
        ))
        if event.kind == QuasiEventKind.HOLD:
            # HOLD 不能“跟着实际基座一起下沉”，否则 IK 只是维持已经塌下去
            # 的高度，关节 PD 不会产生恢复力。这里根据当前支撑足的高度估计
            # 一个水平机身目标：足端越低，腿目标越伸展，从而把机身托回
            # 名义站立高度；水平位置仍取实测值，避免 HOLD 擅自平移。
            current_base = np.asarray(
                data.xpos[self.base_body_id], dtype=np.float64
            ).copy()
            current_rotation = np.asarray(
                data.xmat[self.base_body_id], dtype=np.float64
            ).reshape(3, 3)
            # HOLD 以事件起始机座位姿和足端锚点为参考，并逐腿修正悬空
            # 误差。若只冻结上一事件关节目标，机身一旦在 PD 误差下发生
            # 俯仰，某条足会悬空且再也不会被拉回，导致后续三足门控永久
            # 等待；这里保持固定参考位姿，才能持续产生恢复接触的关节力。
            self._quasi_event_goal_base[:] = self._quasi_event_start_base
            corrected_anchors = self._support_anchors_with_z_correction(
                data,
                self._quasi_support_active_mask,
                self._quasi_support_anchor_world,
            )
            target = self._joint_targets_for_base_pose(
                data,
                corrected_anchors,
                self._quasi_event_goal_base,
                self._quasi_event_goal_rotation,
            )
        elif event.kind == QuasiEventKind.JOINT_POSE:
            target = cosine_blend(
                self._quasi_event_start_q,
                self._quasi_event_goal_q,
                progress,
            )
        elif event.kind == QuasiEventKind.BODY_POSE:
            # 直接沿“期望机座位姿”生成运动学一致的四足目标。若改用固定
            # 终点关节插值，中途的关节轨迹并不严格对应直线机座位姿，落地
            # 摩擦又会把支撑足推离锚点；在本体上表现为机身抖动和支撑足抬起。
            desired_base = cosine_blend(
                self._quasi_event_start_base,
                self._quasi_event_goal_base,
                progress,
            )
            corrected_anchors = self._support_anchors_with_z_correction(
                data,
                self._quasi_support_active_mask,
                self._quasi_support_anchor_world,
            )
            # 机座指令必须使用事件期望位姿，不能跟随实测机身。固定期望
            # 机座与固定世界足端锚点之间会形成明确的腿长/摆角误差，由 PD
            # 持续产生托举机身的恢复力；若让指令跟随实测机座，误差会被
            # “追着错误走”，机身高度推进就会锁死。
            target = self._joint_targets_for_base_pose(
                data,
                corrected_anchors,
                desired_base,
                self._quasi_event_goal_rotation,
            )
        else:
            # 三段式摆动轨迹：先原地抬到安全高度，再保持高度水平跨越，
            # 最后在目标点竖直落下。旧版线性插值会让足端斜着撞向台阶
            # 立面；三段路径把“越障”和“落地”拆开，尤其适合 0.195 m 高台。
            start = self._quasi_event_start_foot
            goal = self._quasi_event_goal_foot
            safe_z = max(float(start[2]), float(goal[2])) + float(
                self._quasi_event_clearance_m
            )
            if progress < 0.28:
                # 第一段只改变 Z：从触地高度垂直抬到安全高度。
                segment_progress = progress / 0.28
                foot_goal = start.copy()
                foot_goal[2] = cosine_blend(
                    float(start[2]), safe_z, segment_progress
                )
            elif progress < 0.68:
                # 第二段只改变 X/Y：足底在安全高度上水平跨过台阶前缘。
                segment_progress = (progress - 0.28) / 0.40
                foot_goal = start.copy()
                foot_goal[:2] = cosine_blend(
                    start[:2], goal[:2], segment_progress
                )
                foot_goal[2] = safe_z
            else:
                # 第三段只改变 Z：落到台面目标高度，避免把水平速度带进
                # 最后一次接触造成弹跳。
                segment_progress = (progress - 0.68) / 0.32
                foot_goal = goal.copy()
                foot_goal[2] = cosine_blend(
                    safe_z, float(goal[2]), segment_progress
                )
            # CRAWL 的摆动相在“已抬到最高点、正在水平前摆”的第二段
            # 额外增加前伸量，让脚先越过前缘再下落，避免垂直砸向立面。
            # 只作用于水平段：第三段若再次叠加，会让落点越过目标并破坏
            # 三段轨迹的首尾连续性。
            if self.state == FsmState.CRAWL and 0.28 <= progress < 0.68:
                horizontal_progress = float(
                    np.clip((progress - 0.28) / 0.40, 0.0, 1.0)
                )
                foot_goal[0] += CRAWL_SWING_FORWARD_EXTENSION_M * math.sin(
                    math.pi * horizontal_progress
                )
            base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
            rotation = np.asarray(
                data.xmat[self.base_body_id], dtype=np.float64
            ).reshape(3, 3)
            relative = (foot_goal - base) @ rotation
            leg_index = event.leg_index
            hip, thigh, calf = leg_inverse_kinematics(
                relative[0],
                relative[1],
                relative[2],
                HIP_X_M[leg_index],
                HIP_Y_M[leg_index],
                LEG_THIGH_Y_OFFSET_M[leg_index],
                hip_low_rad=self.joint_low[3 * leg_index],
                hip_high_rad=self.joint_high[3 * leg_index],
            )
            target = self._quasi_last_target.copy()
            target[3 * leg_index] = hip
            target[3 * leg_index + 1] = thigh
            target[3 * leg_index + 2] = calf

        # 只有 SWING 需要支撑足有限幅闭环。HOLD 的职责是让刚落地足
        # 真正稳定下来：若继续叠加“事件起始关节 + 支撑 IK 修正”，IK
        # 与 PD 实际关节的微小差异会在数秒内把新落地足抬起。BODY_POSE
        # 本身已经按目标机座位姿直接求四足 IK，也不能再被闭环覆盖。
        if (
            event.kind == QuasiEventKind.SWING
            and np.any(self._quasi_support_active_mask)
        ):
            # HOLD/SWING 的支撑腿必须相对事件开始时锁定的目标机座位姿
            # 求 IK。这样实际基座一旦下沉或后漂，固定足端锚点会形成明确
            # 的腿长/摆角误差，由 PD 产生恢复力；不能使用实时基座，否则
            # 目标会“跟着错误走”，系统没有可恢复的参考。
            # SWING 期间保持“事件起始目标机座位姿”求支撑足 IK。摆动腿
            # 的足端轨迹已经按实时机身位姿求 IK；若三条支撑腿也逐帧追
            # 实时姿态，机身横滚会把支撑目标一起带走，反而放大侧倾。
            support_base = self._quasi_event_goal_base
            support_rotation = self._quasi_event_goal_rotation
            # 三足支撑阶段保持事件开始时的世界足端锚点，并对每条支撑腿
            # 单独做竖向误差积分。这里只修正“实际足端高于锚点”的悬空
            # 误差，不做统一前后或横向推地，避免把另一侧的支撑足拉离地面。
            support_indices = np.flatnonzero(self._quasi_support_active_mask)
            support_anchor_world = self._support_anchors_with_z_correction(
                data,
                self._quasi_support_active_mask,
                self._quasi_support_anchor_world,
            )
            relative_support = (
                support_anchor_world - support_base
            ) @ support_rotation
            for support_index in support_indices:
                support_hip, support_thigh, support_calf = (
                    leg_inverse_kinematics(
                        relative_support[support_index, 0],
                        relative_support[support_index, 1],
                        relative_support[support_index, 2],
                        HIP_X_M[support_index],
                        HIP_Y_M[support_index],
                        LEG_THIGH_Y_OFFSET_M[support_index],
                        hip_low_rad=self.joint_low[3 * support_index],
                        hip_high_rad=self.joint_high[3 * support_index],
                    )
                )
                support_goal = np.array(
                    (support_hip, support_thigh, support_calf),
                    dtype=np.float64,
                )
                joint_slice = slice(3 * support_index, 3 * support_index + 3)
                support_delta = np.clip(
                    support_goal - target[joint_slice],
                    -QUASI_SUPPORT_JOINT_STEP_LIMIT_RAD,
                    QUASI_SUPPORT_JOINT_STEP_LIMIT_RAD,
                )
                target[joint_slice] += support_delta
        target = self._clip_joint_target(
            target, f"QUASI_{self._quasi_event_name}", now_s=now_s
        )
        self._quasi_last_target = np.asarray(target, dtype=np.float64).copy()
        return self._quasi_last_target

    def _refresh_gait(self, data: mujoco.MjData) -> None:
        """更新摆动开始/结束事件和足端世界锚点。

        参数：
            data：当前 MuJoCo 数据。
        说明：
            只在 50 Hz 控制边界调用；每条腿进入摆动相时规划新落点，
            摆动结束时把目标锁为新的 stance anchor。这样 stance 足端在
            世界系中保持不动，机身由关节反作用力推动前进。
        重要性：A。
        """

        if self.state not in self.patterns:
            return
        now_s = float(data.time)
        if now_s < self._gait_start_s:
            return
        current_feet = self._current_foot_centers(data)
        if not self._gait_rebased:
            # 站立预热结束的实际足位才是后续世界锚点。不能沿用切换瞬间可能
            # 仍处于摆动相的历史位置，否则第一条腿会被 IK 拉向远处。
            self._anchor_world = current_feet.copy()
            self._swing_start_world[:] = self._anchor_world
            self._swing_goal_world[:] = self._anchor_world
            self._body_foot_height_ref_m = float(
                data.xpos[self.base_body_id, 2] - np.mean(self._anchor_world[:, 2])
            )
            profile = self.patterns[self.state]
            # 从明确的四足落地窗口重建相位，而不是把周期直接放到某只腿
            # 的摆动区间。当前顺序步态的相位间隔为 0.25、摆动占比为
            # 0.24，周期点 0.245 正好位于 FL 落地后、下一条腿离地前。
            # 同时按该周期点设置 ``_prev_local_phase``，使下一次摆动事件
            # 能被准确识别；旧实现把“上一相位”与新的 gait 起点混用，
            # 导致实际抬 FR 时按 FL 计算支撑偏移，机体迅速侧滑。
            rebased_cycle = float(profile.swing_fraction + 0.005) % 1.0
            self._gait_start_s = now_s - rebased_cycle * profile.period_s
            cycle = (now_s - self._gait_start_s) / profile.period_s
            self._prev_local_phase = (
                cycle + profile.phase_offsets
            ) % 1.0
            self._gait_rebased = True
            return
        profile = self.patterns[self.state]
        cycle = (now_s - self._gait_start_s) / profile.period_s
        for leg_index in range(4):
            local_phase = (cycle + profile.phase_offsets[leg_index]) % 1.0
            prev_phase = float(self._prev_local_phase[leg_index])
            swing_now = local_phase < profile.swing_fraction
            swing_before = prev_phase < profile.swing_fraction
            wrapped_into_swing = local_phase < 0.08 and prev_phase > 0.90
            if swing_now and (not swing_before or wrapped_into_swing):
                # 用触地时的真实足位作为轨迹起点，避免长时间 stance 驱动后
                # 世界锚点与几何位置不一致造成瞬移。
                self._swing_start_world[leg_index] = current_feet[leg_index].copy()
                self._swing_goal_world[leg_index] = self._plan_foothold(
                    leg_index, data, now_s
                )
            elif swing_before and not swing_now:
                # 落地后重新以实际足位建立锚点；即使接触求解器产生几毫米
                # 穿透，也不会在下一 stance 周期累积为拖滑。
                self._anchor_world[leg_index] = current_feet[leg_index].copy()
            self._prev_local_phase[leg_index] = local_phase

    def _desired_world_feet(
        self,
        data: mujoco.MjData,
    ) -> np.ndarray:
        """计算当前时刻四足足端球心世界目标，shape=(4,3)。

        说明：
            摆动相使用余弦插值连接起止落点，并叠加正弦净空弧线；
            支撑相保持世界锚点。CLIMB_PUSH 额外把所有足端目标向后推，
            通过摩擦让机身向 +X 前移，而不是直接修改基座位置。
        重要性：A。
        """

        desired = self._anchor_world.copy()
        swing_mask = np.zeros(4, dtype=bool)
        if self.state in self.patterns and float(data.time) >= self._gait_start_s:
            profile = self.patterns[self.state]
            cycle = (float(data.time) - self._gait_start_s) / profile.period_s
            for leg_index in range(4):
                local_phase = (cycle + profile.phase_offsets[leg_index]) % 1.0
                if local_phase < profile.swing_fraction:
                    swing_mask[leg_index] = True
                    progress = local_phase / profile.swing_fraction
                    alpha = cosine_blend(0.0, 1.0, progress)
                    start = self._swing_start_world[leg_index]
                    goal = self._swing_goal_world[leg_index]
                    xyz = start + alpha * (goal - start)
                    arc = math.sin(math.pi * float(np.clip(progress, 0.0, 1.0)))
                    xyz[2] += self._swing_clearance(leg_index) * arc
                    desired[leg_index] = xyz
                else:
                    desired[leg_index] = self._anchor_world[leg_index]
                    if self.state in (FsmState.HIGH_STEP, FsmState.CRAWL):
                        stance_progress = (
                            (local_phase - profile.swing_fraction)
                            / max(1e-6, 1.0 - profile.swing_fraction)
                        )
                        # 支撑足在世界系缓慢后扫，地面摩擦反推机身向 +X。
                        # 这是爬阶前进的主要来源，不能仅靠摆动腿向前摆。
                        desired[leg_index, 0] -= (
                            STAIR_STANCE_DRIVE_M
                            * float(np.clip(stance_progress, 0.0, 1.0))
                        )

        if self.state == FsmState.CLIMB_PUSH:
            push_progress = float(
                np.clip(
                    (float(data.time) - self.state_started_s) / 1.05,
                    0.0,
                    1.0,
                )
            )
            push_distance = 0.115 * cosine_blend(0.0, 1.0, push_progress)
            desired[:, 0] -= push_distance

        if self.state in (
            FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
        ):
            # 单腿摆动时，摆动腿保持原轨迹；其余三只支撑足整体向支撑
            # 三角形内部平移。足端目标相对机身平移会让地面对腿产生反向
            # 作用，从而把质心推入三足支撑三角形。
            target_shift_x_m, target_shift_y_m = self._crawl_support_shift_target(data)
            alpha = 1.0 - math.exp(-CONTROL_DT_S / CRAWL_SHIFT_FILTER_TAU_S)
            self._support_shift_x_m += alpha * (
                target_shift_x_m - self._support_shift_x_m
            )
            self._support_shift_y_m += alpha * (
                target_shift_y_m - self._support_shift_y_m
            )
            profile = self.patterns[self.state]
            cycle = (float(data.time) - self._gait_start_s) / profile.period_s
            for leg_index in range(4):
                local_phase = (cycle + profile.phase_offsets[leg_index]) % 1.0
                if local_phase >= profile.swing_fraction:
                    desired[leg_index, 0] += self._support_shift_x_m
                    desired[leg_index, 1] += self._support_shift_y_m

        if self.state in (FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH):
            # 姿态闭环只作用于支撑足。摆动足仍严格沿已规划的抬腿轨迹运动，
            # 因而不会被姿态修正拉回地面。目标机身高度/俯仰由当前支撑足
            # 锚点估算：前足上台阶后自然形成一个前高后低的爬阶姿态。
            stance_mask = ~swing_mask
            if np.any(stance_mask):
                stance_z = desired[stance_mask, 2]
                target_base_z = (
                    float(np.mean(stance_z)) + self._body_foot_height_ref_m
                )
                front_mask = stance_mask & np.array((True, True, False, False))
                rear_mask = stance_mask & np.array((False, False, True, True))
                target_pitch = 0.0
                if np.any(front_mask) and np.any(rear_mask):
                    front_z = float(np.mean(desired[front_mask, 2]))
                    rear_z = float(np.mean(desired[rear_mask, 2]))
                    target_pitch = math.atan2(
                        front_z - rear_z, NOMINAL_FRONT_REAR_WHEELBASE_M
                    )
                base_z = float(data.xpos[self.base_body_id, 2])
                base_vz = float(data.qvel[2])
                current_pitch, current_roll = quaternion_to_pitch_roll(data.qpos[3:7])
                pitch_rate = float(data.qvel[4])
                roll_rate = float(data.qvel[3])
                # 世界足端 Z 降低会让腿伸长并把机身顶高，因此高度误差的
                # 符号必须与“期望机身高度”相反；速度项则在机身上升时
                # 减小伸长量，避免高度环和地面反力形成正反馈振荡。
                common_z = (
                    -CRAWL_BODY_Z_GAIN * (target_base_z - base_z)
                    + CRAWL_BODY_Z_DAMPING * base_vz
                )
                pitch_z = (
                    CRAWL_PITCH_GAIN_M_RAD * (target_pitch - current_pitch)
                    - CRAWL_PITCH_DAMPING_M_S_RAD * pitch_rate
                )
                roll_z = (
                    CRAWL_ROLL_GAIN_M_RAD * (0.0 - current_roll)
                    - CRAWL_ROLL_DAMPING_M_S_RAD * roll_rate
                )
                limit = CRAWL_POSTURE_CORRECTION_LIMIT_M
                common_z = float(np.clip(common_z, -limit, limit))
                pitch_z = float(np.clip(pitch_z, -limit, limit))
                roll_z = float(np.clip(roll_z, -limit, limit))
                front_sign = np.array((1.0, 1.0, -1.0, -1.0))
                left_sign = np.array((1.0, -1.0, 1.0, -1.0))
                posture_z = common_z - front_sign * pitch_z - left_sign * roll_z
                desired[stance_mask, 2] += posture_z[stance_mask]
        return desired

    def _flat_trot_joint_targets(self, data: mujoco.MjData) -> np.ndarray:
        """生成已验证的低幅对角 Trot 关节目标。

        参数：
            data：当前 MuJoCo 状态；使用 ``data.time`` 计算连续相位。
        返回：
            shape=(12,)，单位 rad。
        算法：
            对角腿相位为 FL+RR / FR+RL；髋关节仅做 0.02 rad 重心转移，
            大腿正弦振幅 0.18 rad，膝关节额外在摆动相屈曲。该参数来自
            仓库中已经验证可稳定前进的 Trot 基线，先保证接近台阶阶段不摔。
        重要性：A（平地接近和高阶状态切换前的稳定性基础）。
        """

        gait_time_s = max(0.0, float(data.time) - self._gait_start_s)
        leg_phase = 2.0 * math.pi * (
            1.8 * gait_time_s
            + np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
        )
        swing_weight = np.maximum(0.0, -np.sin(leg_phase))
        ramp = float(np.clip(gait_time_s / 0.80, 0.0, 1.0))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        hip_sign = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float64)
        knee_gain = np.array((4.4, 4.4, 11.0, 11.0), dtype=np.float64)
        target = STAND_JOINT_RAD.copy()
        for leg_index in range(4):
            sine = float(np.sin(leg_phase[leg_index]))
            target[3 * leg_index] = 0.02 * hip_sign[leg_index] * sine * ramp
            target[3 * leg_index + 1] = 0.9 + 0.18 * sine * ramp
            target[3 * leg_index + 2] = (
                -1.8
                + 0.14 * sine * ramp
                - knee_gain[leg_index] * 0.045 * swing_weight[leg_index] * ramp
            )
        return np.clip(target, self.joint_low + 0.03, self.joint_high - 0.03)

    def _high_step_trot_joint_targets(self, data: mujoco.MjData) -> np.ndarray:
        """生成接近台阶时的高抬腿 Trot 目标。

        参数：
            data：当前 MuJoCo 状态，使用连续仿真时间生成步态相位。
        返回：
            shape=(12,)，单位 rad，已做关节限位安全裁剪。
        算法：
            沿用已验证稳定的平地 Trot，只在两只前足的摆动相额外屈膝。
            这样 HIGH_STEP 不会为了“准备”而突然刹停；真正的三足低速
            切换留到更近的 CRAWL 状态。额外屈膝对应约 0.08 m 净空增量。
        重要性：A（高位预判阶段保持动态稳定的关键）。
        """

        target = self._flat_trot_joint_targets(data)
        gait_time_s = max(0.0, float(data.time) - self._gait_start_s)
        phase = 2.0 * math.pi * (
            1.8 * gait_time_s
            + np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
        )
        swing_weight = np.maximum(0.0, -np.sin(phase))
        ramp = float(np.clip(gait_time_s / 0.55, 0.0, 1.0))
        ramp = ramp * ramp * (3.0 - 2.0 * ramp)
        target[2] -= 0.20 * swing_weight[0] * ramp
        target[5] -= 0.20 * swing_weight[1] * ramp
        return np.clip(target, self.joint_low + 0.03, self.joint_high - 0.03)

    def _crawl_support_shift_target(
        self,
        data: mujoco.MjData,
    ) -> tuple[float, float]:
        """返回 CRAWL/CLIMB_PUSH 支撑足期望的公共 X/Y 平移量，单位 m。

        参数：
            data：当前 MuJoCo 数据，只读取 ``data.time``。
        返回：
            ``(shift_x_m, shift_y_m)``。正 X 表示支撑足目标向前，反作用力
            使机身向后；正 Y 表示支撑足目标向左侧，反作用力使机身向右。
            摆动腿自身不应用该偏移。
        算法：
            FL/FR 摆动时支撑三角形重心在机身后方，因此支撑足目标向前为
            ``+CRAWL_SUPPORT_SHIFT_X_M``；RL/RR 摆动时符号相反。左腿摆动时
            支撑足目标向 ``+Y``，右腿时向 ``-Y``。``CRAWL_SHIFT_LEAD_PHASE``
            让重心在腿离地前开始转移，之后由一阶低通继续平滑。
        边界：
            尚未进入 CRAWL、步态尚未启动或没有单腿摆动/预判窗口时返回零。
        重要性：A（CRAWL 三足支撑不侧翻的核心约束）。
        """

        if self.state not in (
            FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
        ):
            return 0.0, 0.0
        if float(data.time) < self._gait_start_s:
            return 0.0, 0.0

        profile = self.patterns[self.state]
        cycle = (float(data.time) - self._gait_start_s) / profile.period_s
        lead_phase = float(np.clip(CRAWL_SHIFT_LEAD_PHASE, 0.0, 0.20))
        local_phases = (
            cycle + np.asarray(profile.phase_offsets, dtype=np.float64)
        ) % 1.0

        # 第一优先级必须由“本帧真正离地的腿”决定。若先扫描 lead 窗口，
        # 当上一条腿仍在摆动、下一条腿已进入预移窗口时会选错方向，形成
        # 正反馈侧滑。仅有四足全部着地时才使用最接近摆动的 lead 腿。
        swing_legs = np.flatnonzero(local_phases < profile.swing_fraction)
        if swing_legs.size:
            selected_leg = int(swing_legs[0])
        else:
            lead_legs = np.flatnonzero(local_phases > 1.0 - lead_phase)
            if not lead_legs.size:
                return 0.0, 0.0
            selected_leg = int(lead_legs[np.argmax(local_phases[lead_legs])])

        # 前腿缺失时三足三角形重心在后方；后腿缺失时在前方。支撑足目标
        # 朝三足三角形内部移动，地面反作用力把机身推向补全后的支撑中心。
        shift_x_sign = 1.0 if selected_leg < 2 else -1.0
        shift_y_sign = 1.0 if selected_leg in (0, 2) else -1.0
        return (
            shift_x_sign * CRAWL_SUPPORT_SHIFT_X_M,
            shift_y_sign * CRAWL_SUPPORT_SHIFT_Y_M,
        )

    def _joint_targets(self, data: mujoco.MjData) -> np.ndarray:
        """把世界足端目标转换为 12 个关节目标角。

        返回：
            shape=(12,)，float64，单位 rad，已按关节限位留 0.03 rad 安全边界。
        说明：
            足端世界坐标先转到机身局部系，再做矢状面 IK。髋关节目标保持 0，
            因为四种步态均使用标称左右足距；若未来加入横移，需要扩展髋部 IK。
        重要性：A。
        """

        base_position = np.asarray(
            data.xpos[self.base_body_id], dtype=np.float64
        )
        base_rotation = np.asarray(
            data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        desired_world = self._desired_world_feet(data)
        relative = (desired_world - base_position) @ base_rotation
        target = np.zeros(12, dtype=np.float64)
        # 每条腿独立做三轴 IK：足端 Y/Z 共同决定髋侧摆角，X/Z 决定大腿与
        # 小腿角。不能给四条腿复用同一个髋角，否则 CRAWL 时支撑足会被拉
        # 出原位并迅速侧翻。
        for leg_index in range(4):
            hip, thigh, calf = leg_inverse_kinematics(
                relative[leg_index, 0],
                relative[leg_index, 1],
                relative[leg_index, 2],
                HIP_X_M[leg_index],
                HIP_Y_M[leg_index],
                LEG_THIGH_Y_OFFSET_M[leg_index],
                hip_low_rad=self.joint_low[3 * leg_index],
                hip_high_rad=self.joint_high[3 * leg_index],
            )
            target[3 * leg_index] = hip
            target[3 * leg_index + 1] = thigh
            target[3 * leg_index + 2] = calf
        return self._clip_joint_target(
            target, "DESIRED_FEET_IK", now_s=float(data.time)
        )

    def get_target_joint_rad(self) -> np.ndarray:
        """返回教师 FSM 最近一次计算的 12 维关节目标角，单位 rad。

        返回：
            shape=(12,)、float64 的独立副本；调用方可叠加 PPO 残差而不会
            修改控制器内部目标缓存。
        副作用：
            只读，不推进状态机或物理。
        重要性：A（Teacher-Student 融合接口）。
        """

        return np.asarray(self._target_joint_rad, dtype=np.float64).copy()

    def update_control_state(self, data: mujoco.MjData) -> None:
        """执行一次 50 Hz 感知、状态切换和足端节律更新。

        参数：
            data：已推进到当前控制边界的 MuJoCo 数据。
        副作用：
            更新 LiDAR 缓存、FSM 状态、步态相位、锚点和运行指标；不推进物理。
        重要性：A。
        """

        now_s = float(data.time)
        # 接触年龄必须每个控制周期先更新；感知扫描和事件门控随后共享同一
        # 时间快照，避免同一周期内重复累加。
        self._update_contact_age(data)
        self.last_scan = self.perception.scan(data)
        base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        base_rotation = np.asarray(
            data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        candidate = estimate_step_feature(
            self.last_scan, base, base_rotation
        )
        if candidate.visible:
            self._feature_streak += 1
        else:
            self._feature_streak = 0
        self._feature_visible_now = self._feature_streak >= 2
        if self._feature_visible_now:
            # 连续两帧确认后锁存第一级台阶前缘。近距离时 LiDAR 射线主要
            # 命中踏面，若持续用本帧最靠后的高台点更新，会把前缘误判到
            # 机身下方。单级/宽踏面测试锁定首帧前缘；后续多级楼梯需由
            # 更完整的台阶跟踪器显式切换层级。
            self.last_feature = candidate
            self._last_valid_step_time_s = now_s
            observed_edge_x = float(base[0]) + candidate.nearest_x_m
            if self._edge_world_x_m is None:
                self._edge_world_x_m = observed_edge_x
            else:
                # 允许前缘向 +X 小幅修正（例如首帧只扫到踏面后部），但拒绝
                # 因踏面占据视野而产生的向后跳变；该阈值小于一个 5 cm 格。
                self._edge_world_x_m = min(
                    float(self._edge_world_x_m),
                    observed_edge_x + 0.025,
                )
            self.metrics.min_step_distance_m = min(
                self.metrics.min_step_distance_m,
                candidate.nearest_x_m,
            )
        elif (
            now_s - self._last_valid_step_time_s > 2.0
            and self.state == FsmState.FLAT_TROT
        ):
            # 仅在平地接近阶段允许清除特征。进入 HIGH_STEP 后，LiDAR
            # 可能因为机身逼近而看不到踏面，但已经锁存的几何仍是落足和
            # 登台判定的唯一依据，不能在爬阶途中被清零。
            self.last_feature = StepFeature()

        if not self._initialized:
            if now_s >= self.warmup_end_s:
                self._set_initial_gait_start(data)
            else:
                self._target_joint_rad = STAND_JOINT_RAD.copy()
                self._update_metrics(data)
                return

        feature = self.last_feature
        edge_x = self._estimated_edge_world_x()
        feet_on_step = self._feet_on_platform_mask(data, feature, edge_x)
        front_on_step = bool(np.all(feet_on_step[:2]))
        rear_on_step = bool(np.all(feet_on_step[2:]))
        all_on_step = bool(np.all(feet_on_step))
        self._front_on_step = front_on_step
        self._rear_on_step = rear_on_step

        pitch, roll = quaternion_to_pitch_roll(data.qpos[3:7])
        if (
            float(base[2]) < 0.14
            or abs(pitch) > 0.95
            or abs(roll) > ROLL_FAILURE_LIMIT_RAD
        ):
            self._transition_to(FsmState.FAILED, data, gait_delay_s=0.0)
        elif self.state == FsmState.FLAT_TROT:
            # 触发距离必须相对锁存前缘计算。前足比机身中心前伸约
            # 0.28 m，因此必须在接触立面之前切到准静态轨迹。低台阶
            # 0.04 m 虽然不高，但当前 Trot 只生成原地周期摆动，不能形成
            # 可累计的世界足端推进；实测会卡在台前。这里恢复已验证能把
            # 两只前足真实放上台面的 LOW_STEP -> HIGH_STEP 分支。
            edge_distance_m = (
                float(edge_x) - float(base[0])
                if edge_x is not None
                else math.inf
            )
            near_high = (
                self._feature_visible_now
                and feature.rise_m > 0.10
                and edge_distance_m < 0.52
            )
            near_low = (
                self._feature_visible_now
                and 0.022 < feature.rise_m <= 0.10
                and edge_distance_m < LOW_STEP_TRIGGER_DISTANCE_M
            )
            if near_high or near_low:
                self._transition_to(FsmState.HIGH_STEP, data)
        elif self.state == FsmState.HIGH_STEP:
            # 只有内部事件计划全部执行完，才允许依据实际足端和机体重心
            # 推进状态。否则计划刚开始时旧的单帧 LiDAR 判定可能提前切走，
            # 留下尚未落稳的前腿并造成机身塌陷。
            edge_distance_m = (
                float(edge_x) - float(base[0])
                if edge_x is not None
                else math.inf
            )
            if (
                self._quasi_plan_done
                and front_on_step
                and edge_x is not None
            ):
                self._transition_to(FsmState.CRAWL, data)
        elif self.state == FsmState.CRAWL:
            # CRAWL 计划负责把后足逐条送上台面并把重心继续前移；计划完成
            # 前不提前进入 CLIMB_PUSH，避免两组准静态轨迹互相覆盖。
            if (
                self._quasi_plan_done
                and all_on_step
                and edge_x is not None
                and base[0] > edge_x + 0.035
            ):
                self._transition_to(
                    FsmState.CLIMB_PUSH, data, gait_delay_s=0.0
                )
        elif self.state == FsmState.CLIMB_PUSH:
            # 前压计划完成后再做一次严格空间验收：四足都在台面内且重心
            # 越过前缘才可进入 DONE；任何一项不满足都保持当前状态，不通过
            # 计时器或单帧高度猜测伪造成功。
            if self._quasi_plan_done:
                self._pushed_once = True
                if (
                    all_on_step
                    and edge_x is not None
                    and base[0] > edge_x + 0.035
                ):
                    self._transition_to(
                        FsmState.DONE, data, gait_delay_s=0.0
                    )
                    self.metrics.done = True

        if self.state == FsmState.FLAT_TROT:
            # 平地仍使用连续对角 Trot；只有进入三个爬阶状态后才切换为准静态
            # 事件引擎。这样雷达扫描和移动控制保持原有职责边界。
            self._refresh_gait(data)
            self._target_joint_rad = self._flat_trot_joint_targets(data)
        elif self.state in (
            FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
        ):
            # 准静态状态的成功性依赖支撑腿目标在事件内保持不变，因此绝不能
            # 再调用 _refresh_gait/_joint_targets 逐帧重算四条腿。
            self._target_joint_rad = self._update_quasi_control(data, now_s)
        elif self.state == FsmState.DONE:
            self._target_joint_rad = np.clip(
                STAND_JOINT_RAD,
                self.joint_low + 0.03,
                self.joint_high - 0.03,
            )
        elif self.state == FsmState.FAILED:
            self._target_joint_rad = np.clip(
                np.asarray(data.qpos[self.qpos_adr], dtype=np.float64),
                self.joint_low + 0.03,
                self.joint_high - 0.03,
            )

        # 普通状态切换使用一次全局余弦过渡；准静态状态已由内部事件逐段做
        # 余弦插值并锁定支撑腿，不能再被全局混合覆盖。
        blend_age_s = now_s - self._transition_start_s
        if (
            self.state not in (
                FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
            )
            and 0.0 <= blend_age_s < STATE_BLEND_TIME_S
        ):
            blend_progress = blend_age_s / STATE_BLEND_TIME_S
            self._target_joint_rad = cosine_blend(
                self._transition_start_q,
                self._target_joint_rad,
                blend_progress,
            )
        self._update_metrics(data)

    def _update_metrics(self, data: mujoco.MjData) -> None:
        """更新运行诊断指标；只读 MuJoCo 状态。重要性：B。"""

        base = np.asarray(data.xpos[self.base_body_id], dtype=np.float64)
        foot_bottoms = self._current_foot_bottoms_z(data)
        floor_z = self.last_feature.floor_z_m if self.last_feature.visible else 0.0
        self.metrics.max_base_x_m = max(self.metrics.max_base_x_m, float(base[0]))
        self.metrics.max_base_z_m = max(self.metrics.max_base_z_m, float(base[2]))
        self.metrics.max_foot_clearance_m = max(
            self.metrics.max_foot_clearance_m,
            float(np.max(foot_bottoms - floor_z)),
        )
        if self.state == FsmState.FAILED:
            self.metrics.failed = True


# ---------------------------------------------------------------------------
# 单文件运行入口：PD 输出、实时窗口、终端日志和关键帧截图。
# ---------------------------------------------------------------------------

def write_motor_command(
    controller: RuleFsmController,
    data: mujoco.MjData,
    target_joint_rad: np.ndarray,
) -> None:
    """把 12 维关节目标角转换为 MuJoCo 电机力矩。

    参数：
        controller：已初始化的 FSM 控制器，提供关节地址和力矩范围。
        data：当前物理状态。
        target_joint_rad：shape=(12,)，单位 rad。
    副作用：
        写入 ``data.ctrl``；不修改 qpos/qvel。
    算法：
        ``tau = Kp * (q_target - q) - Kd * qdot + k_ff * qfrc_bias``，再按
        执行器 ctrlrange 裁剪。当前 Go2 电机以力矩模式工作，位置控制必须
        在外层完成；``qfrc_bias`` 只补偿关节重力/科氏力，不会推动自由基座。
    边界：
        前馈比例过大会造成落地瞬间上冲，过小则单腿支撑时保持高度不足。
        修改 ``GRAVITY_FEEDFORWARD_SCALE`` 后应优先观察 HOLD 机身 Z 和足端
        接触时长是否稳定。
    重要性：A（控制器与物理引擎的唯一执行边界）。
    """

    q = np.asarray(data.qpos[controller.qpos_adr], dtype=np.float64)
    qd = np.asarray(data.qvel[controller.dof_adr], dtype=np.float64)
    bias = np.asarray(data.qfrc_bias[controller.dof_adr], dtype=np.float64)
    torque = (
        PD_KP * (target_joint_rad - q)
        - PD_KD * qd
        + GRAVITY_FEEDFORWARD_SCALE * bias
    )
    data.ctrl[:] = np.clip(torque, controller.torque_low, controller.torque_high)


def _status_line(controller: RuleFsmController, data: mujoco.MjData) -> str:
    """生成一行满足验收格式的实时 FSM 状态日志。重要性：B。"""

    edge_world_x_m = controller._estimated_edge_world_x()
    base_x_m = float(data.xpos[controller.base_body_id, 0])
    distance_text = (
        f"{max(0.0, edge_world_x_m - base_x_m):.2f}m"
        if edge_world_x_m is not None
        else "--"
    )
    base_z_m = float(data.xpos[controller.base_body_id, 2])
    pitch_rad, roll_rad = quaternion_to_pitch_roll(data.qpos[3:7])
    event_suffix = (
        f" [事件: {controller._quasi_event_name}]"
        if controller.state in (
            FsmState.HIGH_STEP, FsmState.CRAWL, FsmState.CLIMB_PUSH
        )
        else ""
    )
    return (
        f"[当前状态: {controller.state.value}] "
        f"[距台阶: {distance_text}] "
        f"[机身Z高度: {base_z_m:.2f}m] "
        f"[pitch: {math.degrees(pitch_rad):+.1f}deg] "
        f"[roll: {math.degrees(roll_rad):+.1f}deg]"
        f"{event_suffix}"
    )


def _save_keyframe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    output_path: Path,
) -> None:
    """用 MuJoCo 离屏渲染保存一张 3D 关键帧 PNG。

    参数：
        model/data：当前场景和物理状态。
        output_path：目标 PNG 路径；父目录不存在时自动创建。
    可能异常：
        当前机器没有可用 OpenGL 后端时，MuJoCo Renderer 会抛 RuntimeError。
        演示主流程会捕获该异常并继续，而不是让仿真失败。
    重要性：B（交付截图，不参与控制）。
    """

    import cv2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    step_x = _scene_step_center_x(model)
    camera.lookat[:] = (step_x - 0.55, 0.468, 0.28)
    camera.distance = 2.5
    camera.azimuth = 135.0
    camera.elevation = -18.0
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        renderer.update_scene(data, camera=camera)
        rgb = renderer.render()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr)
        if not ok:
            raise RuntimeError("PNG 编码失败")
        # Windows 中文路径下 cv2.imwrite 可能静默返回 False；显式写 bytes
        # 并检查文件大小，保证日志中的截图路径真实对应产物。
        output_path.write_bytes(encoded.tobytes())
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"截图文件未成功写入：{output_path}")
    finally:
        renderer.close()


def _scene_step_center_x(model: mujoco.MjModel) -> float:
    """返回已知测试台阶中心的 X，仅用于截图相机取景。重要性：C。"""

    if model.ngeom <= 0:
        return -0.9
    # 不参与控制的展示辅助：优先找命名明确的宽台阶；完整钢梯则用其第一级
    # 附近的固定取景中心。即使找不到，也回退到出生点前方。
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wide_step_01")
    if geom_id >= 0:
        return float(model.geom_pos[geom_id, 0])
    first_step = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "step_01")
    if first_step >= 0:
        return float(model.geom_pos[first_step, 0])
    return -0.9


def run_demo(args: argparse.Namespace) -> int:
    """运行一次完整 FSM 演示并返回进程退出码。

    参数：
        args：命令行参数，包含场景、时长、是否无窗口和截图目录。
    返回：
        0 表示运行结束且未检测到跌倒；1 表示 FSM 报跌倒或超时未完成。
    副作用：
        推进 MuJoCo 物理、写终端日志、可选打开窗口和写 PNG。
    时序：
        物理步长沿用 XML；控制目标严格按 50 Hz 更新，中间用同一目标角
        推进多个物理子步，使控制器频率不随模型 timestep 改变。
    重要性：A。
    """

    model, keyframe_id, scene_path = load_scene(args.scene)
    data = mujoco.MjData(model)
    controller = RuleFsmController(model, keyframe_id, args.scene)
    controller.reset(data)

    physics_dt = float(model.opt.timestep)
    if physics_dt <= 0.0:
        raise ValueError("模型 timestep 必须为正数")
    substeps = max(1, int(round(CONTROL_DT_S / physics_dt)))
    screenshot_dir = Path(args.screenshot_dir).resolve() if args.screenshot_dir else None
    captured_states: set[str] = set()
    next_log_s = 0.0
    last_state = controller.state
    previous_control_target = controller._target_joint_rad.copy()
    start_wall_s = time.perf_counter()
    simulated_s = 0.0
    done_hold = False

    print(f"[场景] {scene_path}", flush=True)
    print(
        f"[控制] 50 Hz 目标更新，{substeps} 个物理子步/控制周期，"
        f"物理步长 {physics_dt:.3f}s",
        flush=True,
    )
    print(
        f"[相机/雷达] LiDAR site="
        f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, controller.perception.site_id)}"
        f"，射线={controller.perception.ray_count}",
        flush=True,
    )

    viewer_context = (
        contextlib.nullcontext(None)
        if args.headless
        else mujoco.viewer.launch_passive(model, data)
    )
    try:
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = (_scene_step_center_x(model) - 0.55, 0.468, 0.30)
                viewer.cam.distance = 2.8
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -18.0
                viewer.sync()

            # 主循环在一次墙钟循环内完成“控制目标更新 + N 个物理子步”。
            # 视觉/雷达只在 50 Hz 控制边界扫描，不进入物理子步。
            while viewer is None or viewer.is_running():
                loop_start_s = time.perf_counter()
                if simulated_s >= args.duration and not done_hold:
                    print("[结果] 达到最大演示时长，停止推进。", flush=True)
                    break

                controller.update_control_state(data)
                target = controller._target_joint_rad.copy()
                for substep_index in range(substeps):
                    # 50 Hz 控制目标内部再做一次余弦插值，使 500 Hz 物理
                    # 子步看到连续的目标角，降低状态切换时的瞬时 PD 冲击。
                    progress = (substep_index + 1) / float(substeps)
                    substep_target = cosine_blend(
                        previous_control_target, target, progress
                    )
                    write_motor_command(controller, data, substep_target)
                    mujoco.mj_step(model, data)
                previous_control_target = target
                simulated_s = float(data.time)

                if viewer is not None:
                    viewer.sync()

                if controller.state != last_state:
                    last_state = controller.state
                    print(_status_line(controller, data), flush=True)
                    print(f"[状态切换] -> {controller.state.value}", flush=True)
                    if screenshot_dir is not None and controller.state.value not in captured_states:
                        captured_states.add(controller.state.value)
                        shot = screenshot_dir / f"{args.scene}_{controller.state.value.lower()}.png"
                        try:
                            _save_keyframe(model, data, shot)
                            print(f"[截图] {shot}", flush=True)
                        except Exception as exc:  # 截图失败不能中断物理仿真。
                            print(f"[截图跳过] {exc}", flush=True)

                if simulated_s >= next_log_s:
                    print(_status_line(controller, data), flush=True)
                    next_log_s = simulated_s + 0.50

                if controller.state == FsmState.FAILED:
                    print("[结果] 控制器检测到跌倒，演示结束。", flush=True)
                    break
                if controller.state == FsmState.DONE:
                    print(
                        "[结果] 四足已登顶；保持最终姿态，可直接观察 3D 窗口。",
                        flush=True,
                    )
                    if args.headless:
                        break
                    # GUI 模式继续推进物理和站立 PD，直到用户关闭窗口；
                    # 不冻结 mj_step，也不因固定秒数自动退出。
                    done_hold = True

                # 1 倍速是默认行为；headless 默认尽量快，便于自动化验证。
                if not args.headless and not args.fast:
                    target_wall_s = simulated_s / max(args.speed, 0.1)
                    sleep_s = target_wall_s - (time.perf_counter() - start_wall_s)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)

            if screenshot_dir is not None:
                final_shot = screenshot_dir / f"{args.scene}_final.png"
                try:
                    _save_keyframe(model, data, final_shot)
                    print(f"[截图] {final_shot}", flush=True)
                except Exception as exc:
                    print(f"[截图跳过] {exc}", flush=True)

            print(
                f"[指标] 最大机身X={controller.metrics.max_base_x_m:.3f}m，"
                f"最大机身Z={controller.metrics.max_base_z_m:.3f}m，"
                f"最大足端净空={controller.metrics.max_foot_clearance_m:.3f}m，"
                f"最近台阶距离={controller.metrics.min_step_distance_m:.3f}m",
                flush=True,
            )
            print(
                f"[指标] CRAWL={controller.metrics.crawl_entries}次，"
                f"CLIMB_PUSH={controller.metrics.push_entries}次，"
                f"状态链={' -> '.join(controller.metrics.state_history)}",
                flush=True,
            )
    except KeyboardInterrupt:
        print("[退出] 用户中断，正常关闭窗口。", flush=True)
        return 0

    return 0 if controller.state == FsmState.DONE else 1


def build_arg_parser() -> argparse.ArgumentParser:
    """构造此单文件演示的命令行参数。重要性：C。"""

    parser = argparse.ArgumentParser(description="Go2 纯规则 FSM 上台阶演示")
    parser.add_argument(
        "--scene",
        choices=tuple(SCENE_CONFIGS),
        default="low",
        help="low=0.04m 单级低台阶，full=完整三层钢梯",
    )
    parser.add_argument("--headless", action="store_true", help="不打开 3D 窗口")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="最大仿真时长，单位 s"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="GUI 播放倍率，默认 1.0 倍速"
    )
    parser.add_argument(
        "--fast", action="store_true", help="GUI 模式也尽可能快地推进物理"
    )
    parser.add_argument(
        "--screenshot-dir",
        default=None,
        help="可选关键帧截图目录；默认不生成截图",
    )
    return parser


def main() -> int:
    """解析参数并运行演示；异常时输出可排查的错误信息。重要性：A。"""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_arg_parser().parse_args()
    return run_demo(args)


if __name__ == "__main__":
    raise SystemExit(main())









