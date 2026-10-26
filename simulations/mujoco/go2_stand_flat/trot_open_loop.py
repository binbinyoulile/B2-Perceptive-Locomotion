"""在 MuJoCo 中实时演示 Go2 开环抬腿 Trot，并测试单级低台阶跨越。

文件目的：
    在平地 PD 站立基线上叠加带显式摆动相抬腿的 12 关节正弦目标。机器人
    可在平地上持续前进，也可切换到 0.08 m 单级台阶场景，从台阶前起步并
    记录是否撞上立面、是否真实登阶。整个过程保留实时 3D 窗口。
整体输入：
    scene.xml 或 ../go2_stairs_static/scene_single_step.xml；官方 Go2 MJCF、
    对应关键帧，以及 stand_pd.py 中的关节顺序和 PD 增益。
整体输出：
    平地默认写入 trot_metrics.csv、trot_xy.png、trot_z.png；
    台阶模式写入 trot_single_step_metrics.csv、
    trot_single_step_xy.png、trot_single_step_z.png。
    终端同时打印前进距离、机身高度、倾角、各足摆动最大净空、支撑状态以及
    台阶立面/顶面接触结论。
依赖与版本要求：
    Python 3.13、MuJoCo 3.16、NumPy 2.x、matplotlib 3.11。
    实时窗口需要 Windows 图形桌面和可用 OpenGL；不依赖视频编码库。
运行方式：
    平地持续运行，手动关闭窗口结束：
    D:/Python313/python.exe D:/桌面/机器狗/simulations/mujoco/go2_stand_flat/trot_open_loop.py
    平地固定运行 10 s：
    D:/Python313/python.exe .../trot_open_loop.py --scene flat --duration 10
    单级台阶固定运行 20 s：
    D:/Python313/python.exe .../trot_open_loop.py --scene single-step --duration 20
    无窗口快速验证（按物理速度运行，不按墙钟等待）：
    D:/Python313/python.exe .../trot_open_loop.py --scene flat --duration 10 --headless
关键假设：
    1. 关节和 motor 顺序与 JOINT_NAMES 完全一致，启动时用断言校验。
    2. 官方左右腿局部 X 轴同向，因此不对左/右腿角度做额外镜像。
    3. 摆动相定义为 sin(phase) < 0；通过负向 calf 偏置让足端抬高。
    4. 平地关键帧为 stand_home；台阶关键帧先用于取得正常站立姿态，
       随后在运行时把 base 移到台阶前，避免从台阶顶面开始测试。
    5. 0.08 m 台阶的前立面位于世界 X=-1.15 m，狗朝 +X 前进。
    6. 控制器仍为开环：没有机身姿态、速度、落脚点或接触反馈。
已知限制：
    前后腿对相同膝关节目标的跟踪误差不同，因此抬腿增益分别取不同值，
    这是当前 PD 与官方模型下的经验基线，不是通用参数。实际足端净空受
    接触、俯仰和负载影响，不能用指令抬腿高度直接代替测量值。脚本不会在
    失败后自动调频、增大振幅或重试，以免掩盖开环控制的真实问题。
重要性等级：A（后续步态闭环、地形适应和强化学习对照的实时入口）。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# 用 Agg 后端生成复盘图，避免窗口关闭后依赖交互式绘图窗口。
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from stand_pd import (
    JOINT_NAMES,
    KD,
    KP,
    LEG_NAMES,
    Q_TARGET as Q_STAND,
    configure_chinese_plot_font,
    quaternion_to_euler,
)


HERE = Path(__file__).resolve().parent
FLAT_XML_PATH = HERE / "scene.xml"
SINGLE_STEP_XML_PATH = HERE.parent / "go2_stairs_static" / "scene_single_step.xml"
FLAT_CSV_PATH = HERE / "trot_metrics.csv"
FLAT_XY_FIGURE_PATH = HERE / "trot_xy.png"
FLAT_Z_FIGURE_PATH = HERE / "trot_z.png"
STEP_CSV_PATH = HERE / "trot_single_step_metrics.csv"
STEP_XY_FIGURE_PATH = HERE / "trot_single_step_xy.png"
STEP_Z_FIGURE_PATH = HERE / "trot_single_step_z.png"

# 完整 Trot 周期频率。提高频率会增加单位时间步数，但也增加接触冲击。
TROT_FREQUENCY_HZ = 2.0
# 髋外摆只用于轻微重心转移；过大会让足端横向漂移和侧翻。
HIP_ABDUCTION_AMPLITUDE_RAD = 0.02
# 大腿正弦振幅决定前后迈步幅度；0.18 rad 为当前开环基线。
THIGH_AMPLITUDE_RAD = 0.18
# 膝关节基础正弦振幅；显式抬腿量在其上再叠加一个只作用于摆动相的负偏置。
KNEE_AMPLITUDE_RAD = 0.14
KNEE_PHASE_OFFSET_RAD = 0.0
# 对角腿同相：FL+RR=0，FR+RL=0.5 周期，两组相差 180 度。
PAIR_PHASE_CYCLES = np.array((0.0, 0.5, 0.5, 0.0), dtype=np.float64)
# 左右腿局部 X 轴同向，用相反符号形成左右对称的小幅髋外摆。
HIP_SIGNS = np.array((1.0, -1.0, 1.0, -1.0), dtype=np.float64)

# 期望足端抬腿高度。3~5 cm 足以离开平地，但低于 8 cm 台阶立面，因此
# 单级台阶测试预期可能被立面挡住，不能把它当作爬楼成功参数。
SWING_HEIGHT_M = 0.045
# 足端净空对 calf 关节偏置的经验换算斜率，顺序 FL/FR/RL/RR，单位 rad/m。
# 前腿跟踪误差较大，后腿需要更大的关节偏置才能得到相近实际净空。
# 增大该值会抬脚更高，但可能造成过度屈膝、髋部下沉和落地冲击。
SWING_KNEE_GAIN_RAD_PER_M = np.array((4.4, 4.4, 11.0, 11.0), dtype=np.float64)
# 从站立平滑过渡到 Trot，避免第一个周期突然施加满幅动作。
GAIT_RAMP_S = 0.8
WARMUP_S = 0.5

SAMPLE_INTERVAL_STEPS = 5  # 0.002 s * 5 = 100 Hz
VIEWER_SYNC_HZ = 60.0
CONSOLE_PRINT_INTERVAL_S = 2.0
FALL_HEIGHT_M = 0.15
FALL_TILT_DEG = 60.0
CSV_FLUSH_INTERVAL_SAMPLES = 100
# 足端净空至少达到该值，才在总结中算作「真正离地」。
CLEARANCE_TARGET_M = 0.03
# 单级台阶边界识别与起步距离。起步点放在前立面前方 0.55 m。
STEP_GEOM_NAME = "curriculum_step_01"
STEP_START_GAP_M = 0.55
STEP_INITIAL_BASE_HEIGHT_M = 0.30
# 顶面接触腿数至少为 2 且 base 比台阶顶面高 0.20 m，才判为已登阶。
STEP_CLIMB_BASE_CLEARANCE_M = 0.20
STEP_CLIMB_MIN_TOP_CONTACTS = 2

CSV_FIELDS = (
    "time_s",
    "trot_time_s",
    "phase_cycle",
    "base_x_m",
    "base_y_m",
    "base_z_m",
    "forward_distance_m",
    "lateral_offset_m",
    "roll_deg",
    "pitch_deg",
    "yaw_deg",
    "tilt_deg",
    "support_count",
    "step_top_contact_count",
    "step_face_contact",
    "fallen",
) + tuple(f"contact_{name}" for name in LEG_NAMES) + tuple(
    f"floor_contact_{name}" for name in LEG_NAMES
) + tuple(f"step_contact_{name}" for name in LEG_NAMES) + tuple(
    f"swing_{name}" for name in LEG_NAMES
) + tuple(f"clearance_{name}_m" for name in LEG_NAMES) + tuple(
    f"target_{name}_rad" for name in JOINT_NAMES
) + tuple(f"q_{name}_rad" for name in JOINT_NAMES)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass(frozen=True)
class StepGeometry:
    """描述单级台阶的轴对齐包围盒及其前进方向。

    参数：
        geom_name：MuJoCo 中台阶 geom 的名称。
        center_x_m、center_y_m、center_z_m：台阶中心的世界坐标，单位 m。
        half_x_m、half_y_m、half_z_m：三个方向的半尺寸，单位 m。
        front_x_m、rear_x_m：狗沿 +X 上楼时先遇到的立面与后边界，单位 m。
        left_y_m、right_y_m：台阶左右边界，单位 m。
        top_z_m：顶面世界高度，单位 m。
    坐标系：
        MuJoCo 世界坐标；当前场景无旋转、无倾斜，因此可用轴对齐盒简化判断。
    重要性等级：B（接触分类、净空计算和结果判定都依赖该几何）。
    """

    geom_name: str
    center_x_m: float
    center_y_m: float
    center_z_m: float
    half_x_m: float
    half_y_m: float
    half_z_m: float
    front_x_m: float
    rear_x_m: float
    left_y_m: float
    right_y_m: float
    top_z_m: float

    def contains_xy(self, x_m: float, y_m: float) -> bool:
        """判断世界 X-Y 点是否落在台阶顶面的水平投影内。

        参数：
            x_m、y_m：待判断点的世界坐标，单位 m。
        返回：
            布尔值；边界点按包含处理。
        重要性等级：C。
        """
        return (
            self.front_x_m <= x_m <= self.rear_x_m
            and self.left_y_m <= y_m <= self.right_y_m
        )


@dataclass(frozen=True)
class RunConfig:
    """保存一次平地或单级台阶运行所需的场景与输出配置。

    参数：
        scene_name：flat 或 single-step，用于终端输出和结果分类。
        xml_path：场景 XML 绝对路径。
        keyframe_name：初始姿态关键帧名称。
        csv_path、xy_figure_path、z_figure_path：本次输出文件绝对路径。
    生命周期：
        由 main 根据命令行参数创建，运行期间只读。
    重要性等级：B。
    """

    scene_name: str
    xml_path: Path
    keyframe_name: str
    csv_path: Path
    xy_figure_path: Path
    z_figure_path: Path

    @property
    def is_step_scene(self) -> bool:
        """返回当前是否为单级台阶测试场景。重要性等级：C。"""
        return self.scene_name == "single-step"


@dataclass(frozen=True)
class ContactSnapshot:
    """保存四足与地面/台阶的一次接触分类结果。

    字段形状均为 4，顺序 FL/FR/RL/RR：
        foot_contacts：足端与地面或台阶任一表面接触；
        floor_contacts：足端只与 floor 接触；
        step_contacts：足端与台阶任一表面接触；
        step_top_contacts：足端与台阶水平顶面接触；
        step_face_contacts：足端与台阶竖直立面接触。
    生命周期：
        每次 100 Hz 采样新建一次，供 CSV 和统计使用。
    重要性等级：B（支撑计数和台阶卡阻判断依赖该分类）。
    """

    foot_contacts: tuple[bool, ...]
    floor_contacts: tuple[bool, ...]
    step_contacts: tuple[bool, ...]
    step_top_contacts: tuple[bool, ...]
    step_face_contacts: tuple[bool, ...]

    @property
    def support_count(self) -> int:
        """返回当前支撑足数量，范围 0..4。重要性等级：C。"""
        return sum(self.foot_contacts)

    @property
    def step_top_count(self) -> int:
        """返回当前接触台阶顶面的足端数量，范围 0..4。重要性等级：C。"""
        return sum(self.step_top_contacts)

    @property
    def has_step_face_contact(self) -> bool:
        """返回任一足端是否接触台阶立面。重要性等级：C。"""
        return any(self.step_face_contacts)


@dataclass
class RunStats:
    """汇总一次实时仿真中用于终端报告和画图的指标。

    参数：
        initial_x_m、initial_y_m：记录起时机身在世界坐标系的位置，单位 m。
        step：单级台阶几何；平地模式为 None。
    关键字段：
        swing_peak_clearance_m：顺序 FL/FR/RL/RR 的摆动相最大足底净空；
        contact_counts：足端接触地面或台阶的总采样次数；
        step_face_contact_seen：是否曾撞击台阶立面；
        climbed_step：是否同时满足顶面接触和机身抬升条件。
    生命周期：
        每次运行创建一次，在 update 中持续更新，仿真结束后只读。
    重要性等级：B（决定终端总结和复盘图标注）。
    """

    initial_x_m: float
    initial_y_m: float
    step: StepGeometry | None = None
    first_time_s: float | None = None
    last_time_s: float = 0.0
    final_x_m: float = 0.0
    final_y_m: float = 0.0
    final_z_m: float = 0.0
    max_forward_m: float = 0.0
    max_base_x_m: float = float("-inf")
    max_abs_lateral_m: float = 0.0
    min_z_m: float = float("inf")
    max_z_m: float = float("-inf")
    max_abs_tilt_deg: float = 0.0
    support_sum: float = 0.0
    min_support: int = 4
    final_support: int = 0
    sample_count: int = 0
    contact_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(LEG_NAMES), dtype=np.int64)
    )
    swing_peak_clearance_m: np.ndarray = field(
        default_factory=lambda: np.zeros(len(LEG_NAMES), dtype=np.float64)
    )
    swing_sample_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(LEG_NAMES), dtype=np.int64)
    )
    max_step_top_contact_count: int = 0
    step_face_contact_samples: int = 0
    step_face_contact_seen: bool = False
    step_top_contact_seen: bool = False
    climbed_step: bool = False
    fallen: bool = False
    first_fallen_time_s: float | None = None

    def update(
        self,
        row: dict[str, Any],
        contacts: ContactSnapshot,
        clearances_m: np.ndarray,
        swing_mask: np.ndarray,
        tilt_deg: float,
        fallen: bool,
    ) -> None:
        """用当前 100 Hz 采样更新轨迹、抬腿和台阶接触统计。

        参数：
            row：sample_to_row 返回的 CSV 字典，包含 time_s、base_* 等字段。
            contacts：四足地面/台阶接触分类结果。
            clearances_m：四足足底相对正下方地形的高度，shape=(4,)，单位 m；
                支撑相可能因软接触略为负值。
            swing_mask：四足当前是否在摆动相，shape=(4,)，dtype 可转 bool。
            tilt_deg：机身倾角，单位 deg。
            fallen：当前是否满足高度或倾角跌倒判据。
        返回：
            无。
        可能异常：
            数组 shape 不是 (4,) 时抛出 ValueError。
        副作用：
            原地更新统计字段；不修改 MuJoCo 状态。
        重要性等级：B。
        """
        clearances = np.asarray(clearances_m, dtype=np.float64)
        swing = np.asarray(swing_mask, dtype=bool)
        if clearances.shape != (len(LEG_NAMES),):
            raise ValueError(f"clearances_m 应为 (4,)，实际 {clearances.shape}")
        if swing.shape != (len(LEG_NAMES),):
            raise ValueError(f"swing_mask 应为 (4,)，实际 {swing.shape}")

        time_s = float(row["time_s"])
        x_m = float(row["base_x_m"])
        y_m = float(row["base_y_m"])
        z_m = float(row["base_z_m"])
        if self.first_time_s is None:
            self.first_time_s = time_s

        self.last_time_s = time_s
        self.final_x_m = x_m
        self.final_y_m = y_m
        self.final_z_m = z_m
        self.max_forward_m = max(self.max_forward_m, x_m - self.initial_x_m)
        self.max_base_x_m = max(self.max_base_x_m, x_m)
        self.max_abs_lateral_m = max(self.max_abs_lateral_m, abs(y_m - self.initial_y_m))
        self.min_z_m = min(self.min_z_m, z_m)
        self.max_z_m = max(self.max_z_m, z_m)
        self.max_abs_tilt_deg = max(self.max_abs_tilt_deg, abs(float(tilt_deg)))
        self.support_sum += float(contacts.support_count)
        self.min_support = min(self.min_support, contacts.support_count)
        self.final_support = contacts.support_count
        self.sample_count += 1

        for index, is_contact in enumerate(contacts.foot_contacts):
            if is_contact:
                self.contact_counts[index] += 1
            if swing[index]:
                self.swing_sample_counts[index] += 1
                self.swing_peak_clearance_m[index] = max(
                    self.swing_peak_clearance_m[index], clearances[index]
                )

        self.max_step_top_contact_count = max(
            self.max_step_top_contact_count, contacts.step_top_count
        )
        if contacts.has_step_face_contact:
            self.step_face_contact_samples += 1
            self.step_face_contact_seen = True
        if contacts.step_top_count > 0:
            self.step_top_contact_seen = True
        if (
            self.step is not None
            and contacts.step_top_count >= STEP_CLIMB_MIN_TOP_CONTACTS
            and z_m >= self.step.top_z_m + STEP_CLIMB_BASE_CLEARANCE_M
        ):
            self.climbed_step = True

        if fallen:
            self.fallen = True
            if self.first_fallen_time_s is None:
                self.first_fallen_time_s = time_s


def load_model(config: RunConfig) -> mujoco.MjModel:
    """加载指定场景并返回 MuJoCo 模型。

    参数：
        config：包含场景 XML 路径的运行配置。
    返回：
        编译后的 MjModel。
    可能异常：
        XML 或资源缺失、模型语法错误时向上抛出。
    副作用：
        临时切换进程工作目录，兼容 Windows 下含中文路径和相对 mesh 资源；
        函数结束前恢复原目录。
    重要性等级：A（模型加载失败会阻止所有仿真）。
    """
    previous_cwd = Path.cwd()
    try:
        os.chdir(HERE)
        relative_xml = os.path.relpath(config.xml_path, HERE)
        return mujoco.MjModel.from_xml_path(relative_xml)
    finally:
        os.chdir(previous_cwd)


def validate_joint_mapping(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """校验 12 个关节与 actuator 的映射并返回 qpos/qvel 地址。

    参数：
        model：已编译的 Go2 场景。
    返回：
        qpos_adr、dof_adr，shape 均为 (12,)，顺序与 JOINT_NAMES 一致。
    可能异常：
        关节缺失、actuator 数量不为 12 或顺序不一致时抛出 AssertionError。
    副作用：
        无。
    重要性等级：A（顺序错误会把动作发给错误关节，直接导致摔倒）。
    """
    joint_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES],
        dtype=np.int32,
    )
    assert np.all(joint_ids >= 0), f"缺少关节：{JOINT_NAMES}"
    qpos_adr = np.array(
        [model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32
    )
    dof_adr = np.array(
        [model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32
    )
    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    assert model.nu == len(JOINT_NAMES) == 12, "Go2 必须正好有 12 个 motor"
    assert np.array_equal(actuator_joint_ids, joint_ids), (
        "actuator 顺序与 JOINT_NAMES 不一致，不能直接写 data.ctrl"
    )
    return qpos_adr, dof_adr


def read_step_geometry(model: mujoco.MjModel, data: mujoco.MjData) -> StepGeometry:
    """从 MuJoCo 台阶 geom 读取轴对齐包围盒。

    参数：
        model：已编译单级台阶场景。
        data：已执行 mj_forward 的状态，用于读取 geom 的世界位置。
    返回：
        StepGeometry；世界坐标单位为 m。
    可能异常：
        台阶 geom 缺失或尺寸不合法时抛出 RuntimeError。
    副作用：
        无。
    假设：
        台阶 body 当前无旋转；若以后倾斜台阶，应改用 MuJoCo 碰撞距离而非
        该轴对齐盒进行地形高度判断。
    重要性等级：A（后续接触分类和净空测量依赖它）。
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, STEP_GEOM_NAME)
    if geom_id < 0:
        raise RuntimeError(f"单级台阶场景缺少 geom: {STEP_GEOM_NAME}")
    center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    half = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if np.any(half <= 0.0):
        raise RuntimeError(f"台阶 geom 尺寸非法：{half}")
    return StepGeometry(
        geom_name=STEP_GEOM_NAME,
        center_x_m=float(center[0]),
        center_y_m=float(center[1]),
        center_z_m=float(center[2]),
        half_x_m=float(half[0]),
        half_y_m=float(half[1]),
        half_z_m=float(half[2]),
        front_x_m=float(center[0] - half[0]),
        rear_x_m=float(center[0] + half[0]),
        left_y_m=float(center[1] - half[1]),
        right_y_m=float(center[1] + half[1]),
        top_z_m=float(center[2] + half[2]),
    )


def configure_initial_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: RunConfig,
) -> StepGeometry | None:
    """复位关键帧，并在台阶模式中把机器人移到台阶前。

    参数：
        model：已编译场景。
        data：待初始化的 MjData。
        config：运行配置，决定关键帧及是否重定位。
    返回：
        台阶模式返回 StepGeometry；平地模式返回 None。
    可能异常：
        关键帧或台阶 geom 缺失时抛出 RuntimeError。
    副作用：
        清空 qvel 并覆盖 data.qpos、data.ctrl 和正向运动学缓存。
        ！这里不修改 XML，也不覆盖关卡场景文件。
    重要性等级：A（若沿用 single_step_home，四足会直接站在台面上）。
    """
    key_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, config.keyframe_name
    )
    if key_id < 0:
        raise RuntimeError(f"场景缺少关键帧: {config.keyframe_name}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)

    step = None
    if config.is_step_scene:
        step = read_step_geometry(model, data)
        data.qpos[0] = step.front_x_m - STEP_START_GAP_M
        data.qpos[1] = step.center_y_m
        data.qpos[2] = STEP_INITIAL_BASE_HEIGHT_M
        data.qpos[3:7] = np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        data.qvel[:] = 0.0
        data.ctrl[:] = Q_STAND
        mujoco.mj_forward(model, data)
    return step


def gait_phase_and_swing(trot_time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """计算四条腿的 Trot 相位和摆动相权重。

    参数：
        trot_time_s：从 Trot 开始计时的时间，单位 s，必须非负且有限。
    返回：
        leg_phase：shape=(4,)，单位 rad；
        swing_weight：shape=(4,)，范围 0..1。定义 -sin(phase)>0 为摆动相，
        权重随正弦连续变化，避免膝关节阶跃。
    坐标系/算法：
        phase_i = 2*pi*(f*t + offset_i)，offset 为 FL/FR/RL/RR 的周期相位。
    可能异常：
        时间非法时抛出 ValueError。
    重要性等级：A（抬腿与对角步态节奏都在此定义）。
    """
    if not np.isfinite(trot_time_s) or trot_time_s < 0.0:
        raise ValueError(f"trot_time_s 必须是非负有限值，实际 {trot_time_s}")
    leg_phase = 2.0 * np.pi * (
        TROT_FREQUENCY_HZ * trot_time_s + PAIR_PHASE_CYCLES
    )
    swing_weight = np.maximum(0.0, -np.sin(leg_phase))
    return leg_phase, swing_weight


def smoothstep01(value: float) -> float:
    """把输入裁剪到 0..1 后返回平滑的 3x^2-2x^3 过渡值。

    参数：
        value：无量纲比例。
    返回：
        0..1 的平滑值；用于避免步态启动时目标角速度突变。
    重要性等级：C。
    """
    clipped = float(np.clip(value, 0.0, 1.0))
    return clipped * clipped * (3.0 - 2.0 * clipped)


def generate_joint_targets(
    trot_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成指定 Trot 时刻的 12 关节目标和摆动相掩码。

    参数：
        trot_time_s：从步态开始计时的时间，单位 s，必须非负有限。
    返回：
        joint_target_rad：shape=(12,)，关节顺序为每条腿 hip/thigh/calf；
        swing_mask：shape=(4,) 布尔数组，True 表示明确抬腿的摆动相；
        leg_phase：shape=(4,)，单位 rad。
    算法：
        ramp = smoothstep(t / GAIT_RAMP_S)；
        hip = q0_hip + 0.02*sign*sin(phase)*ramp；
        thigh = 0.9 + 0.18*sin(phase)*ramp；
        calf = -1.8 + 0.14*sin(phase)*ramp
               - gain*SWING_HEIGHT_M*max(0,-sin(phase))*ramp。
        calf 更负表示进一步屈膝，从而把足端抬高。
    可能异常：
        输入或结果非有限时抛出 ValueError。
    副作用：
        无。
    重要性等级：A（所有关节动作和显式抬腿都由该函数生成）。
    """
    leg_phase, swing_weight = gait_phase_and_swing(trot_time_s)
    ramp = smoothstep01(trot_time_s / GAIT_RAMP_S)
    target = np.asarray(Q_STAND, dtype=np.float64).copy()
    for leg_index in range(len(LEG_NAMES)):
        sine = float(np.sin(leg_phase[leg_index]))
        hip_offset = HIP_ABDUCTION_AMPLITUDE_RAD * HIP_SIGNS[leg_index] * sine * ramp
        thigh_offset = THIGH_AMPLITUDE_RAD * sine * ramp
        knee_offset = KNEE_AMPLITUDE_RAD * float(
            np.sin(leg_phase[leg_index] + KNEE_PHASE_OFFSET_RAD)
        ) * ramp
        swing_knee_offset = (
            SWING_KNEE_GAIN_RAD_PER_M[leg_index]
            * SWING_HEIGHT_M
            * swing_weight[leg_index]
            * ramp
        )
        target[3 * leg_index] = Q_STAND[3 * leg_index] + hip_offset
        target[3 * leg_index + 1] = Q_STAND[3 * leg_index + 1] + thigh_offset
        target[3 * leg_index + 2] = (
            Q_STAND[3 * leg_index + 2] + knee_offset - swing_knee_offset
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("关节目标包含 NaN 或 Inf")
    swing_mask = swing_weight > 1e-12
    return target, swing_mask, leg_phase


def target_at_time(
    sim_time_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回给定仿真时刻的关节目标、摆动掩码和腿相位。

    参数：
        sim_time_s：MuJoCo 仿真时间，单位 s。
    返回：
        前 WARMUP_S 返回站立目标、全 False 摆动掩码和零相位；之后返回
        generate_joint_targets 的结果。
    边界：
        负时间按 0 处理；WARMUP_S 当刻开始步态，但 ramp 从 0 平滑增加。
    重要性等级：A（预热阶段必须避免突然施加周期动作）。
    """
    if sim_time_s < WARMUP_S:
        return (
            np.asarray(Q_STAND, dtype=np.float64).copy(),
            np.zeros(len(LEG_NAMES), dtype=bool),
            np.zeros(len(LEG_NAMES), dtype=np.float64),
        )
    return generate_joint_targets(max(0.0, float(sim_time_s) - WARMUP_S))


def analyze_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: StepGeometry | None,
) -> ContactSnapshot:
    """按足端分别识别地面和单级台阶的接触类型。

    参数：
        model：已编译场景。
        data：当前状态，应在 mj_step 或 mj_forward 后读取。
        step：单级台阶几何；平地场景为 None。
    返回：
        ContactSnapshot。顺序均为 FL/FR/RL/RR。
    接触法向：
        contact.frame 前三个分量是世界坐标系接触法向。台阶法向的 |x|>0.8
        视为竖直立面接触，|z|>0.8 视为水平顶面接触；不用法向正负号，因为
        MuJoCo 会按 geom1/geom2 顺序翻转法向方向。
    可能异常：
        不抛出；缺失 geom 名称时对应接触被忽略。
    副作用：
        无。
    重要性等级：A（台阶卡阻和登阶判定依赖该分类）。
    """
    floor_contacts = np.zeros(len(LEG_NAMES), dtype=bool)
    step_contacts = np.zeros(len(LEG_NAMES), dtype=bool)
    step_top_contacts = np.zeros(len(LEG_NAMES), dtype=bool)
    step_face_contacts = np.zeros(len(LEG_NAMES), dtype=bool)

    for contact in data.contact[: data.ncon]:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1) or ""
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2) or ""
        foot_name = ""
        other_name = ""
        if geom1 in LEG_NAMES:
            foot_name, other_name = geom1, geom2
        elif geom2 in LEG_NAMES:
            foot_name, other_name = geom2, geom1
        if not foot_name:
            continue
        foot_index = LEG_NAMES.index(foot_name)

        if other_name == "floor":
            floor_contacts[foot_index] = True
        elif step is not None and other_name == step.geom_name:
            step_contacts[foot_index] = True
            normal = np.asarray(contact.frame[:3], dtype=np.float64)
            if abs(float(normal[0])) > 0.8:
                step_face_contacts[foot_index] = True
            if abs(float(normal[2])) > 0.8:
                step_top_contacts[foot_index] = True

    foot_contacts = floor_contacts | step_contacts
    return ContactSnapshot(
        foot_contacts=tuple(bool(value) for value in foot_contacts),
        floor_contacts=tuple(bool(value) for value in floor_contacts),
        step_contacts=tuple(bool(value) for value in step_contacts),
        step_top_contacts=tuple(bool(value) for value in step_top_contacts),
        step_face_contacts=tuple(bool(value) for value in step_face_contacts),
    )


def terrain_height_at_xy(
    step: StepGeometry | None,
    x_m: float,
    y_m: float,
) -> float:
    """返回水平位置正下方用于净空估算的地形高度。

    参数：
        step：单级台阶几何；平地模式为 None。
        x_m、y_m：足端中心的世界坐标，单位 m。
    返回：
        台阶顶面高度或平地高度 0，单位 m。
    限制：
        只在足端中心落在台阶投影内时使用 0.08 m；台阶边缘处可能与真实
        球体接触点略有差异，因此净空是近似测量而非碰撞穿透量。
    重要性等级：B（用于判断足端是否真的离地）。
    """
    if step is not None and step.contains_xy(x_m, y_m):
        return step.top_z_m
    return 0.0


def measure_foot_clearances(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    step: StepGeometry | None,
) -> np.ndarray:
    """测量四个足端球体底部相对正下方地形的净空。

    参数：
        model：已编译场景。
        data：当前状态，geom_xpos 应已更新。
        step：单级台阶几何或 None。
    返回：
        shape=(4,)，顺序 FL/FR/RL/RR，单位 m。正值表示离地，轻微负值可能
        来自软接触穿透或地形投影误差。
    可能异常：
        足端 geom 缺失时抛出 RuntimeError。
    重要性等级：A（摆动相实测离地高度直接用于验收报告）。
    """
    clearances = np.zeros(len(LEG_NAMES), dtype=np.float64)
    for index, foot_name in enumerate(LEG_NAMES):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        if geom_id < 0:
            raise RuntimeError(f"缺少足端 geom: {foot_name}")
        radius_m = float(model.geom_size[geom_id, 0])
        foot_center = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        ground_z_m = terrain_height_at_xy(
            step, float(foot_center[0]), float(foot_center[1])
        )
        clearances[index] = float(foot_center[2] - radius_m - ground_z_m)
    return clearances


def sample_to_row(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adr: np.ndarray,
    joint_target_rad: np.ndarray,
    swing_mask: np.ndarray,
    leg_phase: np.ndarray,
    initial_x_m: float,
    initial_y_m: float,
    step: StepGeometry | None,
) -> tuple[dict[str, Any], ContactSnapshot, np.ndarray, float, bool]:
    """把一个 MuJoCo 状态转换为 CSV 行及统计辅助值。

    参数：
        model：已编译模型。
        data：当前仿真状态。
        qpos_adr：12 个关节在 qpos 中的地址，shape=(12,)。
        joint_target_rad：当前控制目标，shape=(12,)，单位 rad。
        swing_mask：四足摆动相掩码，shape=(4,)。
        leg_phase：四足相位，shape=(4,)，单位 rad。
        initial_x_m、initial_y_m：记录起点 base 世界坐标，单位 m。
        step：台阶几何或 None。
    返回：
        row、contacts、clearances_m、tilt_deg、fallen。row 可直接交给
        csv.DictWriter；clearances_m shape=(4,)，单位 m。
    可能异常：
        数组 shape 不匹配或四元数非法时抛出异常。
    副作用：
        无；仅读取 MuJoCo 状态。
    重要性等级：A（CSV 数据入口）。
    """
    qpos = np.asarray(data.qpos, dtype=np.float64)
    roll_deg, pitch_deg, yaw_deg, tilt_deg = quaternion_to_euler(qpos[3:7])
    joint_position = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
    contacts = analyze_contacts(model, data, step)
    clearances_m = measure_foot_clearances(model, data, step)
    fallen = bool(qpos[2] < FALL_HEIGHT_M or tilt_deg > FALL_TILT_DEG)
    trot_time_s = max(0.0, float(data.time) - WARMUP_S)
    phase_cycle = float(
        TROT_FREQUENCY_HZ * trot_time_s + PAIR_PHASE_CYCLES[0]
    )

    row: dict[str, Any] = {
        "time_s": float(data.time),
        "trot_time_s": trot_time_s,
        "phase_cycle": phase_cycle,
        "base_x_m": float(qpos[0]),
        "base_y_m": float(qpos[1]),
        "base_z_m": float(qpos[2]),
        "forward_distance_m": float(qpos[0] - initial_x_m),
        "lateral_offset_m": float(qpos[1] - initial_y_m),
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "tilt_deg": tilt_deg,
        "support_count": contacts.support_count,
        "step_top_contact_count": contacts.step_top_count,
        "step_face_contact": int(contacts.has_step_face_contact),
        "fallen": fallen,
    }
    for index, leg_name in enumerate(LEG_NAMES):
        row[f"contact_{leg_name}"] = int(contacts.foot_contacts[index])
        row[f"floor_contact_{leg_name}"] = int(contacts.floor_contacts[index])
        row[f"step_contact_{leg_name}"] = int(contacts.step_contacts[index])
        row[f"swing_{leg_name}"] = int(swing_mask[index])
        row[f"clearance_{leg_name}_m"] = float(clearances_m[index])
    for index, joint_name in enumerate(JOINT_NAMES):
        row[f"target_{joint_name}_rad"] = float(joint_target_rad[index])
        row[f"q_{joint_name}_rad"] = float(joint_position[index])
    if leg_phase.shape != (len(LEG_NAMES),):
        raise ValueError(f"leg_phase 应为 (4,)，实际 {leg_phase.shape}")
    return row, contacts, clearances_m, tilt_deg, fallen


def run_realtime_trot(
    config: RunConfig,
    duration_s: float | None,
    headless: bool,
) -> tuple[dict[str, list[float]], RunStats]:
    """运行带显式抬腿的实时 Trot，并在结束前持续写入 CSV。

    参数：
        config：平地或台阶运行配置。
        duration_s：固定仿真时长，单位 s；None 表示实时窗口持续到用户关闭。
            无窗口模式必须给出有限时长，避免无限运行。
        headless：True 时不启动 3D 窗口且不按墙钟休眠，用于自动验证；
            False 时使用 launch_passive 窗口，并按约 1 倍速播放。
    返回：
        history：画图与复盘所需的时间序列字典，包含 base 位置、支撑数、
            四足净空和台阶顶面接触数。
        stats：累计指标。
    可能异常：
        XML/关键帧/关节映射/OpenGL 错误向上抛出；Ctrl+C 被捕获并保存数据。
    副作用：
        打开并关闭 3D 窗口，覆盖写入配置中的 CSV 文件，修改内存 model/data。
    实时性：
        非 headless 模式每个物理步后按绝对截止时间休眠；若计算超时，不补步，
        避免窗口卡顿后突然加速。headless 模式直接推进物理步。
    重要性等级：A。
    """
    if duration_s is not None and (
        not np.isfinite(duration_s) or float(duration_s) <= 0.0
    ):
        raise ValueError(f"duration_s 必须为正有限值或 None，实际 {duration_s}")
    if headless and duration_s is None:
        raise ValueError("--headless 模式必须同时提供 --duration")

    model = load_model(config)
    timestep = float(model.opt.timestep)
    assert abs(timestep - 0.002) < 1e-12, "场景时间步应为 0.002 s"
    qpos_adr, dof_adr = validate_joint_mapping(model)
    data = mujoco.MjData(model)
    step = configure_initial_state(model, data, config)

    initial_x_m = float(data.qpos[0])
    initial_y_m = float(data.qpos[1])
    stats = RunStats(initial_x_m=initial_x_m, initial_y_m=initial_y_m, step=step)
    history: dict[str, list[float]] = {
        "time_s": [],
        "base_x_m": [],
        "base_y_m": [],
        "base_z_m": [],
        "support_count": [],
        "step_top_contact_count": [],
    }
    for leg_name in LEG_NAMES:
        history[f"clearance_{leg_name}_m"] = []

    print("=" * 78)
    print("Go2 开环抬腿 Trot")
    print("=" * 78)
    print(f"场景: {config.scene_name}")
    print(f"初始 base: x={data.qpos[0]:.3f} m, y={data.qpos[1]:.3f} m, z={data.qpos[2]:.3f} m")
    print(
        f"步态: {TROT_FREQUENCY_HZ:.2f} Hz，指令抬腿 {SWING_HEIGHT_M * 100:.1f} cm，"
        f"启动平滑时间 {GAIT_RAMP_S:.1f} s"
    )
    if step is not None:
        print(
            f"台阶: X=[{step.front_x_m:.3f}, {step.rear_x_m:.3f}] m，"
            f"Y=[{step.left_y_m:.3f}, {step.right_y_m:.3f}] m，顶面 Z={step.top_z_m:.3f} m"
        )
        print(f"台阶前立面: X={step.front_x_m:.3f} m；初始距离={STEP_START_GAP_M:.3f} m")

    viewer = None
    if not headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        if viewer is None:
            raise RuntimeError("mujoco.viewer.launch_passive 未返回窗口句柄")
        viewer.cam.distance = 2.35 if step is None else 2.8
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        viewer.cam.lookat[:] = (
            float(data.qpos[0]) + (0.25 if step is not None else 0.0),
            float(data.qpos[1]),
            0.22,
        )
        if duration_s is None:
            print("3D 窗口已启动；关闭窗口即可结束并自动保存轨迹。")
        else:
            print(f"3D 窗口已启动；将运行 {duration_s:.1f} s 或提前关闭窗口结束。")
    else:
        print(f"无窗口验证模式；按物理时间运行 {duration_s:.1f} s（不等待墙钟）。")

    interrupted = False
    try:
        with config.csv_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            def record_current_state(
                target: np.ndarray,
                swing_mask: np.ndarray,
                leg_phase: np.ndarray,
            ) -> None:
                """记录当前状态并更新画图和累计统计。重要性等级：B。"""
                row, contacts, clearances_m, tilt_deg, fallen = sample_to_row(
                    model,
                    data,
                    qpos_adr,
                    target,
                    swing_mask,
                    leg_phase,
                    initial_x_m,
                    initial_y_m,
                    step,
                )
                writer.writerow(row)
                history["time_s"].append(float(row["time_s"]))
                history["base_x_m"].append(float(row["base_x_m"]))
                history["base_y_m"].append(float(row["base_y_m"]))
                history["base_z_m"].append(float(row["base_z_m"]))
                history["support_count"].append(float(row["support_count"]))
                history["step_top_contact_count"].append(
                    float(row["step_top_contact_count"])
                )
                for index, leg_name in enumerate(LEG_NAMES):
                    history[f"clearance_{leg_name}_m"].append(
                        float(clearances_m[index])
                    )
                stats.update(
                    row,
                    contacts,
                    clearances_m,
                    swing_mask,
                    tilt_deg,
                    fallen,
                )
                if stats.sample_count % CSV_FLUSH_INTERVAL_SAMPLES == 0:
                    csv_file.flush()

            initial_target, initial_swing, initial_phase = target_at_time(float(data.time))
            record_current_state(initial_target, initial_swing, initial_phase)

            max_steps = (
                None
                if duration_s is None
                else int(np.ceil(float(duration_s) / timestep))
            )
            step_count = 0
            next_step_wall_s = time.perf_counter()
            last_viewer_sync_wall_s = 0.0
            last_console_time_s = float(data.time)

            while True:
                if viewer is not None and not viewer.is_running():
                    break
                if max_steps is not None and step_count >= max_steps:
                    break

                joint_target, swing_mask, leg_phase = target_at_time(float(data.time))
                joint_position = np.asarray(data.qpos[qpos_adr], dtype=np.float64)
                joint_velocity = np.asarray(data.qvel[dof_adr], dtype=np.float64)
                torque = KP * (joint_target - joint_position) - KD * joint_velocity
                data.ctrl[:] = np.clip(
                    torque,
                    model.actuator_ctrlrange[:, 0],
                    model.actuator_ctrlrange[:, 1],
                )
                mujoco.mj_step(model, data)
                step_count += 1

                if step_count % SAMPLE_INTERVAL_STEPS == 0:
                    record_current_state(joint_target, swing_mask, leg_phase)

                if float(data.time) - last_console_time_s >= CONSOLE_PRINT_INTERVAL_S:
                    contacts = analyze_contacts(model, data, step)
                    console_line = (
                        f"t={data.time:6.2f}s  x={data.qpos[0]:+.3f}m  "
                        f"z={data.qpos[2]:.3f}m  支撑={contacts.support_count}/4"
                    )
                    if step is not None:
                        console_line += (
                            f"  台阶顶接触={contacts.step_top_count}/4"
                            f"  立面接触={'是' if contacts.has_step_face_contact else '否'}"
                        )
                    print(console_line)
                    last_console_time_s = float(data.time)

                wall_now_s = time.perf_counter()
                if viewer is not None and (
                    wall_now_s - last_viewer_sync_wall_s >= 1.0 / VIEWER_SYNC_HZ
                ):
                    viewer.cam.lookat[:] = (
                        float(data.qpos[0]) + (0.20 if step is not None else 0.0),
                        float(data.qpos[1]),
                        0.22,
                    )
                    viewer.sync()
                    last_viewer_sync_wall_s = wall_now_s

                if viewer is not None:
                    next_step_wall_s += timestep
                    sleep_s = next_step_wall_s - time.perf_counter()
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    else:
                        next_step_wall_s = time.perf_counter()

            if float(data.time) > stats.last_time_s + 0.5 * timestep:
                final_target, final_swing, final_phase = target_at_time(float(data.time))
                record_current_state(final_target, final_swing, final_phase)
            csv_file.flush()
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl+C，正在结束仿真并保存已记录的数据。")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                # 用户已关闭窗口时，不同 GLFW 版本可能让重复 close 抛错；
                # 这里只释放资源，不应覆盖真正的控制或数据错误。
                pass

    if interrupted:
        print("仿真由终端中断，已保存中断前的数据。")
    elif not headless:
        print("3D 窗口运行结束，正在生成复盘图。")
    else:
        print("无窗口验证运行结束，正在生成复盘图。")
    if stats.sample_count < 1:
        raise RuntimeError("没有有效采样，无法生成轨迹图")
    return history, stats


def save_plots(
    history: dict[str, list[float]],
    stats: RunStats,
    config: RunConfig,
) -> None:
    """保存 X-Y 轨迹、机身高度和四足净空曲线。

    参数：
        history：run_realtime_trot 返回的时间序列。
        stats：累计指标，用于标题和台阶判定线。
        config：运行配置，决定输出文件名和台阶标注。
    返回：
        无。
    可能异常：
        目录无写权限、数据为空或 matplotlib 后端错误时向上抛出。
    副作用：
        覆盖写入配置中的两张 PNG，并关闭 matplotlib figure。
    重要性等级：B（用户复盘步态和台阶卡阻的主要可视化输出）。
    """
    configure_chinese_plot_font()
    time_s = np.asarray(history["time_s"], dtype=np.float64)
    x_m = np.asarray(history["base_x_m"], dtype=np.float64)
    y_m = np.asarray(history["base_y_m"], dtype=np.float64)
    z_m = np.asarray(history["base_z_m"], dtype=np.float64)
    if time_s.size == 0:
        raise ValueError("history 为空，无法画图")

    figure_xy, axis_xy = plt.subplots(figsize=(7.8, 6.2))
    axis_xy.plot(x_m, y_m, color="#1f77b4", linewidth=1.8, label="机身轨迹")
    axis_xy.scatter(x_m[0], y_m[0], color="#2ca02c", s=55, label="起点", zorder=4)
    axis_xy.scatter(x_m[-1], y_m[-1], color="#d62728", s=55, label="终点", zorder=4)
    if stats.step is not None:
        step = stats.step
        axis_xy.add_patch(
            matplotlib.patches.Rectangle(
                (step.front_x_m, step.left_y_m),
                step.rear_x_m - step.front_x_m,
                step.right_y_m - step.left_y_m,
                facecolor="#c44e52",
                alpha=0.22,
                edgecolor="#8c2d30",
                linewidth=1.2,
                label="单级台阶投影",
                zorder=1,
            )
        )
        axis_xy.axvline(
            step.front_x_m,
            color="#8c2d30",
            linestyle="--",
            linewidth=1.0,
            label="台阶前立面",
        )
    axis_xy.set_xlabel("世界 X 方向 / m")
    axis_xy.set_ylabel("世界 Y 方向 / m")
    axis_xy.set_title(
        f"Go2 开环抬腿 Trot X-Y：前进 {x_m[-1] - x_m[0]:+.3f} m，"
        f"侧移 {y_m[-1] - y_m[0]:+.3f} m"
    )
    axis_xy.axis("equal")
    axis_xy.grid(alpha=0.25)
    axis_xy.legend(loc="best")
    figure_xy.tight_layout()
    figure_xy.savefig(config.xy_figure_path, dpi=160)
    plt.close(figure_xy)

    figure_z, (axis_height, axis_clearance) = plt.subplots(
        2, 1, figsize=(9.4, 7.2), sharex=True, gridspec_kw={"height_ratios": (1.0, 1.15)}
    )
    axis_height.plot(time_s, z_m, color="#ff7f0e", linewidth=1.5, label="机身高度")
    axis_height.axhline(
        float(np.mean(z_m)),
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        label=f"均值 {np.mean(z_m):.4f} m",
    )
    if stats.step is not None:
        axis_height.axhline(
            stats.step.top_z_m,
            color="#8c2d30",
            linestyle=":",
            linewidth=1.2,
            label=f"台阶顶面 {stats.step.top_z_m:.3f} m",
        )
    axis_height.set_ylabel("机身高度 z / m")
    axis_height.set_title(
        f"Go2 开环抬腿 Trot：机身高度范围 "
        f"[{stats.min_z_m:.4f}, {stats.max_z_m:.4f}] m"
    )
    axis_height.grid(alpha=0.25)
    axis_height.legend(loc="best")

    clearance_colors = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728")
    for index, leg_name in enumerate(LEG_NAMES):
        clearance = np.asarray(
            history[f"clearance_{leg_name}_m"], dtype=np.float64
        )
        axis_clearance.plot(
            time_s,
            clearance,
            color=clearance_colors[index],
            linewidth=1.0,
            alpha=0.85,
            label=f"{leg_name} 最大 {stats.swing_peak_clearance_m[index] * 100:.1f} cm",
        )
    axis_clearance.axhline(
        CLEARANCE_TARGET_M,
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label="3 cm 参考线",
    )
    axis_clearance.axhline(0.0, color="#999999", linewidth=0.8)
    if stats.step is not None:
        axis_clearance.axhline(
            stats.step.top_z_m,
            color="#8c2d30",
            linestyle=":",
            linewidth=1.2,
            label="台阶高度 8 cm",
        )
    axis_clearance.set_xlabel("时间 / s")
    axis_clearance.set_ylabel("足底相对地形净空 / m")
    axis_clearance.grid(alpha=0.25)
    axis_clearance.legend(ncol=2, fontsize=8, loc="best")

    figure_z.tight_layout()
    figure_z.savefig(config.z_figure_path, dpi=160)
    plt.close(figure_z)


def classify_step_result(stats: RunStats) -> str:
    """根据累计接触和位移返回单级台阶测试结论。

    参数：
        stats：已包含台阶几何和接触统计的运行指标。
    返回：
        成功跨上台阶 / 卡在台阶立面 / 到达台阶附近但未跨上 / 尚未到达台阶。
    重要性等级：B（避免把「碰到台阶」误报成「登上台阶」）。
    """
    if stats.step is None:
        return "不适用（平地模式）"
    if stats.climbed_step:
        return "成功跨上台阶"
    if stats.step_face_contact_seen:
        return "卡在台阶立面"
    if stats.max_base_x_m >= stats.step.front_x_m - 0.20:
        return "到达台阶附近，但未跨上"
    return "尚未到达台阶"


def print_summary(stats: RunStats, config: RunConfig) -> None:
    """打印运动、抬腿、稳定性和单级台阶结果摘要。

    参数：
        stats：仿真累计指标。
        config：用于选择输出路径与场景说明。
    返回：
        无。
    可能异常：
        无；仅格式化输出。
    副作用：
        写入标准输出。
    重要性等级：B。
    """
    duration_s = stats.last_time_s - (stats.first_time_s or 0.0)
    forward_m = stats.final_x_m - stats.initial_x_m
    lateral_m = stats.final_y_m - stats.initial_y_m
    mean_support = stats.support_sum / max(stats.sample_count, 1)
    contact_duty = stats.contact_counts / max(stats.sample_count, 1)
    total_swing_samples = int(np.sum(stats.swing_sample_counts))
    legs_reaching_target = int(np.sum(stats.swing_peak_clearance_m >= CLEARANCE_TARGET_M))

    print("\n" + "=" * 78)
    print("Go2 开环抬腿 Trot 总结")
    print("=" * 78)
    print(f"场景: {config.scene_name}")
    print(f"仿真时长: {duration_s:.2f} s")
    print(f"总前进距离: {forward_m:+.4f} m")
    print(f"侧向偏移: {lateral_m:+.4f} m")
    print(f"最大绝对侧向偏移: {stats.max_abs_lateral_m:.4f} m")
    print(
        f"机身高度: 范围=[{stats.min_z_m:.4f}, {stats.max_z_m:.4f}] m，"
        f"最终={stats.final_z_m:.4f} m"
    )
    print(f"最大倾角: {stats.max_abs_tilt_deg:.3f}°")
    print(
        f"支撑: 平均={mean_support:.2f}/4，最少={stats.min_support}/4，"
        f"末态={stats.final_support}/4"
    )
    print(
        "各足接触占比: "
        f"FL={contact_duty[0] * 100:.1f}%, FR={contact_duty[1] * 100:.1f}%, "
        f"RL={contact_duty[2] * 100:.1f}%, RR={contact_duty[3] * 100:.1f}%"
    )
    print(
        "摆动相实测最大足底净空: "
        f"FL={stats.swing_peak_clearance_m[0] * 100:.2f} cm, "
        f"FR={stats.swing_peak_clearance_m[1] * 100:.2f} cm, "
        f"RL={stats.swing_peak_clearance_m[2] * 100:.2f} cm, "
        f"RR={stats.swing_peak_clearance_m[3] * 100:.2f} cm"
    )
    print(
        f"达到 {CLEARANCE_TARGET_M * 100:.0f} cm 抬腿目标的足端: "
        f"{legs_reaching_target}/4（摆动采样共 {total_swing_samples} 次）"
    )
    print(f"是否跌倒: {'是' if stats.fallen else '否'}")

    if stats.fallen:
        print(f"首次跌倒时间: {stats.first_fallen_time_s:.3f} s")
        print("\n现象与可能原因（本次未自动修改任何参数）：")
        print("  - 对角相位或关节正负号可能与当前机体坐标约定不完全匹配")
        print("  - 2 Hz 步频、0.18 rad 大腿振幅或抬腿脉冲可能造成接触冲击")
        print("  - 足端摩擦不足、触地滑移或摆动腿落地时机不合适")
        print("  - 开环控制没有机身姿态和落脚反馈，无法主动纠正倾斜")
    else:
        print("限制: 未触发跌倒判据，但开环控制不等于稳定步态或爬楼验收。")

    if stats.step is not None:
        print("\n单级台阶测试:")
        print(f"结论: {classify_step_result(stats)}")
        print(f"台阶前立面 X: {stats.step.front_x_m:.4f} m")
        print(f"base 最大 X: {stats.max_base_x_m:.4f} m")
        print(f"曾接触台阶立面: {'是' if stats.step_face_contact_seen else '否'}")
        print(
            f"台阶顶面最多同时接触足数: "
            f"{stats.max_step_top_contact_count}/4"
        )
        if not stats.climbed_step:
            peak_clearance_cm = 100.0 * float(np.max(stats.swing_peak_clearance_m))
            print(
                f"当前最大实测抬脚约 {peak_clearance_cm:.1f} cm，"
                f"台阶高 {stats.step.top_z_m * 100:.1f} cm。"
            )
            print(
                "可能原因: 摆动净空低于立面高度，且开环步态没有机身抬头、"
                "落脚点选择或接触反馈，足端撞到立面后无法继续抬升。"
            )

    print(f"CSV: {config.csv_path}")
    print(f"X-Y 轨迹图: {config.xy_figure_path}")
    print(f"高度/净空图: {config.z_figure_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析场景、运行时长和无窗口选项。

    参数：
        argv：可选参数列表；None 时从 sys.argv 读取。
    返回：
        argparse.Namespace。scene 为 flat 或 single-step；duration 单位 s；
        headless=True 时不启动 3D 窗口。
    可能异常：
        非法参数由 argparse 以 SystemExit 终止。
    重要性等级：B。
    """
    parser = argparse.ArgumentParser(
        description="Go2 开环抬腿 Trot：平地实时运行与单级低台阶测试"
    )
    parser.add_argument(
        "--scene",
        choices=("flat", "single-step"),
        default="flat",
        help="flat=平地；single-step=0.08 m 单级台阶（默认 flat）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="固定仿真时长（秒）。省略时实时窗口持续到用户关闭；headless 必须指定",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不打开 3D 窗口、不按墙钟等待，仅用于自动验证",
    )
    return parser.parse_args(argv)


def build_config(scene_name: str) -> RunConfig:
    """根据场景名称构建输入输出路径配置。

    参数：
        scene_name：flat 或 single-step。
    返回：
        RunConfig；路径均为绝对路径。
    可能异常：
        scene_name 非预期值时抛出 ValueError。
    重要性等级：C。
    """
    if scene_name == "flat":
        return RunConfig(
            scene_name=scene_name,
            xml_path=FLAT_XML_PATH,
            keyframe_name="stand_home",
            csv_path=FLAT_CSV_PATH,
            xy_figure_path=FLAT_XY_FIGURE_PATH,
            z_figure_path=FLAT_Z_FIGURE_PATH,
        )
    if scene_name == "single-step":
        return RunConfig(
            scene_name=scene_name,
            xml_path=SINGLE_STEP_XML_PATH,
            keyframe_name="single_step_home",
            csv_path=STEP_CSV_PATH,
            xy_figure_path=STEP_XY_FIGURE_PATH,
            z_figure_path=STEP_Z_FIGURE_PATH,
        )
    raise ValueError(f"未知场景: {scene_name}")


def main() -> int:
    """解析参数，运行实时抬腿 Trot，并保存、打印本次复盘结果。

    返回：
        0 表示运行结束且 CSV/PNG 已生成；即使摔倒或未登阶也返回 0，因为
        本脚本用于测量现象，不把一次运动结果当作程序执行失败。
    可能异常：
        3D 窗口/OpenGL 初始化失败、模型不匹配或输出目录不可写时向上抛出。
    副作用：
        打开桌面 3D 窗口（非 headless），写入 CSV 和两张 PNG。
    重要性等级：A。
    """
    args = parse_args()
    config = build_config(args.scene)
    history, stats = run_realtime_trot(
        config=config,
        duration_s=args.duration,
        headless=bool(args.headless),
    )
    save_plots(history, stats, config)
    print_summary(stats, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
