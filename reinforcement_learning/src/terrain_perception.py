"""局部 LiDAR 点云到机身高度图的通用感知模块。

文件目的：
    使用 MuJoCo 官方 ``mj_ray`` 的低分辨率前向射线扫描单级台阶，并把点云栅格化为
    机器人机身坐标系下的局部高度图。仿真训练、MuJoCo 回放和未来 ROS2 真机
    接入必须共用 ``pointcloud_to_heightmap``，避免重复实现两套感知逻辑。
整体输入：
    MuJoCo ``MjModel``/``MjData``；模型 ``base`` body 内必须存在
    ``lidar_site``。仿真扫描输出 shape=(N,3) 的机身局部点云，单位 m。
整体输出：
    ``TerrainScan``，其中默认 ``heightmap`` shape=(10,10)、dtype=float32、
    单位 m，行沿机身 +X（前向），列沿机身 +Y（左向）；无有效点的格子填
    机身当前离地高度的负值（平地约为 -base_z），便于直接比较台阶相对高度。
依赖与版本要求：
    Python 3.10+、mujoco 3.16.0、numpy 2.x。
运行方式：
    由 ``Go2Env(terrain="single_step")`` 初始化并调用，不单独启动窗口。
关键假设：
    1. 机器人机身局部坐标遵循 MuJoCo 约定：+X 前、+Y 左、+Z 上。
    2. 高度图为“机身局部高程”，不是绝对地形高度。平地稳定时同一格约等于
       机身离地高度的负值；踩上课程台阶后该格数值约抬高对应台阶高度。
    3. LiDAR 使用 13×11 共 143 条前向射线，水平 ±30°、垂直 -75°～-25°。
       当前高度图窗口为机身局部 x=0.20～0.70 m、y=-0.25～0.25 m；
       近端无命中的格子使用当前机身离地高度补值，避免把空白误判为障碍。
    4. mj_ray 对无命中射线返回负距离；这些射线必须先过滤，不能把
       可能为零向量的 hit point 当作真实障碍点。
已知限制：
    当前默认高度图展平为 100 维；旧 60 维高度图模型和旧 110 维策略输入语义
    均不兼容，必须使用端到端 150 维观测重新训练。高度图仍同时保留在
    ``info`` 中供回放诊断。
    当前还假设机身近似水平，未做点云去滚转。
重要性等级：A（仿真与未来 ROS2 真机共用的地形感知边界）。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class HeightmapProfile:
    """描述一整套 LiDAR pattern 与局部高度图栅格参数。

    字段：
        name：配置名，仅用于日志和故障排查。
        rows：沿机身 +X 的栅格数。
        cols：沿机身 +Y 的栅格数。
        resolution_m：单格边长，单位 m，必须为正数。
        x_min_m：栅格最近 X 边界，单位 m，相对 ``base``。
        y_min_m：栅格最右 Y 边界，单位 m，相对 ``base``。
        horizontal_rays：水平方向射线数。
        vertical_rays：垂直方向射线数。
    生命周期：
        模块加载时创建并作为不可变配置共享；训练和回放应显式使用同一 profile。
    重要性等级：A（改变网格即改变策略输入维度，旧模型不再兼容）。
    """

    name: str
    rows: int
    cols: int
    resolution_m: float
    x_min_m: float
    y_min_m: float
    horizontal_rays: int
    vertical_rays: int


# 0.04 m 台阶至少占据一行 0.05 m 高度图；窗口聚焦机头前方近场，避免把
# 计算浪费在远处地形。13×11=143 条射线仍可维持 50 Hz 训练。
FINE_HEIGHTMAP_PROFILE = HeightmapProfile(
    name="fine_0p05",
    rows=10,
    cols=10,
    resolution_m=0.05,
    x_min_m=0.20,
    y_min_m=-0.25,
    horizontal_rays=13,
    vertical_rays=11,
)

# 兼容旧导入名的默认值；新代码应优先显式传递 ``HeightmapProfile``。
HEIGHTMAP_ROWS = FINE_HEIGHTMAP_PROFILE.rows
HEIGHTMAP_COLS = FINE_HEIGHTMAP_PROFILE.cols
HEIGHTMAP_RESOLUTION_M = FINE_HEIGHTMAP_PROFILE.resolution_m
HEIGHTMAP_X_MIN_M = FINE_HEIGHTMAP_PROFILE.x_min_m
HEIGHTMAP_Y_MIN_M = FINE_HEIGHTMAP_PROFILE.y_min_m

LIDAR_SITE_NAME = "lidar_site"
LIDAR_CUTOFF_DIST_M = 1.2
LIDAR_HORIZONTAL_RAY_COUNT = FINE_HEIGHTMAP_PROFILE.horizontal_rays
LIDAR_VERTICAL_RAY_COUNT = FINE_HEIGHTMAP_PROFILE.vertical_rays
LIDAR_HORIZONTAL_FOV_RAD = (math.radians(-30.0), math.radians(30.0))
# lidar_site 相对 base 为 (0.32, 0, 0.07) m，并在 XML 中下倾 5°；
# 因此射线相对水平面的有效仰角约为 -80°～-30°。高度图保留机身前向
# x=0.20～0.70 m、y=-0.25～0.25 m 的近场；调大/调小仰角会改变近端覆盖与
# 台阶进入视野的距离，必须连同扫描 pattern 一起重新验证。
LIDAR_VERTICAL_FOV_RAD = (math.radians(-75.0), math.radians(-25.0))


@dataclass(frozen=True)
class TerrainScan:
    """保存一次前向 LiDAR 扫描的可复用结果。

    字段：
        heightmap：shape=(rows,cols)、float32、单位 m，机身局部高度图；默认
            shape=(10,10)，无点格填当前机身离地高度的负值，平地约为 -base_z。
        points_m：shape=(N,3)、float64、单位 m，过滤无命中后的机身局部点云。
        hit_count：有效命中点数，非负整数。
        ray_count：本帧发射的射线总数；默认细网格 profile 为 143。
        scan_ms：扫描、坐标变换和栅格化总耗时，单位 ms，用于训练性能排查。
    数据生命周期：
        每次 ``TerrainPerception.scan()`` 创建独立对象，不与 MuJoCo 内部数组共享。
    重要性等级：B（训练 info、回放诊断和未来真机适配共同读取）。
    """

    heightmap: np.ndarray
    points_m: np.ndarray
    hit_count: int
    ray_count: int
    scan_ms: float


def pointcloud_to_heightmap(
    points_m: np.ndarray,
    *,
    x_min_m: float,
    y_min_m: float,
    resolution_m: float,
    rows: int,
    cols: int,
    default_height_m: float = 0.0,
) -> np.ndarray:
    """把机身局部点云栅格化为每个网格的最大高度。

    参数：
        points_m：shape=(N,3)、单位 m 的机身局部点云；列顺序为 X/Y/Z。
        x_min_m：高度图第一行左边界，单位 m；行沿 +X 方向增长。
        y_min_m：高度图第一列下边界，单位 m；列沿 +Y 方向增长。
        resolution_m：单格边长，单位 m，必须为正数。
        rows：X 方向网格数，必须为正整数。
        cols：Y 方向网格数，必须为正整数。
        default_height_m：无有效点的格子填充值，单位 m；TerrainPerception 在
            平地传入 -base_z，使无命中格与地面基准高度一致。
    返回：
        shape=(rows, cols)、dtype=float32 的高度图。每格取落入该格所有有效点的
        最大 Z 值；NaN、Inf 和越界点被忽略。
    可能异常：
        points_m shape 不是 (N,3)、rows/cols 非正或 resolution_m 非正时抛出
        ValueError。函数不访问 MuJoCo，也不会修改输入数组。
    算法与数值边界：
        先以 -inf 初始化，避免负高程点被填充值覆盖；最后再把从未命中的格子
        填为 ``default_height_m``。索引使用 floor 半开区间
        ``[min, min + count * resolution)``，边界点不会越界写入。
    真机复用说明：
        仿真侧由 MuJoCo ``mj_ray`` 直接产生点云；真机侧订阅 ROS2 ``PointCloud2``，先经
        TF 转到机器人 base 坐标系，再调用本函数。感知逻辑完全复用，不允许在
        ROS2 节点里另写一套栅格化规则。
    重要性等级：A（仿真与真机共用的纯函数感知接口）。
    """
    points = np.asarray(points_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_m shape 应为 (N,3)，实际 {points.shape}")
    if not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"rows 必须为正整数，实际 {rows!r}")
    if not isinstance(cols, int) or cols <= 0:
        raise ValueError(f"cols 必须为正整数，实际 {cols!r}")
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError(f"resolution_m 必须为正数，实际 {resolution_m!r}")

    heightmap = np.full((rows, cols), -np.inf, dtype=np.float64)
    if points.shape[0] == 0:
        heightmap.fill(default_height_m)
        return heightmap.astype(np.float32)

    finite_mask = np.all(np.isfinite(points), axis=1)
    points = points[finite_mask]
    if points.shape[0] == 0:
        heightmap.fill(default_height_m)
        return heightmap.astype(np.float32)

    x_max_m = float(x_min_m) + rows * float(resolution_m)
    y_max_m = float(y_min_m) + cols * float(resolution_m)
    valid_mask = (
        (points[:, 0] >= float(x_min_m))
        & (points[:, 0] < x_max_m)
        & (points[:, 1] >= float(y_min_m))
        & (points[:, 1] < y_max_m)
    )
    valid_points = points[valid_mask]
    if valid_points.shape[0] == 0:
        heightmap.fill(default_height_m)
        return heightmap.astype(np.float32)

    row_indices = np.floor(
        (valid_points[:, 0] - float(x_min_m)) / float(resolution_m)
    ).astype(np.int32)
    col_indices = np.floor(
        (valid_points[:, 1] - float(y_min_m)) / float(resolution_m)
    ).astype(np.int32)
    np.maximum.at(heightmap, (row_indices, col_indices), valid_points[:, 2])
    heightmap[~np.isfinite(heightmap)] = float(default_height_m)
    return heightmap.astype(np.float32)



def heightmap_to_nearest_step_distance(
    heightmap: np.ndarray,
    *,
    x_min_m: float,
    y_min_m: float,
    resolution_m: float,
    height_threshold_m: float = 0.015,
    max_distance_m: float = 1.0,
) -> float:
    """从局部高度图中估计机身到最近台阶边缘的水平距离。

    参数：
        heightmap：shape=(rows, cols)、单位 m 的机身局部高度图。行沿机身
            +X，列沿机身 +Y；无效格可为 NaN 或 Inf。
        x_min_m：高度图第一行左边界，单位 m，相对机身中心。
        y_min_m：高度图第一列下边界，单位 m，相对机身中心。
        resolution_m：单格边长，单位 m，必须为正数。
        height_threshold_m：判定“高于地面”的阈值，单位 m，默认 0.015。
            4 cm 台阶在 5 mm 雷达噪声下仍应稳定超过该值；调得太小会误把
            噪声当台阶，调得太大则低台阶可能完全不可见。
        max_distance_m：无台阶或距离超出量程时的返回值，单位 m，默认 1.0。
            该值必须为正数，并与策略观测上界一致。

    返回：
        float，单位 m，范围 [0, max_distance_m]。返回值是高度图中首个高台
        网格中心到机身原点的最小水平欧氏距离；无有效台阶时返回量程上限。

    算法与边界：
        1. 过滤 NaN/Inf 网格；
        2. 用 20% 分位数估计局部地面高度，避免台阶占多数网格时被中位数污染；
        3. 只把高于基线 ``height_threshold_m`` 的网格视为台阶；
        4. 使用网格中心坐标计算水平距离，因此存在不超过半格加半格的量化误差。

    真机复用说明：
        仿真侧输入 MuJoCo 射线生成的高度图；真机侧输入 PointCloud2 经 TF 转到
        base 坐标系后、由 ``pointcloud_to_heightmap`` 生成的同一格式高度图。
        两条链路共享本函数，避免训练和部署对“台阶距离”的定义漂移。

    重要性等级：A（新增 1 维真机可获取的本体/雷达相对观测）。
    """
    heightmap_array = np.asarray(heightmap, dtype=np.float64)
    if heightmap_array.ndim != 2:
        raise ValueError(f"heightmap 应为二维数组，实际 shape={heightmap_array.shape}")
    if not math.isfinite(resolution_m) or resolution_m <= 0.0:
        raise ValueError(f"resolution_m 必须为正数，实际 {resolution_m!r}")
    if not math.isfinite(height_threshold_m) or height_threshold_m < 0.0:
        raise ValueError(
            f"height_threshold_m 必须为有限非负数，实际 {height_threshold_m!r}"
        )
    if not math.isfinite(max_distance_m) or max_distance_m <= 0.0:
        raise ValueError(f"max_distance_m 必须为正数，实际 {max_distance_m!r}")

    finite_mask = np.isfinite(heightmap_array)
    if not np.any(finite_mask):
        return float(max_distance_m)

    finite_heights = heightmap_array[finite_mask]
    ground_baseline_m = float(np.percentile(finite_heights, 20.0))
    elevated_mask = finite_mask & (
        heightmap_array > ground_baseline_m + float(height_threshold_m)
    )
    if not np.any(elevated_mask):
        return float(max_distance_m)

    row_indices, col_indices = np.nonzero(elevated_mask)
    grid_center_x_m = (
        float(x_min_m) + (row_indices.astype(np.float64) + 0.5) * float(resolution_m)
    )
    grid_center_y_m = (
        float(y_min_m) + (col_indices.astype(np.float64) + 0.5) * float(resolution_m)
    )
    horizontal_distance_m = np.hypot(grid_center_x_m, grid_center_y_m)
    nearest_distance_m = float(np.min(horizontal_distance_m))
    return float(np.clip(nearest_distance_m, 0.0, float(max_distance_m)))


class TerrainPerception:
    """封装 MuJoCo LiDAR 扫描和机身局部高度图生成。

    功能：
        使用 MuJoCo 官方 ``mj_ray``，按所选 profile 每帧发射前向射线，将命中点
        从 LiDAR 局部坐标转换到 world，再转换到 ``base`` body 局部坐标，最后
        调用纯函数 ``pointcloud_to_heightmap``。
    输入：
        model：已编译的 MuJoCo 模型，必须包含 ``base`` body 和 ``lidar_site``。
        base_body_name：坐标参考机身名，默认 ``"base"``。
        site_name：LiDAR 安装 site 名，默认 ``"lidar_site"``。
        heightmap_profile：高度图与射线 pattern；默认 ``FINE_HEIGHTMAP_PROFILE``，
            输出 10×10、0.05 m 栅格和 143 条射线。
    输出：
        每次 ``scan(data)`` 返回 ``TerrainScan``。
    可能异常：
        缺少模型 body/site、射线方向包含非法数值或 MuJoCo 查询失败时在
        构造或扫描阶段抛出异常。
    副作用：
        构造时只预计算射线方向；``scan`` 只读取 MuJoCo 状态，不修改物理
        状态、不打开窗口、不写文件。
    性能与排查：
        若 ``scan_ms`` 明显增大，先检查是否误用了全向/高线束 pattern，再检查
        是否把机器人自身碰撞组纳入扫描。当前 group 3 为 Go2 碰撞体，必须屏蔽。
    重要性等级：A（台阶环境的地形感知入口）。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        base_body_name: str = "base",
        site_name: str = LIDAR_SITE_NAME,
        heightmap_profile: HeightmapProfile = FINE_HEIGHTMAP_PROFILE,
    ) -> None:
        self.heightmap_profile = heightmap_profile
        if heightmap_profile.rows <= 0 or heightmap_profile.cols <= 0:
            raise ValueError("高度图 rows/cols 必须为正整数")
        if not math.isfinite(heightmap_profile.resolution_m) or (
            heightmap_profile.resolution_m <= 0.0
        ):
            raise ValueError("高度图 resolution_m 必须为正数")
        if heightmap_profile.horizontal_rays <= 0 or heightmap_profile.vertical_rays <= 0:
            raise ValueError("LiDAR 水平和垂直射线数必须为正整数")
        self.base_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        if self.base_body_id < 0:
            raise ValueError(f"模型缺少机身 body：{base_body_name}")
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, site_name
        )
        if self.site_id < 0:
            raise ValueError(f"模型缺少 LiDAR site：{site_name}")

        self.model_storage = model
        self.site_name = site_name
        # 用 NumPy 生成前向扫描网格，直接调用 MuJoCo 官方 ``mj_ray``。
        # profile 同时决定计算量与高度图分辨率，回放和训练必须保持一致。
        theta_grid, phi_grid = np.meshgrid(
            np.linspace(
                LIDAR_HORIZONTAL_FOV_RAD[0],
                LIDAR_HORIZONTAL_FOV_RAD[1],
                heightmap_profile.horizontal_rays,
            ),
            np.linspace(
                LIDAR_VERTICAL_FOV_RAD[0],
                LIDAR_VERTICAL_FOV_RAD[1],
                heightmap_profile.vertical_rays,
            ),
        )
        ray_theta = theta_grid.reshape(-1)
        ray_phi = phi_grid.reshape(-1)
        cos_phi = np.cos(ray_phi)
        self.ray_directions_local = np.column_stack(
            (
                cos_phi * np.cos(ray_theta),
                cos_phi * np.sin(ray_theta),
                np.sin(ray_phi),
            )
        ).astype(np.float64, copy=False)
        self.ray_theta = ray_theta.astype(np.float32, copy=False)
        self.ray_phi = ray_phi.astype(np.float32, copy=False)

        # Go2 视觉 geom 在 group 2、碰撞 geom 在 group 3；地形碰撞体和地面在
        # group 0。geomgroup 只保留 0/1，排除全部 Go2 视觉/碰撞 geom，也避免
        # 命中纯视觉钢梯 STL。bodyexclude=base 会递归排除 base 及其腿部子 body，
        # 作为第二道保险。
        self.geomgroup = np.ones(mujoco.mjNGROUP, dtype=np.ubyte)
        self.geomgroup[2:] = 0

    @property
    def ray_count(self) -> int:
        """返回每帧射线数量，便于日志和性能断言。"""
        return int(self.ray_theta.shape[0])

    def scan(self, data: mujoco.MjData) -> TerrainScan:
        """执行一次前向扫描并返回机身局部高度图。

        参数：
            data：当前 MuJoCo 数据；调用前应已执行 ``mj_step`` 或 ``mj_forward``，
                使 site、geom 和 body 的世界位姿为最新值。
        返回：
            ``TerrainScan``；``heightmap`` shape=(10,10)，``points_m`` shape=(N,3)。
        可能异常：
            MuJoCo 未返回有效 site 位姿或射线数组异常时抛出 RuntimeError，避免
            静默返回伪造平地高度图。
        副作用：
            只读取 MuJoCo 状态并发射只读射线；不推进物理、不写文件。
        重要性等级：A（训练每一步的地形信号来源）。
        """
        start_s = time.perf_counter()
        site_position = np.asarray(
            data.site_xpos[self.site_id], dtype=np.float64
        ).copy()
        site_rotation = np.asarray(
            data.site_xmat[self.site_id], dtype=np.float64
        ).reshape(3, 3)
        if not np.all(np.isfinite(site_position)) or not np.all(
            np.isfinite(site_rotation)
        ):
            raise RuntimeError("LiDAR site 世界位姿包含 NaN/Inf")

        ray_directions_world = self.ray_directions_local @ site_rotation.T
        world_hit_points: list[np.ndarray] = []
        # 单条 `mj_ray` 的 bodyexclude 会排除 base 及全部子 body；geomgroup
        # 再排除 Go2 的视觉/碰撞 group。两者同时启用，防止扫到腿或机身。
        for ray_direction in ray_directions_world:
            geom_id = np.array((-1,), dtype=np.int32)
            distance_m = float(
                mujoco.mj_ray(
                    self.model_storage,
                    data,
                    site_position,
                    ray_direction,
                    self.geomgroup,
                    True,
                    self.base_body_id,
                    geom_id,
                )
            )
            if not math.isfinite(distance_m) or distance_m < 0.0:
                continue
            world_hit_points.append(
                site_position + distance_m * ray_direction
            )

        if world_hit_points:
            world_points = np.vstack(world_hit_points).astype(
                np.float64, copy=False
            )
        else:
            world_points = np.empty((0, 3), dtype=np.float64)

        base_rotation = np.asarray(
            data.xmat[self.base_body_id], dtype=np.float64
        ).reshape(3, 3)
        base_position = np.asarray(
            data.xpos[self.base_body_id], dtype=np.float64
        )
        base_local_points = (world_points - base_position) @ base_rotation
        profile = self.heightmap_profile
        heightmap = pointcloud_to_heightmap(
            base_local_points,
            x_min_m=profile.x_min_m,
            y_min_m=profile.y_min_m,
            resolution_m=profile.resolution_m,
            rows=profile.rows,
            cols=profile.cols,
            default_height_m=-float(base_position[2]),
        )
        scan_ms = (time.perf_counter() - start_s) * 1000.0
        return TerrainScan(
            heightmap=heightmap,
            points_m=base_local_points,
            hit_count=int(base_local_points.shape[0]),
            ray_count=self.ray_count,
            scan_ms=float(scan_ms),
        )
    def close(self) -> None:
        """释放感知对象。

        当前 mj_ray 实现不持有窗口、线程或文件句柄；保留接口用于未来切换
        其他 backend 时统一生命周期。重要性等级：C。
        """
        return None



