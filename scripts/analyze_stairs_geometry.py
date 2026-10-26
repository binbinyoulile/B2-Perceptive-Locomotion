"""从原始楼梯 STL 提取可落脚面，并生成简化碰撞体几何数据。

文件目的：
    只读解析 ``models/stairs/steel_stair.STL``，判断 STL 格式和尺度，
    识别上行踏步、中间平台和下行踏步，为 MuJoCo 的 box 碰撞体提供参数。
整体输入：
    二进制 STL 文件；脚本不修改原始 STL。
整体输出：
    ``models/stairs/geometry_estimate.json``，包含 STL 元数据、世界包围盒、
    7 个落脚面以及可直接写入 MuJoCo XML 的 box 中心和半尺寸。
依赖与版本要求：
    Python 3.13、NumPy 2.x、trimesh 5.x。当前验证版本为
    NumPy 2.5.4、trimesh 5.2.0。
运行方式：
    ``D:\\Python313\\python.exe D:\\桌面\\机器狗\\scripts\\analyze_stairs_geometry.py``
关键假设：
    1. STL 坐标单位为毫米，和现有场景的 ``scale=0.001`` 一致。
    2. 场景中的世界变换为：缩放 0.001、绕 X 轴 +90°、Y 平移 0.936 m。
    3. 7 个落脚面采用 20 mm 厚薄板碰撞；立柱、侧梁、横撑和栏杆由 scene.xml 另行定义。
已知限制：
    单位是由模型尺寸和已确认场景推断，不是 STL 文件内显式声明的元数据。
    当前以 STL 为几何来源；若后续提供新版 CAD 或实测模型，再重新生成并执行回归验证。
重要性等级：A（碰撞体几何的唯一数据入口，错误会直接影响接触与训练）。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parents[1]
STL_PATH = ROOT / "models" / "stairs" / "steel_stair.STL"
OUTPUT_PATH = ROOT / "models" / "stairs" / "geometry_estimate.json"

# STL 未携带可信单位字段；根据原始尺寸约 4087 x 936 x 1407 和已确认场景，
# 当前唯一采用“毫米”假设。若现场模型改成米制，必须同步修改并重新验证。
STL_SCALE = 0.001
# 落脚面只建薄板，避免把楼梯下方误建为实心体。20 mm 是视觉钢板的保守碰撞厚度；
# 调小会提高足端穿透风险，调大会额外抬高落脚面或压缩板下净空。
TREAD_THICKNESS_M = 0.020
SCENE_TRANSLATION = np.array((0.0, 0.936, 0.0), dtype=np.float64)
UP_NORMAL_Z_MIN = 0.9999
SURFACE_Z_DECIMALS = 4
BBOX_TOUCH_TOLERANCE = 1e-5
MIN_SURFACE_AREA = 0.1
MIN_SURFACE_WIDTH = 0.5


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def inspect_stl(path: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """读取 STL 并返回未变换网格和文件元数据。

    功能：
        检测二进制 STL 文件和三角形数，计算 SHA-256，并用 trimesh 加载网格。
    参数：
        path: STL 绝对路径，文件应存在且可读。
    返回：
        ``(mesh, metadata)``。mesh 为原始坐标的 ``trimesh.Trimesh``；
        metadata 包含文件大小、格式、三角形数、header、哈希和原始包围盒。
    可能异常：
        文件不存在、无权限、STL 损坏或加载结果不是单个 Trimesh 时抛出异常。
    副作用：
        只读取源文件，不写磁盘。
    关键参数：
        size 与 ``84 + 50 * triangle_count`` 的整数关系用于可靠识别二进制 STL。
    排查入口：
        若 trimesh 报解析错误，先用独立 STL 查看器确认文件是否完整。
    重要性等级：A。
    """
    if not path.is_file():
        raise FileNotFoundError(f"找不到原始 STL：{path}")

    data = path.read_bytes()
    size = len(data)
    is_binary = size >= 84 and (size - 84) % 50 == 0
    triangle_count = (size - 84) // 50 if is_binary else None
    header = data[:80].split(b"\0", 1)[0].decode("ascii", errors="replace").strip()

    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"trimesh 未加载到单个网格，而是：{type(loaded)!r}")

    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    metadata = {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(data).hexdigest().upper(),
        "size_bytes": size,
        "format": "binary STL" if is_binary else "ASCII STL 或未知格式",
        "header": header,
        "triangle_count": int(len(loaded.faces)),
        "binary_size_triangle_count": triangle_count,
        "raw_bounds_mm_assumed": {
            "min": bounds[0].round(6).tolist(),
            "max": bounds[1].round(6).tolist(),
            "size": (bounds[1] - bounds[0]).round(6).tolist(),
        },
    }
    return loaded, metadata


def world_mesh(original: trimesh.Trimesh) -> trimesh.Trimesh:
    """把 STL 网格变换到当前 MuJoCo 场景的世界坐标。

    功能：
        依次执行毫米到米缩放、绕世界 X 轴 +90° 和 Y 轴平移。
    参数：
        original: 原始 STL 坐标中的网格，单位为假设毫米。
    返回：
        新的世界坐标网格，单位为米；原对象不被修改。
    可能异常：
        输入不是 Trimesh 时由 trimesh 内部报错。
    副作用：
        仅修改返回的副本。
    算法与坐标：
        ``p_world = R_x(+90°) * (0.001 * p_stl) + (0, 0.936, 0)``。
        该变换与 ``scene.xml`` 中楼梯 body 的姿态一致，不能改成 -90°。
    重要性等级：A。
    """
    mesh = original.copy()
    mesh.apply_scale(STL_SCALE)
    rotation = trimesh.transformations.rotation_matrix(
        angle=np.pi / 2.0,
        direction=(1.0, 0.0, 0.0),
        point=(0.0, 0.0, 0.0),
    )
    mesh.apply_transform(rotation)
    mesh.apply_translation(SCENE_TRANSLATION)
    return mesh


def extract_upward_surfaces(mesh: trimesh.Trimesh) -> list[dict[str, Any]]:
    """提取朝上、共高且空间相连的水平三角面，形成候选落脚面。

    功能：
        先按面法向筛选严格水平向上的三角面，再按同一高度分组，最后用 XY
        包围盒相交关系把被 STL 拆散的三角面重新聚成完整踏面。
    参数：
        mesh: 世界坐标 Trimesh，单位米，Z 轴竖直向上。
    返回：
        候选面列表；每项包含 z_top、面积、x/y 范围、三角形数和中心/半尺寸。
    可能异常：
        空网格或非有限坐标会触发 ValueError；未找到候选面时返回空列表。
    副作用：
        无。
    算法与数值边界：
        - 用叉积得到三角面面积和法向，只保留 normal_z > 0.9999 的面，
          避免把栏杆斜面和踏面倒角误认成主要支持面。
        - 同一 z 的面仅在 XY 包围盒重叠或接触 1e-5 m 时合并。
        - 过滤面积小于 0.1 m² 或宽度小于 0.5 m 的装饰面。
        - 在高度差 0.03 m 内、XY 明显重叠时，只保留面积更大的主落脚面。
    重要性等级：A。
    """
    triangles = np.asarray(mesh.triangles, dtype=np.float64)
    if triangles.size == 0 or not np.isfinite(triangles).all():
        raise ValueError("网格没有有效三角形或包含 NaN/Inf")

    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal_z = np.divide(
        cross[:, 2],
        2.0 * area,
        out=np.zeros_like(area),
        where=area > 1e-12,
    )
    upward_ids = np.where((normal_z > UP_NORMAL_Z_MIN) & (area > 1e-8))[0]
    if upward_ids.size == 0:
        return []

    z_round = np.round(triangles[:, :, 2].mean(axis=1), SURFACE_Z_DECIMALS)
    bbox_min = triangles.min(axis=1)
    bbox_max = triangles.max(axis=1)

    parent = {int(face_id): int(face_id) for face_id in upward_ids}

    def find(item: int) -> int:
        """查找并压缩并查集根节点。

        参数：
            item: 朝上三角面的 face ID，必须已存在于 ``parent``。
        返回：
            该集合的根 face ID。
        可能异常：
            face ID 不存在时抛出 KeyError。
        副作用：
            沿途修改 ``parent`` 做路径压缩。
        重要性等级：C。
        """
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: int, right: int) -> None:
        """合并两个相邻三角面的并查集集合。

        参数：
            left: 左侧三角面 face ID。
            right: 右侧三角面 face ID。
        返回：
            无。
        可能异常：
            face ID 不存在时由 ``find`` 抛出 KeyError。
        副作用：
            修改 ``parent``，把右集合根挂到左集合根下。
        重要性等级：C。
        """
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    groups_by_z: dict[float, list[int]] = {}
    for face_id in upward_ids:
        groups_by_z.setdefault(float(z_round[face_id]), []).append(int(face_id))

    # 同一高度的三角形逐一检查包围盒。当前楼梯只有约 1600 个朝上面，
    # 平方复杂度足够小，同时不依赖 STL 是否复用顶点索引。
    for face_ids in groups_by_z.values():
        for index, left in enumerate(face_ids):
            for right in face_ids[index + 1 :]:
                separated_x = (
                    bbox_max[left, 0] + BBOX_TOUCH_TOLERANCE < bbox_min[right, 0]
                    or bbox_max[right, 0] + BBOX_TOUCH_TOLERANCE < bbox_min[left, 0]
                )
                separated_y = (
                    bbox_max[left, 1] + BBOX_TOUCH_TOLERANCE < bbox_min[right, 1]
                    or bbox_max[right, 1] + BBOX_TOUCH_TOLERANCE < bbox_min[left, 1]
                )
                if not separated_x and not separated_y:
                    union(left, right)

    components: dict[int, list[int]] = {}
    for face_id in upward_ids:
        components.setdefault(find(int(face_id)), []).append(int(face_id))

    candidates: list[dict[str, Any]] = []
    for face_ids in components.values():
        points = triangles[face_ids].reshape(-1, 3)
        component_area = float(area[face_ids].sum())
        x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
        y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
        z_top = float(np.mean(z_round[face_ids]))
        width_y = y_max - y_min
        if z_top <= 0.05 or component_area < MIN_SURFACE_AREA or width_y < MIN_SURFACE_WIDTH:
            continue
        candidates.append(
            {
                "z_top": z_top,
                "area": component_area,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "length_x": x_max - x_min,
                "width_y": width_y,
                "face_count": len(face_ids),
            }
        )

    # z=0.77 m 附近存在平台外沿装饰面，和 z=0.79 m 主平台重叠。
    # 只保留面积更大的主支持面，避免建立重复碰撞体。
    kept: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        overlaps_larger = False
        for existing in kept:
            if abs(candidate["z_top"] - existing["z_top"]) > 0.03:
                continue
            x_overlap = min(candidate["x_max"], existing["x_max"]) - max(
                candidate["x_min"], existing["x_min"]
            )
            y_overlap = min(candidate["y_max"], existing["y_max"]) - max(
                candidate["y_min"], existing["y_min"]
            )
            if x_overlap > 0.05 and y_overlap > 0.5:
                overlaps_larger = True
                break
        if not overlaps_larger:
            kept.append(candidate)

    for surface in kept:
        surface["center_x"] = 0.5 * (surface["x_min"] + surface["x_max"])
        surface["center_y"] = 0.5 * (surface["y_min"] + surface["y_max"])
        surface["center_z"] = surface["z_top"] - 0.5 * TREAD_THICKNESS_M
        surface["half_x"] = 0.5 * surface["length_x"]
        surface["half_y"] = 0.5 * surface["width_y"]
        surface["half_z"] = 0.5 * TREAD_THICKNESS_M

    return sorted(kept, key=lambda item: (item["z_top"], item["x_min"]))


def classify_surfaces(
    surfaces: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """按中间平台位置划分上行踏步、平台和下行踏步。

    参数：
        surfaces: ``extract_upward_surfaces`` 返回的候选面，单位米。
    返回：
        ``(up_steps, platform, down_steps)``；每个面增加稳定的 id 和中文名称。
    可能异常：
        未找到平台，或存在无法归类的宽支持面时抛出 RuntimeError。
    副作用：
        无；返回的字典是原候选面的浅拷贝扩展后的结果。
    关键假设：
        最大面积面是中间平台；上行面完全位于平台左侧，下行面完全位于右侧。
    重要性等级：A。
    """
    if not surfaces:
        raise RuntimeError("没有找到可落脚水平面，请检查 STL 姿态和法向")

    enriched = [dict(surface) for surface in surfaces]
    platform = max(enriched, key=lambda item: item["area"])
    up_steps = sorted(
        (item for item in enriched if item["x_max"] <= platform["x_min"] + 1e-4),
        key=lambda item: item["x_min"],
    )
    down_steps = sorted(
        (item for item in enriched if item["x_min"] >= platform["x_max"] - 1e-4),
        key=lambda item: item["x_min"],
    )
    unclassified = [
        item
        for item in enriched
        if item is not platform and item not in up_steps and item not in down_steps
    ]
    if unclassified:
        raise RuntimeError(f"存在无法归类的落脚面：{unclassified}")
    if len(up_steps) != 3 or len(down_steps) != 3:
        raise RuntimeError(
            f"期望 3 级上行和 3 级下行，实际 {len(up_steps)} 和 {len(down_steps)}"
        )

    for index, surface in enumerate(up_steps, start=1):
        surface["id"] = f"up_step_{index}"
        surface["name_zh"] = f"上行第 {index} 级"
    platform["id"] = "platform"
    platform["name_zh"] = "中间平台"
    for index, surface in enumerate(down_steps, start=1):
        surface["id"] = f"down_step_{index}"
        surface["name_zh"] = f"下行第 {index} 级"
    return up_steps, platform, down_steps


def json_ready(value: Any) -> Any:
    """把 NumPy 标量和数组递归转换为 JSON 可写类型。

    参数：
        value: 任意嵌套的 dict/list/tuple/NumPy 标量或数组。
    返回：
        仅含 Python 内建类型的等价结构。
    可能异常：
        自定义不可序列化对象会原样返回，调用方 ``json.dumps`` 时可能报错。
    副作用：
        无；不修改输入。
    重要性等级：B（决定几何 JSON 能否稳定写出）。
    """
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_summary(
    metadata: dict[str, Any],
    mesh: trimesh.Trimesh,
    up_steps: list[dict[str, Any]],
    platform: dict[str, Any],
    down_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """组装机器可读的几何与碰撞体摘要。

    参数：
        metadata: STL 格式和哈希信息。
        mesh: 已变换到世界坐标的网格，单位米。
        up_steps/platform/down_steps: 已分类并编号的落脚面。
    返回：
        可直接写入 JSON 的字典；所有长度单位为米。
    可能异常：
        无。
    副作用：
        无。
    数据约定：
        落脚面 box 的顶面等于 z_top、厚度等于 TREAD_THICKNESS_M，因此
        center_z=z_top-厚度/2、half_z=厚度/2；下方保持镂空。
    重要性等级：A。
    """
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    ordered = up_steps + [platform] + down_steps
    box_bounds_min = np.array(
        [
            min(item["x_min"] for item in ordered),
            min(item["y_min"] for item in ordered),
            min(item["z_top"] - TREAD_THICKNESS_M for item in ordered),
        ]
    )
    box_bounds_max = np.array(
        [
            max(item["x_max"] for item in ordered),
            max(item["y_max"] for item in ordered),
            max(item["z_top"] for item in ordered),
        ]
    )

    def public_surface(surface: dict[str, Any]) -> dict[str, Any]:
        """筛出 JSON 需要公开的踏面字段并保持稳定顺序。

        参数：
            surface: 内部踏面字典，所有公开键必须存在。
        返回：
            新字典；长度单位米，面积单位平方米。
        可能异常：
            缺少公开键时抛出 KeyError。
        副作用：
            无；不修改输入字典。
        重要性等级：B（JSON 字段是场景生成和回归的接口）。
        """
        keys = (
            "id", "name_zh", "z_top", "area", "x_min", "x_max", "y_min", "y_max",
            "length_x", "width_y", "face_count", "center_x", "center_y", "center_z",
            "half_x", "half_y", "half_z",
        )
        return {key: surface[key] for key in keys}

    up_rises = [0.0] + [item["z_top"] for item in up_steps] + [platform["z_top"]]
    down_rises = [platform["z_top"]] + [item["z_top"] for item in down_steps] + [0.0]
    return {
        "source": metadata["path"],
        "stl": metadata,
        "measurement": "trimesh 从原始 STL 解析并变换到 MuJoCo 世界坐标",
        "status": "当前按 STL 解析并用于碰撞体原型；如后续提供新版 CAD 或实测模型，再重新生成",
        "unit_assumption": "STL 原始单位推断为毫米，场景缩放 0.001 后为米",
        "scene_transform": {
            "scale": STL_SCALE,
            "rotation_axis": "X",
            "rotation_degrees": 90.0,
            "translation_m": SCENE_TRANSLATION.tolist(),
            "quaternion_wxyz": [0.70710678, 0.70710678, 0.0, 0.0],
        },
        "coordinate_system": {
            "x": "楼梯长边，低端指向平台",
            "y": "楼梯宽度",
            "z": "竖直向上",
            "go2_start_side": "低端 -X 一侧，朝 +X",
        },
        "world_visual_bounds": {
            "min": bounds[0].round(6).tolist(),
            "max": bounds[1].round(6).tolist(),
            "size": (bounds[1] - bounds[0]).round(6).tolist(),
        },
        "surface_collision_box_thickness_m": TREAD_THICKNESS_M,
        "surface_collision_union_bounds": {
            "scope": "仅 7 个落脚薄板；完整场景还包含立柱、侧梁、横撑和栏杆",
            "min": box_bounds_min.round(6).tolist(),
            "max": box_bounds_max.round(6).tolist(),
            "size": (box_bounds_max - box_bounds_min).round(6).tolist(),
        },
        "platform": public_surface(platform),
        "up_steps": [public_surface(item) for item in up_steps],
        "down_steps": [public_surface(item) for item in down_steps],
        "up_rises_m": [round(right - left, 6) for left, right in zip(up_rises, up_rises[1:])],
        "down_rises_m": [round(left - right, 6) for left, right in zip(down_rises, down_rises[1:])],
        "notes": [
            "该模型是带中间平台的对称楼梯桥，不是单跑楼梯。",
            "视觉 STL 保留显示但必须关闭碰撞；7 个独立薄板 box 承担落脚面碰撞。",
            "每块落脚薄板厚 20 mm，板下保持镂空；完整场景另有 42 个结构碰撞体。",
            "相邻踏面约 5.2 mm 的真实缝隙已确认保留，不做合并或填补。",
        ],
    }


def main() -> int:
    """执行 STL 解析、落脚面识别、JSON 写入和终端摘要。

    返回：
        0 表示成功；解析或几何断言失败时抛出异常并返回非零进程码。
    副作用：
        覆盖写入 ``models/stairs/geometry_estimate.json``，不修改原始 STL。
    重要性等级：A。
    """
    original, metadata = inspect_stl(STL_PATH)
    mesh = world_mesh(original)
    surfaces = extract_upward_surfaces(mesh)
    up_steps, platform, down_steps = classify_surfaces(surfaces)
    summary = json_ready(build_summary(metadata, mesh, up_steps, platform, down_steps))
    OUTPUT_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("楼梯 STL 解析完成")
    print(f"格式: {metadata['format']}；三角形: {metadata['triangle_count']}")
    print(f"SHA-256: {metadata['sha256']}")
    print(f"单位: 原始推断 mm，场景缩放 {STL_SCALE}")
    print(f"视觉包围盒: {summary['world_visual_bounds']}")
    print("\n可落脚面（米）")
    print("id              z_top    x_range                 y_range                 length  width")
    for surface in up_steps + [platform] + down_steps:
        print(
            f"{surface['id']:<15} {surface['z_top']:>7.4f}  "
            f"[{surface['x_min']:>7.4f}, {surface['x_max']:>7.4f}]  "
            f"[{surface['y_min']:>7.4f}, {surface['y_max']:>7.4f}]  "
            f"{surface['length_x']:>6.4f} {surface['width_y']:>6.4f}"
        )
    print(f"\n7 个落脚薄板 union 包围盒: {summary['surface_collision_union_bounds']}")
    print(f"上行级高（地面→平台）: {summary['up_rises_m']}")
    print(f"下行级高（平台→地面）: {summary['down_rises_m']}")
    print(f"已写入: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
