"""从可选路径的楼梯 STL 中提取可落脚平面，并输出几何估计 JSON。"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "simulations" / "mujoco" / "go2_stairs_static"
OUTPUT = ROOT / "models" / "stairs" / "geometry_estimate.json"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def upward_components(model: mujoco.MjModel, data: mujoco.MjData) -> list[dict]:
    geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "steel_stair_visual"
    )
    if geom_id < 0:
        raise RuntimeError("场景中找不到 steel_stair_visual")

    mesh_id = int(model.geom_dataid[geom_id])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    face_adr = int(model.mesh_faceadr[mesh_id])
    face_num = int(model.mesh_facenum[mesh_id])

    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
    faces = model.mesh_face[face_adr : face_adr + face_num]
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    world_vertices = vertices @ rotation.T + data.geom_xpos[geom_id]
    triangles = world_vertices[faces]

    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    normal = np.divide(
        cross,
        2.0 * area[:, None],
        out=np.zeros_like(cross),
        where=area[:, None] > 1e-12,
    )
    horizontal_up = (normal[:, 2] > 0.9999) & (area > 1e-8)
    plane_z = np.round(triangles[:, :, 2].mean(axis=1), 4)

    candidates: list[dict] = []
    for z_value in sorted(set(plane_z[horizontal_up])):
        face_ids = np.where(horizontal_up & (plane_z == z_value))[0].astype(int)
        total_area = float(area[face_ids].sum())
        if total_area < 0.1:
            continue

        parent = {face_id: face_id for face_id in face_ids}

        def find(item: int) -> int:
            while parent[item] != item:
                parent[item] = parent[parent[item]]
                item = parent[item]
            return item

        def union(left: int, right: int) -> None:
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left

        edges: dict[tuple[int, int], list[int]] = defaultdict(list)
        for face_id in face_ids:
            a, b, c = (int(value) for value in faces[face_id])
            for edge in ((a, b), (b, c), (c, a)):
                edges[tuple(sorted(edge))].append(face_id)
        for connected_faces in edges.values():
            for other in connected_faces[1:]:
                union(connected_faces[0], other)

        groups: dict[int, list[int]] = defaultdict(list)
        for face_id in face_ids:
            groups[find(face_id)].append(face_id)

        for group_faces in groups.values():
            points = triangles[group_faces].reshape(-1, 3)
            component_area = float(area[group_faces].sum())
            x_min, x_max = float(points[:, 0].min()), float(points[:, 0].max())
            y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
            if component_area >= 0.1 and (y_max - y_min) >= 0.5:
                candidates.append(
                    {
                        "z": float(z_value),
                        "area": component_area,
                        "x_min": x_min,
                        "x_max": x_max,
                        "y_min": y_min,
                        "y_max": y_max,
                        "face_count": len(group_faces),
                    }
                )

    if not candidates:
        raise RuntimeError("没有找到可落脚的水平面，请检查楼梯姿态")

    # 同一高度附近的重叠装饰面只保留面积较大的主落脚面。
    kept: list[dict] = []
    for candidate in sorted(candidates, key=lambda item: item["area"], reverse=True):
        overlaps_larger_surface = False
        for existing in kept:
            if abs(candidate["z"] - existing["z"]) > 0.03:
                continue
            x_overlap = min(candidate["x_max"], existing["x_max"]) - max(
                candidate["x_min"], existing["x_min"]
            )
            y_overlap = min(candidate["y_max"], existing["y_max"]) - max(
                candidate["y_min"], existing["y_min"]
            )
            if x_overlap > 0.05 and y_overlap > 0.5:
                overlaps_larger_surface = True
                break
        if not overlaps_larger_surface:
            kept.append(candidate)

    return sorted(kept, key=lambda item: (item["z"], item["x_min"]))


old_cwd = Path.cwd()
try:
    os.chdir(SCENE_DIR)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
    surfaces = upward_components(model, data)
finally:
    os.chdir(old_cwd)

platform = max(surfaces, key=lambda item: item["area"])
platform_mid_x = 0.5 * (platform["x_min"] + platform["x_max"])
up_steps = sorted(
    (surface for surface in surfaces if surface["x_max"] < platform_mid_x),
    key=lambda item: item["x_min"],
)
down_steps = sorted(
    (surface for surface in surfaces if surface["x_min"] > platform_mid_x),
    key=lambda item: item["x_min"],
)

for index, surface in enumerate(up_steps + [platform] + down_steps, start=1):
    surface["id"] = f"surface_{index}"
    surface["length_x"] = round(surface["x_max"] - surface["x_min"], 6)
    surface["width_y"] = round(surface["y_max"] - surface["y_min"], 6)

up_profile = [0.0] + [surface["z"] for surface in up_steps] + [platform["z"]]
up_rises = [round(b - a, 6) for a, b in zip(up_profile, up_profile[1:])]
down_profile = [platform["z"]] + [surface["z"] for surface in down_steps] + [0.0]
down_rises = [round(a - b, 6) for a, b in zip(down_profile, down_profile[1:])]

summary = {
    "source": "models/stairs/steel_stair.STL",
    "measurement": "由 MuJoCo 编译后的世界坐标网格自动提取",
    "status": "几何估计，用于后续碰撞体建模；最终尺寸应以 CAD 图纸或实测为准",
    "coordinate_system": {
        "x": "楼梯长边，低端指向平台",
        "y": "楼梯宽度",
        "z": "竖直向上",
        "go2_start_side": "低端 -X 一侧，朝 +X",
    },
    "platform": platform,
    "up_steps": up_steps,
    "down_steps": down_steps,
    "up_rises_m": up_rises,
    "down_rises_m": down_rises,
    "notes": [
        "该模型是带中间平台的对称楼梯桥，不是单跑楼梯。",
        "MuJoCo 场景中楼梯只作为视觉网格，尚未添加碰撞体。",
        "后续碰撞体建议按这些水平面建立简化长方体，不要直接启用整体 STL 碰撞。",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print("楼梯可落脚面（单位：米）")
print("id              z_top    x_range                 y_range                 length  width")
for surface in up_steps + [platform] + down_steps:
    print(
        f"{surface['id']:<15} {surface['z']:>7.4f}  "
        f"[{surface['x_min']:>7.4f}, {surface['x_max']:>7.4f}]  "
        f"[{surface['y_min']:>7.4f}, {surface['y_max']:>7.4f}]  "
        f"{surface['length_x']:>6.4f} {surface['width_y']:>6.4f}"
    )
print()
print(f"上行级高序列（地面→平台）: {up_rises}")
print(f"下行级高序列（平台→地面）: {down_rises}")
print(f"已写入: {OUTPUT}")