"""验证单级低台阶与完整楼梯场景的碰撞几何、初始接触和短时动态接触。

文件目的：
    在不启动图形界面的情况下编译两个 MuJoCo 场景，自动检查 Go2 初始位置、
    视觉 STL 是否关闭碰撞、完整楼梯的分类碰撞体数量/包围盒、镂空净空、单向路线、四个足端的初始接触，以及单级场景中 1 秒 PD 动力学下的足端接触。
整体输入：
    ``models/stairs/geometry_estimate.json`` 和
    ``simulations/mujoco/go2_stairs_static`` 下的两个 XML 场景。
整体输出：
    终端报告；全部断言通过时返回 0，否则抛出 AssertionError 或编译错误。
依赖与版本要求：
    Python 3.13、MuJoCo 3.16、NumPy 2.x。当前验证版本为
    MuJoCo 3.16.0、NumPy 2.5.4。
运行方式：
    ``D:\\Python313\\python.exe D:\\桌面\\机器狗\\scripts\\validate_collision_scenes.py``
关键假设：
    完整楼梯仍使用官方 ``stairs_home`` 初始位置；单级场景把四足放在课程台阶顶面。
    初始接触检查使用 ``mj_forward``；单级场景额外运行 1 秒固定目标 PD，只验证短时接触与软压缩，不把结果当作步态能力。
已知限制：
    本脚本只验证初始几何和短暂动力学接触；不验证步态、上楼、姿态闭环或长期接触磨损。
重要性等级：A（场景改动后的回归检查入口）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "simulations" / "mujoco" / "go2_stairs_static"
GEOMETRY_PATH = ROOT / "models" / "stairs" / "geometry_estimate.json"
FOOT_NAMES = ("FL", "FR", "RL", "RR")
SINGLE_STEP_TEST_S = 1.0
SINGLE_STEP_JOINT_TARGET = np.array((0.0, 0.9, -1.8) * 4, dtype=np.float64)
SINGLE_STEP_KP = 60.0
SINGLE_STEP_KD = 2.0
MAX_SINGLE_STEP_CONTACT_COMPRESSION_M = 0.002
TREAD_THICKNESS_M = 0.020
TREAD_FRICTION = (0.8, 0.02, 0.01)
STRUCTURE_FRICTION = (0.6, 0.02, 0.01)
SURFACE_NAMES = (
    "up_step_1", "up_step_2", "up_step_3", "platform",
    "down_step_1", "down_step_2", "down_step_3",
)
SUPPORT_COLUMN_NAMES = (
    "support_column_left_01", "support_column_right_01",
    "support_column_left_02", "support_column_right_02",
)
SIDE_BEAM_NAMES = (
    "platform_side_beam_left", "platform_side_beam_right",
    "up_side_beam_left", "up_side_beam_right",
    "down_side_beam_left", "down_side_beam_right",
)
CROSS_BEAM_NAMES = tuple(f"platform_cross_beam_{index:02d}" for index in range(1, 5))
RAILING_PREFIXES = (
    "platform_top_rail_", "platform_mid_rail_", "platform_rail_post_",
    "up_rail_", "down_rail_",
)
EXPECTED_COLLISION_GEOMS = 49


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    """返回 MuJoCo geom 名称，供接触对和回归报告使用。

    参数：
        model: 已编译的 MuJoCo 模型。
        geom_id: 0-based geom ID，应在 ``[0, model.ngeom)`` 内。
    返回：
        geom 名称；匿名 geom 或非法 ID 返回空字符串。
    可能异常：
        无；非法 ID 由 MuJoCo 名称查询返回空值。
    副作用：
        无。
    重要性等级：C。
    """
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def body_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    """返回指定 body 直接拥有的全部 geom ID。

    参数：
        model: 已编译 MuJoCo 模型。
        body_name: 目标 body 名称。
    返回：
        geom ID 列表；body 不存在时断言失败。
    可能异常：
        body 名称不存在时抛出 AssertionError。
    副作用：
        无。
    重要性等级：A。
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    assert body_id >= 0, f"找不到 body：{body_name}"
    return [index for index in range(model.ngeom) if model.geom_bodyid[index] == body_id]


def load_keyframed_scene(xml_name: str, key_name: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """加载场景、复位指定关键帧并执行一次 mj_forward。

    参数：
        xml_name: 场景目录内的 XML 文件名。
        key_name: 必须存在的 keyframe 名称。
    返回：
        ``(model, data)``；data 已复位到关键帧。
    可能异常：
        XML 编译失败或关键帧不存在时抛出异常/AssertionError。
    副作用：
        临时切换当前工作目录到场景目录，函数结束前恢复；这是为了兼容当前
        Windows MuJoCo 构建对含中文绝对路径的限制。
    重要性等级：A。
    """
    previous_cwd = Path.cwd()
    try:
        os.chdir(SCENE_DIR)
        model = mujoco.MjModel.from_xml_path(xml_name)
        data = mujoco.MjData(model)
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        assert key_id >= 0, f"{xml_name} 缺少关键帧 {key_name}"
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        return model, data
    finally:
        os.chdir(previous_cwd)


def contact_pairs(model: mujoco.MjModel, data: mujoco.MjData) -> set[tuple[str, str]]:
    """把当前接触转换为无序 geom 名称对，便于检查指定足端接触。

    参数：
        model: 已编译模型。
        data: 已执行 ``mj_forward`` 或 ``mj_step`` 的仿真数据。
    返回：
        集合，元素为按字母排序的 ``(geom_name_1, geom_name_2)`` 二元组。
    可能异常：
        无名 geom 会以空字符串参与配对，不抛异常。
    副作用：
        只读 ``data.contact``，不修改仿真状态。
    重要性等级：B。
    """
    pairs: set[tuple[str, str]] = set()
    for contact in data.contact[: data.ncon]:
        left = geom_name(model, int(contact.geom1))
        right = geom_name(model, int(contact.geom2))
        pairs.add(tuple(sorted((left, right))))
    return pairs


def assert_visual_mesh_is_display_only(model: mujoco.MjModel) -> None:
    """确认楼梯视觉网格存在且碰撞掩码完全关闭。

    参数：
        model: 已编译场景模型。
    返回：
        无；检查通过时正常返回。
    可能异常：
        缺少几何、contype 非 0 或 conaffinity 非 0 时抛出 AssertionError。
    副作用：
        无。
    重要性等级：A。
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "steel_stair_visual")
    assert geom_id >= 0, "缺少 steel_stair_visual"
    assert model.geom_contype[geom_id] == 0, "STL 视觉网格 contype 应为 0"
    assert model.geom_conaffinity[geom_id] == 0, "STL 视觉网格 conaffinity 应为 0"


def validate_single_step() -> dict[str, float | int | list[float]]:
    """验证单级低台阶的几何、初始接触和 1 秒动力学接触。

    参数：
        无；固定读取 ``scene_single_step.xml`` 和关键帧 ``single_step_home``。
    返回：
        单级测试摘要字典；长度单位米，接触数量范围 0..4。动态压缩量定义为
        四个足端球底低于台面最大值的正数，0 表示没有几何重叠。
    可能异常：
        任何几何、位置或接触不符合预期时断言失败；XML 编译错误向上抛出。
    副作用：
        临时切换工作目录，函数返回前恢复；不写文件。
    关键参数：
        ``SINGLE_STEP_TEST_S=1.0 s`` 用于让初始离地约 11.6 mm 的四足落稳。
        ``SINGLE_STEP_KP=60``、``SINGLE_STEP_KD=2.0`` 与 B2 站立基线一致；
        调大增益会增加接触冲击，调小会使机身下沉和足端接触更软。
        最大允许动态软接触压缩为 2 mm，明显小于 22 mm 足端半径。
    关键判据：
        四足球心都在台阶顶面 XY 范围内，初始球底不低于台面 1e-6 m；
        1 秒后仍为 4/4 足端接触，且软接触压缩不超过 2 mm。
    重要性等级：A。
    """
    model, data = load_keyframed_scene("scene_single_step.xml", "single_step_home")
    assert_visual_mesh_is_display_only(model)
    collision_geoms = body_geom_ids(model, "curriculum_step_collision")
    assert len(collision_geoms) == 1, f"单级场景应有 1 个碰撞体，实际 {len(collision_geoms)}"
    step_geom = collision_geoms[0]
    assert geom_name(model, step_geom) == "curriculum_step_01"

    half_size = np.asarray(model.geom_size[step_geom], dtype=float)
    step_center = np.asarray(data.geom_xpos[step_geom], dtype=float)
    expected_center = np.array((-0.875, 0.468, 0.04))
    expected_half = np.array((0.275, 0.40, 0.04))
    np.testing.assert_allclose(step_center, expected_center, atol=1e-6)
    np.testing.assert_allclose(half_size, expected_half, atol=1e-6)
    actual_min = step_center - half_size
    actual_max = step_center + half_size
    np.testing.assert_allclose(actual_min, (-1.15, 0.068, 0.0), atol=1e-6)
    np.testing.assert_allclose(actual_max, (-0.60, 0.868, 0.08), atol=1e-6)

    np.testing.assert_allclose(data.qpos[:3], (-0.875, 0.468, 0.3683725), atol=1e-6)
    initial_base_position = data.qpos[:3].copy()
    pairs = contact_pairs(model, data)
    foot_center_z: dict[str, float] = {}
    for foot_name in FOOT_NAMES:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        assert foot_id >= 0, f"缺少足端 geom：{foot_name}"
        foot_pos = np.asarray(data.geom_xpos[foot_id], dtype=float)
        assert actual_min[0] < foot_pos[0] < actual_max[0], f"{foot_name} 不在台阶 X 范围"
        assert actual_min[1] < foot_pos[1] < actual_max[1], f"{foot_name} 不在台阶 Y 范围"
        foot_center_z[foot_name] = float(foot_pos[2])
        pair = tuple(sorted((foot_name, "curriculum_step_01")))
        assert pair in pairs, f"{foot_name} 未与课程台阶接触"

    foot_radius = float(model.geom_size[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "FL")][0])
    min_clearance = min(foot_center_z.values()) - foot_radius - float(actual_max[2])
    assert min_clearance >= -1e-6, f"单级场景初始足底穿透 {min_clearance:.6g} m"

    # 短暂动力学测试：使用与 B2 相同的固定关节目标 PD，检查软接触是否让足端
    # 明显陷入台阶。公式 tau=clip(kp*(q_target-q)-kd*qdot, ctrlrange)。
    joint_ids = np.array(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in (
                "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
                "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
                "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
                "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            )
        ],
        dtype=np.int32,
    )
    assert np.all(joint_ids >= 0), "单级场景缺少 Go2 关节"
    actuator_joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
    assert np.array_equal(actuator_joint_ids, joint_ids), "actuator 顺序与 PD 目标顺序不一致"
    qpos_adr = np.array([model.jnt_qposadr[joint_id] for joint_id in joint_ids])
    dof_adr = np.array([model.jnt_dofadr[joint_id] for joint_id in joint_ids])

    step_count = int(round(SINGLE_STEP_TEST_S / float(model.opt.timestep)))
    for _ in range(step_count):
        torque = (
            SINGLE_STEP_KP * (SINGLE_STEP_JOINT_TARGET - data.qpos[qpos_adr])
            - SINGLE_STEP_KD * data.qvel[dof_adr]
        )
        data.ctrl[:] = np.clip(
            torque,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
        )
        mujoco.mj_step(model, data)

    dynamic_top = float(data.geom_xpos[step_geom, 2] + model.geom_size[step_geom, 2])
    dynamic_clearances: dict[str, float] = {}
    for foot_name in FOOT_NAMES:
        foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, foot_name)
        radius = float(model.geom_size[foot_id, 0])
        dynamic_clearances[foot_name] = float(data.geom_xpos[foot_id, 2] - radius - dynamic_top)

    dynamic_pairs = contact_pairs(model, data)
    dynamic_contact_feet = sum(
        tuple(sorted((foot_name, "curriculum_step_01"))) in dynamic_pairs
        for foot_name in FOOT_NAMES
    )
    max_contact_compression = max(0.0, -min(dynamic_clearances.values()))
    assert dynamic_contact_feet == 4, f"1 秒后仅 {dynamic_contact_feet}/4 足端与台阶接触"
    assert max_contact_compression <= MAX_SINGLE_STEP_CONTACT_COMPRESSION_M, (
        f"单级场景动态软接触压缩 {max_contact_compression:.6g} m 超过 "
        f"{MAX_SINGLE_STEP_CONTACT_COMPRESSION_M:.6g} m"
    )

    return {
        "collision_geoms": len(collision_geoms),
        "base_position": initial_base_position.round(6).tolist(),
        "box_min": actual_min.round(6).tolist(),
        "box_max": actual_max.round(6).tolist(),
        "foot_contacts": len(FOOT_NAMES),
        "min_initial_clearance_m": min_clearance,
        "dynamic_time_s": float(data.time),
        "dynamic_base_position": data.qpos[:3].round(6).tolist(),
        "dynamic_contact_feet": dynamic_contact_feet,
        "max_contact_compression_m": max_contact_compression,
    }



def geom_world_aabb(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """计算任意 box 碰撞体经过当前旋转后的世界轴对齐包围盒。

    功能：
        把 box 的 8 个局部角点按 ``geom_xmat`` 旋转并加上 ``geom_xpos``，
        再取各轴最小值和最大值。斜侧梁和斜栏杆不能用“中心 ± half”直接算 AABB。
    参数：
        model: 已编译 MuJoCo 模型。
        data: 已执行 ``mj_forward`` 的仿真数据。
        geom_id: 0-based geom ID，必须位于 ``[0, model.ngeom)``。
    返回：
        ``(world_min, world_max)``，shape 均为 ``(3,)``，单位米，世界坐标。
    可能异常：
        geom 不是 box 时抛出 AssertionError；非法 ID 可能导致 MuJoCo 数组越界。
    副作用：
        无；只读取编译模型和仿真数据。
    关键假设：
        当前楼梯简化碰撞体全部为 box；改用 mesh/ellipsoid 时需扩展本函数。
    重要性等级：A（斜梁包围盒验证依赖它）。
    """
    assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX, (
        f"碰撞体 {geom_name(model, geom_id)} 不是 box"
    )
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    local_corners = signs * np.asarray(model.geom_size[geom_id], dtype=np.float64)
    rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    world_corners = local_corners @ rotation.T + np.asarray(data.geom_xpos[geom_id])
    return world_corners.min(axis=0), world_corners.max(axis=0)


def categorize_collision_names(names: set[str]) -> dict[str, set[str]]:
    """按用途把完整楼梯碰撞体名称分成稳定类别。

    参数：
        names: ``steel_stair_collision`` body 中全部非空 geom 名称。
    返回：
        类别到名称集合的字典；类别顺序为落脚面、支撑柱、侧梁、横撑、栏杆。
    可能异常：
        遇到未归类名称时抛出 AssertionError，迫使新增碰撞体同步登记。
    副作用：
        无。
    重要性等级：B（决定回归报告和数量断言的口径）。
    """
    surfaces = set(SURFACE_NAMES)
    support_columns = set(SUPPORT_COLUMN_NAMES)
    side_beams = set(SIDE_BEAM_NAMES)
    cross_beams = set(CROSS_BEAM_NAMES)
    railing = {name for name in names if name.startswith(RAILING_PREFIXES)}
    categories = {
        "落脚薄板": surfaces,
        "平台支撑柱": support_columns,
        "侧梁": side_beams,
        "平台横撑": cross_beams,
        "栏杆及栏杆立柱": railing,
    }
    classified = set().union(*categories.values())
    assert classified == names, f"存在未登记或缺失的碰撞体：{names ^ classified}"
    assert sum(len(items) for items in categories.values()) == len(names), "碰撞体类别发生重叠"
    return categories


def validate_full_stair() -> dict[str, object]:
    """验证完整镂空钢梯的分类碰撞体、净空、路线、初始位置和足端接触。

    功能：
        检查 7 个 20 mm 落脚薄板、4 根支撑柱、6 根侧梁、4 根平台横撑和 28 个
        栏杆/栏杆立柱碰撞件；确认踏面真实缝隙未被合并、平台中心仍可竖直贯通，
        并计算每个碰撞体的世界 AABB。最后验证 Go2 初始位姿和四足地面接触。
    参数：
        无；固定读取 ``scene.xml``、关键帧 ``stairs_home`` 和几何 JSON。
    返回：
        完整场景摘要字典，包括分类数量、镂空净空、路线、union 包围盒和逐个碰撞体 AABB。
    可能异常：
        碰撞体缺失/多出/命名重复、尺寸或摩擦不符、路线不单调、视觉网格参与碰撞、
        初始位置/足端接触异常时抛出 AssertionError；XML 编译错误向上抛出。
    副作用：
        临时切换工作目录，函数返回前恢复；只读模型，不写文件。
    关键参数：
        ``TREAD_THICKNESS_M=0.020 m`` 用于保留钢格栅踏面的薄板近似和板下镂空；
        改为从地面填实会错误阻挡平台下方空间。踏面摩擦采用 0.8/0.02/0.01；
        结构钢摩擦采用 0.6/0.02/0.01，均为未现场标定的临时默认值。
    已知限制：
        结构件采用保守简化 box，不模拟钢格栅孔洞、栏杆焊缝或柔性；静态 AABB
        通过不代表控制器已能避障或完成上下楼。
    重要性等级：A（楼梯场景结构变更后的主回归检查）。
    """
    geometry = json.loads(GEOMETRY_PATH.read_text(encoding="utf-8"))
    model, data = load_keyframed_scene("scene.xml", "stairs_home")
    assert_visual_mesh_is_display_only(model)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "steel_stair_collision")
    assert body_id >= 0, "缺少独立 body：steel_stair_collision"
    np.testing.assert_allclose(model.body_pos[body_id], (0, 0, 0), atol=1e-8)
    collision_geoms = body_geom_ids(model, "steel_stair_collision")
    assert len(collision_geoms) == EXPECTED_COLLISION_GEOMS, (
        f"完整楼梯应有 {EXPECTED_COLLISION_GEOMS} 个碰撞体，实际 {len(collision_geoms)}"
    )

    collision_names = [geom_name(model, geom_id) for geom_id in collision_geoms]
    assert all(collision_names), "完整楼梯碰撞体必须全部命名"
    assert len(set(collision_names)) == len(collision_names), "碰撞体名称存在重复"
    categories = categorize_collision_names(set(collision_names))
    expected_category_counts = {
        "落脚薄板": 7,
        "平台支撑柱": 4,
        "侧梁": 6,
        "平台横撑": 4,
        "栏杆及栏杆立柱": 28,
    }
    actual_category_counts = {name: len(items) for name, items in categories.items()}
    assert actual_category_counts == expected_category_counts, (
        f"碰撞体分类数量异常：{actual_category_counts} != {expected_category_counts}"
    )

    geom_by_name = {geom_name(model, geom_id): geom_id for geom_id in collision_geoms}
    for geom_id in collision_geoms:
        assert model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX, (
            f"{geom_name(model, geom_id)} 不是 box"
        )
        assert model.geom_contype[geom_id] != 0 or model.geom_conaffinity[geom_id] != 0, (
            f"{geom_name(model, geom_id)} 未启用碰撞"
        )

    for name in SURFACE_NAMES:
        np.testing.assert_allclose(
            model.geom_friction[geom_by_name[name]], TREAD_FRICTION, atol=1e-8,
            err_msg=f"{name} 踏面摩擦不符",
        )
    structure_names = set(collision_names) - set(SURFACE_NAMES)
    for name in structure_names:
        np.testing.assert_allclose(
            model.geom_friction[geom_by_name[name]], STRUCTURE_FRICTION, atol=1e-8,
            err_msg=f"{name} 结构钢摩擦不符",
        )

    expected_surfaces = geometry["up_steps"] + [geometry["platform"]] + geometry["down_steps"]
    aabb_by_name: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: geom_world_aabb(model, data, geom_id)
        for name, geom_id in geom_by_name.items()
    }
    surface_bounds: dict[str, dict[str, list[float]]] = {}
    for surface in expected_surfaces:
        name = surface["id"]
        actual_min, actual_max = aabb_by_name[name]
        expected_min = np.array(
            (surface["x_min"], surface["y_min"], surface["z_top"] - TREAD_THICKNESS_M)
        )
        expected_max = np.array((surface["x_max"], surface["y_max"], surface["z_top"]))
        np.testing.assert_allclose(actual_min, expected_min, atol=2e-6, err_msg=name)
        np.testing.assert_allclose(actual_max, expected_max, atol=2e-6, err_msg=name)
        actual_thickness = float(actual_max[2] - actual_min[2])
        np.testing.assert_allclose(actual_thickness, TREAD_THICKNESS_M, atol=2e-6)
        surface_bounds[name] = {
            "min": actual_min.round(6).tolist(),
            "max": actual_max.round(6).tolist(),
        }

    ordered_surfaces = expected_surfaces
    gaps = [
        float(right["x_min"] - left["x_max"])
        for left, right in zip(ordered_surfaces, ordered_surfaces[1:])
    ]
    assert all(gap > 0.0 for gap in gaps), f"踏面/平台之间的真实缝隙被合并：{gaps}"
    close_gap_indices = (0, 1, 4, 5)
    for index in close_gap_indices:
        np.testing.assert_allclose(gaps[index], 0.005262, atol=1e-5)

    platform_min, platform_max = aabb_by_name["platform"]
    open_x = 2.043433
    open_y = 0.468000
    blocking_under_platform = []
    for name, (bbox_min, bbox_max) in aabb_by_name.items():
        contains_xy = (
            bbox_min[0] <= open_x <= bbox_max[0]
            and bbox_min[1] <= open_y <= bbox_max[1]
        )
        if contains_xy and bbox_min[2] < platform_min[2] - 1e-6:
            blocking_under_platform.append(name)
    assert not blocking_under_platform, (
        f"平台中心竖直通道被碰撞体填实：{blocking_under_platform}"
    )
    open_vertical_clearance = float(platform_min[2])

    np.testing.assert_allclose(data.qpos[:3], (-0.90, 0.468, 0.27), atol=1e-6)
    np.testing.assert_allclose(data.qpos[3:7], (1, 0, 0, 0), atol=1e-8)
    pairs = contact_pairs(model, data)
    for foot_name in FOOT_NAMES:
        pair = tuple(sorted((foot_name, "floor")))
        assert pair in pairs, f"{foot_name} 初始未与地面接触"

    visual_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "steel_stair_visual")
    mesh_id = int(model.geom_dataid[visual_geom])
    vert_adr = int(model.mesh_vertadr[mesh_id])
    vert_num = int(model.mesh_vertnum[mesh_id])
    vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
    rotation = data.geom_xmat[visual_geom].reshape(3, 3)
    world_vertices = vertices @ rotation.T + data.geom_xpos[visual_geom]
    visual_min = world_vertices.min(axis=0)
    visual_max = world_vertices.max(axis=0)
    expected_visual = geometry["world_visual_bounds"]
    np.testing.assert_allclose(visual_min, expected_visual["min"], atol=2e-6)
    np.testing.assert_allclose(visual_max, expected_visual["max"], atol=2e-6)

    all_mins = np.vstack([bounds[0] for bounds in aabb_by_name.values()])
    all_maxs = np.vstack([bounds[1] for bounds in aabb_by_name.values()])
    collision_geoms_summary = [
        {
            "name": name,
            "type": "box",
            "min": aabb_by_name[name][0].round(6).tolist(),
            "max": aabb_by_name[name][1].round(6).tolist(),
        }
        for name in collision_names
    ]

    return {
        "collision_geoms": len(collision_geoms),
        "category_counts": actual_category_counts,
        "base_position": data.qpos[:3].round(6).tolist(),
        "foot_floor_contacts": len(FOOT_NAMES),
        "tread_thickness_m": TREAD_THICKNESS_M,
        "surface_gaps_m": [round(gap, 6) for gap in gaps],
        "open_vertical_clearance_m": open_vertical_clearance,
        "union_min": all_mins.min(axis=0).round(6).tolist(),
        "union_max": all_maxs.max(axis=0).round(6).tolist(),
        "visual_min": visual_min.round(6).tolist(),
        "visual_max": visual_max.round(6).tolist(),
        "surface_bounds": surface_bounds,
        "collision_geoms_detail": collision_geoms_summary,
    }


def main() -> int:
    """依次验证单级场景和完整镂空钢梯，并打印逐个碰撞体 AABB。

    参数：
        无。
    返回：
        成功时返回 0；断言失败或场景编译失败时向上抛出异常。
    可能异常：
        MuJoCo XML 编译错误、关键帧缺失、分类几何或接触断言失败。
    副作用：
        向标准输出打印报告；不写文件。
    重要性等级：A。
    """
    single = validate_single_step()
    full = validate_full_stair()

    print("=" * 78)
    print("A. 楼梯碰撞场景验证")
    print("=" * 78)
    print("\n[单级低台阶：独立课程测试，不是真实楼梯结构]")
    print(f"碰撞体数量: {single['collision_geoms']}")
    print(f"初始 base 位置: {single['base_position']}")
    print(f"台阶包围盒: min={single['box_min']}, max={single['box_max']}")
    print(f"足端—台阶接触: {single['foot_contacts']}/4（FL/FR/RL/RR）")
    print(f"初始最小足底间隙: {single['min_initial_clearance_m']:.9f} m（>= 0 表示无穿透）")
    print(f"1 s 动力学接触: {single['dynamic_contact_feet']}/4（FL/FR/RL/RR）")
    print(f"1 s 后 base 位置: {single['dynamic_base_position']}")
    print(
        f"动态最大软接触压缩: {single['max_contact_compression_m'] * 1000:.3f} mm"
        f"（判据 <= {MAX_SINGLE_STEP_CONTACT_COMPRESSION_M * 1000:.1f} mm）"
    )

    print("\n[完整镂空钢梯]")
    print(f"碰撞体总数: {full['collision_geoms']}")
    for category, count in full["category_counts"].items():
        print(f"  {category}: {count}")
    print(f"初始 base 位置: {full['base_position']}")
    print(f"足端—地面接触: {full['foot_floor_contacts']}/4（FL/FR/RL/RR）")
    print(f"踏面薄板厚度: {full['tread_thickness_m']:.3f} m")
    print(f"保留的相邻面间隙: {full['surface_gaps_m']} m")
    print(f"平台中心竖向净空: {full['open_vertical_clearance_m']:.6f} m")
    print(f"可落脚面: {len(full['surface_bounds'])} 个（3 级上行 + 平台 + 3 级下行）")
    print(f"全部碰撞体 union 包围盒: min={full['union_min']}, max={full['union_max']}")
    print(f"视觉 STL 包围盒: min={full['visual_min']}, max={full['visual_max']}")
    print("\n全部碰撞体（世界 AABB，单位 m）:")
    for item in full["collision_geoms_detail"]:
        print(f"  {item['name']:<30} type={item['type']:<4} min={item['min']} max={item['max']}")
    print("\n结果: 通过；视觉 STL 仅显示，楼梯为薄踏面/平台 + 立柱/侧梁/栏杆简化碰撞。")
    print("边界: 本验证只覆盖静态几何、初始接触和单级 1 秒短时接触，不代表已具备上下楼步态。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
