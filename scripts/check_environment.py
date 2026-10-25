"""检查 Go2 项目的基础环境、模型和静态场景。"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCENE_DIR = ROOT / "simulations" / "mujoco" / "go2_stairs_static"
GO2_DIR = ROOT / "models" / "unitree_go2"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

failures: list[str] = []


def ok(message: str) -> None:
    print(f"[OK]   {message}")


def fail(message: str) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


print("=" * 66)
print("宇树 Go2 机器狗项目 - 环境与静态场景检查")
print("=" * 66)
info(f"项目根目录: {ROOT}")
info(f"Python: {sys.version.split()[0]} ({sys.executable})")

if sys.version_info >= (3, 10):
    ok("Python 版本满足当前 MuJoCo 脚本要求")
else:
    fail("建议使用 Python 3.10 或更高版本")

required_paths = (
    ROOT / "README.md",
    ROOT / "docs" / "PROJECT_PLAN.md",
    SCENE_DIR / "scene.xml",
    SCENE_DIR / "go2.xml",
    SCENE_DIR / "assets" / "steel_stair.STL",
    GO2_DIR / "go2.xml",
    GO2_DIR / "LICENSE",
)
for path in required_paths:
    if path.exists():
        ok(f"文件存在: {path.relative_to(ROOT)}")
    else:
        fail(f"缺少文件: {path}")

try:
    import mujoco
    import numpy as np
except Exception as exc:  # noqa: BLE001 - 环境检查需要显示原始错误
    fail(f"无法导入 MuJoCo/NumPy: {exc}")
    print("\n请先按 requirements-static.txt 安装基础依赖。")
    raise SystemExit(1)

ok(f"MuJoCo 可导入，版本 {mujoco.__version__}")
ok(f"NumPy 可导入，版本 {np.__version__}")

if package_version("Pillow") == "未安装":
    fail("Pillow 未安装，无法生成场景预览")
else:
    ok(f"Pillow 已安装，版本 {package_version('Pillow')}")

# 官方 Go2 模型编译检查。先切换到模型目录并使用相对 XML 路径。
old_cwd = Path.cwd()
try:
    os.chdir(GO2_DIR)
    go2_model = mujoco.MjModel.from_xml_path("go2.xml")
    if (go2_model.nq, go2_model.nv, go2_model.nu) == (19, 18, 12):
        ok("官方 Go2 MJCF 编译通过: nq/nv/nu = 19/18/12")
    else:
        fail(
            "官方 Go2 MJCF 的自由度或执行器数量异常: "
            f"nq/nv/nu = {go2_model.nq}/{go2_model.nv}/{go2_model.nu}"
        )
except Exception as exc:  # noqa: BLE001
    fail(f"官方 Go2 MJCF 编译失败: {exc}")

# Go2 + 楼梯静态场景检查。
try:
    os.chdir(SCENE_DIR)
    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    if (model.nq, model.nv, model.nu) == (19, 18, 12):
        ok("Go2 + 楼梯场景编译通过: nq/nv/nu = 19/18/12")
    else:
        fail(
            "静态场景自由度或执行器数量异常: "
            f"nq/nv/nu = {model.nq}/{model.nv}/{model.nu}"
        )

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home")
    if key_id < 0:
        fail("静态场景缺少 stairs_home 关键帧")
    else:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        ok("stairs_home 关键帧复位与 mj_forward 通过")

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    if base_id < 0:
        fail("静态场景缺少 Go2 base body")
    else:
        base_position = data.xpos[base_id].copy()
        if np.allclose(base_position[:2], (-0.90, 0.468), atol=1e-4):
            ok(f"Go2 初始位置正确: {base_position.round(4).tolist()}")
        else:
            fail(f"Go2 初始位置异常: {base_position.round(4).tolist()}")

    stair_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "steel_stair")
    if stair_body < 0:
        fail("静态场景缺少 steel_stair body")
    else:
        expected_quat = np.array((2**-0.5, 2**-0.5, 0.0, 0.0))
        if np.allclose(model.body_quat[stair_body], expected_quat, atol=1e-6):
            ok("楼梯姿态正确: 绕 X 轴 +90 度，正向立在地面")
        else:
            fail(f"楼梯姿态异常: quat={model.body_quat[stair_body].tolist()}")

    stair_geom = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "steel_stair_visual"
    )
    if stair_geom < 0:
        fail("静态场景缺少 steel_stair_visual 几何体")
    elif model.geom_contype[stair_geom] != 0 or model.geom_conaffinity[stair_geom] != 0:
        fail("楼梯显示网格不应开启碰撞")
    else:
        ok("楼梯当前为仅显示网格，未开启碰撞")

    foot_names = ("FL", "FR", "RL", "RR")
    missing_feet = [
        name
        for name in foot_names
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0
    ]
    if missing_feet:
        fail(f"缺少脚部碰撞体: {', '.join(missing_feet)}")
    else:
        ok("四个 Go2 足端碰撞体齐全: FL/FR/RL/RR")

    floor_contacts = 0
    for contact in data.contact[: data.ncon]:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        if "floor" in (geom1, geom2):
            floor_contacts += 1
    if floor_contacts == 4:
        ok("初始状态下四个足端与地面接触")
    else:
        fail(f"初始地面接触数量异常: {floor_contacts}（预期 4）")

    if stair_geom >= 0:
        mesh_id = int(model.geom_dataid[stair_geom])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        mesh_vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
        rotation = data.geom_xmat[stair_geom].reshape(3, 3)
        world_vertices = mesh_vertices @ rotation.T + data.geom_xpos[stair_geom]
        stair_min = world_vertices.min(axis=0)
        stair_max = world_vertices.max(axis=0)
        stair_size = stair_max - stair_min
        expected_size = np.array((4.0868662, 0.936, 1.40685))
        if np.allclose(stair_size, expected_size, atol=1e-4):
            ok(f"楼梯世界尺寸正确: {stair_size.round(6).tolist()}")
        else:
            fail(f"楼梯世界尺寸异常: {stair_size.round(6).tolist()}")
        if abs(float(stair_min[2])) < 1e-5:
            ok("楼梯最低点贴地: z = 0")
        else:
            fail(f"楼梯最低点未贴地: z = {float(stair_min[2]):.6f}")

    info(f"场景统计: geoms={model.ngeom}, meshes={model.nmesh}, keyframes={model.nkey}")
except Exception as exc:  # noqa: BLE001
    fail(f"Go2 + 楼梯静态场景检查失败: {exc}")
finally:
    os.chdir(old_cwd)

print("\n可选强化学习依赖:")
optional_packages = (
    ("Gymnasium", "gymnasium"),
    ("PyTorch", "torch"),
    ("Stable-Baselines3", "stable_baselines3"),
    ("JAX", "jax"),
)
for label, module in optional_packages:
    version = package_version(module.replace("_", "-"))
    state = f"已安装 {version}" if version != "未安装" else "未安装"
    info(f"{label}: {state}")
info(f"MuJoCo MJX 模块: {'可用' if module_available('mujoco.mjx') else '不可用'}")

print("\n" + "=" * 66)
if failures:
    print(f"检查完成：{len(failures)} 项失败")
    for message in failures:
        print(f"  - {message}")
    raise SystemExit(1)

print("检查完成：基础环境和静态场景全部通过")
print("注意：这里验证的是静态展示场景，不代表已经具备站立、步态或爬梯控制能力。")