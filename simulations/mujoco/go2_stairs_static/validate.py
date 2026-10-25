from pathlib import Path
import os

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # MuJoCo 当前 Windows 构建无法直接打开含中文的绝对路径
XML = Path("scene.xml")

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home")
assert key_id >= 0, "缺少 stairs_home 关键帧"
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
assert base_id >= 0, "缺少 Go2 base body"
assert model.nq == 19 and model.nv == 18 and model.nu == 12, "Go2 自由度或执行器数量不符合预期"

stair_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "steel_stair")
assert stair_body >= 0, "缺少楼梯 body"
np.testing.assert_allclose(model.body_quat[stair_body], (2**-0.5, 2**-0.5, 0, 0), atol=1e-6)

stair_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "steel_stair_visual")
assert stair_geom >= 0, "缺少楼梯几何体"
assert model.geom_contype[stair_geom] == 0, "楼梯当前应仅用于显示"
assert model.geom_conaffinity[stair_geom] == 0, "楼梯当前应仅用于显示"

foot_names = ("FL", "FR", "RL", "RR")
for name in foot_names:
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0, f"缺少脚部碰撞体 {name}"

floor_contacts = 0
for contact in data.contact[:data.ncon]:
    geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    if "floor" in (geom1, geom2):
        floor_contacts += 1

# 用编译后的世界坐标网格顶点检查：长边沿 X、宽度沿 Y、高度沿 Z，且最低点贴地。
mesh_id = model.geom_dataid[stair_geom]
vert_adr = model.mesh_vertadr[mesh_id]
vert_num = model.mesh_vertnum[mesh_id]
mesh_vertices = model.mesh_vert[vert_adr:vert_adr + vert_num]
rotation = data.geom_xmat[stair_geom].reshape(3, 3)
world_vertices = mesh_vertices @ rotation.T + data.geom_xpos[stair_geom]
stair_min = world_vertices.min(axis=0)
stair_max = world_vertices.max(axis=0)
stair_size = stair_max - stair_min
np.testing.assert_allclose(stair_size, (4.0868662, 0.936, 1.40685), atol=1e-4)
assert abs(stair_min[2]) < 1e-5, "楼梯最低点没有贴在地面上"
np.testing.assert_allclose(data.xpos[base_id][:2], (-0.90, 0.468), atol=1e-4)

print(f"MuJoCo: {mujoco.__version__}")
print(f"nq/nv/nu: {model.nq}/{model.nv}/{model.nu}")
print(f"geoms/meshes: {model.ngeom}/{model.nmesh}")
print(f"keyframes: {model.nkey} (using stairs_home)")
print(f"base position: {data.xpos[base_id]}")
print(f"stair world min: {stair_min}")
print(f"stair world max: {stair_max}")
print(f"stair world size (x/y/z): {stair_size}")
print(f"initial floor contacts: {floor_contacts}")
print("compile + keyframe reset + mj_forward: OK")
print("Note: the official Go2 keyframe is only an initial pose; static display is intentional without a gait controller.")


