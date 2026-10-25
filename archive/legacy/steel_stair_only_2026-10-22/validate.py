from pathlib import Path
import mujoco

xml = Path(__file__).with_name("steel_stair.xml")
model = mujoco.MjModel.from_xml_path(str(xml))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

print(f"MuJoCo {mujoco.__version__}")
print(f"mesh assets: {model.nmesh}")
print(f"geoms: {model.ngeom}")
print(f"mesh vertices: {model.mesh_vert.shape[0]}")
print(f"mesh triangles: {model.mesh_face.shape[0]}")
print("compile + forward: OK")
