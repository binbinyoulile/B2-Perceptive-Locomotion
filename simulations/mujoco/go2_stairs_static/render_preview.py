from pathlib import Path
import os

import mujoco
from PIL import Image

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # MuJoCo 当前 Windows 构建无法直接打开含中文的绝对路径
XML = Path("scene.xml")
OUT = HERE / "preview.png"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model, height=720, width=1280)
camera = mujoco.MjvCamera()
camera.lookat[:] = [1.70, 0.468, 0.70]
camera.distance = 4.8
camera.azimuth = 128.0
camera.elevation = -18.0
renderer.update_scene(data, camera)
Image.fromarray(renderer.render()).save(OUT)
renderer.close()
print(f"saved: {OUT}")
