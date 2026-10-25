from pathlib import Path
import time

import mujoco
import mujoco.viewer

# 模型与脚本位于同一目录，直接按 XML 加载 STL。
xml_path = Path(__file__).with_name("steel_stair.xml")
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    # 初始视角：楼梯长边沿 Y 轴，从入口端斜向观察。
    viewer.cam.lookat[:] = [0.0, 1.95, 0.52]
    viewer.cam.distance = 3.9
    viewer.cam.azimuth = 180.0
    viewer.cam.elevation = -30.0

    print("MuJoCo viewer started. Close the window to exit.")
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(1.0 / 60.0)
