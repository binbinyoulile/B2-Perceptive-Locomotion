from pathlib import Path
import os
import time

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent
os.chdir(HERE)  # MuJoCo 当前 Windows 构建无法直接打开含中文的绝对路径
XML = Path("scene.xml")

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)

key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stairs_home")
if key_id < 0:
    raise RuntimeError("scene.xml 中找不到 stairs_home 关键帧")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

print("Go2 + steel stair loaded.")
print("静态展示模式：当前阶段不运行步态控制器，模型保持初始站姿。")
print("鼠标左键旋转，右键平移，滚轮缩放；关闭窗口退出。")

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.lookat[:] = [1.70, 0.468, 0.70]
    viewer.cam.distance = 4.8
    viewer.cam.azimuth = 128.0
    viewer.cam.elevation = -18.0

    while viewer.is_running():
        # 只刷新场景，不调用 mj_step，避免无控制器时狗因重力倒下。
        mujoco.mj_forward(model, data)
        viewer.sync()
        time.sleep(1.0 / 60.0)
