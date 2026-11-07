"""三机同台：Go2 / B2 / AS2 并排站在真实尺寸钢梯前，PD 静置 + 3D 截图 + 体量对比。

文件目的：
    现场实测钢梯（级高 0.197 m、踏面深 0.272 m、宽 0.81 m）后，需要一眼看清三台狗
    在这个梯子前的体量、机头高度、腿长差异。做法是把三个官方 MJCF 用 MjSpec.attach
    拼进同一个场景（各自的网格、关节、执行器都带前缀），同时用各自标定过的 PD 增益
    让它们并排站住，再渲染一张图。

输入：
    simulations/mujoco/steel_stair/scene_only.xml（只有钢梯和地面，无机器人）
    models/unitree_go2/go2.xml、models/unitree_b2_mujoco/xml/b2.xml、
    models/unitree_as2_mujoco/xml/as2.xml
输出：
    experiments/feasibility/multi_robot_compare.png
    experiments/feasibility/multi_robot_compare.txt
依赖：
    mujoco 3.16（MjSpec.attach）、numpy
运行：
    D:\\Python313\\python.exe .\\experiments\\feasibility\\multi_robot_compare.py [--stand]

关键假设：
    - 三台狗用各自标定的 PD 增益和站立关节角（Go2 130/3.5、B2 800/20、AS2 400/12），
      这不是"同一控制器"的比较，而是"各自正常站着"的体量对比。
    - attach 时父 spec 的 impratio/cone 会压过子 spec（MuJoCo 的 attach 冲突策略），
      本文只做站立对比，不影响结论。

已知限制：
    - 每个机器人各自的 `<option>`（timestep/integrator）被父场景的覆盖，统一用 0.002 s。
    - 只比体量与机头高度，不比爬梯能力（那是 stair_0272_limits.py 的事）。
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "simulations" / "mujoco" / "steel_stair" / "scene_only.xml"
OUT = ROOT / "experiments" / "feasibility" / "multi_robot_compare"

# 三个机器人：前缀、根 body、出生点 y、站立关节角（hip/thigh/calf）、PD 增益
ROBOTS = (
    dict(name="Go2", prefix="g_", base="base", y=0.10, drop=0.28,
         xml="models/unitree_go2/go2.xml", stand=(0.0, 0.9, -1.8), kp=130.0, kd=3.5),
    dict(name="B2", prefix="b_", base="base_link", y=0.468, drop=0.425,
         xml="models/unitree_b2_mujoco/xml/b2.xml", stand=(0.0, 0.9649, -1.9298),
         kp=800.0, kd=20.0),
    dict(name="AS2", prefix="a_", base="base_link", y=0.92, drop=0.345,
         xml="models/unitree_as2_mujoco/xml/as2.xml", stand=(0.0, 0.7906, -1.563),
         kp=400.0, kd=12.0),
)
LEGS = ("FL", "FR", "RL", "RR")


def load_spec(path: Path):
    """MuJoCo 打不开含中文的绝对路径，统一 chdir 到文件所在目录再用文件名。"""
    import mujoco

    previous = Path.cwd()
    os.chdir(path.parent)
    try:
        return mujoco.MjSpec.from_file(path.name)
    finally:
        os.chdir(previous)


def rebase_assets(spec, model_path: Path, base_dir: Path) -> None:
    """把子模型里相对的网格/贴图路径改成"相对 base_dir"的路径。

    合并三个模型时，编译发生在 base_dir（steel_stair）下，子模型的 meshdir 若不重写就会
    指向错误位置；又不能写成中文绝对路径（MuJoCo 打不开）。
    """
    d = model_path.parent.resolve()
    meshdir = spec.meshdir or ""
    for m in spec.meshes:
        m.file = os.path.relpath((d / meshdir / m.file).resolve(), base_dir)
    spec.meshdir = ""
    texdir = spec.texturedir or ""
    for t in spec.textures:
        if t.file:
            t.file = os.path.relpath((d / texdir / t.file).resolve(), base_dir)
    spec.texturedir = ""


def main() -> int:
    import mujoco
    from PIL import Image

    base_dir = SCENE.parent.resolve()
    spec = load_spec(SCENE)
    frame = spec.worldbody.add_frame(pos=[0, 0, 0])
    for r in ROBOTS:
        child = load_spec(ROOT / r["xml"])
        rebase_assets(child, ROOT / r["xml"], base_dir)
        spec.attach(child, prefix=r["prefix"], frame=frame)
    previous = Path.cwd()
    os.chdir(base_dir)
    try:
        model = spec.compile()
    finally:
        os.chdir(previous)
    data = mujoco.MjData(model)

    lines: list[str] = []

    def say(t: str = "") -> None:
        lines.append(t)

    # ---- 放到楼梯前并排站好：每台狗的自由关节各自设位置/四元数/站立角 ----
    for r in ROBOTS:
        body = model.body(r["prefix"] + r["base"])
        # 自由关节在 attach 后名字为空，只能按"挂在根 body 上 + 类型是 free"来定位
        free = [model.joint(j) for j in range(model.njnt)
                if model.joint(j).type == mujoco.mjtJoint.mjJNT_FREE
                and int(model.jnt_bodyid[j]) == body.id][0]
        adr = int(np.asarray(free.qposadr).ravel()[0])
        data.qpos[adr:adr + 3] = [-0.55, r["y"], r["drop"]]
        data.qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
        for i, leg in enumerate(LEGS):
            for k, part in enumerate(("hip", "thigh", "calf")):
                j = model.joint(f"{r['prefix']}{leg}_{part}_joint")
                data.qpos[j.qposadr] = r["stand"][k]
    mujoco.mj_forward(model, data)

    # ---- PD 静置 2 s ----
    targets = {}
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        name = model.joint(j).name
        prefix = next(r["prefix"] for r in ROBOTS if name.startswith(r["prefix"]))
        r = next(x for x in ROBOTS if x["prefix"] == prefix)
        rest = name[len(prefix):]
        leg, part = rest.split("_")[0], rest.split("_")[1]
        targets[a] = r["stand"][("hip", "thigh", "calf").index(part)]
    gains = {}
    for r in ROBOTS:
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            if model.joint(j).name.startswith(r["prefix"]):
                gains[a] = (r["kp"], r["kd"])

    for _ in range(100):  # 2 s @ 50 Hz
        for a in range(model.nu):
            j = int(model.actuator_trnid[a, 0])
            dof = int(model.jnt_dofadr[j])
            q = data.qpos[model.jnt_qposadr[j]]
            kp, kd = gains[a]
            tau = kp * (targets[a] - q) - kd * data.qvel[dof] + data.qfrc_bias[dof]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = min(hi, max(lo, tau))
        for _ in range(10):
            mujoco.mj_step(model, data)

    # ---- 统计 ----
    say("=" * 88)
    say("三机同台：真实钢梯 0.197 m 级高 / 0.272 m 踏面 / 0.81 m 宽")
    say("=" * 88)
    say(f"{'机型':<6}{'机身 z(m)':>10}{'总高(m)':>10}{'质量(kg)':>10}{'髋距(m)':>9}{'腿长(m)':>9}")
    for r in ROBOTS:
        body = model.body(r["prefix"] + r["base"])
        z = float(data.xpos[body.id][2])
        # 总高：机器人子树里所有可视 geom 的最高点（近似）
        top = 0.0
        for g in range(model.ngeom):
            if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
                if model.body_rootid[model.geom_bodyid[g]] == body.id:
                    top = max(top, float(data.geom_xpos[g][2] + model.geom_rbound[g]))
        mass = float(np.sum([model.body_mass[i] for i in range(model.nbody)
                             if model.body_rootid[i] == body.id]))
        thigh = float(abs(model.body(r["prefix"] + "FL_calf").pos[2]))
        hip = float(model.body(r["prefix"] + "FL_hip").pos[0])
        say(f"{r['name']:<6}{z:>10.3f}{top:>10.3f}{mass:>10.2f}{2*abs(hip):>9.3f}{thigh:>9.3f}")

    # ---- 渲染 ----
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1  # 碰撞体（踏面）默认不显示，打开才能看清钢梯
    renderer = mujoco.Renderer(model, 720, 1280)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (-0.15, 0.51, 0.42)
    cam.distance = 3.1
    cam.azimuth = 0.0
    cam.elevation = -20.0
    renderer.update_scene(data, camera=cam, scene_option=opt)
    png = OUT.with_suffix(".png")
    Image.fromarray(renderer.render()).save(png)
    renderer.close()
    say()
    say("已渲染 " + str(png))

    text = "\n".join(lines)
    OUT.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print("WROTE", OUT.with_suffix(".txt"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
