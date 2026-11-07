"""三机同踩+四拍的三帧图：起步 → 第一只脚上二级 → 失稳瞬间（3 行 × 3 列对照）。

文件目的：
    把 three_robot_gait_0272.py 的实测结果做成一眼能看懂的对照图：
    每行一款机型，三列分别是【起步踩在一级】【第一只脚踩上二级】【失稳那一刻】。
    光看数字很难体会"B2 是同踩站姿就后仰、AS2 是抬脚那下力矩先饱和"，图能直接看出来。

输入：
    simulations/mujoco/{go2,b2,as2}_stairs_static/scene.xml（同一份 0.197/0.272 钢梯）
输出：
    experiments/feasibility/three_robot_gait_0272.png（1920×1440 拼图）
    experiments/feasibility/frames/{key}_{1,2,3}.png（单帧）
依赖：mujoco 3.16（需 OpenGL 离屏渲染）、Pillow、numpy。
运行：D:\\Python313\\python.exe .\\experiments\\feasibility\\three_robot_gait_0272_frames.py

关键假设：
    - 只跑第 1 级（steps=1），因为三机都在第二拍就翻了，画多级没意义。
    - 相机固定相对机身，三张图同一视角，便于横向比较。
已知限制：
    - 是"关键时刻抽帧"，不是连续视频；要看运动过程用 --view。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT))

import mujoco  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import three_robot_gait_0272 as G  # noqa: E402

OUT = ROOT / "experiments" / "feasibility" / "three_robot_gait_0272.png"
FRAMES = ROOT / "experiments" / "feasibility" / "frames"
W, H = 640, 480


def shots(key: str) -> list[Image.Image]:
    rig = G.load_rig(key)
    fg = G.foot_gids(rig)
    segs, _notes = G.build_plan(rig, steps=1)
    if segs is None:
        return []
    model, data = rig.model, rig.data
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 1          # 碰撞体（踏面）默认不显示，打开才能看清钢梯
    renderer = mujoco.Renderer(model, H, W)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 104.0, -9.0, 2.9
    out: list[Image.Image] = []
    state = {"start": False, "peak": False, "fall": False, "lift": segs[2].lift if len(segs) > 2 else "FL"}

    def grab(tag: str) -> None:
        cam.lookat[:] = (0.56, G.BODY_Y, 0.34)
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = Image.fromarray(renderer.render())
        out.append(img)
        order = {"start": 1, "foot_up": 2, "fall": 3, "end": 4}[tag]
        img.save(FRAMES / f"{key}_{order}_{tag}.png")

    def hook(seg, u: float) -> None:
        if not state["start"] and seg.name == "稳住":
            return
        if not state["start"] and seg.name != "稳住":
            state["start"] = True
            grab("start")
        # 顺序必须固定：先"脚上去"，再"失稳"，否则拼图列标题会对不上
        if not state["peak"] and seg.lift == state["lift"] and u >= 0.85:
            state["peak"] = True
            grab("foot_up")
        if state["peak"] and not state["fall"] and G.tilt_of(data) > 28.0:
            state["fall"] = True
            grab("fall")

    G.run_plan(rig, segs, fg, hook=hook)
    if len(out) < 3:
        grab("end")
    renderer.close()
    return out[:3]


def main() -> int:
    FRAMES.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in G.ORDER:
        imgs = shots(key)
        print(f"{G.LABEL[key]}: {len(imgs)} 帧")
        rows.append((G.LABEL[key], imgs))

    pad, bar = 8, 30
    canvas = Image.new("RGB", (W * 3 + pad * 4, (H + bar) * len(rows) + pad * (len(rows) + 1)),
                       (18, 18, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(20)
    except TypeError:
        font = ImageFont.load_default()
    for r, (label, imgs) in enumerate(rows):
        y = pad + r * (H + bar + pad)
        for c, img in enumerate(imgs):
            x = pad + c * (W + pad)
            canvas.paste(img, (x, y))
        draw.text((pad + 4, y + H + 3), f"{label}   start | FL on step 2 | lost stability",
                  fill=(235, 235, 235), font=font)
    canvas.save(OUT)
    print("WROTE", OUT, canvas.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
