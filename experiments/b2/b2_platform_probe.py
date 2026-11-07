"""临时探针：B2 四足全部站在平台（第 4 级，顶 0.788，后沿 x=1.0326）上，有没有静态可行位姿。

只回答一个问题：落点选在平台深走面上时，(x,y,z,pitch) 里能不能搜到满足
"四腿可达 + 伸展<95% + 非足端碰撞体离梯级>=30mm + 重心在支撑多边形内"的位姿。
运行：D:\\Python313\\python.exe .\\experiments\\b2\\b2_platform_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "b2"))
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))
sys.path.insert(0, str(ROOT))

import three_robot_gait_0272 as G  # noqa: E402
import b2_diag_stance as D  # noqa: E402

PLAT_REAR = 0.2166 + 0.272 * 3          # 1.0326
PLAT_TOP = 0.788
PRELOAD = 0.003


def feet_on_platform(rig, x_front: float, x_rear: float) -> dict:
    z = PLAT_TOP + rig.r + PRELOAD
    return {leg: ((x_front if leg in ("FL", "FR") else x_rear),
                  G.BODY_Y + rig.foot_y[leg], z) for leg in G.LEGS}


def main() -> int:
    rig = G.load_rig("b2")
    fg = G.foot_gids(rig)
    print(f"r={rig.r:.4f} foot_y={ {k: round(v,4) for k,v in rig.foot_y.items()} }")
    print(f"平台后沿 x={PLAT_REAR:.4f} 顶 {PLAT_TOP}  足心目标 z={PLAT_TOP + rig.r + PRELOAD:.4f}")
    for span in ((0.34, 0.10), (0.40, 0.08), (0.30, 0.06), (0.45, 0.05)):
        feet = feet_on_platform(rig, PLAT_REAR + span[0], PLAT_REAR + span[1])
        cands = D.enumerate_poses(rig, feet, pitch_lo=0.0, pitch_hi=20.0, ext_max=0.95,
                                  xs=np.arange(1.00, 1.62, 0.02),
                                  ys=np.arange(G.BODY_Y - 0.08, G.BODY_Y + 0.081, 0.02),
                                  zs=np.arange(1.02, 1.42, 0.02))
        ok = D.filter_clear(rig, cands, fg)
        tag = f"前足 x+{span[0]:.2f} / 后足 x+{span[1]:.2f}（跨 {span[0]-span[1]:.2f} m）"
        if not ok:
            print(f"  {tag}: 可行 0（几何候选 {len(cands)}，净空过滤后 0）")
            continue
        best = max(ok, key=lambda c: c["margin"] - 0.02 * abs(c["body"][3]))
        b = best["body"]
        print(f"  {tag}: 可行 {len(ok):4d}  最优 机身 ({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) "
              f"pitch {np.degrees(b[3]):+5.1f}°  余量 {best['margin']*1000:5.1f} mm  "
              f"伸展 {best['max_ext']*100:3.0f}%  净空 {best['leg_clear']*1000:4.1f} mm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
