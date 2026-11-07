"""临时探针：入口第 1 步（FR 上第 1 级）为什么无可行位姿。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "b2"))
sys.path.insert(0, str(ROOT / "experiments" / "feasibility"))
sys.path.insert(0, str(ROOT))

import three_robot_gait_0272 as G  # noqa: E402
import b2_diag_stance as D  # noqa: E402

rig = G.load_rig("b2")
fg = G.foot_gids(rig)
gx, gz = -0.115, 0.550
feet = {"FL": (gx + 0.30, G.BODY_Y + rig.foot_y["FL"], rig.r + 0.003),
        "FR": (gx + 0.30, G.BODY_Y + rig.foot_y["FR"], rig.r + 0.003),
        "RL": (gx - 0.357, G.BODY_Y + rig.foot_y["RL"], rig.r + 0.003),
        "RR": (gx - 0.357, G.BODY_Y + rig.foot_y["RR"], rig.r + 0.003)}
new = dict(feet)
new["FR"] = (0.4466, feet["FR"][1], 0.197 + rig.r + 0.003)
sup = ("FL", "RL", "RR")
ic = G.incenter3([(new[l][0], new[l][1]) for l in sup])
print("incenter", np.round(ic, 3))

cands = D.enumerate_poses(rig, new, pitch_lo=0.0, pitch_hi=18.0, ext_max=0.95,
                          xs=np.arange(ic[0] - 0.06, ic[0] + 0.061, 0.02),
                          ys=np.arange(G.BODY_Y - 0.10, G.BODY_Y + 0.101, 0.02),
                          zs=np.arange(gz - 0.06, gz + 0.061, 0.02), support=sup)
print(f"几何可行 {len(cands)}")
gids = D.self_gids(rig, fg)
if cands:
    cl = [D.min_clear(rig, c["body"], c["joints"], gids) for c in cands]
    print(f"净空 max {max(cl)*1000:.1f} mm  min {min(cl)*1000:.1f} mm  "
          f">=30mm 的 {sum(1 for v in cl if v >= 0.030)}")
    ok = D.filter_clear(rig, cands, fg)
    print(f"净空过滤后 {len(ok)}")
    if ok:
        best = max(ok, key=lambda c: c["margin"])
        print("best", np.round(best["body"], 3), f"margin {best['margin']*1000:.0f} mm")
else:
    # 看看是卡在伸展还是可达
    for x in np.arange(ic[0] - 0.06, ic[0] + 0.061, 0.02):
        for z in np.arange(gz - 0.06, gz + 0.061, 0.02):
            for pitch in np.radians(np.arange(0, 18.1, 3.0)):
                body = (float(x), G.BODY_Y, float(z), float(pitch))
                j, ext = G.legs_ik(rig, body, new)
                if j is None:
                    tag = "IK 不可达"
                elif max(ext.values()) > 0.95:
                    tag = f"伸展 {max(ext.values()):.2f}"
                else:
                    com, gap = G.pose_metrics(rig, body, j)
                    m = G.margin_in_poly(com, [(new[l][0], new[l][1]) for l in sup])
                    tag = f"max_ext {max(ext.values()):.2f} gap {gap:.3f} margin {m*1000:.0f}mm"
                print(f"  x{x:+.2f} z{z:.2f} p{np.degrees(pitch):.0f}: {tag}")
