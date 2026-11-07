import json, io, sys
sys.stdout.reconfigure(encoding="utf-8")
d = json.load(io.open(r"experiments/b2/b2_follow_gait.json", encoding="utf-8"))
for i, s in enumerate(d["steps"], 1):
    b = s["body"]
    print(f"{i:2d} {s['leg']:<4} {s['label']:<14} body=({b[0]:.3f},{b[1]:.3f},{b[2]:.3f}) pitch={s['pitch']:5.1f}")
