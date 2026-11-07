import re, io, sys
p = r"experiments/b2/b2_follow_gait.py"
s = io.open(p, encoding="utf-8").read()
s = s.replace("else load_plan(cycles, base))", "else load_plan(cycles, base, entry))")
s = s.replace("save_plan(segs, steps, cycles, base)", "save_plan(segs, steps, cycles, base, entry)")
io.open(p, "w", encoding="utf-8", newline="").write(s)
print("patched")
