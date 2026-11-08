# 机器狗：B2 爬钢梯

![version](https://img.shields.io/badge/version-v1.0-blue?style=flat-square)
![MuJoCo](https://img.shields.io/badge/MuJoCo-3.x-orange?style=flat-square)

基于 MuJoCo 的核应急四足机器人（Unitree B2）感知运动控制，实现从平地起步到 20 步攀爬高架钢梯的 FSM 与姿态伺服闭环。

本项目依托上海交大校企联培项目背景，聚焦核应急场景。

> **当前状态（2026-11-08）**：B2 从**平地正常站姿起步，20 步全程爬上钢梯平台（0.788 m）**，
> 四足都落在平台上、全程未失稳。基线标签 `v1.0-FSM-20Steps-Success`。
> 这是 MuJoCo 仿真结果；真机尚未标定。旧版本 README 写的"Go2 尚未登顶"属于更早的 Go2 阶段，见文末「历史阶段」。

## 最快看到结果

```powershell
cd "D:\桌面\机器狗"
D:\Python313\python.exe .\experiments\b2\b2_long_stair.py --view      # 任意级数：8 级爬到顶（56 步，约 3 min）
D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py --demo --view   # 完整作业流（36 步，约 2.5 min）
D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py --view         # 只播 20 步爬梯：1.4 s 起播、约 91 s
D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py --demo         # 完整作业流跑数据：约 7 s（读缓存）
```

**任意级数（`b2_long_stair.py`）**：场景由参数生成（`StairSpec`，级数任意），一级一个 `CLIMB_ONE_STEP`
（4 步），主循环每轮从**实测足端高度**读当前级号、再规划这一级，直到"再往上一级是平台"。
8 级实测 **56/56 步、全程未失稳**：四足都站上 `platform`（顶 1.773 m），机身 (3.259, 0.428, 2.193)、
pitch 0°；峰值力矩 196%（FL_thigh 抬脚瞬间）、最大倾角 29°。详见 [`docs/B2_LONG_STAIR.md`](docs/B2_LONG_STAIR.md)。

**完整作业流**（`--demo`）四段连播、中间不停：
**平地走近 8 步**（机身 x −0.715 → −0.115，步幅 0.30）→ **按实测位姿重规划入口 6 步** → **爬梯 14 步** →
**登顶后边走边站直 + 平台上再走 8 步**（x 1.86、z 1.258、pitch 0°）。实测 **36/36 步、全程未失稳**，
末态四足都在 `platform` 上，峰值力矩 128%（FL_thigh，爬梯段）。

只跑爬梯段（20 步，无走近/平台段）：**20/20 步**、末态四足接触 `platform` 164/158/178/176 N、
机身 0.680 → 1.198 m、末态重心余量 183 mm，峰值力矩 187%（FL_thigh 瞬态）。
完整数据与失败档案见 [`docs/B2_PLATFORM_CLIMB.md`](docs/B2_PLATFORM_CLIMB.md)，文档地图见 [`docs/INDEX.md`](docs/INDEX.md)。

## 步态一句话

「交替跟进」：一次只动一条腿，每循环 FR+1 → RL+1 → FL+1 → RR+1，四足永远"前足成对一级、后足成对一级"错开，
不会出现 B2 站不住的"四足同踩一级"。平地起步先走 **6 步入口**进入标准状态 S0（FL2/FR1/RR1/RL0），
之后是**两段式**：入口跑完读一次实测机身/足端位姿 → 按实测重规划爬梯 14 步 → 不重置地接着跑
（入口落点与计划差 46 mm，不分段只有 17/20）。控制器 = 50 Hz 关节 PD(800/20) + 重力前馈 + 机身位姿伺服 k=1.2。

## 完整作业流（`--demo`）

```
平地站姿 x=−0.715
  → 走近 8 步（四拍走，步幅 0.30，机身 −0.715 → −0.115）       ← 平地上的四拍：机身 pitch 0、锚点=上一位姿+1/4 步幅
  → 按实测位姿重规划入口 6 步（→ S0：FL2/FR1/RR1/RL0）          ← 走过来的落点差 ~7 cm，用实测状态重规划
  → 爬梯 14 步（按入口后的实测状态重规划）                      ← 与 20 步演示完全同一套代码
  → 登顶停顿 1.2 s → 平台上走 8 步，边走边把机身放平、抬到站高   ← 仰 7.3°→0°、z 1.199→1.258，没有单独的"站直"段
```

- 四段之间**不停**：第一段跑完直接读实测状态、接着跑下一段（`run_plan(reset=False)`）。
- 走近段/平台段是"平地上的四拍走"：一次只动一条腿、腿落点 = 自己原位 + 步幅，位姿仍由搜索给出
  （机身姿态固定、俯仰 ≤10°），所以规划是秒级；入口段/爬梯段仍是细网格位姿搜索（各一次、缓存）。
- 登顶恢复**不摆"站直位姿"**：以前用 4 s 硬站直到平地全高（0.550，腿伸展 84%），实测第一拍就前倾 45° 摔倒，
  只好把平台站高降到 1.258（0.470）。现在改成"停顿一下然后继续走，机身每走一步放平一点、抬一点"
  （俯仰按步线性过渡到 0），同一个 0.470 能自然走上去，8 级长楼梯也是 8/8。再高到 0.550 仍会在第 6 步翻 ——
  这是这套"纯关节 PD + 比例机身伺服"的能力上限。

## 常用命令

| 命令 | 作用 | 耗时 |
|---|---|---|
| `b2_long_stair.py --view` | **任意级数 3D 连播**（默认 8 级：走近 8 + 入口 6 + 6×4 级 + 登顶 10 + 平台恢复 8 = 56 步） | 秒级起播 + 约 3 min（动作时长 193 s） |
| `b2_long_stair.py --steps 6 --view` | 同样的流程跑 6 级楼梯 | 秒级起播 + 约 2.5 min |
| `b2_long_stair.py` | 任意级数跑数据，写 `b2_long_stair.txt/.json` | 约 3 s（缓存命中） |
| `... --demo --view` | **完整作业流 3D 连播**：走近 8 + 入口 6 + 爬梯 14 + 平台恢复 8 = 36 步 | 2.4 s 起播 + 约 2.5 min（动作时长 130 s） |
| `... --demo` | 完整作业流跑数据，写 `b2_full_course.txt/.json` | 约 7 s（缓存命中） |
| `... --view` | 只播 20 步爬梯；`--slow` = 0.5 倍速 | 1.4 s 起播 + 91 s |
| `... b2_follow_gait.py` | 平地起步 20 步，写 `b2_follow_gait.txt/.json` | 2 s（缓存命中） |
| `... --no-entry --rebuild` | 只跑爬梯段（S0 起步，14/14），强制重算计划 | 约 5 min |
| `... --one-phase` | 对照：不分段（17/20） | 2 s |
| `... --plan` | 只规划、打印每步余量/伸展表 | 4–6 min |
| `... --png` | 额外输出四联过程图 | +5 s |
| `... --rebuild` | 忽略规划缓存重新算 | 4–7 min |

## 目录（当前链路）

```text
机器狗/
├─ experiments/b2/b2_follow_gait.py            ★ 主脚本（3 级：步态生成 + 两段式执行 + 报告）
├─ experiments/b2/b2_long_stair.py             ★ 任意级数：StairSpec 生成场景 + 一级一循环 + REALIGN
├─ experiments/b2/b2_diag_stance.py            位姿枚举 / 非足端避碰 / PD 保持（被主脚本复用）
├─ experiments/feasibility/three_robot_gait_0272.py   公共工具：StairSpec / 场景生成 / rig / IK / run_plan
├─ experiments/b2/b2_long_stair.{txt,json}     长楼梯产物（逐级实测表 + 每拍数据）
├─ experiments/b2/b2_follow_gait.{txt,json}    跑数据产物（报告 + 逐拍数据）
├─ experiments/b2/b2_follow_gait.png           四联过程图（起步 / 第4 / 第8 / 第12 / 第20 步）
├─ experiments/b2/b2_follow_gait_plan*.json    规划缓存（删了首跑要重算：走近/平台 25/10 s、入口 17 s、每级 2–4 min）
├─ simulations/mujoco/b2_stairs_static/scene.xml       场景：级高 0.197 / 踏面 0.272 / 宽 0.81 / μ=1.0
├─ simulations/mujoco/steel_stair/stair_bodies.xml     钢梯几何（三机共用同一份）
└─ models/                                     官方模型资产：B2 / Go2 / AS2 + 钢梯 —— **不要删，留作研究**
```

## 环境

Python 3.13（`D:\Python313\python.exe`）、MuJoCo 3.16、NumPy 2.x；依赖见 `requirements-static.txt`。
不需要虚拟环境。MuJoCo 打不开含中文的绝对路径 —— 脚本内部会 `chdir`，所以**命令必须在项目根目录下运行**。

## 已知限制

- 纯关节 PD + 一个比例型机身位姿伺服，没有真正的姿态/力矩闭环，也没有足端力控；同一条轨迹纯开环只能走 1/14 步。
- 每步 3 s（挪重心 1.0 + 抬腿 1.5 + 落定 0.5；入口段再慢 1.4 倍），比真机可接受的速度慢。
- 机身只按 pitch 参数化，yaw/横向没有闭环。
- 没有真机标定、质量/摩擦域随机化或安全盾；单跑一段楼梯（8 级）到平台，还没连爬多段（楼梯—平台—下一段楼梯）。
- "登顶检测"用的是场景几何查询（场景由本程序生成），不是感知；上真机要换成高度图 + 里程计。
- 第一次跑每条楼梯要逐级现规划（每级 2–4 min），之后按「规划链状态 + 局部几何」读缓存秒开。

## 历史阶段（保留，不参与当前链路）

| 阶段 | 入口 | 结论 |
|---|---|---|
| Go2 + 规则 FSM + PPO 残差（2026-10） | `run_demo.py`、`fsm_stairs_controller.py`、`reinforcement_learning/`、`docs/{ARCHITECTURE_PLAN,RL_EXPERIMENT_LOG,FSM_STAGE_REPORT}.md` | 0/20 登顶，转纯脚本化步态 |
| 机型评估与场景复测（2026-11-06/07） | `experiments/as2/`、`experiments/feasibility/`、`docs/{AS2_*,GO2_LIMIT_REPORT,B2_VS_M1_REPORT,THREE_ROBOT_COMPARE,STAIR_0272_UPDATE}.md` | 定 B2、钢梯改 0.197/0.272 |
| 姿态与窄踏面探索（2026-11-06/07） | `experiments/b2/b2_{step_pose,staggered_stance,stair_gait,crawl_0272,stance_compare}.py`、`docs/B2_{CROSS_STEP,STAGGERED_STANCE,STAIR_GAIT,STAND_LOAD}_REPORT.md` | 定「交替跟进」+ 机身位姿伺服 |
