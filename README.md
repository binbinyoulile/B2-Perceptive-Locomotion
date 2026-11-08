# 机器狗：B2 爬钢梯

> **当前状态（2026-11-08）**：B2 从**平地正常站姿起步，20 步全程爬上钢梯平台（0.788 m）**，
> 四足都落在平台上、全程未失稳。基线标签 `v1.0-FSM-20Steps-Success`。
> 这是 MuJoCo 仿真结果；真机尚未标定。旧版本 README 写的"Go2 尚未登顶"属于更早的 Go2 阶段，见文末「历史阶段」。

## 最快看到结果

```powershell
cd "D:\桌面\机器狗"
D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py --view   # 3D 实时播放：1.4 s 起播、20 步约 91 s
D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py          # 跑数据：约 2 s（读缓存），写 txt/json
```

实测：**20/20 步**走通、全程未失稳；末态四足接触 `platform` 164/158/178/176 N；
机身 0.680 → 1.198 m；末态重心余量 183 mm（静置 2 s 不倾覆）；峰值力矩 187%（FL_thigh 瞬态）。
完整数据与失败档案见 [`docs/B2_PLATFORM_CLIMB.md`](docs/B2_PLATFORM_CLIMB.md)，文档地图见 [`docs/INDEX.md`](docs/INDEX.md)。

## 步态一句话

「交替跟进」：一次只动一条腿，每循环 FR+1 → RL+1 → FL+1 → RR+1，四足永远"前足成对一级、后足成对一级"错开，
不会出现 B2 站不住的"四足同踩一级"。平地起步先走 **6 步入口**进入标准状态 S0（FL2/FR1/RR1/RL0），
之后是**两段式**：入口跑完读一次实测机身/足端位姿 → 按实测重规划爬梯 14 步 → 不重置地接着跑
（入口落点与计划差 46 mm，不分段只有 17/20）。控制器 = 50 Hz 关节 PD(800/20) + 重力前馈 + 机身位姿伺服 k=1.2。

## 常用命令

| 命令 | 作用 | 耗时 |
|---|---|---|
| `... b2_follow_gait.py` | 平地起步 20 步，写 `b2_follow_gait.txt/.json` | 2 s（缓存命中） |
| `... --view` | 3D 实时播放；`--slow` = 0.5 倍速 | 1.4 s 起播 + 91 s |
| `... --no-entry --rebuild` | 只跑爬梯段（S0 起步，14/14），强制重算计划 | 约 5 min |
| `... --one-phase` | 对照：不分段（17/20） | 2 s |
| `... --plan` | 只规划、打印每步余量/伸展表 | 4–6 min |
| `... --png` | 额外输出四联过程图 | +5 s |
| `... --rebuild` | 忽略规划缓存重新算 | 4–7 min |

## 目录（当前链路）

```text
机器狗/
├─ experiments/b2/b2_follow_gait.py            ★ 主脚本（步态生成 + 两段式执行 + 报告）
├─ experiments/b2/b2_diag_stance.py            位姿枚举 / 非足端避碰 / PD 保持（被主脚本复用）
├─ experiments/feasibility/three_robot_gait_0272.py   公共工具：rig / IK / PD 循环 / run_plan / pacer
├─ experiments/b2/b2_follow_gait.{txt,json}    跑数据产物（报告 + 逐拍数据）
├─ experiments/b2/b2_follow_gait.png           四联过程图（起步 / 第4 / 第8 / 第12 / 第20 步）
├─ experiments/b2/b2_follow_gait_plan*.json    规划缓存（65 KB；删了首跑要重算 6–7 min）
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
- 没有真机标定、质量/摩擦域随机化或安全盾；目前只到第一段楼梯的平台，还没连爬多段。

## 历史阶段（保留，不参与当前链路）

| 阶段 | 入口 | 结论 |
|---|---|---|
| Go2 + 规则 FSM + PPO 残差（2026-10） | `run_demo.py`、`fsm_stairs_controller.py`、`reinforcement_learning/`、`docs/{ARCHITECTURE_PLAN,RL_EXPERIMENT_LOG,FSM_STAGE_REPORT}.md` | 0/20 登顶，转纯脚本化步态 |
| 机型评估与场景复测（2026-11-06/07） | `experiments/as2/`、`experiments/feasibility/`、`docs/{AS2_*,GO2_LIMIT_REPORT,B2_VS_M1_REPORT,THREE_ROBOT_COMPARE,STAIR_0272_UPDATE}.md` | 定 B2、钢梯改 0.197/0.272 |
| 姿态与窄踏面探索（2026-11-06/07） | `experiments/b2/b2_{step_pose,staggered_stance,stair_gait,crawl_0272,stance_compare}.py`、`docs/B2_{CROSS_STEP,STAGGERED_STANCE,STAIR_GAIT,STAND_LOAD}_REPORT.md` | 定「交替跟进」+ 机身位姿伺服 |
| 旧版交接文档 | `PROJECT_HANDOFF.md`（2026-11-05，925 行） | 机器狗系统说明的历史快照 |
