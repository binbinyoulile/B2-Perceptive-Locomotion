# 文档地图（先读这一页）

> 目的：新会话（人或 AI）只读这一页就能知道**当前在做什么、跑哪条命令、改哪里、别删什么**，
> 不用把 925 行的 `PROJECT_HANDOFF.md` 从头读一遍。

## 1. 当前基线（2026-11-08）

- 里程碑：B2 平地起步 **20 步**爬上钢梯平台，20/20 全程未失稳 → 标签 `v1.0-FSM-20Steps-Success`
- 唯一主脚本：`experiments/b2/b2_follow_gait.py`
- 结果文档：**`B2_PLATFORM_CLIMB.md`**（本文档目录里唯一"活的"结论文档）
- 复现：`D:\Python313\python.exe .\experiments\b2\b2_follow_gait.py --view`（在项目根目录运行）

## 2. 文档分类

| 文档 | 角色 | 还要不要读 |
|---|---|---|
| `B2_PLATFORM_CLIMB.md` | **当前结论**：20 步全程数据、失败档案、耗时 | ✅ 必读 |
| `STAIR_0272_UPDATE.md` | 钢梯尺寸改成 0.197/0.272/0.81 的由来与几何极限 | ✅ 背景 |
| `THREE_ROBOT_COMPARE.md` | 三机同标准步态对比（为什么选 B2） | 🔸 背景 |
| `B2_STAGGERED_STANCE_REPORT.md` / `B2_CROSS_STEP_REPORT.md` | 错位站姿与跨级姿态的静态可行域 | 🔸 步态依据 |
| `B2_STAIR_GAIT_REPORT.md` | 0.18 m 窄踏面时代的步态失败分析 | ⛔ 历史 |
| `B2_STAND_LOAD_REPORT.md` | B2 参数固化与场景摩擦 | 🔸 参数来源 |
| `GO2_LIMIT_REPORT.md` / `B2_VS_M1_REPORT.md` | Go2 极限、换平台评估 | ⛔ 历史 |
| `AS2_PLATFORM_REPORT.md` / `AS2_CLIMB_ATTEMPT_REPORT.md` | AS2 尝试后放弃 | ⛔ 历史 |
| `ARCHITECTURE_PLAN.md` / `RL_EXPERIMENT_LOG.md` / `FSM_STAGE_REPORT.md` | Go2 + FSM/PPO 阶段（0/20 登顶） | ⛔ 历史 |
| `../PROJECT_HANDOFF.md` | 2026-11-05 的系统交接文档（925 行） | ⛔ 历史快照 |

## 3. 代码地图

| 文件 | 作用 |
|---|---|
| `experiments/b2/b2_follow_gait.py` | 主脚本：入口/爬梯计划生成、两段式执行、报告与出图 |
| `experiments/feasibility/three_robot_gait_0272.py` | 公共工具：`load_rig` / `legs_ik` / PD 循环 `Loop` / `run_plan` / `pacer` / `hold` |
| `experiments/b2/b2_diag_stance.py` | 位姿枚举 `enumerate_poses`、非足端避碰 `filter_clear`、PD 静置 |
| `simulations/mujoco/b2_stairs_static/scene.xml` | 场景入口（include B2 官方 MJCF + 钢梯） |
| `simulations/mujoco/steel_stair/stair_bodies.xml` | 钢梯几何唯一副本（三机场景都 include 它） |

## 4. 别删 / 别改

- `models/`（72 文件 / 88.6 MB）：B2、Go2、AS2 官方模型与钢梯资产 —— **一个都不要删**。
- `experiments/b2/b2_follow_gait_plan.json`、`b2_follow_gait_plan_climb.json`：规划缓存（65 KB）。
  它们让 `--view` 1.4 s 起播；删了首次运行要重算 6–7 分钟。改了任何规划参数会自动失效并重算。
- `experiments/b2/b2_follow_gait_trace*.txt`：两次失败的时间历程，文档 §5 失败档案引用的证据。
- **已验证段落的规划参数不要动**：`PITCH_HI`、`WIN_X/WIN_Z`、`EDGE`、`SHIFT_S/LIFT_S/SETTLE_S`、`PLAT_*`、
  `GROUND_*` 都是逐步试出来的；改 2–4 cm 就会把整条跑通的轨迹改坏（实测 14/14 → 2/14）。
  新需求用"追加段"的方式加，别改已验证段。

## 5. 已知的坑（省时间用）

- 命令必须在项目根目录执行；MuJoCo 打不开含中文的绝对路径（脚本内部 `chdir` 规避）。
- 只规划不跑动力学用 `--plan`（4–6 min）；跑动力学本身只要 2 s —— 慢的都是规划，别看错方向。
- 换机器后第一次运行会重新规划 6–7 min，属正常；之后秒开。
- `--view` 的窗口播完不会自动关，按 X 或 Ctrl+C 结束。
