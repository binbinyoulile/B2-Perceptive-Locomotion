# 文档地图（先读这一页）

> ## 🚨 硬性红线：公开仓库前必须做「个人信息与敏感数据」扫描
>
> 任何人 / 任何 AI，在把本仓库**转为 public、推送到公开远端、或上传压缩包给外部**之前，
> 都**必须先跑一遍扫描并确认无残留**。这不是建议，是红线。
>
> 扫描必须覆盖四类内容，范围必须包含**全部 git 历史**（不只是当前工作区）：
>
> 1. **内部文档**：类似已删除的 `PROJECT_HANDOFF.md` 那种「给 AI 的交接说明书」——
>    里面有个人背景、AI 辅助编程说明、内部规划与能力边界，一律不得出现在公开仓库。
> 2. **个人身份信息**：学校 / 单位 / 姓名 / 学号 / 电话 / 邮箱 / 对个人的评价性描述。
> 3. **凭证**：`ghp_*`、`github_pat_*`、`sk-*`、`AKIA*`、`BEGIN ... PRIVATE KEY`、密码赋值、`.env`、`.pem`、`id_rsa`。
> 4. **本机数据**：记忆库 `.dsh-meow/`、数据库、日志、含用户名的绝对路径（`C:\Users\...`、`/home/...`）。
>
> 参考命令（搜全历史）：
> `git grep -I -n -E '<关键词>' $(git rev-list --all)`
> 以及 `git log --all --diff-filter=A --name-only | Sort-Object -Unique`（查敏感文件名是否曾入库）。
>
> 发现**任何**拿不准的内容：先停下、报给用户决定，不要自行判断“无害”后公开。
> 只做 `git rm` + commit 是**不够**的——历史里还能按旧 SHA 取到；2026-11-08 就是这么泄露过一次，
> 最终靠「改写全历史 + 删库重建」才彻底清除（详见 §5 最后一条）。

> 目的：新会话（人或 AI）只读这一页就能知道**当前在做什么、跑哪条命令、改哪里、别删什么**，
> 不用去翻早已移除的内部交接文档。

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

## 3. 代码地图

| 文件 | 作用 |
|---|---|
| `experiments/b2/b2_follow_gait.py` | 主脚本：入口/爬梯计划生成、两段式执行、报告与出图；`--demo` = 完整作业流（走近→入口→爬梯→平台前行） |
| `experiments/b2/b2_follow_gait_plan*.json` | 四段规划缓存：`approach` / `entry` / `climb` / `platform`（各按「实测状态 + 参数指纹」缓存） |
| `experiments/b2/b2_full_course.{txt,json}` | `--demo` 的产物：四段汇总 + 36 拍实测数据 + 末态核验 |
| `experiments/feasibility/three_robot_gait_0272.py` | 公共工具：`load_rig` / `legs_ik` / PD 循环 `Loop` / `run_plan` / `pacer` / `hold` |
| `experiments/b2/b2_diag_stance.py` | 位姿枚举 `enumerate_poses`、非足端避碰 `filter_clear`、PD 静置 |
| `simulations/mujoco/b2_stairs_static/scene.xml` | 场景入口（include B2 官方 MJCF + 钢梯） |
| `simulations/mujoco/steel_stair/stair_bodies.xml` | 钢梯几何唯一副本（三机场景都 include 它） |

## 4. 别删 / 别改 / 公开前必查

- 🔴 **公开前必查**：见页首红线 —— 个人信息、内部文档、凭证、本机数据（含**全历史**）扫描通过后，
  才允许转 public 或推到公开远端。**任何**拿不准的内容先报给用户，不要自行判断“无害”。
- `models/`（72 文件 / 88.6 MB）：B2、Go2、AS2 官方模型与钢梯资产 —— **一个都不要删**。
- `experiments/b2/b2_follow_gait_plan*.json`（四个：`approach`/`entry`/`climb`/`platform`，共约 78 KB）：
  规划缓存。它们让 `--demo --view` 2.4 s 起播、`--demo` 跑数据 7 s；删了首次运行要重算
  （走近/平台各约 25/10 s，入口 17 s，爬梯 6–7 min）。改了任何规划参数会自动失效并重算。
- `experiments/b2/b2_follow_gait_trace*.txt`：两次失败的时间历程，文档 §5 失败档案引用的证据。
- **已验证段落的规划参数不要动**：`PITCH_HI`、`WIN_X/WIN_Z`、`EDGE`、`SHIFT_S/LIFT_S/SETTLE_S`、`PLAT_*`、
  `GROUND_*` 都是逐步试出来的；改 2–4 cm 就会把整条跑通的轨迹改坏（实测 14/14 → 2/14）。
  新需求用"追加段"的方式加，别改已验证段。

## 5. 已知的坑（省时间用）

- 命令必须在项目根目录执行；MuJoCo 打不开含中文的绝对路径（脚本内部 `chdir` 规避）。
- 只规划不跑动力学用 `--plan`（4–6 min）；跑动力学本身只要 2 s —— 慢的都是规划，别看错方向。
- 换机器后第一次运行会重新规划 6–7 min，属正常；之后秒开。
- `--view` 的窗口播完不会自动关，按 X 或 Ctrl+C 结束。
- **隐私泄露教训（2026-11-08，必读）**：内部交接文档 `PROJECT_HANDOFF.md` 曾随仓库公开（含个人背景与
  「代码主要依赖 AI 辅助生成」等表述）。注意这条链条：
  ① 只做 `git rm` + commit + push **无效**——旧提交里的文件仍可用
  `raw.githubusercontent.com/<repo>/<旧SHA>/PROJECT_HANDOFF.md` 下载；
  ② 用 `git filter-branch` 改写全历史 + `push --force` 后，GitHub **仍能按旧 SHA** 取到内容；
  ③ 只有**删库重建**（新 repo id）才真正清除；第三方已下载的副本无法召回。
  结论：敏感文件**永远不要提交进 git**；一旦提交过，只能靠删库重建兜底。
  验证要用 `gh api repos/<o>/<r>/commits/<旧SHA>`（应返回 `422 No commit found`）和直连 raw（应 404），
  别用带缓存的小工具，它会返回 200 骗人。
