# Unitree Go2 楼梯运动项目

> 当前状态（2026-10-30）：保留 MuJoCo 完整钢梯场景、0.04 m 单级台阶、带机头 LiDAR 的 Go2 模型、规则 FSM Teacher，以及最终 5 万步 FSM + PPO 残差模型。当前模型尚未实现四足稳定登顶，不能宣称已通过 0.195 m 高台阶或完整三层钢梯。

## 项目简介

本项目研究 Unitree Go2 四足机器人在 MuJoCo 中的楼梯运动与分层控制。主线路为 **规则 FSM Teacher + PPO 有界残差 Student**：FSM 生成可解释的基础动作，PPO 只输出小幅关节修正。当前仓库已完成阶段 0/1 的实时展示、感知环境和低台阶短测，后续重点是解决后腿跟随、支撑足锚点与重心转移问题。

## 一键运行

在 PowerShell 中运行完整钢梯 + Go2 实时演示：

```powershell
cd D:\桌面\机器狗; D:\Python313\python.exe .\run_demo.py
```

运行 0.04 m 低台阶 FSM 实时演示：

```powershell
cd D:\桌面\机器狗; D:\Python313\python.exe .\fsm_stairs_controller.py --scene low
```

加载最终 50k PPO 残差模型进行 3D 回放：

```powershell
cd D:\桌面\机器狗; D:\Python313\python.exe .\reinforcement_learning\eval_fsm_residual.py --model .\experiments\checkpoints\ppo_fsm_residual_50k\ppo_fsm_residual_50k_latest.zip
```

运行时不要求创建或激活虚拟环境；关闭 MuJoCo 窗口或按 `Ctrl+C` 结束。

## 当前能力

- 官方 Unitree Go2 MJCF 与完整钢梯可视化/碰撞资产。
- 带机头前向 `lidar_site` 的派生 Go2 模型，不修改官方模型。
- 完整三层钢梯场景及对应 LiDAR 场景。
- 0.04 m 单级低台阶课程场景。
- FSM 状态：`FLAT_TROT -> HIGH_STEP -> CRAWL -> CLIMB_PUSH -> DONE/FAILED`。
- 175 维策略观测，包含关节与机身状态、局部高度图和关节物理限位距离。
- 最终 PPO 残差模型：`ppo_fsm_residual_50k_latest.zip`。

## 5 万步短测结论

| 方案 | 四足登顶率 | 平均前进距离 | IK 裁剪 | 动作饱和率 |
|---|---:|---:|---:|---:|
| 纯 FSM Teacher | 0/20 | +0.267 m | 257 次 | 0% |
| FSM + 0.10 × PPO | 0/20 | +0.485 m | 0 次 | 0% |

短测消除了 IK 裁剪和关节限位违规，但尚未解决四足登顶。完整对比见 `experiments/evaluations/fsm_residual_50k_compare.txt`，阶段结论见 `docs/FSM_STAGE_REPORT.md`。

## 精简目录

```text
机器狗/
├─ README.md
├─ requirements-static.txt
├─ run_demo.py
├─ fsm_stairs_controller.py
├─ docs/
│  ├─ ARCHITECTURE_PLAN.md
│  ├─ RL_EXPERIMENT_LOG.md
│  └─ FSM_STAGE_REPORT.md
├─ experiments/
│  ├─ checkpoints/ppo_fsm_residual_50k/ppo_fsm_residual_50k_latest.zip
│  └─ evaluations/fsm_residual_50k_compare.txt
├─ models/                         官方 Go2、LiDAR 派生模型与钢梯资产
├─ reinforcement_learning/
│  ├─ eval_fsm_residual.py         最终模型 3D 回放入口
│  └─ src/                         环境、奖励、观测、步态与地形感知源码
└─ simulations/mujoco/go2_stairs_static/
   ├─ scene.xml                    完整三层钢梯
   ├─ scene_single_step_004.xml    0.04 m 单级低台阶
   └─ scene_full_lidar.xml         完整钢梯 LiDAR 版本
```

`reinforcement_learning/eval_fsm_residual.py` 是核心运行入口之外的唯一必要展示文件；没有它，保留的最终权重无法直接加载到 3D 窗口。

## 环境要求

- Python 3.10+，当前 Windows 验证环境为 `D:\Python313\python.exe`。
- MuJoCo 3.x、NumPy 2.x、Gymnasium、Stable-Baselines3、PyTorch。
- 具备 OpenGL 的桌面图形环境，用于打开被动式 MuJoCo 3D 窗口。
- 依赖版本记录见 `requirements-static.txt`。

## 核心文档

- `docs/ARCHITECTURE_PLAN.md`：Teacher-Student 架构与后续路线。
- `docs/RL_EXPERIMENT_LOG.md`：强化学习尝试、失败模式与调参记录。
- `docs/FSM_STAGE_REPORT.md`：FSM 阶段进展、短测结果和已知限制。
- `docs/GO2_LIMIT_REPORT.md`：0.195 m / 0.18 m 钢梯的物理极限测试与死局因素分析。
- `docs/B2_VS_M1_REPORT.md`：B2 与 M1 的任务可行性、仿真生态对比与 Sim2Sim 接口规划。

## 已知限制

- 0.04 m 低台阶尚未实现稳定的四足登顶。
- 后腿跟随、支撑足锚点与机身姿态改变时的 IK 可达性仍是主要问题。
- 0.195 m 高台阶和完整三层钢梯尚未完成跨越验证。
- 当前没有真机标定、质量/摩擦域随机化或完整安全盾。
