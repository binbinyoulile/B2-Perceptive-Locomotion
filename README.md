# 宇树 Go2 机器狗项目

面向两项主要任务的 Go2 仿真与控制研究项目：

1. 机器狗上下楼梯的 MuJoCo 仿真；
2. 机器狗运动控制强化学习的学习、训练与仿真实践。

当前阶段只做项目整理、环境固化、可复现场景和技术路线准备，不开始训练或楼梯越障算法。

## 当前状态

- 官方 Unitree Go2 MuJoCo 模型已归档并验证。
- 钢制楼梯原始 STL、正确立式姿态和 Go2 + 楼梯静态场景已归档并验证。
- 已从 STL 提取楼梯可落脚面：3 级上行、2.5 m 平台、3 级下行。
- 已归档 `go2_gait_planner` 作为步态与控制参考，但尚未集成。
- 已归档 `go2_omniverse` 作为 IsaacLab/RL 参考，但不作为当前 MuJoCo 主链路。
- 已建立仿真、强化学习、实验记录、数据集和历史归档目录。
- 已提供环境检查、楼梯几何提取、场景验证和预览生成脚本。

![Go2 与楼梯静态场景](simulations/mujoco/go2_stairs_static/preview.png)

## 目录结构

```text
机器狗/
├─ docs/                         项目计划、学习路线、环境和资料说明
├─ models/
│  ├─ unitree_go2/               MuJoCo Menagerie 官方 Go2 模型
│  └─ stairs/                    楼梯原始 STL、几何估计和说明
├─ simulations/
│  └─ mujoco/
│     └─ go2_stairs_static/      当前可运行的静态 Go2 + 楼梯场景
├─ control/
│  └─ upstream/
│     └─ go2_gait_planner/       外部步态规划参考工程
├─ reinforcement_learning/
│  ├─ src/                       后续自研 Gymnasium/RL 代码
│  ├─ configs/                   后续训练和课程配置
│  └─ upstream/go2_omniverse/    IsaacLab/RL 外部参考工程
├─ datasets/                     后续地形、轨迹和评测数据
├─ experiments/                  训练运行、检查点与报告
├─ scripts/                      环境检查和项目辅助脚本
└─ archive/legacy/               历史版本与旧项目资料
```

## 快速检查

双击项目根目录的 `检查项目环境.bat`，或运行：

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\check_environment.py"
```

该脚本会检查 Python、MuJoCo、NumPy、Pillow、官方 Go2 模型、静态场景、楼梯姿态、世界尺寸和初始接地状态，并列出尚未安装的强化学习依赖。

重新提取楼梯几何：

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\analyze_stairs_geometry.py"
```

验证静态场景：

```powershell
cd "D:\桌面\机器狗\simulations\mujoco\go2_stairs_static"
D:\Python313\python.exe validate.py
```

打开静态场景：双击 `打开Go2楼梯场景.bat`。

## 关键文档

- [项目总体路线图](docs/PROJECT_PLAN.md)
- [楼梯几何估计](docs/STAIR_GEOMETRY.md)
- [强化学习学习路线](docs/LEARNING_ROADMAP.md)
- [环境与运行说明](docs/ENVIRONMENT.md)
- [资料与来源清单](docs/SOURCE_CATALOG.md)
- [优先阅读资料](docs/READING_LIST.md)
- [技术决策记录](docs/DECISION_LOG.md)
- [辅助脚本说明](scripts/README.md)

## 推荐工作顺序

1. 固化环境检查流程并建立 Git 基线。
2. 学习 MuJoCo 基础 API、坐标系、四足运动学、接触与执行器。
3. 建立平地低层控制器基线，先做到稳定站立，再尝试小跑步态。
4. 依据已测量的几何建立楼梯简化碰撞体，并验证足端落脚。
5. 在平地上完成第一版 Gymnasium 强化学习环境和 PPO 训练闭环。
6. 用“平地 → 低台阶 → 多级台阶 → 缩小楼梯 → 完整楼梯”的课程逐步增加难度。
7. 统一评测成功率、时间、能耗、脚滑、跌倒和鲁棒性，并形成报告。

## 重要约定

- `models/` 存放原始或官方模型，不作为实验改动的直接工作区。
- `simulations/` 存放可直接运行的场景；训练场景与静态展示场景分开。
- `reinforcement_learning/` 存放自研环境、算法和配置；控制类外部代码放在 `control/upstream/`，RL/IsaacLab 类外部代码放在 `reinforcement_learning/upstream/`。
- `experiments/` 中的训练结果不手工覆盖；每次运行使用独立目录。
- 楼梯整体 STL 不直接开启碰撞，后续必须建立简化碰撞体。
- 重要模型和配置变更要记录来源、日期和验证结果。

## Windows 中文路径说明

当前 Python MuJoCo 接口无法稳定直接打开含中文的绝对 XML 路径。项目内脚本会先切换到 XML 所在目录，再使用相对路径加载，因此当前目录可以直接运行。后续接入 WSL、JAX/MJX 或其他训练框架时，需要重新验证中文路径兼容性；如遇问题，可为项目建立 ASCII 路径别名。

## 当前未完成

- 楼梯尺寸估计已完成；逐级碰撞体尚未建立，正式 CAD 尺寸仍待确认。
- 尚无站立平衡、步态控制或强化学习控制器。
- 尚未安装 Gymnasium、PyTorch、Stable-Baselines3、JAX/MJX 等训练依赖。
- `go2_gait_planner` 的 ROS2 构建尚未验证，许可证声明仍不明确，仅作为参考。
- `go2_omniverse` 依赖 Isaac Sim/IsaacLab 和 Linux/ROS2，当前只归档不运行。