# 资料与来源清单

整理日期：2026-10-25

## 已导入资料

| 资料 | 原位置 | 项目位置 | 用途 | 状态 |
|---|---|---|---|---|
| Unitree Go2 官方 MJCF | `D:\codex\2026-10-22\new-chat\work\unitree_go2` | `models/unitree_go2/` | 官方机器人模型、MJX 变体、场景和许可证 | 已复制并验证 |
| 钢制楼梯 STL | `D:\桌面\steel_stair.STL` | `models/stairs/steel_stair.STL` | 楼梯原始几何 | 已复制 |
| 楼梯几何估计 | 由本地 STL 自动提取 | `models/stairs/geometry_estimate.json` | 后续简化碰撞体尺寸依据 | 已生成 |
| Go2 + 楼梯静态场景 | `D:\codex\2026-10-22\new-chat\outputs\mujoco_go2_stair` | `simulations/mujoco/go2_stairs_static/` | 当前可运行场景和预览 | 已复制并验证 |
| 仅有楼梯的旧场景 | `D:\codex\2026-10-22\new-chat\outputs\mujoco_steel_stair` | `archive/legacy/steel_stair_only_2026-10-22/` | 历史快照 | 已归档 |
| Go2 步态规划参考工程 | `D:\桌面\go2_gait_planner-main\go2_gait_planner-main` | `control/upstream/go2_gait_planner/` | ROS2 步态规划、运动学和底层控制参考 | 已复制，未构建 |
| Go2 IsaacLab/RL 参考工程 | `D:\桌面\公司之前项目\go2_omniverse-master` | `reinforcement_learning/upstream/go2_omniverse/` | 地形配置、RL 环境、Isaac Sim/ROS2 研究参考 | 已完整复制，未运行 |
| 旧机器狗辐射探测程序 | `D:\桌面\公司之前项目\机器狗辐射探测` | `archive/legacy/company_radiation_detection_app_2025/` | 既往机器狗应用资料 | 仅归档源码，排除 `.vs/bin/obj` |

## 文件校验值

SHA-256：

```text
steel_stair.STL
839ABB1DB0238B32E9F5B001F77386A9957AFCF28ECE9CCB1DB337759FC59289

models\unitree_go2\go2.xml
50ADB09A4365293E2ACDAF2010AE35A82B0F09FEA18AE51806FEF91E310ED04A

simulations\mujoco\go2_stairs_static\scene.xml
1B5D7DF5BFF399D060EDCF3EC33609CE77C91525DC02257E26D345FBA6E9D112
```

## 官方 Go2 模型说明

`models/unitree_go2/` 包含：

- `go2.xml`：常规 MuJoCo 模型；
- `go2_mjx.xml`：MJX 版本；
- `scene.xml`、`scene_mjx.xml`：官方场景包装；
- `assets/`：16 个 OBJ 网格；
- `LICENSE`：BSD-3-Clause；
- `README.md`、`CHANGELOG.md`：来源和版本信息。

常规 MJCF 已验证：

```text
nq/nv/nu = 19/18/12
ngeom/nmesh = 57/16
```

`simulations/mujoco/go2_stairs_static/` 保留独立自包含的 Go2 资源和楼梯资源，避免以后调整目录时破坏可复现场景。

## 楼梯资料

- `models/stairs/steel_stair.STL`：原始网格；
- `models/stairs/geometry_estimate.json`：自动提取的可落脚面、级高和坐标范围；
- `docs/STAIR_GEOMETRY.md`：可读尺寸表、结构判断和控制影响；
- 当前估计为 3 级上行、中间平台、3 级下行；
- 踏面深度约 `0.18 m`，上行级高约 `0.1950 / 0.1853 / 0.1852 / 0.2245 m`；
- 平台尺寸约 `2.5 × 0.8 m`。

## 外部参考工程

### `go2_gait_planner`

README 提到 ROS2、`unitree_ros2`、`unitree_mujoco` 和 Eigen，本地包依赖 `rclcpp`、`unitree_go` 等。

当前状态：

- 仅作为代码阅读和算法参考；
- 尚未安装 ROS2，未执行 `colcon build`；
- `package.xml` 中许可证仍为 `TODO: License declaration`；
- 不应在未确认许可证前直接复制其核心代码到可分发项目中。

### `go2_omniverse`

- BSD-2-Clause，许可证已随源码保存在该目录；
- 面向 Isaac Sim 5.x / IsaacLab / ROS2，不是当前 MuJoCo 主链路；
- 包含 `custom_rl_env.py`、`terrain_cfg.py`、`terrain_generator_cfg.py`；训练检查点已移入 `archive/待确认/go2_omniverse_logs/`；
- 当前只作为后续 RL、地形课程和域随机化的阅读材料，尚未运行或集成；
- 其中包含 G1 和数字孪生相关内容，与本项目当前 Go2 楼梯目标无关的部分不要混入自研代码。

### 旧辐射探测程序

- 仅用于保留既往机器狗应用资料，不是当前上下楼梯或 RL 任务的依赖；
- 原工程中 `.vs`、`bin`、`obj` 等缓存和编译产物没有复制；
- 该工程的许可证和维护状态未确认，不作为可复用核心代码。

## 当前未导入的资料

目前项目目录中没有发现以下材料：

- 教师任务书或评分标准；
- Go2 用户手册、SDK 文档或 URDF 原始包；
- 强化学习课程讲义、论文 PDF 或视频；
- 楼梯正式 CAD 图纸、二维尺寸图或现场实测记录；
- 实机试验数据、动作捕捉或传感器日志。

拿到这些资料后，分别放入 `docs/`、`datasets/raw/` 或 `docs/references/`，并在本清单追加来源和日期。
