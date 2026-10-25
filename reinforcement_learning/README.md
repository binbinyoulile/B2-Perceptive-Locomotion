# 强化学习工作区

当前已建立目录，并归档外部 IsaacLab/RL 参考工程；不包含自研训练代码，也不包含已安装的训练依赖。

## 目录

```text
reinforcement_learning/
├─ src/                         自研 Gymnasium 环境、奖励和训练入口
├─ configs/                     场景、PPO 和课程配置
├─ tests/                       环境单元测试
├─ upstream/go2_omniverse/      IsaacLab/RL 外部参考，不直接修改
└─ README.md
```

## 外部参考工程

`upstream/go2_omniverse/` 是 BSD-2-Clause 的 Go2 Isaac Sim/IsaacLab 参考快照，可用于阅读：

- `custom_rl_env.py`：RL 环境组织方式；
- `terrain_cfg.py`、`terrain_generator_cfg.py`：地形和课程配置；
- `agent_cfg.py`：训练配置；
- ROS2 和数字孪生相关代码。

限制：

- 本次不安装、不运行、不集成；
- 它依赖 Linux、Isaac Sim/IsaacLab 和 ROS2，与当前 Windows MuJoCo 静态场景不同；
- 不直接复制其核心代码，先理解并重写适合本项目的环境。

## 第一版环境的最小要求

- 行为符合 Gymnasium API；
- 明确的动作空间、观测空间和终止条件；
- 固定随机种子；
- 可配置控制频率和仿真步长；
- 奖励项可单独关闭，便于消融；
- 能记录速度、姿态、足端接触、力矩和能量；
- 能在无渲染模式下批量运行；
- 有随机策略和固定动作策略测试。

## 开发顺序

1. 平地站立环境；
2. 平地速度跟踪；
3. 小台阶；
4. 多级楼梯；
5. 域随机化和鲁棒性评测。

在完成平地训练闭环前，不直接搭建完整楼梯训练。