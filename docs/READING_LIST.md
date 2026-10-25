# 优先阅读资料

## P0：先读

1. MuJoCo 官方文档：模型结构、Python API、接触、执行器。
2. MuJoCo Menagerie `unitree_go2` 的 README、`go2.xml` 和 `scene.xml`。
3. Gymnasium 官方 environment API。
4. Stable-Baselines3 PPO 文档。
5. Unitree Go2 官方产品页、SDK2 和 ROS2 资料。

## P1：四足控制基础

- 四足机器人运动学与雅可比矩阵；
- Trot 步态和足端轨迹；
- 接触力、摩擦锥和关节力矩限制；
- 状态估计、姿态控制和速度跟踪；
- 模型预测控制、全身控制和强化学习的基本区别。

## P2：强化学习论文

建议按以下顺序阅读：

1. PPO：Schulman et al., *Proximal Policy Optimization Algorithms*。
2. Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*。
3. Lee et al., *Learning Quadrupedal Locomotion over Challenging Terrain*。
4. Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots*。
5. Kumar et al., *Rapid Locomotion via Reinforcement Learning*。
6. 后续再阅读 parkour、stairs、teacher-student 和 sim-to-real 方向论文。

阅读要求：

- 每篇论文记录问题、观测、动作、奖励、课程、仿真器和实验指标；
- 不只看最终视频，重点看失败案例、消融和鲁棒性；
- 把可复用的设计记录到 `docs/`。

## P3：工具与工程

- MuJoCo MJX；
- JAX；
- Stable-Baselines3；
- TensorBoard；
- W&B 或 MLflow（可选）；
- IsaacLab、legged_gym（后期评估，不必现在安装）。

## 常用官方入口

- MuJoCo：https://mujoco.readthedocs.io/
- MuJoCo Menagerie：https://github.com/google-deepmind/mujoco_menagerie
- Gymnasium：https://gymnasium.farama.org/
- Stable-Baselines3：https://stable-baselines3.readthedocs.io/
- Unitree Go2：https://www.unitree.com/go2/
- Unitree ROS2：https://github.com/unitreerobotics/unitree_ros2
- Unitree MuJoCo：https://github.com/unitreerobotics/unitree_mujoco
