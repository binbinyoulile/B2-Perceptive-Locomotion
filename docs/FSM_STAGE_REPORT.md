# Go2 楼梯项目阶段性交付报告

> 更新时间：2026-10-30  
> 交付目标：保留可运行的“机器狗 + 楼梯 + MuJoCo 实时窗口 + 可解释 FSM + PPO 残差回放”成果，并为后续真实楼梯跨越研究提供基线。  
> 结论先说：**阶段 0/1 的展示链路已建立；0.04 m 低台阶尚未完成稳定的四足登顶，完整三层钢梯尚未完成跨越验证。**

## 1. 当前保留成果

### 1.1 运行入口

- `run_demo.py`：打开完整三层钢梯，展示 Go2 站立、Trot 接近并停稳。
- `fsm_stairs_controller.py`：在 0.04 m 低台阶或完整钢梯 LiDAR 场景中运行规则 FSM。
- `reinforcement_learning/eval_fsm_residual.py`：加载最终 50k PPO 残差模型进行 3D 回放。

### 1.2 核心源码

- `reinforcement_learning/src/go2_env.py`：Gymnasium 环境生命周期、物理推进、当前帧判定与终止。
- `reinforcement_learning/src/obs_builder.py`：175 维观测拼接与观测边界。
- `reinforcement_learning/src/reward_calculator.py`：奖励分项与诊断值。
- `reinforcement_learning/src/terrain_perception.py`：前向 LiDAR 扫描与局部高度图。
- `reinforcement_learning/src/terrain_task.py`：台阶几何、接触与任务判定。
- `reinforcement_learning/src/gait_controller.py`、`gait_config.py`：Trot 先验和步态参数。
- `reinforcement_learning/src/__init__.py`：包导出入口。

### 1.3 场景与模型

- `simulations/mujoco/go2_stairs_static/scene.xml`：完整三层钢梯阶段 0 展示场景。
- `simulations/mujoco/go2_stairs_static/scene_single_step_004.xml`：0.04 m 单级低台阶课程场景。
- `simulations/mujoco/go2_stairs_static/scene_full_lidar.xml`：完整钢梯的 LiDAR 派生场景；当前已验证资产保留，不代表已完成爬梯。
- `models/unitree_go2/go2.xml`：官方 Go2 模型。
- `models/unitree_go2/go2_lidar.xml`：机头 LiDAR 派生模型，不修改官方模型。
- `experiments/checkpoints/ppo_fsm_residual_50k/ppo_fsm_residual_50k_latest.zip`：唯一保留的最终权重。

## 2. Teacher-Student 短测结果

最终对比日志：`experiments/evaluations/fsm_residual_50k_compare.txt`。

| 方案 | 四足登顶率 | 平均前进距离 | IK 裁剪 | 动作饱和率 |
|---|---:|---:|---:|---:|
| 纯 FSM Teacher | 0/20 | +0.267 m | 257 次 | 0% |
| FSM + 0.10 × PPO | 0/20 | +0.485 m | 0 次 | 0% |

结论：

- PPO 有界残差显著改善了平均前进距离；
- IK 裁剪从 257 次降到 0 次，说明混合动作没有继续把关节推到硬限位；
- 两种方案均未实现四足稳定登顶，不能把低台阶短测解释为完整楼梯能力。

## 3. 失败现象与根因

1. **前足可搭台，后腿跟随不足**：前足能够进入台面并触发 CRAWL/CLIMB_PUSH，但后腿收起时机与重心前移不同步。
2. **支撑足锚点与机身姿态冲突**：BODY_POSE 调整机身处姿时，固定足端目标可能超出腿长工作空间，造成 IK 裁剪或支撑脚突然离地。
3. **接触门控与真实物理存在时间差**：MuJoCo 足端接触是短时、离散事件，理想化的三足支撑门控容易误判。
4. **奖励与判定仍可能保留局部最优**：仅“前足搭台”不能等同于任务成功，后续必须坚持当前帧四足台面判定。

## 4. 运行方式

完整钢梯阶段 0/1 实时窗口：

```powershell
D:\Python313\python.exe .\run_demo.py
```

低台阶 FSM 实时窗口：

```powershell
D:\Python313\python.exe .\fsm_stairs_controller.py --scene low --speed 1.0
```

最终 FSM + PPO 残差模型回放：

```powershell
D:\Python313\python.exe .\reinforcement_learning\eval_fsm_residual.py
```

无窗口快速验证：

```powershell
D:\Python313\python.exe .\fsm_stairs_controller.py --scene low --headless --duration 8.0
```

## 5. 已知限制

- 0.04 m 低台阶尚未实现稳定的四足登顶；
- 完整三层钢梯尚未完成跨越验证；
- 现有参数面向当前 Go2 质量、摩擦和课程场景，迁移到真机前必须重新标定；
- 当前没有质量/摩擦域随机化、真机传感器外参标定或完整安全盾；
- 不应在低台阶回归未稳定前启动长周期 PPO 训练。

## 6. 下一步

1. 先把 FSM 的后腿跟随、支撑足锚点和重心转移做成可重复单回合回归；
2. 用 0.04 m 台阶验证“当前帧四足台面”成功判定，避免历史锁存造成假成功；
3. 稳定后再迁移到 0.195 m 单级台阶；
4. 最后再测试完整三层钢梯，不能跳过中间课程。
