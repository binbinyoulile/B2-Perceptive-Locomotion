# 资料与来源清单

整理日期：2026-10-25

本清单只记录当前项目仍在使用的资产。已删除的旧课程、重复模型快照和外部参考工程不再占据仓库空间；相关技术判断保留在 `docs/DECISION_LOG.md`。

## 当前资产

| 资料 | 项目位置 | 用途 | 状态 |
|---|---|---|---|
| Unitree Go2 官方 MJCF | `models/unitree_go2/` | 官方机器人模型、MJX 变体、OBJ 网格和许可证 | 已验证 |
| 钢制楼梯 STL | `models/stairs/steel_stair.STL` | 楼梯原始视觉几何 | 已验证 |
| 楼梯几何估计 | `models/stairs/geometry_estimate.json` | 简化碰撞体和课程参数依据 | 可重复生成 |
| Go2 + 完整钢梯场景 | `simulations/mujoco/go2_stairs_static/scene.xml` | 当前阶段 0 实时演示场景 | 已验证 |
| 单级低台阶场景 | `simulations/mujoco/go2_stairs_static/scene_single_step.xml` | 后续单级台阶课程起点 | 已验证 |
| 平地站立/Trot 基线 | `simulations/mujoco/go2_stand_flat/` | 确定性控制器与步态基线 | 已验证 |

## 资产约束

- `models/unitree_go2/` 视为只读官方资产；实验修改放在 `simulations/`。
- 原始 STL 保持不变；派生尺寸写入 `geometry_estimate.json`。
- 场景目录不再保存重复 OBJ/STL，避免同一模型出现多份不同版本。
- 完整钢梯的 STL 只负责显示，碰撞由场景中的 49 个分类 box 承担；这些碰撞实体是后续楼梯训练的必需部分，不得删除。
- 外部参考工程已移除，不作为当前运行或训练依赖。

## 文件校验

钢梯 STL 与官方 Go2 模型的校验值由脚本读取并在需要时重新计算。当前回归入口：

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\check_environment.py
D:\Python313\python.exe D:\桌面\机器狗\scripts\validate_collision_scenes.py
```

## 尚未导入的资料

- 教师任务书或正式评分标准；
- 楼梯 CAD 图纸、现场实测尺寸和材质参数；
- Go2 实机试验数据；
- 强化学习训练预算和最终验收指标。
