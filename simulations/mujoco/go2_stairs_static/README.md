# Go2 完整钢梯场景

## 文件

- `scene.xml`：Go2 + 完整钢梯，当前实时入口使用此场景。
- `scene_single_step.xml`：0.08 m 单级低台阶课程，用于后续控制/RL 测试。
- `preview.png`：当前场景预览。

## 资产引用

场景不复制模型文件：

```text
Go2 MJCF: ../../../models/unitree_go2/go2.xml
Go2 mesh: ../../../models/unitree_go2/assets/
钢梯 STL: models/stairs/steel_stair.STL
```

MuJoCo 的 `meshdir` 会参与场景 mesh 路径解析，因此钢梯路径写作 `../../stairs/steel_stair.STL`。修改相对路径后必须重新编译两个场景。

## 几何与碰撞

- 钢梯为 3 级上行、2.5 m 平台、3 级下行。
- STL 只负责显示，`contype=0`、`conaffinity=0`。
- 碰撞由 49 个分类 box 承担。
- 7 个落脚薄板保留约 5.26 mm 的真实缝隙。
- 足端/踏面摩擦暂取 `0.8 / 0.02 / 0.01`，真机前需标定。

## 验证

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\check_environment.py
D:\Python313\python.exe D:\桌面\机器狗\scripts\validate_collision_scenes.py
```
