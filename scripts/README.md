# 辅助脚本

| 脚本 | 作用 |
|---|---|
| `check_environment.py` | 检查依赖、根目录模型资产、场景编译和初始接触 |
| `validate_collision_scenes.py` | 验证完整钢梯和单级台阶的碰撞几何与动力学接触 |
| `analyze_stairs_geometry.py` | 从原始 STL 重建楼梯几何 JSON |

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\check_environment.py
D:\Python313\python.exe D:\桌面\机器狗\scripts\validate_collision_scenes.py
D:\Python313\python.exe D:\桌面\机器狗\scripts\analyze_stairs_geometry.py
```

MuJoCo 脚本先切换到对应 XML 目录，再使用相对路径加载，避免 Windows 中文绝对路径兼容问题。
