# 辅助脚本

| 脚本 | 作用 |
|---|---|
| `check_environment.py` | 检查 Python、MuJoCo、依赖、官方 Go2 模型和静态楼梯场景 |
| `analyze_stairs_geometry.py` | 从静态场景中的 STL 提取可落脚面并生成几何估计 JSON |

## 环境检查

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\check_environment.py"
```

也可以双击项目根目录的 `检查项目环境.bat`。

## 楼梯几何提取

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\analyze_stairs_geometry.py"
```

输出文件：

```text
D:\桌面\机器狗\models\stairs\geometry_estimate.json
```

MuJoCo 相关脚本必须在对应 XML 目录中使用相对路径加载模型，以避开当前 Windows 版本对中文绝对路径的兼容问题。