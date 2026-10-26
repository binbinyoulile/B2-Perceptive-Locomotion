# 环境与运行说明

## 当前环境

| 项目 | 当前值 |
|---|---|
| 操作系统 | Windows |
| 项目目录 | `D:\桌面\机器狗` |
| Python | 3.13.12，`D:\Python313\python.exe` |
| MuJoCo | 3.16.0 |
| NumPy | 2.5.x |
| Pillow | 13.x |
| trimesh | 5.2.0 |
| matplotlib | 3.11.2 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU，6 GB |
| Gymnasium / PyTorch / Stable-Baselines3 | 尚未安装 |

当前依赖足以运行实时仿真、几何解析和碰撞验证，不包含强化学习训练框架。

## 实时运行

```powershell
cd D:\桌面\机器狗; D:\Python313\python.exe .\run_demo.py
```

当前项目直接使用全局 Python，不要求创建或激活虚拟环境。

## 基础依赖

```powershell
D:\Python313\python.exe -m pip install -r D:\桌面\机器狗\requirements-static.txt
```

## 环境检查

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\check_environment.py
```

检查内容：

- Python、MuJoCo、NumPy 和 Pillow；
- 根目录 `models/unitree_go2/` 与 `models/stairs/` 资产；
- 官方 Go2 MJCF 编译，`nq/nv/nu = 19/18/12`；
- 完整楼梯场景和单级台阶场景编译；
- `stairs_home` 关键帧、Go2 初始位置；
- 视觉 STL 关闭碰撞；
- 四足初始接触。

## 碰撞验证

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\validate_collision_scenes.py
```

检查 49 个分类碰撞体、7 个落脚面、5.26 mm 真实缝隙、平台净空、单级台阶初始接触和 1 s 动力学接触。该验证只证明静态几何和短时接触正确，不证明机器狗会上楼。

## 楼梯几何

```powershell
D:\Python313\python.exe D:\桌面\机器狗\scripts\analyze_stairs_geometry.py
```

原始 STL 只读；输出写入 `models/stairs/geometry_estimate.json`。

## Windows 中文路径

MuJoCo 对含中文的绝对 XML 路径支持不稳定。现有脚本会切换到场景目录并使用相对 XML 路径。新增脚本必须沿用这一方式，不能把 `D:\桌面\...` 写进 `from_xml_path()`。

## 强化学习环境

训练框架建议单独安装：

```powershell
D:\Python313\python.exe -m pip install gymnasium stable-baselines3 tensorboard
```

安装前先确认 PyTorch 与 Python 3.13、CUDA 驱动兼容。若版本不兼容，再考虑 Python 3.11/3.12 的独立训练环境；不要为了安装训练库破坏当前已通过验证的实时仿真环境。
