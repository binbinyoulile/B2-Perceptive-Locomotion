# 环境与运行说明

## 当前已验证环境

| 项目 | 当前值 |
|---|---|
| 操作系统 | Windows |
| 项目目录 | `D:\桌面\机器狗` |
| Python | 3.13.12，`D:\Python313\python.exe` |
| MuJoCo | 3.16.0 |
| NumPy | 2.5.4 |
| Pillow | 13.0.0 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU，6 GB |
| Gymnasium | 未安装 |
| PyTorch | 未安装 |
| Stable-Baselines3 | 未安装 |
| JAX/MJX | 未安装 |
| ROS2 | 当前终端未发现 |
| CMake/colcon | 当前终端未发现 |

当前环境足够运行并验证静态 MuJoCo 场景，不包含强化学习训练依赖。

## Windows 中文路径注意事项

MuJoCo 当前 Windows Python 构建无法可靠通过绝对路径打开 `D:\桌面\机器狗\...\scene.xml`。项目中的 Python 脚本会：

1. 取得脚本所在目录；
2. 执行 `os.chdir(HERE)`；
3. 使用相对文件名 `scene.xml` 编译。

这条路径已经验证可用。新增 MuJoCo 脚本时也应遵循同样方式，不要直接拼接含中文的绝对 XML 路径。

后续如果 WSL、JAX/MJX 或其他工具出现类似的 Unicode 路径问题，可以给项目建立一个 ASCII 路径别名，例如 `D:\robot_dog`，不要复制出第二套项目。

## 项目环境检查

推荐先运行统一检查脚本：

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\check_environment.py"
```

当前已通过的检查包括：

- Python 3.13.12、MuJoCo 3.16.0、NumPy 2.5.4、Pillow 13.0.0；
- 官方 Go2 MJCF 编译，`nq/nv/nu = 19/18/12`；
- Go2 + 楼梯静态场景编译和关键帧复位；
- 楼梯姿态为绕 X 轴 `+90°`，最低点贴地；
- 楼梯世界尺寸 `4.086866 × 0.936000 × 1.406850 m`；
- 四个足端初始与地面接触。

## 楼梯几何提取

```powershell
D:\Python313\python.exe "D:\桌面\机器狗\scripts\analyze_stairs_geometry.py"
```

结果写入 `models/stairs/geometry_estimate.json`，可读说明见 `docs/STAIR_GEOMETRY.md`。
## 静态场景验证

```powershell
cd "D:\桌面\机器狗\simulations\mujoco\go2_stairs_static"
D:\Python313\python.exe validate.py
```

期望结果包括：

```text
nq/nv/nu: 19/18/12
base position: [-0.9    0.468  0.27 ]
stair world size (x/y/z): [4.0868662  0.93600003 1.40685003]
initial floor contacts: 4
compile + keyframe reset + mj_forward: OK
```

## 重新生成预览

```powershell
cd "D:\桌面\机器狗\simulations\mujoco\go2_stairs_static"
D:\Python313\python.exe render_preview.py
```

## 推荐环境策略

### 当前阶段

继续使用现有 Python 3.13 + MuJoCo 3.16 验证模型和编写基础脚本，不立即安装大型训练框架。

### 强化学习阶段

建议建立独立虚拟环境，并先做依赖兼容性检查。Python 3.11 或 3.12 通常更容易匹配教学资料和深度学习生态；如果 Python 3.13 下 PyTorch、Gymnasium、Stable-Baselines3 均能正常工作，也可以继续使用。

候选基础栈：

- MuJoCo + NumPy + Pillow；
- Gymnasium；
- PyTorch；
- Stable-Baselines3 + TensorBoard；
- 可选 JAX/MJX 用于批量仿真；
- 可选 WSL2/Linux 用于 MJX 或 IsaacLab。

在安装前先检查：

1. GPU 驱动和 CUDA 兼容性；
2. PyTorch 是否有对应 Python 版本和 CUDA 构建；
3. MuJoCo 与 Gymnasium 版本兼容性；
4. 单环境每秒步数；
5. 批量环境显存和内存占用。

## 版本与结果记录规则

每次重要实验至少记录：

- Git 提交号；
- Python、MuJoCo、深度学习框架版本；
- GPU 型号与驱动；
- 场景文件和奖励配置哈希；
- 随机种子；
- 训练命令；
- 日志、检查点和评测结果路径。
