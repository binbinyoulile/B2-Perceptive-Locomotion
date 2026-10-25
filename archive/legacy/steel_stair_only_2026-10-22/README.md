# MuJoCo STL 导入测试（竖向摆放）

## 结果

- 源文件：`D:\桌面\steel_stair.STL`
- 格式：二进制 STL
- 三角面：41,290
- 顶点：20,258
- 原始尺寸：4086.87 × 1406.85 × 936.00 mm
- XML 缩放：0.001，模型尺寸约 4.087 × 1.407 × 0.936 m
- 已按需求绕 Z 轴旋转 90°：长边沿 Y 轴，俯视时为竖向
- 已验证：MuJoCo 3.16.0 可编译并完成 `mj_forward`

## 可视化

双击 `打开MuJoCo模型.bat`。窗口打开后可用鼠标：

- 左键拖动：旋转视角
- 右键拖动：平移
- 滚轮：缩放

也可以直接运行：

```powershell
D:\Python313\python.exe "D:\codex\2026-10-22\new-chat\outputs\mujoco_steel_stair\visualize.py"
```

或使用 MuJoCo 自带查看器：

```powershell
D:\Python313\python.exe -m mujoco.viewer --mjcf "D:\codex\2026-10-22\new-chat\outputs\mujoco_steel_stair\steel_stair.xml"
```

验证模型：

```powershell
D:\Python313\python.exe "D:\codex\2026-10-22\new-chat\outputs\mujoco_steel_stair\validate.py"
```

## 碰撞说明

当前 XML 将网格设为可视化几何体（`contype="0" conaffinity="0"`）。MuJoCo 能显示任意三角网格，但普通网格碰撞会使用整体凸包。这个模型有 35 个互不相连的网格部件，且楼梯和栏杆是凹形结构；如果直接开启碰撞，阶梯凹槽会被凸包填平，因此不能作为可靠的可攀爬楼梯碰撞体。

若下一步需要机器人/物体与楼梯真实接触，应为踏步、平台、立柱和栏杆建立简化碰撞几何体（长方体/凸多面体），或先做凸分解后导入。
