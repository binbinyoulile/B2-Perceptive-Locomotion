# 控制代码

## 当前内容

`upstream/go2_gait_planner/` 是外部 Go2 步态规划工程，用于阅读以下内容：

- 四足运动学；
- 关节和连杆建模；
- Trot 步态；
- 机身和足端控制；
- ROS2/Unitree 通信接口。

## 使用限制

- 当前未安装 ROS2 和 Unitree 相关依赖；
- 尚未构建或运行；
- `package.xml` 许可证为 TODO；
- 只作为架构和算法参考，核心自研代码应放在本目录的独立子工程中。

## 后续目录建议

```text
control/
├─ upstream/                 外部参考代码
├─ go2_control/              自研控制库
├─ configs/                  控制器参数
└─ tests/                    控制测试
```
