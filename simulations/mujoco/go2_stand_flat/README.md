# Go2 平地控制基线

本目录保留两个确定性控制基线，供后续强化学习对照：

- `stand_pd.py`：固定关节目标 PD 站立；
- `trot_open_loop.py`：开环 Trot 和 0.08 m 单级台阶测试；
- `scene.xml`：无楼梯平地场景。

## 运行

```powershell
D:\Python313\python.exe D:\桌面\机器狗\simulations\mujoco\go2_stand_flat\stand_pd.py
D:\Python313\python.exe D:\桌面\机器狗\simulations\mujoco\go2_stand_flat\trot_open_loop.py
```

单级台阶：

```powershell
D:\Python313\python.exe D:\桌面\机器狗\simulations\mujoco\go2_stand_flat\trot_open_loop.py --scene single-step --duration 20
```

脚本会在当前目录生成 CSV、JSON 或 PNG 结果。它们是可复现实验输出，不作为固定源码提交；重要结果应整理后放入 `experiments/reports/`。

## 已知限制

- 当前 Trot 是开环关节轨迹，不是完整平衡控制器。
- 机身辅助力只服务于演示，不可直接迁移到真机。
- 单级台阶测试用于暴露问题，不代表已经具备上楼能力。
- 摩擦、质量和控制延迟尚未做域随机化或真机标定。
