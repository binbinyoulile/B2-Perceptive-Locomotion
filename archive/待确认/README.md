# 待确认归档

整理日期：2026-10-25

此目录保存“目前不必进入主工作区，但可能有参考价值、尚不能确认可删除”的内容。

## 内容

- `go2_omniverse_logs/`：从
  `reinforcement_learning/upstream/go2_omniverse/logs/` 移出的两个训练检查点，
  来源为外部 `go2_omniverse` 参考工程。它们不属于当前 MuJoCo 静态场景，
  也不是当前可运行链路的依赖。
  - `model_2050.pt`：G1 rough 地形训练检查点，约 7.48 MB；
  - `model_7850.pt`：Unitree Go2 rough 地形训练检查点，约 6.56 MB。

## 处理建议

- 若要复现或研究外部工程的训练结果，再移回原目录或单独建立实验记录；
- 若后续确认只保留其环境与地形配置，可以在负责人确认后删除。
