"""FSM Teacher + PPO 残差模型的 MuJoCo 3D 实时回放入口。

文件目的：
    加载本轮 5 万步 Teacher-Student 模型，在 0.04 m 单级台阶场景中打开
    MuJoCo passive viewer，直接观察 FSM 教师、PPO 微调量、IK 裁剪和关节
    限位诊断。该脚本只负责可视化，不训练、不改模型。

整体输入：
    Stable-Baselines3 PPO ``.zip`` 模型；默认读取
    ``experiments/checkpoints/ppo_fsm_residual_50k/ppo_fsm_residual_50k_latest.zip``。
整体输出：
    一个实时 3D 窗口，以及每个控制步的 FSM 状态、动作/残差、奖励、关节使用率
    和 IK 裁剪终端日志。关闭窗口或按 Ctrl+C 结束。

依赖与版本要求：
    Python 3.10+、MuJoCo 3.x（需要可用的 OpenGL 图形环境）、Gymnasium 1.x、
    Stable-Baselines3 2.x、PyTorch、NumPy。

运行方式：
    ``python .\\reinforcement_learning\\eval_fsm_residual.py``。
    如需自动冒烟测试，可追加 ``--max-steps 1``。

关键假设：
    1. 模型观测空间必须与环境当前 175 维观测完全一致；
    2. 环境使用与训练相同的 0.04 m 台阶和域随机化默认参数；
    3. 一个控制步包含 ``n_substeps`` 个物理子步，回放按 50 Hz 控制周期限速；
    4. 回合结束（登顶、摔倒、超时等）后自动 reset，不关闭窗口。

已知限制：
    ``launch_passive`` 需要桌面图形会话；无 OpenGL/远程无显示环境时窗口无法创建。
    回合自动 reset 会让每次回放使用递增种子，因此不是单回合确定性的录像。

重要性等级：A（本轮最终人工视觉验收入口）。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REINFORCEMENT_DIR = PROJECT_ROOT / "reinforcement_learning"
sys.path.insert(0, str(REINFORCEMENT_DIR))

from src.go2_env import (  # noqa: E402  (必须先设置项目导入路径)
    DEFAULT_HEIGHTMAP_NOISE_STD_M,
    DEFAULT_STEP_HEIGHT_RANDOMIZATION_M,
    DEFAULT_STEP_X_RANDOMIZATION_M,
    Go2Env,
)

SCENE_PATH = (
    PROJECT_ROOT
    / "simulations"
    / "mujoco"
    / "go2_stairs_static"
    / "scene_single_step_004.xml"
)
MODEL_TAG = "ppo_fsm_residual_50k"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "checkpoints"
    / MODEL_TAG
    / f"{MODEL_TAG}_latest.zip"
)


def _parse_args() -> argparse.Namespace:
    """解析模型、随机种子、设备和自动退出步数。

    返回：
        argparse.Namespace。``max_steps=0`` 表示持续回放直到关闭窗口；
        正整数主要用于自动化冒烟测试。
    可能异常：
        argparse 会在参数格式错误时抛出 SystemExit；模型是否存在由 main 检查。
    重要性等级：B。
    """
    parser = argparse.ArgumentParser(
        description="FSM + PPO 残差模型 MuJoCo 3D 实时回放"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="PPO zip 模型路径；相对路径按项目根目录解析。",
    )
    parser.add_argument("--seed", type=int, default=42, help="首个回合随机种子。")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="PPO 推理设备，默认 cpu。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="最多执行多少个控制步；0 表示持续回放。",
    )
    args = parser.parse_args()
    if args.seed < 0:
        parser.error("--seed 必须为非负整数")
    if args.max_steps < 0:
        parser.error("--max-steps 必须为非负整数")
    return args


def _resolve_model_path(path: Path) -> Path:
    """将命令行模型路径解析为绝对路径。"""
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _format_vector(values) -> str:
    """把短数值序列格式化为便于终端观察的紧凑文本。"""
    if values is None:
        return "[]"
    return "[" + ",".join(f"{float(value):+.3f}" for value in values) + "]"


def main() -> int:
    """加载模型并打开实时 3D 回放窗口。

    返回：
        0 表示正常退出。模型缺失、观测/动作维度不匹配或 viewer 创建失败时，
        异常会向上传播并由文件末尾打印明确错误。
    副作用：
        推进 MuJoCo 物理、打开图形窗口并持续打印诊断；不修改业务文件。
    重要性等级：A。
    """
    args = _parse_args()
    model_path = _resolve_model_path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"找不到融合模型：{model_path}\n"
            "请先完成 5 万步训练，或通过 --model 指定已有 ZIP。"
        )

    from stable_baselines3 import PPO

    model = PPO.load(str(model_path), device=args.device)
    env = Go2Env(
        scene_path=SCENE_PATH,
        terrain="single_step",
        domain_randomization=True,
        step_x_randomization_m=DEFAULT_STEP_X_RANDOMIZATION_M,
        step_height_randomization_m=DEFAULT_STEP_HEIGHT_RANDOMIZATION_M,
        heightmap_noise_std_m=DEFAULT_HEIGHTMAP_NOISE_STD_M,
        controller_mode="fsm_residual",
    )
    try:
        if model.observation_space.shape != env.observation_space.shape:
            raise ValueError(
                "模型观测维度与环境不一致："
                f"model={model.observation_space.shape}, env={env.observation_space.shape}"
            )
        if model.action_space.shape != env.action_space.shape:
            raise ValueError(
                "模型动作维度与环境不一致："
                f"model={model.action_space.shape}, env={env.action_space.shape}"
            )

        print(f"[回放] 模型: {model_path}", flush=True)
        print(f"[回放] 场景: {SCENE_PATH}", flush=True)
        print("[回放] 控制: target = FSM + 0.10 * PPO；关闭窗口结束。", flush=True)

        observation, reset_info = env.reset(seed=args.seed)
        episode_index = 1
        episode_reward = 0.0
        step_in_episode = 0
        total_steps = 0

        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.lookat[:] = env.data.qpos[:3]
            viewer.cam.distance = 1.8
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -18.0
            viewer.sync()

            control_period_s = env.model.opt.timestep * env.n_substeps
            next_tick_s = time.perf_counter()
            while viewer.is_running():
                action, _ = model.predict(observation, deterministic=True)
                action_array = np.asarray(action, dtype=np.float64).reshape(-1)
                observation, reward, terminated, truncated, info = env.step(action_array)
                episode_reward += float(reward)
                step_in_episode += 1
                total_steps += 1

                if step_in_episode % 10 == 0 or terminated or truncated:
                    state = str(info.get("fsm_state", "UNKNOWN"))
                    residual = info.get("residual_joint_rad")
                    usage = float(info.get("joint_limit_max_usage_ratio", 0.0))
                    ik_clip_count = int(info.get("fsm_ik_clip_count", 0))
                    print(
                        f"[回放 {episode_index:02d}] step={step_in_episode:3d} | "
                        f"state={state:16s} | reward={float(reward):+.3f} | "
                        f"joint_usage={usage:.3f} | IK裁剪={ik_clip_count} | "
                        f"residual={_format_vector(residual)}",
                        flush=True,
                    )

                if terminated or truncated:
                    reason = str(info.get("termination_reason", "unknown"))
                    print(
                        f"[回合结束] episode={episode_index} reason={reason} "
                        f"steps={step_in_episode} reward={episode_reward:+.3f}",
                        flush=True,
                    )
                    episode_index += 1
                    observation, reset_info = env.reset(seed=args.seed + episode_index - 1)
                    episode_reward = 0.0
                    step_in_episode = 0

                if args.max_steps > 0 and total_steps >= args.max_steps:
                    print(f"[回放] 达到 --max-steps={args.max_steps}，正常退出。", flush=True)
                    break

                viewer.cam.lookat[:] = env.data.qpos[:3]
                viewer.sync()
                next_tick_s += control_period_s
                remaining_s = next_tick_s - time.perf_counter()
                if remaining_s > 0.0:
                    time.sleep(remaining_s)
                else:
                    next_tick_s = time.perf_counter()
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"[回放错误] {error}", file=sys.stderr, flush=True)
        raise
