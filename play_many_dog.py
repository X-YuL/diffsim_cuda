# -*- coding: utf-8 -*-
"""
play.py - 使用 many_dog_walk.py 训练好的策略，在 Isaac Gym 中实时播放多狗行走。

用法示例：
    python play.py                           # 默认用 4 只狗 + 随机速度指令
    python play.py --num_envs 16            # 16 只狗一起跑
    python play.py --no_rand_cmd            # 固定 0.2 m/s 行走
    python play.py --gait_mode 1            # 固定 trot 步态
    python play.py --weights your_ckpt.pth  # 指定权重文件
"""

import os
import argparse


from many_dog_walk_vectorized import EnvCfg, RealQuadEnv, Policy, PURE_PAPER_MODE

import torch


def load_policy(weight_path: str,
                device: torch.device,
                dim_obs: int = 36,
                dim_action: int = 12) -> Policy:
    """构建 Policy，并从权重文件加载参数（如果存在）"""
    policy = Policy(dim_obs, dim_action).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"✅ 已从 {weight_path} 加载策略权重。")
    else:
        print(f"⚠️ 未找到权重文件 {weight_path}，将使用随机初始化策略。")
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default="quad_diffsim_srbd_align_multi_robot.pth",
        help="策略权重文件路径（默认使用训练脚本保存的 pth）"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="并行机器狗数量（不填则使用 EnvCfg 默认值）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="运行设备：cuda / cpu"
    )
    parser.add_argument(
        "--no_rand_cmd",
        action="store_true",
        help="关闭随机速度指令，使用固定 0.2 m/s"
    )
    parser.add_argument(
        "--gait_mode",
        type=int,
        default=None,
        help="步态模式：-1=每个 env 随机, 0=stand, 1=trot, 2=pace, 3=bound"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="最多运行多少步；0 表示无限运行，直到关闭 viewer"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    # -------- 构建 EnvCfg --------
    cfg = EnvCfg()
    # 并行狗数
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs

    # 播放模式：必须打开 viewer
    cfg.use_viewer = True
    cfg.use_gpu_pipeline = True

    # 速度指令开关
    if args.no_rand_cmd:
        cfg.rand_cmd = False  # 固定 vx_star = 0.2
    # 步态模式（可选）
    if args.gait_mode is not None:
        cfg.gait_mode = args.gait_mode

    # -------- 构建环境 & 策略 --------
    env = RealQuadEnv(cfg, device=device)
    env.reset()  # 初始化到“站立 + 随机/固定指令”

    policy = load_policy(args.weights, device)
    B = env.B
    dim_action = 12

    # 按照 many_dog_walk.train 的逻辑做 action_hold / 平滑
    hx = None
    a_prev = torch.zeros(B, dim_action, device=device)
    hx_hold = None

    t = 0
    steps_done = 0

    print("🎮 开始播放 DiffSim Quadruped 多狗环境")
    print(f"   并行机器狗数量 B = {B}")
    print("   提示：在 Isaac Gym viewer 窗口中自由移动摄像机，关闭窗口即可退出。")

    try:
        while True:
            # 观测：与训练完全一致 (B, 36)
            with torch.no_grad():
                s = env.get_obs().to(device)

                if PURE_PAPER_MODE:
                    # 论文版：只做 action_hold，不做动作平滑
                    if (t % cfg.action_hold) == 0:
                        a, hx = policy(s, hx)  # (B, 12)
                        a_prev = a
                        hx_hold = hx
                    else:
                        a = a_prev
                        hx = hx_hold
                else:
                    # 工程版：action_hold + 简单 EMA 平滑
                    if (t % cfg.action_hold) == 0:
                        a_raw, hx = policy(s, hx)  # (B, 12)
                        a_smooth = 0.7 * a_prev + 0.3 * a_raw
                        a_prev = a_smooth
                        hx_hold = hx
                        a = a_smooth
                    else:
                        a = a_prev
                        hx = hx_hold

            # 环境前进一步（里面会自动调用 IsaacGym simulate + viewer 刷新）
            obs, extra, q_err, q_ref = env.step(a)

            done = extra["done"]  # (B,)
            if done.any():
                fallen_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                print(f"[play] 局部 reset envs: {fallen_ids.cpu().tolist()}")
                env.reset_envs(fallen_ids)
                # 如果以后换成 RNN 策略，可以在这里对对应 env 的 hidden state 清零
                # 目前 Policy 不用隐藏状态，可以忽略 hx

            t += 1
            steps_done += 1

            # viewer 被关掉就退出
            if env.viewer is None:
                print("Viewer 已关闭，退出 play。")
                break

            # 限制最大步数（可选）
            if args.max_steps > 0 and steps_done >= args.max_steps:
                print(f"达到最大步数 {args.max_steps}，退出 play。")
                break

    except KeyboardInterrupt:
        print("收到 Ctrl+C，退出 play。")


if __name__ == "__main__":
    main()
