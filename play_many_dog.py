# -*- coding: utf-8 -*-
"""
play.py - Use trained policy to play multi-dog walking in Isaac Gym in real-time.

Usage examples:
    python play_many_dog.py                           # Default: 4 dogs + random velocity commands
    python play_many_dog.py --num_envs 16            # 16 dogs running together
    python play_many_dog.py --no_rand_cmd            # Fixed 0.5 m/s walking
    python play_many_dog.py --gait_mode 1            # Fixed trot gait
    python play_many_dog.py --weights your_ckpt.pth  # Specify weight file
"""

import os
import argparse
import shutil
import subprocess

# ⚠️ CRITICAL: Isaac Gym must be imported BEFORE torch
try:
    from isaacgym import gymapi
except Exception:
    pass

import torch

from config import EnvCfg, PURE_PAPER_MODE
from env import RealQuadEnv
from policy import Policy


class FfmpegVideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: int):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg was not found on PATH; install ffmpeg to record MP4 video.")

        self.path = path
        self.width = int(width)
        self.height = int(height)
        self.frames = 0

        out_dir = os.path.dirname(os.path.abspath(path))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        cmd = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}",
            "-r", str(int(fps)),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            path,
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame):
        if frame.shape != (self.height, self.width, 3):
            raise ValueError(f"Expected frame shape {(self.height, self.width, 3)}, got {frame.shape}")
        try:
            self.proc.stdin.write(frame.tobytes())
            self.frames += 1
        except BrokenPipeError as exc:
            raise RuntimeError("ffmpeg stopped while writing video frames.") from exc

    def close(self):
        if self.proc.stdin and not self.proc.stdin.closed:
            self.proc.stdin.close()
        stderr = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
        ret = self.proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {ret}: {stderr.strip()}")


def load_policy(weight_path: str,
                device: torch.device,
                dim_obs: int = 36,
                dim_action: int = 12) -> Policy:
    """Build Policy and load parameters from weight file (if exists)"""
    policy = Policy(dim_obs, dim_action).to(device)
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location=device)
        policy.load_state_dict(state)
        print(f"✅ Loaded policy weights from {weight_path}.")
    else:
        print(f"⚠️ Weight file {weight_path} not found, using randomly initialized policy.")
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join("results", "quad_diffsim_srbd_align_multi_robot.pth"),
        help="Policy weight file path (default: pth saved by training script)"
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel quadrupeds (if not specified, use EnvCfg default)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on: cuda / cpu"
    )
    parser.add_argument(
        "--no_rand_cmd",
        action="store_true",
        help="Disable random velocity commands, use fixed 0.2 m/s"
    )
    parser.add_argument(
        "--gait_mode",
        type=int,
        default=None,
        help="Gait mode: -1=random per env, 0=stand, 1=trot, 2=pace, 3=bound"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Maximum steps to run; 0 means run indefinitely until viewer is closed"
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Run without the Isaac Gym viewer; useful for SSH/headless recording"
    )
    parser.add_argument(
        "--record_video",
        type=str,
        default="",
        help="Write an MP4 video to this path using an Isaac Gym camera sensor"
    )
    parser.add_argument(
        "--video_width",
        type=int,
        default=1280,
        help="Recording width in pixels"
    )
    parser.add_argument(
        "--video_height",
        type=int,
        default=720,
        help="Recording height in pixels"
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=50,
        help="Frames per second for the output video"
    )
    parser.add_argument(
        "--video_every",
        type=int,
        default=10,
        help="Capture one video frame every N simulation steps"
    )
    parser.add_argument(
        "--fixed_camera",
        action="store_true",
        help="Use the fixed overview camera for recording instead of following robot 0"
    )
    parser.add_argument(
        "--asset_root",
        type=str,
        default=None,
        help="Directory containing go2_description.urdf and dae/ assets"
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.record_video and args.video_every <= 0:
        raise ValueError("--video_every must be positive when recording video.")

    # -------- Build EnvCfg --------
    cfg = EnvCfg()
    # Number of parallel dogs
    if args.num_envs is not None:
        cfg.num_envs = args.num_envs

    # Play mode: viewer is optional when using the off-screen recorder
    viewer_requested = not args.no_viewer
    cfg.use_viewer = viewer_requested
    cfg.use_gpu_pipeline = True
    if args.asset_root is not None:
        cfg.asset_root = args.asset_root

    # Velocity command switch
    if args.no_rand_cmd:
        cfg.rand_cmd = False  # Fixed vx_star = 0.2
    # Gait mode (optional)
    if args.gait_mode is not None:
        cfg.gait_mode = args.gait_mode

    # -------- Build environment & policy --------
    env = RealQuadEnv(cfg, device=device)
    env.reset()  # Initialize to “standing + random/fixed commands”

    video_writer = None
    if args.record_video:
        env.create_recording_camera(
            width=args.video_width,
            height=args.video_height,
            follow=not args.fixed_camera,
        )
        video_writer = FfmpegVideoWriter(
            args.record_video,
            width=args.video_width,
            height=args.video_height,
            fps=args.video_fps,
        )
        print(f"[video] Recording to {args.record_video} at {args.video_width}x{args.video_height}@{args.video_fps}fps")

    policy = load_policy(args.weights, device)
    B = env.B
    dim_action = 12

    # Follow many_dog_walk.train logic for action_hold / smoothing
    hx = None
    a_prev = torch.zeros(B, dim_action, device=device)
    hx_hold = None

    t = 0
    steps_done = 0

    print("🎮 Starting DiffSim Quadruped multi-dog environment playback")
    print(f"   Number of parallel quadrupeds B = {B}")
    if env.viewer is not None:
        print("   Tip: Move camera freely in Isaac Gym viewer window, close window to exit.")
    elif video_writer is not None:
        print("   Viewer disabled; recording video from the Isaac Gym camera sensor.")
        if args.max_steps <= 0:
            print("   Note: --max_steps is 0, so stop the run with Ctrl+C when you have enough footage.")

    try:
        while True:
            # Observation: exactly same as training (B, 36)
            with torch.no_grad():
                s = env.get_obs().to(device)

                if PURE_PAPER_MODE:
                    # Paper version: only action_hold, no action smoothing
                    if (t % cfg.action_hold) == 0:
                        a, hx = policy(s, hx)  # (B, 12)
                        a_prev = a
                        hx_hold = hx
                    else:
                        a = a_prev
                        hx = hx_hold
                else:
                    # Engineering version: action_hold + simple EMA smoothing
                    if (t % cfg.action_hold) == 0:
                        a_raw, hx = policy(s, hx)  # (B, 12)
                        a_smooth = 0.7 * a_prev + 0.3 * a_raw
                        a_prev = a_smooth
                        hx_hold = hx
                        a = a_smooth
                    else:
                        a = a_prev
                        hx = hx_hold

            # Environment step forward (automatically calls IsaacGym simulate + viewer refresh)
            obs, extra, q_err, q_ref = env.step(a)

            done = extra["done"]  # (B,)
            if done.any():
                fallen_ids = torch.nonzero(done, as_tuple=False).squeeze(-1)
                print(f"[play] Partial reset envs: {fallen_ids.cpu().tolist()}")
                env.reset_envs(fallen_ids)
                # If switching to RNN policy later, can zero out hidden state for corresponding envs here
                # Currently Policy doesn't use hidden state, can ignore hx

            t += 1
            steps_done += 1

            if video_writer is not None and (steps_done % args.video_every) == 0:
                video_writer.write(env.capture_recording_frame())

            # Exit if viewer is closed
            if viewer_requested and env.viewer is None:
                print("Viewer closed, exiting play.")
                break

            # Limit maximum steps (optional)
            if args.max_steps > 0 and steps_done >= args.max_steps:
                print(f"Reached maximum steps {args.max_steps}, exiting play.")
                break

    except KeyboardInterrupt:
        print("Received Ctrl+C, exiting play.")
    finally:
        if video_writer is not None:
            video_writer.close()
            print(f"[video] Saved {video_writer.frames} frames to {args.record_video}.")


if __name__ == "__main__":
    main()
