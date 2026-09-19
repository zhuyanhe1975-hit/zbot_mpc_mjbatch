"""Replay closed-loop ZBot stepping in MuJoCo viewer or export an MP4 video."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

import mujoco

from .model import load_model
from .stepping import StepRun, simulate_steps


def render_video(
    run: StepRun,
    output: Path,
    width: int = 800,
    height: int = 600,
    speed: float = 1.0,
) -> None:
    """Render measured simulation poses and stream frames directly to ffmpeg."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    if width <= 0 or height <= 0 or width % 2 or height % 2:
        raise ValueError("video width and height must be positive even numbers")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to write MP4 video")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model()
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = (0.0, 0.0, 0.25)
    camera.distance = 1.0
    camera.azimuth = 135
    camera.elevation = -20
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
    feet = [model.geom(f"foot_{i}_visuals").id for i in (0, 1)]
    frame_dt = float(run.times_s[1] - run.times_s[0]) if len(run.times_s) > 1 else float(run.times_s[0])
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{speed / frame_dt:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    renderer = mujoco.Renderer(model, height=height, width=width)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert process.stdin is not None
        assert process.stderr is not None
        for i, qpos in enumerate(run.qpos):
            data.qpos[:] = qpos
            data.qvel[:] = run.qvel[i]
            data.ctrl[:] = run.ctrl[i]
            data.time = float(run.times_s[i])
            swing = int(run.swing_feet[i])
            model.geom_rgba[feet[swing]] = (1.0, 0.35, 0.12, 1.0)
            model.geom_rgba[feet[1 - swing]] = (0.15, 0.55, 1.0, 1.0)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=option)
            process.stdin.write(renderer.render().tobytes())
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed: {error.strip()}")
    except BaseException:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        renderer.close()


def main() -> None:
    from .warmstart import LinearWarmStartPolicy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=4.2)
    parser.add_argument("--candidates", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--steps", type=int, help="stop after this many completed steps")
    parser.add_argument("--hold-seconds", type=float, default=0.1)
    parser.add_argument("--warm-start", type=Path, help="learned MPC proposal policy (.npz)")
    parser.add_argument("--speed", type=float, default=1.0, help="video speed multiplier")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--output", type=Path, default=Path("dist/zbot_stepping.mp4"))
    parser.add_argument("--viewer", action="store_true", help="replay in MuJoCo's interactive viewer")
    parser.add_argument("--once", action="store_true", help="close the viewer after one replay")
    args = parser.parse_args()
    if not math.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be finite and positive")
    if args.width <= 0 or args.height <= 0 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be positive even numbers")
    try:
        policy = LinearWarmStartPolicy.load(args.warm_start) if args.warm_start else None
        run = simulate_steps(
            args.frequency,
            args.seconds,
            args.candidates,
            args.iterations,
            args.threads,
            args.seed,
            policy,
            args.steps,
            args.hold_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.viewer:
        from .viewer import play_viewer

        print(
            json.dumps(
                {**run.summary(), "landing_times_s": run.landing_times_s.round(3).tolist()},
                indent=2,
            ),
            flush=True,
        )
        print("MuJoCo viewer：空格暂停/继续，N 单帧前进，R 从头播放。关闭窗口退出。", flush=True)
        try:
            play_viewer(run, args.speed, args.once)
        except KeyboardInterrupt:
            pass
        return
    render_video(run, args.output, args.width, args.height, args.speed)
    print(
        json.dumps(
            {
                **run.summary(),
                "landing_times_s": run.landing_times_s.round(3).tolist(),
                "video": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    if run.stop_reason is not None:
        raise SystemExit(run.stop_reason)


if __name__ == "__main__":
    main()
