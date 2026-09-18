"""Interactive MuJoCo viewer replay for a measured stepping trajectory."""

from __future__ import annotations

import math
import threading
import time

import mujoco
import mujoco.viewer

from .model import load_model
from .stepping import StepRun


def play_viewer(run: StepRun, speed: float = 1.0, once: bool = False) -> None:
    """Replay measured states; Space pauses, N advances, and R restarts."""
    if not math.isfinite(speed) or speed <= 0:
        raise ValueError("speed must be finite and positive")
    model = load_model()
    data = mujoco.MjData(model)
    foot_geoms = [model.geom(f"foot_{foot}_visuals").id for foot in (0, 1)]
    frame_dt = float(run.times_s[1] - run.times_s[0]) if len(run.times_s) > 1 else float(run.times_s[0])
    control_lock = threading.Lock()
    paused = False
    advance = False
    restart = False

    def on_key(keycode: int) -> None:
        nonlocal paused, advance, restart
        with control_lock:
            if keycode == 32:  # Space
                paused = not paused
            elif keycode in (78, 110):  # N
                paused = True
                advance = True
            elif keycode in (82, 114):  # R
                restart = True
                advance = True

    before = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=on_key)
    # MuJoCo 3.11's passive handle requests shutdown without joining its
    # render thread. Joining before Python exits avoids a GLFW teardown crash.
    viewer_threads = [thread for thread in threading.enumerate() if thread not in before]
    try:
        with viewer.lock():
            viewer.cam.lookat[:] = (0.0, 0.0, 0.25)
            viewer.cam.distance = 1.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -20
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        viewer.sync()
        index = 0
        deadline = time.perf_counter()
        while viewer.is_running():
            with control_lock:
                if restart:
                    index = 0
                    restart = False
                    deadline = time.perf_counter()
                draw = not paused or advance
                advance = False
            if not draw:
                deadline = time.perf_counter()
                time.sleep(0.02)
                continue
            if index == len(run.times_s):
                if once:
                    break
                index = 0
                deadline = time.perf_counter()
            swing = int(run.swing_feet[index])
            with viewer.lock():
                data.qpos[:] = run.qpos[index]
                data.qvel[:] = run.qvel[index]
                data.ctrl[:] = run.ctrl[index]
                data.time = float(run.times_s[index])
                model.geom_rgba[foot_geoms[swing]] = (1.0, 0.35, 0.12, 1.0)
                model.geom_rgba[foot_geoms[1 - swing]] = (0.15, 0.55, 1.0, 1.0)
                mujoco.mj_forward(model, data)
            viewer.sync()
            index += 1
            deadline += frame_dt / speed
            delay = deadline - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        viewer.close()
        for thread in viewer_threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in viewer_threads):
            raise RuntimeError("MuJoCo viewer did not finish shutting down")
