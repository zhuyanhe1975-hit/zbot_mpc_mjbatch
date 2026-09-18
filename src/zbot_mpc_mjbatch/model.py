"""Load the ZBot model with a numerically stable MuJoCo integrator."""

from pathlib import Path

import mujoco
import numpy as np

ASSET_DIR = Path(__file__).parent / "assets"
SCENE_XML = ASSET_DIR / "scene.xml"


def load_model() -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    # The source model's position servos are unstable with MuJoCo's Euler default.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    if (model.nq, model.nv, model.nu) != (13, 12, 6):
        raise ValueError("expected the ZBot model with one free root and six hinges")
    return model


def standing_joint_targets(model: mujoco.MjModel) -> np.ndarray:
    keyframe = model.key("stand")
    return np.asarray(
        [keyframe.qpos[model.joint(model.actuator(i).name).qposadr[0]] for i in range(model.nu)],
        dtype=np.float64,
    )
