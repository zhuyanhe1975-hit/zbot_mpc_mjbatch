import mujoco
import numpy as np

from zbot_mpc_mjbatch.model import load_model


def test_subtree_com_jacobian_matches_joint_perturbations():
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
    mujoco.mj_forward(model, data)
    root = model.body("foot_0").id
    jac_com = np.zeros((3, model.nv))
    jac_support = np.zeros((3, model.nv))
    mujoco.mj_jacSubtreeCom(model, data, jac_com, root)
    mujoco.mj_jacBody(model, data, jac_support, None, root)
    expected = (jac_com - jac_support)[:, 6:]
    baseline = data.subtree_com[root].copy() - data.body("foot_0").xpos.copy()
    for joint in (0, 2, 5):
        perturbed = mujoco.MjData(model)
        perturbed.qpos[:] = data.qpos
        perturbed.qpos[7 + joint] += 1e-6
        mujoco.mj_forward(model, perturbed)
        observed = (
            perturbed.subtree_com[root] - perturbed.body("foot_0").xpos - baseline
        ) / 1e-6
        np.testing.assert_allclose(observed, expected[:, joint], atol=1e-6)
