"""Foot support polygons and ground-plane center of pressure."""

from __future__ import annotations

import mujoco
import numpy as np


def mesh_pair_distance(raw_distance: np.ndarray, fromto: np.ndarray) -> np.ndarray:
    """Repair zero mesh distances that still have separated witness points."""
    witness_gap = np.linalg.norm(fromto[..., 3:] - fromto[..., :3], axis=-1)
    return np.where((raw_distance == 0) & (witness_gap > 1e-4), witness_gap, raw_distance)


def center_of_pressure(
    position_w: np.ndarray,
    force_w: np.ndarray,
    torque_w: np.ndarray,
    minimum_normal_force_n: float = 1.0,
) -> np.ndarray:
    """Intersect the net ground-contact wrench with the z=0 plane.

    The netforce contact sensors report a synthetic contact in world coordinates.
    Their torque is about ``position_w``. A contact below the force threshold has
    no meaningful center of pressure and is returned as NaN.
    """
    moment_at_origin = np.cross(position_w, force_w) + torque_w
    normal_force = force_w[..., 2]
    valid = normal_force > minimum_normal_force_n
    cop = np.full(force_w.shape[:-1] + (2,), np.nan)
    np.divide(-moment_at_origin[..., 1], normal_force, out=cop[..., 0], where=valid)
    np.divide(moment_at_origin[..., 0], normal_force, out=cop[..., 1], where=valid)
    return cop


def _convex_hull_indices(points: np.ndarray) -> np.ndarray:
    """Return indices of the counterclockwise 2D convex hull."""
    order = np.lexsort((points[:, 1], points[:, 0]))

    def cross(a: int, b: int, c: int) -> float:
        ab = points[b] - points[a]
        ac = points[c] - points[a]
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    lower: list[int] = []
    for index in order:
        i = int(index)
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 1e-12:
            lower.pop()
        lower.append(i)
    upper: list[int] = []
    for index in reversed(order):
        i = int(index)
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 1e-12:
            upper.pop()
        upper.append(i)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.int32)
    if len(hull) < 3:
        raise ValueError("foot collision mesh has no usable sole polygon")
    return hull


def signed_polygon_margin(point_xy: np.ndarray, polygon_xy: np.ndarray) -> np.ndarray:
    """Minimum signed edge half-space margin: positive inside, negative outside."""
    next_vertex = np.roll(polygon_xy, -1, axis=-2)
    edge = next_vertex - polygon_xy
    offset = point_xy[..., None, :] - polygon_xy
    cross = edge[..., 0] * offset[..., 1] - edge[..., 1] * offset[..., 0]
    area_twice = np.sum(
        polygon_xy[..., 0] * next_vertex[..., 1]
        - next_vertex[..., 0] * polygon_xy[..., 1],
        axis=-1,
    )
    orientation = np.where(area_twice >= 0, 1.0, -1.0)
    distance = orientation[..., None] * cross / np.maximum(np.linalg.norm(edge, axis=-1), 1e-9)
    return np.min(distance, axis=-1)


class FootSupportGeometry:
    """Sole boundary from the collision mesh vertices near the standing floor."""

    def __init__(self, model: mujoco.MjModel) -> None:
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)
        mujoco.mj_forward(model, data)
        self.geom_ids = tuple(model.geom(f"foot_{foot}_collisions").id for foot in (0, 1))
        polygons = []
        for geom_id in self.geom_ids:
            mesh_id = int(model.geom_dataid[geom_id])
            first = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = model.mesh_vert[first : first + count]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            world = vertices @ rotation.T + data.geom_xpos[geom_id]
            bottom = np.flatnonzero(world[:, 2] <= world[:, 2].min() + 0.004)
            hull = _convex_hull_indices(world[bottom, :2])
            polygons.append(vertices[bottom[hull]].copy())
        self.local_polygons = tuple(polygons)

    def polygon_world(
        self, geom_pos_w: np.ndarray, geom_mat_w: np.ndarray, foot: int
    ) -> np.ndarray:
        """Sole polygon in world coordinates over arbitrary leading axes."""
        rotation = geom_mat_w.reshape(geom_mat_w.shape[:-1] + (3, 3))
        return (
            np.einsum("...ij,kj->...ki", rotation, self.local_polygons[foot])
            + geom_pos_w[..., None, :]
        )

    def margin(
        self,
        point_xy: np.ndarray,
        geom_pos_w: np.ndarray,
        geom_mat_w: np.ndarray,
        foot: int,
    ) -> np.ndarray:
        """Signed point-to-sole margin for one foot over arbitrary leading axes."""
        return signed_polygon_margin(
            point_xy, self.polygon_world(geom_pos_w, geom_mat_w, foot)[..., :2]
        )

    def double_support_margin(
        self, point_xy: np.ndarray, geom_pos_w: np.ndarray, geom_mat_w: np.ndarray
    ) -> float:
        """Signed ZMP margin within the convex hull of both sole polygons."""
        points = np.concatenate(
            [
                self.polygon_world(geom_pos_w[foot], geom_mat_w[foot], foot)[:, :2]
                for foot in (0, 1)
            ]
        )
        hull = points[_convex_hull_indices(points)]
        return float(signed_polygon_margin(point_xy, hull))
