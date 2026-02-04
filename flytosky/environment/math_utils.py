from __future__ import annotations

import torch
import numpy as np

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (w, x, y, z format)."""
    conj = q.clone()
    conj[..., 1:] *= -1
    return conj


def rotate_vector_by_quaternion(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion (w, x, y, z format).
    v is in body frame and q is the transform from body to world frame.
    The result is v in world frame.
    """
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotated[..., 1:]

def rotate_vector_by_quaternion_conj(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion (w, x, y, z format).
    v is in world frame and q is the transform from body to world frame.
    The result is v in body frame.
    """
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q_conj, q_v), q)
    return rotated[..., 1:]


def quaternion_error_axis_angle(q_current: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor:
    """
    Compute the axis-angle error between current and desired quaternions.
    Returns the axis-angle representation (3D vector where magnitude is angle).
    Result is in the body frame of q_current.

    q_current, q_desired: quaternions in (w, x, y, z) format
    """
    # q_error = q_desired * q_current^-1 gives rotation from current to desired
    # But we want it in body frame, so: q_error = q_current^-1 * q_desired
    q_current_inv = quaternion_conjugate(q_current)
    q_error = quaternion_multiply(q_current_inv, q_desired)

    # Ensure we take the shortest path (q and -q represent same rotation)
    # If w < 0, negate the quaternion
    sign = torch.sign(q_error[..., 0:1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q_error = q_error * sign

    # Extract axis-angle from quaternion
    # q = [cos(theta/2), sin(theta/2) * axis]
    w = q_error[..., 0:1]
    xyz = q_error[..., 1:]

    # sin(theta/2) = ||xyz||
    sin_half_angle = torch.norm(xyz, dim=-1, keepdim=True)

    # Handle small angles to avoid division by zero
    # For small angles: theta ≈ 2 * sin(theta/2), axis ≈ xyz / sin(theta/2)
    # So axis_angle ≈ 2 * xyz
    small_angle_mask = sin_half_angle < 1e-6

    # For larger angles: theta = 2 * atan2(sin(theta/2), cos(theta/2))
    half_angle = torch.atan2(sin_half_angle, w)
    angle = 2.0 * half_angle

    # axis = xyz / sin(theta/2), axis_angle = angle * axis
    axis_angle = torch.where(
        small_angle_mask,
        2.0 * xyz,  # Small angle approximation
        angle * xyz / (sin_half_angle + 1e-10)  # Normal case
    )

    return axis_angle


def quaternion_from_z_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """
    Create quaternion from yaw angle (rotation about z-axis).
    yaw: tensor of yaw angles in radians
    Returns: quaternion in (w, x, y, z) format
    """
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    z = torch.sin(half_yaw)
    return torch.stack([w, x, y, z], dim=-1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix (w, x, y, z format)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x*y - w*z)
    R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y)
    R[..., 2, 1] = 2 * (y*z + w*x)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return R