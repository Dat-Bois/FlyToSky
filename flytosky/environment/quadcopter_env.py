from __future__ import annotations

import json
import torch
import numpy as np
import gymnasium as gym
import pufferlib # needs to be imported after gym or else logging breaks???
from pathlib import Path

from typing import Optional, Tuple, Dict, Any

from .math_utils import *
from .track_gen import VectorizedTrackGenerator, TrackSettings
from ..logging import Log

'''
Done:
Update goal logic for passing through waypoint (plane passed normal)
Add logic for sequenced waypoints
Action smoothing (low-pass filter) 
Delta progress reward
Expand observation space to include next N waypoints
Randomized start positions infront of waypoints.
'''

class QuadcopterEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int,
        config_path: str,
        max_episode_length: int,
        dt: float,
        progress_reward_scale: float,
        wp_passed_reward_scale: float,
        action_smoothness_reward_scale: float,
        action_magnitude_reward_scale: float,
        ang_vel_reward_scale: float,
        orientation_reward_scale: float,
        alive_reward_scale: float,
        crash_penalty: float,
        dynamics_randomization_delta: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        num_track_points: int = 10,
        **kwargs
    ):
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32) # should be (-1, 1) but in isaaclab it's inf so we're sticking with that
        # Observations: velocity_body (3) + angular_velocity (3) + gravity_body (3) +
        #               orientation_error_axis_angle (3) + rpm_scaled (4) + future_wp (3*lookahead(3) = 9) = 25
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        self.num_envs = num_envs
        self.num_agents = num_envs  # For PufferLib compatibility
        
        Log.info(f"Initializing QuadcopterEnv with {num_envs} envs on {device}")
        
        super().__init__()

        self.device = torch.device(device)
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.max_episode_length_s = max_episode_length * dt
        
        # reward / penalty scales
        self.progress_reward_scale = progress_reward_scale
        self.wp_passed_reward_scale = wp_passed_reward_scale
        self.action_smoothness_reward_scale = action_smoothness_reward_scale
        self.action_magnitude_reward_scale = action_magnitude_reward_scale
        self.ang_vel_reward_scale = ang_vel_reward_scale
        self.orientation_reward_scale = orientation_reward_scale
        self.alive_reward_scale = alive_reward_scale
        self.crash_penalty = crash_penalty
        
        # dynamics randomization range (percentage)
        self.dynamics_randomization_delta = dynamics_randomization_delta

        # Setup track gen
        self.curriculum_level = 0
        self.num_track_points = num_track_points
        self.track_generator = VectorizedTrackGenerator(self.device)
        self.track_settings = TrackSettings(num_points=num_track_points)

        self._wp_positions = torch.zeros(self.num_envs, self.track_settings.num_points, 3, device=self.device)
        self._wp_normals = torch.zeros(self.num_envs, self.track_settings.num_points, 3, device=self.device)
        self._target_wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # target orientation (handled internally, always facing towards next waypoint)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)

        # intital speed map by curriculum level
        self._speed_map = {
            "0": (0.0, 1.0), # level 0: stationary start / slightly forward
            "1": (0.0, 3.0), # level 1: random speed between 0 m/s and 3 m/s
            "2": (0.0, 3.0), # level 2: random speed between 0 m/s and 3 m/s
            "3": (2.0, 5.0), # level 3: random speed between 2 m/s and 5 m/s
            "4": (5.0, 10.0), # level 4: random speed between 5 m/s and 10 m/s
            "5": (5.0, 10.0), # level 5: random speed between 5 m/s and 10 m/s
        }

        # Define action and observation spaces
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        # Load quadcopter parameters
        config_file = Path(__file__).parent / config_path
        Log.info(f"Loading quadcopter params from {config_file}")
        with open(config_file) as f:
            params = json.load(f)

        # Quadcopter state
        self._position = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._velocity = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._quaternion = torch.zeros(self.num_envs, 4, device=self.device)  # (w, x, y, z)
        self._quaternion[:, 0] = 1.0  # identity quaternion
        self._angular_velocity = torch.zeros(self.num_envs, 3, device=self.device)  # body frame

        # Actions and forces
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_actions = torch.zeros(self.num_envs, 4, device=self.device)  # normalized [-1, 1]
        self._last_actions_rpm = torch.zeros(self.num_envs, 4, device=self.device)
        self._rotor_speeds = torch.zeros(self.num_envs, 4, device=self.device)
        self._total_thrust_body = torch.zeros(self.num_envs, 3, device=self.device)

        # Physics parameters
        self._mass = params['mass']
        self._inertia = torch.tensor(params['inertia_diag'], device=self.device)
        self._inertia_inv = 1.0 / self._inertia
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self._gravity_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        self._max_rpm = params['max_measured_rpm']

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["progress", "wp_passed", "action_smoothness", "action_magnitude", "ang_vel", "orientation", "alive"]
        }
        self._cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Completed episode statistics (stores most recent completed episode for each env)
        self._completed_episode_lengths = torch.full((self.num_envs,), float('nan'), dtype=torch.float, device=self.device)
        self._completed_episode_rewards = torch.full((self.num_envs,), float('nan'), dtype=torch.float, device=self.device)

        # Store nominal (original) dynamics parameters
        self._nominal_thrust_coefficients = torch.tensor(params['thrust_coefficients'], device=self.device, dtype=torch.float32)
        self._nominal_thrust_directions = torch.tensor(params['rotor_thrust_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_directions = torch.tensor(params['rotor_torque_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_constants = torch.tensor(params['rotor_torque_constants'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_positions = torch.tensor(params['rotor_positions'], dtype=torch.float32, device=self.device)
        self._nominal_rising_delay_constants = 1.0 / torch.tensor(params['delay_rising_constants'], dtype=torch.float32, device=self.device)
        self._nominal_falling_delay_constants = 1.0 / torch.tensor(params['delay_falling_constants'], dtype=torch.float32, device=self.device)

        # Create per-environment randomized dynamics parameters
        self._thrust_coefficients = self._nominal_thrust_coefficients.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._thrust_directions = self._nominal_thrust_directions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rotor_torque_directions = self._nominal_rotor_torque_directions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rotor_torque_constants = self._nominal_rotor_torque_constants.unsqueeze(0).repeat(self.num_envs, 1)
        self._rotor_positions = self._nominal_rotor_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rising_delay_constants = self._nominal_rising_delay_constants.unsqueeze(0).repeat(self.num_envs, 1)
        self._falling_delay_constants = self._nominal_falling_delay_constants.unsqueeze(0).repeat(self.num_envs, 1)
        self._decimation_steps = 2

        # torch.compile setup
        self.use_compile = use_compile
        if self.use_compile:
            # Use 'default' mode instead of 'reduce-overhead' to avoid CUDA graph issues
            # with tensor reuse across multiple step() calls
            effective_mode = compile_mode if compile_mode != "reduce-overhead" else "default"
            Log.info(f"Compiling physics step with torch.compile (mode={effective_mode})")
            self._compiled_physics_step = torch.compile(
                self._physics_step_impl,
                mode=effective_mode,
                fullgraph=False,
            )
        else:
            self._compiled_physics_step = self._physics_step_impl

        Log.info(f"QuadcopterEnv initialized: dt={dt}, max_ep_len={max_episode_length}, "
                 f"track_points={num_track_points}, dynamics_rand={dynamics_randomization_delta}")

    def step(self, actions: torch.Tensor) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Ensure actions are on correct device and shape
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Ensure actions have batch dimension
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # (4,) -> (1, 4)

        # Process actions and apply physics
        self._actions = actions.clone().clamp(-1.0, 1.0)
        actions_rpm = self._max_rpm * (self._actions + 1.0) / 2.0

        # Accumulate rewards and reward components across decimation sub-steps.
        # Only the final sub-step's observations, terminals, etc. are kept.
        total_rewards = torch.zeros(self.num_envs, device=self.device)
        total_unscaled_components = torch.zeros(self.num_envs, 7, device=self.device)
        total_scaled_components = torch.zeros(self.num_envs, 7, device=self.device)

        for _ in range(self._decimation_steps):
            self._physics_substep(actions_rpm)
            total_rewards += self.rewards
            total_unscaled_components += self._last_unscaled_reward_components
            total_scaled_components += self._last_scaled_reward_components

        self._last_actions = self._actions.clone()
        self._last_actions_rpm = actions_rpm.clone()

        self.rewards = total_rewards
        component_names = ["progress", "wp_passed", "action_smoothness", "action_magnitude", "ang_vel", "orientation", "alive"]
        # Unscaled rewards dict for episode tracking
        rewards_dict = {
            name: total_unscaled_components[:, i]
            for i, name in enumerate(component_names)
        }
        # Scaled rewards dict for tensorboard
        scaled_rewards_dict = {
            name: total_scaled_components[:, i]
            for i, name in enumerate(component_names)
        }
        for key, value in rewards_dict.items():
            self._episode_sums[key] += value
        self._cumulative_rewards += self.rewards
        self.truncations = self.episode_length_buf >= self.max_episode_length - 1
        self.episode_length_buf += 1
        reset_envs = torch.where(self.terminals | self.truncations)[0]
        if len(reset_envs) > 0:
            # Store completed episode stats before resetting
            self._completed_episode_lengths[reset_envs] = self.episode_length_buf[reset_envs].float()
            self._completed_episode_rewards[reset_envs] = self._cumulative_rewards[reset_envs]
            avg_len = self.episode_length_buf[reset_envs].float().mean().item()
            avg_rew = self._cumulative_rewards[reset_envs].mean().item()
            # Log.debug(f"Resetting {len(reset_envs)} envs | avg ep length: {avg_len:.0f} | avg ep reward: {avg_rew:.2f}")
            self._reset_idx(reset_envs)
            # Recompute observations for reset envs so returned obs reflects
            # the new episode start rather than the stale terminal state.
            fresh_obs = self._get_observations()
            self.observations[reset_envs] = fresh_obs[reset_envs]

        Log.render(
            position=self._position[0].detach().cpu().numpy(),
            quaternion_wxyz=self._quaternion[0].detach().cpu().numpy(),
            actions=self._actions[0].detach().cpu().numpy(),
            observations=self.observations[0].detach().cpu().numpy(),
            angular_velocity_rad=self._angular_velocity[0].detach().cpu().numpy(),
            rotor_speeds=self._rotor_speeds[0].detach().cpu().numpy(),
            total_thrust_body=self._total_thrust_body[0].detach().cpu().numpy(),
            velocity_world=self._velocity[0].detach().cpu().numpy(),
            goal_quaternion_wxyz=self._desired_quat_w[0].detach().cpu().numpy(),
            wp_positions=self._wp_positions[0].detach().cpu().numpy(),
            target_wp_idx=self._target_wp_idx[0].item(),
        )

        info = {
            "mean_reward": self.rewards.mean().item(),
        }
        for key, value in rewards_dict.items():
            info[f"mean_{key}"] = value.mean().item()
        for key, value in scaled_rewards_dict.items():
            info[f"mean_scaled_{key}"] = value.mean().item()
        valid_mask = ~torch.isnan(self._completed_episode_lengths)
        if valid_mask.any():
            valid_lengths = self._completed_episode_lengths[valid_mask]
            valid_rewards = self._completed_episode_rewards[valid_mask]
            info["episode_length_min"] = valid_lengths.min().item()
            info["episode_length_max"] = valid_lengths.max().item()
            info["episode_length_mean"] = valid_lengths.mean().item()
            info["episode_reward_min"] = valid_rewards.min().item()
            info["episode_reward_max"] = valid_rewards.max().item()
            info["episode_reward_mean"] = valid_rewards.mean().item()
        else:
            info["episode_length_min"] = 0.0
            info["episode_length_max"] = 0.0
            info["episode_length_mean"] = 0.0
            info["episode_reward_min"] = 0.0
            info["episode_reward_max"] = 0.0
            info["episode_reward_mean"] = 0.0
        self.infos = [info]
        return (self.observations, self.rewards, self.terminals,
            self.truncations, self.infos)
    
    def _physics_step_impl(
        self,
        actions_rpm: torch.Tensor,
        actions_normalized: torch.Tensor,
        last_actions_normalized: torch.Tensor,
        last_actions_rpm: torch.Tensor,
        rotor_speeds: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        wp_positions: torch.Tensor,
        wp_normals: torch.Tensor,
        target_wp_idx: torch.Tensor,
        rotor_positions: torch.Tensor,
        thrust_directions: torch.Tensor,
        thrust_coefficients: torch.Tensor,
        rotor_torque_constants: torch.Tensor,
        rotor_torque_directions: torch.Tensor,
        rising_delay_constants: torch.Tensor,
        falling_delay_constants: torch.Tensor,
        mass: float,
        inertia: torch.Tensor,
        inertia_inv: torch.Tensor,
        gravity: torch.Tensor,
        gravity_unit: torch.Tensor,
        dt: float,
        max_rpm: float,
        progress_reward_scale: float,
        wp_passed_reward_scale: float,
        action_smoothness_reward_scale: float,
        action_magnitude_reward_scale: float,
        ang_vel_reward_scale: float,
        orientation_reward_scale: float,
        alive_reward_scale: float,
        crash_penalty: float,
    ):
        """Pure computation kernel for physics step - compiled by torch.compile."""
        # Apply motor delay
        rising_mask = actions_rpm > rotor_speeds
        diffs = actions_rpm - rotor_speeds
        delay_constants = torch.where(rising_mask, rising_delay_constants, falling_delay_constants)
        new_rotor_speeds = rotor_speeds + diffs * delay_constants * dt

        # Compute thrust from rotor speeds (quadratic thrust curve)
        actions_polynomial = torch.stack([
            torch.ones_like(new_rotor_speeds),
            new_rotor_speeds,
            torch.square(new_rotor_speeds)
        ], dim=-1)  # N x 4 x 3
        thrust_magnitude = torch.einsum('ijk,ijk->ij', actions_polynomial, thrust_coefficients)  # N x 4
        rotor_thrust = thrust_magnitude[..., None] * thrust_directions

        # Compute torques
        # Yaw moment (torque in z axis)
        torque_body = torch.sum(
            thrust_magnitude[..., None] *
            rotor_torque_constants[..., None] *
            rotor_torque_directions,
            dim=1
        )
        # Roll and pitch moment (torque in x and y axis) - vectorized cross product
        cross_prod = torch.cross(rotor_positions, rotor_thrust, dim=-1).sum(dim=1)
        torque_body = torque_body + cross_prod

        # Total thrust in body frame
        total_thrust_body = rotor_thrust.sum(dim=1)

        # Integrate physics
        # Convert thrust from body to world frame
        thrust_world = rotate_vector_by_quaternion(total_thrust_body, quaternion)

        # Linear acceleration (F/m + g)
        linear_acc = thrust_world / mass + gravity

        # Update velocity and position
        new_velocity = velocity + linear_acc * dt
        new_position = position + new_velocity * dt

        # Angular acceleration (I^-1 * (tau - omega x (I * omega)))
        I_omega = inertia * angular_velocity
        gyroscopic = torch.cross(angular_velocity, I_omega, dim=-1)
        angular_acc = inertia_inv * (torque_body - gyroscopic)

        # Update angular velocity
        new_angular_velocity = torch.clamp(angular_velocity + angular_acc * dt, -100, 100)

        # Update quaternion
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = torch.cat([
            torch.zeros_like(new_angular_velocity[..., :1]),
            new_angular_velocity
        ], dim=-1)
        q_dot = 0.5 * quaternion_multiply(quaternion, omega_quat)
        new_quaternion = quaternion + q_dot * dt

        # Normalize quaternion
        new_quaternion = new_quaternion / torch.norm(new_quaternion, dim=-1, keepdim=True)

        # Determine target wp
        batch_idx = torch.arange(wp_positions.shape[0], device=wp_positions.device)
        current_wp = wp_positions[batch_idx, target_wp_idx] #(N, 3)
        current_normal = wp_normals[batch_idx, target_wp_idx] #(N, 3)
        vec_old = position - current_wp
        dist_plane_old = torch.sum(vec_old * current_normal, dim=1)
        vec_new = new_position - current_wp
        dist_plane_new = torch.sum(vec_new * current_normal, dim=1)
        # Perpendicular distance to gate center (in the plane of the gate)
        perp_offset = vec_new - dist_plane_new.unsqueeze(-1) * current_normal
        dist_to_center_perp = torch.linalg.norm(perp_offset, dim=1)
        # We crossed if we went from negative (behind) to positive (ahead)
        # AND we are within a reasonable perpendicular distance to count it.
        within_radius = dist_to_center_perp < 1.0
        crossed_plane = (dist_plane_old < 0) & (dist_plane_new >= 0)
        wp_passed = crossed_plane & within_radius
        false_pass = crossed_plane & ~within_radius # kill or penalize quad if it misses wp #test
        new_target_idx = target_wp_idx + wp_passed.long()
        new_target_idx = torch.clamp(new_target_idx, max=wp_positions.shape[1]-1)
        desired_pos_w = wp_positions[batch_idx, new_target_idx]

        # determine target orientation (always point towards next waypoint)
        target_vec = desired_pos_w - position
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        half_yaw = target_yaw * 0.5
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        desired_quat_w = torch.stack([cy, torch.zeros_like(cy), torch.zeros_like(cy), sy], dim=-1)

        # Compute observations
        velocity_body = rotate_vector_by_quaternion_conj(new_velocity, new_quaternion)
        gravity_body = rotate_vector_by_quaternion_conj(gravity_unit.unsqueeze(0).expand(new_position.shape[0], -1), new_quaternion)
        rpm_scaled = new_rotor_speeds / max_rpm
        orientation_error = quaternion_error_axis_angle(new_quaternion, desired_quat_w)

        future_rel_wp = []
        lookahead = 3
        batch_idx = torch.arange(wp_positions.shape[0], device=wp_positions.device)
        for i in range(lookahead):
            idx = torch.clamp(new_target_idx + i, max=wp_positions.shape[1]-1)
            wp_pos = wp_positions[batch_idx, idx]
            rel_pos_world = wp_pos - new_position
            rel_pos_body = rotate_vector_by_quaternion_conj(rel_pos_world, new_quaternion)
            rel_pos_body = torch.clamp(rel_pos_body * (0.3**(i+1)), -1.0, 1.0)
            future_rel_wp.append(rel_pos_body)

        # create copy of new angular velocity and orientation error to scale down
        o_new_angular_velocity = new_angular_velocity.clone()
        o_new_angular_velocity[:, :2] *= 0.025
        o_new_angular_velocity[:, 2] *= 0.3
        o_orientation_error = orientation_error.clone()
        o_orientation_error[:, :2] *= 0.3
        o_orientation_error[:, 2] *= 0.5

        # apply clamps
        o_velocity_body = torch.clamp(velocity_body * 0.07, -1.0, 1.0)
        o_gravity_body = torch.clamp(gravity_body, -1.0, 1.0)
        o_new_angular_velocity = torch.clamp(o_new_angular_velocity, -1.0, 1.0)
        o_orientation_error = torch.clamp(o_orientation_error, -1.0, 1.0)

        observations = torch.cat([
            o_velocity_body,           # 3
            o_new_angular_velocity,    # 3
            o_gravity_body,            # 3
            o_orientation_error,       # 3
            rpm_scaled,              # 4
            *future_rel_wp           # 3 * lookahead(3) = 9
        ], dim=-1)

        # rewards

        # Delta progress reward (encourage moving towards goal)
        # Use desired_pos_w (the target *after* any gate advance) so there is no
        # reward discontinuity when the target switches to the next waypoint.
        dist_old = torch.linalg.norm(position - desired_pos_w, dim=1)
        dist_new = torch.linalg.norm(new_position - desired_pos_w, dim=1)
        delta_progress = torch.clamp(dist_old - dist_new, min=-0.5, max=0.2) # cap reward 
        #scaled since it's a small number (meters per 0.01s)
        r_progress = delta_progress * progress_reward_scale
        #pulse reward
        r_wp = wp_passed.float() * wp_passed_reward_scale

        # penalties

        # action penalty (use normalized [-1,1] actions so penalty is RPM-independent)
        action_diff = torch.norm(actions_normalized - last_actions_normalized, dim=1)
        r_smooth = action_smoothness_reward_scale * action_diff * dt
        action_mag = torch.square(actions_normalized).mean(dim=1)
        r_magnitude = action_magnitude_reward_scale * action_mag * dt
        # velocity penalty (encourage smooth flight)
        wx = new_angular_velocity[:, 0] # Roll rate
        wy = new_angular_velocity[:, 1] # Pitch rate
        wz = new_angular_velocity[:, 2] # Yaw rate
        xy_spin_penalty = torch.clamp(torch.square(wx) + torch.square(wy), max=400.0)
        z_spin_penalty = torch.square(wz)
        ang_vel = xy_spin_penalty + z_spin_penalty
        r_ang_vel = (xy_spin_penalty * ang_vel_reward_scale * dt) + (z_spin_penalty * -0.05 * dt)
        # orientation penalty (encourage facing towards goal)
        orientation_error_magnitude = torch.linalg.norm(orientation_error, dim=1)
        orientation_error_mapped = torch.tanh(orientation_error_magnitude / 0.5)
        r_orient = orientation_error_mapped * orientation_reward_scale * dt
        r_alive = alive_reward_scale * dt # small penalty just for being alive each step

        # Check for termination / crash
        dist_to_target = torch.linalg.norm(desired_pos_w - new_position, dim=1)
        lost = dist_to_target > 8.0
        track_completed = (target_wp_idx + wp_passed.long()) >= (wp_positions.shape[1])
        spinning_out = torch.linalg.norm(new_angular_velocity, dim=1) > 50.0 # 50 rad/s is an extreme spin, likely unrecoverable
        died = (new_position[:, 2] < 0.1) | (new_position[:, 2] > 5.0) | lost | track_completed | spinning_out | false_pass
        crashed = (new_position[:, 2] < 0.1) | (new_position[:, 2] > 5.0) | lost | false_pass | spinning_out
        r_crashed = crashed.float() * crash_penalty

        rewards = (
            r_progress +
            r_wp +
            r_smooth +
            r_magnitude +
            r_ang_vel +
            r_orient +
            r_alive +
            r_crashed
        )

        # Reward components for logging
        # Unscaled (raw values before reward scaling)
        unscaled_reward_components = torch.stack([
            delta_progress,
            wp_passed.float(),
            action_diff * dt,
            action_mag * dt,
            ang_vel * dt,
            orientation_error_mapped * dt,
            torch.ones_like(delta_progress) * dt, # represents alive reward, dp is just for shape reference
        ], dim=-1)
        # Scaled (actual reward contributions)
        scaled_reward_components = torch.stack([
            r_progress,
            r_wp,
            r_smooth,
            r_magnitude,
            r_ang_vel,
            r_orient,
            torch.ones_like(r_progress) * r_alive,
        ], dim=-1)

        return (
            new_rotor_speeds,
            new_position,
            new_velocity,
            new_quaternion,
            new_angular_velocity,
            total_thrust_body,
            observations,
            rewards,
            unscaled_reward_components,
            scaled_reward_components,
            died,
            new_target_idx,
            desired_quat_w
        )

    def _physics_substep(self, actions_rpm: torch.Tensor):
        """Run a single physics sub-step. Updates state in-place.
        Stores observations, rewards, terminals, and reward components
        but does NOT do episode bookkeeping or resets."""
        # Call compiled physics kernel
        (
            self._rotor_speeds,
            self._position,
            self._velocity,
            self._quaternion,
            self._angular_velocity,
            self._total_thrust_body,
            self.observations,
            self.rewards,
            self._last_unscaled_reward_components,
            self._last_scaled_reward_components,
            self.terminals,
            self._target_wp_idx,
            self._desired_quat_w
        ) = self._compiled_physics_step( # this is _physics_step_impl wrapped by torch.compile
            actions_rpm,
            self._actions,
            self._last_actions,
            self._last_actions_rpm,
            self._rotor_speeds,
            self._position,
            self._velocity,
            self._quaternion,
            self._angular_velocity,
            self._wp_positions,
            self._wp_normals,
            self._target_wp_idx,
            self._rotor_positions,
            self._thrust_directions,
            self._thrust_coefficients,
            self._rotor_torque_constants,
            self._rotor_torque_directions,
            self._rising_delay_constants,
            self._falling_delay_constants,
            self._mass,
            self._inertia,
            self._inertia_inv,
            self._gravity,
            self._gravity_unit,
            self.dt,
            self._max_rpm,
            self.progress_reward_scale,
            self.wp_passed_reward_scale,
            self.action_smoothness_reward_scale,
            self.action_magnitude_reward_scale,
            self.ang_vel_reward_scale,
            self.orientation_reward_scale,
            self.alive_reward_scale,
            self.crash_penalty,
        )

    def _get_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        # velocity in body frame
        velocity_body = rotate_vector_by_quaternion_conj(self._velocity, self._quaternion)
        # gravity in body frame
        gravity_body = rotate_vector_by_quaternion_conj(self._gravity_unit.unsqueeze(0).expand(self.num_envs, -1), self._quaternion)
        # scaled rpm
        rpm_scaled = self._rotor_speeds / self._max_rpm
        # orientation error as axis-angle (in body frame)
        orientation_error = quaternion_error_axis_angle(self._quaternion, self._desired_quat_w)

        future_rel_wp = []
        lookahead = 3
        batch_idx = torch.arange(self.num_envs, device=self.device) 
        for i in range(lookahead):
            idx = torch.clamp(self._target_wp_idx + i, max=self._wp_positions.shape[1]-1)
            wp_pos = self._wp_positions[batch_idx, idx]
            rel_pos_world = wp_pos - self._position
            rel_pos_body = rotate_vector_by_quaternion_conj(rel_pos_world, self._quaternion)
            # clamp and scale down future waypoints to keep observations stable
            rel_pos_body = torch.clamp(rel_pos_body * (0.3**(i+1)), -1.0, 1.0)
            future_rel_wp.append(rel_pos_body)

        # create copy of new angular velocity and orientation error to scale down
        o_new_angular_velocity = self._angular_velocity.clone()
        o_new_angular_velocity[:, :2] *= 0.025
        o_new_angular_velocity[:, 2] *= 0.3
        o_orientation_error = orientation_error.clone()
        o_orientation_error[:, :2] *= 0.3
        o_orientation_error[:, 2] *= 0.5

        # apply clamps
        o_velocity_body = torch.clamp(velocity_body * 0.07, -1.0, 1.0)
        o_gravity_body = torch.clamp(gravity_body, -1.0, 1.0)
        o_new_angular_velocity = torch.clamp(o_new_angular_velocity, -1.0, 1.0)
        o_orientation_error = torch.clamp(o_orientation_error, -1.0, 1.0)
        
        obs = torch.cat([
            o_velocity_body,           # 3
            o_new_angular_velocity,  # 3
            o_gravity_body,            # 3
            o_orientation_error,       # 3
            rpm_scaled,               # 4
            *future_rel_wp             # 3 * lookahead
        ], dim=-1)

        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments."""
        Log.info(f"Full reset called on all {self.num_envs} envs (seed={seed})")
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Reset all environments
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)

        obs = self._get_observations()  # Already flattened
        return obs, [dict()]

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        if len(env_ids) == 0:
            return

        # Log.debug(f"_reset_idx: resetting {len(env_ids)} envs")

        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0
        self._rotor_speeds[env_ids] = 0.0
        self._cumulative_rewards[env_ids] = 0.0

        # Reset episode sums
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        # Randomize dynamics parameters for reset environments
        delta = self.dynamics_randomization_delta
        num_reset = len(env_ids)

        if delta > 0:
            # Log.debug(f"Applying dynamics randomization (delta={delta:.3f}) to {num_reset} envs")
            # Generate random multipliers: (1 +- delta)
            self._thrust_coefficients[env_ids] = self._nominal_thrust_coefficients * (
                1.0 + torch.zeros((num_reset, *self._nominal_thrust_coefficients.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rotor_torque_constants[env_ids] = self._nominal_rotor_torque_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_rotor_torque_constants.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rotor_positions[env_ids] = self._nominal_rotor_positions * (
                1.0 + torch.zeros((num_reset, *self._nominal_rotor_positions.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rising_delay_constants[env_ids] = self._nominal_rising_delay_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_rising_delay_constants.shape), device=self.device).uniform_(-delta, delta)
            )
            self._falling_delay_constants[env_ids] = self._nominal_falling_delay_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_falling_delay_constants.shape), device=self.device).uniform_(-delta, delta)
            )

        waypoints, normals = self.track_generator.generate_track(
        level=self.curriculum_level,
        settings=self.track_settings,
        env_ids=env_ids
            )
        self._wp_positions[env_ids] = waypoints
        self._wp_normals[env_ids] = normals

        # Randomized wp start and velocity (always points towards target)
        start_wp_idx = torch.randint(0, max(1, self.track_settings.num_points - 1), (len(env_ids),), device=self.device)
        self._target_wp_idx[env_ids] = start_wp_idx
        target_pos = waypoints[torch.arange(len(env_ids)), start_wp_idx]
        origin_pos = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(len(env_ids), -1)        
        prev_idx = torch.clamp(start_wp_idx - 1, min=0)
        prev_wp_pos = waypoints[torch.arange(len(env_ids)), prev_idx]
        #init spawn
        is_start = (start_wp_idx == 0)
        spawn_pos = torch.where(is_start.unsqueeze(-1), origin_pos, prev_wp_pos)
        spawn_noise = torch.randn_like(spawn_pos) * 0.3 #TODO: adjust with curriculum?
        spawn_pos_noisy = spawn_pos + spawn_noise
        # Clamp z so the drone doesn't spawn underground or too high
        spawn_pos_noisy[:, 2] = torch.clamp(spawn_pos_noisy[:, 2], min=0.5, max=4.5)
        self._position[env_ids] = spawn_pos_noisy
        # point yaw
        aim_vec = target_pos - self._position[env_ids]
        aim_yaw = torch.atan2(aim_vec[:, 1], aim_vec[:, 0])
        half_yaw = aim_yaw * 0.5
        cy = torch.cos(half_yaw)
        sy = torch.sin(half_yaw)
        self._quaternion[env_ids, 0] = cy
        self._quaternion[env_ids, 1] = 0.0
        self._quaternion[env_ids, 2] = 0.0
        self._quaternion[env_ids, 3] = sy
        # Update desired quaternion so observations are correct after reset
        self._desired_quat_w[env_ids, 0] = cy
        self._desired_quat_w[env_ids, 1] = 0.0
        self._desired_quat_w[env_ids, 2] = 0.0
        self._desired_quat_w[env_ids, 3] = sy
        # random speed 
        initial_speed = torch.empty(len(env_ids), 1, device=self.device).uniform_(*self._speed_map[str(self.curriculum_level)])
        # masking speed: If wp 0, speed = 0. Else, speed = random.
        initial_speed = torch.where(is_start.unsqueeze(-1), torch.zeros_like(initial_speed), initial_speed)
        aim_dir = torch.nn.functional.normalize(aim_vec, dim=1)
        self._velocity[env_ids] = aim_dir * initial_speed
        self._angular_velocity[env_ids] = 0.0
        self._last_actions[env_ids] = 0.0
        self._last_actions_rpm[env_ids] = 0.0

    def close(self):
        Log.info("QuadcopterEnv closed.")