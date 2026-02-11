from __future__ import annotations

import json
import torch
import numpy as np
import pufferlib
import gymnasium as gym

from typing import Optional, Tuple, Dict, Any

from math_utils import *
from .track_gen import VectorizedTrackGenerator, TrackSettings

#TODO:
'''
Done:
Update goal logic for passing through waypoint (plane passed normal)
Add logic for sequenced waypoints
Action smoothing (low-pass filter) 
Delta progress reward

Not Done:
Expand observation space to include next N waypoints
Randomized start positions infront of waypoints.
'''

class QuadcopterEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        config_path: str = "quad_params.json",
        max_episode_length: int = 1000,
        dt: float = 0.01,
        progress_reward_scale: float = 100.0,
        wp_passed_reward_scale: float = 5.0,
        action_smoothness_reward_scale: float = -0.7,
        ang_vel_reward_scale: float = -0.01,
        orientation_reward_scale: float = 10.0,
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
        super().__init__()

        self.device = torch.device(device)
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.max_episode_length_s = max_episode_length * dt
        
        # reward / penalty scales
        self.progress_reward_scale = progress_reward_scale
        self.wp_passed_reward_scale = wp_passed_reward_scale
        self.action_smoothness_reward_scale = action_smoothness_reward_scale
        self.ang_vel_reward_scale = ang_vel_reward_scale
        self.orientation_reward_scale = orientation_reward_scale
        
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

        #TODO: init rerun

        # Define action and observation spaces
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        # Load quadcopter parameters
        params = json.load(open(config_path))

        # Quadcopter state
        self._position = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._velocity = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._quaternion = torch.zeros(self.num_envs, 4, device=self.device)  # (w, x, y, z)
        self._quaternion[:, 0] = 1.0  # identity quaternion
        self._angular_velocity = torch.zeros(self.num_envs, 3, device=self.device)  # body frame

        # Actions and forces
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._last_actions_0_1 = torch.zeros(self.num_envs, 4, device=self.device)
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
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "orientation"]
        }
        self._cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Completed episode statistics (stores most recent completed episode for each env)
        self._completed_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._completed_episode_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

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
            self._compiled_physics_step = torch.compile(
                self._physics_step_impl,
                mode=effective_mode,
                fullgraph=False,
            )
        else:
            self._compiled_physics_step = self._physics_step_impl

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
        actions_0_1 = self._max_rpm * (self._actions + 1.0) / 2.0
        for _ in range(self._decimation_steps-1):
            self._step_once(actions_0_1)
        results = self._step_once(actions_0_1)
        self._last_actions_0_1 = actions_0_1.clone()
        return results
    
    def _physics_step_impl(
        self,
        actions_0_1: torch.Tensor,
        last_actions_0_1: torch.Tensor,
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
        ang_vel_reward_scale: float,
        orientation_reward_scale: float,
    ):
        """Pure computation kernel for physics step - compiled by torch.compile."""
        # Apply motor delay
        rising_mask = actions_0_1 > rotor_speeds
        diffs = actions_0_1 - rotor_speeds
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
        new_angular_velocity = torch.clamp(angular_velocity + angular_acc * dt, -1e12, 1e12)

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
        dist_to_center = torch.linalg.norm(vec_new, dim=1)
        # We crossed if we went from negative (behind) to positive (ahead)
        # AND we are within a reasonable distance (e.g. 1m radius) to count it.
        within_radius = dist_to_center < 1
        crossed_plane = (dist_plane_old < 0) & (dist_plane_new >= 0)
        wp_passed = crossed_plane & within_radius
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
            idx = torch.clamp(target_wp_idx + i, max=wp_positions.shape[1]-1)
            wp_pos = wp_positions[batch_idx, idx]
            rel_pos_world = wp_pos - position
            rel_pos_body = rotate_vector_by_quaternion_conj(rel_pos_world, quaternion)
            future_rel_wp.append(rel_pos_body)

        observations = torch.cat([
            velocity_body,           # 3
            new_angular_velocity,    # 3
            gravity_body,            # 3
            orientation_error,       # 3
            rpm_scaled,              # 4
            *future_rel_wp           # 3 * lookahead(3) = 9
        ], dim=-1)

        # rewards

        # Delta progress reward (encourage moving towards goal)
        dist_old = torch.linalg.norm(position - current_wp, dim=1)
        dist_new = torch.linalg.norm(new_position - current_wp, dim=1)
        delta_progress = dist_old - dist_new
        #scaled since it's a small number (meters per 0.01s)
        r_progress = delta_progress * progress_reward_scale
        #pulse reward
        r_wp = wp_passed.float() * wp_passed_reward_scale

        # penalties

        # action penalty
        action_diff = torch.norm(actions_0_1 - last_actions_0_1, dim=1)
        r_smooth = action_smoothness_reward_scale * action_diff * dt
        # velocity penalty (encourage smooth flight)
        ang_vel = torch.sum(torch.square(new_angular_velocity), dim=1)
        r_ang_vel = ang_vel * ang_vel_reward_scale * dt
        # orientation penalty (encourage facing towards goal)
        orientation_error_magnitude = torch.linalg.norm(orientation_error, dim=1)
        orientation_reward_mapped = 1 - torch.tanh(orientation_error_magnitude / 0.5)
        r_orient = orientation_reward_mapped * orientation_reward_scale * dt

        rewards = (
            r_progress +
            r_wp +
            r_smooth +
            r_ang_vel +
            r_orient
        )

        # Reward components for logging (unscaled)
        reward_components = torch.stack([
            delta_progress,
            wp_passed.float(),
            action_diff * dt,
            ang_vel * dt,
            orientation_reward_mapped * dt,
        ], dim=-1)

        # Check for termination
        dist_to_target = torch.linalg.norm(desired_pos_w - new_position, dim=1)
        lost = dist_to_target > 8.0
        died = (new_position[:, 2] < 0.1) | (new_position[:, 2] > 5.0) | lost

        return (
            new_rotor_speeds,
            new_position,
            new_velocity,
            new_quaternion,
            new_angular_velocity,
            total_thrust_body,
            observations,
            rewards,
            reward_components,
            died,
            new_target_idx,
            desired_quat_w
        )

    def _step_once(self, actions_0_1: torch.Tensor):
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
            reward_components,
            self.terminals,
            self._target_wp_idx,
            self._desired_quat_w
        ) = self._compiled_physics_step( # this is _physics_step_impl wrapped by torch.compile
            actions_0_1,
            self._last_actions_0_1,
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
            self.ang_vel_reward_scale,
            self.orientation_reward_scale,
        )

        # Build rewards dict for logging (outside compiled region)
        rewards_dict = {
            "progress": reward_components[:, 0],
            "wp_passed": reward_components[:, 1],
            "action_smoothness": reward_components[:, 2],
            "ang_vel": reward_components[:, 3],
            "orientation": reward_components[:, 4],
        }

        # Update episode sums for logging
        for key, value in rewards_dict.items():
            self._episode_sums[key] += value

        # Accumulate rewards for episode tracking
        self._cumulative_rewards += self.rewards

        # Check for truncation (timeout) - terminals already computed in kernel
        self.truncations = self.episode_length_buf >= self.max_episode_length - 1

        # Update episode length
        self.episode_length_buf += 1

        # Handle resets
        reset_envs = torch.where(self.terminals | self.truncations)[0]
        if len(reset_envs) > 0:
            # Store completed episode stats before resetting
            self._completed_episode_lengths[reset_envs] = self.episode_length_buf[reset_envs].float()
            self._completed_episode_rewards[reset_envs] = self._cumulative_rewards[reset_envs]
            self._reset_idx(reset_envs)

        # log data
        # self._render()

        # Compute reward statistics across all environments
        info = {
            "mean_reward": self.rewards.mean().item(),
        }

        # Add mean for each reward component
        for key, value in rewards_dict.items():
            info[f"mean_{key}"] = value.mean().item()

        # Add episode statistics (min/max/mean across most recent completed episode per env)
        info["episode_length_min"] = self._completed_episode_lengths.min().item()
        info["episode_length_max"] = self._completed_episode_lengths.max().item()
        info["episode_length_mean"] = self._completed_episode_lengths.mean().item()
        info["episode_reward_min"] = self._completed_episode_rewards.min().item()
        info["episode_reward_max"] = self._completed_episode_rewards.max().item()
        info["episode_reward_mean"] = self._completed_episode_rewards.mean().item()

        self.infos = [info]
        return (self.observations, self.rewards, self.terminals,
            self.truncations, self.infos)

    def _get_observations(self) -> torch.Tensor:
        """Compute observations for all environments."""
        # velocity in body frame
        velocity_body = rotate_vector_by_quaternion_conj(self._velocity, self._quaternion)
        # gravity in body frame
        gravity_body = rotate_vector_by_quaternion_conj(self._gravity_unit.unsqueeze(0).expand(self.num_envs, -1), self._quaternion)
        # scaled rpm
        rpm_scaled = self._rotor_speeds / self._max_rpm
        # orientation error as axis-angle (in body frame) #TODO bring out desired quat somehow
        orientation_error = quaternion_error_axis_angle(self._quaternion, self._desired_quat_w)

        future_rel_wp = []
        lookahead = 3
        batch_idx = torch.arange(self.num_envs, device=self.device) 
        for i in range(lookahead):
            idx = torch.clamp(self._target_wp_idx + i, max=self._wp_positions.shape[1]-1)
            wp_pos = self._wp_positions[batch_idx, idx]
            rel_pos_world = wp_pos - self._position
            rel_pos_body = rotate_vector_by_quaternion_conj(rel_pos_world, self._quaternion)
            future_rel_wp.append(rel_pos_body)
        
        obs = torch.cat([
            velocity_body,           # 3
            self._angular_velocity,  # 3
            gravity_body,            # 3
            orientation_error,       # 3
            rpm_scaled,               # 4
            *future_rel_wp             # 3 * lookahead
        ], dim=-1)

        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()

        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[torch.Tensor, Dict]:
        """Reset all environments."""
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

        #TODO: Start in front of random wp, and point towards it
        self._target_wp_idx[env_ids] = 0

        # Reset quadcopter state to origin with identity orientation
        self._position[env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._velocity[env_ids] = 0.0
        self._quaternion[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._angular_velocity[env_ids] = 0.0

    def close(self):
        pass

    # def _render(self):
    #     """Render the environment using rerun logging."""
        # Log the first environment's state (index 0)
        # pass
        # position = self._position[0].detach().cpu().numpy()
        # quaternion = self._quaternion[0].detach().cpu().numpy()
        # quat_xyzw = np.array([quaternion[1], quaternion[2],
        #                       quaternion[3], quaternion[0]])

        # for i, action_val in enumerate(self._actions[0].cpu().numpy()):
        #     rr.log(f"actions/motor_{i}", rr.Scalars(float(action_val)))

        # for i, observation_val in enumerate(self.observations[0].cpu().numpy()):
        #     rr.log(f"observations/{i}", rr.Scalars(float(observation_val)))

        # log_drone_pose(position, quat_xyzw)

        # # Log angular velocity in degrees/s as time series
        # angular_vel_rad = self._angular_velocity[0].detach().cpu().numpy()
        # angular_vel_deg = np.degrees(angular_vel_rad)
        # rr.log("angular_velocity_deg_s/roll", rr.Scalars(float(angular_vel_deg[0])))
        # rr.log("angular_velocity_deg_s/pitch", rr.Scalars(float(angular_vel_deg[1])))
        # rr.log("angular_velocity_deg_s/yaw", rr.Scalars(float(angular_vel_deg[2])))

        # # Log RPMs as time series
        # rpms = self._rotor_speeds[0].detach().cpu().numpy()
        # rr.log("rotor_speeds_rpm/motor_0", rr.Scalars(float(rpms[0])))
        # rr.log("rotor_speeds_rpm/motor_1", rr.Scalars(float(rpms[1])))
        # rr.log("rotor_speeds_rpm/motor_2", rr.Scalars(float(rpms[2])))
        # rr.log("rotor_speeds_rpm/motor_3", rr.Scalars(float(rpms[3])))

        # # Log total thrust in body frame as time series
        # total_thrust = self._total_thrust_body[0].detach().cpu().numpy()
        # rr.log("total_thrust_body_N/x", rr.Scalars(float(total_thrust[0])))
        # rr.log("total_thrust_body_N/y", rr.Scalars(float(total_thrust[1])))
        # rr.log("total_thrust_body_N/z", rr.Scalars(float(total_thrust[2])))

        # # Log velocity in world frame as time series
        # velocity_world = self._velocity[0].detach().cpu().numpy()
        # rr.log("velocity_world_m_s/x", rr.Scalars(float(velocity_world[0])))
        # rr.log("velocity_world_m_s/y", rr.Scalars(float(velocity_world[1])))
        # rr.log("velocity_world_m_s/z", rr.Scalars(float(velocity_world[2])))

        # # Log goal position with goal orientation
        # goal_position = self._desired_pos_w[0].detach().cpu().numpy()
        # goal_quaternion = self._desired_quat_w[0].detach().cpu().numpy()
        # goal_quat_xyzw = np.array([goal_quaternion[1], goal_quaternion[2],
        #                            goal_quaternion[3], goal_quaternion[0]])
        # rr.log(
        #     "goal",
        #     rr.Transform3D(
        #         translation=goal_position,
        #         quaternion=goal_quat_xyzw,
        #     ),
        #     rr.TransformAxes3D(0.5),
        #     static=False,
        # )