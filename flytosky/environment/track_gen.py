import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple

'''
Generate tracks (a set of waypoints in meters) with setable levels of difficulty.

Level 0: Straight line track (adjustable length)
Level 1: Straight track with variable height (adjustable length and height range)
Level 2: circular track (adjustable radius, direction)
Level 3: circular track with variable height (adjustable radius, height, direction)

Are these actually better?
Level 4: random-walk track (adjustable distance between points, number of points)
Level 5: random-walk track with variable height (adjustable distance between points, number of points, height range)
'''

#TODO:
'''
Rather than only generate new difficult tracks, do a distribution
So Level 0: 100% straight
Level 1: 50% straight, 50% straight with variable height
Level 2: 50% circular, 25% circular with variable height, 25% straight with variable height etc
'''

@dataclass
class TrackSettings:
    """
    Settings for track generation. 
    """
    num_points: int = 10        # Number of waypoints
    length: float = 20.0        # Total length for straight tracks
    height_range: tuple = (1.0, 2.5) # Min/Max height
    radius: float = 5.0         # Radius for circular tracks
    step_size: float = 4.0      # Distance between gates (Random Walk)

class VectorizedTrackGenerator:
    def __init__(self, device: torch.device):
        self.device = device

    def generate_track(self, 
                       level: int, 
                       settings: TrackSettings, 
                       env_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates tracks for a batch of environments.
        
        Args:
            level: Curriculum level (0-5)
            settings: TrackSettings object
            env_ids: Tensor of environment indices that need reset
            
        Returns:
            waypoints: Tensor [Num_Reset, Num_Points, 3]
            normals: Tensor [Num_Reset, Num_Points, 3] (Direction to next gate)
        """
        if level == 0:
            return self._generate_straight(settings, env_ids, variable_height=False)
        elif level == 1:
            return self._generate_straight(settings, env_ids, variable_height=True)
        elif level == 2:
            return self._generate_circle(settings, env_ids, variable_height=False)
        elif level == 3:
            return self._generate_circle(settings, env_ids, variable_height=True)
        elif level == 4:
            return self._generate_random_walk(settings, env_ids, variable_height=False)
        elif level == 5:
            return self._generate_random_walk(settings, env_ids, variable_height=True)
        else:
            raise ValueError(f"Invalid track level: {level}")

    def _compute_normals(self, waypoints: torch.Tensor) -> torch.Tensor:
        """Calculates direction vector (normal) for each gate plane."""
        #vector from Gate i -> Gate i+1
        diffs = waypoints[:, 1:] - waypoints[:, :-1]
        normals = torch.nn.functional.normalize(diffs, dim=-1)
        #pad the last gate with the previous normal (since there is no i+1)
        last_normal = normals[:, -1:].clone()
        return torch.cat([normals, last_normal], dim=1)

    def _generate_straight(self, settings: TrackSettings, env_ids: torch.Tensor, variable_height: bool):
        num_reset = len(env_ids)
        waypoints = torch.zeros(num_reset, settings.num_gates, 3, device=self.device)    
        x = torch.linspace(0, settings.length, settings.num_points, device=self.device)
        waypoints[:, :, 0] = x.unsqueeze(0).expand(num_reset, -1)
        waypoints[:, :, 1] = 0.0
        if variable_height:
            waypoints[:, :, 2] = torch.empty(num_reset, settings.num_points, device=self.device).uniform_(*settings.height_range)
        else:
            waypoints[:, :, 2] = 1.5 # Fixed height
        return waypoints, self._compute_normals(waypoints)

    def _generate_circle(self, settings: TrackSettings, env_ids: torch.Tensor, variable_height: bool):
        num_reset = len(env_ids)
        waypoints = torch.zeros(num_reset, settings.num_points, 3, device=self.device)
        angles = torch.linspace(0, 2 * np.pi, settings.num_points, device=self.device)
        angles = angles.unsqueeze(0).expand(num_reset, -1)
        direction = torch.randint(0, 2, (num_reset, 1), device=self.device).float() * 2 - 1
        angles = angles * direction
        waypoints[:, :, 0] = settings.radius * torch.cos(angles)
        waypoints[:, :, 1] = settings.radius * torch.sin(angles)
        if variable_height:
            waypoints[:, :, 2] = torch.empty(num_reset, settings.num_points, device=self.device).uniform_(*settings.height_range)
        else:
            waypoints[:, :, 2] = 1.5
        return waypoints, self._compute_normals(waypoints)

    def _generate_random_walk(self, settings: TrackSettings, env_ids: torch.Tensor, variable_height: bool):
        num_reset = len(env_ids)
        waypoints = torch.zeros(num_reset, settings.num_points, 3, device=self.device)
        start_z = 1.5
        current_pos = torch.zeros(num_reset, 3, device=self.device)
        current_pos[:, 2] = start_z
        waypoints[:, 0] = current_pos
        for i in range(1, settings.num_points):
            if variable_height:
                #3D Noise
                offset = torch.randn(num_reset, 3, device=self.device)
                offset = torch.nn.functional.normalize(offset, dim=1) * settings.step_size
                new_z = current_pos[:, 2] + offset[:, 2]
                new_z = torch.clamp(new_z, settings.height_range[0], settings.height_range[1])
                offset[:, 2] = new_z - current_pos[:, 2] # Re-adjust offset to match clamped Z
            else:
                # 2D Noise (XY only)
                angle = torch.rand(num_reset, device=self.device) * 2 * np.pi
                dx = torch.cos(angle) * settings.step_size
                dy = torch.sin(angle) * settings.step_size
                offset = torch.stack([dx, dy, torch.zeros(num_reset, device=self.device)], dim=1)
            # Here we just add the offset
            current_pos = current_pos + offset
            waypoints[:, i] = current_pos

        return waypoints, self._compute_normals(waypoints)