import numpy as np
from dataclasses import dataclass

'''
Generate tracks (a set of waypoints in meters) with setable levels of difficulty.

Level 0: Straight line track (adjustable length)
Level 1: Straight track with variable height (adjustable length and height range)
Level 2: circular track (adjustable radius, direction)
Level 3: circular track with variable height (adjustable radius, height, direction)
Level 4: random-walk track (adjustable distance between points, number of points)
Level 5: random-walk track with variable height (adjustable distance between points, number of points, height range)
'''

@dataclass
class TrackSettings:
    num_points: int = 10 # number of waypoints
    length: float = 10.0  # for straight tracks
    height_range: tuple = (1, 5)  # for variable height tracks
    radius: float = 5.0  # for circular tracks
    direction: str = 'clockwise'  # for circular tracks
    step_size: float = 3.0  # for random walk tracks

@dataclass
class Track:
    level: int
    settings: TrackSettings  # settings used to generate the track
    waypoints: np.ndarray  # shape (N, 3) for N waypoints in 3D space

class TrackGenerator:
    def __init__(self):
        pass

    def generate_track(self, level: int, settings: TrackSettings) -> Track:
        if level == 0:
            return self._generate_straight_line_track(settings)
        elif level == 1:
            return self._generate_variable_height_straight_track(settings)
        elif level == 2:
            return self._generate_circular_track(settings)
        elif level == 3:
            return self._generate_variable_height_circular_track(settings)
        elif level == 4:
            return self._generate_random_walk_track(settings)
        elif level == 5:
            return self._generate_variable_height_random_walk_track(settings)
        else:
            raise ValueError(f"Invalid track level: {level}")

    def _generate_straight_line_track(self, settings: TrackSettings) -> Track:
        waypoints = np.array([[i, 0, 2] for i in np.linspace(0, settings.length, num=settings.num_points)])
        return Track(level=0, settings=settings, waypoints=waypoints)

    def _generate_variable_height_straight_track(self, settings: TrackSettings) -> Track:
        waypoints = np.array([[i, 0, np.random.uniform(*settings.height_range)] for i in np.linspace(0, settings.length, num=settings.num_points)])
        return Track(level=1, settings=settings, waypoints=waypoints)

    def _generate_circular_track(self, settings: TrackSettings) -> Track:
        angle_step = np.pi / 50 if settings.direction == 'clockwise' else -np.pi / 50
        waypoints = np.array([[settings.radius * np.cos(i * angle_step), settings.radius * np.sin(i * angle_step), 0] for i in range(settings.num_points)])
        return Track(level=2, settings=settings, waypoints=waypoints)

    def _generate_variable_height_circular_track(self, settings: TrackSettings) -> Track:
        angle_step = np.pi / 50 if settings.direction == 'clockwise' else -np.pi / 50
        waypoints = np.array([[settings.radius * np.cos(i * angle_step), settings.radius * np.sin(i * angle_step), np.random.uniform(*settings.height_range)] for i in range(settings.num_points)])
        return Track(level=3, settings=settings, waypoints=waypoints)
    
    def _generate_random_walk_track(self, settings: TrackSettings) -> Track:
        waypoints = np.zeros((settings.num_points, 3))
        step_size = settings.step_size * np.random.uniform(0.5, 1.5)  # add some variability to step size
        for i in range(1, settings.num_points):
            settings.direction = np.random.uniform(0, 2 * np.pi)
            waypoints[i] = waypoints[i-1] + step_size * np.array([np.cos(settings.direction), np.sin(settings.direction), 0])
        return Track(level=4, settings=settings, waypoints=waypoints)
    
    def _generate_variable_height_random_walk_track(self, settings: TrackSettings) -> Track:
        waypoints = np.zeros((settings.num_points, 3))
        step_size = settings.step_size * np.random.uniform(0.5, 1.5)  # add some variability to step size
        for i in range(1, settings.num_points):
            settings.direction = np.random.uniform(0, 2 * np.pi)
            waypoints[i] = waypoints[i-1] + step_size * np.array([np.cos(settings.direction), np.sin(settings.direction), np.random.uniform(*settings.height_range)])
        return Track(level=5, settings=settings, waypoints=waypoints)