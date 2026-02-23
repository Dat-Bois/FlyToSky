import time
import inspect
import logging
import numpy as np
from typing import Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation

import rerun as rr
import rerun.blueprint as rrb

file_path = Path(__file__).parent

class ColoredFormatter(logging.Formatter):
    def __init__(self, template: str) -> None:
        super().__init__(template)
        # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        grey = "\x1b[38;20m"
        cyan = "\x1b[36;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        self.FORMATS = {
            logging.DEBUG: grey + template + reset,
            logging.INFO: cyan + template + reset,
            logging.WARNING: yellow + template + reset,
            logging.ERROR: red + template + reset,
            logging.CRITICAL: bold_red + template + reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Log:
    """A high-level logger that supports multiple named loggers with
    independent log files while sharing a single Rerun session."""

    _rerun_initialized: bool = False
    _output_path: Path | None = None
    _loggers: dict[str, logging.Logger] = {}  # filename_stem -> Logger

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def init(rrd_filename: str = "") -> None:
        """Set up the shared output directory and Rerun session.

        Safe to call multiple times — subsequent calls are no-ops.
        """
        if Log._output_path is not None:
            return

        output_path = file_path / "Logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
        output_path.mkdir(parents=True, exist_ok=True)
        Log._output_path = output_path

        Log._init_rerun(output_path, (rrd_filename or "flytosky") + ".rrd")

    # ------------------------------------------------------------------
    # Caller-aware logger resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _get_caller_logger() -> logging.Logger:
        """Return the ``logging.Logger`` for the file that called into
        ``Log``.  Creates one (with its own log file) on first access.

        Stack: [0] _get_caller_logger → [1] Log.info/debug/… → [2] caller
        """
        frame = inspect.stack()[2]
        name = Path(frame.filename).stem  # e.g. "train", "quadcopter_env"
        if name not in Log._loggers:
            Log._loggers[name] = Log._create_console_logger(name, Log._output_path)
        return Log._loggers[name]

    # ------------------------------------------------------------------
    # Rerun (shared)
    # ------------------------------------------------------------------

    @staticmethod
    def _send_layout():
        """Constructs and sends the blueprint with panels for all logged data."""
        drone_3d = rrb.Spatial3DView(name="Drone View", origin="local")

        actions_plot = rrb.TimeSeriesView(name="Actions", origin="actions")
        observations_plot = rrb.TimeSeriesView(name="Observations", origin="observations")
        angular_vel_plot = rrb.TimeSeriesView(name="Angular Velocity (deg/s)", origin="angular_velocity_deg_s")
        rotor_speeds_plot = rrb.TimeSeriesView(name="Rotor Speeds (RPM)", origin="rotor_speeds_rpm")
        thrust_plot = rrb.TimeSeriesView(name="Thrust (N)", origin="total_thrust_body_N")
        velocity_plot = rrb.TimeSeriesView(name="Velocity (m/s)", origin="velocity_world_m_s")
        episode_reward_plot = rrb.TimeSeriesView(name="Episode Reward Mean", origin="episode_reward_mean")

        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    drone_3d,
                    rrb.TextLogView(name="System Logs", origin="console"),
                    row_shares=[3, 1],
                ),
                rrb.Vertical(
                    rrb.Horizontal(actions_plot, rotor_speeds_plot),
                    rrb.Horizontal(angular_vel_plot, velocity_plot),
                    rrb.Horizontal(thrust_plot, observations_plot),
                    episode_reward_plot,
                ),
                column_shares=[1, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def _init_rerun(logs_path: Path, rrd_filename: str) -> None:
        if Log._rerun_initialized:
            return
        rr.init(f"[{rrd_filename.removesuffix('.rrd')}] " + str(logs_path))
        # Drone model setup
        rr.log("local/drone", rr.Asset3D(path=file_path / "drone.obj"), static=True)
        rr.log(
            "local/drone/camera",
            rr.Transform3D(translation=[0.03, 0, 0.04], quaternion=Rotation.from_euler('y', -20, degrees=True).as_quat()),
            rr.Pinhole(
                fov_y=1.2,
                aspect_ratio=1.7777778,
                camera_xyz=rr.ViewCoordinates.FLU,
                image_plane_distance=0.1,
            ),
        )
        Log._send_layout()
        rr.save(str(logs_path / rrd_filename))
        Log._rerun_initialized = True

    @staticmethod
    def _create_console_logger(
        name: str, output_path: Path | None = None
    ) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        logger.propagate = False 
        # Avoid adding duplicate handlers if get_logger is called twice
        # for the same underlying logging name.
        if logger.handlers:
            return logger

        logger.setLevel(logging.DEBUG)

        if output_path:
            file_handler = logging.FileHandler(output_path / f"{name}.log")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            "%(name)s: %(asctime)s %(levelname)s %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        return logger

    @staticmethod
    def debug(message: str) -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log._get_caller_logger().debug(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.DEBUG))

    @staticmethod
    def info(message: str) -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log._get_caller_logger().info(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.INFO))

    @staticmethod
    def warning(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        logger = Log._get_caller_logger()
        logger.warning(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.WARN))
        if traceback:
            logger.debug(f"{message}\n{traceback}")

    @staticmethod
    def error(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        logger = Log._get_caller_logger()
        logger.error(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.ERROR))
        if traceback:
            logger.debug(f"{message}\n{traceback}")

    @staticmethod
    def critical(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        logger = Log._get_caller_logger()
        logger.critical(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.CRITICAL))
        if traceback:
            logger.debug(f"{message}\n{traceback}")

    @staticmethod
    def log_drone_pose(position: np.ndarray, quaternion: np.ndarray):
        rr.set_time("unix_time", timestamp=time.time())
        rr.log(
            "local/drone",
            rr.Transform3D(
                translation=position,
                quaternion=(Rotation.from_quat(quaternion)).as_quat(),
                axis_length=0.1,
            ),
            static=False,
        )

    @staticmethod
    def log_velocity(
        position: np.ndarray, velocity: np.ndarray, model_name="drone/velocity"
    ):
        rr.log(
            model_name,
            rr.Arrows3D(
                origins=[position],
                vectors=[velocity * 0.5],  # Scale for visibility
                colors=[0, 0, 255],
            ),
        )


    @staticmethod
    def log_actions(actions: np.ndarray) -> None:
        """Log motor action values as time series."""
        for i, action_val in enumerate(actions):
            rr.log(f"actions/motor_{i}", rr.Scalars(float(action_val)))

    @staticmethod
    def log_observations(observations: np.ndarray) -> None:
        """Log observation vector as time series."""
        observations_names = [
            "velocity_body_x", "velocity_body_y", "velocity_body_z",
            "angular_velocity_body_x", "angular_velocity_body_y", "angular_velocity_body_z",
            "gravity_body_x", "gravity_body_y", "gravity_body_z",
            "orientation_error_x", "orientation_error_y", "orientation_error_z",
            "rpm_1", "rpm_2", "rpm_3", "rpm_4",
            "wp1_rel_x", "wp1_rel_y", "wp1_rel_z",
            "wp2_rel_x", "wp2_rel_y", "wp2_rel_z",
            "wp3_rel_x", "wp3_rel_y", "wp3_rel_z",
        ]
        for i, obs_val in enumerate(observations):
            rr.log(f"observations/{observations_names[i]}", rr.Scalars(float(obs_val)))

    @staticmethod
    def log_angular_velocity(angular_velocity_rad: np.ndarray) -> None:
        """Log angular velocity (converted to deg/s) as time series."""
        angular_vel_deg = np.degrees(angular_velocity_rad)
        rr.log("angular_velocity_deg_s/roll", rr.Scalars(float(angular_vel_deg[0])))
        rr.log("angular_velocity_deg_s/pitch", rr.Scalars(float(angular_vel_deg[1])))
        rr.log("angular_velocity_deg_s/yaw", rr.Scalars(float(angular_vel_deg[2])))

    @staticmethod
    def log_rotor_speeds(rpms: np.ndarray) -> None:
        """Log rotor speeds in RPM as time series."""
        for i, rpm in enumerate(rpms):
            rr.log(f"rotor_speeds_rpm/motor_{i}", rr.Scalars(float(rpm)))

    @staticmethod
    def log_thrust(total_thrust: np.ndarray) -> None:
        """Log total thrust in body frame (N) as time series."""
        rr.log("total_thrust_body_N/x", rr.Scalars(float(total_thrust[0])))
        rr.log("total_thrust_body_N/y", rr.Scalars(float(total_thrust[1])))
        rr.log("total_thrust_body_N/z", rr.Scalars(float(total_thrust[2])))

    @staticmethod
    def log_world_velocity(velocity: np.ndarray) -> None:
        """Log world-frame velocity (m/s) as time series."""
        rr.log("velocity_world_m_s/x", rr.Scalars(float(velocity[0])))
        rr.log("velocity_world_m_s/y", rr.Scalars(float(velocity[1])))
        rr.log("velocity_world_m_s/z", rr.Scalars(float(velocity[2])))

    @staticmethod
    def log_episode_reward_mean(reward_mean: float) -> None:
        """Log rolling mean episode reward as a time series."""
        rr.set_time("unix_time", timestamp=time.time())
        rr.log("episode_reward_mean/reward", rr.Scalars(reward_mean))

    @staticmethod
    def log_goal(goal_position: np.ndarray, goal_quat_xyzw: np.ndarray) -> None:
        """Log goal position and orientation."""
        rr.log(
            "goal",
            rr.Transform3D(
                translation=goal_position,
                quaternion=goal_quat_xyzw,
                axis_length=0.5,
            ),
            static=False,
        )

    @staticmethod
    def log_waypoints(wp_positions: np.ndarray, target_wp_idx: int) -> None:
        """Log the full waypoint track and highlight the current target.

        Args:
            wp_positions: (N, 3) array of all waypoint positions.
            target_wp_idx: Index of the current target waypoint.
        """
        # Full track in red
        rr.log(
            "local/track",
            rr.LineStrips3D([wp_positions], colors=[[255, 0, 0]]),
        )
        rr.log(
            "local/track/waypoints",
            rr.Points3D(wp_positions, colors=[[255, 0, 0]] * len(wp_positions), radii=0.08),
        )
        # Current target waypoint in blue
        rr.log(
            "local/track/current_goal",
            rr.Points3D([wp_positions[target_wp_idx]], colors=[[0, 120, 255]], radii=0.15),
        )

    @staticmethod
    def render(
        position: np.ndarray,
        quaternion_wxyz: np.ndarray,
        actions: np.ndarray,
        observations: np.ndarray,
        angular_velocity_rad: np.ndarray,
        rotor_speeds: np.ndarray,
        total_thrust_body: np.ndarray,
        velocity_world: np.ndarray,
        goal_quaternion_wxyz: np.ndarray,
        wp_positions: np.ndarray,
        target_wp_idx: int,
    ) -> None:
        """Render the environment state using rerun logging.

        All array inputs should be 1-D numpy arrays (already extracted for a
        single environment and moved to CPU).

        Quaternions are expected in **wxyz** order and are converted to xyzw
        internally where needed.
        """
        rr.set_time("unix_time", timestamp=time.time())

        # Convert wxyz -> xyzw for rerun
        quat_xyzw = np.array([
            quaternion_wxyz[1], quaternion_wxyz[2],
            quaternion_wxyz[3], quaternion_wxyz[0],
        ])
        goal_quat_xyzw = np.array([
            goal_quaternion_wxyz[1], goal_quaternion_wxyz[2],
            goal_quaternion_wxyz[3], goal_quaternion_wxyz[0],
        ])

        Log.log_drone_pose(position, quat_xyzw)
        Log.log_actions(actions)
        Log.log_observations(observations)
        Log.log_angular_velocity(angular_velocity_rad)
        Log.log_rotor_speeds(rotor_speeds)
        Log.log_thrust(total_thrust_body)
        Log.log_world_velocity(velocity_world)
        Log.log_goal(wp_positions[target_wp_idx], goal_quat_xyzw)
        Log.log_waypoints(wp_positions, target_wp_idx)