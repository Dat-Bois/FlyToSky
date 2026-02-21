import time
import psutil
import logging
import numpy as np
from typing import Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation

import rerun as rr
import rerun.blueprints as rrb

file_path = Path(__file__).parent
logged_model_names = {}


class ColoredFormatter(logging.Formatter):
    def __init__(self, template: str) -> None:
        super().__init__(template)
        # https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
        grey = "\x1b[38;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        self.FORMATS = {
            logging.DEBUG: grey + template + reset,
            logging.INFO: grey + template + reset,
            logging.WARNING: yellow + template + reset,
            logging.ERROR: red + template + reset,
            logging.CRITICAL: bold_red + template + reset,
        }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class Log:
    """A centralized and standardized high-level logger interface. </br>
    Incorporates both console, file logging, and rerun logging."""

    initialized: bool = False
    console: logging.Logger

    @staticmethod
    def init(rrd_filename: str = "") -> None:
        if not Log.initialized:
            output_path = file_path / "Logs" / time.strftime("%Y-%m-%d_%H-%M-%S")
            output_path.mkdir(parents=True, exist_ok=True)
            Log.console = Log.__create_console_logger(
                rrd_filename, output_path
            )
            Log.__init_rerun(output_path, rrd_filename+".rrd")
            Log.initialized = True

    @staticmethod
    def _send_layout():
        """Constructs and sends the blueprint with panels for all logged data."""
        Log.debug("Sending Rerun layout blueprint.")

        drone_3d = rrb.Spatial3DView(name="Drone View", origin="local")

        actions_plot = rrb.TimeSeriesView(name="Actions", origin="actions")
        observations_plot = rrb.TimeSeriesView(name="Observations", origin="observations")
        angular_vel_plot = rrb.TimeSeriesView(name="Angular Velocity (deg/s)", origin="angular_velocity_deg_s")
        rotor_speeds_plot = rrb.TimeSeriesView(name="Rotor Speeds (RPM)", origin="rotor_speeds_rpm")
        thrust_plot = rrb.TimeSeriesView(name="Thrust (N)", origin="total_thrust_body_N")
        velocity_plot = rrb.TimeSeriesView(name="Velocity (m/s)", origin="velocity_world_m_s")

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
                ),
                column_shares=[1, 1],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def __init_rerun(logs_path: Path, rrd_filename: str) -> None:
        rr.init(f"[{rrd_filename.removesuffix(".rrd")}] " + str(logs_path))
        Log._send_layout()
        rr.save(str(logs_path / rrd_filename))

    @staticmethod
    def __create_console_logger(
        name: str, output_path: Path | None = None
    ) -> logging.Logger:
        logger: logging.Logger = logging.getLogger(name)
        # logger.handlers.clear()  # Clear any existing handlers
        logger.setLevel(logging.DEBUG)  # Set the overall logging level

        if output_path:
            file_handler = logging.FileHandler(output_path / f"{name}.log")
            file_handler.setLevel(logging.DEBUG)  # Capture all logs
            file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)  # Only show INFO and above on console

        # Define log format
        console_formatter = ColoredFormatter(
            "%(name)s: %(asctime)s %(levelname)s %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        logger.setLevel(-1)

        return logger

    @staticmethod
    def debug(message: str) -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log.console.debug(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.DEBUG))

    @staticmethod
    def info(message: str) -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log.console.info(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.INFO))

    @staticmethod
    def warning(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log.console.warning(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.WARN))
        if traceback:
            Log.console.debug(f"{message}\n{traceback}")

    @staticmethod
    def error(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log.console.error(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.ERROR))
        if traceback:
            Log.console.debug(f"{message}\n{traceback}")

    @staticmethod
    def critical(message: str, traceback: str = "") -> None:
        rr.set_time("unix_time", timestamp=time.time())
        Log.console.critical(message)
        rr.log("console", rr.TextLog(message, level=rr.TextLogLevel.CRITICAL))
        if traceback:
            Log.console.debug(f"{message}\n{traceback}")

    @staticmethod
    def log_drone_pose(position: np.ndarray, quaternion: np.ndarray, model_name="drone/drone_model"):
        if model_name not in logged_model_names:
            rr.log(model_name, rr.Asset3D(path=file_path / "drone.obj"), static=True)
            rr.log(
                f"{model_name}/camera",
                rr.Transform3D(translation=[0.03, 0, 0.04], quaternion=Rotation.from_euler('y', -20, degrees=True).as_quat()),
                rr.Pinhole(
                    fov_y=1.2,
                    aspect_ratio=1.7777778,
                    camera_xyz=rr.ViewCoordinates.FLU,
                    image_plane_distance=0.1,
                    color=[255, 128, 0],
                    line_width=0.003,
                ),
            )
            logged_model_names[model_name] = True
        rr.log(
            model_name,
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
        for i, obs_val in enumerate(observations):
            rr.log(f"observations/{i}", rr.Scalars(float(obs_val)))

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