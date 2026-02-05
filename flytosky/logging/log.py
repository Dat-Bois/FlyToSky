import time
import psutil
import logging
import rerun as rr
import numpy as np
from typing import Tuple
from pathlib import Path
from scipy.spatial.transform import Rotation

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


class ProcessMemoryUsage:
    BYTES_TO_GB = 1 / (1024**3)

    def __init__(self) -> None:
        # important: it gets the current process wherever this class is initialized
        self.proc = psutil.Process()

    def __call__(self) -> Tuple[float, float]:
        """Returns (process_rss_gb, system_used_gb)."""
        vm = psutil.virtual_memory()
        process_rss_gb = self.proc.memory_info().rss * self.BYTES_TO_GB
        system_used_gb = (vm.total - vm.available) * self.BYTES_TO_GB
        return process_rss_gb, system_used_gb


class Log:
    """A centralized and standardized high-level logger interface. </br>
    Incorporates both console, file logging, and rerun logging."""

    initialized = False
    bench_initialized = False
    console: logging.Logger
    _pm: ProcessMemoryUsage
    _last_hz_log_time: float

    @staticmethod
    def init_online(
        output_path: Path | None, rrd_filename: str = "perception.rrd"
    ) -> None:
        """Initialize logger for live/online recording."""
        if not Log.initialized:
            Log.console = Log.__create_console_logger(
                "perception", output_path
            )
            Log.__init_rerun(output_path, rrd_filename)
            Log._pm = ProcessMemoryUsage()
            Log._last_hz_log_time = time.time()
            Log.initialized = True

    @staticmethod
    def _send_layout(show_benchmarks: bool = False):
        """Constructs and sends the blueprint, conditionally adding benchmarks."""
        Log.debug("Sending Rerun layout blueprint.")
        if show_benchmarks:
            benchmark_section = rrb.Tabs(
                rrb.TimeSeriesView(
                    name="Main Thread Memory (GB)",
                    origin="benchmarking",
                    contents=["benchmarking/process_rss", "benchmarking/system_used"],
                ),
            )
            left_pane = rrb.Vertical(
                rrb.Spatial3DView(name="Drone View", origin="local"),
                benchmark_section,  # type: ignore
                row_shares=[3, 1],
            )
        else:
            left_pane = rrb.Vertical(
                rrb.Spatial3DView(name="Drone View", origin="local")
            )
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                left_pane,
                rrb.Vertical(
                    rrb.Spatial2DView(name="Camera Feed", origin="local/drone/camera"),
                    rrb.TextLogView(name="System Logs", origin="console"),
                ),
                column_shares=[30, 40],
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)

    @staticmethod
    def __init_rerun(logs_path: Path | None, rrd_filename: str) -> None:
        if logs_path:
            rr.init("[Perception] " + str(logs_path))
        else:
            rr.init("[Perception] Unlogged Run")

        Log._send_layout(show_benchmarks=False)
        # Set series line colors (static)
        rr.log(
            "benchmarking/process_rss",
            rr.SeriesLines(colors=[[0, 255, 0]], names=["Process RSS"]),
            static=True,
        )
        rr.log(
            "benchmarking/system_used",
            rr.SeriesLines(colors=[[255, 0, 0]], names=["System Used"]),
            static=True,
        )
        rr.log(
            "benchmarking/loop_hz",
            rr.SeriesLines(colors=[[255, 255, 0]], names=["Loop Hz"]),
            static=True,
        )
        # Set annotation context: blue for tent (0), yellow for mannequin (1)
        rr.log(
            "local/drone/camera/detections",
            rr.AnnotationContext(
                [
                    rr.AnnotationInfo(id=0, color=(0, 0, 255)),  # blue for tent
                    rr.AnnotationInfo(
                        id=1, color=(255, 255, 0)
                    ),  # yellow for mannequin
                ]
            ),
            static=True,
        )

        if logs_path:
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
    def benchmark() -> None:
        now = time.time()
        if not Log.bench_initialized:
            Log._send_layout(show_benchmarks=True)
            Log.bench_initialized = True

        elapsed = now - Log._last_hz_log_time
        if elapsed >= 1.0:
            process_rss_gb, system_used_gb = Log._pm()

            rr.set_time("unix_time", timestamp=now)
            rr.log("benchmarking/process_rss", rr.Scalars(process_rss_gb))
            rr.log("benchmarking/system_used", rr.Scalars(system_used_gb))

            Log._last_hz_log_time = now

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
    def log_drone_pose(pose: Pose) -> None:
        rr.set_time("unix_time", timestamp=pose.timestamp)
        rr.log(
            "local/drone",
            rr.Transform3D(
                translation=pose.position,
                quaternion=pose.rotation.as_quat(),
                axis_length=1,
            ),
        )



# def log_drone_pose(
#     position: np.ndarray, quaternion: np.ndarray, model_name="drone/drone_model"
# ):
#     if model_name not in logged_model_names:
#         rr.log(model_name, rr.Asset3D(path=file_path / "drone.obj"), static=True)
#         rr.log(
#             f"{model_name}/camera",
#             rr.Transform3D(translation=[0.03, 0, 0.04], quaternion=Rotation.from_euler('y', -20, degrees=True).as_quat()),
#             rr.Pinhole(
#                 fov_y=1.2,
#                 aspect_ratio=1.7777778,
#                 camera_xyz=rr.ViewCoordinates.FLU,
#                 image_plane_distance=0.1,
#                 color=[255, 128, 0],
#                 line_width=0.003,
#             ),
#         )
#         logged_model_names[model_name] = True
#     rr.log(
#         model_name,
#         rr.Transform3D(
#             translation=position,
#             quaternion=(Rotation.from_quat(quaternion)).as_quat(),
#         ),
#         rr.TransformAxes3D(0.1),
#         static=False,
#     )


# def log_gates(gate_info: list[tuple[np.ndarray, float]]):
#     """
#     Takes gate info as
#     list of (position, yaw) tuples
#     """
#     for i, (gate_pos, gate_yaw) in enumerate(gate_info):
#         # Log 3d gate model
#         obj_file_path = file_path / "gate.obj"

#         instance_path = f"gate_models/gate_{i}"
#         rr.log(
#             instance_path,
#             rr.Transform3D(
#                 translation=gate_pos,
#                 rotation=rr.Quaternion(
#                     xyzw=(
#                         Rotation.from_euler("XYZ", [0.0, 0.0, gate_yaw])
#                     ).as_quat()
#                 ),
#             ),
#             static=True,
#         )
#         rr.log(
#             f"{instance_path}/model",
#             rr.Asset3D(path=obj_file_path, albedo_factor=[0.9, 0.9, 0.9, 1.0]),
#             static=True,
#         )


# def log_velocity(
#     position: np.ndarray, velocity: np.ndarray, model_name="drone/velocity"
# ):
#     rr.log(
#         model_name,
#         rr.Arrows3D(
#             origins=[position],
#             vectors=[velocity * 0.5],  # Scale for visibility
#             colors=[0, 0, 255],
#         ),
#     )