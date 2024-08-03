import threading
import time
from queue import Queue
from typing import Any, Literal

from mbodied.data.recording import Recorder
from mbodied.robots import Robot


class RobotRecorder:
    """A class for recording robot observation and actions.

    Recording the observation and action of a robot hardware interface on the given robot
    from the constructor at a specified frequency. It leverages a queue and a worker
    thread to handle the recording asynchronously, ensuring that the main operations of the
    robot are not blocked.

    Robot class must implement the `get_robot_state` and `calculate_action` methods.` for the recorder to work.
    get_robot_state() gets the current state/pose of the robot. calculate_action() calculates the action between
    the new and old states.

    Usage:
        # Optional: Specify the kwargs for the recorder explicitly.
        recorder_kwargs = {
            "observation_space": spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)}),
            "action_space": HandControl().space(),
        }

        robot = SomeRobot()
        robot_recorder = RobotRecorder(robot, frequency_hz=5, recorder_kwargs=recorder_kwargs)
        with robot_recorder.record("pick up the fork"):
            # Recording automatically starts here
            robot.do(motion1)
            robot.do(motion2)

    Alternatively, you can use the start_recording() and stop_recording() methods to start and stop recording manually.
    """

    def __init__(
        self,
        robot: Robot,
        frequency_hz: int = 5,
        recorder_kwargs: dict[str, Any] = {},
        on_static: Literal["record", "omit"] = "omit",
    ) -> None:
        """Initializes the RobotRecorder.

        This constructor sets up the recording mechanism on the given robot, including the recorder instance,
        recording frequency, and the asynchronous processing queue and worker thread. It also
        initializes attributes to track the last recorded pose and the current instruction.

        Args:
            robot: The robot hardware interface to record.
            frequency_hz: Frequency at which to record pose and image data (in Hz).
            recorder_kwargs: Keyword arguments to pass to the Recorder constructor.
            on_static: Whether to record on static poses or not. If "record", it will record when the robot is not moving.
        """
        self.robot = robot

        self.recorder = Recorder(**recorder_kwargs)
        self.task = None

        self.last_recorded_pose = None
        self.last_image = None

        self.recording = False
        self.frequency_hz = frequency_hz
        self.record_on_static = on_static == "record"
        self.recording_queue = Queue()

        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()

    def __enter__(self):
        """Enter the context manager, starting the recording."""
        self.start_recording(self.task)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager, stopping the recording."""
        self.stop_recording()

    def record(self, task: str) -> "RobotRecorder":
        """Set the task and return the context manager."""
        self.task = task
        return self

    def reset_recorder(self) -> None:
        """Reset the recorder."""
        while self.recording:
            time.sleep(0.1)
        self.recorder.reset()

    def record_pose_and_image(self) -> None:
        """Records the current pose and captures an image at the specified frequency."""
        while self.recording:
            start_time = time.perf_counter()
            self.record_current_state()
            elapsed_time = time.perf_counter() - start_time
            # Sleep for the remaining time to maintain the desired frequency
            sleep_time = max(0, (1.0 / self.frequency_hz) - elapsed_time)
            time.sleep(sleep_time)

    def start_recording(self, task: str = "") -> None:
        """Starts the recording of pose and image."""
        if not self.recording:
            self.task = task
            self.recording = True
            self.recording_thread = threading.Thread(target=self.record_pose_and_image)
            self.recording_thread.start()

    def stop_recording(self) -> None:
        """Stops the recording of pose and image."""
        if self.recording:
            self.recording = False
            self.recording_thread.join()

    def _process_queue(self) -> None:
        """Processes the recording queue asynchronously."""
        while True:
            image, action, instruction = self.recording_queue.get()
            self.recorder.record(observation={"image": image, "instruction": instruction}, action=action)
            self.recording_queue.task_done()

    def record_current_state(self) -> None:
        """Records the current pose and image if the pose has changed."""
        pose = self.robot.get_robot_state()
        image = self.robot.capture()

        # This is the beginning of the episode
        if self.last_recorded_pose is None:
            self.last_recorded_pose = pose
            self.last_image = image
            return

        if pose != self.last_recorded_pose or self.record_on_static:
            action = self.robot.calculate_action(self.last_recorded_pose, pose)
            self.recording_queue.put(
                (
                    self.last_image,
                    action,
                    self.task,
                ),
            )
            self.last_image = image
            self.last_recorded_pose = pose

    def record_last_state(self) -> None:
        """Records the final pose and image after the movement completes."""
        self.record_current_state()
