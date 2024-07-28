import time
from pathlib import Path

from gymnasium import spaces

from mbodied.data.replaying import Replayer
from mbodied.hardware.recording_interface import RecordingHardwareInterface
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image


class SimRecordingInterface(RecordingHardwareInterface):
    """A simulated recording interface for testing and validating purposes.

    This class simulates the interface between the robot arm and the control system.

    Attributes:
        home_pos: The home position of the robot arm.
        current_pos: The current position of the robot arm.

    """

    def __init__(self, record_frequency=5, recorder_kwargs=None):
        """Initializes the SimInterface and sets up the robot arm.

        position: [x, y, z, r, p, y, grasp]
        """
        self.home_pos = [0, 0, 0, 0, 0, 0, 0]
        self.current_pos = self.home_pos
        if recorder_kwargs is None:
            recorder_kwargs = {
                "name": "sim_record",
                "observation_space": spaces.Dict(
                    {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
                ),
                "action_space": HandControl().space(),
                "out_dir": "sim_dataset",
            }
        super().__init__(record_frequency, recorder_kwargs)

    def do(self, motion: HandControl) -> list[float]:
        """Executes a given HandControl motion and returns the new position of the robot arm.

        This simulates the execution of a motion for 1 second. It divides the motion into 10 steps.

        Args:
            motion: The HandControl motion to be executed.
        """
        print("Executing motion:", motion)  # noqa: T201

        # Number of steps to divide the motion into
        steps = 10
        sleep_duration = 0.1
        step_motion = [value / steps for value in motion.flatten()]

        for _ in range(steps):
            self.current_pos = [round(x + y, 5) for x, y in zip(self.current_pos, step_motion)]
            time.sleep(sleep_duration)

        print("New position:", self.current_pos)  # noqa: T201
        return self.current_pos

    def get_robot_state(self) -> HandControl:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return HandControl.unflatten(self.current_pos)

    def calculate_action(self, old_pose: HandControl, new_pose: HandControl) -> HandControl:
        """Calculates the action between two poses."""
        # Calculate the difference between the old and new poses. Use absolute value for grasp.
        old = list(old_pose.flatten())
        new = list(new_pose.flatten())
        result = [(new[i] - old[i]) for i in range(len(new) - 1)] + [new[-1]]
        print(f"Record action: {result}, instruction: {self.current_instruction}")  # noqa: T201
        return HandControl.unflatten(result)

    def capture(self, **_) -> Image:
        """Captures an image."""
        resource = Path("resources") / "xarm.jpeg"
        return Image(resource, size=(224, 224))


if __name__ == "__main__":
    # Test the SimRecordingInterface
    sim_recording_interface = SimRecordingInterface()
    motion = HandControl.unflatten([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0])
    sim_recording_interface.do_and_record("pick up the fork", motion)
    replayer = Replayer("sim_dataset/sim_record.h5")
    print("Replaying recorded actions in dataset:")  # noqa: T201
    for _, action in replayer:
        print("Action:", action)  # noqa: T201
