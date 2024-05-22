from mbodied.types.controls import HandControl
from mbodied.hardware.interface import HardwareInterface


class SimInterface(HardwareInterface):
    """A simulated interface for testing and validating purposes."""

    def __init__(self):
        """Initializes the SimInterface and sets up the robot arm.

        position: [x, y, z, r, p, y, grasp]
        """
        self.home_pos = [0, 0, 0, 0, 0, 0, 0]
        self.current_pos = self.home_pos

    def do(self, motion: HandControl) -> None:
        """Executes a given HandControl motion.

        Args:
            motion: The HandControl motion to be executed.
        """
        print("Executing motion:", motion)
        self.current_pos[0] += motion.pose.x
        self.current_pos[1] += motion.pose.y
        self.current_pos[2] += motion.pose.z
        self.current_pos[3] += motion.pose.roll
        self.current_pos[4] += motion.pose.pitch
        self.current_pos[5] += motion.pose.yaw
        self.current_pos[6] = motion.grasp.value
        print("New position:", self.current_pos)

        return self.current_pos

    def get_pose(self) -> list[float]:
        """Gets the current pose of the robot arm.

        Returns:
            list[float]: A list of the current pose values [x, y, z, r, p, y, grasp].
        """
        return self.current_pos
