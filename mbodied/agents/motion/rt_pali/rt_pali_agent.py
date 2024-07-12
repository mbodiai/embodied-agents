from enum import Enum

from mbodied.agents.motion.motor_agent import MotorAgent
from mbodied.agents.motion.rt_pali.model.model import VLAModel
from mbodied.types.motion import Motion, RelativeMotionField


class TerminationStatus(Enum):
    ACTIVE = 0
    TERMINATED = 1

class GraspStatus(Enum):
    RELEASED = 0
    GRASPED = 1


class RtPaliMotion(Motion):
    """RtPaliMotion defines the motion characteristics for the RtPaliAgent.

    Attributes:
        terminated (TerminationStatus): Indicates if the motion is terminated.
        x (float): Relative motion in the x direction. Defaults to 0.0 with bounds [-1.0, 1.0].
        y (float): Relative motion in the y direction. Defaults to 0.0 with bounds [-1.0, 1.0].
        z (float): Relative motion in the z direction. Defaults to 0.0 with bounds [-1.0, 1.0].
        roll (float): Rotational motion around the x-axis. Defaults to 0.0 with bounds [-π, π].
        pitch (float): Rotational motion around the y-axis. Defaults to 0.0 with bounds [-π, π].
        yaw (float): Rotational motion around the z-axis. Defaults to 0.0 with bounds [-π, π].
        grasp (bool): Indicates if the grasp action is triggered.
    """
    terminated: TerminationStatus = TerminationStatus.ACTIVE
    x: float = RelativeMotionField(default=0.0, bounds=[-1.0, 1.0])
    y: float = RelativeMotionField(default=0.0, bounds=[-1.0, 1.0])
    z: float = RelativeMotionField(default=0.0, bounds=[-1.0, 1.0])
    roll: float = RelativeMotionField(default=0.0, bounds=['-pi', 'pi'])
    pitch: float = RelativeMotionField(default=0.0, bounds=['-pi', 'pi'])
    yaw: float = RelativeMotionField(default=0.0, bounds=['-pi', 'pi'])
    grasp: GraspStatus = GraspStatus.RELEASED


class RtPaliAgent(MotorAgent):
    """RtPaliAgent is responsible for generating motion actions based on image and task instructions.

    Attributes:
        model (VLAModel): The model used to generate action maps.
    """

    def __init__(self):
        """Initializes the RtPaliAgent and loads the VLAModel from a checkpoint, setting it to evaluation mode."""
        super().__init__()
        self.model = VLAModel().load_from_checkpoint(
            'mbodied/agents/motion/rt_pali/checkpoints/ra_pali_gemma-epoch=36.ckpt')
        self.model = self.model.to('cuda')
        self.model.eval()

    def act(self, image, task_instruction) -> Motion:
        """Generates a motion action based on the given image and task instruction.

        Args:
            image: The input image.
            task_instruction: The task instruction.

        Returns:
            RtPaliMotion: The generated motion action.
        """
        action_map = self.model.generate_action_map(image, task_instruction)
        return RtPaliMotion(
            terminated=TerminationStatus(action_map['terminated']),
            x=action_map['x'],
            y=action_map['y'],
            z=action_map['z'],
            roll=action_map['roll'],
            pitch=action_map['pitch'],
            yaw=action_map['yaw'],
            grasp=GraspStatus(action_map['grasp']),
        )
