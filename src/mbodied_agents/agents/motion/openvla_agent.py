from mbodied_agents.agents.motion.motor_agent import MotorAgent
from mbodied_agents.types.motion_controls import HandControl, JointControl, Motion, Pose6D
from mbodied_agents.types.sense.vision import Image


class OpenVlaAgent(MotorAgent):
    """OpenVLA agent to generate robot actions.

    Either specify gradio remote_server_name or set run_local=True to run locally.
    Note that OpenVLA is quite large and requires a lot of memory to run locally.

    See openvla_example_server.py for the an exmaple of the gradio server code.

    Remote is a gradio server taking: image, instruction, and unnorm_key as input.

    Example:
        >>> openvla_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")
        >>> hand_control = openvla_agent.remote_act("move forward", image)  # xyzrpyg
    """

    def __init__(
        self,
        recorder="omit",
        recorder_kwargs=None,
        model_src=None,
        model_kwargs=None,
        local_only: bool = False,
        **kwargs,
    ):
        # Specify the gradio server name.
        super().__init__(
            recorder=recorder,
            recorder_kwargs=recorder_kwargs,
            model_src=model_src,
            model_kwargs=model_kwargs,
            local_only=local_only,
            **kwargs,
        )

    def remote_act(self, instruction: str, image: Image, unnorm_key: str = "bridge_orig") -> Motion:
        """Act based on the instruction and image using the remote server."""
        response = self.remote_actor.predict(image.base64, instruction, unnorm_key)
        items = response.strip("[]").split()
        action = [float(item) for item in items]
        return HandControl(
            pose=Pose6D(x=action[0], y=action[1], z=action[2], roll=action[3], pitch=action[4], yaw=action[5]),
            grasp=JointControl(value=action[6]),
        )


# Example usage:
if __name__ == "__main__":
    openvla_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")
    image = Image("resources/xarm.jpeg")
    response = openvla_agent.remote_act("move forward", image)
    print(response)
