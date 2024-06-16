from typing import List, Union
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch

from mbodied_agents.types.controls import Motion
from mbodied_agents.agents.motion.motor_agent import MotorAgent
from mbodied_agents.types.sense.vision import Image
from mbodied_agents.types.controls import HandControl, Pose6D, JointControl


class OpenVlaAgent(MotorAgent):
    """OpenVLA agent to generate robot actions.

    Either specify gradio remote_server_name or set run_local=True to run locally.
    Note that OpenVLA is quite large and requires a lot of memory to run locally.

    See openvla_example_server.py for the an exmaple of the gradio server code.

    Remote is a gradio server taking: image, instruction, and unnorm_key as input.

    >>> openvla_agent = OpenVlaAgent(remote_server_name="http://1.2.3.4:1234")
    >>> hand_control = openvla_agent.remote_act("move forward", image) # xyzrpyg
    """

    def __init__(self, recorder="omit", remote_server_name: str | None = None, run_local=False, device="cuda", **kwargs):
        if remote_server_name is None and not run_local:
            raise ValueError(
                "Either remote_server_name or run_local must be provided.")
        super().__init__(recorder, remote_server_name, **kwargs)
        if run_local:
            self.device = device
            self.processor = AutoProcessor.from_pretrained(
                "openvla/openvla-7b", trust_remote_code=True)
            self.model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b",
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(self.device)

    def response_to_hand_control(self, response: str) -> HandControl:
        actions = response.strip('[]').split()
        actions = [float(item) for item in actions]
        return HandControl(
            pose=Pose6D(x=actions[0], y=actions[1], z=actions[2],
                        roll=actions[3], pitch=actions[4], yaw=actions[5]),
            grasp=JointControl(value=actions[6]),
        )

    def act(self, instruction: str, image: Image, unnorm_key: str = "bridge_orig") -> List['Motion']:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image.pil).to(
            self.device, dtype=torch.bfloat16)
        response = self.model.predict_action(
            **inputs, unnorm_key=unnorm_key, do_sample=False)
        return [self.response_to_hand_control(response)]

    def remote_act(self, instruction: str, image: Image, unnorm_key: str = "bridge_orig") -> List['Motion']:
        """Act based on the instruction and image using the remote server."""
        response = self.remote_actor.predict(
            image.base64, instruction, unnorm_key)
        return [self.response_to_hand_control(response)]


# Example usage:
# if __name__ == "__main__":
#     openvla_agent = OpenVlaAgent(
#         remote_server_name="http://1.2.3.4:1234/")
#     image = Image("resources/xarm.jpeg")
#     response = openvla_agent.remote_act("move forward", image)
#     print(response)
