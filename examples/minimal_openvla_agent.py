# Copyright 2024 Mbodi AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from mbodied.hardware.sim_interface import SimInterface
from mbodied.types.sense.vision import Image
from mbodied.agents.motion.openvla_agent import OpenVlaAgent


def get_image_from_camera() -> Image:
    """Get an image from the camera. Using a static example here."""
    resource = Path("resources") / "xarm.jpeg"
    return Image(resource, size=(224, 224))


def main() -> None:
    """Minimal exmaple running a motor agent, i.e. OpenVLA on your robot."""
    # Initialize the OpenVLA agent (currently pointing to mbodi's gradio endpoint for OpenVLA).
    motor_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")

    hardware_interface = SimInterface()  # SimInterface is a placeholder for hardware interface.

    instruction = input("Your instruction to OpenVLA:")

    while True:
        hand_control = motor_agent.act(instruction, get_image_from_camera())
        hardware_interface.do(hand_control)


if __name__ == "__main__":
    main()
