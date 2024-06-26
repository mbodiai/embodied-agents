# Copyright 2024 mbodi ai
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

from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.hardware.sim_interface import SimInterface
from mbodied.types.sense.vision import Image


def main() -> None:
    motor_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")

    # Subclass HardwareInterface and implement the do() method for your specific hardware.
    hardware_interface = SimInterface()
    
    # Use the same instruction throughout.
    instruction = input("Your instruction:")

    while True:
        # Image can be initialized with most image types, including file paths.
        image = Image(Path("resources") / "xarm.jpeg")
        hand_control = motor_agent.act(instruction, image)
        hardware_interface.do(hand_control)

if __name__ == "__main__":
    main()
