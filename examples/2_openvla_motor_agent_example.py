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

"""This scripts provides a minimal exmaple to run robotic transformers, i.e. OpenVLA, as motor agent on the robot."""

from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.hardware.sim_interface import SimInterface


def main() -> None:
    # Use the remote gradio server as the agent actor.
    motor_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")

    # Subclass HardwareInterface and implement the do() method for your specific hardware.
    robot = SimInterface()

    # Use the same instruction throughout.
    instruction = input("Your instruction:")

    while True:
        # Image can be initialized with most image types, including file paths.
        hand_control = motor_agent.act(instruction, robot.capture())
        robot.do(hand_control)


if __name__ == "__main__":
    main()
