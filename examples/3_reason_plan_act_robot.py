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
"""Example for layered (chained) robotic control.

It uses a LanguageAgent for planning and reasoning, and an OpenVLA agent as motor agent for generating robot actions.
This is a minimal example to demonstrate the interaction between the cognitive agent and the motor agent.
More complex applications can be built using the modules on top of this example.
"""

import json
import os
from pathlib import Path

import rich_click as click

from mbodied.agents.language import LanguageAgent
from mbodied.agents.motion.openvla_agent import OpenVlaAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.robots import SimRobot
from mbodied.types.message import Message


@click.command()
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai"]))
@click.option("--backend_api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
@click.option("--disable_audio", default=False, help="Disable audio input/output")
def main(backend: str, backend_api_key: str, disable_audio: bool) -> None:
    """Example for layered robotic control.

    It uses a LanguageAgent for planning and reasoning, and an OpenVLA agent for generating robot actions.
    This is a minimal example to demonstrate the interaction between the cognitive agent and the motor agent.
    More complex applications can be built using the modules on top of this example.

    Args:
        backend: The backend to use for the LanguageAgent (e.g., "openai").
        disable_audio: If True, disables audio input/output.
        record_dataset: If True, enables recording of the interaction data for training.

    Example:
        To run the script with OpenAI backend and disable audio:
        python script.py --backend openai --disable_audio
    """
    context = [
        Message(
            role="user",
            content="You are a robot action planner with vision capabilities. For each task given, you respond a JSON list of strings of actions to take from what you see.",
        ),
        Message(role="assistant", content="Understood!"),
    ]
    # Use the specified backend, i.e. OpenAI as the cognitive agent for planning and reasoning.
    cognitive_agent = LanguageAgent(
        context=context,
        api_key=backend_api_key,
        model_src=backend,
        # Pass in "default" to recorder to record the dataset automatically.
        # recorder="default",
    )

    # Use OpenVLA as the motor agent for generating robot actions.
    motor_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")
    # A mocked out interface for robot control.
    robot = SimRobot()

    if disable_audio:
        os.environ["NO_AUDIO"] = "1"
    # Prefer to use use_pyaudio=False for MAC.
    audio = AudioAgent(use_pyaudio=False)

    # Listen to the instruction from the user.
    instruction = audio.listen()

    # Get the plan from cognitive agent.
    response = cognitive_agent.act(instruction, robot.capture())
    response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
    plans = json.loads(response)
    print("Cognitive agent's plan: ", plans)  # noqa: T201

    # Execute the plan from cognitive agent with motor agent.
    for step in plans:
        print("\nMotor agent is executing step: ", step, "\n")  # noqa: T201
        counter = 5  # A counter for testing purposes only as done signal.
        while True:
            hand_control = motor_agent.act(step, robot.capture())
            robot.do(hand_control)
            counter -= 1
            if counter == 0:
                break


if __name__ == "__main__":
    main()
