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

"""This is a simple script to collect dataset on robot's execution of actions.

The actions are recorded with a frequency specified when the robot's moving.
This dataset can be used to train a robotic transformer model directly.

Usage:
    export OPENAI_API_KEY=<your_openai_api_key>
    python examples/5_teach_robot_record_dataset.py --task "pick up the remote" --backend "openai"
"""

import os

import click

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.data.replaying import Replayer
from mbodied.robots import SimRobot
from mbodied.robots.robot_recording import RobotRecorder
from mbodied.types.message import Message
from mbodied.types.motion.control import HandControl


@click.command()
@click.option("--task", default="pick up the remote", help="The task to perform/record")
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai"]))
@click.option("--backend_api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
@click.option("--disable_audio", default=False, help="Disable audio input/output")
def main(task: str, backend: str, backend_api_key: str, disable_audio: bool) -> None:
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"

    context = [
        Message(
            role="user",
            content=f"""You are a robot. Respond in the following json schema:{HandControl.model_json_schema()}""",
        ),
        Message(role="assistant", content="Understood!"),
    ]

    cognitive_agent = LanguageAgent(
        context=context,
        api_key=backend_api_key,
        model_src=backend,
    )

    # Prefer to use use_pyaudio=False for MAC.
    audio = AudioAgent(use_pyaudio=False)

    # Use sim robot for this example.
    robot = SimRobot()
    robot_recorder = RobotRecorder(robot, frequency_hz=5)

    with robot_recorder.record(task):
        # Recording automatically starts here
        # You can say something like "move forward by 0.5 meters".
        # We are recording a single action here. You can put this in a while loop to record multiple actions.
        instruction = audio.listen()
        print("Instruction:", instruction)  # noqa
        action = cognitive_agent.act_and_parse(instruction, robot.capture(), HandControl)
        robot.do(action)

    # Let's look at the dataset we just collected!
    replayer = Replayer("example_dataset/example_record.h5")
    print("Replaying recorded actions in dataset:")  # noqa: T201
    for observation, action in replayer:
        print("Observation:", observation)  # noqa: T201
        print("Action:", action)  # noqa: T201


if __name__ == "__main__":
    main()
