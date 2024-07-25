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

We are using SimRecordingInterface as a mock robot interface to record the dataset.
SimRecordingInterface.do() execute every robot action in 1 second with 0.1s steps.

Usage:
    export OPENAI_API_KEY=<your_openai_api_key>
    python examples/5_teach_robot_record_dataset.py --task "pick up the remote" --backend "openai"
"""

import os

import click
from gymnasium import spaces
from pydantic import Field

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.data.replaying import Replayer
from mbodied.hardware.sim_recording_interface import SimRecordingInterface
from mbodied.types.message import Message
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image


@click.command("hri")
@click.option("--task", default="pick up the remote", help="The task to perform/record")
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai"]))
@click.option("--backend_api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
@click.option("--disable_audio", default=False, help="Disable audio input/output")
def main(task: str, backend: str, backend_api_key: str, disable_audio: bool) -> None:
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"

    class ActionsList(Sample):
        """List of actions to be taken by the robot."""

        actions: list[HandControl] | None = Field(
            default=[],
            description="List of actions to be taken by the robot.",
        )

    context = [
        Message(
            role="user",
            content=f"""You are a robot with vision capabilities.
                    For each instruction given, you respond in JSON format. Here's the JSON schema:
                    {ActionsList.model_json_schema()}""",
        ),
        Message(role="assistant", content="Understood!"),
    ]

    cognitive_agent = LanguageAgent(
        context=context,
        api_key=backend_api_key,
        model_src=backend,
    )
    # Specify the recorder's kwargs for the dataset.
    recorder_kwargs = {
        "name": "example_record",
        "observation_space": spaces.Dict(
            {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
        ),
        "action_space": HandControl().space(),
        "out_dir": "example_dataset",
    }
    # Use SimRecordingInterface as a mock robot interface to record the dataset.
    robot = SimRecordingInterface(record_frequency=5, recorder_kwargs=recorder_kwargs)

    # Prefer to use use_pyaudio=False for MAC.
    audio = AudioAgent(use_pyaudio=False)

    while True:
        # You can say something like "move hand 0.5 forward", "move hand down by 0.5" etc.
        # Say stop to stop recording and look at the dataset.
        instruction = audio.listen()
        # Exit this loop by saying "stop".
        if instruction.lower().strip(".") == "stop":
            break
        print("Instruction:", instruction)  # noqa
        # act_and_parse will get us the AnswerAndActionsList object directly.
        actions = cognitive_agent.act_and_parse(instruction, robot.capture(), ActionsList)
        if actions.actions:
            # Execute the actions with the robot interface.
            robot.do_and_record(task, actions)

    # Let's look at the dataset we just collected!
    replayer = Replayer("example_dataset/example_record.h5")
    print("Replaying recorded actions in dataset:")  # noqa: T201
    for observation, action in replayer:
        print("Observation:", observation)  # noqa: T201
        print("Action:", action)  # noqa: T201


if __name__ == "__main__":
    main()
