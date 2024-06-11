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

import os
import logging

import click
from pydantic import BaseModel, Field
from pydantic_core import from_json
from gym import spaces

from mbodied_agents.agents.language import CognitiveAgent
from mbodied_agents.agents.sense.audio_handler import AudioHandler
from mbodied_agents.base.sample import Sample
from mbodied_agents.hardware.sim_interface import SimInterface
from mbodied_agents.types.controls import HandControl
from mbodied_agents.types.vision import Image
from mbodied_agents.data.recording import Recorder


class AnswerAndActionsList(Sample):
    """A customized pydantic type for the robot's reply and actions.

    Attributes:
        answer: A short, one sentence answer to the user's question or request.
        actions: A list of actions (HandControl objects) to be taken by the robot.

    Example:
        >>> from mbodied_agents.types.controls import HandControl
        >>> data = {"answer": "Hello, world!", "actions": [{"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}]}
        >>> response = AnswerAndActionsList.model_validate(data)
        >>> response.answer
        'Hello, world!'
    """
    answer: str | None = Field(
        default="",
        description="Short, one sentence answer to the user's question or request.",
    )
    actions: list[HandControl] | None = Field(
        default=[],
        description="List of actions to be taken by the robot.",
    )


# This prompt is used to provide context to the CognitiveAgent.
SYSTEM_PROMPT = f"""
    You are a robot with vision capabilities. 
    For each task given, you respond in JSON format. Here's the JSON schema:
    {AnswerAndActionsList.model_json_schema()}
    """


@click.command("hri")
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai", "mbodi"]))
@click.option("--disable_audio", default=False, help="Disable audio input/output")
@click.option("--record_dataset", default=True, help="Record dataset for training automatically.")
def main(backend: str, disable_audio: bool, record_dataset: bool) -> None:
    """Main function to initialize and run the robot interaction.

    Args:
        backend: The backend to use for the CognitiveAgent (e.g., "openai").
        disable_audio: If True, disables audio input/output.
        record_dataset: If True, enables recording of the interaction data for training.

    Example:
        To run the script with OpenAI backend and disable audio:
        python script.py --backend openai --disable_audio
    """
    # Initialize the intelligent Robot Agent.
    robot_agent = CognitiveAgent(context=SYSTEM_PROMPT, api_service=backend)

    # Use a mock robot interface for movement visualization.
    robot_interface = SimInterface()

    # Enable or disable audio input/output capabilities.
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"
    audio = AudioHandler(use_pyaudio=False)  # Prefer to use use_pyaudio=False for MAC.

    # Data recorder for every conversation and action.
    if record_dataset:
        observation_space = spaces.Dict({
            'image': Image(size=(224, 224)).space(),
            'instruction': spaces.Text(1000)
        })
        action_space = AnswerAndActionsList(actions=[HandControl()] * 6).space()
        recorder = Recorder(
            'example_recorder',
            out_dir='saved_datasets',
            observation_space=observation_space,
            action_space=action_space
        )

    while True:
        # Listen for instructions.
        instruction = audio.listen()
        print("Instruction:", instruction)

        # Note: This is just an example vision image.
        image = Image("resources/xarm.jpeg", size=(224, 224))

        # Get the robot's response and actions based on the instruction and image.
        response = robot_agent.act(instruction, image)[0]
        response = response.replace("```json", "").replace("```", "")
        print("Response:", response)

        # Validate the response to the pydantic object.
        answer_actions = AnswerAndActionsList.model_validate(from_json(response))

        # Let the robot speak.
        if answer_actions.answer:
            audio.speak(answer_actions.answer)

        # Execute the actions with the robot interface.
        if answer_actions.actions:
            for action in answer_actions.actions:
                robot_interface.do(action)

        # Record the dataset for training.
        if record_dataset:
            recorder.record(
                observation={
                    'image': image,
                    'instruction': instruction,
                },
                action=answer_actions
            )


if __name__ == "__main__":
    main()
