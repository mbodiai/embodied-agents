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

"""
For dataset recording, you can pass in the "default" argument to the agent to record the dataset automatically.

For example:
    >>> LanguageAgent(context=SYSTEM_PROMPT, model_src=backend, recorder="default")

Alternatively, you can define the recorder separately to record the space you want.
For example, to record the dataset with the image and instruction observation and AnswerAndActionsList as action:
    >>> observation_space = spaces.Dict({"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)})
    >>> action_space = AnswerAndActionsList(actions=[HandControl()] * 6).space()
    >>> recorder = Recorder(
    ...     'example_recorder',
    ...     out_dir='saved_datasets',
    ...     observation_space=observation_space,
    ...     action_space=action_space

To record:
    >>> recorder.record(
    ...     observation={
    ...         "image": image,
    ...         "instruction": instruction,
    ...     },
    ...     action=answer_actions,
    ... )
"""

import os
from pathlib import Path

import click
from gymnasium import spaces
from mbodied_agents.agents.language import LanguageAgent
from mbodied_agents.agents.sense.audio.audio_handler import AudioHandler
from mbodied_agents.base.sample import Sample
from mbodied_agents.data.recording import Recorder
from mbodied_agents.hardware.sim_interface import SimInterface
from mbodied_agents.types.motion_controls import HandControl
from mbodied_agents.types.sense.vision import Image
from pydantic import Field


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


# This prompt is used to provide context to the LanguageAgent.
SYSTEM_PROMPT = f"""
    You are a robot with vision capabilities.
    For each task given, you respond in JSON format. Here's the JSON schema:
    {AnswerAndActionsList.model_json_schema()}
    """


@click.command("hri")
@click.option(
    "--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai", "mbodi"])
)
@click.option("--disable_audio", default=False, help="Disable audio input/output")
@click.option(
    "--record_dataset", default="default", help="Recording action to take", type=click.Choice(["default", "omit"])
)
def main(backend: str, disable_audio: bool, record_dataset: bool) -> None:
    """Example for using LLMs for robot control. In this example, the language agent will perform double duty as both the cognitive and motor agent.

    Args:
        backend: The backend to use for the LanguageAgent (e.g., "openai").
        disable_audio: If True, disables audio input/output.
        record_dataset: If True, enables recording of the interaction data for training.

    Example:
        To run the script with OpenAI backend and disable audio:
        python script.py --backend openai --disable_audio
    """
    # Pass in "default" to recorder to record the dataset automatically.
    cognitive_agent = LanguageAgent(context=SYSTEM_PROMPT, model_src=backend, recorder=record_dataset)

    hatdware_interface = SimInterface()

    # Enable or disable audio input/output capabilities.
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"
    # Prefer to use use_pyaudio=False for MAC.
    audio = AudioHandler(use_pyaudio=False)

    while True:
        instruction = audio.listen()
        print("Instruction:", instruction)  # noqa

        resource = Path("resources") / "xarm.jpeg"
        example_image = Image(resource, size=(224, 224))

        response = cognitive_agent.act(instruction, example_image)

        # Since we are using
        # the language agent as the motor agent here, we'll have to parse the strings.
        response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
        print("Response:", response)  # noqa

        # Validate the response against the pydantic object.
        answer_actions = AnswerAndActionsList.model_validate_json(response)

        # Let the robot speak.
        if answer_actions.answer:
            audio.speak(answer_actions.answer)

        # Execute the actions with the robot interface.
        for action in answer_actions.actions:
            hatdware_interface.do(action)


if __name__ == "__main__":
    main()
