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

"""Run a LanguageAgent with memory, optional remote acting, and optional automatic dataset creation capabilities.

Optional:
While it is always recommended to explicitly define your observation and action spaces,
which can be set with a gym.Space object or any python object using the Sample class
(see examples/using_sample.py for a tutorial), you can have the recorder infer the spaces
by setting recorder="default" for automatic dataset recording.

For example:
    >>> agent = LanguageAgent(context=SYSTEM_PROMPT, model_src=backend, recorder="default")
    >>> agent.act_and_record("pick up the fork", image)

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

import click
from pydantic import Field

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.hardware.sim_interface import SimInterface
from mbodied.types.message import Message
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample


@click.command("hri")
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai"]))
@click.option("--backend_api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
@click.option("--disable_audio", default=False, help="Disable audio input/output")
@click.option(
    "--record_dataset",
    default="default",
    help="Recording action to take",
    type=click.Choice(["default", "omit"]),
)
def main(backend: str, backend_api_key: str, disable_audio: bool, record_dataset: bool) -> None:
    """Example for using LLMs for robot control. In this example, the language agent will perform double duty as both the cognitive and motor agent.

    Args:
        backend: The backend to use for the LanguageAgent (e.g., "openai").
        backend_api_key: The API key for the backend, i.e. OpenAI, Anthropic.
        disable_audio: If True, disables audio input/output.
        record_dataset: If True, enables recording of the interaction data for training.

    Example:
        To run the script with OpenAI backend and disable audio:
        python script.py --backend openai --disable_audio
    """
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"

    class AnswerAndActionsList(Sample):
        """A customized pydantic type for the robot's reply and actions."""

        answer: str | None = Field(
            default="",
            description="Short, one sentence answer to the user's question or request.",
        )
        actions: list[HandControl] | None = Field(
            default=[],
            description="List of actions to be taken by the robot.",
        )

    context = [
        Message(
            role="user",
            content=f"""You are a robot with vision capabilities.
        For each task given, you respond in JSON format. Here's the JSON schema:
        {AnswerAndActionsList.model_json_schema()}""",
        ),
        Message(role="assistant", content="Understood!"),
    ]

    cognitive_agent = LanguageAgent(
        context=context,
        api_key=backend_api_key,
        model_src=backend,
        recorder=record_dataset,  # Pass in "default" to recorder to record the dataset automatically.
    )

    robot = SimInterface()

    # Prefer to use use_pyaudio=False for MAC.
    audio = AudioAgent(use_pyaudio=False)

    while True:
        instruction = audio.listen()
        print("Instruction:", instruction)  # noqa

        # act_and_parse will get us the AnswerAndActionsList object directly.
        answer_actions = cognitive_agent.act_and_parse(instruction, robot.capture(), AnswerAndActionsList, record=True)
        print("Response:", answer_actions)  # noqa

        # Let the robot speak.
        if answer_actions.answer:
            audio.speak(answer_actions.answer)

        # Execute the actions with the robot interface.
        robot.do(answer_actions.actions)


if __name__ == "__main__":
    main()
