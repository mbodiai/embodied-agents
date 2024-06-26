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
    >>> recorder.push_to_hub()
"""

import os
from pathlib import Path

import click
from gymnasium import spaces
from pydantic import Field

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.audio.audio_handler import AudioHandler
from mbodied.base.motion import AbsoluteMotionField, RelativeMotionField
from mbodied.base.sample import Sample
from mbodied.data.recording import Recorder
from mbodied.hardware.sim_interface import SimInterface
from mbodied.types.message import Message
from mbodied.types.motion_controls import FullJointControl, HandControl, JointControl
from mbodied.types.sense.vision import Image


@click.command("hri")
@click.option(
    "--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai"])
)
@click.option("--backend_api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
@click.option("--disable_audio", default=False, help="Disable audio input/output")
@click.option(
    "--record_dataset", default="auto", help="Recording action to take", type=click.Choice(["auto", "omit"])
)
def main(backend: str, backend_api_key: str, disable_audio: bool, record_dataset: bool) -> None:
    os.environ["NO_AUDIO"] = "1" if disable_audio else "0"

    # Customize a  Motion by adding or overriding fields.
    class FineGrainedHandControl(HandControl):
        """Custom HandControl with an additional field."""
        comment: str = Field(None, description="A comment to voice aloud.")
        
        # Any attempted validation will fail if the bounds are not satisfied.
        index: FullJointControl = AbsoluteMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
        thumb: FullJointControl = RelativeMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
    
    # Define the observation and action spaces.
    observation_space = spaces.Dict(
        {
            "image": Image(size=(224, 224)).space(),
            "instruction": spaces.Text(1000),
        },
    )
    action_space = FineGrainedHandControl().space()
        
    system_prompt = """You are a robot with vision capabilities. 
    For each task given, you respond with a plan of actions as a json list."""

    cognition = LanguageAgent(context=system_prompt, api_key=backend_api_key, model_src=backend)
    
    system_prompt = f"""You control a robot's hand based on the instructions given and the image provided.
        You always respond with an action of the form: {FineGrainedHandControl.model_json_schema()}.
   """ # Descriptions are taken from the docstrings or MotionField arguments.

    # Use a language agent as a motor agent proxy.
    motion = LanguageAgent(
        context=Message(role="system", content=system_prompt),
        model_src=backend,
        recorder="auto",
        recorder_kwargs={"observation_space": observation_space, "action_space": action_space},
    )
    
    # Subclass HardwareInterface and implement the do() method for your specific hardware.
    hardware_interface = SimInterface() 
    audio = AudioHandler(use_pyaudio=False) # PyAudio is buggy on Mac.
    
    # Recieve inputs.
    image = Image(Path("resources") / "xarm.jpeg")
    instruction = audio.listen()
    
    # Get the plan.
    plan = cognition.act(instruction, image)

    for instruction in plan.strip("[]").split(","):
        # Pydantic de-serializes and validates json under the hood.
        response = motion.act_and_parse(instruction, image, parse_target=FineGrainedHandControl)

        # Listen to the robot's reasoning.
        if response.comment:
            audio.speak(response.comment)
        
        hardware_interface.do(response)


if __name__ == "__main__":
    main()
