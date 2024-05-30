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
from pydantic import Field
from pydantic_core import from_json
from gym import spaces

from mbodied_agents.agents.language import CognitiveAgent
from mbodied_agents.agents.sense.audio_handler import AudioHandler
from mbodied_agents.base.sample import Sample
from mbodied_agents.hardware.sim_interface import SimInterface
from mbodied_agents.types.controls import HandControl
from mbodied_agents.types.vision import Image
from mbodied_agents.data.recording import Recorder


# This can be customized to be anything you like.
# In this example, we have an answer field, which is what the robot speaks to you,
# and a list of hand controls (X,Y,Z,R,P,Y) which is the end effector delta for robot's hand.
class AnswerAndActionsList(Sample):
    """A customized pydantic type for robot's reply and actions."""
    answer: str | None = Field(
        default="",
        description="Short, one sentence answer to user's question or request.",
    )
    actions: list[HandControl] | None = Field(
        default=[],
        description="List of actions to be taken by the robot.",
    )


SYSTEM_PROMPT = f"""
    You are robot with vision capabilities. 
    For each task given, you respond in JSON format. Here's the JSON schema:
    {AnswerAndActionsList.model_json_schema()}
    """


@click.command("hri")
@click.option("--backend", default="openai", help="The backend to use", type=click.Choice(["anthropic", "openai", "mbodi"]))
@click.option("--disable_audio", default=False, help="Disable audio input/output")
@click.option("--record_dataset", default=True, help="Record dataset for training automatically.")
def main(backend, disable_audio, record_dataset) -> None:
    # Initialize the intelligent Robot Agent.
    # For this example, we use OpenAI by default as the backend.
    # Cognitive Agent is the entry point for your intelligent robot.
    # It's responsible of handling the planning, reasoning, or action output.
    # You may even have multiple Cognitive Agent in parallel handling different aspect for your robot,i.e. planning, motion control, sensory, etc.
    # For the simplest example, we can use OpenAI's GPT for the cognitive Agent here. Anthropic is also supported here.
    # Mbodi's backend and HuggingFace backend is also upcoming.
    # The system prompt is minimal. You can provide more context if you like.
    robot_agent = CognitiveAgent(context=SYSTEM_PROMPT, api_service=backend)

    # Use a mock robot interface for movement visualization.
    robot_interface = SimInterface()

    # Enable audio input/output capabilities.
    if disable_audio:
        os.environ["NO_AUDIO"] = "1"
    audio = AudioHandler(
        use_pyaudio=False)  # For MAC, prefer to use use_pyaudio=False.

    # Data recorder for every conversation and every action.
    # A new dataset is created each time you interact, teach or train the robot.
    # Available for further augmentation finetuning and model training.
    if record_dataset:
        observation_space = spaces.Dict({
            'image': Image(size=(224, 224)).space(),
            'instruction': spaces.Text(1000)
        })
        action_space = AnswerAndActionsList(actions=[HandControl()] * 6).space()
        recorder = Recorder('example_recorder',
                            out_dir='saved_datasets',
                            observation_space=observation_space,
                            action_space=action_space)

    while True:
        instruction = audio.listen()
        print("Instruction:", instruction)

        # Note: This is just an example vision image.
        # Customize this for your robot's observation.
        image = Image("resources/xarm.jpeg", size=(224, 224))

        response = robot_agent.act(instruction, image)[0]
        response = response.replace("```json", "").replace("```", "")
        print("Response:", response)
        # Validate the response to the pydantic object.
        answer_actions = AnswerAndActionsList.model_validate(
            from_json(response))

        # Let the robot speak.
        if answer_actions.answer:
            audio.speak(answer_actions.answer)

        # Execute the actions with the robot interface.
        if answer_actions.actions:
            for action in answer_actions.actions:
                robot_interface.do(action)

        if record_dataset:
            # Record the dataset for training.
            recorder.record(observation={
                'image': image,
                'instruction': instruction,
            },
                            action=answer_actions)


if __name__ == "__main__":
    main()
