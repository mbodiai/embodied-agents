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

import click

from mbodied.agents.language import LanguageAgent
from mbodied.data.replaying import Replayer
from mbodied.robots import SimRobot
from mbodied.types.motion.control import HandControl


@click.command()
@click.option("--task", default="pick up the remote", help="The task to perform/record")
@click.option("--api_key", default=None, help="The API key for the backend, i.e. OpenAI, Anthropic")
def main(task: str, api_key: str) -> None:
    cognitive_agent = LanguageAgent(
        context=f"""You are a robot. Respond in the following json schema:{HandControl.model_json_schema()}""",
        api_key=api_key,
        model_src="openai",
    )

    robot = SimRobot()
    # Initialize the recorder for the robot.
    robot.init_recorder(
        frequency_hz=5,
        recorder_kwargs={
            "name": "example_record.h5",
            "out_dir": "example_dataset",
        },
    )

    # You can use the context manager to record the actions.
    with robot.record(task):
        # Recording automatically starts here
        instruction = "move arm forward by 0.5 meter please!"
        print("Instruction:", instruction)  # noqa
        action = cognitive_agent.act_and_parse(instruction, robot.capture(), HandControl)
        robot.do(action)

    # Alternatively, you can use the start_recording and stop_recording methods.
    robot.start_recording(task=task)
    # Recording starts here
    instruction = "move arm left by 0.5 meter please!"
    print("Instruction:", instruction)
    action = cognitive_agent.act_and_parse(instruction, robot.capture(), HandControl)
    robot.do(action)
    robot.stop_recording()

    # Let's look at the dataset we just collected!
    replayer = Replayer("example_dataset/example_record.h5")
    print("Replaying recorded actions in dataset:")  # noqa: T201
    for observation, action, state in replayer:
        print("Observation:", type(observation))  # noqa: T201
        print("Action:", action)  # noqa: T201
        print("State:", state)  # noqa: T201


if __name__ == "__main__":
    main()
