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


"""This is a simple script to use RAG with embodied agents.

First insert some robot skills documents into the RAG database.
Then query LLM to output action on the robot with the skill retreived.

Usage:
    export OPENAI_API_KEY=<your_openai_api_key>
    python examples/6_robot_with_rag.py
"""

from pydantic import Field

from mbodied.agents.language import LanguageAgent, RagAgent
from mbodied.robots import SimRobot
from mbodied.types.motion.control import HandControl
from mbodied.types.sample import Sample


class AnswerAndActionsList(Sample):
    """A customized pydantic type for the robot's reply and actions."""

    answer: str | None = Field(
        description="Short, one sentence answer to the user's question or request.",
    )
    actions: list[HandControl] | None = Field(
        description="List of actions to be taken by the robot.",
    )


def main() -> None:
    rag_agent = RagAgent(collection_name="test_collection", path="./chroma", distance_threshold=1.5)
    # Add document to Rag database
    documents = [
        "You are a robot. To wave at the audience, go left and right by 0.1m for 3 times.",
        "You are a robot. To high five, move up and then move forward by 0.2m.",
    ]
    rag_agent.upsert(documents)

    # Initialize the language agent to control a robot.
    language_agent = LanguageAgent(
        model_src="openai", context=f"Respond in the following json schema:{AnswerAndActionsList.model_json_schema()}"
    )

    robot = SimRobot()

    instruction = "Hi robot, wave at the audience please."
    # This will retrieve the document about waving at the audience from the Rag database.
    instruction_with_rag = rag_agent.act(instruction, n_results=1)
    result = language_agent.act_and_parse(instruction=instruction_with_rag, parse_target=AnswerAndActionsList)
    print(result)  # noqa: T201
    robot.do(result.actions)

    instruction = "Hi robot, give me a high five!"
    # This will retrieve the document about giving a high five from the Rag database.
    instruction_with_rag = rag_agent.act(instruction, n_results=1)
    result = language_agent.act_and_parse(instruction=instruction_with_rag, parse_target=AnswerAndActionsList)
    print(result)  # noqa: T201
    robot.do(result.actions)


if __name__ == "__main__":
    main()
