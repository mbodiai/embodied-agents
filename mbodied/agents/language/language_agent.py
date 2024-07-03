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
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import List, Literal, TypeAlias

from art import text2art
from pydantic import AnyUrl, DirectoryPath, FilePath, NewPath

from mbodied.agents import Agent
from mbodied.agents.backends import OpenAIBackend
from mbodied.types.message import Message
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image

SupportsOpenAI: TypeAlias = OpenAIBackend


@dataclass
class Reminder:
    """A reminder to show the agent a prompt every n messages."""

    prompt: str | Image | Message
    n: int


def make_context_list(context: list[str | Image | Message] | Image | str | Message | None) -> List[Message]:
    """Convert the context to a list of messages."""
    if isinstance(context, list):
        return [Message(content=c) if not isinstance(c, Message) else c for c in context]
    if isinstance(context, Message):
        return [context]
    if isinstance(context, str | Image):
        return [Message(role="user", content=[context]), Message(role="assistant", content="Understood.")]
    return []


class LanguageAgent(Agent):
    """An agent that can interact with users using natural language.

    This class extends the functionality of a base Agent to handle natural language interactions. It manages memory, dataset-recording, and asynchronous remote inference, supporting multiple platforms including OpenAI, Anthropic, and Gradio.

    Attributes:
        reminders (List[Reminder]): A list of reminders that prompt the agent every n messages.
        context (List[Message]): The current context of the conversation.
        Inherits all attributes from the parent class `Agent`.

    Examples:
        Basic usage with OpenAI:
            >>> cognitive_agent = LanguageAgent(api_key="...", model_src="openai", recorder="default")
            >>> cognitive_agent.act("your instruction", image)

        Automatically act and record to dataset:
            >>> cognitive_agent.act_and_record("your instruction", image)
    """

    _art_printed = False

    def __init__(  # noqa
        self,
        model_src: Literal["openai", "anthropic"]
        | SupportsOpenAI
        | AnyUrl
        | FilePath
        | DirectoryPath
        | NewPath = "openai",
        context: list | Image | str | Message = None,
        api_key: str | None = os.getenv("OPENAI_API_KEY"),
        model_kwargs: dict = None,  # noqa
        recorder: Literal["default", "omit"] | str = "omit",
        recorder_kwargs: dict = None,  # noqa
        local_only: bool = False,  # noqa
    ) -> None:
        """LanguageAgent with memory, dataset-recording, and remote infererence support. Always returns a string.

        Supported datasets: HDF5, Datasets, JSON, CSV, Parquet.
        Supported inference backends: OpenAI, Anthropic, Gradio.

        Methods:
            - act(instruction: str, image: Image = None, context: list | str | Image | Message = None, model=None, **kwargs) -> str
            - forget_last() -> Message
            - forget(everything=False, last_n: int = -1) -> None
            - remind_every(prompt: str | Image | Message, n: int) -> None

        Args:
            model: The model or weights path to setup and preload if applicable.
            context: The starting context to use for the conversation. Can be a list of messages, an image, a string,
                or a message. If a string is provided, it will be interpreted as a user message.
            api_key: The API key to use for the remote actor (if applicable).
            model_src: Any of:
                1. A path to a model's weights.
                2. A string or mbodied.agents.backends.openai_backend.OpenAIBackendMixin subclass representing a backend API.
                3. Any huggingface spaces path (mbodiai/openvla-quantized) or URL hosting a gradio server. See https://www.gradio.app/guides/getting-started-with-the-python-client for more details.
            model_kwargs: Additional keyword arguments to pass to the model source. See mbodied.agents.backends.
            recorder: The recorder config or name to use for the agent to record observations and actions.
            recorder_kwargs: Additional keyword arguments to pass to the recorder such as push_to_cloud.
            local_only: Whether to use the local model only. If True, the agent will not use a remote actor for inference.
        """
        if not LanguageAgent._art_printed:
            print("Welcome to")  # noqa: T201
            print(text2art("mbodi"))  # noqa: T201
            print("A platform for intelligent embodied agents.\n\n")  # noqa: T201
            LanguageAgent._art_printed = True
        self.reminders: List[Reminder] = []
        print(f"Initializing language agent for robot using : {model_src}")  # noqa: T201

        super().__init__(
            recorder=recorder,
            recorder_kwargs=recorder_kwargs,
            model_src=model_src,
            model_kwargs=model_kwargs,
            api_key=api_key,
            local_only=local_only,
        )

        self.context = make_context_list(context)

    def forget_last(self) -> Message:
        """Forget the last message in the context."""
        try:
            return self.context.pop(-1)
        except IndexError:
            logging.warning("No message to forget in the context")

    def forget(self, everything=False, last_n: int = -1) -> None:
        """Forget the last n messages in the context."""
        if everything:
            self.context = []
            return
        for _ in range(last_n):
            self.forget_last()

    def history(self) -> List[Message]:
        """Return the conversation history."""
        return self.context

    def remind_every(self, prompt: str | Image | Message, n: int) -> None:
        """Remind the agent of the prompt every n messages."""
        message = Message([prompt]) if not isinstance(prompt, Message) else prompt
        self.reminders.append(Reminder(message, n))

    def _check_for_reminders(self) -> None:
        """Check if there are any reminders to show."""
        for reminder, n in self.reminders:
            if len(self.context) % n == 0:
                self.context.append(reminder)

    def act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: Sample = Sample,
        context: list | str | Image | Message = None,
        model=None,
        **kwargs,
    ) -> Sample:
        """Responds to the given instruction, image, and context and parses the response into a Sample object."""
        response = self.act(instruction, image, context, model, **kwargs)
        response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
        try:
            response = parse_target.model_validate_json(response)
        except Exception as e:
            error = f"Error parsing response: {e}"
            logging.error(error)
            logging.info(f"Recieved response: {response}. Retrying with error message.")

            instruction = instruction + "avoid the following error: " + error
            response = self.act(instruction, image, context, model, **kwargs)
            response = response.replace("```json", "").replace("```", "").replace("\n", "").strip()
            response = parse_target.model_validate_json(response)
        return response

    async def async_act_and_parse(
        self,
        instruction: str,
        image: Image = None,
        parse_target: Sample = Sample,
        context: list | str | Image | Message = None,
        model=None,
        **kwargs,
    ) -> Sample:
        """Responds to the given instruction, image, and context asynchronously and parses the response into a Sample object."""
        return await asyncio.to_thread(
            self.act_and_parse, instruction, image, parse_target, context, model=model, **kwargs
        )

    def act(
        self, instruction: str, image: Image = None, context: list | str | Image | Message = None, model=None, **kwargs
    ) -> str:
        """Responds to the given instruction, image, and context.

        Uses the given instruction and image to perform an action.

        Args:
            instruction: The instruction to be processed.
            image: The image to be processed.
            context: Additonal context to include in the response. If context is a list of messages, it will be interpreted
                as new memory.
            model: The model to use for the response.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The response to the instruction.

        Example:
            >>> agent.act("Hello, world!", Image("scene.jpeg"))
            "Hello! What can I do for you today?"
            >>> agent.act("Return a plan to pickup the object as a python list.", Image("scene.jpeg"))
            "['Move left arm to the object', 'Move right arm to the object']"
        """
        self._check_for_reminders()
        memory = self.context
        if context and all(isinstance(c, Message) for c in context):
            memory += context
            context = []

        # Prepare the inputs
        inputs = [instruction]
        if image is not None:
            inputs.append(image)
        if context:
            inputs.extend(context if isinstance(context, list) else [context])
        message = Message(role="user", content=inputs)

        model = model or kwargs.pop("model", None)
        response = self.actor.act(message, memory, model=model, **kwargs)

        self.context.append(message)
        self.context.append(Message(role="assistant", content=response))
        return response
