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

from typing import List, Union
import logging
from art import text2art

from mbodied_agents.agents.backends import AnthropicBackend, OpenAIBackend, OllamaBackend, MbodiBackend, HuggingFaceBackend
from mbodied_agents.agents.language.language_agent import LanguageAgent
from mbodied_agents.base.sample import Sample
from mbodied_agents.data.recording import Recorder
from mbodied_agents.types.message import Message
from mbodied_agents.types.vision import Image


class CognitiveAgent(LanguageAgent):
    """The entry point for the Embodied Agents.
    
    This language-based Cognitive Agent that has the capability to connect to different backend services.

    This class extends `LanguageAgent` and includes methods for recording conversations,
    managing context, looking up messages, forgetting messages, storing context, and acting
    based on an instruction and an image.
    """
    _art_printed = False
    def __init__(
        self,
        model: str = None,
        temperature: float = 0,
        context: Union[list, str, Message] = None,
        api_key: str | None = None,
        api_service: str = "openai",  # 'openai' or 'anthropic' or 'ollama'
        # 'default' or 'vision' or Recorder
        recorder: Union[str, Recorder] = None,
        client=None,
        backend=None,
        **kwargs,
    ) -> None:
        """Initializes the CognitiveAgent with the given parameters.

        Args:
            model: The model name to be used.
            temperature: The temperature setting for the language model.
            context: Initial context which can be a list, a string, or a Message.
            api_key: The API key for the backend service.
            api_service: The backend service to use ('openai' or 'anthropic').
            recorder: The recorder to use for recording conversations.
            client: The client to use for API requests.
            backend: An optional backend instance.
            **kwargs: Additional keyword arguments.
        """
        if not CognitiveAgent._art_printed:
            print("Welcome to")
            print(text2art("Mbodi"))
            print("A platform for intelligent embodied agents.\n\n")
            CognitiveAgent._art_printed = True        
        print(f"Initializing cognitive agent for robot using backend: {api_service}")

        super().__init__(recorder=recorder, **kwargs)
        if backend is not None:
            self.backend = backend
        elif api_service == "openai":
            self.starting_context = OpenAIBackend.INITIAL_CONTEXT
            self.backend = OpenAIBackend(api_key, client=client, **kwargs)
            self.model = OpenAIBackend.DEFAULT_MODEL if model is None else model
        elif api_service == "anthropic":
            self.starting_context = AnthropicBackend.INITIAL_CONTEXT
            self.backend = AnthropicBackend(api_key, client=client, **kwargs)
            self.model = AnthropicBackend.DEFAULT_MODEL if model is None else model
        # # Upcoming backends:
        elif api_service == "mbodi":
            self.starting_context = MbodiBackend.INITIAL_CONTEXT
            self.backend = MbodiBackend(url=MbodiBackend.API_URL)
            self.model = MbodiBackend.DEFAULT_MODEL if model is None else model
        # elif api_service == "huggingface":
        #     self.starting_context = HuggingFaceBackend.INITIAL_CONTEXT
        #     self.backend = HuggingFaceBackend()
        #     self.model = HuggingFaceBackend.DEFAULT_MODEL if model is None else model
        # Upcoming:
        # elif api_service == "mbodi":
        #     self.starting_context = MbodiBackend.INITIAL_CONTEXT
        #     self.backend = MbodiBackend(url=MbodiBackend.API_URL)
        #     self.model = MbodiBackend.DEFAULT_MODEL if model is None else model
        # elif api_service == "ollama":
        #     self.starting_context = OllamaBackend.INITIAL_CONTEXT
        #     self.backend = OllamaBackend(api_key, client=client, **kwargs)
        #     self.model = OllamaBackend.DEFAULT_MODEL if model is None else model
        else:
            raise ValueError(f"Unsupported API service {api_service}")
        self.temperature = temperature

        self.context = []
        if isinstance(context, list):
            self.context = context
        elif isinstance(context, Message):
            self.context.append(context)
        elif isinstance(context, str):
            self.context.append(
                Message(role="user", content=[Sample(context)]))
            self.context.append(
                Message(role="assistant", content=[Sample("Got it!")]))
        else:
            self.context = self.starting_context

    def on_completion_response(self, response: str) -> List[str]:
        """Processes the response from the completion API.

        Args:
            response: The response from the completion API.

        Returns:
            List[str]: The processed response as a list of strings.
        """
        return [response]

    def act(self, text_or_inputs: Union[str, list] = None, image: Image = None, context=None, **kwargs) -> List[str]:
        """Performs an action based on the given instruction.

        This method allows the option to use accumulated context or start from scratch with new context.

        Args:
            text_or_inputs: The instruction to be processed.
            image: The image associated with the instruction, can be a NumPy array, Pillow Image, or file path.
            context: The accumulated context. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            List[str]: A list containing the response text and the updated context.
        """
        if context is None:
            context = self.context
            logging.info("Using internal context")

        inputs = []
        if text_or_inputs is not None:
            if isinstance(text_or_inputs, list):
                inputs = text_or_inputs
            else:
                inputs.append(text_or_inputs)
        if image is not None:
            inputs.append(Image(image))
        message = Message(role="user", content=inputs)
        response = self.backend.create_completion(
            message, context, model=self.model, **kwargs)

        logging.log(2, f"Response: {response}")
        context.append(Message(role="assistant", content=[Sample(response)]))

        return self.on_completion_response(response)

    def forget(self, observation=None, action=None, num_msgs: int = -1) -> list:
        """Removes messages from the context.

        Args:
            observation: The observation to use for looking up messages to forget.
            action: The action to use for looking up messages to forget.
            num_msgs: The number of messages to forget. Defaults to -1 (all).

        Returns:
            list: The list of targets removed from the context.
        """
        targets = self.rag_lookup(observation, action, num_msgs)
        for target in targets:
            self.context.remove(target)

    def store_context(self, input, response) -> None:
        """Stores the input and response in the context.

        Args:
            input: The input to store.
            response: The response to store.
        """
        self.context.append(Message(role="user", content=input))
        self.context.append(Message(role="assistant", content=response))
