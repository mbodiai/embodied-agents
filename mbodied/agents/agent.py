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


import asyncio
from inspect import signature
import logging
from pathlib import Path
from typing import Literal

from mbodied.agents.backends import AnthropicBackend, GradioBackend, OpenAIBackend, HttpxBackend, OllamaBackend
from mbodied.types.sample import Sample
from mbodied.data.recording import Recorder
from huggingface_hub import repo_exists

Backend = AnthropicBackend | GradioBackend | OpenAIBackend | HttpxBackend | OllamaBackend

class Agent:
    """Abstract base class for agents.

    This class provides a template for creating agents that can
    optionally record their actions and observations.

    Attributes:
        recorder (Recorder): The recorder to record observations and actions.
        actor (Union[OpenAIBackend, AnthropicBackend, GradioClient]): The remote actor to interact with.
        kwargs (dict): Additional arguments to pass to the recorder.
    """
    ACTOR_MAP = {
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "gradio": GradioBackend,
        "ollama": OllamaBackend,
    }

    @staticmethod
    def init_backend(model_src: str, model_kwargs: dict, api_key: str) -> type:
        """Initialize the backend based on the model source.

        Args:
            model_src: The model source to use.
            model_kwargs: The additional arguments to pass to the model.
            api_key: The API key to use for the remote actor.

        Returns:
            type: The backend class to use.
        """
        if model_src in Agent.ACTOR_MAP:
            return Agent.ACTOR_MAP[model_src](**model_kwargs, api_key=api_key)
        return Agent.handle_default(model_src, model_kwargs)


    @staticmethod
    def handle_default(model_src: str, model_kwargs: dict) -> None:
        """Default to gradio then httpx backend if the model source is not recognized.

        Args:
            model_src: The model source to use.
            model_kwargs: The additional arguments to pass to the model.
        """
        if repo_exists(model_src) or repo_exists(model_src.split("https://huggingface.co/")[1]):
            try:
                backend = GradioBackend(model_src=model_src, **model_kwargs)
                return backend
            except Exception as e:
                logging.warning(f"Tried loading model from {model_src}: {e}. Using httpx backend.")
           
        try:
            backend = HttpxBackend(model_src=model_src, **model_kwargs)
            return backend
        except Exception as e:
            raise ValueError(f"Could not load model from {model_src}. Ensure the repo exists on HuggingFace or a valid Http endpoint is provided.")

    def __init__(
        self,
        recorder: Literal["omit", "auto"] | str = "omit",
        recorder_kwargs=None,
        api_key: str = None,
        model_src=None,
        model_kwargs=None,
        local_only: bool = False,
    ):
        """Initialize the agent, optionally setting up a recorder, remote actor, or loading a local model.

        Args:
            recorder: The recorder config or name to use for recording observations and actions.
            recorder_kwargs: Additional arguments to pass to the recorder.
            api_key: The API key to use for the remote actor (if applicable).
            model_src: The model or inference client or weights path to setup and preload if applicable.
                       You can pass in for example, "openai", "anthropic", "gradio", or a gradio endpoint,
                       or a path to a weights file.
            model_kwargs: Additional arguments to pass to the remote actor.
        """
        if model_src is None:
            raise ValueError("Model source must be provided.")
        if not isinstance(model_src, str):
            raise ValueError("Model source must be a string.")

        self.recorder = None
        recorder_kwargs = recorder_kwargs or {}
        if recorder == "auto":
            self.recorder = Recorder("base_agent", out_dir="outs", **recorder_kwargs)
        elif recorder != "omit":
            self.recorder = Recorder(recorder, out_dir="outs", **recorder_kwargs)

        model_kwargs = model_kwargs or {}
        self.actor = None
        if Path(model_src).exists():
            self.load_model(model_src, **model_kwargs)
        elif local_only:
            raise ValueError("'local_only' requested yet model source not found.")
        else:
            self.actor: Backend = self.init_backend(model_src, model_kwargs, api_key)

    def load_model(self, model: str) -> None:
        """Load a model from a file or path. Required if the model is a weights path.

        Args:
            model: The path to the model file.
        """
        pass

    def act(self, *args, **kwargs) -> Sample:
        """Act based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.act() correctly to perform the actions.
        """
        raise NotImplementedError("Subclass should implement this method.")

    async def async_act(self, *args, **kwargs) -> Sample:
        """Act asynchronously based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.async_act() correctly to perform the actions.
        """
        return await asyncio.to_thread(self.act, *args, **kwargs)

    def act_and_record(self, *args, **kwargs) -> Sample:
        """Peform action based on the observation and record the action, if applicable.

        Args:
            *args: Additional arguments to customize the action.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
        action = self.act(*args, **kwargs)
        if self.recorder is not None:
            observation = self.create_observation_from_args(
                self.recorder.observation_space, self.act_and_record, args, kwargs
            )
            self.recorder.record(observation=observation, action=action)
        return action

    async def async_act_and_record(self, *args, **kwargs) -> Sample:
        """Act asynchronously based on the observation.

        Subclass should implement this method.

        For remote actors, this method should call actor.async_act() correctly to perform the actions.
        """
        return await asyncio.to_thread(self.act_and_record, *args, **kwargs)

    @staticmethod
    def create_observation_from_args(observation_space, function, args, kwargs) -> dict:
        """Helper method to create an observation from the arguments of a function."""
        param_names = list(signature(function).parameters.keys())

        # Create the observation from the arguments
        params = {**kwargs}
        for arg, val in zip(param_names, args, strict=False):
            params[arg] = val
        if observation_space is not None:
            observation = observation_space.sample()
            return {k: v for k, v in params.items() if k in observation}

        return {k: v for k, v in params.items() if v is not None and k not in ["self", "kwargs"]}
