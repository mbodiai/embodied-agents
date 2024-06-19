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


import time
from inspect import signature
from pathlib import Path
from typing import Literal, TypeAlias

from gradio_client import Client as GradioClient
from gradio_client.client import Job

from mbodied_agents.agents.backends import AnthropicBackend, OpenAIBackend
from mbodied_agents.base.sample import Sample
from mbodied_agents.data.recording import Recorder


def create_observation_from_args(observation_space, function, args, kwargs) -> dict:
    """Create an observation from the arguments of a function."""
    param_names = list(signature(function).parameters.keys())

    # Create the observation from the arguments
    params = {**kwargs}
    for arg, val in zip(param_names, args, strict=False):
        params[arg] = val
    if observation_space is not None:
        observation = observation_space.sample()
        return {k: v for k, v in params.items() if k in observation}

    return {k: v for k, v in params.items() if v is not None and k not in ["self", "kwargs"]}


class Agent:
    """Abstract base class for agents.

    This class provides a template for creating agents that can
    optionally record their actions and observations.

    Attributes:
        recorder (Recorder): The recorder to record observations and actions.
        kwargs (dict): Additional arguments to pass to the recorder.
    """

    REMOTE_ACTOR_MAP = {
        "openai": OpenAIBackend,
        "anthropic": AnthropicBackend,
        "gradio": GradioClient,
    }

    def __init__(
        self,
        recorder: Literal["omit", "default"] | str = "omit",
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

        self.recorder = None
        if recorder == "default":
            self.recorder = Recorder("base_agent", out_dir="outs", **recorder_kwargs)
        elif recorder != "omit":
            self.recorder = Recorder(recorder, out_dir="outs", **recorder_kwargs)

        model_kwargs = model_kwargs or {}
        self.remote_actor = None
        if Path(model_src).exists():
            self.load_model(model_src, **model_kwargs)
        elif local_only:
            raise ValueError("'local_only' requested yet model source not found.")
        else:
            actor_class = self.REMOTE_ACTOR_MAP.get(model_src, GradioClient)
            # If the model source is a gradio endpoint, pass it as the src.
            # Note that isinstance() does not work here. Thus, we use the class name Client for GradioClient.
            if actor_class.__name__ == "Client":
                model_kwargs.update({"src": model_src})
            if api_key is not None:
                model_kwargs["api_key"] = api_key
            self.remote_actor = actor_class(**model_kwargs)

    def act(self, **kwargs) -> Sample:
        """Act based on the observation.

        Record the observation and action.

        Args:
            *args: Observation fields.
            **kwargs: Observation fields as keyword arguments.

        Returns:
            Sample: The action sample created by the agent.
        """
        raise NotImplementedError

    def load_model(self, model: str) -> None:
        """Load a model from a file or path. Required if the model is a weights path.

        Args:
            model: The path to the model file.
        """
        pass

    def _act(self, *args, **kwargs) -> Sample:
        """Act based on the observation and record the action, if applicable.

        Args:
            *args: Additional arguments to customize the action.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
        action = self.remote_actor.act(*args, **kwargs)
        if self.recorder is not None:
            observation = create_observation_from_args(self.recorder.observation_space, self._act, args, kwargs)
            self.recorder.record(observation=observation, action=action)
        return action

    def remote_act(self, observation_dict: dict, endpoint: str, result_callbacks=None, blocking=False, **kwargs) -> Job:
        """Act remotely and asynchronously using a gradio client.

        Args:
            observation: The current observation based on which the agent acts.
            endpoint: The endpoint to send the action to.
            remote_actor: The remote actor to send the action to.
            result_callbacks: Callbacks to be executed on the result.
            blocking: Whether to block until the job is done.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
        if not hasattr(self, "remote_actor"):
            raise AttributeError("No remote actor defined.")
        job = self.remote_actor.submit(observation_dict, api_name=endpoint, result_callbacks=result_callbacks, **kwargs)
        tic = time.time()
        if blocking:
            while not job.done() and time.time() - tic < 10:
                pass
            if self.recorder is not None:
                self.recorder.record(observation=observation_dict, action=job.result)
        return job
