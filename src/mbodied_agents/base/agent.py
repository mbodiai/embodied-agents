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

from gradio_client import Client
from gradio_client.client import Job

from mbodied_agents.base.sample import Sample
from mbodied_agents.data.recording import Recorder


class Agent:
    """Abstract base class for agents.

    This class provides a template for creating agents that can 
    optionally record their actions and observations.

    Attributes:
        recorder (Recorder): The recorder to record observations and actions.
        kwargs (dict): Additional arguments to pass to the recorder.
    """

    def __init__(self, recorder="omit",  remote_server_name: str | None = None, **kwargs):
        """Initialize the agent.

        Args:
            recorder (str): The recorder to record the observations and actions.
            remote_server_name (str): The name of the remote server to connect to.
            **kwargs: Additional arguments to pass to the recorder.
        """
        if recorder is None or recorder == 'omit':
            self.recorder = None
        elif recorder == 'default':
            self.recorder = Recorder('base_agent', out_dir='outs', **kwargs)
        else:
            self.recorder = Recorder(recorder, out_dir='outs', **kwargs)
        self.kwargs = kwargs

        if remote_server_name is not None:
            self.remote_actor = Client(remote_server_name, **kwargs)

    def __call__(self, observation, verify=False, **kwargs) -> Sample:
        """Act based on the observation.

        Record the observation and action.

        Args:
            observation: The current observation based on which the agent acts.
            verify (bool): Whether to verify the action before sending.
            **kwargs: Additional arguments to pass to the `act` method.

        Returns:
            Sample: The action sample created by the agent.
        """
        action = self.act(observation, **kwargs)
        if self.recorder is not None:
            self.recorder.record(observation, action)
        if verify:
            is_verified = input("Cancel Action? (y/n)")
        if not verify or "y" not in is_verified:
            self.recorder.record(observation, action)
        return action

    def act(self, observation, **kwargs) -> Sample:
        """Abstract method to define the action based on the observation.

        Args:
            observation: The current observation based on which the agent acts.
            **kwargs: Additional arguments to customize the action.

        Returns:
            Sample: The action sample created by the agent.
        """
        raise NotImplementedError

    def remote_act(self, observation, endpoint, result_callbacks=None, blocking=False, **kwargs) -> Job:
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
        if not hasattr(self, 'remote_actor'):
            raise AttributeError("No remote actor defined.")
        job = self.remote_actor.submit(
            observation, api_name=endpoint, result_callbacks=result_callbacks, **kwargs)
        tic = time.time()
        if blocking:
            while not job.done() and time.time()-tic < 10:
                pass
        return job
