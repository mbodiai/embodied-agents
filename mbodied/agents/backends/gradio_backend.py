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

from gradio_client import Client
from gradio_client.client import Job


class GradioBackend:
    """Gradio backend that handles connections to gradio servers."""

    def __init__(
        self,
        remote_server: str = None,
        **kwargs,
    ) -> None:
        self.remote_server = remote_server
        self.client = Client(src=remote_server, **kwargs)

    def act(self, *args, **kwargs) -> str:
        """Forward queries to the gradio api endpoint `predict`.

        Args:
            *args: The arguments to pass to the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio server.
        """
        return self.client.predict(*args, **kwargs)

    def submit(self, *args, api_name="/predict", result_callbacks=None, **kwargs) -> Job:
        """Asynchronous submit queries to the gradio api endpoint.

        Args:
            *args: The arguments to pass to the gradio server.
            api_name: The name of the api endpoint to submit the job.
            result_callbacks: The callbacks to apply to the result.
            **kwargs: The keywrod arguments to pass to the gradio server.

        Returns:
            Job: Gradio job object.
        """
        return self.client.submit(api_name=api_name, result_callbacks=result_callbacks, *args, **kwargs)
