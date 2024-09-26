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

import logging
from typing import TYPE_CHECKING, Any

from mbodied.utils.import_utils import smart_import

if TYPE_CHECKING:
    try:
        from gradio.job import Job
        from gradio_client import Client
    except ImportError:
        Client = Any
        Job = Any
        gradio = Any
else:
    try:
        from gradio_client import Client
    except Exception:
        
        logging.info("To use this backend, install gradio-client with `pip install gradio-client`")
        

from mbodied.agents.backends.backend import Backend


class GradioBackend(Backend):
    """Gradio backend that handles connections to gradio servers."""

    def __init__(
        self,
        endpoint: str = None,
        **kwargs,
    ) -> None:
        """Initializes the GradioBackend.

        Args:
            endpoint: The url of the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio client.
        """
        smart_import("gradio_client")
        self.endpoint = endpoint
        self.client = Client(src=endpoint, **kwargs)


    def predict(self, *args, **kwargs) -> str:
        """Forward queries to the gradio api endpoint `predict`.

        Args:
            *args: The arguments to pass to the gradio server.
            **kwargs: The keywrod arguments to pass to the gradio server.
        """
        return self.client.predict(*args, **kwargs)

    def submit(self, *args, api_name="/predict", result_callbacks=None, **kwargs) -> "Job":
        """Submit queries asynchronously without need of asyncio.

        Args:
            *args: The arguments to pass to the gradio server.
            api_name: The name of the api endpoint to submit the job.
            result_callbacks: The callbacks to apply to the result.
            **kwargs: The keywrod arguments to pass to the gradio server.

        Returns:
            Job: Gradio job object.
        """
        return self.client.submit(api_name=api_name, result_callbacks=result_callbacks, *args, **kwargs)
