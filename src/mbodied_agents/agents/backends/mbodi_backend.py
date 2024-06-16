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
from typing import Any, List

from mbodied_agents.base.backend import Backend
from mbodied_agents.base.serializer import Serializer
from mbodied_agents.types.message import Message
from mbodied_agents.types.sense.vision import Image
import requests
import logging
import json


class MbodiSerializer(Serializer):
    """Serializer for Mbodi-specific data formats."""

    @classmethod
    def serialize_image(cls, image: Image) -> dict[str, Any]:
        """Serializes an image to the Mbodi format.

        Args:
            image: The image to be serialized.

        Returns:
            A dictionary representing the serialized image.
        """
        return {
            "type": "image_url",
            "image_url": {
                "url": image.url,
            },
        }

    @classmethod
    def serialize_text(cls, text: str) -> dict[str, Any]:
        """Serializes a text string to the Mbodi format.

        Args:
            text: The text to be serialized.

        Returns:
            A dictionary representing the serialized text.
        """
        return {"type": "text", "text": text}


class MbodiBackend(Backend):
    """Backend for interacting with Mbodi's API."""

    INITIAL_CONTEXT = [
        Message(role="user",
                content='You are a robot with advanced spatial reasoning.'),
        Message(role="assistant", content='Understood.'),
    ]
    DEFAULT_MODEL = "mbodi"
    # Note: This is a placeholder URL and will be updated when the API is available.
    API_URL = "http://80.188.223.202:11102/process"

    def __init__(self, url: str, **kwargs):
        """Initializes the MbodiBackend with Mbodi API url.

        Args:
            url: Mbodi API url.
            **kwargs: Additional keyword arguments.
        """
        self.url = url
        self.serializer = MbodiSerializer()

    def _create_completion(self, messages: List[Message], model: str = "phi3v-spatial", stream: bool = False, **kwargs) -> str:
        """Creates a completion for the given messages using the Mbodi API.

        Args:
            messages: A list of messages to be sent to the completion API.
            model: The model to be used for the completion.
            stream: Whether to stream the response. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The content of the completion response.
        """
        serialized_messages = [
            self.serializer.serialize_msg(msg) for msg in messages]
        payload = {"messages": serialized_messages}
        response = requests.post(self.url, json=payload, headers={
                                 'Content-Type': 'application/json'})
        if response.status_code != 200:
            logging.warning(
                f"Failed to get response from Mbodi backend. Status code: {response.status_code}")

        return response.json()["response"]
