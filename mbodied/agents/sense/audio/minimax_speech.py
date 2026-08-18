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

"""MiniMax speech provider for text to speech synthesis.

MiniMax exposes text to audio (T2A) endpoints for both the global and China
regions. This module implements the synchronous HTTP synthesis call used by
``AudioAgent.speak``, the asynchronous task API for longer inputs, and the
response parsing that decodes the hex encoded audio returned by the service.
"""

import os
from typing import Any, Literal

import httpx

MINIMAX_SPEECH_BASE_URLS = {
    "global_en": "https://api.minimax.io",
    "cn_zh": "https://api.minimaxi.com",
}
MINIMAX_SPEECH_REGION_ALIASES = {
    "global": "global_en",
    "cn": "cn_zh",
}
MINIMAX_T2A_PATH = "/v1/t2a_v2"
MINIMAX_T2A_ASYNC_CREATE_PATH = "/v1/t2a_async_v2"
MINIMAX_T2A_ASYNC_QUERY_PATH = "/v1/query/t2a_async_query_v2"
MINIMAX_T2A_WEBSOCKET_PATH = "/ws/v1/t2a_v2"

DEFAULT_MINIMAX_SPEECH_MODEL = "speech-2.8-hd"
MINIMAX_SPEECH_MODELS = (
    "speech-2.8-hd",
    "speech-2.8-turbo",
    "speech-2.6-hd",
    "speech-2.6-turbo",
    "speech-02-hd",
    "speech-02-turbo",
    "speech-01-hd",
    "speech-01-turbo",
)
MINIMAX_AUDIO_FORMATS = ("mp3", "wav", "flac", "pcm")

DEFAULT_MINIMAX_SPEECH_TIMEOUT = 60


class MiniMaxSpeechError(RuntimeError):
    """Raised when the MiniMax speech API reports an error or returns no audio."""


class MiniMaxSpeechBackend:
    """Backend for MiniMax text to audio synthesis.

    The backend targets one regional endpoint and one speech model, and exposes
    the synchronous T2A call, the asynchronous T2A task API, and the websocket
    endpoint of the configured region.

    Attributes:
        api_key: The API key for the MiniMax service.
        region: The normalized regional endpoint name.
        base_url: The regional base URL used for every request.
        model: The default speech model.
        timeout: The default request timeout in seconds.
        headers: The request headers, including bearer authorization.

    Usage:
    ```python
    speech = MiniMaxSpeechBackend(region="global_en")
    audio = speech.synthesize("How can I help you?")
    ```
    """

    def __init__(
        self,
        api_key: str | None = None,
        region: str = "global_en",
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_MINIMAX_SPEECH_TIMEOUT,
    ):
        """Initializes the backend for one region and one default speech model.

        Args:
            api_key: The API key for the MiniMax service. Defaults to the
                ``MINIMAX_API_KEY`` environment variable.
            region: The regional endpoint to use, either ``"global_en"`` or
                ``"cn_zh"``. The aliases ``"global"`` and ``"cn"`` are also
                accepted. Ignored when ``base_url`` is provided.
            base_url: An explicit base URL. Takes precedence over ``region``.
            model: The default speech model. Defaults to
                ``DEFAULT_MINIMAX_SPEECH_MODEL``. See ``MINIMAX_SPEECH_MODELS``
                for the models the service documents.
            timeout: The default request timeout in seconds.

        Raises:
            ValueError: If the region is unknown and no base URL is given.
        """
        normalized_region = MINIMAX_SPEECH_REGION_ALIASES.get(region, region)
        if base_url is None and normalized_region not in MINIMAX_SPEECH_BASE_URLS:
            raise ValueError(
                f"region must be one of {sorted(MINIMAX_SPEECH_BASE_URLS)} "
                f"or {sorted(MINIMAX_SPEECH_REGION_ALIASES)}, got {region!r}.",
            )

        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        self.region = normalized_region
        self.base_url = (base_url or MINIMAX_SPEECH_BASE_URLS[normalized_region]).rstrip("/")
        self.model = model or DEFAULT_MINIMAX_SPEECH_MODEL
        self.timeout = timeout
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    @property
    def websocket_url(self) -> str:
        """The T2A websocket endpoint of the configured region."""
        return self.base_url.replace("https://", "wss://", 1) + MINIMAX_T2A_WEBSOCKET_PATH

    def request_speech(
        self,
        text: str,
        model: str | None = None,
        stream: bool | None = None,
        language_boost: str | None = None,
        output_format: Literal["hex", "url"] | None = None,
        voice_setting: dict[str, Any] | None = None,
        pronunciation_dict: dict[str, Any] | None = None,
        audio_setting: dict[str, Any] | None = None,
        voice_modify: dict[str, Any] | None = None,
        subtitle_enable: bool | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Synthesizes speech with the synchronous T2A endpoint and returns the parsed response.

        Args:
            text: The text to synthesize.
            model: The speech model. Defaults to the model of this backend.
            stream: Whether the service should stream the audio chunks.
            language_boost: The language to bias the synthesis towards.
            output_format: Whether the audio is returned inline as ``"hex"`` or
                as a ``"url"``. The service defaults to hex.
            voice_setting: The voice options, such as the voice id, speed, volume and pitch.
            pronunciation_dict: The pronunciation replacements to apply to the text.
            audio_setting: The output audio options, such as sample rate, bitrate,
                channel and one of ``MINIMAX_AUDIO_FORMATS`` as format.
            voice_modify: The voice timbre modification options.
            subtitle_enable: Whether the service should also return subtitles.
            timeout: The request timeout in seconds. Defaults to the backend timeout.

        Returns:
            The parsed response body, holding the audio and status under ``data``.

        Raises:
            MiniMaxSpeechError: If the service reports a non zero status code.
            ValueError: If the requested audio format is not supported.
        """
        self._validate_audio_setting(audio_setting)
        options = {
            "stream": stream,
            "language_boost": language_boost,
            "output_format": output_format,
            "voice_setting": voice_setting,
            "pronunciation_dict": pronunciation_dict,
            "audio_setting": audio_setting,
            "voice_modify": voice_modify,
            "subtitle_enable": subtitle_enable,
        }
        return self._post(MINIMAX_T2A_PATH, self._payload(text, model, options), timeout)

    def synthesize(self, text: str, model: str | None = None, **kwargs) -> bytes:
        """Synthesizes speech and returns the decoded audio bytes.

        Args:
            text: The text to synthesize.
            model: The speech model. Defaults to the model of this backend.
            **kwargs: Additional request options accepted by ``request_speech``.

        Returns:
            The decoded audio bytes, ready to be written to a file.
        """
        return self.extract_audio(self.request_speech(text, model, **kwargs))

    def create_async_task(
        self,
        text: str,
        model: str | None = None,
        voice_setting: dict[str, Any] | None = None,
        audio_setting: dict[str, Any] | None = None,
        language_boost: str | None = None,
        pronunciation_dict: dict[str, Any] | None = None,
        voice_modify: dict[str, Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Creates an asynchronous T2A task, which suits inputs that are too long for one request.

        Args:
            text: The text to synthesize.
            model: The speech model. Defaults to the model of this backend.
            voice_setting: The voice options, such as the voice id, speed, volume and pitch.
            audio_setting: The output audio options, such as sample rate, bitrate,
                channel and one of ``MINIMAX_AUDIO_FORMATS`` as format.
            language_boost: The language to bias the synthesis towards.
            pronunciation_dict: The pronunciation replacements to apply to the text.
            voice_modify: The voice timbre modification options.
            timeout: The request timeout in seconds. Defaults to the backend timeout.

        Returns:
            The parsed response body, holding the identifier of the created task.

        Raises:
            ValueError: If the requested audio format is not supported.
        """
        self._validate_audio_setting(audio_setting)
        options = {
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
            "language_boost": language_boost,
            "pronunciation_dict": pronunciation_dict,
            "voice_modify": voice_modify,
        }
        return self._post(MINIMAX_T2A_ASYNC_CREATE_PATH, self._payload(text, model, options), timeout)

    def query_async_task(self, task_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Queries the state of an asynchronous T2A task.

        Args:
            task_id: The identifier returned by ``create_async_task``.
            timeout: The request timeout in seconds. Defaults to the backend timeout.

        Returns:
            The parsed response body, holding the state of the task.
        """
        return self._post(MINIMAX_T2A_ASYNC_QUERY_PATH, {"task_id": task_id}, timeout)

    @staticmethod
    def extract_audio(response: dict[str, Any]) -> bytes:
        """Decodes the hex encoded audio of a T2A response.

        Args:
            response: The parsed response body of a synthesis request.

        Returns:
            The decoded audio bytes.

        Raises:
            MiniMaxSpeechError: If the response carries no inline hex encoded audio.
        """
        data = response.get("data") or {}
        audio = data.get("audio")
        if not audio:
            raise MiniMaxSpeechError(f"The speech response carried no audio, status {data.get('status')!r}.")
        try:
            return bytes.fromhex(audio)
        except ValueError as e:
            raise MiniMaxSpeechError(
                "The speech response did not carry hex encoded audio. "
                "Use request_speech to read the response of a url output format.",
            ) from e

    def _payload(self, text: str, model: str | None, options: dict[str, Any]) -> dict[str, Any]:
        """Builds a request body from the required fields and the options that are set."""
        payload: dict[str, Any] = {"model": model or self.model, "text": text}
        payload.update({key: value for key, value in options.items() if value is not None})
        return payload

    def _post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Posts a request body to a regional endpoint and returns the parsed response."""
        with httpx.Client(trust_env=True) as client:
            response = client.post(
                self.base_url + path,
                headers=self.headers,
                json=payload,
                timeout=timeout or self.timeout,
            )
            response.raise_for_status()
            return self._parse_response(response.json())

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> dict[str, Any]:
        """Raises when the service reported an error and returns the response otherwise."""
        base_resp = response.get("base_resp") or {}
        status_code = base_resp.get("status_code") or 0
        if status_code:
            raise MiniMaxSpeechError(
                f"The speech request failed with status code {status_code}: {base_resp.get('status_msg', '')}",
            )
        return response

    @staticmethod
    def _validate_audio_setting(audio_setting: dict[str, Any] | None) -> None:
        """Raises when the audio setting requests an unsupported output format."""
        audio_format = (audio_setting or {}).get("format")
        if audio_format is not None and audio_format not in MINIMAX_AUDIO_FORMATS:
            raise ValueError(f"format must be one of {list(MINIMAX_AUDIO_FORMATS)}, got {audio_format!r}.")
