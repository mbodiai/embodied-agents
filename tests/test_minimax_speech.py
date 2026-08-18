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

import httpx
import pytest
from mbodied.agents.sense.audio.audio_agent import AudioAgent
from mbodied.agents.sense.audio.minimax_speech import (
    DEFAULT_MINIMAX_SPEECH_MODEL,
    MINIMAX_SPEECH_BASE_URLS,
    MINIMAX_SPEECH_MODELS,
    MINIMAX_T2A_ASYNC_CREATE_PATH,
    MINIMAX_T2A_ASYNC_QUERY_PATH,
    MINIMAX_T2A_PATH,
    MiniMaxSpeechBackend,
    MiniMaxSpeechError,
)

AUDIO_BYTES = b"minimax audio"
AUDIO_HEX = AUDIO_BYTES.hex()


class FakeResponse:
    """Minimal stand in for an httpx response."""

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeClient:
    """Records the request the backend sends and returns a canned payload."""

    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, url, headers=None, json=None, timeout=None):
        self.requests.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(self.payload)


@pytest.fixture
def fake_client(mocker):
    def install(payload):
        client = FakeClient(payload)
        mocker.patch("mbodied.agents.sense.audio.minimax_speech.httpx.Client", return_value=client)
        return client

    return install


def speech_payload(audio=AUDIO_HEX, status=2, status_code=0):
    return {"data": {"audio": audio, "status": status}, "base_resp": {"status_code": status_code, "status_msg": "bad"}}


def test_default_region_model_and_headers():
    backend = MiniMaxSpeechBackend(api_key="test-api-key")

    assert backend.region == "global_en"
    assert backend.base_url == MINIMAX_SPEECH_BASE_URLS["global_en"]
    assert backend.model == DEFAULT_MINIMAX_SPEECH_MODEL
    assert backend.model in MINIMAX_SPEECH_MODELS
    assert backend.headers["Authorization"] == "Bearer test-api-key"


def test_regional_endpoints_and_aliases():
    assert MiniMaxSpeechBackend(region="cn_zh").base_url == MINIMAX_SPEECH_BASE_URLS["cn_zh"]
    assert MiniMaxSpeechBackend(region="cn").region == "cn_zh"
    assert MiniMaxSpeechBackend(region="global").region == "global_en"
    assert MiniMaxSpeechBackend(base_url="https://example.test/").base_url == "https://example.test"


def test_unknown_region_is_rejected():
    with pytest.raises(ValueError):
        MiniMaxSpeechBackend(region="mars")


def test_websocket_url_uses_region():
    assert MiniMaxSpeechBackend(region="cn_zh").websocket_url.startswith("wss://")
    assert MiniMaxSpeechBackend().websocket_url.endswith("/ws/v1/t2a_v2")


def test_synthesize_posts_required_fields_and_decodes_audio(fake_client):
    client = fake_client(speech_payload())
    backend = MiniMaxSpeechBackend(api_key="test-api-key")

    audio = backend.synthesize("How can I help you?")

    assert audio == AUDIO_BYTES
    request = client.requests[0]
    assert request["url"] == MINIMAX_SPEECH_BASE_URLS["global_en"] + MINIMAX_T2A_PATH
    assert request["json"] == {"model": DEFAULT_MINIMAX_SPEECH_MODEL, "text": "How can I help you?"}
    assert request["timeout"] == backend.timeout


def test_synthesize_forwards_request_options(fake_client):
    client = fake_client(speech_payload())
    backend = MiniMaxSpeechBackend(api_key="test-api-key", model="speech-2.8-turbo")

    backend.synthesize(
        "Hello",
        language_boost="English",
        output_format="hex",
        voice_setting={"voice_id": "test-voice", "speed": 1.0},
        audio_setting={"format": "wav", "sample_rate": 32000},
        pronunciation_dict={"tone": ["a/b"]},
        voice_modify={"pitch": 1},
        subtitle_enable=True,
        stream=False,
        timeout=5,
    )

    request = client.requests[0]
    assert request["json"] == {
        "model": "speech-2.8-turbo",
        "text": "Hello",
        "stream": False,
        "language_boost": "English",
        "output_format": "hex",
        "voice_setting": {"voice_id": "test-voice", "speed": 1.0},
        "pronunciation_dict": {"tone": ["a/b"]},
        "audio_setting": {"format": "wav", "sample_rate": 32000},
        "voice_modify": {"pitch": 1},
        "subtitle_enable": True,
    }
    assert request["timeout"] == 5


def test_unset_options_are_omitted(fake_client):
    client = fake_client(speech_payload())

    MiniMaxSpeechBackend().request_speech("Hello", voice_setting={"voice_id": "test-voice"})

    assert set(client.requests[0]["json"]) == {"model", "text", "voice_setting"}


def test_unsupported_audio_format_is_rejected():
    with pytest.raises(ValueError):
        MiniMaxSpeechBackend().synthesize("Hello", audio_setting={"format": "ogg"})


def test_error_status_code_raises(fake_client):
    fake_client(speech_payload(status_code=1004))

    with pytest.raises(MiniMaxSpeechError):
        MiniMaxSpeechBackend().synthesize("Hello")


def test_missing_audio_raises(fake_client):
    fake_client(speech_payload(audio=None))

    with pytest.raises(MiniMaxSpeechError):
        MiniMaxSpeechBackend().synthesize("Hello")


def test_url_output_format_is_reported_for_synthesize(fake_client):
    fake_client(speech_payload(audio="https://example.test/audio.mp3"))
    backend = MiniMaxSpeechBackend()

    with pytest.raises(MiniMaxSpeechError):
        backend.synthesize("Hello", output_format="url")

    assert backend.request_speech("Hello", output_format="url")["data"]["audio"].startswith("https://")


def test_async_task_create_and_query(fake_client):
    client = fake_client({"task_id": "task-1", "base_resp": {"status_code": 0}})
    backend = MiniMaxSpeechBackend()

    created = backend.create_async_task("Hello", audio_setting={"format": "mp3"}, language_boost="English")
    queried = backend.query_async_task("task-1")

    assert created["task_id"] == "task-1"
    assert queried["task_id"] == "task-1"
    assert client.requests[0]["url"].endswith(MINIMAX_T2A_ASYNC_CREATE_PATH)
    assert client.requests[0]["json"] == {
        "model": DEFAULT_MINIMAX_SPEECH_MODEL,
        "text": "Hello",
        "audio_setting": {"format": "mp3"},
        "language_boost": "English",
    }
    assert client.requests[1]["url"].endswith(MINIMAX_T2A_ASYNC_QUERY_PATH)
    assert client.requests[1]["json"] == {"task_id": "task-1"}


def test_http_errors_propagate(mocker):
    class FailingResponse(FakeResponse):
        def raise_for_status(self):
            raise httpx.HTTPStatusError("unauthorized", request=None, response=None)

    class FailingClient(FakeClient):
        def post(self, url, headers=None, json=None, timeout=None):
            return FailingResponse(self.payload)

    mocker.patch(
        "mbodied.agents.sense.audio.minimax_speech.httpx.Client",
        return_value=FailingClient(speech_payload()),
    )

    with pytest.raises(httpx.HTTPStatusError):
        MiniMaxSpeechBackend().synthesize("Hello")


def test_audio_agent_speaks_with_provider(mocker, monkeypatch, tmp_path, fake_client):
    monkeypatch.delenv("NO_AUDIO", raising=False)
    client = fake_client(speech_payload())
    mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI")
    thread = mocker.patch("mbodied.agents.sense.audio.audio_agent.threading.Thread")
    speak_file = tmp_path / "tmp_speak.mp3"

    agent = AudioAgent(
        tmp_speak_filename=str(speak_file),
        tts_provider="minimax",
        tts_kwargs={"api_key": "test-api-key", "region": "cn_zh"},
    )
    agent.speak("Hello", voice_setting={"voice_id": "test-voice"})

    assert isinstance(agent.speech_backend, MiniMaxSpeechBackend)
    assert speak_file.read_bytes() == AUDIO_BYTES
    assert client.requests[0]["url"] == MINIMAX_SPEECH_BASE_URLS["cn_zh"] + MINIMAX_T2A_PATH
    assert client.requests[0]["json"]["voice_setting"] == {"voice_id": "test-voice"}
    thread.assert_called_once()


def test_audio_agent_defaults_to_existing_speech_path(mocker, monkeypatch):
    monkeypatch.delenv("NO_AUDIO", raising=False)
    mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI")

    assert AudioAgent(api_key="test-api-key").speech_backend is None


def test_audio_agent_rejects_unknown_provider(mocker):
    mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI")

    with pytest.raises(ValueError):
        AudioAgent(tts_provider="unknown")


if __name__ == "__main__":
    pytest.main([__file__])
