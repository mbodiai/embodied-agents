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

import pytest
from unittest.mock import MagicMock, mock_open
from mbodied.agents.sense.audio.audio_agent import AudioAgent


@pytest.fixture
def audio_agent(mocker):
    mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI")
    mocker.patch("mbodied.agents.sense.audio.audio_agent.pyaudio.PyAudio")
    mock_openai = mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI").return_value
    mock_openai.audio.transcriptions.create.return_value = MagicMock(text="test transcription")
    handler = AudioAgent(api_key="test-api-key")
    return handler


@pytest.mark.extras
def test_listen_type_mode(mocker, audio_agent):
    mocker.patch("builtins.input", side_effect=["exit"])
    result = audio_agent.listen(mode="type")
    assert result == "exit\n##\n"


@pytest.mark.extras
def test_listen_speak_mode(mocker, audio_agent):
    mocker.patch("builtins.input", side_effect=["", ""])
    mock_thread = mocker.patch("mbodied.agents.sense.audio.audio_agent.threading.Thread")
    mock_thread_instance = mock_thread.return_value
    mock_thread_instance.start.side_effect = lambda: setattr(audio_agent, "recording", False)

    # Use the mock_open provided by unittest.mock and apply it with pytest-mocker
    m_open = mock_open(read_data=b"audio data")
    mocker.patch("builtins.open", m_open)

    result = audio_agent.listen(mode="speak")

    assert result == "test transcription"
    mock_thread_instance.start.assert_called_once()
    mock_thread_instance.join.assert_called_once()


@pytest.mark.extras
def test_speak(mocker, audio_agent):
    mock_openai = mocker.patch("mbodied.agents.sense.audio.audio_agent.OpenAI").return_value
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = [b"chunk1", b"chunk2"]
    mock_openai.with_streaming_response.audio.speech.create.return_value = mock_response

    # Correctly use mock_open from mocker, and assign to a different variable name
    m_open = mocker.mock_open()
    mocker.patch("builtins.open", m_open, create=True)

    mock_thread = mocker.patch("mbodied.agents.sense.audio.audio_agent.threading.Thread")
    audio_agent.speak("Hello")
    mock_thread.assert_called_once()
    # Check if the file was correctly opened for writing
    m_open.assert_called_with(audio_agent.speak_filename, "wb")


@pytest.mark.extras
def test_play_audio(mocker, audio_agent):
    mock_playsound = mocker.patch("mbodied.agents.sense.audio.audio_agent.playsound.playsound")
    mocker.patch("mbodied.agents.sense.audio.audio_agent.os.path.exists", return_value=True)
    mock_remove = mocker.patch("mbodied.agents.sense.audio.audio_agent.os.remove")

    audio_agent.play_audio("test.mp3")

    mock_playsound.assert_called_once_with("test.mp3")
    mock_remove.assert_called_once_with("test.mp3")


if __name__ == "__main__":
    pytest.main([__file__])
