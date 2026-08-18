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
import os
import platform
import threading
import wave
from pathlib import Path

try:
    import playsound
    import pyaudio
except ImportError:
    logging.warning("playsound or pyaudio is not installed. Please run `pip install pyaudio playsound` to install.")

from openai import OpenAI
from typing_extensions import Literal

from mbodied.agents import Agent
from mbodied.agents.sense.audio.minimax_speech import MiniMaxSpeechBackend

TTS_PROVIDERS = {"minimax": MiniMaxSpeechBackend}


class AudioAgent(Agent):
    """Handles audio recording, playback, and speech-to-text transcription.

    This module uses OpenAI's API to transcribe audio input and synthesize speech.
    Speech synthesis can also be delegated to another provider listed in `TTS_PROVIDERS`.
    Set Environment Variable NO_AUDIO=1 to disable audio recording and playback.
    It will then take input from the terminal.

    Usage:
    ```python
    audio_agent = AudioAgent(api_key="your-openai-api-key", use_pyaudio=False)
    audio_agent.speak("How can I help you?")
    message = audio_agent.listen()
    ```
    """

    mode = Literal["speak", "type", "speak_or_type"]

    def __init__(
        self,
        listen_filename: str = "tmp_listen.wav",
        tmp_speak_filename: str = "tmp_speak.mp3",
        use_pyaudio: bool = True,
        api_key: str = None,
        run_local: bool = False,
        tts_provider: str = None,
        tts_kwargs: dict = None,
    ):
        """Initializes the AudioAgent with specified parameters.

        Args:
            listen_filename: The filename for storing recorded audio.
            tmp_speak_filename: The filename for storing synthesized speech.
            use_pyaudio: Whether to use PyAudio for playback. Prefer setting to False for Mac.
            api_key: The API key for OpenAI.
            run_local: Whether to run the whisper model locally instead of using OpenAI.
            tts_provider: The provider to synthesize speech with, one of `TTS_PROVIDERS`.
                Defaults to None, which keeps the default speech path.
            tts_kwargs: Additional arguments to pass to the speech provider, such as
                its api_key, region or model.
        """
        self.recording = False
        self.record_lock = threading.Lock()
        self.listen_filename = listen_filename
        self.speak_filename = tmp_speak_filename
        self.use_pyaudio = use_pyaudio
        self.speech_backend = self.init_speech_backend(tts_provider, tts_kwargs)
        if os.getenv("NO_AUDIO"):
            return
        self.run_local = False
        if run_local:
            try:
                import whisper
            except ImportError:
                logging.warning("whisper is not installed. Please run `pip install openai-whisper` to install.")
            self.run_local = True
            self.model = whisper.load_model("base")
        else:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def init_speech_backend(tts_provider: str | None, tts_kwargs: dict | None) -> MiniMaxSpeechBackend | None:
        """Initializes the speech provider to synthesize with, if one was requested.

        Args:
            tts_provider: The provider name, one of `TTS_PROVIDERS`.
            tts_kwargs: Additional arguments to pass to the provider.

        Returns:
            The speech backend, or None to use the default speech path.

        Raises:
            ValueError: If the provider name is not recognized.
        """
        if tts_provider is None:
            return None
        if tts_provider not in TTS_PROVIDERS:
            raise ValueError(f"tts_provider must be one of {sorted(TTS_PROVIDERS)}, got {tts_provider!r}.")
        return TTS_PROVIDERS[tts_provider](**(tts_kwargs or {}))

    def act(self, *args, **kwargs):
        return self.listen(*args, **kwargs)

    def listen(self, keep_audio: bool = False, mode: str = "speak") -> str:
        """Listens for audio input and transcribes it using OpenAI's API.

        Args:
            keep_audio: Whether to keep the recorded audio file.
            mode: The mode of input (speak, type, speak_or_type).

        Returns:
            The transcribed text from the audio input.
        """
        logging.debug(f"Listening with mode: {mode}")
        if os.getenv("NO_AUDIO") or mode in ["type", "speak_or_type"]:
            user_input = input("Please type your input [Type 'exit' to exit]: ") + "\n##\n"
            if os.getenv("NO_AUDIO") or mode == "type":
                return user_input
        else:
            user_input = ""

        typed_input = user_input
        thread = threading.Thread(target=self.record_audio)
        user_input = input("Press ENTER to speak [Type 'exit' to exit]")
        if user_input.lower() == "exit":
            exit()

        with self.record_lock:
            self.recording = True
        thread.start()
        input("Press ENTER to stop recording")
        with self.record_lock:
            self.recording = False
        thread.join()
        transcription = None
        try:
            with open(self.listen_filename, "rb") as audio_file:
                if self.run_local:
                    transcription = self.model.transcribe(self.listen_filename)["text"]
                else:
                    transcription = self.client.audio.transcriptions.create(model="whisper-1", file=audio_file).text
                return typed_input + transcription
        except Exception as e:
            logging.error(f"Failed to read or transcribe audio file: {e}")
            return ""
        finally:
            if not keep_audio and os.path.exists(self.listen_filename):
                os.remove(self.listen_filename)
            return typed_input + transcription if transcription else ""

    def record_audio(self) -> None:
        """Records audio from the microphone and saves it to a file."""
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 44100
        p = pyaudio.PyAudio()
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        frames = []

        try:
            while self.recording:
                data = stream.read(chunk)
                frames.append(data)
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        try:
            with wave.open(self.listen_filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(sample_format))
                wf.setframerate(fs)
                wf.writeframes(b"".join(frames))
        except Exception as e:
            logging.error(f"Failed to save audio: {e}")

    def speak(self, message: str, voice: str = "onyx", api_key: str = None, **kwargs) -> None:
        """Synthesizes speech from text and plays it back.

        Args:
            message: The text message to synthesize.
            voice: The voice model to use for synthesis. Only used by the default speech path.
            api_key: The API key for OpenAI.
            **kwargs: Additional synthesis options for the configured speech provider,
                such as its voice_setting or audio_setting.
        """
        if os.environ.get("NO_AUDIO"):
            return
        try:
            if self.speech_backend is not None:
                Path(self.speak_filename).write_bytes(self.speech_backend.synthesize(message, **kwargs))
            else:
                client = self.client or OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
                with (
                    client.with_streaming_response.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=message,
                    ) as response,
                    open(self.speak_filename, "wb") as out_file,
                ):
                    for chunk in response.iter_bytes():
                        out_file.write(chunk)
        except Exception as e:
            logging.error(f"Failed to create or save speech: {e}")
            return

        self.playback_thread = threading.Thread(target=self.play_audio, args=(self.speak_filename,))
        self.playback_thread.start()

    def play_audio(self, filename: str) -> None:
        """Plays an audio file.

        Args:
            filename: The filename of the audio file to play.
        """
        try:
            if platform.system() == "Darwin" and not self.use_pyaudio:
                # Only works on mac.
                os.system("afplay " + filename)
            else:
                playsound.playsound(filename)
        except Exception as e:
            logging.error(f"Error playing audio file {filename}: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
