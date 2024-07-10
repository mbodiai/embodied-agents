import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from queue import Queue
from typing import Any, List, Optional
import asyncio

from mbodied.data.recording import Recorder
from mbodied.hardware.interface import HardwareInterface
from mbodied.types.sample import Sample
from mbodied.types.sense.vision import Image


class RecordingHardwareInterface(HardwareInterface, ABC):
    """Abstract base class for recording hardware interfaces with pose and image data.

    This class provides a framework for recording the state of a hardware interface, including
    its pose and captured images, at a specified frequency. It leverages a queue and a worker
    thread to handle the recording asynchronously, ensuring that the main operations of the
    hardware are not blocked.

    Subclass must implement the `do`, `capture`, and `get_pose` methods.
    Subclass should call the `do_and_record` method to execute the main operation and record.

    Note that we are recording the absolute pose of the hardware, not the relative motion.
    """

    def __init__(self, record_frequency: int = 2, **kwargs: Any):
        """Initializes the RecordingHardwareInterface.

        This constructor sets up the recording mechanism, including the recorder instance,
        recording frequency, and the asynchronous processing queue and worker thread. It also
        initializes attributes to track the last recorded pose and the current instruction.

        Args:
            record_frequency: Frequency at which to record pose and image data (in Hz).
            **kwargs: Additional arguments passed to the Recorder class initializer.
        """
        self.recorder = Recorder(**kwargs)
        self.recording = False
        self.record_frequency = record_frequency
        self.recording_queue = Queue()
        self.worker_thread = threading.Thread(target=self._process_queue)
        self.worker_thread.daemon = True
        self.worker_thread.start()
        self.last_recorded_pose = None
        self.current_instruction = ""

    @abstractmethod
    def do(self, *args: Any, **kwargs: Any) -> None:
        """Executes the main operation of the hardware."""
        raise NotImplementedError

    @abstractmethod
    def capture(self, *args: Any, **kwargs: Any) -> Image:
        """Captures an image using the hardware's vision system."""
        raise NotImplementedError

    @abstractmethod
    def get_pose(self) -> Sample:
        """Gets the current pose of the hardware."""
        raise NotImplementedError

    def reset_recorder(self) -> None:
        """Reset the recorder."""
        while self.recording:
            time.sleep(0.1)
        self.recorder.reset()

    def do_and_record(self, instruction: str, *args: Any, **kwargs: Any) -> None:
        """Executes the main operation and records pose and image with the instruction.

        Args:
            instruction: The instruction to be recorded along with pose and image.
            *args: Additional arguments to pass to do method.
            **kwargs: Additional keyword arguments to pass to the do method.
        """
        self.current_instruction = instruction
        self.start_recording()
        try:
            self.do(*args, **kwargs)
        finally:
            self.stop_recording()
            self.record_final_state()

    async def async_do_and_record(self, instruction: str, *args: Any, **kwargs: Any) -> None:
        """Asynchronously executes the main operation and records pose and image with the instruction.

        Args:
            instruction: The instruction to be recorded along with pose and image.
            *args: Additional arguments to pass to do method.
            **kwargs: Additional keyword arguments to pass to the do method.
        """
        return await asyncio.to_thread(self.do_and_record, instruction, *args, **kwargs)

    def record_pose_and_image(self) -> None:
        """Records the current pose and captures an image at the specified frequency."""
        while self.recording:
            self.record_current_state()
            time.sleep(1.0 / self.record_frequency)

    def start_recording(self) -> None:
        """Starts the recording of pose and image."""
        if not self.recording:
            self.recording = True
            self.recording_thread = threading.Thread(target=self.record_pose_and_image)
            self.recording_thread.start()

    def stop_recording(self) -> None:
        """Stops the recording of pose and image."""
        if self.recording:
            self.recording = False
            self.recording_thread.join()

    def _process_queue(self) -> None:
        """Processes the recording queue asynchronously."""
        while True:
            image, pose, instruction = self.recording_queue.get()
            self.recorder.record(observation={"image": image, "instruction": instruction}, action=pose)
            self.recording_queue.task_done()

    def record_current_state(self) -> None:
        """Records the current pose and image if the pose has changed."""
        pose = self.get_pose()
        if pose != self.last_recorded_pose:
            logging.info("Recording:", pose)
            image = self.capture()  # Capture an image
            self.recording_queue.put((image, pose, self.current_instruction))
            self.last_recorded_pose = pose

    def record_final_state(self) -> None:
        """Records the final pose and image after the movement completes."""
        self.record_current_state()
