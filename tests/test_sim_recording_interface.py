import pytest
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image
from mbodied.hardware.sim_recording_interface import SimRecordingInterface
from tempfile import TemporaryDirectory
from gymnasium import spaces


@pytest.fixture
def tempdir():
    with TemporaryDirectory("test") as temp_dir:
        yield temp_dir


@pytest.fixture
def sim_recording_interface(tempdir):
    recorder_kwargs = {
        "name": "sim_record",
        "observation_space": spaces.Dict(
            {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
        ),
        "action_space": HandControl().space(),
        "out_dir": tempdir,
    }
    return SimRecordingInterface(record_frequency=5, recorder_kwargs=recorder_kwargs)


def test_initialization(sim_recording_interface):
    assert sim_recording_interface.current_pos == [0, 0, 0, 0, 0, 0, 0], "Initial position is incorrect"


def test_capture(sim_recording_interface):
    image = sim_recording_interface.capture()
    assert isinstance(image, Image), "capture() did not return an Image instance"
    assert image.size == (224, 224), "capture() returned an Image of incorrect size"


def test_do_and_record(sim_recording_interface, tempdir):
    motion = HandControl.unflatten([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0])
    sim_recording_interface.do_and_record("pick up the fork", motion)


if __name__ == "__main__":
    pytest.main()
