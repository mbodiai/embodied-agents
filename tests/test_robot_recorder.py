import pytest
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image
from mbodied.robots.robot_recording import RobotRecorder
from mbodied.robots import SimRobot
from mbodied.data.replaying import Replayer
from tempfile import TemporaryDirectory
from gymnasium import spaces
from pathlib import Path


@pytest.fixture
def tempdir():
    with TemporaryDirectory("test") as temp_dir:
        yield temp_dir


@pytest.fixture
def robot_recorder(tempdir):
    robot = SimRobot()
    recorder_kwargs = {
        "name": "sim_record",
        "observation_space": spaces.Dict(
            {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
        ),
        "action_space": HandControl().space(),
        "out_dir": tempdir,
    }
    return RobotRecorder(robot=robot, frequency_hz=5, recorder_kwargs=recorder_kwargs)


def test_robot_recorder_record(tempdir):
    robot = SimRobot()
    recorder_kwargs = {
        "name": "sim_record.h5",
        "observation_space": spaces.Dict(
            {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
        ),
        "action_space": HandControl().space(),
        "out_dir": tempdir,
    }
    robot_recorder = RobotRecorder(robot=robot, frequency_hz=5, recorder_kwargs=recorder_kwargs)

    robot_recorder.start_recording(task="pick up the fork")
    robot.do(HandControl.unflatten([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    robot_recorder.stop_recording()

    assert robot_recorder.task == "pick up the fork"
    assert robot_recorder.recording is False

    # Replay the dataset and verify the recorded data.
    replayer = Replayer(Path(tempdir) / "sim_record.h5")
    assert replayer.size == 5  # 5 hz * 1 second
    for observation, action in replayer:
        assert observation["instruction"] == "pick up the fork"


def test_robot_recorder_record_context_manager(tempdir):
    robot = SimRobot()
    recorder_kwargs = {
        "name": "sim_record.h5",
        "observation_space": spaces.Dict(
            {"image": Image(size=(224, 224)).space(), "instruction": spaces.Text(1000)},
        ),
        "action_space": HandControl().space(),
        "out_dir": tempdir,
    }
    robot_recorder = RobotRecorder(robot=robot, frequency_hz=5, recorder_kwargs=recorder_kwargs)

    with robot_recorder.record("pick up the fork"):
        robot.do(HandControl.unflatten([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))

    assert robot_recorder.task == "pick up the fork"
    assert robot_recorder.recording is False

    # Replay the dataset and verify the recorded data.
    replayer = Replayer(Path(tempdir) / "sim_record.h5")
    assert replayer.size == 5  # 5 hz * 1 second
    for observation, action in replayer:
        assert observation["instruction"] == "pick up the fork"


if __name__ == "__main__":
    pytest.main()
