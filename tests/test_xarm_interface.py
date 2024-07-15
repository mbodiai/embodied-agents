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
from unittest.mock import MagicMock
import math

from mbodied.types.motion.control import HandControl, Pose6D
from mbodied.hardware.xarm_interface import XarmInterface


@pytest.fixture
def mock_xarm_api(mocker):
    # Mock the XArmAPI methods that are used in XarmInterface
    mock = mocker.patch("mbodied.hardware.xarm_interface.XArmAPI")
    mock_instance = mock.return_value
    mock_instance.motion_enable.return_value = None
    mock_instance.clean_error.return_value = None
    mock_instance.set_mode.return_value = None
    mock_instance.set_state.return_value = None
    mock_instance.set_gripper_mode.return_value = None
    mock_instance.set_gripper_enable.return_value = None
    mock_instance.set_gripper_speed.return_value = None
    mock_instance.set_position.return_value = None
    mock_instance.set_gripper_position.return_value = None
    mock_instance.get_position.return_value = (0, [300, 0, 325, -3.14, 0, 0])
    mock_instance.get_gripper_position.return_value = (0, 800)
    return mock_instance


@pytest.fixture
def xarm(mock_xarm_api):
    # Explicitly initialize XarmInterface to trigger the mocked calls
    return XarmInterface(ip="192.168.1.228")


def test_initialization(mock_xarm_api, xarm):
    # Test initialization calls
    mock_xarm_api.motion_enable.assert_called_with(True)
    mock_xarm_api.clean_error.assert_called_once()
    mock_xarm_api.set_mode.assert_called_with(0)
    mock_xarm_api.set_state.assert_called_with(0)
    mock_xarm_api.set_gripper_mode.assert_called_with(0)
    mock_xarm_api.set_gripper_enable.assert_called_with(True)
    mock_xarm_api.set_gripper_speed.assert_called_with(2000)
    mock_xarm_api.set_position.assert_called_with(300, 0, 325, math.radians(180), 0, 0, wait=True, speed=200)
    mock_xarm_api.set_gripper_position.assert_called_with(800, wait=True)


def test_do(mock_xarm_api, xarm):
    mock_motion = HandControl(
        pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3), grasp=MagicMock(value=0.6)
    )
    mock_xarm_api.get_position.return_value = (0, [300, 0, 325, -3.14, 0, 0])

    xarm.do(mock_motion)

    expected_position = [
        300 + 0.1 * 1000,
        0 + 0.2 * 1000,
        325 + 0.3 * 1000,
        -3.14 + 0.1,
        0 + 0.2,
        0 + 0.3,
    ]

    mock_xarm_api.set_position.assert_called_with(*expected_position, wait=False, speed=200)
    mock_xarm_api.set_gripper_position.assert_called_with(480.0, wait=False)


def test_get_pose(mock_xarm_api, xarm):
    mock_xarm_api.get_position.return_value = (0, [300, 0, 325, 3.14, 0, 0])

    pose = xarm.get_pose()

    expected_pose = [0.3, 0.0, 0.325, 3.14, 0.0, 0.0, 1.0]
    hand_pose = HandControl.unflatten(expected_pose)
    assert pose == hand_pose


def test_do_and_record(mock_xarm_api, mocker, xarm):
    # Mock the necessary methods
    mocker.patch.object(xarm, "start_recording", autospec=True)
    mocker.patch.object(xarm, "stop_recording", autospec=True)
    mocker.patch.object(xarm, "record_final_state", autospec=True)
    mocker.patch.object(xarm, "do", autospec=True)

    mock_motion = HandControl(
        pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3), grasp=MagicMock(value=0.6)
    )

    instruction = "move to position"
    xarm.do_and_record(instruction, mock_motion)

    xarm.start_recording.assert_called_once()
    xarm.stop_recording.assert_called_once()
    xarm.record_final_state.assert_called_once()
    xarm.do.assert_called_once_with(mock_motion)
    assert xarm.current_instruction == instruction


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
