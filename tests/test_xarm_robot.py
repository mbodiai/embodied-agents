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
from mbodied.robots.xarm_robot import XarmRobot


@pytest.fixture
def mock_xarm_api():
    """Create a properly configured mock for XArmAPI"""
    mock = MagicMock()
    mock.motion_enable.return_value = 0  # Success code
    mock.clean_error.return_value = 0
    mock.set_mode.return_value = 0
    mock.set_state.return_value = 0
    mock.set_gripper_mode.return_value = 0
    mock.set_gripper_enable.return_value = 0
    mock.set_gripper_speed.return_value = 0
    mock.set_position.return_value = 0
    mock.set_gripper_position.return_value = 0
    mock.get_position.return_value = (0, [300, 0, 325, -3.14, 0, 0])
    mock.get_gripper_position.return_value = (0, 800)
    mock.set_collision_sensitivity.return_value = 0
    mock.set_self_collision_detection.return_value = 0
    return mock


@pytest.fixture
def xarm(mock_xarm_api, monkeypatch):
    """Create XarmRobot with mocked XArmAPI"""
    # Patch the XArmAPI constructor to return our mock
    monkeypatch.setattr("mbodied.robots.xarm_robot.XArmAPI", lambda *args, **kwargs: mock_xarm_api)
    
    # Create and return XarmRobot instance
    robot = XarmRobot(ip="192.168.1.228")
    return robot



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

    mock_xarm_api.set_position.assert_called_with(*expected_position, wait=True, speed=200)
    mock_xarm_api.set_gripper_position.assert_called_with(800, wait=True)


def test_get_state(mock_xarm_api, xarm):
    mock_xarm_api.get_position.return_value = (0, [300, 0, 325, 3.14, 0, 0])

    pose = xarm.get_state()

    expected_pose = [0.3, 0.0, 0.325, 3.14, 0.0, 0.0, 1.0]
    hand_pose = HandControl.unflatten(expected_pose)
    assert pose == hand_pose


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
