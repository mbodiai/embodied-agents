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



from embdata.motion.control import HandControl
from embdata.coordinate import Pose6D
from mbodied.robots.xarm_robot import XarmRobot


@pytest.fixture
def mock_xarm_api(monkeypatch):
    # Create a MagicMock class to stand in for XArmAPI so XArmRobot() returns this instance
    mock = MagicMock()
    monkeypatch.setattr("mbodied.robots.xarm_robot.XArmAPI", mock)

    mock_instance = mock.return_value
    # Stub the XArmAPI instance methods used by XarmRobot
    mock_instance.motion_enable.return_value = None
    mock_instance.clean_error.return_value = None
    mock_instance.set_mode.return_value = None
    mock_instance.set_state.return_value = None
    mock_instance.set_gripper_mode.return_value = None
    mock_instance.set_gripper_enable.return_value = None
    mock_instance.set_gripper_speed.return_value = None
    mock_instance.set_position.return_value = None
    mock_instance.set_gripper_position.return_value = None
    mock_instance.set_collision_sensitivity.return_value = None
    mock_instance.set_self_collision_detection.return_value = None
    mock_instance.get_position.return_value = (0, [300, 0, 325, -3.14, 0, 0])
    mock_instance.get_gripper_position.return_value = (0, 800)
    return mock_instance



def test_initialization(mock_xarm_api):
    XarmRobot(ip="192.168.1.228")
    # Test initialization calls
    mock_xarm_api.motion_enable.assert_called_with(True)
    mock_xarm_api.clean_error.assert_called_once()
    mock_xarm_api.set_mode.assert_called_with(0)
    mock_xarm_api.set_state.assert_called_with(0)
    mock_xarm_api.set_gripper_mode.assert_called_with(0)
    mock_xarm_api.set_gripper_enable.assert_called_with(True)
    mock_xarm_api.set_gripper_speed.assert_called_with(2000)
    mock_xarm_api.set_position.assert_called_with(300, 0, 325, math.radians(180), 0, 0, is_radian=True, wait=True, speed=200)
    mock_xarm_api.set_gripper_position.assert_called_with(800, wait=True)


def test_do(mock_xarm_api):
    motion = HandControl(
        pose=Pose6D(*[0.1, 0.2, 0.3, 0.1, 0.2, 0.3]), grasp=0.6
    )
    mock_xarm_api.get_position.return_value = (0, [300, 0, 325, -3.14, 0, 0])

    xarm = XarmRobot(ip="192.168.1.228")
    xarm.do(motion)
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    assert motion.info["fields"]["pose"]["motion_type"] == "relative"
    cur_t = np.array([300, 0, 325], dtype=float) / 1000.0
    cur_rpy = np.array([-3.14, 0.0, 0.0], dtype=float)
    d_t = np.array([0.1, 0.2, 0.3], dtype=float)
    d_rpy = np.array([0.1, 0.2, 0.3], dtype=float)
    R_cur = R.from_euler("XYZ", cur_rpy, degrees=False).as_matrix()
    R_d = R.from_euler("XYZ", d_rpy, degrees=False).as_matrix()
    t_target = cur_t + R_cur @ d_t
    rpy_target = R.from_matrix(R_cur @ R_d).as_euler("XYZ", degrees=False)
    expected_position = [*(t_target * 1000.0), *rpy_target]

    args, kwargs = mock_xarm_api.set_position.call_args
    for a, e in zip(args, expected_position):
        assert a == pytest.approx(e, rel=1e-6, abs=1e-6)
    assert kwargs.get("wait") is True
    assert kwargs.get("speed") == 200
    expected_grip = xarm.clip_scale_grasp(0.6)
    mock_xarm_api.set_gripper_position.assert_called_with(expected_grip, wait=True)


def test_get_state(mock_xarm_api):
    mock_xarm_api.get_position.return_value = (0, [300, 0, 325, 3.14, 0, 0])

    state = XarmRobot(ip="192.168.1.228").get_state()

    expected_pose = Pose6D(x=0.3, y=0.0, z=0.325, roll=3.14, pitch=0.0, yaw=0.0)
    assert state.pose == expected_pose


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
