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
from mbodied.types.motion.control import HandControl, Pose6D, JointControl
from mbodied.robots import SimRobot


@pytest.fixture
def sim_robot():
    """Create a SimRobot instance for testing."""
    # Use 0.1 to speed up test run.
    return SimRobot(execution_time=0.1)


def test_initial_pose(sim_robot):
    """Test that the initial pose is set correctly."""
    expected_pose = HandControl.unflatten([0, 0, 0, 0, 0, 0, 0])
    assert sim_robot.get_state() == expected_pose


def test_do(sim_robot):
    """Test that the do method updates the current position correctly."""
    motion = HandControl(pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3), grasp=JointControl(value=0.5))
    sim_robot.do(motion)
    expected_pose = HandControl.unflatten([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5])
    assert sim_robot.get_state() == expected_pose

    # Perform another motion to ensure the position updates cumulatively
    another_motion = HandControl(
        pose=Pose6D(x=-0.1, y=-0.2, z=-0.3, roll=-0.1, pitch=-0.2, yaw=-0.3), grasp=JointControl(value=0.0)
    )
    sim_robot.do(another_motion)
    expected_pose = HandControl.unflatten([0, 0, 0, 0, 0, 0, 0.5])
    assert sim_robot.get_state() == expected_pose


def test_do_list(sim_robot):
    """Test that the do method updates the current position correctly."""
    motion_list = [
        HandControl(pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3), grasp=JointControl(value=0.0)),
        HandControl(pose=Pose6D(x=0.2, y=0.4, z=0.5, roll=0.5, pitch=0.6, yaw=0.7), grasp=JointControl(value=0.5)),
    ]
    sim_robot.do(motion_list)
    expected_pose = HandControl.unflatten([0.3, 0.6, 0.8, 0.6, 0.8, 1.0, 0.5])
    assert sim_robot.get_state() == expected_pose


@pytest.mark.asyncio
async def test_async_do(sim_robot):
    motion = HandControl(pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3), grasp=JointControl(value=0.5))
    await sim_robot.async_do(motion)
    expected_pose = HandControl.unflatten([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5])
    assert sim_robot.get_state() == expected_pose


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
