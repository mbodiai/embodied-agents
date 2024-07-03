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
import numpy as np
import json
import logging
from pathlib import Path
import h5py
from tempfile import TemporaryDirectory

from mbodied.types.motion.control import (
    LocationAngle,
    Pose6D,
    JointControl,
    FullJointControl,
    HandControl,
    HeadControl,
    MobileSingleArmControl,
)
from mbodied.data.recording import Recorder


@pytest.fixture(autouse=True)
def mock_file():
    with TemporaryDirectory() as tmpdirname:
        filepath = Path(tmpdirname) / "test.h5"
        yield filepath


def test_location_angle_serialization():
    la = LocationAngle(x=0.5, y=-0.5, theta=1.57)
    json_data = json.dumps(la.dict())
    expected = {"x": 0.5, "y": -0.5, "theta": 1.57}
    assert json.loads(json_data) == expected


def test_location_angle_deserialization():
    json_data = '{"x": 0.5, "y": -0.5, "theta": 1.57}'
    la = LocationAngle.model_validate_json(json_data)
    assert la.x == 0.5 and la.y == -0.5 and la.theta == 1.57


def test_pose6d_serialization():
    pose = Pose6D(x=1, y=0.9, z=0.9, roll=0.1, pitch=0.2, yaw=0.3)
    json_data = json.dumps(pose.dict())
    expected = {"x": 1, "y": 0.9, "z": 0.9, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    assert json.loads(json_data) == expected


def test_pose6d_deserialization():
    json_data = '{"x": 1, "y": 0.9, "z": 0.9, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}'
    pose = Pose6D.model_validate_json(json_data)
    assert (pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw) == (1, 0.9, 0.9, 0.1, 0.2, 0.3)


def test_joint_control_serialization():
    jc = JointControl(value=2.5)
    json_data = json.dumps(jc.dict())
    expected = {"value": 2.5}
    assert json.loads(json_data) == expected


def test_joint_control_deserialization():
    json_data = '{"value": 2.5}'
    jc = JointControl.model_validate_json(json_data)
    assert jc.value == 2.5


def test_full_joint_control_serialization():
    fjc = FullJointControl(joints=[2.5, -1.0], names=["elbow", "wrist"])
    json_data = json.dumps(fjc.dict())
    expected = {"joints": [2.5, -1.0], "names": ["elbow", "wrist"]}
    assert json.loads(json_data) == expected


def test_full_joint_control_deserialization():
    json_data = '{"joints": [2.5, -1.0], "names": ["elbow", "wrist"]}'
    fjc = FullJointControl.model_validate_json(json_data)
    assert fjc.joints == [2.5, -1.0]
    assert fjc.names == ["elbow", "wrist"]


def test_mobile_single_arm_control_serialization():
    msac = MobileSingleArmControl(
        base=LocationAngle(x=0.5, y=-0.5, theta=1.57),
        arm=FullJointControl(joints=[2.5, -1.0], names=["elbow", "wrist"]),
        head=HeadControl(tilt=JointControl(value=1.0), pan=JointControl(value=-1.0)),
    )
    json_data = json.dumps(msac.dict())
    expected = {
        "base": {"x": 0.5, "y": -0.5, "theta": 1.57},
        "arm": {"joints": [2.5, -1.0], "names": ["elbow", "wrist"]},
        "head": {"tilt": {"value": 1.0}, "pan": {"value": -1.0}},
    }
    assert json.loads(json_data) == expected


def test_mobile_single_arm_control_deserialization():
    json_data = '{"base": {"x": 0.5, "y": -0.5, "theta": 1.57}, "arm": {"joints": [2.5, -1.0], "names": ["elbow", "wrist"]}, "head": {"tilt": {"value": 1.0}, "pan": {"value": -1.0}}}'
    msac = MobileSingleArmControl.model_validate_json(json_data)
    assert (msac.base.x, msac.base.y, msac.base.theta) == (0.5, -0.5, 1.57)
    assert msac.arm.joints == [2.5, -1.0]
    assert msac.head.tilt.value == 1.0
    assert msac.head.pan.value == -1.0


def test_hand_control_serialization():
    hc = HandControl(pose=Pose6D(x=0.5, y=-0.5, z=0.5, roll=0.5, pitch=-0.5, yaw=0.5), grasp=JointControl(value=1.0))
    json_data = json.dumps(hc.dict())
    expected = {
        "pose": {"x": 0.5, "y": -0.5, "z": 0.5, "roll": 0.5, "pitch": -0.5, "yaw": 0.5},
        "grasp": {"value": 1.0},
    }
    assert json.loads(json_data) == expected


def test_hand_control_deserialization():
    json_data = (
        '{"pose": {"x": 0.5, "y": -0.5, "z": 0.5, "roll": 0.5, "pitch": -0.5, "yaw": 0.5}, "grasp": {"value": 1.0}}'
    )
    hc = HandControl.model_validate_json(json_data)
    assert (hc.pose.x, hc.pose.y, hc.pose.z) == (0.5, -0.5, 0.5)
    assert (hc.pose.roll, hc.pose.pitch, hc.pose.yaw) == (0.5, -0.5, 0.5)
    assert hc.grasp.value == 1.0


def test_recording_location_angle(mock_file):
    recorder = Recorder(mock_file.stem, out_dir=mock_file.parent)
    la = LocationAngle(x=0.5, y=-0.5, theta=1.57)
    recorder.record(la)
    with h5py.File(mock_file, "r") as file:
        assert file["observation/x"][0] == 0.5
        assert file["observation/y"][0] == -0.5
        assert file["observation/theta"][0] == 1.57
    recorder.close()


def test_recording_pose(mock_file):
    recorder = Recorder(mock_file.stem, out_dir=mock_file.parent)
    pose = Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3)
    recorder.record(pose)
    with h5py.File(mock_file, "r") as file:
        # Assuming the recorder saves each attribute separately
        assert file["observation/x"][0] == 0.1
        assert file["observation/y"][0] == 0.2
        assert file["observation/z"][0] == 0.3
        assert file["observation/roll"][0] == 0.1
        assert file["observation/pitch"][0] == 0.2
        assert file["observation/yaw"][0] == 0.3
    recorder.close()


def test_unflatten():
    original_pose = LocationAngle(x=0.5, y=-0.5, theta=1.57)
    flattened_pose = original_pose.flatten(output_type="dict")

    schema = {
        "type": "object",
        "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "theta": {"type": "number"},
        },
    }
    unflattened_pose = LocationAngle.unflatten(flattened_pose, schema)

    assert unflattened_pose.x == original_pose.x
    assert unflattened_pose.y == original_pose.y
    assert unflattened_pose.theta == original_pose.theta


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
