from typing import List
from mbodied.types.sense.vision import Image
from mbodied.types.geometry import Pose6D
from mbodied.base.sample import Sample


class ObjectPose(Sample):
    object_name: str
    object_pose: Pose6D


class SceneData(Sample):
    image: Image
    object_poses: List[ObjectPose]
