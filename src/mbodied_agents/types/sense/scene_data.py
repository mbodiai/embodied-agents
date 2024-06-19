from typing import List
from mbodied_agents.types.sense.vision import Image
from mbodied_agents.types.geometry import Pose6D
from mbodied_agents.base.sample import Sample


class ObjectPose(Sample):
    object_name: str
    object_pose: Pose6D


class SceneData(Sample):
    image: Image.Image
    object_poses: List[ObjectPose]
