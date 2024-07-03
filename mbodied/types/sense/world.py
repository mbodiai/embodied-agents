from typing import List
from pydantic import  Field
from mbodied.base.sample import Sample

from mbodied.types.sense.vision import Image
from functools import partial
from mbodied.types.geometry import Pose6D
    
class SceneObjects(Sample):
    """Model for Scene Object."""
    object_name: List[str] | str = Field(default_factory=str, description="Name of the object in the scene")

class ObjectsPose(Sample):
    """Model for Object Pose."""
    object_pose: List[Pose6D] | Pose6D = Field(default_factory=Pose6D, description="Pose of the object with respect to a world frame")


class SceneData(Sample):
    """Model for Scene Data."""
    image: Image = Field(default_factory=partial(Image, size=(224,224)), description="Image of the scene")
    objects: SceneObjects = Field(default_factory=lambda: [SceneObjects()], description="List of objects in the scene")
    object_poses: ObjectsPose = Field(default_factory=lambda: [Pose6D()], description="List of object poses in the scene")

