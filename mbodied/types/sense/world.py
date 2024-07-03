from typing import List
from pydantic import  Field
from mbodied.base.sample import Sample

from mbodied.types.sense.vision import Image
from functools import partial
from mbodied.types.geometry import Pose6D

class SceneObject(Sample):
    """Model for Scene Object Poses."""
    object_name: str = Field(default=lambda: str, description="The name of an object in the scene")
    object_pose: Pose6D = Field(default_factory=lambda: Pose6D(), description="Pose of the object with respect to a reference frame")


class SceneData(Sample):
    """Model for Scene Data."""
    image: Image = Field(default_factory=partial(Image, size=(224,224)), description="Annotated Image")
    scene_objects: List[SceneObject] = Field(default_factory=lambda: [SceneObject()], description="List of Scene Objects")

