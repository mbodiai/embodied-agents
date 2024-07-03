from typing import List, Optional, Any
from pydantic import  Field
from mbodied.base.sample import Sample

from mbodied.types.sense.vision import Image
from mbodied.types.sense.camera import CameraParameters
from functools import partial
    
class SceneObject(Sample):
    """Model for Ground Truth Objects."""
    rotation_object_to_camera: List[float] = Field(
        default_factory= lambda: [0.0], description="Rotation matrix from model to camera coordinate system")
    translation_object_to_camera: List[float] = Field(
        default_factory=lambda: [0.0], description="Translation vector from model to camera coordinate system")
    object_id: int = -1
    shapenet_name: str = "unknown"
    object_name: str = "unknown"

class SceneData(Sample):
    """Model for Scene Data."""
    camera: CameraParameters = Field(default_factory=CameraParameters, description="Camera parameters")
    objects: List[SceneObject] = Field(default_factory=lambda: [SceneObject()], description="List of objects in the scene")
    image: Image | str = Field(default_factory=partial(Image, size=(224,224)), description="Image of the scene")