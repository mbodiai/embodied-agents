from typing import List

from mbodied_agents.base.sample import Sample
from pydantic import Field


class SceneObject(Sample):
    """Model for Ground Truth Objects."""
    object_name:str = Field(..., description="Label of the object")
    centroid: tuple = Field(..., description="")
