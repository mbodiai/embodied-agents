# Copyright 2024 Mbodi AI
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


from mbodied_agents.base.sample import Sample
from pydantic import Field


class BoundingBox(Sample):
    """Represents a bounding box with minimum and maximum x and y coordinates.

    Attributes:
        x_min (int): x min coordinate
        y_min (int): y min coordinate
        x_max (int): x max coordinate
        y_max (int): y max coordinate
    """
    x_min: int = Field(..., description="x min")
    y_min: int = Field(..., description="y min")
    x_max: int = Field(..., description="x max")
    y_max: int = Field(..., description="y max")


class SceneObject(Sample):
    """Represents an object detected in a scene.

    Attributes:
        object_name (str): Label of the object
        centroid (tuple): Centroid coordinates of the object
        score (float): Detection confidence score
        bounding_box (BoundingBox): The bounding box of the object
    """
    object_name: str = Field(..., description="Label of the object")
    centroid: tuple = Field(..., description="Centroid coordinates of the object")
    score: float = Field(..., description="Detection confidence score")
    bounding_box: BoundingBox
    