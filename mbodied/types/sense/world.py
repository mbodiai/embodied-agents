from typing import List, NamedTuple

from pydantic import Field

from mbodied.types.geometry import Pose
from mbodied.types.ndarray import NumpyArray
from mbodied.types.sample import Sample
from mbodied.types.sense.image import Image


class BBox2D(NamedTuple):
    """Model for 2D Bounding Box."""

    x1: float
    y1: float
    x2: float
    y2: float


class BBox3D(NamedTuple):
    """Model for 3D Bounding Box."""

    x1: float
    y1: float
    z1: float
    x2: float
    y2: float
    z2: float


class PixelCoords(NamedTuple):
    """Model for Pixel Coordinates."""

    u: int
    v: int


class WorldObject(Sample):
    """Model for World Object. It describes the objects in the world.

    Attributes:
        name (str): The name of the object.
        bbox_2d (BBox2D | None): The 2D bounding box of the object.
        bbox_3d (BBox3D | None): The 3D bounding box of the object.
        pose (Pose | None): The pose of the object.
        pixel_coords (PixelCoords | None): The pixel coordinates of the object.
    """

    name: str = ""
    bbox_2d: BBox2D | None = None
    bbox_3d: BBox3D | None = None
    pose: Pose | None = None
    pixel_coords: PixelCoords | None = None
    mask: NumpyArray | None = None


class World(Sample):
    """Model for World Data.

    Attributes:
        image (Image | None): The image of the world.
        depth (Image | None): The depth image of the world.
        annotated (Image | None): The annotated image of the world.
        objects (List[WorldObject]): The list of world objects.
    """

    image: Image | None = None
    depth: Image | None = None
    annotated: Image | None = None
    objects: List[WorldObject] = Field(default_factory=list, description="List of world objects")
