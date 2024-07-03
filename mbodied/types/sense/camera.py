from typing import List
from pydantic import Field
from mbodied.base.sample import Sample

class IntrinsicParameters(Sample):
    """Model for Camera Intrinsic Parameters."""
    focal_length_x: float = 0.0
    focal_length_y: float = 0.0
    optical_center_x: float = 0.0
    optical_center_y: float = 0.0

class ExtrinsicParameters(Sample):
    """Model for Camera Extrinsic Parameters."""
    rotation_matrix: List[float] = Field (
        default_factory= lambda: [0.0], description="Rotation matrix from world to camera coordinate system")
    translation_vector: List[float] = Field(
            default_factory=lambda: [0.0], description="Translation vector from world to camera coordinate system")

class DistortionParameters(Sample):
    """Model for Camera Distortion Parameters."""
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

class CameraParameters(Sample):
    """Model for Camera Parameters."""
    intrinsic: IntrinsicParameters = Field(
        default_factory=IntrinsicParameters, description="Intrinsic parameters of the camera")
    distortion: DistortionParameters = Field(
        default_factory=DistortionParameters, description="Distortion parameters of the camera")
    extrinsic: ExtrinsicParameters = Field(
        default_factory=ExtrinsicParameters, description="Extrinsic parameters of the camera")
    depth_scale: float = 1.0