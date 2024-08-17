import numpy as np
from PIL import Image
import pytest
from mbodied.types.sense.depth import Depth

def test_depth_initialization():
  depth = Depth(mode="I", points=None, array=None)
  assert depth.mode == "I"
  assert depth.points is None
  assert depth.array is None

def test_depth_from_pil():
  pil_image = Image.new("RGB", (100, 100))
  depth = Depth.from_pil(pil_image)
  assert depth.mode == "I"
  assert depth.points is None
  assert depth.array is not None
  assert isinstance(depth.array, np.ndarray)

def test_depth_cluster_points():
  depth = Depth()
  depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
  labels = depth.cluster_points(n_clusters=2)
  assert len(labels) == 3
  assert set(labels) == {0, 1}

def test_depth_segment_plane():
  depth = Depth()
  depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
  inlier_mask, plane_coefficients = depth.segment_plane()
  assert inlier_mask.shape == (3,)
  assert plane_coefficients.shape == (3,)

def test_depth_segment_cylinder():
    depth = Depth()
    depth.points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    inlier_points, inlier_indices = depth.segment_cylinder()
    
    # Ensure that inlier_points and inlier_indices have the correct shapes
    assert len(inlier_points.shape) == 2
    assert inlier_points.shape[1] == 3  # Each point should have 3 coordinates
    
    assert len(inlier_indices.shape) == 1
    assert inlier_indices.shape[0] == inlier_points.shape[0]  # Number of inliers should match
