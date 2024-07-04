# Object Pose Estimation

The `ObjectPoseEstimator3D` is a subclass of the `SensoryAgent` class. It outputs sensory readings by utilizing information from a sensor, such as a depth camera. 
This script demonstrates object detection and pose estimation using a 2D object detector and ArUco markers. 

It interacts with the Gradio server to process images and estimate object poses.

## Usage

To use the `act` method, instantiate the ObjectPoseEstimator3D class with the server URL and call the `act` method with appropriate arguments. For optimal results:

- Use an image with an ArUco marker present.
- Provide the camera intrinsic parameters and distortion coefficients.
- Provide the translation and rotational offsets required to align the ArUco marker frame with the base frame. If the ArUco marker is the base frame, provide `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.
- Provide a list of `object_classes` to detect in the image.

## Data Types

### CameraParameters

Represents the intrinsic and distortion parameters of the camera.

```python

class IntrinsicParameters(Sample):
    focal_length_x: float
    focal_length_y: float
    optical_center_x: float
    optical_center_y: float

class DistortionParameters(Sample):
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float

class CameraParameters(Sample):
    intrinsic: IntrinsicParameters
    distortion: DistortionParameters

```

### Pose6D
Represents the 6D pose of an object.

```python
class Pose6D(Sample):
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    
```

### SceneObjects
Represents the names of objects in the scene.

```python
class SceneObjects(Sample):
    object_name: List[str]

```

### ObjectsPose
Represents the poses of objects in the scene.

```python
class ObjectsPose(Sample):
    object_pose: List[Pose6D]

```

### SceneData
Represents the data of a scene, including the image, objects, and their poses.

```python
class SceneData(Sample):
    image: Image
    objects: SceneObjects
    object_poses: ObjectsPose

```


## Example
```python
from mbodied.types.sense.camera import IntrinsicParameters, DistortionParameters, CameraParameters
from mbodied.types.geometry import Pose6D
from mbodied.types.sense.world import SceneObjects, ObjectsPose

# Create intrinsic parameters
intrinsic_params = IntrinsicParameters(
    focal_length_x=911.0,
    focal_length_y=911.0,
    optical_center_x=653.0,
    optical_center_y=371.0
)

# Create distortion parameters
distortion_params = DistortionParameters(
    k1=0.0,
    k2=0.0,
    p1=0.0,
    p2=0.0,
    k3=0.0
)

# Create camera parameters
camera_intrinsics = CameraParameters(
    intrinsic=intrinsic_params,
    distortion=distortion_params
)

# Create ArUco pose in the world frame
aruco_pose_world_frame = Pose6D(
    x=0.0,
    y=0.2032,
    z=0.0,
    roll=-90.0,
    pitch=0.0,
    yaw=-90.0
)

# Create object classes
object_classes = SceneObjects(
    object_name=[
        "Remote Control",
        "Basket",
        "Fork",
        "Spoon",
        "Red Marker"
    ]
)

# Initialize the estimator
estimator = ObjectPoseEstimator3D(server_url="https://api.mbodi.ai/3d-object-pose-detection/")

# Call the act method
scene_data = estimator.act(
    rgb_image=Image("path/to/image.jpg"),
    depth_image=Image("path/to/depth_image.jpg"),
    camera_intrinsics=camera_intrinsics,
    distortion_coeffs=distortion_params,
    aruco_pose_world_frame=aruco_pose_world_frame,
    object_classes=object_classes,
    confidence_threshold=0.5,
    using_realsense=False
)

# Display the annotated image
scene_data.image.show()

# Print object names and poses
for obj, pose in zip(scene_data.objects.object_name, scene_data.object_poses.object_pose):
    print(f"Object: {obj}, Pose: {pose}")

```

# Example Pose Estimation Server

## Overview

The server script sets up a Gradio interface to perform object detection and pose estimation using images. The script includes the following functionalities:

- Image Processing Function (process_images): Processes RGB and depth images to detect objects and estimate their poses.
- Gradio Interface: A user-friendly interface to input images and parameters, and visualize results.

## Key Components
- Object Detection: Returns bounding boxes for detected objects.
- Object Segmentation: Returns the 2D centroid of objects as pixel values.
- 3D Object Pose Estimation: Deprojects the 2D points into 3D coordinates with respect to a base frame.
