# Object Pose Estimation

The `ObjectPoseEstimator3D` is a subclass of the `SensoryAgent` class. It outputs sensory readings by utilizing information from a sensor, such as a depth camera. 
This script demonstrates object detection and pose estimation using a 2D object detector and ArUco markers. 

It interacts with the Gradio server to process images and estimate object poses.

## Usage

To use the sense function, instantiate the `ObjectPoseEstimator3D` class with the server URL and call the `sense` method with appropriate arguments. 
For optimal results:
- Use an image with an ArUco marker present.
- Provide the path of the camera intrinsic mati and distortion coefficient list.
- Provide the translation and rotational offsets required to align the ArUco marker frame with the base frame. If the ArUco marker is the base frame, provide `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`.
- Provide a list of object names to detect in the image.
  
```python
estimator = ObjectPoseEstimator3D(server_url="https://api.mbodi.ai/3d-object-pose-detection/")
result = estimator.sense(
    rgb_image_path="color_image.png",
    depth_image_path="depth_image.png",
    camera_intrinsics="intrinsic_matrix.npy",
    distortion_coeffs=[0.0, 0.0, 0.0, 0.0, 0.0],
    aruco_pose_world_frame=[0.0, 0.2032, 0.0, -90, 0, -90],
    object_classes=["Remote Control", "Basket", "Fork", "Spoon", "Red Marker"],
    confidence_threshold=0.5,
    using_realsense=False,
)
```

# Example Pose Estimation Server

## Overview

The server script sets up a Gradio interface to perform object detection and pose estimation using images. The script includes the following functionalities:

- Image Processing Function (`process_images`): Processes RGB and depth images to detect objects and estimate their poses.
- Gradio Interface: A user-friendly interface to input images and parameters, and visualize results.

## Key Components
- Object Detection: Returns bounding boxes for detected objects.
- Object Segmentation: Returns the 2D centroid of objects as pixel values.
- 3D Object Pose Estimation: Deprojects the 2D points into 3D coordinates with respect to a base frame.
