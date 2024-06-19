from typing import Dict, List, Optional

import gradio as gr
import numpy as np
from mbodied_agents.agents.sense.utils.object_pose_calculations import ObjectPoseEstimation
from mbodied_agents.types.sense.vision import Image
from PIL import Image as PILImage
from mbodied_agents.agents.sense.utils.object_detect_segment import ObjectDetector2D

detect_segmentor = ObjectDetector2D(detector_type="dino")

client_results = {}


def process_images(
    image: PILImage,
    depth: PILImage,
    camera_intrinsics: List[float],
    distortion_coeffs: List[float],
    aruco_to_base_offset: List[float],
    object_classes: List[str],
    confidence_threshold: float,
    camera_source: str,
    client_id: Optional[int] = None,
) -> Dict:
    """Process images and perform object detection and pose estimation.

    Args:
        image (PILImage): RGB image.
        depth (PILImage): Depth image.
        camera_intrinsics (List[float]): Camera intrinsics.
        distortion_coeffs (List[float]): Distortion coefficients.
        target_frame_offset (List[float]): Target frame offset.
        object_classes (List[str]): Object classes.
        confidence_threshold (float): Confidence threshold.
        camera_source (str): Camera source.
        client_id (Optional[int]): Client ID.

    Returns:
        Dict: Object poses.
        Image: Annotated image.
        int: Client ID.

    Example:
        >>> process_images(
        ...     image,
        ...     depth,
        ...     [618.0, 0.0, 320.0, 0.0, 618.0, 240.0, 0.0, 0.0, 1.0],
        ...     [0.0, 0.0, 0.0, 0.0, 0.0],
        ...     [0.3, 0.3, 0.1, 0.0, 0.0, 0.0],
        ...     ["person"],
        ...     0.3,
        ...     "realsense",
        ... )
    """
    if client_id in client_results:
        return client_results[client_id]

    camera_intrinsics = np.array(camera_intrinsics, dtype=float).reshape(3, 3)
    distortion_coeffs = np.array(distortion_coeffs, dtype=float).reshape(
        5,
    )
    aruco_to_base_offset = np.array(aruco_to_base_offset, dtype=float).reshape(
        6,
    )

    detection_results = detect_segmentor.detect_and_segment(
        image=Image(image),
        object_classes=object_classes[0],
        threshold=confidence_threshold,
        polygon_refinement=True,
        save_name="segmented_image.png",
    )

    centroids = [detection.centroid for detection in detection_results]
    detection_labels = [detection.label for detection in detection_results]
    centroids = dict(zip(detection_labels, centroids))

    object_poses, annotated_images = ObjectPoseEstimation(
        color_image=image,
        depth_image=depth,
        intrinsic_matrix=camera_intrinsics,
        centroids=centroids,
        distortion_coeffs=distortion_coeffs,
        aruco_to_base_offset=aruco_to_base_offset,
        camera_source=camera_source,
    ).pose_estimation()

    if client_id is None:
        client_id = len(client_results) + 1

    client_results[client_id] = (object_poses, annotated_images)
    return object_poses, annotated_images, client_id


MARKDOWN = """
## Object Detection and Pose Estimation

This demonstration showcases object detection and pose estimation using a 2D object detector and ArUco markers. Follow the instructions below to upload your images and provide necessary parameters to perform the detection and estimation.

### Instructions
1. **Upload a RGB Image (3 channels)**: Select and upload a color image in RGB format.
2. **Upload a Depth Image (16-bit raw depth)**: Select and upload a depth image that provides distance information for each pixel.
3. **Enter the Camera Intrinsic Matrix**: Provide the camera intrinsic matrix as a list of nine floats, representing the 3x3 matrix used for camera calibration.
4. **Enter the Distortion Coefficients**: Provide the distortion coefficients as a list of five floats. These coefficients correct the image distortion caused by the camera lens.
5. **Enter the Target Frame Offset**: Specify the translational and rotational offset between the ArUco marker and the base frame as a list of six floats. If the ArUco marker is the target frame, this can be zero.
6. **Enter the Object Labels**: List the object labels you wish to detect in the image.
7. **Enter the Confidence Threshold**: Set the confidence threshold for object detection. This value determines the minimum confidence score required to consider a detection valid.
8. **Select the Camera Source**: Choose the source of the camera, such as "realsense" or "webcam".
9. **Click 'Run'**: After providing all the inputs, click the 'Run' button to execute object detection and pose estimation.
10. **View Results**: The detected object poses and the annotated image will be displayed.
"""

# Example Camera Intrinsics as a list[float]
CAMERA_INTRINSICS = [
    [618.0, 0.0, 320.0],
    [0.0, 618.0, 240.0],
    [0.0, 0.0, 1.0],
]

# Example Distortion Coefficients as a list[float]
DISTORTION_COEFFS = [
    [0.0, 0.0, 0.0, 0.0, 0.0],
]

# Example Target Frame Offset as a list[float]
TARGET_FRAME_OFFSET = [
    [0.3, 0.3, 0.1, 0.0, 0.0, 0.0],
]

# Example Object Labels as a list[str]
OBJECT_LABELS = [
    ["person"],
]

# Example Confidence Threshold as a float
CONFIDENCE_THRESHOLD = 0.3

# Example Camera Source as a str
CAMERA_SOURCE = "realsense"

IMAGE_EXAMPLES = [
    [
        "https://media.roboflow.com/supervision/image-examples/people-walking.png",
        "https://media.roboflow.com/supervision/image-examples/people-walking.png",
        CAMERA_INTRINSICS,
        DISTORTION_COEFFS,
        TARGET_FRAME_OFFSET,
        OBJECT_LABELS,
        CONFIDENCE_THRESHOLD,
        CAMERA_SOURCE,
    ],
]

with gr.Blocks() as demo:
    current_client_id = gr.Number(label="Client ID", value=None)
    new_client_id = gr.Number(label="New Client ID", value=None)

    gr.Markdown(MARKDOWN)

    interface = gr.Interface(
        fn=process_images,
        examples=IMAGE_EXAMPLES,
        inputs=[
            gr.Image(label="RGB Image", image_mode="RGB"),
            gr.Image(label="Depth Image", image_mode="I"),
            gr.List(label="Camera Intrinsics", row_count=3, col_count=3),
            gr.List(label="Distortion Coefficients", row_count=1, col_count=5),
            gr.List(
                label="Target Frame Offset",
                row_count=1,
                col_count=6,
                headers=["Z(m)", "Y(m)", "X(m)", "Roll(degrees)", "Pitch(degrees)", "Yaw(degrees)"],
            ),
            gr.List(label="Object Labels", datatype="str"),
            gr.Slider(label="Confidence Threshold", minimum=0.0, maximum=1.0, step=0.1, value=0.3),
            gr.Radio(label="Camera Source", choices=["realsense", "webcam"]),
            current_client_id,
        ],
        outputs=[
            gr.JSON(label="3D Pose"),
            gr.Image(label="Annotated Image"),
            new_client_id,
        ],
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=8081, share=False, show_api=True)
