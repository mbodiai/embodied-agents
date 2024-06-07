from mbodied_agents.agents.sense.pose_estimator.dino.realsense_functions import capture_realsense_images, pose_detector
from mbodied_agents.agents.sense.pose_estimator.dino.sam_with_grounding_dino import detect_and_segment

color_image, depth_image, intrinsics_matrix = capture_realsense_images()


detection_results = detect_and_segment(
        image=color_image,
        labels=["Robot", "Remote control", "Blue Memory Card", "White Charger", "Whiteboard Eraser", "Tray"],
        threshold=1.0,
        detector_id="IDEA-Research/grounding-dino-base",
        segmenter_id="facebook/sam-vit-huge",
        polygon_refinement=True,
        save_name="experimental/images_for_segmentation/Segmented_image.png",
    )

object_poses = pose_detector(detection_results.centroids)
