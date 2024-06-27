from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline
from ultralytics import YOLOv10
from mbodied.agents.sense.utils.detect_segment_utils import (
    DetectionResult,
    get_boxes,
    plot_detections,
    refine_masks,
)


class ObjectDetector2D:
    """A class to detect and segment objects in 2D images using DINO or YOLO detectors and SAM segmenter.

    Attributes:
        device (str): The device to run the models on ('cuda' or 'cpu').
        detector_type (str): The type of object detector ('dino' or 'yolo').
        detector_id (str): The identifier for the object detector model.
        segmenter_id (str): The identifier for the segmenter model.
        object_detector: The object detector model.
        segmentator: The segmenter model.
        processor: The processor for the segmenter model.
    """

    def __init__(
        self, detector_type: str = "dino", detector_id: str = None, segmenter_id: str = "facebook/sam-vit-base"
    ) -> None:
        """Initializes the ObjectDetector2D class with specified detector and segmenter.

        Args:
            detector_type (str): The type of object detector to use ('dino' or 'yolo').
            detector_id (str): The identifier for the object detector model.
            segmenter_id (str): The identifier for the segmenter model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.detector_type = detector_type
        self.detector_id = (
            detector_id
            if detector_id
            else ("jameslahm/yolov10b" if detector_type == "yolo" else "IDEA-Research/grounding-dino-tiny")
        )
        self.segmenter_id = segmenter_id

        if self.detector_type == "dino":
            self.object_detector = pipeline(
                model=self.detector_id, task="zero-shot-object-detection", device=self.device
            )
        elif self.detector_type == "yolo":
            self.object_detector = YOLOv10.from_pretrained(self.detector_id)

        self.segmentator = AutoModelForMaskGeneration.from_pretrained(self.segmenter_id).to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.segmenter_id)

    def detect(self, image: Image, labels: List[str] = None, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detects objects in the given image.

        Args:
            image (Image): The input image for object detection.
            labels (List[str]): The list of object classes to detect.
            threshold (float): The detection threshold.

        Returns:
            List[Dict[str, Any]]: The detection results.

        Example:
            >>> detector = ObjectDetector2D()
            >>> image = Image.open("example.jpg")
            >>> results = detector.detect(image, ["person", "dog"], 0.3)
        """
        if self.detector_type == "dino":
            labels = [label if label.endswith(".") else label + "." for label in labels]
            results = self.object_detector(image, candidate_labels=labels, threshold=threshold)
            results = [DetectionResult.from_dict(result) for result in results]
        elif self.detector_type == "yolo":
            results = self.object_detector.predict(source=image, save=True)
            for detection_result in results:
                detection_result.show()  # Display to screen
                detection_result.save(filename="result.jpg")  # Save to disk
            results = [
                DetectionResult.from_dict(result) for result in results
            ]  # Convert to DetectionResult format if necessary
        return results

    def segment(
        self, image: Image, detection_results: List[Dict[str, Any]], polygon_refinement: bool = False
    ) -> List[DetectionResult]:
        """Segments objects in the given image based on detection results.

        Args:
            image (Image): The input image for object segmentation.
            detection_results (List[Dict[str, Any]]): The detection results.
            polygon_refinement (bool): Whether to refine the segmentation masks into polygons.

        Returns:
            List[DetectionResult]: The segmentation results.

        Example:
            >>> detector = ObjectDetector2D()
            >>> image = Image.open("example.jpg")
            >>> detections = detector.detect(image, ["person", "dog"], 0.3)
            >>> segments = detector.segment(image, detections, True)
        """
        boxes = get_boxes(detection_results)
        boxes = [[[float(coord) for coord in box] for box in boxes]]
        inputs = self.processor(images=image, input_boxes=boxes, return_tensors="pt").to(self.device)

        outputs = self.segmentator(**inputs)
        masks = self.processor.post_process_masks(
            masks=outputs.pred_masks,
            original_sizes=inputs.original_sizes,
            reshaped_input_sizes=inputs.reshaped_input_sizes,
        )[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks, strict=False):
            detection_result.mask = mask

        return detection_results

    def detect_and_segment(
        self,
        image: Image,
        object_classes: List[str] = None,
        threshold: float = 1.0,
        polygon_refinement: bool = False,
        save_name: str = "experimental/images_for_segmentation/Segmented_image.png",
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detects and segments objects in the given image.

        Args:
            image (Image): The input image for detection and segmentation.
            object_classes (List[str]): The list of object classes to detect.
            threshold (float): The detection threshold.
            polygon_refinement (bool): Whether to refine the segmentation masks into polygons.
            save_name (str): The name to save the segmented image.

        Returns:
            Tuple[np.ndarray, List[DetectionResult]]: The detected and segmented results.

        Example:
            >>> detector = ObjectDetector2D()
            >>> image = Image.open("example.jpg")
            >>> results = detector.detect_and_segment(image, ["person", "dog"], 0.3, True, "segmented.png")
        """
        detections = self.detect(image.pil, object_classes, threshold)
        detections = self.segment(image.pil, detections, polygon_refinement)
        image_array = np.array(image.pil)
        plot_detections(image_array, detections, save_name)
        return detections


if __name__ == "__main__":
    detect_segmentor = ObjectDetector2D(
        detector_type="yolo",  # Change to "dino" for DINO detector
        detector_id="jameslahm/yolov10b",
        segmenter_id="facebook/sam-vit-huge",
    )

    detect_segmentor.detect_and_segment(
        image=Image.open("captured_images/images_for_segmentation/rgb_image.png"),
        object_classes=["Robot", "Remote control", "Blue Memory Card", "White Charger", "Whiteboard Eraser", "Tray"],
        threshold=0.3,
        polygon_refinement=True,
        save_name="experimental/images_for_segmentation/Segmented_image.png",
    )
