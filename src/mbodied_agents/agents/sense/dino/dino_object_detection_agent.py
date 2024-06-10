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

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mbodied_agents.agents.sense.dino.utils import DetectionResult, get_boxes, load_image, refine_masks
from mbodied_agents.agents.sense.world_agent import WorldAgent
from mbodied_agents.types.vision import Image
from mbodied_agents.types.world import BoundingBox, SceneObject
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


class DinoObjectDetectionAgent(WorldAgent):
    def __init__(self,
                 detector_id: str = "IDEA-Research/grounding-dino-base",
                 segmenter_id: str = "facebook/sam-vit-huge",
                 threshold: float = 0.3,
                 polygon_refinement: bool = False,
                 **kwargs):
        """Initialization of DinoObjectDetectionAgent.

        Args:
            detector_id (str): The model ID for the object detector. Defaults to "IDEA-Research/grounding-dino-base".
            segmenter_id (str): The model ID for the mask generator. Defaults to "facebook/sam-vit-huge".
            threshold (float): The confidence threshold for object detection. Defaults to 0.3.
            polygon_refinement (bool): Flag to enable polygon refinement for segmentation masks. Defaults to False.
            **kwargs: Additional keyword arguments.
        """
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.threshold = threshold
        self.polygon_refinement = polygon_refinement

    def detect(self,
               image: Image,
               labels: List[str],
               threshold: float = 0.3,
               detector_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Detect objects in an image using Grounding DINO.

        Args:
            image (Image): The image in which to detect objects.
            labels (List[str]): List of labels to detect.
            threshold (float, optional): The confidence threshold for object detection. Defaults to 0.3.
            detector_id (Optional[str], optional): The model ID for the object detector. Defaults to "IDEA-Research/grounding-dino-tiny" if None.

        Returns:
            List[Dict[str, Any]]: A list of detection results, where each result is a dictionary containing detected object information.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        detector_id = detector_id if detector_id is not None else "IDEA-Research/grounding-dino-tiny"
        object_detector = pipeline(
            model=detector_id, task="zero-shot-object-detection", device=device)

        labels = [label if label.endswith(
            ".") else label+"." for label in labels]

        results = object_detector(
            image,  candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]

        return results

    def segment(
            self,
            image: Image,
            detection_results: List[Dict[str, Any]],
            polygon_refinement: bool = False,
            segmenter_id: Optional[str] = None) -> List[DetectionResult]:
        """Generate segmentation masks for detected objects in an image using SAM.

        Args:
            image (Image): The image for which to generate masks.
            detection_results (List[Dict[str, Any]]): List of detection results to segment.
            polygon_refinement (bool, optional): Flag to enable polygon refinement for segmentation masks. Defaults to False.
            segmenter_id (Optional[str], optional): The model ID for the mask generator. Defaults to "facebook/sam-vit-huge" if None.

        Returns:
            List[DetectionResult]: A list of DetectionResult objects, each containing information about a detected object's mask.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        segmenter_id = segmenter_id if segmenter_id is not None else "facebook/sam-vit-huge"

        segmentator = AutoModelForMaskGeneration.from_pretrained(
            segmenter_id).to(device)
        processor = AutoProcessor.from_pretrained(segmenter_id)

        boxes = get_boxes(detection_results)
        inputs = processor(images=image, input_boxes=boxes,
                           return_tensors="pt").to(device)

        outputs = segmentator(**inputs)
        masks = processor.post_process_masks(
            masks=outputs.pred_masks, original_sizes=inputs.original_sizes, reshaped_input_sizes=inputs.reshaped_input_sizes)[0]

        masks = refine_masks(masks, polygon_refinement)

        for detection_result, mask in zip(detection_results, masks, strict=True):
            detection_result.mask = mask

        return detection_results

    def detect_and_segment(
        self,
        image: Union[str, np.ndarray, Image],
        labels: List[str],
        save_name: str = "./",
    ) -> Tuple[np.ndarray, List[DetectionResult]]:
        """Detect and segment objects in an image.

        This method first detects objects in the given image using the configured
        object detector and threshold. It then segments the detected objects using
        the configured segmenter and optionally refines the segmentation masks.

        Args:
            image (Union[str, np.ndarray, Image]): The image in which to detect and segment objects. Can be a file path,
                a numpy array or an Image object.
            labels (List[str]): A list of labels to detect in the image.
            save_name (str): Optional string for naming the save file. Defaults to "./".

        Returns:
            Tuple[np.ndarray, List[DetectionResult]]: A tuple containing:
              - A numpy array of the image.
              - A list of DetectionResult objects, each containing information about a detected and segmented object.
        """
        image = load_image(image)
        detections = self.detect(image, labels,
                                 self.threshold,
                                 self.detector_id,
                                 )
        detections = self.segment(
            image, detections,
            self.polygon_refinement,
            self.segmenter_id,
        )

        image_array = np.array(image)

        return image_array, detections

    def act(self, image: Image, labels: List[str]) -> List[SceneObject]:
        """Detects and segments objects in an image and returns a list of SceneObjects.

        This method first detects and segments objects in the provided image using the 
        `detect_and_segment` method. It then processes the detections and constructs 
        SceneObject instances for each valid detection, which include the object's name, 
        centroid, score, and bounding box.

        Args:
            image (Image): The image in which to detect and segment objects.
            labels (List[str]): A list of labels used to identify objects in the image.

        Returns:
            List[SceneObject]: A list of SceneObject instances representing 
            the detected and segmented objects.

        """
        _, detections = self.detect_and_segment(image=image, labels=labels)
        detection_list = []
        for detection in detections:
            if detection.centroid is None:
                continue
            detection_bb = detection.box
            detection_list.append(
                SceneObject(
                    object_name=detection.label,
                    centroid=detection.centroid,
                    score=detection.score,
                    bounding_box=BoundingBox(
                        x_min=detection_bb.xmin,
                        y_min=detection_bb.ymin,
                        x_max=detection_bb.xmax,
                        y_max=detection_bb.ymax,
                    ),
                ),
            )
        return detection_list
