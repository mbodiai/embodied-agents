
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from mbodied_agents.agents.sense.dino.utils import DetectionResult, get_boxes, load_image, refine_masks
from mbodied_agents.agents.sense.world_agent import World, WorldAgent
from mbodied_agents.types.vision import Image
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline


class DinoObjectDetectionAgent(WorldAgent):
    def __init__(self,
                 detector_id: str = "IDEA-Research/grounding-dino-base",
                 segmenter_id: str = "facebook/sam-vit-huge",
                 threshold: float = 0.3,
                 polygon_refinement: bool = False,
                 **kwargs):
        self.detector_id = detector_id
        self.segmenter_id = segmenter_id
        self.threshold = threshold
        self.polygon_refinement = polygon_refinement

    def detect(self,
               image: Image,
               labels: List[str],
               threshold: float = 0.3,
               detector_id: Optional[str] = None) -> List[Dict[str, Any]]:

        # Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.

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
            segmenter_id: str | None = None) -> List[DetectionResult]:

        # Use Segment Anything (SAM) to generate masks given an image + a set of bounding boxes.

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

    def act(self, image: Image, labels: List[str],) -> World:
        _, detections = self.detect_and_segment(image=image, labels=labels)
        return World()
