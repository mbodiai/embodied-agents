import numpy as np
from mbodied_agents.agents.sense.dino.dino_object_detection_agent import DinoObjectDetectionAgent
from PIL import Image

# Initialize the instance of DinoPoseEstimator
object_detection_agent = DinoObjectDetectionAgent()

# Load an image
# image_path = "path_to_your_image.jpg"
# image = Image.open(image_path)
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

# Define the labels you are interested in detecting
labels = ["person", "car"]

# Use the detect_and_segment function
image_array, detections = object_detection_agent.act(
    image=image,
    labels=labels,
)

# Output results
print(f"Processed Image Array Shape: {image_array.shape}")
print("Detections:")
for detection in detections:
    print({
        'label': detection.label,
        'score': detection.score,
        'box': detection.box,
        'mask': detection.mask
    })
