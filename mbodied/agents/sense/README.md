# Sensory Agent

The `SensorAgent` is an abstract base class designed for creating agents that interact with various sensory components, such as images and/or depth maps. These agents can connect to different backends to process sensory data, enabling tasks like object detection, depth estimation, and more.

## Key Features

- **Backend Flexibility**: Natively supports multiple API services including OpenAI, Anthropic, vLLM, Ollama, HTTPX, and any Gradio endpoints.
- **Extensibility**: The `SensorAgent` class provides a flexible template for building custom sensory agents tailored to specific tasks or services.

## Quick Start

### 1. Import the Sensory Agent

Begin by importing the `SensorAgent` class from the `mbodied` package.

```python
from mbodied.agents.sense.sensor_agent import SensorAgent
```

### 2. Define a Custom Sensory Agent

Create a subclass of `SensorAgent` to define a custom sensory agent. For example,

```python
class MySensorAgent(SensorAgent):

    def __init__(
        self,
        model_src="https://api.mbodi.ai/sense/",
        model_kwargs=None,
        **kwargs,
    ):
        super().__init__(
            model_src=model_src,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def act(self, *args, **kwargs) -> Any:
        if self.actor is None:
            raise ValueError("Remote actor for agent not initialized.")
        response = self.actor.predict(*args, **kwargs)
        return response
```

In this example, the `MySensorAgent` class is configured to interact with a Gradio endpoint at the specified `model_src`. The `act` method sends data to the backend and returns the response.

### 3. Use the Custom Agent in Your Script

In another script, you can import and use your custom sensory agent as follows:

```python
from path.to.your.agent import MySensorAgent

# Initialize the agent
agent = MySensorAgent()

# Use the agent to process data
response = agent.act(image=your_image_data)
```

Replace `your_image_data` with the actual image or sensory data you wish to process. The `act` method sends the data to the backend and returns the result.

## Example Agents

### Depth Estimation Agent

The `DepthEstimationAgent` estimates depth from an image using a mbodi remote server. Here's how you can use it:

```python
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.types.sense.vision import Image

# Initialize the depth estimation agent
agent = DepthEstimationAgent(model_src="https://api.mbodi.ai/sense/")

# Use the agent to estimate depth from an image
image = Image("path/to/your/image.jpeg")
depth_image = agent.act(image=image)

# Display the depth image (assuming PIL integration)
depth_image.pil.show()
```

### Segmentation Agent

The `SegmentationAgent` performs image segmentation, identifying and segmenting objects within an image. Here's how you can use it:

```python
from mbodied.agents.sense.segmentation_agent import SegmentationAgent
from mbodied.types.sense.vision import Image
from mbodied.types.sense.world import BBox2D, PixelCoords

# Initialize the segmentation agent
agent = SegmentationAgent(model_src="https://api.mbodi.ai/sense/")

# Example 1: Segmenting using bounding boxes
bboxes = [BBox2D(x1=225, y1=196, x2=408, y2=355), BBox2D(x1=378, y1=179, x2=494, y2=236)]
mask_image, masks = agent.act(image=Image("path/to/your/image.jpeg"), input_data=bboxes)

# Display the segmented image
mask_image.pil.show()

# Example 2: Segmenting using pixel coordinates
pixel_coords = PixelCoords(u=800, v=100)
mask_image, masks = agent.act(image=Image("path/to/your/image.jpeg"), input_data=pixel_coords)

# Display the segmented image
mask_image.pil.show()
```

### Object Detection Agent

The `ObjectDetectionAgent` detects objects within an image and provides their locations. Here's how you can use it:

```python
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.types.sense.vision import Image

# Initialize the object detection agent
agent = ObjectDetectionAgent(model_src="https://api.mbodi.ai/sense/")

# Use the agent to detect objects in an image
image = Image("path/to/your/image.jpeg")
detection_result = agent.act(image=image, objects=["object_1", "object_2"], model_type="YOLOWorld")

# Display the annotated image with detected objects
detection_result.annotated.pil.show()
```
