import click

from mbodied.agents.language import LanguageAgent
from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.sense.segmentation_agent import SegmentationAgent
from mbodied.types.sense.vision import Image
from mbodied.types.sense.world import BBox2D, PixelCoords


@click.group()
def cli() -> None:
    """CLI for various AI agents."""
    pass


@cli.command("language_chat")
@click.option(
    "--model-src",
    default="openai",
    help="The model source for the LanguageAgent. i.e. openai, anthropic, gradio url, etc",
)
@click.option("--api-key", default=None, help="API key for the remote actor (if applicable).")
@click.option("--context", default=None, help="Starting context for the conversation.")
@click.option("--instruction", prompt="Instruction", help="Instruction for the LanguageAgent.")
@click.option("--image-path", default=None, help="Optional path to the image file.")
def language_chat(model_src, api_key, context, instruction, image_path) -> None:
    """Run the LanguageAgent to interact with users using natural language."""
    agent = LanguageAgent(model_src=model_src, api_key=api_key, context=context)
    image = Image(image_path) if image_path else None
    response = agent.act(instruction=instruction, image=image, context=context)
    print("Response:", response)


@cli.command("detect_objects")
@click.argument("image_filename")
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option(
    "--objects", prompt="Objects to detect (comma-separated)", help="Comma-separated list of objects to detect."
)
@click.option(
    "--model-type",
    type=click.Choice(["YOLOWorld", "Grounding DINO"], case_sensitive=False),
    prompt="Model type",
    help="The model type to use for detection.",
)
@click.option("--api-name", default="/detect", help="The API endpoint to use.")
def detect_objects(image_filename, model_src, objects, model_type, api_name) -> None:
    """Run the ObjectDetectionAgent to detect objects in an image."""
    image = Image(image_filename, size=(224, 224))
    objects_list = objects.split(",")
    agent = ObjectDetectionAgent(model_src=model_src)
    result = agent.act(image=image, objects=objects_list, model_type=model_type, api_name=api_name)
    result.annotated.pil.show()


@cli.command("estimate_depth")
@click.argument("image_filename")
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--api-name", default="/depth", help="The API endpoint to use.")
def estimate_depth(image_filename, model_src, api_name) -> None:
    """Run the DepthEstimationAgent to estimate depth from an image."""
    image = Image(image_filename, size=(224, 224))
    agent = DepthEstimationAgent(model_src=model_src)
    result = agent.act(image=image, api_name=api_name)
    result.pil.show()


@cli.command("segment")
@click.argument("image_filename")
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option(
    "--segment-type",
    type=click.Choice(["bbox", "coords"], case_sensitive=False),
    prompt="Input type - bounding box or pixel coordinates",
    help="Type of input data `bbox` (bounding box) or `coords` (pixel coordinates).",
)
@click.option(
    "--segment-input",
    prompt="Segment input data - x1,y1,x2,y2 (for bbox) or u,v (for coords)",
    help="Bounding box coordinates as x1,y1,x2,y2 or pixel coordinates as u,v.",
)
@click.option("--api-name", default="/segment", help="The API endpoint to use.")
def segment(image_filename, model_src, segment_type, segment_input, api_name) -> None:
    """Run the SegmentationAgent to segment objects in an image."""
    image = Image(image_filename, size=(224, 224))
    agent = SegmentationAgent(model_src=model_src)

    if segment_type == "bbox":
        bbox_coords = list(map(int, segment_input.split(",")))
        input_data = BBox2D(x1=bbox_coords[0], y1=bbox_coords[1], x2=bbox_coords[2], y2=bbox_coords[3])
    elif segment_type == "coords":
        u, v = map(int, segment_input.split(","))
        input_data = PixelCoords(u=u, v=v)

    mask_image, masks = agent.act(image=image, input_data=input_data, api_name=api_name)
    print("Masks shape:", masks.shape)
    mask_image.pil.show()


if __name__ == "__main__":
    cli()
