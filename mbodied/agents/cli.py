import click

from mbodied.agents.sense.depth_estimation_agent import DepthEstimationAgent
from mbodied.agents.sense.object_detection_agent import ObjectDetectionAgent
from mbodied.agents.sense.segmentation_agent import SegmentationAgent
from mbodied.types.sense.vision import Image
from mbodied.types.sense.world import BBox2D, PixelCoords


@click.group()
def cli():
    """CLI for various AI agents."""
    pass


@cli.command()
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--image-path", prompt="Image path", help="Path to the image file.")
@click.option("--objects", prompt="Objects to detect", help="Comma-separated list of objects to detect.")
@click.option(
    "--model-type",
    type=click.Choice(["YOLOWorld", "Grounding DINO"], case_sensitive=False),
    prompt="Model type",
    help="The model type to use for detection.",
)
@click.option("--api-name", default="/detect", help="The API endpoint to use.")
def detect(model_src, image_path, objects, model_type, api_name):
    """Run the ObjectDetectionAgent to detect objects in an image."""
    image = Image(image_path, size=(224, 224))
    objects_list = objects.split(",")
    agent = ObjectDetectionAgent(model_src=model_src)
    result = agent.act(image=image, objects=objects_list, model_type=model_type, api_name=api_name)
    result.annotated.pil.show()


@cli.command()
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--image-path", prompt="Image path", help="Path to the image file.")
@click.option("--api-name", default="/depth", help="The API endpoint to use.")
def estimate_depth(model_src, image_path, api_name):
    """Run the DepthEstimationAgent to estimate depth from an image."""
    image = Image(image_path, size=(224, 224))
    agent = DepthEstimationAgent(model_src=model_src)
    result = agent.act(image=image, api_name=api_name)
    result.pil.show()


@cli.command()
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--image-path", prompt="Image path", help="Path to the image file.")
@click.option(
    "--input-type",
    type=click.Choice(["bbox", "coords"], case_sensitive=False),
    prompt="Input type",
    help="Type of input data (bbox or coords).",
)
@click.option(
    "--input-data", prompt="Input data", help="Bounding box coordinates as x1,y1,x2,y2 or pixel coordinates as u,v."
)
@click.option("--api-name", default="/segment", help="The API endpoint to use.")
def segment(model_src, image_path, input_type, input_data, api_name):
    """Run the SegmentationAgent to segment objects in an image."""
    image = Image(image_path, size=(224, 224))
    agent = SegmentationAgent(model_src=model_src)

    if input_type == "bbox":
        bbox_coords = list(map(int, input_data.split(",")))
        input_data = BBox2D(x1=bbox_coords[0], y1=bbox_coords[1], x2=bbox_coords[2], y2=bbox_coords[3])
    elif input_type == "coords":
        u, v = map(int, input_data.split(","))
        input_data = PixelCoords(u=u, v=v)

    mask_image, masks = agent.act(image=image, input_data=input_data, api_name=api_name)
    print("Masks shape:", masks.shape)
    mask_image.pil.show()


if __name__ == "__main__":
    cli()
