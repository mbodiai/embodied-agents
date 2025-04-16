import json
from typing import TYPE_CHECKING

import rich_click as click
import rich.box
import unicodedata
from rich.console import Console
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from mbodied import __version__
from mbodied.utils.import_utils import smart_import

console = Console(style="light_goldenrod2")
print = console.print  # type: ignore # noqa
if TYPE_CHECKING:
    from mbodied.agents.language import LanguageAgent
    from mbodied.agents.sense import DepthEstimationAgent, ObjectDetectionAgent


def format_with_rich(doc: str, title: str = "Help") -> None:
    """
    Formats and displays the given docstring in a table-like structure using rich.

    Args:
        doc (str): The docstring to format and display.
        title (str): Title for the panel/container. Defaults to "Help".
    """

    table = Table(show_lines=True, box=rich.box.ROUNDED)

    sections = {
        "Example Command": "",
        "Response": "",
        "Inputs": "",
        "Outputs": "",
        "API Documentation": "",
        "API Endpoint": "",
    }

    current_section = None
    for line in doc.split("\n"):
        stripped = line.strip()
        if stripped.startswith("Example command:"):
            current_section = "Example Command"
            sections[current_section] = stripped.replace("Example command:", "").strip()
        elif stripped.startswith("Response:"):
            current_section = "Response"
            sections[current_section] = stripped.replace("Response:", "").strip()
        elif stripped.startswith("Inputs:"):
            current_section = "Inputs"
        elif stripped.startswith("Outputs:"):
            current_section = "Outputs"
        elif stripped.startswith("API documentation:"):
            current_section = "API Documentation"
            sections[current_section] = stripped.replace("API documentation:", "").strip()
        elif stripped.startswith("API Endpoint:"):
            current_section = "API Endpoint"
            sections[current_section] = stripped.replace("API Endpoint:", "").strip()
        elif current_section:
            if sections[current_section]:
                sections[current_section] += "\n"
            sections[current_section] += stripped

    if sections["Inputs"]:
        inputs_table = Table(box=rich.box.SQUARE)
        inputs_table.add_column("Input Name", style="bold cyan", no_wrap=True)
        inputs_table.add_column("Description", style="dim")
        for input_line in sections["Inputs"].split("\n"):
            normalized_line = unicodedata.normalize("NFKC", input_line.strip())
            if normalized_line.startswith("-") and "[" in normalized_line and "]:" in normalized_line:
                start_idx = normalized_line.index("[") + 1
                end_idx = normalized_line.index("]")
                input_name = normalized_line[start_idx:end_idx]
                description = normalized_line.split("]:", 1)[1].strip()
                inputs_table.add_row(input_name, description)
        table.add_row(
            "[bold yellow]Inputs:[/bold yellow]", inputs_table if inputs_table.row_count > 0 else "[dim]None[/dim]"
        )

    if sections["Outputs"]:
        outputs_table = Table(box=rich.box.SQUARE)
        outputs_table.add_column("Output Name", style="bold cyan", no_wrap=True)
        outputs_table.add_column("Description", style="dim")
        for output_line in sections["Outputs"].split("\n"):
            normalized_line = unicodedata.normalize("NFKC", output_line.strip())
            if normalized_line.startswith("-") and "[" in normalized_line and "]:" in normalized_line:
                start_idx = normalized_line.index("[") + 1
                end_idx = normalized_line.index("]")
                output_name = normalized_line[start_idx:end_idx]
                description = normalized_line.split("]:", 1)[1].strip()
                outputs_table.add_row(output_name, description)
        table.add_row(
            "[bold yellow]Outputs:[/bold yellow]", outputs_table if outputs_table.row_count > 0 else "[dim]None[/dim]"
        )

    if sections["Example Command"]:
        table.add_row("[bold yellow]Example Command:[/bold yellow]", f"[green]{sections['Example Command']}[/green]")
    if sections["Response"]:
        table.add_row("[bold yellow]Response:[/bold yellow]", sections["Response"])

    if sections["API Documentation"]:
        table.add_row(
            "[bold yellow]API Documentation:[/bold yellow]",
            f"[blue underline]{sections['API Documentation']}[/blue underline]",
        )

    if sections["API Endpoint"]:
        table.add_row(
            "[bold yellow]API Endpoint:[/bold yellow]", f"[blue underline]{sections['API Endpoint']}[/blue underline]"
        )

    console.print(Panel(table, title=f"[bold cyan]{title}[/bold cyan]"))


def list_agents(verbose) -> None:
    """List available agents."""
    import inspect
    import sys

    from rich.table import Table

    for mode in ["language", "sense", "motion"]:
        table = Table(title=f"{mode.capitalize()} Agents")
        table.add_column("Agent Name", style="bold cyan")
        table.add_column("Description", style="blue")

        smart_import(f"mbodied.agents.{mode}")
        seen = set()
        for agent in inspect.getmembers(sys.modules[f"mbodied.agents.{mode}"], inspect.isclass):
            if agent[0].endswith("Agent") and agent[0] not in seen:
                description = (
                    inspect.getdoc(agent[1])[:100] if inspect.getdoc(agent[1]) else "No description available."
                )
                if verbose:
                    description = Markdown("""```python\n""" + inspect.getdoc(agent[1]))
                table.add_row(agent[0], description)
                seen.add(agent[0])

        console.print(table, overflow="ellipsis")
    console.print("\n")
    if not verbose:
        console.print("Hint: Rerun with `--verbose` to see full descriptions.")
    console.print(Markdown("For more information, run `mbodied [language | sense | motion] --help`."))
    console.print("\n")


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--dry-run", is_flag=True, help="Simulate the action without executing.")
@click.option("--list", "-l", is_flag=True, help="List available agents.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def cli(ctx: click.Context, verbose, dry_run, list, help) -> None:
    """CLI for various AI agents."""
    if ctx.obj is None:
        ctx.obj = {}

    ctx.obj["VERBOSE"] = verbose
    ctx.obj["DRY_RUN"] = dry_run

    if verbose:
        print("Verbose mode enabled.")
    if dry_run:
        print("Dry run mode enabled.")
    if list:
        list_agents(verbose)
    if not ctx.invoked_subcommand or help:
        ctx.get_help()


@cli.command("language")
@click.option(
    "--model-src",
    default="openai",
    help="The model source for the LanguageAgent. i.e. openai, anthropic, gradio url, etc",
)
@click.option("--api-key", default=None, help="API key for the remote actor (if applicable).")
@click.option("--context", default=None, help="Starting context for the conversation.")
@click.option("--instruction", default=None, help="Instruction for the LanguageAgent.")
@click.option("--image-path", default=None, help="Optional path to the image file.")
@click.option("--loop", is_flag=True, help="Keep the agent running for multiple instructions.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def language_chat(ctx, model_src, api_key, context, instruction, image_path, loop, help) -> None:
    """Run the LanguageAgent to interact with users using natural language.

    Example command:
        mbodied language --instruction "What type of robot is this?" --image-path resources/color_image.png

    Response:
        "This is a robotic arm, specifically a PR2 (Personal Robot 2) developed by Willow Garage."

    Inputs:
        - [model_src]: The model source for the LanguageAgent (e.g., openai, anthropic, or a gradio URL).
        - [api_key]: (Optional) API key for the remote actor, if needed.
        - [context]: Starting context for the conversation (optional).
        - [instruction]: Instruction or query for the LanguageAgent to respond to.
        - [image_path]: (Optional) path to an image file to include as part of the input.
        - [loop]: If set, the agent will continue running and accepting new instructions. Use "exit" to break loop.

    Outputs:
        - [Response]: The natural language response generated by the LanguageAgent.
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]
    LanguageAgent: "LanguageAgent" = smart_import("mbodied.agents.language", attribute="LanguageAgent")  # type: ignore # noqa
    Image = smart_import("mbodied.types.sense", attribute="Image")  # type: ignore # noqa
    if help:
        format_with_rich(language_chat.__doc__, title="Language Help")
        ctx.exit()

    if instruction is None:
        console.print("[cyan]Enter your initial instruction[/cyan]")
        instruction = Prompt.ask("Instruction")

    if verbose:
        print(f"Running language agent from {model_src}")

    if dry_run:
        print(f"Dry run: Would run LanguageAgent with model: {model_src}, instruction: {instruction}")
        ctx.exit()

    agent: "LanguageAgent" = LanguageAgent(model_src=model_src, api_key=api_key, context=context)
    image = Image(image_path) if image_path else None
    response = agent.act(instruction=instruction, image=image, context=context)
    console.print(Panel(response, title="Assistant", expand=False))
    while loop:
        try:
            instruction = Prompt.ask("User('-exit' to stop)")
            if instruction.lower() == "-exit":
                print("Exiting loop")
                break
            response = agent.act(instruction=instruction, image=image, context=context)
            console.print(Panel(response, title="Assistant", expand=False))

        except KeyboardInterrupt:
            print("Interrupted.")
            break


@cli.group(invoke_without_command=True)
@click.option("--list", "-l", is_flag=True, help="List available sensory models.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def sense(ctx, list, help):
    """Commands related to sensing tasks (detection, segmentation, depth estimation)."""
    if list:
        print("Available Sensory Models:")
        print("- Object Detection Models:")
        print("  - Grounding DINO")
        print("  - YOLOWorld")
        print("- Depth Estimation Models:")
        print("  - Depth Anything")
        print("  - Zoe Depth")
        print("- Segmentation Models:")
        print("  - Segment Anything(SAM2)")
        ctx.exit()

    if not ctx.invoked_subcommand or help:
        ctx.get_help()


@sense.command("detect")
@click.argument("image_filename", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--objects", prompt=False, help="Comma-separated list of objects to detect.")
@click.option(
    "--model-type",
    type=click.Choice(["YOLOWorld", "Grounding DINO"], case_sensitive=False),
    prompt=False,
    help="The model type to use for detection.",
)
@click.option("--api-name", default="/detect", help="The API endpoint to use.")
@click.option("--list", "-l", is_flag=True, help="List available models for object detection.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def detect_objects(ctx, image_filename, model_src, objects, model_type, api_name, list, help) -> None:
    """Run the ObjectDetectionAgent to detect objects in an image.

    Example command:
        mbodied sense detect resources/color_image.png --objects "remote, spoon" --model-type "YOLOWorld"

    Response:
        Annotated Image: The image with detected objects highlighted and labeled.

    Inputs:
        - [image_filename]: Path to the image file.
        - [objects]: Comma-separated list of objects to detect (e.g., "car, person").
        - [model_type]: Model type to use for detection (e.g., "YOLOWorld", "Grounding DINO").

    Outputs:
        - [Annotated Image]: Display of the image with detected objects and their bounding boxes.

    API Endpoint: https://api.mbodi.ai/sense/
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]

    if help:
        format_with_rich(detect_objects.__doc__, title="Detect Help")
        ctx.exit()

    if list:
        print("Available Object Detection Models:")
        print("- Grounding DINO")
        print("- YOLOWorld")
        ctx.exit()

    if image_filename is None:
        print("Error: Missing argument 'IMAGE_FILENAME'. Specify an image filename")
        print("Run 'mbodied sense detect --help' for assistance")
        ctx.exit()

    if objects is None:
        objects = click.prompt("Objects to detect (comma-separated)")

    if model_type is None:
        model_type = click.prompt(
            "Model Type", type=click.Choice(["YOLOWorld", "Grounding DINO"], case_sensitive=False)
        )

    if verbose:
        print(f"Running object detection on {image_filename} using {model_type}")

    if dry_run:
        print(f"Dry run: Would detect objects in {image_filename} with model: {model_type}, objects: {objects}")
        ctx.exit()
    Image = smart_import("mbodied.types.sense", attribute="Image")
    ObjectDetectionAgent = smart_import("mbodied.agents.sense", attribute="ObjectDetectionAgent")
    image = Image(image_filename, size=(224, 224))
    objects_list = objects.split(",")
    agent: "ObjectDetectionAgent" = ObjectDetectionAgent(model_src=model_src)
    result = agent.act(image=image, objects=objects_list, model_type=model_type, api_name=api_name)
    if verbose:
        print("Displaying annotated image.")
    result.annotated.pil.show()


@sense.command("depth")
@click.argument("image_filename", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--api-name", default="/depth", help="The API endpoint to use.")
@click.option("--list", "-l", is_flag=True, help="List available models for depth estimation.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def estimate_depth(ctx, image_filename, model_src, api_name, list, help) -> None:
    """Run the DepthEstimationAgent to estimate depth from an image.

    Example command:
        mbodied sense depth path/to/image.png

    Response:
        Depth map image displaying the estimated depth information for each pixel.

    Inputs:
        - [image_filename]: Path to the image file (e.g., PNG or RGBD image).

    Outputs:
        - [Depth Estimation Response]: A depth map image representing the depth information in the image.

    Loaded as API: [https://api.mbodi.ai/sense/depth](https://api.mbodi.ai/sense/depth)
    API Endpoint: https://api.mbodi.ai/sense/
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]

    if help:
        format_with_rich(estimate_depth.__doc__, title="Depth Help")
        ctx.exit()

    if list:
        print("Available Depth Estimation Models:")
        print("- Depth Anything")
        print("- Zoe Depth")
        ctx.exit()

    if image_filename is None:
        print("Error: Missing argument 'IMAGE_FILENAME'. Specify an image filename")
        print("Run 'mbodied sense depth --help' for assistance")
        ctx.exit()

    if verbose:
        print(f"Running depth estimation on {image_filename}")

    if dry_run:
        print(f"Dry run: Would estimate from image in {image_filename}")
        ctx.exit()
    Image = smart_import("mbodied.types.sense", attribute="Image")
    DepthEstimationAgent = smart_import("mbodied.agents.sense", attribute="DepthEstimationAgent")
    image = Image(path=image_filename, size=(224, 224))
    agent: "DepthEstimationAgent" = DepthEstimationAgent(model_src=model_src)
    result, depth_array = agent.act(image=image, api_name=api_name)
    result.pil.show()
    print("Depth array shape", depth_array.shape)


@sense.command("segment")
@click.argument("image_filename", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option(
    "--segment-type",
    type=click.Choice(["bbox", "coords"], case_sensitive=False),
    prompt=False,
    help="Type of input data `bbox` (bounding box) or `coords` (pixel coordinates).",
)
@click.option(
    "--segment-input",
    prompt=False,
    help="Bounding box coordinates as x1,y1,x2,y2 or pixel coordinates as u,v.",
)
@click.option("--api-name", default="/segment", help="The API endpoint to use.")
@click.option("--list", "-l", is_flag=True, help="List available models for segmentation.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def segment(ctx, image_filename, model_src, segment_type, segment_input, api_name, list, help) -> None:
    """Run the SegmentationAgent to segment objects in an image.

    Example command:
        mbodied sense segment resources/color_image.png --segment-type "bbox" --segment-input "50,50,150,150"

    Response:
        Masks shape:
        (1, 720, 1280)

    Inputs:
        - [image_filename]: Path to the image file.
        - [segment-type]: The type of segmentation input, either `bbox` for bounding box or `coords` for pixel coordinates.
        - [segment-input]: The input data, either bounding box coordinates as `x1,y1,x2,y2` or pixel coordinates as `u,v`.

    Outputs:
        - [Masks]: A 2D mask indicating the segmented region in the image.

    Loaded as API: [https://api.mbodi.ai/sense/segment](https://api.mbodi.ai/sense/segment)
    API Endpoint: https://api.mbodi.ai/sense/
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]

    if help:
        format_with_rich(segment.__doc__, title="Segment Help")
        ctx.exit()

    if list:
        print("Available Segmentation Models:")
        print("- Segment Anything(SAM2)")
        ctx.exit()

    if image_filename is None:
        print("Error: Missing argument 'IMAGE_FILENAME'. Specify an image filename")
        print("Run 'mbodied sense segment --help' for assistance")
        ctx.exit()

    if segment_type is None:
        segment_type = click.prompt(
            "Input type - bounding box or pixel coordinates",
            type=click.Choice(["bbox", "coords"], case_sensitive=False),
        )

    if segment_input is None:
        segment_input = click.prompt("Segment input data - x1,y1,x2,y2 (for bbox) or u,v (for coords)")

    if verbose:
        print(f"Running segmentation agent on {image_filename} to segment {segment_input}")

    if dry_run:
        print(f"Dry run: Would segment objects in {image_filename}")
        ctx.exit()
    Image = smart_import("mbodied.types.sense", attribute="Image")
    SegmentationAgent = smart_import("mbodied.agents.sense", attribute="SegmentationAgent")
    BBox2D = smart_import("mbodied.types.sense.world", attribute="BBox2D")
    PixelCoords = smart_import("mbodied.types.sense.world", attribute="PixelCoords")
    image = Image(image_filename, size=(224, 224))
    agent = SegmentationAgent(model_src=model_src)

    if segment_type == "bbox":
        bbox_coords = [int(x) for x in segment_input.split(",")]
        input_data = BBox2D(x1=bbox_coords[0], y1=bbox_coords[1], x2=bbox_coords[2], y2=bbox_coords[3])
    elif segment_type == "coords":
        u, v = map(int, segment_input.split(","))
        input_data = PixelCoords(u=u, v=v)

    mask_image, masks = agent.act(image=image, input_data=input_data, api_name=api_name)
    print("Masks shape:", masks.shape)
    mask_image.pil.show()


@cli.group(invoke_without_command=True)
@click.option("--list", "-l", is_flag=True, help="List available models for motion.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def motion(ctx, list, help):
    """Commands related to robot motion tasks."""
    if list:
        print("Available Motion Models:")
        print("- OPENVLA MODEL")
        ctx.exit()

    if not ctx.invoked_subcommand or help:
        ctx.get_help()
        ctx.exit()


@motion.command("openvla")
@click.argument("image_filename", required=False)
@click.option("--instruction", default=None, help="Instruction for the OpenVlaAgent.")
@click.option("--model-src", default="https://api.mbodi.ai/community-models/", help="The model source URL.")
@click.option("--unnorm-key", default="bridge_orig", help="Key for the unnormalized image.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def openvla_motion(ctx, instruction, image_filename, model_src, unnorm_key, help) -> None:
    """Run the OpenVlaAgent to generate robot motion based on instruction and image.

    Example command:
        mbodied motion openvla resources/xarm.jpeg --instruction "move forward"

    Response:
        Motion Response:
        HandControl(
            pose=Pose6D(
                x=-0.000432461563,
                y=0.000223397129,
                z=-0.000241243806,
                roll=-0.000138880808,
                pitch=0.00122899628,
                yaw=-6.67113405e-05
            ),
            grasp=JointControl(value=0.996078431)
        )

    Inputs:
        - [image_filename]: Path to the image file.
        - [instruction]: Instruction for the robot to act on.
        - [unnorm-key]: Key for the unnormalized image.

    Outputs:
        - [Motion Response]: HandControl object containing pose and grasp information.

    Loaded as API: [https://api.mbodi.ai/community-models/](https://api.mbodi.ai/community-models/)

    API Endpoint: https://api.mbodi.ai/community-models/
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]

    if help:
        format_with_rich(openvla_motion.__doc__, title="OpenVLA Help")
        ctx.exit()

    if image_filename is None:
        print("Error: Missing argument 'IMAGE_FILENAME'. Specify an image filename")
        print("Run 'mbodied motion openvla --help' for assistance")
        ctx.exit()

    if instruction is None:
        instruction = click.prompt("Instruction")

    if verbose:
        print(f"Running OpenVLA motion agent on {image_filename} with instruction: {instruction}")

    if dry_run:
        print(f"Dry run: Would generate robot motion from {image_filename} with instruction: {instruction}")
        ctx.exit()
    Image = smart_import("mbodied.types.sense", attribute="Image")
    OpenVlaAgent = smart_import("mbodied.agents.motion", attribute="OpenVlaAgent")

    image = Image(image_filename, size=(224, 224))
    agent = OpenVlaAgent(model_src=model_src)
    motion_response = agent.act(instruction=instruction, image=image, unnorm_key=unnorm_key)

    print("Motion Response:", motion_response.flatten())


@cli.command("auto")
@click.argument("task", required=False)
@click.option("--image-path", default=None, help="Optional path to the image file (for sense tasks).")
@click.option("--model-src", default="openai", help="Model source for agent")
@click.option("--api-key", default=None, help="API key for the remote model, if applicable.")
@click.option("--params", type=str, help="JSON string with parameters for the agent.")
@click.option("--help", "-h", is_flag=True, help="Show this message and exit.")
@click.pass_context
def auto(ctx, task, image_path, model_src, api_key, params, help):
    r"""Dynamically select and run the correct agent based on the task.

    Example command:
        mbodied auto language --params "{\"instruction\": \"Tell me a math joke?\"}"

    Response:
        Why was the equal sign so humble?
        Because it knew it wasn't less than or greater than anyone else!

    Example command:
        mbodied auto motion-openvla --params "{\"instruction\": \"Move forward\", \"image\": \"resources/bridge_example.jpeg\"}" --model-src "https://api.mbodi.ai/community-models/"

    Response:
        Response: HandControl(pose={'x': -0.00960310545, 'y': -0.0111081966, 'z': -0.00206002074, 'roll': 0.0126330038, 'pitch': -0.000780597846, 'yaw': -0.0177964902}, grasp={'value': 0.996078431})

    Inputs:
        - [task]: Task to be executed by the agent. Choices include:
            language: Run language-related tasks.
            motion-openvla: Use the OpenVlaAgent to generate robot motion.
            sense-object-detection: Run object detection tasks.
            sense-image-segmentation: Run image segmentation tasks.
            sense-depth-estimation: Run depth estimation tasks.

        - [image-path]: (Optional) Path to an image file, required for sense and motion tasks.
        - [model-src]: The source of the model, e.g., "openai", "gradio", etc.
        - [api-key]: (Optional) API key for accessing the remote model.
        - [params]: The parameters for the agent.

    Outputs:
        - [Response]: The output generated by the selected agent based on the task, such as HandControl for motion or detected objects for sensing tasks.
    """
    verbose = ctx.obj["VERBOSE"]
    dry_run = ctx.obj["DRY_RUN"]

    if help:
        format_with_rich(auto.__doc__, title="Auto Help")
        ctx.exit()

    if task is None:
        print("Error: Missing argument 'task'. Specify a task name")
        print("Run 'mbodied auto --help' for assistance")
        ctx.exit()

    if verbose:
        print(f"Executing 'auto' command with task: {task}")

    if dry_run:
        print(f"Dry run: Would execute 'auto' with task: {task}")
        ctx.exit()
    AutoAgent = smart_import("mbodied.agents.auto", attribute="AutoAgent")
    Image = smart_import("mbodied.types.sense", attribute="Image")
    if params:
        try:
            options = json.loads(params)
        except json.JSONDecodeError:
            print("Invalid JSON format for parameters.")
            ctx.exit()

    else:
        options = {}
    if "image" not in options:
        image = Image(image_path) if image_path else None
        options["image"] = image
    else:
        options["image"] = Image(options["image"])
    model_kwargs = {"api_key": api_key} if api_key else {}
    kwargs = options
    auto_agent = AutoAgent(task=task, model_src=model_src, model_kwargs=model_kwargs)

    response = auto_agent.act(**kwargs)
    if verbose:
        print(f"[Verbose] Auto agent response: {response}")
    print(f"Response: {response}")


@cli.command("version")
def version():
    """Display the version of mbodied."""
    print(f"mbodied version: {__version__}")


if __name__ == "__main__":
    cli()
