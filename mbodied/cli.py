import json
from typing import TYPE_CHECKING
import rich_click as click

from mbodied.utils.import_utils import smart_import

if TYPE_CHECKING:
    from mbodied.agents.language import LanguageAgent
    from mbodied.agents.sense import DepthEstimationAgent, ObjectDetectionAgent
    from mbodied.types.sense import Image
    from rich.console import Console

_console = None
def getconsole() -> "Console":
    global _console
    if _console is None:
        Console = smart_import("rich.console", attribute="Console")
        _console = Console(style="light_goldenrod2")
    return _console

def format_with_rich(doc: str, title: str = "Help") -> None:
    """
    Formats and displays the given docstring in a table-like structure using rich.

    Args:
        doc (str): The docstring to format and display.
        title (str): Title for the panel/container. Defaults to "Help".
    """
    if not TYPE_CHECKING:
        rich = smart_import("rich")
        Table = smart_import("rich.table", attribute="Table")
    else:
        import rich
        from rich.table import Table
    table = Table(show_lines=True, box=rich.box.ROUNDED)
    import rich_click as click
    import rich.box
    import unicodedata
    from rich.console import Console
    from rich.prompt import Prompt
    from rich.markdown import Markdown
    from rich.panel import Panel
    from rich.table import Table
    from mbodied import __version__

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

    getconsole().print(Panel(table, title=f"[bold cyan]{title}[/bold cyan]"))


def list_agents(verbose) -> None:
    """List available agents."""
    import inspect
    import sys

    from rich.table import Table
    from rich.markdown import Markdown

    for mode in ["language", "sense", "motion"]:
        table = Table(title=f"{mode.capitalize()} Agents")
        table.add_column("Agent Name", style="bold cyan")
        table.add_column("Description", style="blue")

        smart_import(f"mbodied.agents.{mode}")
        seen = set()
        for agent in inspect.getmembers(sys.modules[f"mbodied.agents.{mode}"], inspect.isclass):
            if agent[0].endswith("Agent") and agent[0] not in seen:
                description = (
                    (inspect.getdoc(agent[1]) or "")[:100] if inspect.getdoc(agent[1]) else "No description available."
                )
                if verbose:
                    description = Markdown("""```python\n""" + (inspect.getdoc(agent[1] + "```") or "```"))
                table.add_row(agent[0], description)
                seen.add(agent[0])

        getconsole().print(table, overflow="ellipsis")
    getconsole().print("\n")
    if not verbose:
        getconsole().print("Hint: Rerun with `--verbose` to see full descriptions.")
    getconsole().print(Markdown("For more information, run `mbodied [language | sense | motion] --help`."))
    getconsole().print("\n")


@click.group(invoke_without_command=True)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.option("--list", "-l", is_flag=True, help="List available agents.")
@click.pass_context
def cli(ctx: click.Context, verbose, list) -> None:
    """CLI for various AI agents."""
    if verbose:
        print("Verbose mode enabled.")

    if list or not ctx.invoked_subcommand:
        list_agents(verbose)




@cli.command("lang")
@click.argument("instruction", required=False,default=None)
@click.option("--image-path", default=None, help="Optional path to an image file.")
@click.option(
    "--model-src",
    default="openai",
    help="The model source for the LanguageAgent. i.e. openai, anthropic, gradio url, etc",
)
@click.option("--api-key", default=None, help="API key for the remote actor (if applicable).")
@click.help_option("--help", "-h")
def language_chat(instruction, image_path, model_src, api_key) -> None:
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
    from rich.prompt import Confirm
    import sys
    try:
        import prompt_toolkit  # Now used for PromptSession
    except ImportError:
        if Confirm.ask("Missing dependencies: prompt_toolkit. Install?"):
            import os
            os.system(f"{sys.executable} -m pip install prompt_toolkit")
        else:
            getconsole().print("Exiting.", style="bold cyan")
            return
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import InMemoryHistory
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers.python import PythonLexer
    from prompt_toolkit.styles import Style
    from prompt_toolkit.enums import EditingMode

    if not TYPE_CHECKING:
        Image = smart_import("mbodied.types.sense", attribute="Image")
        LanguageAgent = smart_import("mbodied.agents.language", attribute="LanguageAgent")

        Panel = smart_import("rich.panel", attribute="Panel")
        Console = smart_import("rich.console", attribute="Console")
        Panel = smart_import("rich.panel", attribute="Panel")
        Console = smart_import("rich.console", attribute="Console")
    else:
        from rich.panel import Panel

    console = getconsole()
    console.print(Panel("Hello! How can I assist you today?", title="[bold cyan]Assistant[/bold cyan]", border_style="cyan"))

    commands = ['/exit', '/help']
    completer = WordCompleter(commands, ignore_case=True)
    session = PromptSession(completer=completer, history=InMemoryHistory())
    help_text = (
                "Type your instructions to interact with the LanguageAssistant.\n"
                "Commands:\n  /exit or /quit - Exit the REPL\n  /help - Show this help message"
            )
    agent: "LanguageAgent" = LanguageAgent(model_src=model_src, api_key=api_key)
    try:
        if instruction is not None:
            if image_path is not None:
                response = agent.act(instruction=instruction, image=Image(image_path))
            else:
                response = agent.act(instruction=instruction)
            console.print(Panel(response, title="[bold cyan]Assistant[/bold cyan]", border_style="cyan"))
            console.print("Run without arguments to enter the REPL.", style="bold cyan")
            return
        while True:
            user_input = session.prompt(">>> ",
                                        editing_mode=EditingMode.VI,
                                        lexer=PygmentsLexer(PythonLexer),
                                        style=Style([('', 'ansigreen')]))
            if user_input.strip().lower() in ['/exit', '/quit']:
                console.print(Panel("Exiting REPL.", style="bold green", border_style="green"))
                break
            elif user_input.strip().lower() == '/help':
                console.print(Panel(help_text, title="[bold yellow]Help[/bold yellow]", border_style="yellow"))
                continue
            if image_path is not None:
                response = agent.act(instruction=user_input, image=Image(image_path))
            else:
                response = agent.act(instruction=user_input)
            

            console.print(Panel(response, title="[bold cyan]Assistant[/bold cyan]", border_style="cyan"))
    except KeyboardInterrupt:
        console.print("\nInterrupted by user.", style="bold red")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
  
@cli.command("detect")
@click.argument("image_filename", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--objects", prompt=False, help="Comma-separated list of objects to detect.")
@click.option(
    "--model-type",
    type=click.Choice(["yolo", "dino"], case_sensitive=False),
    prompt=False,
    help="The model type to use for detection.",
)
@click.option("--api-name", default="/detect", help="The API endpoint to use.")
@click.option("--list", "-l", is_flag=True, help="List available models for object detection.")
@click.help_option("--help", "-h")
@click.pass_context
def detect_objects(ctx:click.RichContext, image_filename, model_src, objects, model_type, api_name, list) -> None:
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
    try:


        if list:
            print("Available Object Detection Models:")
            print("- yolo: YOLOWorld")
            print("- dino: Grounding DINO")

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

    except Exception as e:
        getconsole().print(f"[red]Error: {str(e)}[/red]")
        if ctx.ensure_object(dict).get("VERBOSE", False):
            import traceback
            getconsole().print(traceback.format_exc())
        ctx.exit(1)


@cli.command("depth")
@click.argument("image_filename", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/sense/", help="The model source URL.")
@click.option("--api-name", default="/depth", help="The API endpoint to use.")
@click.option("--list", "-l", is_flag=True, help="List available models for depth estimation.")
@click.help_option("--help", "-h")
@click.pass_context
def estimate_depth(ctx, image_filename, model_src, api_name, list) -> None:
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
    verbose = ctx.ensure_object(dict).get("VERBOSE", False)
    dry_run = ctx.ensure_object(dict).get("DRY_RUN", False)

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
    result = agent.act(image=image, api_name=api_name)
    result.pil.show()


@cli.command("segment")
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
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.help_option("--help", "-h")
def segment(image_filename, model_src, segment_type, segment_input, api_name, list, verbose) -> None:
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
    if list:
        print("Available Segmentation Models:")
        print("- Segment Anything(SAM2)")

    if image_filename is None:
        print("Error: Missing argument 'IMAGE_FILENAME'. Specify an image filename")
        print("Run 'mbodied sense segment --help' for assistance")

    if segment_type is None:
        segment_type = click.prompt(
            "Input type - bounding box or pixel coordinates",
            type=click.Choice(["bbox", "coords"], case_sensitive=False),
        )

    if segment_input is None:
        segment_input = click.prompt("Segment input data - x1,y1,x2,y2 (for bbox) or u,v (for coords)")

    if verbose:
        print(f"Running segmentation agent on {image_filename} to segment {segment_input}")

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





@cli.command("motion", no_args_is_help=True)
@click.argument("instruction")
@click.argument("image_path", required=False)
@click.option("--model-src", default="https://api.mbodi.ai/community-models/", help="The model source URL.")
@click.help_option("--help", "-h")
@click.pass_context
def openvla_motion(ctx, instruction, image_filename, model_src, verbose, dry_run) -> None:
    """Run the OpenVlaAgent to generate robot motion based on instruction and image.
    
    Example command:
        mbodied motion openvla resources/xarm.jpeg --instruction "move forward" -- --unnorm-key "your_custom_key"
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
        - [model_src]: The model source for the OpenVlaAgent.
        - Custom Inputs (Following the `--` separator)

    Outputs:
        - [Motion Response]: HandControl object containing pose and grasp information.

    Loaded as API: [https://api.mbodi.ai/community-models/](https://api.mbodi.ai/community-models/)

    API Endpoint: https://api.mbodi.ai/community-models/
    """

    if image_filename is None:
        image = click.prompt("Image Path")

    if instruction is None:
        instruction = click.prompt("Instruction")

    if not TYPE_CHECKING:
        Image = smart_import("mbodied.types.sense", attribute="Image")
        OpenVlaAgent = smart_import("mbodied.agents.motion", attribute="OpenVlaAgent")
    else:
        from mbodied.types.sense import Image
        from mbodied.agents.motion import OpenVlaAgent

    image = Image(image_filename, size=(224, 224))
    agent = OpenVlaAgent(model_src=model_src)
    motion_response = agent.act(instruction=instruction, image=image, unnorm_key=unnorm_key)
    console = getconsole()
    console.print("Motion Response:", motion_response)


@cli.command("auto", no_args_is_help=True)
@click.argument("instruction", required=True)
@click.option("--image-path", default=None, help="Optional path to the image file (for sense tasks).")
@click.option("--model-src", default="openai", help="Model source for agent")
@click.option("--api-key", default=None, help="API key for the remote model, if applicable.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.help_option("--help", "-h")
def auto(instruction, image_path, model_src, api_key, params, verbose) -> None:
    r"""Dynamically select and run the correct agent based on the task.

    Example command:
        mbodied auto  "Tell me a math joke."

    Response:
        Why was the equal sign so humble?
        Because it knew it wasn't less than or greater than anyone else!

    Example command:
        mbodied auto 

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
    console = getconsole()
    try:


        from mbodied.agents.auto import TaskTypes
        if TYPE_CHECKING:
            from mbodied.agents.auto import AutoAgent
            from mbodied.types.sense import Image
        else:
            AutoAgent = smart_import("mbodied.agents.auto", attribute="AutoAgent")
            Image = smart_import("mbodied.types.sense", attribute="Image")
        if params:
            try:
                options = json.loads(params)
            except json.JSONDecodeError:
                print("Invalid JSON format for parameters.")
                return

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

        print(f"Response: {response}")

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback
        console.print(traceback.format_exc())


@cli.command("version")
def version():
    """Display the version of mbodied."""
    __version__ = smart_import("mbodied", attribute="__version__")
    print(f"mbodied version: {__version__}")


if __name__ == "__main__":
    cli()
