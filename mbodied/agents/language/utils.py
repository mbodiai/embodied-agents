import inspect
import re
from typing import Any, Callable, get_type_hints

from mbodied.types.tool import Tool


def function_to_tool(func: Callable) -> Tool:
    """Convert a Python function to a Tool object for use with LanguageAgent.

    This utility extracts function signature, type hints, and docstring to create
    a properly formatted Tool object that can be used with LanguageAgent's tool calls.

    Args:
        func: The Python function to convert

    Returns:
        Tool object ready to use with LanguageAgent

    Example:
        >>> def get_object_location(object_name: str, reference: str = "end_effector") -> Dict:
        ...     '''Get the pose of an object relative to a reference.
        ...
        ...     Args:
        ...         object_name: The name of the object whose location is being queried
        ...         reference: The reference object for the pose (default: end_effector)
        ...
        ...     Returns:
        ...         Dictionary with position and orientation
        ...     '''
        ...     pass
        >>> tool = function_to_tool(get_object_location)
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get function name and docstring
    name = func.__name__
    doc = inspect.getdoc(func) or ""

    # Get type hints
    type_hints = get_type_hints(func)

    # Extract function description from docstring
    description = doc.split("\n\n")[0].strip() if doc else name

    # Parse parameter descriptions from docstring
    param_desc = {}
    param_pattern = re.compile(r"(\w+):\s*(.*?)(?=\n\s*\w+:|$)", re.DOTALL)
    args_section_match = re.search(r"Args:(.*?)(?=\n\s*Returns:|$)", doc, re.DOTALL)

    if args_section_match:
        args_section = args_section_match.group(1)
        for param_match in param_pattern.finditer(args_section):
            param_name = param_match.group(1).strip()
            param_desc[param_name] = param_match.group(2).strip()

    # Build parameters dictionary
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # Skip self parameter for methods
        if param_name == "self":
            continue

        param_type = type_hints.get(param_name, Any)
        param_info = {"description": param_desc.get(param_name, f"Parameter {param_name}")}

        # Map Python types to JSON schema types
        if param_type == str:
            param_info["type"] = "string"
        elif param_type == int:
            param_info["type"] = "integer"
        elif param_type == float:
            param_info["type"] = "number"
        elif param_type == bool:
            param_info["type"] = "boolean"
        elif param_type == list or getattr(param_type, "__origin__", None) == list:
            param_info["type"] = "array"
        elif param_type == dict or getattr(param_type, "__origin__", None) == dict:
            param_info["type"] = "object"
        else:
            param_info["type"] = "string"

        # Handle default values
        if param.default != inspect.Parameter.empty:
            param_info["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = param_info

    # Create the Tool object
    tool_data = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }

    return Tool.model_validate(tool_data)


def main() -> None:
    """Example usage of function_to_tool utility."""
    from mbodied.agents.language import LanguageAgent

    def move_by(x: float, y: float, z: float, speed: float = 1.0) -> bool:
        """Move the robot arm by the specified position.

        Args:
            x: X-coordinate in meters, + is forward, - is backward
            y: Y-coordinate in meters, + is left, - is right
            z: Z-coordinate in meters, + is up, - is down
            speed: Movement speed (0.1-2.0)

        Returns:
            Success status
        """
        return True

    def grasp(force: float = 0.5) -> bool:
        """Close the gripper with specified force.

        Args:
            force: Grip force between 0.0 and 1.0

        Returns:
            Success status
        """
        return True

    agent = LanguageAgent(context="You are a helpful robot assistant that can call tools.", model_src="openai")

    # Convert multiple functions to tools
    tools = [function_to_tool(move_by), function_to_tool(grasp)]
    print("tools we have: ", tools)
    # Example with streaming
    response = ""
    completed_tool_calls = []

    for content_chunk, completed_tools in agent.act_and_stream(
        "move forward 0.1 meters, then move backward 0.1 meters, repeat 3 times, and then in the end, close the gripper.",
        tools=tools,
    ):
        # Process content chunks
        if content_chunk:
            response += content_chunk
            print(f"Content: {content_chunk}")

        # Process completed tools
        if completed_tools:
            for tool in completed_tools:
                completed_tool_calls.append(tool)
                print(f"\nTool Call: {tool.function.name} with args: {tool.function.arguments}")

    resp, tools = agent.act("Tell me what you just did", tools=tools)
    print(resp)


if __name__ == "__main__":
    main()
