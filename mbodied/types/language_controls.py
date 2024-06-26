"""This module defines varuous language controls.

1. CommandControl: Basic commands such as 'start', 'stop', 'restart', etc.
2. MobileControl: Commands for mobile movement like 'move forward', 'move backward', etc.
3. HeadControl: Commands for head movements such as 'look up', 'look down', etc.
4. HandControl: Commands for hand movements like 'open hand', 'move hand forward', etc.
5. MobileSingleArmControl: Comprehensive commands for a mobile single-arm robot, including movements and rotations.

The dynamically created Enums are:
- LangControl: Combines CommandControl, MobileControl, HeadControl, and HandControl.
- MobileSingleArmLangControl: Based on MobileSingleArmControl.

Example usage:
    execute_command(LangControl.START)  # Output: Starting the system...
    execute_command(LangControl.MOVE_FORWARD)  # Output: Moving forward...

    command_str = "move forward"
    command = get_command_from_string(command_str)
    execute_command(command)  # Execute: Moving forward...
"""

from enum import Enum


def create_enum_from_list(name, string_list) -> Enum:
    """Dynamically create an Enum type from a list of strings.

    Args:
        name (str): The name of the Enum class.
        string_list (list[str]): The list of strings to be converted into Enum members.

    Returns:
        Enum: A dynamically created Enum type with members based on the string list.
    """
    return Enum(value=name, names=[(item.replace(" ", "_").upper(), item) for item in string_list])


def language_control_to_list(enum: Enum) -> list[str]:
    """Convert an Enum type to a list of its values. So it's easier to pass i.e. as prompt."""
    return [command.value for command in enum]


def get_command_from_string(command_str) -> Enum:
    """Get the Enum member corresponding to a given command string."""
    try:
        return LangControl[command_str.replace(" ", "_").upper()]
    except KeyError:
        return None


CommandControl = ["start", "stop", "restart", "pause", "resume", "sleep", "go home"]

MobileControl = ["move forward", "move backward", "turn left", "turn right"]

HeadControl = ["look up", "look down", "look left", "look right"]

HandControl = [
    "open hand",
    "close hand",
    "move hand forward",
    "move hand backward",
    "move hand left",
    "move hand right",
    "move hand up",
    "move hand down",
]

MobileSingleArmControl = [
    "move home",
    "move forward",
    "move backward",
    "turn left",
    "turn right",
    "look up",
    "look down",
    "look left",
    "look right",
    "move hand forward",
    "move hand backward",
    "move hand left",
    "move hand right",
    "move hand up",
    "move hand down",
    "roll hand clock wise",
    "roll hand counter clock wise",
    "pitch hand up",
    "pitch hand down",
    "rotate waist left",
    "yaw hand counter clock wise",
    "yaw hand clock wise",
    "yaw hand counter clock wise a lot",
    "yaw hand clock wise a lot",
    "yaw hand counter clock wise a little",
    "yaw hand clock wise a little",
    "rotate waist right",
    "open hand",
    "close hand",
    "move forward a little",
    "move backward a little",
    "turn left a little",
    "turn right a little",
    "look up a little",
    "look down a little",
    "look left a little",
    "look right a little",
    "move hand forward a little",
    "move hand backward a little",
    "move hand left a little",
    "move hand right a little",
    "move hand up a little",
    "move hand down a little",
    "roll hand clock wise a little",
    "roll hand counter clock wise a little",
    "pitch hand up a little",
    "pitch hand down a little",
    "rotate waist left a little",
    "rotate waist right a little",
    "move forward a lot",
    "move backward a lot",
    "turn left a lot",
    "turn right a lot",
    "look up a lot",
    "look down a lot",
    "look left a lot",
    "look right a lot",
    "move hand forward a lot",
    "move hand backward a lot",
    "move hand left a lot",
    "move hand right a lot",
    "move hand up a lot",
    "move hand down a lot",
    "roll hand clock wise a lot",
    "roll hand counter clock wise a lot",
    "pitch hand up a lot",
    "pitch hand down a lot",
    "rotate waist left a lot",
    "rotate waist right a lot",
    "sleep",
]

LangControl: Enum = create_enum_from_list("LangControl", CommandControl + MobileControl + HeadControl + HandControl)

MobileSingleArmLangControl: Enum = create_enum_from_list("MobileSingleArmLangControl", MobileSingleArmControl)
