import pytest

from mbodied.types.language.control import (
    LangControl,
    MobileSingleArmLangControl,
    language_control_to_list,
    get_command_from_string,
)


def test_enum_members():
    assert LangControl.START.name == "START"
    assert LangControl.START.value == "start"
    assert LangControl.MOVE_FORWARD.name == "MOVE_FORWARD"
    assert LangControl.MOVE_FORWARD.value == "move forward"


def test_enum_iteration():
    expected_values = {
        "START": "start",
        "STOP": "stop",
        "RESTART": "restart",
        "PAUSE": "pause",
        "RESUME": "resume",
        "SLEEP": "sleep",
        "GO_HOME": "go home",
        "MOVE_FORWARD": "move forward",
        "MOVE_BACKWARD": "move backward",
        "TURN_LEFT": "turn left",
        "TURN_RIGHT": "turn right",
        "LOOK_UP": "look up",
        "LOOK_DOWN": "look down",
        "LOOK_LEFT": "look left",
        "LOOK_RIGHT": "look right",
        "OPEN_HAND": "open hand",
        "CLOSE_HAND": "close hand",
        "MOVE_HAND_FORWARD": "move hand forward",
        "MOVE_HAND_BACKWARD": "move hand backward",
        "MOVE_HAND_LEFT": "move hand left",
        "MOVE_HAND_RIGHT": "move hand right",
        "MOVE_HAND_UP": "move hand up",
        "MOVE_HAND_DOWN": "move hand down",
    }

    for command in LangControl:
        assert command.name in expected_values
        assert command.value == expected_values[command.name]


def test_get_command_from_string():
    def get_command_from_string(command_str):
        try:
            return LangControl[command_str.replace(" ", "_").upper()]
        except KeyError:
            return None

    # Test mapping string input to Enum members
    assert get_command_from_string("start") == LangControl.START
    assert get_command_from_string("move forward") == LangControl.MOVE_FORWARD
    assert get_command_from_string("look up") == LangControl.LOOK_UP
    assert get_command_from_string("invalid command") is None


def test_mobile_single_arm_lang_control():
    # Test iterating over MobileSingleArmLangControl members
    expected_values = {
        "MOVE_HOME": "move home",
        "MOVE_FORWARD": "move forward",
        "MOVE_BACKWARD": "move backward",
        "TURN_LEFT": "turn left",
        "TURN_RIGHT": "turn right",
        "LOOK_UP": "look up",
        "LOOK_DOWN": "look down",
        "LOOK_LEFT": "look left",
        "LOOK_RIGHT": "look right",
        "MOVE_HAND_FORWARD": "move hand forward",
        "MOVE_HAND_BACKWARD": "move hand backward",
        "MOVE_HAND_LEFT": "move hand left",
        "MOVE_HAND_RIGHT": "move hand right",
        "MOVE_HAND_UP": "move hand up",
        "MOVE_HAND_DOWN": "move hand down",
        "ROLL_HAND_CLOCK_WISE": "roll hand clock wise",
        "ROLL_HAND_COUNTER_CLOCK_WISE": "roll hand counter clock wise",
        "PITCH_HAND_UP": "pitch hand up",
        "PITCH_HAND_DOWN": "pitch hand down",
        "ROTATE_WAIST_LEFT": "rotate waist left",
        "YAW_HAND_COUNTER_CLOCK_WISE": "yaw hand counter clock wise",
        "YAW_HAND_CLOCK_WISE": "yaw hand clock wise",
        "YAW_HAND_COUNTER_CLOCK_WISE_A_LOT": "yaw hand counter clock wise a lot",
        "YAW_HAND_CLOCK_WISE_A_LOT": "yaw hand clock wise a lot",
        "YAW_HAND_COUNTER_CLOCK_WISE_A_LITTLE": "yaw hand counter clock wise a little",
        "YAW_HAND_CLOCK_WISE_A_LITTLE": "yaw hand clock wise a little",
        "ROTATE_WAIST_RIGHT": "rotate waist right",
        "OPEN_HAND": "open hand",
        "CLOSE_HAND": "close hand",
        "MOVE_FORWARD_A_LITTLE": "move forward a little",
        "MOVE_BACKWARD_A_LITTLE": "move backward a little",
        "TURN_LEFT_A_LITTLE": "turn left a little",
        "TURN_RIGHT_A_LITTLE": "turn right a little",
        "LOOK_UP_A_LITTLE": "look up a little",
        "LOOK_DOWN_A_LITTLE": "look down a little",
        "LOOK_LEFT_A_LITTLE": "look left a little",
        "LOOK_RIGHT_A_LITTLE": "look right a little",
        "MOVE_HAND_FORWARD_A_LITTLE": "move hand forward a little",
        "MOVE_HAND_BACKWARD_A_LITTLE": "move hand backward a little",
        "MOVE_HAND_LEFT_A_LITTLE": "move hand left a little",
        "MOVE_HAND_RIGHT_A_LITTLE": "move hand right a little",
        "MOVE_HAND_UP_A_LITTLE": "move hand up a little",
        "MOVE_HAND_DOWN_A_LITTLE": "move hand down a little",
        "ROLL_HAND_CLOCK_WISE_A_LITTLE": "roll hand clock wise a little",
        "ROLL_HAND_COUNTER_CLOCK_WISE_A_LITTLE": "roll hand counter clock wise a little",
        "PITCH_HAND_UP_A_LITTLE": "pitch hand up a little",
        "PITCH_HAND_DOWN_A_LITTLE": "pitch hand down a little",
        "ROTATE_WAIST_LEFT_A_LITTLE": "rotate waist left a little",
        "ROTATE_WAIST_RIGHT_A_LITTLE": "rotate waist right a little",
        "MOVE_FORWARD_A_LOT": "move forward a lot",
        "MOVE_BACKWARD_A_LOT": "move backward a lot",
        "TURN_LEFT_A_LOT": "turn left a lot",
        "TURN_RIGHT_A_LOT": "turn right a lot",
        "LOOK_UP_A_LOT": "look up a lot",
        "LOOK_DOWN_A_LOT": "look down a lot",
        "LOOK_LEFT_A_LOT": "look left a lot",
        "LOOK_RIGHT_A_LOT": "look right a lot",
        "MOVE_HAND_FORWARD_A_LOT": "move hand forward a lot",
        "MOVE_HAND_BACKWARD_A_LOT": "move hand backward a lot",
        "MOVE_HAND_LEFT_A_LOT": "move hand left a lot",
        "MOVE_HAND_RIGHT_A_LOT": "move hand right a lot",
        "MOVE_HAND_UP_A_LOT": "move hand up a lot",
        "MOVE_HAND_DOWN_A_LOT": "move hand down a lot",
        "ROLL_HAND_CLOCK_WISE_A_LOT": "roll hand clock wise a lot",
        "ROLL_HAND_COUNTER_CLOCK_WISE_A_LOT": "roll hand counter clock wise a lot",
        "PITCH_HAND_UP_A_LOT": "pitch hand up a lot",
        "PITCH_HAND_DOWN_A_LOT": "pitch hand down a lot",
        "ROTATE_WAIST_LEFT_A_LOT": "rotate waist left a lot",
        "ROTATE_WAIST_RIGHT_A_LOT": "rotate waist right a lot",
        "SLEEP": "sleep",
    }

    for command in MobileSingleArmLangControl:
        assert command.name in expected_values
        assert command.value == expected_values[command.name]


def test_get_command_from_string():
    # Test mapping string input to Enum members
    assert get_command_from_string("start") == LangControl.START
    assert get_command_from_string("move forward") == LangControl.MOVE_FORWARD
    assert get_command_from_string("look up") == LangControl.LOOK_UP
    assert get_command_from_string("invalid command") is None


def test_language_control_to_list():
    # Test converting LangControl Enum to a list of its values
    expected_list = [
        "start",
        "stop",
        "restart",
        "pause",
        "resume",
        "sleep",
        "go home",
        "move forward",
        "move backward",
        "turn left",
        "turn right",
        "look up",
        "look down",
        "look left",
        "look right",
        "open hand",
        "close hand",
        "move hand forward",
        "move hand backward",
        "move hand left",
        "move hand right",
        "move hand up",
        "move hand down",
    ]
    assert language_control_to_list(LangControl) == expected_list


if __name__ == "__main__":
    pytest.main()
