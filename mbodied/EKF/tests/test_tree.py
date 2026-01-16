import unittest
from unittest import mock
from typing import List
from mbodied.EKF.tree_of_thought import TreeOfThought
from mbodied.agents.language import LanguageAgent

@mock.patch(
    "mbodied.agents.language.language_agent.LanguageAgent.act",
    side_effect = [
    '["move hand to the right", "close hand"]',
    '["move hand forward", "open hand"]',
    '["rotate wrist clockwise", "close hand"]',
    '["move hand up", "open hand"]',
    '["move hand down", "close hand"]',
    '["slide hand left", "hold position"]',
    '["move hand diagonally right-up", "open hand"]',
    '["move hand diagonally left-down", "close hand"]',
    '["shake hand", "open hand"]',
    '["tap surface lightly", "open hand"]',
    '["clench fist", "hold position"]',
    '["move hand back", "close hand"]',
    '["wave hand", "open hand"]',
    '["stretch fingers", "hold position"]',
    '["move hand in a circle", "close hand"]',
    None
]
)
def test_tree_of_thought(mock_act):

    language_agent = mock.Mock()
    
    language_agent.act.side_effect = [
        '["move hand to the right", "close hand"]',  
        '["move hand forward", "open hand"]',         
    ]

    def parse_json_output(response: str) -> List[str]:
        return eval(response)

    language_agent = LanguageAgent()

    tot = TreeOfThought(language_agent, depth=2)
    
    root = tot.generate_thoughts("What should the robot do next?", parse_function=parse_json_output)

    assert root.state == "move hand to the right", f"Expected root state to be 'move hand to the right', got {root.state}"

    assert len(root.children) == 1, f"Expected root to have 2 children, got {len(root.children)}"

    first_child = root.children[0]
    assert first_child.state == "close hand", f"Expected first child state to be 'close hand', got {first_child.state}"

    nested_child = root.children[0].children[0]
    assert nested_child.state == "move hand forward", f"Expected second child state to be 'move hand forward', got {nested_child.state}"

    tot.traverse_tree(root)


if __name__ == "__main__":
    unittest.main()
