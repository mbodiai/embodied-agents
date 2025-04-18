from unittest import mock
from mbodied.agents.language.language_agent import LanguageAgent
from mbodied.tree.tree_of_thought import TreeOfThought, ThoughtNode

@mock.patch("mbodied.agents.language.language_agent.LanguageAgent.act", side_effect=[
    "['Pick up the remote', 'Place the remote where the fork is', 'Pick up the fork', 'Place the fork where the remote was']",
    "['move forward', 'grasp remote']",
    "['move forward', 'grasp remote']",
    "[]",
    "['move to fork location', 'place remote']",
    "['place remote']",
    "[]",
    "['grasp fork']",
    "[]"
    "['move to remotes original location', 'place fork']",
    "['place fork']",
    "[]"

])
def test_generate_thoughts(mock_act):
    
    language_agent = mock.Mock()

    language_agent.act.side_effect = [
        "['Pick up the remote', 'Place the remote where the fork is', 'Pick up the fork', 'Place the fork where the remote was']",
        "['move forward', 'grasp remote']",
        "['move forward', 'grasp remote']",
        "[]",
        "['move to fork location', 'place remote']",
        "['place remote']",
        "[]",
        "['grasp fork']",
        "[]"
        "['move to remotes original location', 'place fork']",
        "['place fork']",
        "[]"
    ]

    language_agent = LanguageAgent()
    tot = TreeOfThought(language_agent=language_agent, max_depth=2)
    tot.generate_thoughts(instruction="pick up spoon")
    assert tot.root is not None, "Expected root node"
    assert tot.root.thought == "Start", f"Expected Start but got {tot.root.thought}"
    tot.traverse()
    tot.get_actions()
    
def test_add_child():
        parent_node = ThoughtNode("Parent")
        child_node = ThoughtNode("Child")
        parent_node.add_child(child_node)

        assert parent_node.children[0] == child_node, f"Expected {parent_node} to contain {child_node}"

def test_is_leaf():
        leaf_node = ThoughtNode("Leaf")
        assert leaf_node.is_leaf(), "Expected to be a leaf node"

        non_leaf_node = ThoughtNode("Non-leaf")
        non_leaf_node.add_child(ThoughtNode("Child"))
        assert not non_leaf_node.is_leaf(), "Expected to have children nodes"

