from typing import List, Callable, Optional, Any
from mbodied.agents.language import LanguageAgent

class TreeNode:
    """
    A class representing a node in the Tree of Thought.
    Reference: https://arxiv.org/pdf/2305.10601
    
    Attributes:
    -----------
    state : str
        The state or thought represented by this node.
    children : List[TreeNode]
        The list of child nodes, representing further thoughts or decisions.
    """
    def __init__(self, state: str) -> None:
        """
        Initialize the TreeNode with the given state.
        
        Parameters:
        -----------
        state : str
            The thought or state at this node.
        """
        self.state = state
        self.children: List[TreeNode] = []

    def add_child(self, child_state: str) -> 'TreeNode':
        """
        Add a child node with the given state to this node.
        
        Parameters:
        -----------
        child_state : str
            The state or thought to add as a child.
        
        Returns:
        --------
        TreeNode
            The newly added child node.
        """
        child_node = TreeNode(child_state)
        self.children.append(child_node)
        return child_node

class TreeOfThought:
    """
    A class that generates and manages a Tree of Thought (ToT) using a LanguageAgent.
    
    Attributes:
    -----------
    language_agent : LanguageAgent
        The language agent used to generate decisions or text.
    depth : int
        The maximum depth of the tree to explore (i.e., how many steps forward to think).
    root : Optional[TreeNode]
        The root node of the decision tree, representing the initial thought.
    """

    def __init__(self, language_agent: 'LanguageAgent', depth: int = 3) -> None:
        """
        Initialize the Tree of Thought with a given LanguageAgent.
        
        Parameters:
        -----------
        language_agent : LanguageAgent
            The agent used to generate decisions or thoughts.
        depth : int, optional
            The maximum depth of the thought tree to expand (default is 3).
        """
        self.language_agent = language_agent
        self.depth = depth
        self.root: Optional[TreeNode] = None
        self.parse_function = None

    def generate_thoughts(
        self, 
        prompt: str, 
        parse_function: Optional[Callable[[str], List[str]]] = None, 
        max_depth: Optional[int] = None,
        image = None
    ) -> TreeNode:
        """
        Generate a Tree of Thought based on the given prompt.
        
        Parameters:
        -----------
        prompt : str
            The initial prompt or command used to generate the first thought in the tree.
        parse_function : Optional[Callable[[str], List[str]]], optional
            A user-defined function to process the output of the LanguageAgent.
            This function takes the raw output as input and returns a list of thoughts or child states.
        max_depth : Optional[int], optional
            The maximum depth to expand the tree (overrides self.depth if provided).
        
        Returns:
        --------
        TreeNode
            The root node of the generated Tree of Thought.
        """
        if max_depth is None:
            max_depth = self.depth

        initial_response: str = self.language_agent.act(prompt, image=image)

        if parse_function:
            self.parse_function = parse_function
            initial_thoughts: List[str] = self.parse_function(initial_response)
        else:
            initial_thoughts = [initial_response.strip()]

        self.root = TreeNode(initial_thoughts[0])
        
        self.expand(self.root, initial_thoughts[1:], max_depth)
        
        return self.root

    def expand(self, node: TreeNode, thoughts: List[str], depth: int) -> None:
        """
        Recursively expand the Tree of Thought from the given node.
        
        Parameters:
        -----------
        node : TreeNode
            The current node being expanded.
        thoughts : List[str]
            The list of thoughts or child states to expand from.
        depth : int
            The remaining depth to expand (i.e., how many steps ahead to think).
        """
        if depth == 0 or not thoughts:
            return
        
        for thought in thoughts:
            print(thought)
            child: TreeNode = node.add_child(thought)
            
            new_response: str = self.language_agent.act(thought)
            
            if new_response:
                if hasattr(self, 'parse_function') and self.parse_function:
                    new_thoughts: List[str] = self.parse_function(new_response)
                else:
                    new_thoughts = [new_response.strip()]
                print(new_thoughts)
            else:
                new_thoughts = new_response
            
            self.expand(child, new_thoughts, depth - 1)
    
    def traverse_tree(self, node: Optional[TreeNode] = None, depth: int = 0) -> None:
        """
        Traverse the Tree of Thought from the given node and print the decisions at each level.
        
        Parameters:
        -----------
        node : Optional[TreeNode], optional
            The starting node for traversal (default is None, which uses the root).
        depth : int, optional
            The current depth in the tree (used for printing indentation).
        """
        if node is None:
            node = self.root
        
        print(" " * depth + f"Thought: {node.state}")
        
        for child in node.children:
            self.traverse_tree(child, depth + 1)


# Example usage:

# if __name__ == "__main__":
#     from mbodied.agents.language import LanguageAgent

#     context = [
#         {"role": "user", "content": "What should the robot do next?"},
#         {"role": "assistant", "content": "Understood!"}
#     ]
#     language_agent = LanguageAgent(context=context, api_key="your_api_key", model_src="openai")

#     def parse_json_output(response: str) -> List[str]:
#         response = response.replace("```json", "").replace("```", "").strip()
#         return json.loads(response)

#     tot = TreeOfThought(language_agent, depth=3)
    
#     root = tot.generate_thoughts("What should the robot do next?", parse_function=parse_json_output)
    
#     tot.traverse_tree(root)
