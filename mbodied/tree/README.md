# Tree of Thought - README

## Overview
This repository implements a **Tree of Thought** (ToT) framework inspired by the decision-making process in AI systems. The Tree of Thought leverages Large Language Models (LLMs) to generate, evaluate, and expand thoughts (actions or decisions) recursively. The system allows for the exploration of multiple possible actions and evaluates each step to find the best path through a tree-structured reasoning process.

For reference, see the [Tree of Thoughts: Deliberate Problem Solving with LLMs](https://arxiv.org/pdf/2305.10601).

## Table of Contents
- [Key Concepts](#key-concepts)
- [Usage](#usage)
- [Tree of Thought Components](#tree-of-thought-components)
  - [ThoughtNode](#thoughtnode)
  - [TreeOfThought](#treeofthought)
- [Embedding with PCA](#embedding-with-pca)
- [Best Path Calculation](#best-path-calculation)
- [Visualization](#visualization)


## Key Concepts
- **ThoughtNode**: A node in the tree representing a thought (or action). Each node has an evaluation score and can have child nodes representing subsequent thoughts.
- **Tree of Thought (ToT)**: A tree structure that explores various paths of decisions, where each node represents a thought, and the branches represent possible follow-up actions.
- **PCA (Principal Component Analysis)**: A method used in this framework to reduce the dimensionality of thought embeddings for performance optimization.

## Usage
To run the system, you will need to set up a **LanguageAgent** and provide a task/instruction. You can optionally pass an image to be processed along with the instructions.

### Example
```python
from mbodied.agents import LanguageAgent
from mbodied.types.sense.vision import Image
from tree_of_thought import TreeOfThought

image = Image(path="resources/color_image.png")
cognition = LanguageAgent(
  context="You are an embodied planner that responds with a python list of strings and nothing else.",
  api_key=os.getenv("OPENAI_API_KEY"),
  model_src="openai",
  recorder="auto",
)
tree_of_thought = TreeOfThought(language_agent=cognition, n_components=10, max_depth=3,)

tree_of_thought.generate_thoughts(instruction="Switch the position of the remote and the fork", image=image)
tree_of_thought.traverse()
tree_of_thought.get_actions()
```

### Output:
```
Level 0: Thought: Start Evaluation: 0.5
    Level 1: Thought: Pick up the remote Evaluation: 0.5
        Level 2: Thought: move forward Evaluation: 8.0
        Level 3: Thought: grasp remote Evaluation: 9.0
        Level 3: Thought: lift remote Evaluation: 8.0
        Level 2: Thought: grasp remote Evaluation: 9.0
        Level 3: Thought: lift remote Evaluation: 8.0
        Level 2: Thought: lift remote Evaluation: 8.0
    Level 1: Thought: Place the remote where the fork is Evaluation: 0.5
        Level 2: Thought: move to fork location Evaluation: 8.0
        Level 3: Thought: place remote Evaluation: 9.0
        Level 2: Thought: place remote Evaluation: 9.0
    Level 1: Thought: Pick up the fork Evaluation: 0.5
        Level 2: Thought: move to fork Evaluation: 8.0
        Level 3: Thought: grasp fork Evaluation: 9.0
        Level 3: Thought: lift fork Evaluation: 8.0
        Level 2: Thought: grasp fork Evaluation: 9.0
        Level 3: Thought: lift fork Evaluation: 8.0
        Level 2: Thought: lift fork Evaluation: 8.0
    Level 1: Thought: Place the fork where the remote was Evaluation: 0.5
        Level 2: Thought: move to remote location Evaluation: 8.0
        Level 3: Thought: place fork Evaluation: 9.0
        Level 2: Thought: place fork Evaluation: 9.0
        
    Best Path:
    Action: move forward
    Action: grasp remote
    Action: lift remote
    Action: move to fork location
    Action: place remote
    Action: move to fork
    Action: grasp fork
    Action: lift fork
    Action: move to remote location
    Action: place fork
```

### The best action path can also be generated using the language_agent
```python
tree_of_thought.get_actions_with_llm()
```

### The structure of the actions in the tree
```python
Root
│
├── Action: "Pick up the remote"
│   ├── Thought: "move arm to the right" (Evaluation: 0.9)
│   ├── Thought: "lower arm" (Evaluation: 0.9)
│   ├── Thought: "grasp remote" (Evaluation: 1.0)
│   └── Thought: "lift arm" (Evaluation: 0.9)
│
├── Action: "Place the remote where the fork is"
│   ├── Thought: "lower arm" (Evaluation: 0.9)
│   ├── Thought: "grasp remote" (Evaluation: 1.0)
│   └── Thought: "lift arm" (Evaluation: 0.9)
│
├── Action: "Pick up the fork"
│   ├── Thought: "grasp fork" (Evaluation: 1.0)
│   └── Thought: "lift arm" (Evaluation: 0.9)
│
└── Action: "Place the fork where the remote was"
    ├── Thought: "release fork" (Evaluation: 1.0)
    └── Thought: "lift arm" (Evaluation: 0.9)
```

## Tree of Thought Components

# ThoughtNode

A `ThoughtNode` represents an individual decision/action in the reasoning process. It contains the following attributes:

- **thought**: The actual action or decision.
- **embedding**: A high-dimensional representation of the thought (optional).
- **evaluation**: A score representing how promising the thought is.
- **children**: A list of child nodes (follow-up actions).
- **reduced_embedding**: Embedding reduced via PCA for optimization.

### Methods

- **`add_child`**: Adds a child node to the current thought node.
- **`is_leaf`**: Checks if the node has any children.

## TreeOfThought

The `TreeOfThought` manages the tree and recursively expands on thoughts using the `LanguageAgent`. It generates embeddings for thoughts and uses PCA to reduce the dimensionality of these embeddings.

### Parameters

- **language_agent**: The agent responsible for generating new thoughts based on the input instruction.
- **n_components**: Number of PCA components used to reduce the dimensionality of embeddings.
- **max_depth**: Maximum depth for the tree exploration.
- **embed**: Flag to indicate whether embeddings should be generated for thoughts.

### Core Methods

- **`generate_thoughts`**: Initializes the thought tree by querying the `LanguageAgent` with an instruction.
- **`get_actions`**: Retrieves the best action path in the thought tree using a combination of BFS and DFS.
- **`traverse`**: Traverses and prints the structure of the thought tree.

## Embedding with PCA

Each thought can be transformed into a high-dimensional embedding using a SentenceTransformer model. To optimize performance, PCA is used to reduce the embedding dimensionality:

- **Embedding**: Captures the semantic meaning of a thought.
- **PCA Reduction**: Reduces the number of dimensions while retaining as much information as possible.

### Example

```python
thought_node = ThoughtNode("Find the optimal solution", embedding=embedding_vector, n_components=10)
```

## Thought pathfinding system

This system explores a thought tree using a combination of **Breadth-First Search (BFS)** and **Depth-First Search (DFS)** to identify the most promising path based on evaluation scores and depth.

The thought tree consists of nodes representing decisions or actions, each with an associated evaluation score. The goal of the system is to traverse the tree and find the optimal path, avoiding redundant or low-value thoughts unless they are terminal actions (leaf nodes).

- Prioritizing **Depth-First Search (DFS)** to explore the deepest nodes in the tree first, collecting evaluated thoughts from the deepest strategies before backtracking.
- Using **Breadth-First Search (BFS)** within each level to ensure all possible actions at the current depth are explored before backtracking to higher levels.
- Skipping nodes with an evaluation of 0.5 unless they are leaf nodes. Nodes with a score of 0.5` are considered neutral and are skipped unless they represent terminal actions with no further decisions (leaf nodes).

### Example of Best Path Calculation

```python
tree_of_thought.get_actions(recompute=True)
```

### Visualization

The **traverse** function enables visualization of the thought tree. It prints out the thought at each level and display the corresponding evaluation score.
```python
tree_of_thought.traverse()
```




