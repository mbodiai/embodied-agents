from mbodied.types.message import Message
from mbodied.types.sample import Sample
from typing import Optional, List
from pydantic import Field


class Thought(Sample):
    thought: List[str] = Field(
        description="The actions to be taken by the robot"
    )
    evaluation: Optional[List[float]] = Field(
        description="The evaluation of the each thought. It can be a number between 0.1 and 1.0 being 0.1 the worst and 1.0 the best."
    )

def generate_prompt(instruction: str) -> List[Message]:
    """
    Generate the prompt to send to the LLM based on the instruction.
    Parameter:
    - instruction: The instruction from the user.
    return: Context to send to the language agent.
    """
    TREE_OF_THOUGHT_PROMPT = f"""
    You are a robot command agent designed to solve complex tasks by generating, evaluating, and refining actions. Your task is to generate solutions, evaluate the quality of your thoughts, and return a response in a structured format.
    - Use the action provided e.g "turn right" to generate an ACCURATE list of next steps for the robot to follow after the action in the format: ["move forward", "turn left", ...] based on the instruction: {instruction}.
    - Generate only steps that follow the action given and not previous steps taken to get to the action while telling the robot exactly what to do.
    - Return an empty list if there is no further action required to reach the goal from the action provided.

    ### Instructions:

    1. **Understand the Action:**
    - Analyze the initial action or thought provided.
    - Break it down into smaller, manageable steps if necessary.
    - Ensure you fully comprehend the context before proceeding.

    2. **Generate New Actions:**
    - Based on the provided thought, generate a LIST of new actions or reasoning steps.
    - Ensure that the actions are logical, well-justified, and relevant to the initial problem.

    3. **Self-Evaluation:**
    - After generating each action, evaluate it for relevance, feasibility, safety, and efficiency.
    - Strictly follow the instructions below to assign an evaluation score between 1 and 10:
        - **1 to 4:** Flawed or irrelevant action.
        - **5 to 7:** Actions that can be broken into further steps e.g "Place object on the table".
        - **8 to 10:** Only actions that cannot be broken down any further e.g "move forward".

    4. **Iterate on Actions:**
    - Based on the actions generated, further reason and expand on each step.
    - Re-evaluate new actions as necessary.

    Respond in the following json schema:
    {Thought.model_json_schema()}
    """ + """Please provide the response as a JSON array e.g 
    {
        "thought": ["move forward", "turn left", "pick up apple"],
        "evaluation": [8, 9, 6]
    }
    """

    context = [
        Message(
            role="user",
            content=TREE_OF_THOUGHT_PROMPT,
        ),
        Message(role="assistant", content="Understood!"),
    ]

    return context

def create_llm_prompt(formatted_thought_tree: str, instruction: str) -> List[Message]:
    """
    Create the prompt to send to the LLM based on the formatted thought tree.
    Parameter:
    - formatted_thought_tree: The formatted thought sequence tree.
    - instruction: The instruction from the user.
    return: Context to send to the language agent.
    """
    prompt = f"""
        Given the following thought tree, pick the best path through the actions that completes the goal in the instruction {instruction}.
        The path should:
        - Not pick duplicate actions/steps.
        - Not skip important steps.
        - Prioritize steps that cannot be broken down any further e.g "move forward" over "move to table".
        - Prioritize lower-level actions as they are most likely not steps that can be broken down any further e.g "grasp object".

        Here is the thought tree:
            {formatted_thought_tree}

        DO NOT PROVIDE ANY RESPONSE EXCEPT A LIST IN THE FOLLOWING FORMAT:
            ["action1", "action2", "action3",...]
    """

    context = [
        Message(
            role="user",
            content=prompt,
        ),
        Message(role="assistant", content="Understood!"),
    ]

    return context