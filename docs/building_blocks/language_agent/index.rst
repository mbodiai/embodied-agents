Language Agent
=================

Overview
^^^^^^^^^^^^

The `Language Agent <https://github.com/mbodiai/embodied-agents/blob/main/mbodied/agents/language/language_agent.py>`_ is the main entry point for intelligent robot agents. It can connect to different backends or transformers of your choice. It includes methods for recording conversations, managing context, looking up messages, forgetting messages, storing context, and acting based on an instruction and an image.

Currently supported API services are OpenAI and Anthropic. Upcoming API services include Mbodi, Ollama, and HuggingFace. Stay tuned for our Mbodi backend service!

For example, to use OpenAI for your robot backend:

.. code-block:: python

    robot_agent = LanguageAgent(context=context_prompt, model_src="openai")

``context`` can be either a string or list, for example:

.. code-block:: python

    context_prompt = "you are a robot"
    # OR
    context_prompt = [
        Message(role="system", content="you are a robot"),
        Message(role="user", content=["example text", Image("example.jpg")]),
        Message(role="assistant", content="Understood."),
    ]

to execute an instruction:

.. code-block:: python

    response = robot_agent.act(instruction, image)[0]
    # You can also pass an arbituary number of text and image to the agent:
    response = robot_agent.act([instruction1, image1, instruction2, image2])[0]

Language Agent can connect to vLLM as well. For example, suppose you are running a vLLM server Mistral-7B on 1.2.3.4:1234. All you need to do is:

.. code-block:: python

    agent = LanguageAgent(
        context=context,
        model_src="openai",
        model_kwargs={"api_key": "EMPTY", "base_url": "http://1.2.3.4:1234/v1"},
    )
    response = agent.act("Hello, how are you?", model="mistralai/Mistral-7B-Instruct-v0.3")
