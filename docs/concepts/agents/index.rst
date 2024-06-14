Agents
=========

Overview
^^^^^^^^^^

Agents are the primary entities in our system designed to perform specific tasks and return a standardized type called a ``Sample``. Each agent has an act() method that is pivotal to its functionality. This method records data to either an HDF5 file or a HuggingFace dataset, depending on the configuration and requirements of the system. Additionally, agents can be deployed using a client-server pattern, allowing for the invocation of remote actors, which facilitates distributed processing and scalability. Below are the different types of agents and their unique characteristics:

Sensory Agents
^^^^^^^^^^^^^^

These agents interact with the environment to collect sensory data. They always return a ``SensorReading``, which can be various forms of processed sensory input such as **images**, **depth data**, or **audio signals**.

Motor Agents
^^^^^^^^^^^^

These agents are responsible for generating motion or controlling movements. They always return a ``Motion`` object, which encapsulates details like the type of motion, its bounds, and the reference frame in which it operates.

Language Agents
^^^^^^^^^^^^^^^^^^

Designed to handle and generate natural language, these agents always return a ``LanguageAction``. This could be a series of instructions or even executable code, enabling complex interactions through language.

