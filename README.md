# Mbodied Agents

<img src="assets/logo.jpeg" alt="Mbodied Agents Logo" style="width: 200px;">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MacOS | Python 3.12|3.11|3.10](https://github.com/MbodiAI/opensource/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/MbodiAI/opensource/actions/workflows/macos.yml)
[![Ubuntu](https://github.com/MbodiAI/opensource/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/MbodiAI/opensource/actions/workflows/ubuntu.yml)
[![PyPI Version](https://img.shields.io/pypi/v/mbodied-agents.svg)](https://pypi.python.org/pypi/mbodied-agents)
[![Example Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DAQkuuEYj8demiuJS1_10FIyTI78Yzh4?usp=sharing)

Welcome to **Mbodied Agents**, a toolkit and platform for integrating various state-of-the-art transformers in robotics for any embodiments. Mbodied Agents has a consistent interface for calling different AI models, handling multimodal data, using/creating datasets trained on different robots, and more!

<img src="assets/architecture.jpg" alt="Architecture Diagram" style="width: 650px;">

Each time you interact with a robot, the data is automatically recorded into a dataset, which can be augmented and used for model training. To learn more about how to process the dataset, augment the data, or train/finetune a foundation model, please fill out this [form](https://forms.gle/rv5rovK93dLucma37).

<img src="assets/demo_gif.gif" alt="Demo GIF" style="width: 625px;">

We welcome any questions, issues, or PRs!

Please join our [Discord](https://discord.gg/RNzf3RCxRJ) for interesting discussions! **⭐ Give us a star on GitHub if you like us!**

## What is Mbodied Agents for

Mbodied Agents simplifies the integration of advanced AI models in robotics. It offers a unified platform for controlling various robots using state-of-the-art transformers and multimodal data processing. This toolkit enables experimentation with AI models, dataset collection and augmentation, and model training or finetuning for specific tasks. The goal is to develop intelligent, adaptable robots that learn from interactions and perform complex tasks in dynamic environments.

### Example use case

Imagine you are working on a project to develop a home assistant robot capable of performing household chores. Using Mbodied Agents, you can leverage pre-trained AI models to control the robot's actions based on voice commands and visual inputs.

For instance, you can train the robot to understand natural language instructions, and perform tasks such as fetching items, cleaning, etc. By continuously recording the robot's interactions and augmenting the collected data to train models, you can improve its performance over time, making it more efficient and reliable in various scenarios.

## Overview

Mbodied Agents offers the following features:

- **Configurability** : Define your desired Observation and Action spaces and read data into the format that works best for your system.
- **Natural Language Control** : Use verbal prompts to correct a cognitive agent's actions and calibrate its behavior to a new environment.
- **Modularity** : Easily swap out different backends, transformers, and hardware interfaces. For even better results, run multiple agents in separate threads.
- **Validation** : Ensure that your data is in the correct format and that your actions are within the correct bounds before sending them to the robot.

### Support Matrix

If you would like to integrate a new backend, sense, or motion control, it is very easy to do so. Please refer to the [contributing guide](CONTRIBUTING.md) for more information.

- OpenAI
- Anthropic
- Mbodi (Coming Soon)
- HuggingFace (Coming Soon)
- Gemini (Coming Soon)

### In Beta

For access (or just to say hey 😊), don't hesitate to fill out this [form](https://forms.gle/rv5rovK93dLucma37) or reach out to us at info@mbodi.ai.

- **Conductor**: A service for processing and managing datasets, and automatically training your models on your own data.
- **Conductor Dashboard**: See how GPT-4o, Claude Opus, or your custom models are performing on your datasets and open benchmarks.
- **Data Augmentation**: Build invariance to different environments by augmenting your dataset with Mbodi's diffusion-based data augmentation to achieve better generalization.
- **Mbodied SVLM**: A new Spatial Vision Language Model trained specifically for spatial reasoning and robotics control.

### Idea

The core idea behind Mbodied Agents is end-to-end continual learning. We believe that the best way to train a robot is to have it learn from its own experiences.

## Installation

`pip install mbodied-agents`

## Dev Environment Setup

1. Clone this repo:

   ```console
   git clone https://github.com/MbodiAI/mbodied-agents.git
   ```

2. Install system dependencies:

   ```console
   source install.bash
   ```

3. Then for each new terminal, run:

   ```console
   hatch shell
   ```

## Getting Started

Please refer to [examples/simple_robot_agent.py](examples/simple_robot_agent.py) or use the Colab below to get started.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DAQkuuEYj8demiuJS1_10FIyTI78Yzh4?usp=sharing)

To run `simple_robot_agent.py`, if you want to use OpenAI, for example, as your backend:

```shell
export OPENAI_API_KEY=your_api_key
python examples/simple_robot_agent.py --backend=openai
```

Upcoming feature: if you want to use `mbodi` as your backend:

```shell
python examples/simple_robot_agent.py --backend=mbodi
```

## Directory Structure

```
├─ assets/ ............. Images, icons, and other static assets
├─ examples/ ........... Example scripts and usage demonstrations
├─ resources/ .......... Additional resources for examples
├─ src/
│  └─ mbodied/
│     ├─ agents/ ....... Modules for robot agents
│     │  ├─ backends/ .. Backend implementations for different services for agents
│     │  ├─ language/ .. Language based agents modules
│     │  └─ sense/ ..... Sensory, e.g. audio, processing modules
│     ├─ base/ ......... Base classes and core infra modules
│     ├─ data/ ......... Data handling and processing
│     ├─ hardware/ ..... Hardware interface and interaction
│     └─ types/ ........ Common types and definitions
└─ tests/ .............. Unit tests
```

## Glossary

- **Agent**: A unit of intelligent computation that takes in an `Observation` and outputs an `Action`. This can involve multiple sub-agents.

- **Backend**: The system that embodied agents query. This typically involves a vision-language model or other specially purposed models.

- **Control**: An atomic action that is “handed off” to other processes outside the scope of consideration. An example is HandControl, which includes x, y, z, roll, pitch, yaw, and grasp. This is a motion control used to manage the position, orientation, and hand-openness of an end-effector. Typically, this is passed to lower-level hardware interfaces or libraries.

## Building Blocks

### Sample

The Sample class is a base model for serializing, recording, and manipulating arbitrary data. It is designed to be extensible, flexible, and strongly typed. The Sample class supports any JSON API out of the box and can represent arbitrary action and observation spaces in robotics. It integrates seamlessly with H5, Gym, Arrow, PyTorch, DSPY, numpy, and HuggingFace.

#### Creating a Sample

To create a Sample, you can wrap your object in Sample(). By doing so, you automatically get the following functionalities:

   - Gym Space: Create a new gym environment.
   - Flattened List/Array/Tensor: Plug the flattened data into a machine learning model.
   - HuggingFace Dataset: Utilize semantic search capabilities.
   - Pydantic BaseModel: Ensure reliable and quick JSON validation.

Here is an example of creating a Sample and using its methods:

```python
# Creating a Sample instance
sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)

# Flattening the Sample instance
flat_list = sample.flatten()
print(flat_list) # Output: [1, 2, 3, 4, 5]

# Generating a simplified JSON schema
schema = sample.schema()
print(schema)
# Output: {'type': 'object', 'properties': {'x': {'type': 'number'}, 'y': {'type': 'number'}, 'z': {'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}}, 'extra_field': {'type': 'number'}}}

# Unflattening a list into a Sample instance
unflattened_sample = Sample.unflatten(flat_list, schema)
print(unflattened_sample) # Output: Sample(x=1, y=2, z={'a': 3, 'b': 4}, extra_field=5)
```

#### Serialization and Deserialization with Pydantic

The Sample class leverages Pydantic's powerful features for serialization and deserialization, allowing you to easily convert between Sample instances and JSON.

To serialize or deserialize a Sample instance with JSON:

``` python
# Serialize the Sample instance to JSON
sample = Sample(x=1, y=2, z={"a": 3, "b": 4}, extra_field=5)
json_data = sample.model_dump_json()

# Deserialize the JSON data back into a Sample instance
json_data = '{"x": 1, "y": 2, "z": {"a": 3, "b": 4}, "extra_field": 5}'
sample = Sample.model_validate(from_json(json_data))
```

#### Converting to Different Containers

``` python
# Converting to a dictionary
sample_dict = sample.to("dict")

# Converting to a NumPy array
sample_np = sample.to("np")

# Converting to a PyTorch tensor
sample_pt = sample.to("pt")

# Converting to a HuggingFace Dataset
sample_hf = sample.to("hf")
```

#### Gym Space Integration

```python
from gym import spaces

# Creating a Gym space from the Sample instance
gym_space = sample.space()
print(gym_space)
# Output: Dict('a': Box(0, 255, (1,), int64), 'b': Dict('c': Box(0, 255, (1,), int64), 'd': Box(0, 255, (1,), int64)))
```

See [sample.py](src/mbodied_agents/base/sample.py) for more details.

### Message

The [Message](src/mbodied_agents/types/message.py) class represents a single completion sample space. It can be text, image, a list of text/images, Sample, or other modality. The Message class is designed to handle various types of content and supports different roles such as user, assistant, or system.

You can create a `Message` in versatile ways. They can all be understood by mbodi's backend.

```python
Message(role="user", content="example text")
Message(role="user", content=["example text", Image("example.jpg"), Image("example2.jpg")])
Message(role="user", content=[Sample("Hello")])
```

### Backend

The [Backend](src/mbodied_agents/base/backend.py) class is an abstract base class for Backend implementations. It provides the basic structure and methods required for interacting with different backend services, such as API calls for generating completions based on given messages. See [backend directory](src/mbodied_agents/agents/backends) on how various backends are implemented.

### Cognitive Agent

The [Cognitive Agent](src/mbodied_agents/agents/language/cognitive_agent.py) is the main entry point for intelligent robot agents. It can connect to different backends or transformers of your choice. It includes methods for recording conversations, managing context, looking up messages, forgetting messages, storing context, and acting based on an instruction and an image.

Currently supported API services are OpenAI and Anthropic. Upcoming API services include Mbodi, Ollama, and HuggingFace. Stay tuned for our Mbodi backend service!

For example, to use OpenAI for your robot backend:

```python
robot_agent = CognitiveAgent(context=context_prompt, api_service="openai")
```

``context`` can be either a string or a list, for example:

```python
context_prompt = "you are a robot"
# OR
context_prompt = [
    Message(role="system", content="you are a robot"),
    Message(role="user", content=["example text", Image("example.jpg")]),
    Message(role="assistant", content="Understood."),
]
```

To execute an instruction:

```python
response = robot_agent.act(instruction, image)[0]
# You can also pass an arbituary number of text and image to the agent:
response = robot_agent.act([instruction1, image1, instruction2, image2])[0]
```

### Controls

The [controls](src/mbodied_agents/types/controls.py) module defines various motions to control a robot as Pydantic models. They are also subclassed from ``Sample``, thus possessing all the capability of ``Sample`` as mentioned above. These controls cover a range of actions, from simple joint movements to complex poses and full robot control.

### Hardware Interface

Mapping robot actions from any model to any embodiment is very easy. In our example script, we use a mock hardware interface. We also have an [XArm interface](src/mbodied_agents/hardware/xarm_interface.py) as an example.

Upcoming: a remote hardware interface with a communication protocol. This will be very convenient for controlling robots that have a computer attached, e.g., LoCoBot.

### Recorder

Dataset [Recorder](src/mbodied_agents/data/recording.py) can record your conversation and the robot's actions to a dataset as you interact with/teach the robot. You can define any observation space and action space for the Recorder.

Here's an example of recording observation, instruction, and the output HandControl (x, y, z, r, p, y, grasp).

```python
observation_space = spaces.Dict({
    'image': Image(size=(224, 224)).space(),
    'instruction': spaces.Text(1000)
})
action_space = HandControl().space()
recorder = Recorder('example_recorder', out_dir='saved_datasets', observation_space=observation_space, action_space=action_space)

# Every time robot makes a conversation or performs an action:
recorder.record(observation={'image': image, 'instruction': instruction,}, action=hand_control)
```

The dataset is saved to `./saved_datasets`. Please fill out this [form](https://forms.gle/rv5rovK93dLucma37) if you are interested in getting the dataset processed, augmented, or use it for training etc.

## Contributing

We believe in the power of collaboration and open-source development. This platform is designed to be shared, extended, and improved by the community. See the [contributing guide](CONTRIBUTING.md) for more information.

Feel free to report any issues, ask questions, ask for features, or submit PRs.
