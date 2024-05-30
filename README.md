# Mbodied Agents </br> Bringing the Power of Generative AI to Robotics

<img src="assets/logo.jpeg" alt="Mbodied Agents Logo" style="width: 200px;">

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![MacOS | Python 3.12|3.11|3.10](https://github.com/MbodiAI/opensource/actions/workflows/macos.yml/badge.svg?branch=main)](https://github.com/MbodiAI/opensource/actions/workflows/macos.yml)
[![Ubuntu](https://github.com/MbodiAI/opensource/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/MbodiAI/opensource/actions/workflows/ubuntu.yml)
[![Example Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DAQkuuEYj8demiuJS1_10FIyTI78Yzh4?usp=sharing)

Welcome to **Mbodied Agents**! This repository is a toolkit for integrating various state-of-the-art transformers in robotics. We wanted to make a consistent interface for calling different AI models, handling multimodal data, and using/creating datasets trained on different robots. Be sure to checkout the example section for how to automatically fine-tune a foundational model in as little as 10 lines of code.  With that in mind, Mbodied Agents offers the following features:

- **Configurability** : Define your desired Observation and Action spaces and read data into the format that works best for your system.
- **Modularity** : Easily swap out different backends, transformers, and hardware interfaces. For even better results, run multiple agents in separate threads.
- **Validation** : Ensure that your data is in the correct format and that your actions are within the correct bounds.
- **FAISS Indexing** : Use FAISS to index your robot's recent memory and perform RAG rather than pollute its context.
- **Automatic Dataset Creation** : Record your robot's interactions and automatically and save them to a huggingface dataset for continual or offline training.


## Support Matrix

If you would like to integrate a new backend, it is very easy to do so. Please refer to the [contributing guide](CONTRIBUTING.md) for more information.

- OpenAI
- Anthropic
- Mbodi (Coming Soon)
- HuggingFace (Coming Soon)

## In Beta

- **Mbodi Backend** : We are currently working on a backend that will allow you to use Mbodi's models for your robot. This will include a vision-language model, a 3D image segmentation model, and a diffusion-based data augmentation model.

Please fill out this [form](https://forms.gle/rv5rovK93dLucma37) or reach out to us at info@mbodi.ai for access.


## Roadmap

- **Data Augmentation** : Build invariance to different environments by augmenting your dataset with Mbodi's diffusion-based data augmentation.
- **Observability** : See how GPT4o, Claude Opus, or other custom models are performing on various datasets and benchmarks.
- **Few Shot Prompting** : Use verbal or visual prompts to correct your robot's actions and calibrate its behavior to a new environment.


<img src="assets/architecture.jpg" alt="Architecture Diagram" style="width: 650px;">


<img src="assets/demo_gif.gif" alt="Demo GIF" style="width: 625px;">


We welcome any questions, issues, or PRs! Refer to the Contributing section below for more details.

Please join our [Discord](https://discord.gg/RNzf3RCxRJ) for interesting discussions!

**⭐ Give us a star on GitHub if you like us!**

## Installation

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

## Get Started

Please refer to [examples/simple_robot_agent.py](examples/simple_robot_agent.py) or use the Colab below to get started.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1DAQkuuEYj8demiuJS1_10FIyTI78Yzh4?usp=sharing)

To run `simple_robot_agent.py`, if you want to use OpenAI, for example, as your backend:

```shell
export OPENAI_API_KEY=your_api_key
python examples/simple_robot_agent.py --backend=openai
```

Upcoming feature:
if you want to use `mbodi` as your backend:

```shell
python examples/simple_robot_agent.py --backend=mbodi
```

## Glossary

- **Agent**: A unit of intelligent computation that takes in an `Observation` and outputs an `Action`. This can involve multiple sub-agents.

- **Backend**: The system that embodied agents query. This typically involves a vision-language model or other specially purposed models.

- **Control**: An atomic action that is “handed off” to other processes outside the scope of consideration. An example is HandControl, which includes x, y, z, roll, pitch, yaw, and grasp. This is a motion control used to manage the position, orientation, and hand-openness of an end-effector. Typically, this is passed to lower-level hardware interfaces or libraries.

## Details

### Cognitive Agent

The Cognitive Agent is the main entry point for intelligent robot agents. It can connect to different backends or transformers of your choice.

For example, to use OpenAI for your robot backend. Currently supported API services are OpenAI and Anthropic. Upcoming API services include Mbodi, Ollama, and HuggingFace.

Stay tuned for our Mbodi backend service!

```python
robot_agent = CognitiveAgent(context=context_prompt, api_service="openai")
```

To execute an instruction:

```python
response = robot_agent.act(instruction, observation)[0]
```

You can also pass an arbituary number of text and image to the agent:

```python
response = robot_agent.act([instruction1, image1, instruction2, image2])[0]
``` 

### Hardware Interface

Mapping robot actions from any model to any embodiment is very easy.

In our example script, we use a mock hardware interface. We also have an XArm interface as an example at [src/mbodied/hardware/xarm_interface.py)](src/mbodied/hardware/xarm_interface.py).

Upcoming: a remote hardware interface with a communication protocol. This will be very convenient for controlling robots that have a computer attached, e.g., LoCoBot.

### Dataset Recording

To record your conversation and the robot's actions to a dataset as you interact with/teach the robot.

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

## About Mbodi AI

Mbodi AI is an open-source robotics and AI platform designed to support end-to-end robotics applications involving artificial intelligence, data handling and augmentation, human-user interaction, and much more!
