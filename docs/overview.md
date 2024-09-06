# embodied agents

**embodied agents** is a toolkit for integrating large multi-modal models into existing robot stacks with just a few lines of code. It provides consistency, reliability, scalability and is configurable to any observation and action space.


## Overview

This repository is broken down into 3 main components: **Agents**, **Data**, and **Hardware**. Inspired by the efficiency of the central nervous system, each component is broken down into 3 meta-modalities: **Language**, **Motion**, and **Sense**. Each agent has an `act` method that can be overridden and satisfies:

- **Language Agents** always return a string.
- **Motor Agents** always return a `Motion`.
- **Sensory Agents** always return a `SensorReading`.

For convenience, we also provide **AutoAgent** which dynamically initializes the right agent for the specified task. See [API Reference](#auto-agent) below for more.

A call to `act` or `async_act` can perform local or remote inference synchronously or asynchronously. Remote execution can be performed with [Gradio](https://www.gradio.app/docs/python-client/introduction), [httpx](https://www.python-httpx.org/), or different LLM clients. Validation is performed with [Pydantic](https://docs.pydantic.dev/latest/).

<img src="https://raw.githubusercontent.com/mbodiai/embodied-agents/main/assets/architecture.jpg" alt="Architecture Diagram" style="width: 700px;">

- Language Agents natively support OpenAI, Anthropic, Ollama, vLLM, Gradio, etc
- Motor Agents natively support OpenVLA, RT1(upcoming)
- Sensory Agents support Depth Anything, YOLO, Segment Anything 2

Jump to [getting started](#getting-started) to get up and running on [real hardware](https://colab.research.google.com/drive/1KN0JohcjHX42wABBHe-CxXP-NWJjbZts?usp=sharing) or [simulation](https://colab.research.google.com/drive/1gJlfEvsODZWGn3rK8Nx4A0kLnLzJtJG_?usp=sharing). Be sure to join our [Discord](https://discord.gg/BPQ7FEGxNb) for 🥇-winning discussions :)

**⭐ Give us a star on GitHub if you like us!**

### Motivation

There is a signifcant barrier to entry for running SOTA models in robotics. It is currently unrealistic to run state-of-the-art AI models on edge devices for responsive, real-time applications. Furthermore,
the complexity of integrating multiple models across different modalities is a significant barrier to entry for many researchers,
hobbyists, and developers. This library aims to address these challenges by providing a simple, extensible, and efficient way to
integrate large models into existing robot stacks.


### Goals

Facillitate data-collection and sharing among roboticists. This requires reducing much of the complexities involved with setting up inference endpoints, converting between different model formats, and collecting and storing new datasets for future availibility.

We aim to achieve this by:

1. Providing simple, Python-first abstrations that are modular, extensible and applicable to a wide range of tasks.
2. Providing endpoints, weights, and interactive Gradio playgrounds for easy access to state-of-the-art models.
3. Ensuring that this library is observation and action-space agnostic, allowing it to be used with any robot stack.

Beyond just improved robustness and consistency, this architecture makes asynchronous and remote agent execution exceedingly simple. In particular we demonstrate how responsive natural language interactions can be achieved in under 30 lines of Python code.


### Limitations

_Embodied Agents are not yet capable of learning from in-context experience_:

- Frameworks for advanced RAG techniques are clumsy at best for OOD embodied applications however that may improve.
- Amount of data required for fine-tuning is still prohibitively large and expensive to collect.
- Online RL is still in its infancy and not yet practical for most applications.

### Scope

- This library is intended to be used for research and prototyping.
- This library is still experimental and under active development. Breaking changes may occur although they will be avoided as much as possible. Feel free to report issues!

### Features

- Extensible, user-friendly python SDK with explicit typing and modularity
- Asynchronous and remote thread-safe agent execution for maximal responsiveness and scalability.
- Full-compatiblity with HuggingFace Spaces, Datasets, Gymnasium Spaces, Ollama, and any OpenAI-compatible api.
- Automatic dataset-recording and optionally uploads dataset to huggingface hub.

### Endpoints

- [OpenVLA](https://api.mbodi.ai/community-models/)
- [Embodied AI Playground](https://api.mbodi.ai/benchmark/)
- [3D Object Pose Detection](https://api.mbodi.ai/3d-object-pose-detection/)

### Support Matrix

- Closed: OpenAI, Anthropic
- Open Weights: OpenVLA, Idefics2, Llava-1.6-Mistral, Phi-3-vision-128k-instruct
- All gradio endpoints hosted on HuggingFace spaces.

### Roadmap

- [x] OpenVLA Motor Agent
- [x] Automatic dataset recording on Robot
- [x] Yolo, SAM2, DepthAnything Sensory Agents
- [x] Auto Agent
- [ ] ROS integration
- [ ] More Motor Agents, i.e. RT1
- [ ] More device support, i.e. OpenCV camera
- [ ] Fine-tuning Scripts
