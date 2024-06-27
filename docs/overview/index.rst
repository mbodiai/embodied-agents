Overview
========

.. image:: ../../assets/architecture.jpg
   :alt: Architecture Diagram
   :width: 700px

Motivation
----------

It is currently unrealistic to run state-of-the-art AI models on edge devices for responsive, real-time applications. Furthermore, the complexity of integrating multiple models across different modalities is a significant barrier to entry for many researchers, hobbyists, and developers. This library aims to address these challenges by providing a simple, extensible, and efficient way to integrate large models into existing robot stacks.

Goals
-----

Facilitate data-collection and sharing among roboticists by reducing much of the complexities involved with setting up inference endpoints, converting between different model formats, and collecting and storing new datasets for future availability.

We aim to achieve this by:

1. Providing simple, Python-first abstractions that are modular, extensible and applicable to a wide range of tasks.
2. Providing endpoints, weights, and interactive Gradio playgrounds for easy access to state-of-the-art models.
3. Ensuring that this library is observation and action-space agnostic, allowing it to be used with any robot stack.

Beyond just improved robustness and consistency, this architecture makes asynchronous and remote agent execution exceedingly simple. In particular, we demonstrate how responsive natural language interactions can be achieved in under 10 lines of Python code.

Scope
-----

- This library is intended to be used for research and prototyping.
- This library is still experimental and under active development. Breaking changes may occur although they will be avoided as much as possible. Feel free to report issues!

Limitations
-----------

*Agents are not yet capable of learning from experience*:

- Frameworks for advanced RAG techniques are clumsy at best for OOD embodied applications, however, that may improve.
- The amount of data required for fine-tuning is still prohibitively large and expensive to collect.
- Online RL is still in its infancy and not yet practical for most applications.

Features
--------

- User-friendly python SDK with explicit typing and modularity.
- Asynchronous and remote thread-safe agent execution for maximal responsiveness and scalability.
- Full-compatibility with HuggingFace Spaces, Datasets, Gymnasium Spaces, Ollama, and any OpenAI-compatible API.
- Automatic dataset-recording and optionally uploads dataset to the HuggingFace hub.

Example Use Case
----------------

**Local Thread**

- Audio agent listens for a keyword.
- YOLO agent processes camera input and produces bounding boxes.
- Classical MPC module ensures commands don't violate constraints.

**API Services**

- Text-to-speech service further processes natural language input.
- GPU-accelerated 3D object pose detection.
- GPT-4o for high-level plan generation.

**Remote Thread**

- Custom proprietary model continually-learning through RLHF.
