Types
=======

Overview
^^^^^^^^^^

All types returned by agents are subclasses of ``Sample``, ensuring a uniform interface and behavior across different agent outputs. Here are the main types associated with agents:

Sense
^^^^^^^
Sense objects are inputs to ``SensoryAgents``. They provide raw data that the agents process into more structured forms.

**Subclasses:**

- Image: Represents visual data.
- Depth: Captures information about distances within a scene.
- Audio: Contains sound data from the environment.

SensorReading
^^^^^^^^^^^^^^^^

``SensorReading`` objects are outputs from ``SensoryAgents``, representing processed sensory data.

**Subclasses:**

- 2DObjectPoses: Positions of objects in a 2D plane.
- 2DBoundingBoxes: Bounding boxes around objects in a 2D plane.
- 3DObjectPoses: Positions of objects in a 3D space.
- 3DBoundingBoxes: Bounding boxes around objects in a 3D space.
- Embedding: A numerical representation of data, often used in machine learning.

LanguageAction
^^^^^^^^^^^^^^^^

A ``LanguageAction`` contains a list of instructions that can be executed. These instructions can be plain text commands or executable code, facilitating various language-based interactions.

Motion
^^^^^^^^^

``Motion`` objects define movement parameters, including **bounds** (limits of movement), **motion type** (e.g., linear, rotational), and the **reference frame** (context or coordinate system for the motion).

