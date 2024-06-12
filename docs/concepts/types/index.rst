Types
=======

All of the following are subclasses of ``Sample``.

**Sense**

Input to a ``SensoryAgent``

- Subclasses
    - Image
    - Depth
    - Audio

**SensorReading**

Output of a ``SensoryAgent``

- Subclasses
    - 2DObjectPoses
    - 2DBoundingBoxes
    - 3DObjectPoses
    - 3DBoundingBoxes
    - Embedding

**LanguageAction**

Has a list of instructions
    - Can be code

**Motion**

Always has bounds, motion type, reference frame.