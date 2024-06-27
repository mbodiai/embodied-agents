Motor Agent
==============

Overview
^^^^^^^^
`Motor Agent <mbodied/agents/motion/motor_agent.py>`_ is similar to Language Agent but instead of returning a string, it always returns a ``Motion``.
Motor Agent is generally powered by robotic transformer models, i.e., OpenVLA, RT1, Octo, etc.
Some small models, like RT1, can run on edge devices. However, some, like OpenVLA, are too large to run on edge devices.
See `OpenVLA Agent <mbodied/agents/motion/openvla_agent.py>`_ and an `example OpenVLA server <mbodied/agents/motion/openvla_example_server.py>`_

.. code-block:: python

    motor_agent = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/") # OpenVLA model
    hand_control = motor_agent.act("move left", image)
    hardware_interface.do(hand_control)
