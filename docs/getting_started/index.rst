Getting Started
==================

Run a robotics transformer model on a robot
-------------------------------------------

.. code-block:: python

    import os
    from mbodied.agents import LanguageAgent
    from mbodied.agents.motion import OpenVlaAgent
    from mbodied.agents.sense.audio import AudioAgent
    from mbodied.robots import SimRobot

    cognition = LanguageAgent(
      context="You are an embodied planner that responds with a python list of strings and nothing else.",
      api_key=os.getenv("OPENAI_API_KEY"),
      model_src="openai",
      recorder="auto",
    )
    audio = AudioAgent(use_pyaudio=False, api_key=os.getenv("OPENAI_API_KEY")) # pyaudio is buggy on mac
    motion = OpenVlaAgent(model_src="https://api.mbodi.ai/community-models/")

    # Subclass and override do() and capture() methods.
    robot = SimRobot()

    instruction = audio.listen()
    plan = cognition.act(instruction, robot.capture())

    for step in plan.strip('[]').strip().split(','):
      print("\nMotor agent is executing step: ", step, "\n")
      for _ in range(10):
        hand_control = motion.act(step, robot.capture())
        robot.do(hand_control)

Example Scripts
---------------

- `examples/simple_robot_agent.py <examples/simple_robot_agent.py>`_: A very simple language based cognitive agent taking instruction from user and output actions.
- `examples/simple_robot_agent_layered.py <examples/simple_robot_agent_layered.py>`_: Full example of layered language based cognitive agent and motor agent executing task.
- `examples/motor_example_openvla.py <examples/motor_example_openvla.py>`_: Run robotic transformers, i.e. OpenVLA, in several lines on the robot.
- `examples/reason_plan_act_robot.py <examples/reason_plan_act_robot.py>`_: Full example of layered language based cognitive agent and OpenVLA motor agent executing task.

Notebooks
---------

Real Robot Hardware: `Open In Colab <https://colab.research.google.com/drive/1qFoo2h4tD9LYtUwkWtO4XtVAwcKxALn_?usp=sharing>`_

Simulation with SimplerEnv: `Open In Colab <https://colab.research.google.com/drive/1gJlfEvsODZWGn3rK8Nx4A0kLnLzJtJG_?usp=sharing>`_

MotorAgent with OpenVLA: `examples/motor_example_openvla.py <examples/motor_example_openvla.py>`_
