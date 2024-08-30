# Getting Started

### Customize a Motion to fit a robot's action space.

```python
from mbodied.types.motion.control import HandControl, FullJointControl
from mbodied.types.motion import AbsoluteMotionField, RelativeMotionField

class FineGrainedHandControl(HandControl):
    comment: str = Field(None, description="A comment to voice aloud.")
    index: FullJointControl = AbsoluteMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
    thumb: FullJointControl = RelativeMotionField([0,0,0],bounds=[-3.14, 3.14], shape=(3,))
```

### Run a robotics transformer model on a robot.

```python
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
```

Example Scripts:

- [1_simple_robot_agent.py](https://github.com/mbodiai/embodied-agents/blob/main/examples/1_simple_robot_agent.py): A very simple language based cognitive agent taking instruction from user and output voice and actions.
- [2_openvla_motor_agent_example.py](https://github.com/mbodiai/embodied-agents/blob/main/examples/2_openvla_motor_agent_example.py): Run robotic transformers, i.e. OpenVLA, in several lines on the robot.
- [3_reason_plan_act_robot.py](https://github.com/mbodiai/embodied-agents/blob/main/examples/3_reason_plan_act_robot.py): Full example of language based cognitive agent and OpenVLA motor agent executing task.
- [4_language_reason_plan_act_robot.py](https://github.com/mbodiai/embodied-agents/blob/main/examples/4_language_reason_plan_act_robot.py): Full example of all languaged based cognitive and motor agent executing task.
- [5_teach_robot_record_dataset.py](https://github.com/mbodiai/embodied-agents/blob/main/examples/5_teach_robot_record_dataset.py): Example of collecting dataset on robot's action at a specific frequency by just yelling at the robot!

### Notebooks

Real Robot Hardware: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KN0JohcjHX42wABBHe-CxXP-NWJjbZts?usp=sharing)

Simulation with: [SimplerEnv](https://github.com/simpler-env/SimplerEnv.git) : [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18oiuw1yTxO5x-eT7Z8qNyWtjyYd8cECI?usp=sharing)

Run OpenVLA with embodied-agents in simulation: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1flnMrqyepGOO8J9rE6rehzaLdZPsw6lX?usp=sharing)

Record dataset on a robot: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/15UuFbMUJGEjqJ_7I_b5EvKvLCKnAc8bB?usp=sharing)
