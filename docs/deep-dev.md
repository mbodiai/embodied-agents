# Deep Dive

### The [Sample](mbodied/base/sample.py) Class

The Sample class is a base model for serializing, recording, and manipulating arbitrary data. It is designed to be extendable, flexible, and strongly typed. By wrapping your observation or action objects in the [Sample](mbodied/base/sample.py) class, you'll be able to convert to and from the following with ease:

- A Gym space for creating a new Gym environment.
- A flattened list, array, or tensor for plugging into an ML model.
- A HuggingFace dataset with semantic search capabilities.
- A Pydantic BaseModel for reliable and quick json serialization/deserialization.

To learn more about all of the possibilities with embodied agents, check out the [documentation](https://mbodi-ai-mbodied-agents.readthedocs-hosted.com/en/latest/)

### 💡 Did you know

- You can `pack` a list of `Sample`s or Dicts into a single `Sample` or `Dict` and `unpack` accordingly?
- You can `unflatten` any python structure into a `Sample` class so long you provide it with a valid json schema?

#### Creating a Sample

Creating a Sample requires just wrapping a python dictionary with the `Sample` class. Additionally, they can be made from kwargs, Gym Spaces, and Tensors to name a few.

```python
from mbodied.types.sample import Sample
# Creating a Sample instance
sample = Sample(observation=[1,2,3], action=[4,5,6])

# Flattening the Sample instance
flat_list = sample.flatten()
print(flat_list) # Output: [1, 2, 3, 4, 5, 6]

# Generating a simplified JSON schema
>>> schema = sample.schema()
{'type': 'object', 'properties': {'observation': {'type': 'array', 'items': {'type': 'integer'}}, 'action': {'type': 'array', 'items': {'type': 'integer'}}}}

# Unflattening a list into a Sample instance
Sample.unflatten(flat_list, schema)
>>> Sample(observation=[1, 2, 3], action=[4, 5, 6])
```

#### Serialization and Deserialization with Pydantic

The Sample class leverages Pydantic's powerful features for serialization and deserialization, allowing you to easily convert between Sample instances and JSON.

```python
# Serialize the Sample instance to JSON
sample = Sample(observation=[1,2,3], action=[4,5,6])
json_data = sample.model_dump_json()
print(json_data) # Output: '{"observation": [1, 2, 3], "action": [4, 5, 6]}'

# Deserialize the JSON data back into a Sample instance
json_data = '{"observation": [1, 2, 3], "action": [4, 5, 6]}'
sample = Sample.model_validate(from_json(json_data))
print(sample) # Output: Sample(observation=[1, 2, 3], action=[4, 5, 6])
```

#### Converting to Different Containers

```python
# Converting to a dictionary
sample_dict = sample.to("dict")
print(sample_dict) # Output: {'observation': [1, 2, 3], 'action': [4, 5, 6]}

# Converting to a NumPy array
sample_np = sample.to("np")
print(sample_np) # Output: array([1, 2, 3, 4, 5, 6])

# Converting to a PyTorch tensor
sample_pt = sample.to("pt")
print(sample_pt) # Output: tensor([1, 2, 3, 4, 5, 6])
```

#### Gym Space Integration

```python
gym_space = sample.space()
print(gym_space)
# Output: Dict('action': Box(-inf, inf, (3,), float64), 'observation': Box(-inf, inf, (3,), float64))
```

See [sample.py](mbodied/base/sample.py) for more details.

### Message

The [Message](mbodied/types/message.py) class represents a single completion sample space. It can be text, image, a list of text/images, Sample, or other modality. The Message class is designed to handle various types of content and supports different roles such as user, assistant, or system.

You can create a `Message` in versatile ways. They can all be understood by mbodi's backend.

```python
from mbodied.types.message import Message

Message(role="user", content="example text")
Message(role="user", content=["example text", Image("example.jpg"), Image("example2.jpg")])
Message(role="user", content=[Sample("Hello")])
```

### Backend

The [Backend](mbodied/base/backend.py) class is an abstract base class for Backend implementations. It provides the basic structure and methods required for interacting with different backend services, such as API calls for generating completions based on given messages. See [backend directory](mbodied/agents/backends) on how various backends are implemented.

### Agent

[Agent](mbodied/base/agent.py) is the base class for various agents listed below. It provides a template for creating agents that can talk to a remote backend/server and optionally record their actions and observations.

### Language Agent

The [Language Agent](mbodied/agents/language/language_agent.py) can connect to different backends or transformers of your choice. It includes methods for recording conversations, managing context, looking up messages, forgetting messages, storing context, and acting based on an instruction and an image.

Natively supports API services: OpenAI, Anthropic, vLLM, Ollama, HTTPX, or any gradio endpoints. More upcoming!

To use OpenAI for your robot backend:

```python
from mbodied.agents.language import LanguageAgent

agent = LanguageAgent(context="You are a robot agent.", model_src="openai")
```

To execute an instruction:

```python
instruction = "pick up the fork"
response = robot_agent.act(instruction, image)
```

Language Agent can connect to vLLM as well. For example, suppose you are running a vLLM server Mistral-7B on 1.2.3.4:1234. All you need to do is:

```python
agent = LanguageAgent(
    context=context,
    model_src="openai",
    model_kwargs={"api_key": "EMPTY", "base_url": "http://1.2.3.4:1234/v1"},
)
response = agent.act("Hello, how are you?", model="mistralai/Mistral-7B-Instruct-v0.3")
```

### Motor Agent

[Motor Agent](mbodied/agents/motion/motor_agent.py) is similar to Language Agent but instead of returning a string, it always returns a `Motion`. Motor Agent is generally powered by robotic transformer models, i.e. OpenVLA, RT1, Octo, etc.
Some small model, like RT1, can run on edge devices. However, some, like OpenVLA, may be challenging to run without quantization. See [OpenVLA Agent](mbodied/agents/motion/openvla_agent.py) and an [example OpenVLA server](examples/servers/gradio_example_openvla.py)

### Sensory Agent

These agents interact with the environment to collect sensor data. They always return a `SensorReading`, which can be various forms of processed sensory input such as images, depth data, or audio signals.

Currently, we have:

- [depth estimation](mbodied/agents/sense/depth_estimation_agent.py)
- [object detection](mbodied/agents/sense/object_detection_agent.py)
- [image segmentation](mbodied/agents/sense/segmentation_agent.py)
- [3D object pose estimator](mbodied/agents/sense/object_pose_estimator_3d.py)

agents that process robot's sensor information.

### Auto Agent

[Auto Agent](mbodied/agents/auto/auto_agent.py) dynamically selects and initializes the correct agent based on the task and model.

```python
from mbodied.agents.auto.auto_agent import AutoAgent

# This makes it a LanguageAgent
agent = AutoAgent(task="language", model_src="openai")
response = agent.act("What is the capital of France?")

# This makes it a motor agent: OpenVlaAgent
auto_agent = AutoAgent(task="motion-openvla", model_src="https://api.mbodi.ai/community-models/")
action = auto_agent.act("move hand forward", Image(size=(224, 224)))

# This makes it a sensory agent: DepthEstimationAgent
auto_agent = AutoAgent(task="sense-depth-estimation", model_src="https://api.mbodi.ai/sense/")
depth = auto_agent.act(image=Image(size=(224, 224)))
```

Alternatively, you can use `get_agent` method in [auto_agent](mbodied/agents/auto/auto_agent.py) as well.

```python
language_agent = get_agent(task="language", model_src="openai")
```

### Motions

The [motion_controls](mbodied/types/motion_controls.py) module defines various motions to control a robot as Pydantic models. They are also subclassed from `Sample`, thus possessing all the capability of `Sample` as mentioned above. These controls cover a range of actions, from simple joint movements to complex poses and full robot control.

### Robot

You can integrate your custom robot hardware by subclassing [Robot](mbodied/robot/robot.py) quite easily. You only need to implement `do()` function to perform actions (and some additional methods if you want to record dataset on the robot). In our examples, we use a [mock robot](mbodied/robot/sim_robot.py). We also have an [XArm robot](mbodied/robot/xarm_robot.py) as an example.

#### Recording a Dataset

Recording a dataset on a robot is very easy! All you need to do is implement the `get_observation()`, `get_state()`, and `prepare_action()` methods for your robot. After that, you can record a dataset on your robot anytime you want. See [examples/5_teach_robot_record_dataset.py](examples/5_teach_robot_record_dataset.py) and this colab: [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/15UuFbMUJGEjqJ_7I_b5EvKvLCKnAc8bB?usp=sharing) for more details.

```python
from mbodied.robots import SimRobot
from mbodied.types.motion.control import HandControl, Pose

robot = SimRobot()
robot.init_recorder(frequency_hz=5)
with robot.record("pick up the fork"):
  motion = HandControl(pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3))
  robot.do(motion)
```

### Recorder

Dataset [Recorder](mbodied/data/recording.py) is a lower level recorder to record your conversation and the robot's actions to a dataset as you interact with/teach the robot. You can define any observation space and action space for the Recorder. See [gymnasium](https://github.com/Farama-Foundation/Gymnasium) for more details about spaces.

```python
from mbodied.data.recording import Recorder
from mbodied.types.motion.control import HandControl
from mbodied.types.sense.vision import Image
from gymnasium import spaces

observation_space = spaces.Dict({
    'image': Image(size=(224, 224)).space(),
    'instruction': spaces.Text(1000)
})
action_space = HandControl().space()
recorder = Recorder('example_recorder', out_dir='saved_datasets', observation_space=observation_space, action_space=action_space)

# Every time robot makes a conversation or performs an action:
recorder.record(observation={'image': image, 'instruction': instruction,}, action=hand_control)
```

The dataset is saved to `./saved_datasets`.

### Replayer

The [Replayer](mbodied/data/replaying.py) class is designed to process and manage data stored in HDF5 files generated by `Recorder`. It provides a variety of functionalities, including reading samples, generating statistics, extracting unique items, and converting datasets for use with HuggingFace. The Replayer also supports saving specific images during processing and offers a command-line interface for various operations.

Example for iterating through a dataset from Recorder with Replayer:

```python
from mbodied.data.replaying import Replayer

replayer = Replayer(path=str("path/to/dataset.h5"))
for observation, action in replayer:
   ...
```
