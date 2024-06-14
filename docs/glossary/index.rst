Glossary
================

- **Agent**: A unit of intelligent computation that takes in an Observation and outputs an Action. This can involve multiple sub-agents.

- **Backend**: The system that embodied agents query. This typically involves a vision-language model or other specially purposed models.

- **Control**: An atomic action that is “handed off” to other processes outside the scope of consideration. An example is HandControl, which includes x, y, z, roll, pitch, yaw, and grasp. This is a motion control used to manage the position, orientation, and hand-openness of an end-effector. Typically, this is passed to lower-level hardware interfaces or libraries.

- **Simulation**: A SimplerEnv environment takes the Action from the Control and applies it to a robot over a specified number of timesteps, demonstrating the simulation. It can also be used to benchmark the accuracy of your agents (models) within the simulation environment.