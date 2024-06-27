Sensory Agent
=================

These agents interact with the environment to collect sensory data. They always return a ``SensorReading``, which can be various forms of processed sensory input such as images, depth data, or audio signals.

For example, `object_pose_estimator_3d <mbodied/agents/sense/object_pose_estimator_3d.py>`_ is a sensory agent that senses objects' 3d coordinates as the robot sees.
