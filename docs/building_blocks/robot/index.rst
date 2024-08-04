Robot
=====

You can integrate your custom robot hardware by subclassing :py:class:`Robot <mbodied.robot.robot.Robot>` quite easily. You only need to implement the ``do()`` function to perform actions (and some additional methods if you want to record a dataset on the robot). In our examples, we use a :py:class:`mock robot <mbodied.robot.sim_robot.SimRobot>`. We also have an :py:class:`XArm robot <mbodied.robot.xarm_robot.XArmRobot>` as an example.

Recording to a Dataset on a Robot
---------------------------------

Recording a dataset on a robot is very easy! All you need to do is implement the ``get_observation()``, ``get_state()``, and ``prepare_action()`` methods for your robot. Then you can record a dataset on your robot anytime you want. See ``examples/5_teach_robot_record_dataset.py`` for more details!

.. code-block:: python

    from mbodied.robots import SimRobot
    from mbodied.types.motion.control import HandControl, Pose

    robot = SimRobot()
    robot.init_recorder(frequency_hz=5)
    with robot.record("pick up the fork"):
        motion = HandControl(pose=Pose(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3))
        robot.do(motion)
