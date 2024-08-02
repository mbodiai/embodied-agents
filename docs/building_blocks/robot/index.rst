Robot
=====

You can integrate your custom robot hardware by subclassing `Robot <mbodied/robot/robot.py>`_ quite easily. You only need to implement the ``do()`` function to perform actions (and some additional methods if you want to record datasets on the robot). In our examples, we use a `mock robot <mbodied/robot/sim_robot.py>`_. We also have an `XArm robot <mbodied/robot/xarm_robot.py>`_ as an example.

Recording dataset on robot
--------------------------

Recording datasets on the robot is very easy using `RobotRecorder <mbodied/robot/robot_recording.py>`_. All you need to do is specify recorder arguments, and you can start and stop recording anytime you want on the robot! See `examples/5_teach_robot_record_dataset.py` for more details!

.. code-block:: python

    robot = SimRobot()
    robot_recorder = RobotRecorder(robot, record_frequency=5)

    robot_recorder.start_recording(task="pick up the fork")
    robot.do(motion1)
    robot.do(motion2)
    robot_recorder.stop_recording()
