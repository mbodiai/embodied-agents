from mbodied.EKF.extended_kalman_filter import ExtendedKalmanFilter
from mbodied.EKF.world import world_to_vector
from mbodied.types.sense.world import World, WorldObject, BBox2D, BBox3D, Pose, PixelCoords
from mbodied.types.sense.vision import Image
from mbodied.types.motion.control import HandControl
from mbodied.EKF.trajectory import HandControlVector
import numpy as np

def test_ekf():
    world = World(
        image=Image(path="resources/color_image.png"),
        objects=[
            WorldObject(
                name="box",
                bbox_2d=BBox2D(10, 20, 50, 60),
                bbox_3d=BBox3D(0, 0, 0, 1, 1, 1),
                pose=Pose(x=1.0, y=2.0, z=3.0, roll=0.1, pitch=0.2, yaw=0.3),
                pixel_coords=PixelCoords(100, 150)
            ),
            WorldObject(
                name="sphere",
                bbox_2d=BBox2D(15, 25, 55, 65),
                bbox_3d=BBox3D(1, 1, 1, 2, 2, 2),
                pose=Pose(x=4.0, y=5.0, z=6.0, roll=0.4, pitch=0.5, yaw=0.6),
                pixel_coords=PixelCoords(110, 160)
            )
        ]
    )

    hand = HandControl.unflatten([20, 30, 10, 40, 50, 70, 1])

    world_vector = world_to_vector(world)
    motion_vector = HandControlVector(hand).to_vector()
    state_vector = np.concatenate([world_vector, motion_vector])

    state_dim = state_vector.size
    control_dim = motion_vector.size
    observation_dim = state_vector.size
    Q = np.eye(state_dim) * 0.1
    R = np.eye(observation_dim) * 0.5
    
    ekf = ExtendedKalmanFilter(state_dim, control_dim, observation_dim, initial_state=state_vector, process_noise_cov=Q, measurement_noise_cov=R)

    num_iterations = 10
    innovations = []
    state_estimates = []
    robot_poses = []

    for i in range(num_iterations):
        print(f"\n--- Iteration {i + 1} ---")

        control_input = np.random.randint(1, 100, size=7)

        ekf.predict(control_input)

        predicted_state = ekf.get_state().flatten()
        print(f"Predicted State: {predicted_state}")

        noise = np.random.normal(0, 0.1, observation_dim)
        simulated_observation = predicted_state + noise
        print(f"Simulated Observation: {simulated_observation}")

        ekf.update(simulated_observation)

        updated_state = ekf.get_state().flatten()
        print(f"Updated State Estimate: {updated_state}")

        predicted_measurement = ekf.get_measurement_prediction()
        innovation = simulated_observation - predicted_measurement
        innovations.append(innovation)
        print(f"Innovation: {innovation}")

        state_estimates.append(updated_state)

        robot_pose = {
            "position": updated_state[-7:-4],
            "orientation": updated_state[-4:-1]
        }
        robot_poses.append(robot_pose)

    print(f"Robot POSE: {robot_poses}")

    return state_estimates, innovations

test_ekf()
