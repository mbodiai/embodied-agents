import pytest
from mbodied_agents.types.controls import HandControl, Pose6D, JointControl
from mbodied_agents.hardware.sim_interface import SimInterface


@pytest.fixture
def sim_interface():
    """Create a SimInterface instance for testing."""
    return SimInterface()


def test_initial_pose(sim_interface):
    """Test that the initial pose is set correctly."""
    expected_pose = [0, 0, 0, 0, 0, 0, 0]
    assert sim_interface.get_pose() == expected_pose


def test_do_motion(sim_interface):
    """Test that the do method updates the current position correctly."""
    motion = HandControl(
        pose=Pose6D(x=0.1, y=0.2, z=0.3, roll=0.1, pitch=0.2, yaw=0.3),
        grasp=JointControl(value=0.5)
    )
    sim_interface.do(motion)
    expected_pose = [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.5]
    assert sim_interface.get_pose() == expected_pose

    # Perform another motion to ensure the position updates cumulatively
    another_motion = HandControl(
        pose=Pose6D(x=-0.1, y=-0.2, z=-0.3, roll=-0.1, pitch=-0.2, yaw=-0.3),
        grasp=JointControl(value=0.0)
    )
    sim_interface.do(another_motion)
    expected_pose = [0, 0, 0, 0, 0, 0, 0.0]
    assert sim_interface.get_pose() == expected_pose

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
