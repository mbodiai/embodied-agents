import pytest
from unittest.mock import MagicMock, patch
from mbodied.agents.motion.rt_pali.rt_pali_agent import RtPaliAgent, RtPaliMotion

@pytest.fixture
def mocked_vla_model():
    """Fixture to mock the VLAModel used in RtPaliAgent."""
    with patch('mbodied.agents.motion.rt_pali.rt_pali_agent.VLAModel') as MockModel:
        mock_model = MockModel.return_value
        mock_model.load_from_checkpoint.return_value = mock_model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        yield mock_model

@pytest.fixture
def image():
    """Fixture for an example image input."""
    return MagicMock(name='image')

@pytest.fixture
def task_instruction():
    """Fixture for an example task instruction input."""
    return "example task instruction"

@pytest.fixture
def action_map():
    """Fixture for the expected action map returned by the VLAModel."""
    return {
        'terminated': False,
        'x': 0.5,
        'y': -0.5,
        'z': 0.1,
        'roll': 0.0,
        'pitch': 0.1,
        'yaw': -0.1,
        'grasp': True
    }

def test_act(mocked_vla_model, image, task_instruction, action_map):
    """Test the act method of RtPaliAgent."""
    # Mock the generate_action_map method to return the fixture action_map
    mocked_vla_model.generate_action_map.return_value = action_map

    # Initialize RtPaliAgent
    agent = RtPaliAgent()

    # Perform the act method
    motion = agent.act(image, task_instruction)

    # Assert that the generated motion matches expected values from action_map
    assert isinstance(motion, RtPaliMotion)
    assert motion.terminated == action_map['terminated']
    assert motion.x == action_map['x']
    assert motion.y == action_map['y']
    assert motion.z == action_map['z']
    assert motion.roll == action_map['roll']
    assert motion.pitch == action_map['pitch']
    assert motion.yaw == action_map['yaw']
    assert motion.grasp == action_map['grasp']

    # Verify that the mocked methods were called as expected
    mocked_vla_model.load_from_checkpoint.assert_called()
    mocked_vla_model.to.assert_called_with('cuda')
    mocked_vla_model.eval.assert_called()
    mocked_vla_model.generate_action_map.assert_called_once_with(image, task_instruction)
