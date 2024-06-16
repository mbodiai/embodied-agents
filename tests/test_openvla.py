import pytest
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from unittest.mock import patch, MagicMock

from mbodied_agents.types.controls import Motion, HandControl, Pose6D, JointControl
from mbodied_agents.agents.motion.motor_agent import MotorAgent
from mbodied_agents.types.sense.vision import Image
from mbodied_agents.agents.motion.openvla_agent import OpenVlaAgent


@pytest.fixture
def mock_image():
    # Mock the Image class and its attributes/methods as needed
    mock_image = MagicMock()
    mock_image.pil = MagicMock()
    mock_image.base64 = "mock_base64_string"
    return mock_image


@pytest.fixture
def openvla_agent_local():
    with patch("mbodied_agents.agents.motion.openvla_agent.AutoProcessor.from_pretrained") as mock_processor, \
            patch("mbodied_agents.agents.motion.openvla_agent.AutoModelForVision2Seq.from_pretrained") as mock_model:

        mock_processor_instance = MagicMock()
        mock_model_instance = MagicMock()

        mock_processor.return_value = mock_processor_instance
        mock_model.return_value = mock_model_instance

        # Ensure predict_action returns the mock response directly
        mock_model_instance.to.return_value.predict_action.return_value = "[1 2 3 4 5 6 7]"

        agent = OpenVlaAgent(run_local=True, device="cpu")
        return agent


def test_local_act(openvla_agent_local, mock_image):
    instruction = "move forward"
    response = openvla_agent_local.act(instruction, mock_image)
    assert isinstance(response, list)
    assert len(response) == 1
    assert isinstance(response[0], HandControl)
    assert response[0].pose.x == 1.0
    assert response[0].pose.y == 2.0
    assert response[0].pose.z == 3.0
    assert response[0].pose.roll == 4.0
    assert response[0].pose.pitch == 5.0
    assert response[0].pose.yaw == 6.0
    assert response[0].grasp.value == 7.0


if __name__ == "__main__":
    pytest.main()
