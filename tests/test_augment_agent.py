# Copyright 2024 mbodi ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from unittest.mock import patch, MagicMock
from mbodied.types.sense.vision import Image
from mbodied.agents.sense.augment_agent import AugmentAgent
from mbodied.agents.backends.gradio_backend import GradioBackend


@pytest.fixture
def mock_gradio_backend():
    with patch.object(GradioBackend, "__init__", lambda x, model_src=None, **kwargs: None):
        with patch.object(GradioBackend, "predict", return_value="resources/xarm.jpeg"):
            yield GradioBackend(endpoint="http://1.2.3.4:1234")


@pytest.fixture
def augment_agent(mock_gradio_backend):
    agent = AugmentAgent(model_src="http://1.2.3.4:1234/")
    agent.actor = mock_gradio_backend
    return agent


def test_augment_agent_initialization(augment_agent):
    assert isinstance(augment_agent, AugmentAgent)
    assert augment_agent.actor is not None


def test_augment_agent_act(augment_agent):
    instruction = "change lighting"
    mock_image = Image("resources/xarm.jpeg", size=(224, 224))
    result = augment_agent.act(instruction=instruction, image=mock_image)
    assert isinstance(result, Image)
