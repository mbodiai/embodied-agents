from huggingface_hub import repo_exists
import pytest
from mbodied.agents.agent import Agent
from mbodied.agents.backends import GradioBackend, HttpxBackend


def test_model_src_exists():
    model_src = "openai/clip-vit-base-patch32"
    assert repo_exists(model_src) == True

    with pytest.raises(Exception):
        agent = Agent(model_src=model_src)
        agent.act("Hello, world!")


def test_space_exists():
    model_src = "https://api.mbodi.ai/community-models"
    agent = Agent(model_src=model_src)
    assert isinstance(agent.actor, GradioBackend)


def test_reka_model():
    model_src = "https://api.reka.ai/v1/chat"
    agent = Agent(model_src=model_src)
    assert isinstance(agent.actor, HttpxBackend)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
