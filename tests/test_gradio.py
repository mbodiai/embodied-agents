import pytest
from unittest.mock import patch, MagicMock
from mbodied.agents.backends import GradioBackend


def test_init():
    with patch("mbodied.agents.backends.gradio_backend.Client") as mock_client:
        backend = GradioBackend(remote_server="http://fake-server.com")
        mock_client.assert_called_with(src="http://fake-server.com")
        assert backend.remote_server == "http://fake-server.com"


def test_act():
    test_args = ("input1", "input2")
    test_kwargs = {"key": "value"}
    expected_result = "predicted value"

    with patch("mbodied.agents.backends.gradio_backend.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.predict.return_value = expected_result
        backend = GradioBackend(remote_server="http://fake-server.com")

        result = backend.act(*test_args, **test_kwargs)

        mock_instance.predict.assert_called_once_with(*test_args, **test_kwargs)
        assert result == expected_result


def test_submit():
    test_args = ("input1", "input2")
    test_kwargs = {"key": "value"}
    api_name = "/test_api"
    result_callbacks = None
    expected_job = MagicMock()

    with patch("mbodied.agents.backends.gradio_backend.Client") as mock_client:
        mock_instance = mock_client.return_value
        mock_instance.submit.return_value = expected_job
        backend = GradioBackend(remote_server="http://fake-server.com")

        job = backend.submit(*test_args, api_name=api_name, result_callbacks=result_callbacks, **test_kwargs)

        mock_instance.submit.assert_called_once_with(
            api_name=api_name, result_callbacks=result_callbacks, *test_args, **test_kwargs
        )
        assert job == expected_job
        assert not job.done.called  # Ensuring that job.done was not called as it's non-blocking


if __name__ == "__main__":
    pytest.main()
