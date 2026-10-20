"""Exercise MiniMax model selection at the SDK's serialized HTTP boundary."""

import asyncio
import json

import httpx
import pytest
from openai import AsyncOpenAI

from mbodied.agents.backends import MiniMaxBackend
from mbodied.agents.language import LanguageAgent
from mbodied.types.message import Message
from mbodied.types.sense.vision import Image


@pytest.fixture
def transport_capture():
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((request, payload))
        model = payload["model"]
        if request.url.path.endswith("/messages"):
            message = {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [{"type": "text", "text": "Ready"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
            if payload.get("stream"):
                events = [
                    ("message_start", {"message": {**message, "content": []}}),
                    ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
                    ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "Ready"}}),
                    ("content_block_stop", {"index": 0}),
                    ("message_stop", {}),
                ]
                content = "".join(
                    f"event: {kind}\ndata: {json.dumps({'type': kind, **body})}\n\n" for kind, body in events
                )
                return httpx.Response(200, text=content, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=message)
        message = {
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Ready"}}],
        }
        if payload.get("stream"):
            message["object"] = "chat.completion.chunk"
            message["choices"] = [{"index": 0, "delta": {"content": "Ready"}}]
            return httpx.Response(
                200,
                text=f"data: {json.dumps(message)}\n\ndata: [DONE]\n\n",
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=message)

    return requests, httpx.MockTransport(respond)


@pytest.mark.parametrize("api_format", ["openai", "anthropic"])
@pytest.mark.parametrize("region", ["global_en", "cn_zh", "global", "cn"])
@pytest.mark.parametrize("model", [None, "MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
@pytest.mark.parametrize("streaming", [False, True])
def test_language_agent_serialized_model_and_history(transport_capture, api_format, region, model, streaming):
    requests, transport = transport_capture
    with httpx.Client(transport=transport) as client:
        agent = LanguageAgent(
            model_src="minimax",
            api_key="test-only",
            context=[Message(role="user", content="Remember this"), Message(role="assistant", content="Remembered")],
            model_kwargs={"api_format": api_format, "region": region, "http_client": client},
        )
        image = Image(size=(2, 2))
        kwargs = {} if model is None else {"model": model}
        if streaming:
            assert "".join(agent.act_and_stream("Describe", image=image, **kwargs)) == "Ready"
        else:
            assert agent.act("Describe", image=image, **kwargs) == "Ready"
        request, payload = requests[-1]
        assert payload["model"] == (model or "MiniMax-M3.1-Flash")
        assert request.url.host == ("api.minimaxi.com" if region in ("cn", "cn_zh") else "api.minimax.io")
        assert request.url.path == ("/v1/chat/completions" if api_format == "openai" else "/anthropic/v1/messages")
        assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "user"]
        assert payload["messages"][0]["content"] == [{"type": "text", "text": "Remember this"}]
        assert payload["messages"][1]["content"] == [{"type": "text", "text": "Remembered"}]
        assert payload["messages"][-1]["content"][0] == {"type": "text", "text": "Describe"}
        image_block = payload["messages"][-1]["content"][1]
        if api_format == "openai":
            assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")
        else:
            assert image_block["source"]["data"] == image.base64
        assert "service_tier" not in payload
        assert "thinking" not in payload
        assert "reasoning_effort" not in payload
        assert "output_config" not in payload


@pytest.mark.parametrize("api_format", ["openai", "anthropic"])
@pytest.mark.parametrize("streaming", [False, True])
def test_custom_endpoint_and_explicit_effort(transport_capture, api_format, streaming):
    requests, transport = transport_capture
    with httpx.Client(transport=transport) as client:
        backend = MiniMaxBackend(
            api_key="test-only",
            api_format=api_format,
            region="custom",
            base_url="https://custom.invalid/api",
            http_client=client,
        )
        kwargs = (
            {"reasoning_effort": "high"}
            if api_format == "openai"
            else {"extra_body": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}}
        )
        call = backend.stream if streaming else backend.predict
        result = call(Message("Hello"), **kwargs)
        assert ("".join(result) if streaming else result) == "Ready"
        request, payload = requests[-1]
        assert request.url.host == "custom.invalid"
        assert payload["model"] == "MiniMax-M3.1-Flash"
        if api_format == "openai":
            assert payload["reasoning_effort"] == "high"
        else:
            assert payload["thinking"] == {"type": "adaptive"}
            assert payload["output_config"] == {"effort": "high"}


@pytest.mark.parametrize("api_format", ["openai", "anthropic"])
def test_explicit_m3_disabled_thinking(transport_capture, api_format):
    requests, transport = transport_capture
    with httpx.Client(transport=transport) as client:
        backend = MiniMaxBackend(api_key="test-only", api_format=api_format, http_client=client)
        assert (
            backend.predict(Message("Hello"), model="MiniMax-M3", extra_body={"thinking": {"type": "disabled"}})
            == "Ready"
        )
    assert requests[-1][1]["thinking"] == {"type": "disabled"}
    assert requests[-1][1]["model"] == "MiniMax-M3"


@pytest.mark.parametrize("model", [None, "MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
def test_direct_async_stream_model(transport_capture, model):
    requests, transport = transport_capture

    async def run():
        backend = MiniMaxBackend(api_key="test-only", client=object())
        async with httpx.AsyncClient(transport=transport) as client:
            backend.aclient = AsyncOpenAI(api_key="test-only", base_url=backend.base_url, http_client=client)
            kwargs = {} if model is None else {"model": model}
            return "".join([part async for part in backend.astream(Message("Hello"), **kwargs)])

    assert asyncio.run(run()) == "Ready"
    assert requests[-1][1]["model"] == (model or "MiniMax-M3.1-Flash")
