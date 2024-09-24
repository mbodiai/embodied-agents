import asyncio
import json
import os
import sys
import time
import traceback
import uuid
from typing import Tuple

import pytest
from pydantic import BaseModel

import litellm.litellm_core_utils
import litellm.litellm_core_utils.litellm_logging
from litellm.utils import ModelResponseListIterator

sys.path.insert(0, os.path.abspath("../.."))  # Adds the parent directory to the system path
from dotenv import load_dotenv

load_dotenv()
import random

import litellm
from litellm import (
    AuthenticationError,
    BadRequestError,
    ModelResponse,
    RateLimitError,
    acompletion,
    completion,
)

litellm.logging = False
litellm.set_verbose = True
litellm.num_retries = 3
litellm.cache = None

score = 0




def logger_fn(model_call_object: dict):
    print(f"model call details: {model_call_object}")


user_message = "Hello, how are you?"
messages = [{"content": user_message, "role": "user"}]


first_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": ""},
            "finish_reason": None,  # it's null
        }
    ],
}


def validate_first_format(chunk):
    # write a test to make sure chunk follows the same format as first_openai_chunk_example
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert isinstance(choice["delta"]["role"], str), "'role' should be a string."
        assert "messages" not in choice
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


second_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [
        {"index": 0, "delta": {"content": "Hello"}, "finish_reason": None}  # it's null
    ],
}


def validate_second_format(chunk):
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert hasattr(choice["delta"], "role"), "'role' should be a string."
        # openai v1.0.0 returns content as None
        assert (choice["finish_reason"] is None) or isinstance(
            choice["finish_reason"], str
        ), "'finish_reason' should be None or a string."


from typing import List, Optional

#### STREAMING + FUNCTION CALLING ###
from pydantic import BaseModel


class Function(BaseModel):
    name: str
    arguments: str


class ToolCalls(BaseModel):
    index: int
    id: str
    type: str
    function: Function


class Delta(BaseModel):
    role: str
    content: Optional[str]
    tool_calls: List[ToolCalls]


class Choices(BaseModel):
    index: int
    delta: Delta
    logprobs: Optional[str]
    finish_reason: Optional[str]


class Chunk(BaseModel):
    id: str
    object: str
    created: int
    model: str
    # system_fingerprint: str
    choices: List[Choices]
class Function2(BaseModel):
    arguments: str


class ToolCalls2(BaseModel):
    index: int
    function: Optional[Function2]


class Delta2(BaseModel):
    tool_calls: List[ToolCalls2]


class Choices2(BaseModel):
    index: int
    delta: Delta2
    logprobs: Optional[str]
    finish_reason: Optional[str]


class Chunk2(BaseModel):
    id: str
    object: str
    created: int
    model: str
    system_fingerprint: Optional[str]
    choices: List[Choices2]




@pytest.mark.skip(reason="flaky test")
@pytest.mark.asyncio
async def test_hf_completion_tgi_stream():
    try:
        response = await acompletion(
            model="huggingface/HuggingFaceH4/zephyr-7b-beta",
            messages=[{"content": "Hello, how are you?", "role": "user"}],
            stream=True,
        )
        # Add any assertions here to check the response
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        idx = 0
        async for chunk in response:
            chunk, finished = streaming_format_tests(idx, chunk)
            complete_response += chunk
            if finished:
                break
            idx += 1
        print(f"completion_response: {complete_response}")
    except litellm.ServiceUnavailableError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# hf_test_completion_tgi_stream()

# def test_completion_aleph_alpha():
#     try:
#         response = completion(
#             model="luminous-base", messages=messages, stream=True
#         )
#         # Add any assertions here to check the response
#         has_finished = False
#         complete_response = ""
#         start_time = time.time()
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finished = finished
#             complete_response += chunk
#             if finished:
#                 break
#         if has_finished is False:
#             raise Exception("finished reason missing from final chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# # test_completion_aleph_alpha()

# def test_completion_aleph_alpha_bad_key():
#     try:
#         api_key = "bad-key"
#         response = completion(
#             model="luminous-base", messages=messages, stream=True, api_key=api_key
#         )
#         # Add any assertions here to check the response
#         has_finished = False
#         complete_response = ""
#         start_time = time.time()
#         for idx, chunk in enumerate(response):
#             chunk, finished = streaming_format_tests(idx, chunk)
#             has_finished = finished
#             complete_response += chunk
#             if finished:
#                 break
#         if has_finished is False:
#             raise Exception("finished reason missing from final chunk")
#         if complete_response.strip() == "":
#             raise Exception("Empty response received")
#     except InvalidRequestError as e:
#         pass
#     except Exception as e:
#         pytest.fail(f"Error occurred: {e}")

# test_completion_aleph_alpha_bad_key()


# test on openai completion call
def test_openai_chat_completion_call():
    litellm.set_verbose = False
    litellm.return_response_headers = True
    print(f"making openai chat completion call")
    response = completion(model="gpt-3.5-turbo", messages=messages, stream=True)
    assert isinstance(
        response._hidden_params["additional_headers"]["llm_provider-x-ratelimit-remaining-requests"],
        str,
    )

    print(f"response._hidden_params: {response._hidden_params}")
    complete_response = ""
    start_time = time.time()
    for idx, chunk in enumerate(response):
        chunk, finished = streaming_format_tests(idx, chunk)
        print(f"outside chunk: {chunk}")
        if finished:
            break
        complete_response += chunk
        # print(f'complete_chunk: {complete_response}')
    if complete_response.strip() == "":
        raise Exception("Empty response received")
    print(f"complete response: {complete_response}")


# test_openai_chat_completion_call()


def test_openai_chat_completion_complete_response_call():
    try:
        complete_response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            complete_response=True,
        )
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_openai_chat_completion_complete_response_call()
@pytest.mark.parametrize(
    "model",
    ["gpt-3.5-turbo", "claude-3-haiku-20240307"],  #
)
@pytest.mark.parametrize(
    "sync",
    [True, False],
)
@pytest.mark.asyncio
async def test_openai_stream_options_call(model, sync):
    litellm.set_verbose = True
    usage = None
    chunks = []
    if sync:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "user", "content": "say GM - we're going to make it "},
            ],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )
        for chunk in response:
            print("chunk: ", chunk)
            chunks.append(chunk)
    else:
        response = await litellm.acompletion(
            model=model,
            messages=[{"role": "user", "content": "say GM - we're going to make it "}],
            stream=True,
            stream_options={"include_usage": True},
            max_tokens=10,
        )

        async for chunk in response:
            print("chunk: ", chunk)
            chunks.append(chunk)

    last_chunk = chunks[-1]
    print("last chunk: ", last_chunk)

    """
    Assert that:
    - Last Chunk includes Usage
    - All chunks prior to last chunk have usage=None
    """

    assert last_chunk.usage is not None
    assert isinstance(last_chunk.usage, litellm.Usage)
    assert last_chunk.usage.total_tokens > 0
    assert last_chunk.usage.prompt_tokens > 0
    assert last_chunk.usage.completion_tokens > 0

    # assert all non last chunks have usage=None
    # Improved assertion with detailed error message
    non_last_chunks_with_usage = [chunk for chunk in chunks[:-1] if hasattr(chunk, "usage") and chunk.usage is not None]
    assert not non_last_chunks_with_usage, f"Non-last chunks with usage not None:\n" + "\n".join(
        f"Chunk ID: {chunk.id}, Usage: {chunk.usage}, Content: {chunk.choices[0].delta.content}"
        for chunk in non_last_chunks_with_usage
    )


def test_openai_stream_options_call_text_completion():
    litellm.set_verbose = False
    for idx in range(3):
        try:
            response = litellm.text_completion(
                model="gpt-3.5-turbo-instruct",
                prompt="say GM - we're going to make it ",
                stream=True,
                stream_options={"include_usage": True},
                max_tokens=10,
            )
            usage = None
            chunks = []
            for chunk in response:
                print("chunk: ", chunk)
                chunks.append(chunk)

            last_chunk = chunks[-1]
            print("last chunk: ", last_chunk)

            """
            Assert that:
            - Last Chunk includes Usage
            - All chunks prior to last chunk have usage=None
            """

            assert last_chunk.usage is not None
            assert last_chunk.usage.total_tokens > 0
            assert last_chunk.usage.prompt_tokens > 0
            assert last_chunk.usage.completion_tokens > 0

            # assert all non last chunks have usage=None
            assert all(chunk.usage is None for chunk in chunks[:-1])
            break
        except Exception as e:
            if idx < 2:
                pass
            else:
                raise e


def test_openai_text_completion_call():
    try:
        litellm.set_verbose = True
        response = completion(model="gpt-3.5-turbo-instruct", messages=messages, stream=True)
        complete_response = ""
        start_time = time.time()
        for idx, chunk in enumerate(response):
            chunk, finished = streaming_format_tests(idx, chunk)
            print(f"chunk: {chunk}")
            complete_response += chunk
            if finished:
                break
            # print(f'complete_chunk: {complete_response}')
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# test_openai_text_completion_call()



def test_completion_openai_with_functions():
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    try:
        litellm.set_verbose = False
        response = completion(
            model="gpt-3.5-turbo-1106",
            messages=[{"role": "user", "content": "what's the weather in SF"}],
            functions=function1,
            stream=True,
        )
        # Add any assertions here to check the response
        print(response)
        for chunk in response:
            print(chunk)
            if chunk["choices"][0]["finish_reason"] == "stop":
                break
            print(chunk["choices"][0]["finish_reason"])
            print(chunk["choices"][0]["delta"]["content"])
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


# asyncio.run(ai21_async_completion_call())


async def completion_call():
    try:
        response = completion(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
            logger_fn=logger_fn,
            max_tokens=10,
        )
        print(f"response: {response}")
        complete_response = ""
        start_time = time.time()
        # Change for loop to async for loop
        idx = 0
        async for chunk in response:
            chunk, finished = streaming_format_tests(idx, chunk)
            if finished:
                break
            complete_response += chunk
            idx += 1
        if complete_response.strip() == "":
            raise Exception("Empty response received")
        print(f"complete response: {complete_response}")
    except:
        print(f"error occurred: {traceback.format_exc()}")
        pass


# asyncio.run(completion_call())

#### Test Function Calling + Streaming ####

final_openai_function_call_example = {
    "id": "chatcmpl-7zVNA4sXUftpIg6W8WlntCyeBj2JY",
    "object": "chat.completion",
    "created": 1694892960,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "function_call": {
                    "name": "get_current_weather",
                    "arguments": '{\n  "location": "Boston, MA"\n}',
                },
            },
            "finish_reason": "function_call",
        }
    ],
    "usage": {"prompt_tokens": 82, "completion_tokens": 18, "total_tokens": 100},
}

function_calling_output_structure = {
    "id": str,
    "object": str,
    "created": int,
    "model": str,
    "choices": [
        {
            "index": int,
            "message": {
                "role": str,
                "content": (type(None), str),
                "function_call": {"name": str, "arguments": str},
            },
            "finish_reason": str,
        }
    ],
    "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
}

last_openai_chunk_example = {
    "id": "chatcmpl-7zSKLBVXnX9dwgRuDYVqVVDsgh2yp",
    "object": "chat.completion.chunk",
    "created": 1694881253,
    "model": "gpt-4-0613",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
}

"""
Final chunk (sdk):
chunk: ChatCompletionChunk(id='chatcmpl-96mM3oNBlxh2FDWVLKsgaFBBcULmI', 
choices=[Choice(delta=ChoiceDelta(content=None, function_call=None, role=None, 
tool_calls=None), finish_reason='stop', index=0, logprobs=None)], 
created=1711402871, model='gpt-3.5-turbo-0125', object='chat.completion.chunk', system_fingerprint='fp_3bc1b5746c')
"""


def validate_last_format(chunk):
    """
    Ensure last chunk has no remaining content or tools
    """
    assert isinstance(chunk, ModelResponse), "Chunk should be a dictionary."
    assert isinstance(chunk["id"], str), "'id' should be a string."
    assert isinstance(chunk["object"], str), "'object' should be a string."
    assert isinstance(chunk["created"], int), "'created' should be an integer."
    assert isinstance(chunk["model"], str), "'model' should be a string."
    assert isinstance(chunk["choices"], list), "'choices' should be a list."
    assert not hasattr(chunk, "usage"), "Chunk cannot contain usage"

    for choice in chunk["choices"]:
        assert isinstance(choice["index"], int), "'index' should be an integer."
        assert choice["delta"]["content"] is None
        assert choice["delta"]["function_call"] is None
        assert choice["delta"]["role"] is None
        assert choice["delta"]["tool_calls"] is None
        assert isinstance(choice["finish_reason"], str), "'finish_reason' should be a string."


def streaming_format_tests(idx, chunk) -> Tuple[str, bool]:
    extracted_chunk = ""
    finished = False
    print(f"chunk: {chunk}")
    if idx == 0:  # ensure role assistant is set
        validate_first_format(chunk=chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx == 1:  # second chunk
        validate_second_format(chunk=chunk)
    if idx != 0:  # ensure no role
        if "role" in chunk["choices"][0]["delta"]:
            pass  # openai v1.0.0+ passes role = None
    if chunk["choices"][0]["finish_reason"]:  # ensure finish reason is only in last chunk
        validate_last_format(chunk=chunk)
        finished = True
    if "content" in chunk["choices"][0]["delta"] and chunk["choices"][0]["delta"]["content"] is not None:
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    print(f"extracted chunk: {extracted_chunk}")
    return extracted_chunk, finished


tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

def validate_final_structure(item, structure=function_calling_output_structure):
    if isinstance(item, list):
        if not all(validate_final_structure(i, structure[0]) for i in item):
            return Exception("Function calling final output doesn't match expected output format")
    elif isinstance(item, dict):
        if not all(k in item and validate_final_structure(item[k], v) for k, v in structure.items()):
            return Exception("Function calling final output doesn't match expected output format")
    else:
        if not isinstance(item, structure):
            return Exception("Function calling final output doesn't match expected output format")
    return True


first_openai_function_call_example = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": None,
                "function_call": {"name": "get_current_weather", "arguments": ""},
            },
            "finish_reason": None,
        }
    ],
}


def validate_first_function_call_chunk_structure(item):
    if not (isinstance(item, dict) or isinstance(item, litellm.ModelResponse)):
        raise Exception(f"Incorrect format, type of item: {type(item)}")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in item:
            raise Exception("Incorrect format")

    if not isinstance(item["choices"], list) or not item["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in item["choices"]:
        if not (isinstance(choice, dict) or isinstance(choice, litellm.utils.StreamingChoices)):
            raise Exception(f"Incorrect format, type of choice: {type(choice)}")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

        if not (isinstance(choice["delta"], dict) or isinstance(choice["delta"], litellm.utils.Delta)):
            raise Exception(f"Incorrect format, type of choice: {type(choice['delta'])}")

        required_keys_in_delta = {"role", "content", "function_call"}
        for key in required_keys_in_delta:
            if key not in choice["delta"]:
                raise Exception("Incorrect format")

        if not (
            isinstance(choice["delta"]["function_call"], dict)
            or isinstance(choice["delta"]["function_call"], BaseModel)
        ):
            raise Exception(f"Incorrect format, type of function call: {type(choice['delta']['function_call'])}")

        required_keys_in_function_call = {"name", "arguments"}
        for key in required_keys_in_function_call:
            if not hasattr(choice["delta"]["function_call"], key):
                raise Exception(
                    f"Incorrect format, expected key={key};  actual keys: {choice['delta']['function_call']}, eval: {hasattr(choice['delta']['function_call'], key)}"
                )

    return True


second_function_call_chunk_format = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo-0613",
    "choices": [
        {
            "index": 0,
            "delta": {"function_call": {"arguments": "{\n"}},
            "finish_reason": None,
        }
    ],
}


def validate_second_function_call_chunk_structure(data):
    if not isinstance(data, dict):
        raise Exception("Incorrect format")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in data:
            raise Exception("Incorrect format")

    if not isinstance(data["choices"], list) or not data["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in data["choices"]:
        if not isinstance(choice, dict):
            raise Exception("Incorrect format")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

        if "function_call" not in choice["delta"] or "arguments" not in choice["delta"]["function_call"]:
            raise Exception("Incorrect format")

    return True


final_function_call_chunk_example = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo-0613",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}],
}


def validate_final_function_call_chunk_structure(data):
    if not (isinstance(data, dict) or isinstance(data, litellm.ModelResponse)):
        raise Exception("Incorrect format")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in data:
            raise Exception("Incorrect format")

    if not isinstance(data["choices"], list) or not data["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in data["choices"]:
        if not (isinstance(choice, dict) or isinstance(choice["delta"], litellm.utils.Delta)):
            raise Exception("Incorrect format")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

    return True


def streaming_and_function_calling_format_tests(idx, chunk):
    extracted_chunk = ""
    finished = False
    print(f"idx: {idx}")
    print(f"chunk: {chunk}")
    decision = False
    if idx == 0:  # ensure role assistant is set
        decision = validate_first_function_call_chunk_structure(chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx != 0:  # second chunk
        try:
            decision = validate_second_function_call_chunk_structure(data=chunk)
        except:  # check if it's the last chunk (returns an empty delta {} )
            decision = validate_final_function_call_chunk_structure(data=chunk)
            finished = True
    if "content" in chunk["choices"][0]["delta"]:
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    if decision == False:
        raise Exception("incorrect format")
    return extracted_chunk, finished


@pytest.mark.parametrize(
    "model",
    [
        # "gpt-3.5-turbo",
        # "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku-20240307",
    ],
)
def test_streaming_and_function_calling(model):
    import json

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    try:
        # litellm.set_verbose = True
        response: litellm.CustomStreamWrapper = completion(
            model=model,
            tools=tools,
            messages=messages,
            stream=True,
            tool_choice="required",
        )  # type: ignore
        # Add any assertions here to check the response
        json_str = ""
        for idx, chunk in enumerate(response):
            # continue
            # print("\n{}\n".format(chunk))
            if idx == 0:
                assert chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                assert isinstance(chunk.choices[0].delta.tool_calls[0].function.arguments, str)
            if chunk.choices[0].delta.tool_calls is not None:
                json_str += chunk.choices[0].delta.tool_calls[0].function.arguments

        print(json.loads(json_str))
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e


def validate_first_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk(**chunk.model_dump())

final_function_call_chunk_example = {
    "id": "chatcmpl-7zVRoE5HjHYsCMaVSNgOjzdhbS3P0",
    "object": "chat.completion.chunk",
    "created": 1694893248,
    "model": "gpt-3.5-turbo-0613",
    "choices": [{"index": 0, "delta": {}, "finish_reason": "function_call"}],
}


def validate_second_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk2(**chunk.model_dump())


class Delta3(BaseModel):
    content: Optional[str] = None
    role: Optional[str] = None
    function_call: Optional[dict] = None
    tool_calls: Optional[List] = None


class Choices3(BaseModel):
    index: int
    delta: Delta3
    logprobs: Optional[str]
    finish_reason: str


class Chunk3(BaseModel):
    id: str
    object: str
    created: int
    model: str
    # system_fingerprint: str
    choices: List[Choices3]


def validate_final_streaming_function_calling_chunk(chunk: ModelResponse):
    chunk_instance = Chunk3(**chunk.model_dump())


def validate_final_function_call_chunk_structure(data):
    if not (isinstance(data, dict) or isinstance(data, litellm.ModelResponse)):
        raise Exception("Incorrect format")

    required_keys = {"id", "object", "created", "model", "choices"}
    for key in required_keys:
        if key not in data:
            raise Exception("Incorrect format")

    if not isinstance(data["choices"], list) or not data["choices"]:
        raise Exception("Incorrect format")

    required_keys_in_choices_array = {"index", "delta", "finish_reason"}
    for choice in data["choices"]:
        if not (isinstance(choice, dict) or isinstance(choice["delta"], litellm.utils.Delta)):
            raise Exception("Incorrect format")
        for key in required_keys_in_choices_array:
            if key not in choice:
                raise Exception("Incorrect format")

    return True


def streaming_and_function_calling_format_tests(idx, chunk):
    extracted_chunk = ""
    finished = False
    print(f"idx: {idx}")
    print(f"chunk: {chunk}")
    decision = False
    if idx == 0:  # ensure role assistant is set
        decision = validate_first_function_call_chunk_structure(chunk)
        role = chunk["choices"][0]["delta"]["role"]
        assert role == "assistant"
    elif idx != 0:  # second chunk
        try:
            decision = validate_second_function_call_chunk_structure(data=chunk)
        except:  # check if it's the last chunk (returns an empty delta {} )
            decision = validate_final_function_call_chunk_structure(data=chunk)
            finished = True
    if "content" in chunk["choices"][0]["delta"]:
        extracted_chunk = chunk["choices"][0]["delta"]["content"]
    if decision == False:
        raise Exception("incorrect format")
    return extracted_chunk, finished


@pytest.mark.parametrize(
    "model",
    [
        # "gpt-3.5-turbo",
        # "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku-20240307",
    ],
)
def test_streaming_and_function_calling(model):
    import json

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    messages = [{"role": "user", "content": "What is the weather like in Boston?"}]
    try:
        # litellm.set_verbose = True
        response: litellm.CustomStreamWrapper = completion(
            model=model,
            tools=tools,
            messages=messages,
            stream=True,
            tool_choice="required",
        )  # type: ignore
        # Add any assertions here to check the response
        json_str = ""
        for idx, chunk in enumerate(response):
            # continue
            # print("\n{}\n".format(chunk))
            if idx == 0:
                assert chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                assert isinstance(chunk.choices[0].delta.tool_calls[0].function.arguments, str)
            if chunk.choices[0].delta.tool_calls is not None:
                json_str += chunk.choices[0].delta.tool_calls[0].function.arguments

        print(json.loads(json_str))
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")
        raise e





def test_success_callback_streaming():
    def success_callback(kwargs, completion_response, start_time, end_time):
        print(
            {
                "success": True,
                "input": kwargs,
                "output": completion_response,
                "start_time": start_time,
                "end_time": end_time,
            }
        )

    litellm.success_callback = [success_callback]

    messages = [{"role": "user", "content": "hello"}]
    print("TESTING LITELLM COMPLETION CALL")
    response = litellm.completion(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
        max_tokens=5,
    )
    print(response)

    for chunk in response:
        print(chunk["choices"][0])





def test_completion_claude_3_function_call_with_streaming():
    litellm.set_verbose = True
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="claude-3-opus-20240229",
            messages=messages,
            tools=tools,
            tool_choice="required",
            stream=True,
        )
        idx = 0
        for chunk in response:
            print(f"chunk in response: {chunk}")
            if idx == 0:
                assert chunk.choices[0].delta.tool_calls[0].function.arguments is not None
                assert isinstance(chunk.choices[0].delta.tool_calls[0].function.arguments, str)
                validate_first_streaming_function_calling_chunk(chunk=chunk)
            elif idx == 1 and chunk.choices[0].finish_reason is None:
                validate_second_streaming_function_calling_chunk(chunk=chunk)
            elif chunk.choices[0].finish_reason is not None:  # last chunk
                assert "usage" in chunk._hidden_params
                validate_final_streaming_function_calling_chunk(chunk=chunk)
            idx += 1
        # raise Exception("it worked!")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")

if __name__ == "__main__":
    pytest.main(["-s", __file__])