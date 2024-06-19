# Copyright 2024 Mbodi AI
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

from typing import List, Literal

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from mbodied_agents.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied_agents.base.serializer import Serializer
from mbodied_agents.types.message import Message
from mbodied_agents.types.sense.vision import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
attn_implementation = "flash_attention_2" if torch.cuda.is_available() else "eager"


system_prompt = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def format_instruction(instruction: str) -> str:
    return f"{system_prompt} USER: What action should the robot take to {instruction}? ASSISTANT:"


class HuggingFaceSerializer(Serializer):
    pass


class HuggingFaceBackend(OpenAIBackendMixin):
    DEFAULT_SYSTEM_PROMPT = system_prompt
    DEFAULT_TASK = "vision-to-text modeling"
    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(
        self,
        model: str = "openvla/openvla-v01-7b",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        attn_implementation: Literal["flash_attention_2", "eager"] = "eager",
        torch_dtype: torch.dtype = torch.float16,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        # Load Processor & VLA
        self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=torch.cuda.is_available(),
            trust_remote_code=True,
            **kwargs,
        ).to(device)

    def _create_completion(self, messages: List[Message] | None = None, model: str | None = None, **_) -> str:
        model = model or self.model
        content = messages[0].content
        item = None
        instruction = ""
        for item in content:
            if isinstance(item, Image):
                image = item.pil
            else:
                instruction += item
        inputs = self.processor(format_instruction(instruction), image).to(device, dtype=dtype)
        action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        return str(action)
