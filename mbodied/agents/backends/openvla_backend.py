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

from typing import List, Literal

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from mbodied.agents.backends.serializer import Serializer
from mbodied.types.sense.vision import Image


class OpenVLASerializer(Serializer):
    pass


class OpenVLABackend:
    """OpenVLA backend that runs locally to generate robot actions.

    Beware of the memory requirements of OpenVLA, which is quite large.

    Attributes:
        model_id (str): The model to use for the OpenVLA backend.
        device (torch.device): The device to run the model on.
        torch_dtype (torch.dtype): The torch data type to use.
        processor (AutoProcessor): The processor for the model.
        model (AutoModelForVision2Seq): The model for the OpenVLA backend.
    """

    DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ATTN_IMPLEMENTATION = "flash_attention_2" if torch.cuda.is_available() else "eager"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        attn_implementation: Literal["flash_attention_2", "eager"] = ATTN_IMPLEMENTATION,
        torch_dtype: torch.dtype = DTYPE,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.torch_dtype = torch_dtype
        # Load Processor & VLA
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            attn_implementation=attn_implementation,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=torch.cuda.is_available(),
            trust_remote_code=True,
            **kwargs,
        ).to(device)

    def act(self, instruction: str, image: Image, unnorm_key: str = "bridge_orig") -> str:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image.pil).to(self.device, dtype=self.torch_dtype)
        response = self.model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return str(response)
