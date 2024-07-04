import os
from typing import Generator, List
import httpx
from mbodied.agents.backends.serializer import Serializer
from mbodied.types.sense import Image
from mbodied.agents.backends.openai_backend import OpenAIBackendMixin
from mbodied.types.message import Message

class HttpxSerializer(Serializer):
    @classmethod
    def serialize_image(cls, image : Image):
        return {
            "type": "image_url",
            "image_url": image.url
        }
    
    @classmethod
    def serialize_text(cls, text):
        return {
            "type": "text",
            "text": text
        }
    
    @classmethod
    def extract_response(cls, response):
        return response["responses"][0]["message"]["content"]



class HttpxBackend(OpenAIBackendMixin):
    SERIALIZER = HttpxSerializer
    DEFAULT_SRC = "https://api.reka.ai/v1/chat"
    DEFAULT_MODEL = "reka-core"

    def __init__(self, api_key=None, model_src: str| None=None, serializer: Serializer | None =None, **kwargs):
      """Initializes the CompleteBackend. Defaults to using the API key from the environment and 
      
      Args:
          api_key (Optional[str]): The API key for the Complete service.
          model_src (str): The base URL for the Complete API.
          serializer (Optional[Serializer]): The serializer to use for serializing messages.
      """
      self.base_url = model_src or self.DEFAULT_SRC
      self.api_key = api_key or os.getenv("MBB_API_KEY")
      self.headers = {
          "X-Api-Key": self.api_key,
          "Content-Type": "application/json"
      }
      self.serialized = serializer or self.SERIALIZER
    
    def _create_completion(self,messages: List[Message], model: str | None = None, **kwargs) -> str:
      model = model or self.DEFAULT_MODEL
      data = {
        "messages": self.serialized(messages)(),
        "model": model,
        "stream": False,
        **kwargs
      }
      with httpx.Client(follow_redirects=True) as client:
          response = client.post(self.base_url, headers=self.headers, json=data, timeout=kwargs.get("timeout", 60))
          if response.status_code == 200:
              response_data = response.json()
              return self.serialized.extract_response(response_data)
          else:
              response.raise_for_status()
    
    def _stream_completion(self, messages: List[Message], model:str|None= None, **kwargs) -> Generator[str, None, None]:
      model = model or self.DEFAULT
      data = {
          "messages": self.serialized(messages)(),
          "model": model,
          "stream": True,
          **kwargs
      }
      with httpx.Client(follow_redirects=True) as client:
          with client.stream("post", self.base_url, headers=self.headers, json=data, timeout=60) as stream:
              for chunk in stream.iter_text():
                  yield chunk


    async def _acreate_completion(self,messages: List[Message], model: str | None= None, **kwargs) -> str:
          model = model or self.DEFAULT_MODEL
          data = {
              "messages": self.serialized(messages)(),
              "model": model,
              "stream": False
              **kwargs
          }
          if "use_search_engine" in kwargs:
              data["use_search_engine"] = kwargs["use_search_engine"]
            
          async with httpx.Client(follow_redirects=True) as client:
              response = await client.post(self.base_url, headers=self.headers, json=data)
              if response.status_code == 200:
                  response_data = response.json()
                  print(response_data)
                  return self.serialized.extract_response(response_data)
              else:
                  response.raise_for_status()

    async def _astream_completion(self, messages: List[Message], model: str | None = None, **kwargs) -> str:
        model = model or self.DEFAULT_MODEL
        data = {
            "messages": self.serialized(messages)(),
            "model": model,
            "stream": True
            **kwargs
        }
        if "use_search_engine" in kwargs:
          data["use_search_engine"] = kwargs["use_search_engine"]
        
        async with httpx.Client(follow_redirects=True) as client:
            async with client.stream("post", self.base_url, headers=self.headers, json=data, timeout=60) as run:
                async for chunk in run.aiter_text():
                    yield chunk

  


if __name__ == "__main__":
    # Usage
    client = HttpxBackend()
    image_url = "https://v0.docs.reka.ai/_images/000000245576.jpg"
    text = "What animal is this? Answer briefly."
    response = client._create_completion([Message(role="user", content="text")])
    print(response)