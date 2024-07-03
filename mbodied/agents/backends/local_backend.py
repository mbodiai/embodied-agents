import logging
from functools import partial
from typing import Any, Callable

import regex
from rellm import complete_re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mbodied.types.message import Message


class CausalLLMBackend:
  def __init__(self, model: str = "Locutusque/TinyMistral-248M-Instruct", **kwargs) -> None:
    self.model = AutoModelForCausalLM.from_pretrained(model)
    self.tokenizer = AutoTokenizer.from_pretrained(model)
    if "pattern"not  in kwargs:
      self.model = pipeline("text-generation", model=model, tokenizer=self.tokenizer, use_fast=True)
      return
    self.pattern = kwargs["pattern"]
    self.user_token = kwargs.get("user_token", "<|USER|>")
    self.assistant_token = kwargs.get("assistant_token", "<|ASSISTANT|>")
  
  @staticmethod
  def load_message(message: Message, bos="", eos="") -> str:
    prompt_list = []
    for c in message.content:
      if not isinstance(c, str):
        raise ValueError("Content must be a string or list of strings.")
      prompt_list.append(c)
    if len(prompt_list) == 0:
      raise ValueError("Content must not be empty.")
    prompt = " ".join(prompt_list)
    return f"{bos} {prompt} {eos}"
    
  def act(self, message: Message, context: Any, max_length=20, **kwargs) -> str:
    if not isinstance(message, Message):
      raise ValueError("Message must be a Message object.")
    prompt_list = []
    prompt_list.append(self.load_message(message))
    for m in context:
      if isinstance(m, Message):
        if m.role == "user":
          prompt_list.append(self.load_message(m, bos=self.user_token, eos=""))
        elif m.role == "assistant":
          prompt_list.append(self.load_message(m, bos=self.assistant_token, eos=""))
      
    
    
    prompt = " ".join(prompt_list)
    if "pattern" in kwargs or hasattr(self, "pattern") and self.pattern is not None:
      pattern = kwargs.get("pattern", self.pattern if hasattr(self, "pattern") else None)
      pattern = regex.compile(pattern)
      return complete_re(tokenizer=self.tokenizer, 
                    model=self.model, 
                    prompt= prompt,
                    pattern=pattern,
                    do_sample=True,
                    max_new_tokens=max_length)

    return self.model(prompt)[0]['generated_text']


  

    
