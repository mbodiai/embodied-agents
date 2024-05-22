from abc import ABC, abstractmethod
from typing import Any, Optional, List

import backoff
from anthropic import RateLimitError as AnthropicRateLimitError
from openai._exceptions import RateLimitError as OpenAIRateLimitError

from mbodied.types.message import Message

ERRORS = (
    OpenAIRateLimitError,
    AnthropicRateLimitError,
)


class Backend(ABC):
    """Abstract base class for Backend implementations.

    Attributes:
        api_key (Optional[str]): The API key for authentication.
        client (Optional[Any]): The client instance used for API calls.
    """

    def __init__(self, api_key: Optional[str] = None, client: Optional[Any] = None):
        """Initializes the Backend instance.

        Args:
            api_key (Optional[str]): The API key for authentication, if any.
            client (Optional[Any]): The client instance, if any.
        """
        self.api_key = api_key
        self.client = client

    @abstractmethod
    def _create_completion(self, messages: Optional[List[Message]] = None, model: Optional[Any] = None, **kwargs) -> str:
        """Abstract method to create a completion based on the given messages and model.

        Args:
            messages (Optional[List[Message]]): List of messages to process.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.

        Raises:
            NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @backoff.on_exception(
        backoff.expo,
        ERRORS,
        max_tries=3,
    )
    def create_completion(self, message: Message, context: Optional[List[Message]] = None, model: Optional[Any] = None, **kwargs) -> str:
        """Create a completion based on the given message and context.

        Args:
            message (Message): The message to process.
            context (Optional[List[Message]]): The context of messages.
            model (Optional[Any]): The model used for processing the messages.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The result of the completion.
        """
        if context is None:
            context = []
        context.append(message)
        for msg in context:
            for c in msg.content:
                if not isinstance(c, str):
                    pass
                else:
                    pass

        if model is not None:
            kwargs["model"] = model

        return self._create_completion(context, **kwargs)
