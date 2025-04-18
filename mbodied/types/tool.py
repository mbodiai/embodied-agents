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

# This module defines tool-related type classes that are compatible with the OpenAI API format
# while being tailored for mbodi's specific needs.

from typing import Any, Dict, Literal, Optional

from pydantic import ConfigDict

from mbodied.types.sample import Sample

Role = Literal["user", "assistant", "system"]


class Function(Sample):
    arguments: str | dict[str, Any]
    """
    Contains the arguments for function execution as provided by the model.
    
    May be in JSON format (as a string) or as a dictionary. Models may occasionally
    produce invalid JSON or include parameters not in your schema, so validation
    is recommended before executing any functions.
    """

    name: str
    """The identifier for the function to be executed."""


class ToolCall(Sample):
    """Represents a call to a tool by the model.

    In mbodi, a ToolCall is a structured way for models to interact with
    external capabilities through well-defined interfaces.
    """

    id: str
    """A unique identifier for this specific tool call."""

    function: Function
    """The function details including name and arguments."""

    type: Literal["function"]
    """The tool type identifier. Currently supports 'function' only."""


# Type alias for function parameters to match API compatibility
FunctionParameters = Dict[str, object]


class FunctionDefinition(Sample):
    model_config = ConfigDict(extra="allow")
    name: str
    """
    The function's identifier name.
    
    Should use alphanumeric characters, underscores, or dashes.
    Maximum length is 64 characters.
    """

    description: Optional[str] = None
    """
    Explains the function's purpose and capabilities.
    
    This helps the model determine when and how to use this function.
    """

    parameters: FunctionParameters = {}
    """
    Schema defining the function's expected parameters.
    
    Uses JSON Schema format to specify parameter types, requirements, 
    and constraints. An empty dictionary indicates no parameters.
    """

    strict: Optional[bool] = False
    """
    Controls parameter validation strictness.
    
    When enabled, the model must adhere exactly to the parameter schema
    with no deviations. This provides more predictable outputs but requires
    careful schema design.
    """


class Tool(Sample):
    function: FunctionDefinition
    type: Literal["function"]
    """Defines the tool category. Currently, our implementation only supports function-based tools."""
