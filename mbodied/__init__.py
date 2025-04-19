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

__version__ = "1.5.0"

import logging

from rich.logging import RichHandler
from rich.pretty import install
from rich.traceback import install as install_traceback

logger = logging.getLogger()
logger.addHandler(RichHandler())
install_traceback(word_wrap=True, max_frames=10)
install(max_length=100, max_string=100, overflow="fold")
