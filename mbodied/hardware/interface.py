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

from abc import ABC, abstractmethod


class HardwareInterface(ABC):
    """Abstract base class for hardware interfaces.

    This class provides a template for creating hardware interfaces that can
    control robots or other hardware devices.
    """

    def __init__(self, **kwargs):
        """Initializes the hardware interface.

        Args:
            kwargs: Additional arguments to pass to the hardware interface.
        """
        raise NotImplementedError

    @abstractmethod
    def do(self, *args, **kwargs) -> None: # noqa
        """Executes motion.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
        raise NotImplementedError
    
    def fetch(self, *args, **kwargs) -> None:
        """Fetches data from the hardware.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
        raise NotImplementedError
    
    def capture(self, *args, **kwargs) -> None:
        """Captures continuous data from the hardware.

        Args:
            args: Arguments to pass to the hardware interface.
            kwargs: Additional arguments to pass to the hardware interface.
        """
        raise NotImplementedError
