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
    def do(self, **kwargs) -> None:
        """Executes motion.

        Args:
            kwargs: Additional arguments to pass to the hardware interface.
        """
        raise NotImplementedError
