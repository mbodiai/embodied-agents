
class Backend:
    """Base class for agent backends."""

    def predict(self, *args, **kwargs) -> str:
        raise NotImplementedError