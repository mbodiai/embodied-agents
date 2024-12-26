import importlib
import sys
from importlib.util import LazyLoader, find_spec, module_from_spec
from types import ModuleType
from typing import Literal, Optional, Any


def reload(module: str) -> None:
    """Reload an existing module or import it if not already imported.

    If the specified module is already present in the global namespace,
    it will be reloaded. Otherwise, the module will be imported.

    Args:
        module (str): The name of the module to reload or import.

    Returns:
        None
    """
    if module in globals():
        return importlib.reload(globals()[module])
    return importlib.import_module(module)


def smart_import(name: str, mode: Literal["lazy"] | None = None, attribute: Optional[str] = None) -> Any:
    """Import a module with optional lazy loading.

    This function imports a module by name. If the module is already
    present in the global namespace, it will return the existing module.
    If the `mode` is set to "lazy", the module will be loaded lazily,
    meaning that the module will only be fully loaded when an attribute
    or function within the module is accessed.

    Args:
        name (str): The name of the module to import.
        mode (Literal["lazy"] | None, optional): If "lazy", the module will
            be imported using a lazy loader. Defaults to None.
        attribute (str, optional): The name of an attribute, class, or symbol
            to import from the module. Defaults to None.

    Returns:
        Any: The imported module or the specified attribute from the module.

    Raises:
        NameError: If the module cannot be found or imported.
    """
    if name in globals():
        return globals()[name]
    if mode == "lazy":
        spec = find_spec(name)
        if spec is None:
            msg = f"Module `{name}` not found"
            raise NameError(msg)
        loader = LazyLoader(spec.loader)
        spec.loader = loader
        module = module_from_spec(spec)
        sys.modules[name] = module
        loader.exec_module(module)

    try:
        module = importlib.import_module(name)
    except ImportError as e:
        msg = f"Module {name} not found"
        raise NameError(msg) from e

    if attribute:
        try:
            return getattr(module, attribute)
        except AttributeError as e:
            msg = f"Attribute `{attribute}` not found in module `{name}`"
            raise NameError(msg) from e

    return module
