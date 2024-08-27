import importlib
import sys
from importlib.util import LazyLoader, find_spec, module_from_spec
from types import ModuleType
from typing import Literal


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


def smart_import(name: str, mode: Literal["lazy"] | None = None) -> ModuleType:
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

    Returns:
        ModuleType: The imported module.

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
        return module

    try:
        return importlib.import_module(name)
    except ImportError as e:
        msg = f"Module {name} not found"
        raise NameError(msg) from e
