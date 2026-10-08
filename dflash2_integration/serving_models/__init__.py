"""Isolated external model package for SGLang worker discovery.

Import the trusted implementation here. SGLang imports the package outside its
optional-module exception handler, so extension failures stop the worker rather
than silently leave the stock DFlash2 registry entry in place.
"""

from ..serving_model import DFlash2DraftModel

__all__ = ["DFlash2DraftModel"]
