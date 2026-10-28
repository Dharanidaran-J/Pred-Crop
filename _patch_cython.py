"""
AgriPredict AI - Windows Application Control / WDAC Compatibility Hook
Ensures pure-Python implementations are used when Windows Defender Application
Control or AppLocker blocks Cython .pyd compiled extensions.
"""

import sys
import os
import importlib.abc
import importlib.util

class DisablePydFinder(importlib.abc.MetaPathFinder):
    """Intercepts module resolution for SQLAlchemy to load pure .py files instead of blocked .pyd."""
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("sqlalchemy"):
            if path is None:
                return None
            for p in path:
                base_name = fullname.split(".")[-1]
                py_path = os.path.join(p, base_name + ".py")
                if os.path.isfile(py_path):
                    return importlib.util.spec_from_file_location(fullname, py_path)
        return None

# Register hook if not already registered
if not any(isinstance(f, DisablePydFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, DisablePydFinder())
