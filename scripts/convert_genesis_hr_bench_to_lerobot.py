"""Compatibility entry point for :mod:`scripts.data_conversion.lerobot`.

Imports share the implementation module, including private helpers and mutable
configuration. The command-line interface is unchanged.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.data_conversion import lerobot as _implementation  # noqa: E402

if __name__ == "__main__":
    _implementation.main()
else:
    sys.modules[__name__] = _implementation
