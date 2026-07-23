# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.

"""
Shared library loader utility for ctypes bindings.

Handles finding and loading .so files by name, supporting both system-installed
libraries and custom paths via environment variables.
"""

import ctypes
import ctypes.util
import os


def load_library(name):
    """
    Load a shared library by short name (e.g. "fabric", "ibverbs", "efa").

    Search order:
      1. OFI_LIB_PATH environment variable (colon-separated directories)
      2. LD_LIBRARY_PATH (handled by ctypes/dlopen)
      3. System library paths (via ctypes.util.find_library)

    Args:
        name: Library short name without "lib" prefix or ".so" suffix.

    Returns:
        ctypes.CDLL object.

    Raises:
        OSError: If the library cannot be found.
    """
    lib_path = os.environ.get("OFI_LIB_PATH", "")
    for directory in lib_path.split(":"):
        if not directory:
            continue
        for suffix in (".so", ".dylib"):
            path = os.path.join(directory, f"lib{name}{suffix}")
            if os.path.isfile(path):
                return ctypes.CDLL(path)

    system_path = ctypes.util.find_library(name)
    if system_path:
        return ctypes.CDLL(system_path)

    return ctypes.CDLL(f"lib{name}.so")
