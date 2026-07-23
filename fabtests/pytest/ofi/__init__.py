# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.

"""
Python ctypes bindings for libfabric and related libraries.

This package provides thin ctypes wrappers around libfabric's C API,
allowing pytest tests to call libfabric functions directly from Python
instead of shelling out to compiled C test binaries.

Usage:
    from ofi import libfabric
    info = libfabric.get_info(provider="efa")

    from ofi import verbs
    devices = verbs.get_device_list()
"""

from ofi.libfabric import FI_VERSION, get_info, version  # noqa: F401
