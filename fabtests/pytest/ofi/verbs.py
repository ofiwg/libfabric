# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.

"""
ctypes bindings for libibverbs and libefa (rdma-core).

Provides access to ibv_get_device_list, ibv_open_device, efadv_query_device
and related functions needed for device capability queries.
"""

import ctypes
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_uint8,
    c_uint16,
    c_uint32,
    c_uint64,
    c_void_p,
)

from ofi._loader import load_library

# -- ibv structures (minimal, opaque where possible) --


class ibv_device(Structure):
    _fields_ = [
        ("_ops", c_void_p),
        ("node_type", c_int),
        ("transport_type", c_int),
        ("name", ctypes.c_char * 64),
        ("dev_name", ctypes.c_char * 64),
        ("dev_path", ctypes.c_char * 256),
        ("ibdev_path", ctypes.c_char * 256),
    ]


class ibv_context(Structure):
    _fields_ = [
        ("device", POINTER(ibv_device)),
        ("cmd_fd", c_int),
        ("async_fd", c_int),
    ]


class efadv_device_attr(Structure):
    _fields_ = [
        ("comp_mask", c_uint32),
        ("max_sq_wr", c_uint32),
        ("max_rq_wr", c_uint32),
        ("max_sq_sge", c_uint16),
        ("max_rq_sge", c_uint16),
        ("inline_buf_size", c_uint16),
        ("_reserved", c_uint8),
        ("device_caps", c_uint32),
        ("max_rdma_size", c_uint32),
    ]


# efadv device capability flags
EFADV_DEVICE_ATTR_CAPS_RDMA_READ = 1 << 0
EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE = 1 << 1
EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV = 1 << 2


# -- Library handles --

_ibverbs = None
_efadv = None


def _get_ibverbs():
    global _ibverbs
    if _ibverbs is None:
        _ibverbs = load_library("ibverbs")
        _ibverbs.ibv_get_device_list.argtypes = [POINTER(c_int)]
        _ibverbs.ibv_get_device_list.restype = POINTER(POINTER(ibv_device))
        _ibverbs.ibv_free_device_list.argtypes = [POINTER(POINTER(ibv_device))]
        _ibverbs.ibv_free_device_list.restype = None
        _ibverbs.ibv_open_device.argtypes = [POINTER(ibv_device)]
        _ibverbs.ibv_open_device.restype = POINTER(ibv_context)
        _ibverbs.ibv_close_device.argtypes = [POINTER(ibv_context)]
        _ibverbs.ibv_close_device.restype = c_int
        _ibverbs.ibv_get_device_name.argtypes = [POINTER(ibv_device)]
        _ibverbs.ibv_get_device_name.restype = c_char_p
    return _ibverbs


def _get_efadv():
    global _efadv
    if _efadv is None:
        _efadv = load_library("efa")
        _efadv.efadv_query_device.argtypes = [
            POINTER(ibv_context), POINTER(efadv_device_attr), c_uint32,
        ]
        _efadv.efadv_query_device.restype = c_int
    return _efadv


# -- Public API --


class DeviceList:
    """
    Context manager for ibv_device_list. Device pointers are only valid
    while this object is alive (before free() is called).
    """

    def __init__(self):
        lib = _get_ibverbs()
        self._lib = lib
        num_devices = c_int(0)
        self._raw = lib.ibv_get_device_list(ctypes.byref(num_devices))
        self._count = num_devices.value if self._raw else 0
        self.devices = []
        for i in range(self._count):
            dev = self._raw[i]
            name = lib.ibv_get_device_name(dev)
            self.devices.append((dev, name.decode() if name else ""))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()

    def __len__(self):
        return self._count

    def __iter__(self):
        return iter(self.devices)

    def free(self):
        if self._raw:
            self._lib.ibv_free_device_list(self._raw)
            self._raw = None


def query_efa_device(device_ptr):
    """
    Query EFA device attributes for a given ibv_device pointer.

    The caller must ensure the device list has not been freed.

    Args:
        device_ptr: Pointer to ibv_device from DeviceList.

    Returns:
        efadv_device_attr on success, None if not an EFA device.

    Raises:
        RuntimeError: On unexpected query failure.
    """
    ibv = _get_ibverbs()
    efa = _get_efadv()

    ctx = ibv.ibv_open_device(device_ptr)
    if not ctx:
        raise RuntimeError("ibv_open_device failed")

    attr = efadv_device_attr()
    ctypes.memset(ctypes.byref(attr), 0, ctypes.sizeof(attr))
    err = efa.efadv_query_device(ctx, ctypes.byref(attr), ctypes.sizeof(attr))
    ibv.ibv_close_device(ctx)

    if err != 0:
        import errno as _errno
        if err == _errno.EOPNOTSUPP:
            return None
        raise RuntimeError(f"efadv_query_device failed with error {err}")

    return attr


def check_rdma_capability(operation):
    """
    Check whether EFA RDMA read/write/writedata is enabled on this host.

    This is the Python equivalent of fi_efa_rdma_checker.

    Args:
        operation: One of "read", "write", "writedata"

    Returns:
        True if the capability is enabled, False otherwise.

    Raises:
        ValueError: If operation is not recognized.
        RuntimeError: If no EFA device is found or query fails.
    """
    if operation not in ("read", "write", "writedata"):
        raise ValueError(f"Unknown operation '{operation}'. Allowed: read, write, writedata")

    with DeviceList() as dev_list:
        if not len(dev_list):
            raise RuntimeError("No ibv device found")

        for device_ptr, name in dev_list:
            attr = query_efa_device(device_ptr)
            if attr is None:
                continue

            if attr.max_rdma_size == 0:
                return False

            if operation == "read":
                return True

            if operation == "write":
                return bool(attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE)

            if operation == "writedata":
                return bool(attr.device_caps & EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV)

    raise RuntimeError("No EFA device found")
