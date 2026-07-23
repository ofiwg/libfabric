# SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
# SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved.

"""
ctypes bindings for libfabric core API.

Wraps fi_getinfo, fi_freeinfo, fi_version and related structures.
"""

import ctypes
import ctypes.util
from ctypes import (
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_size_t,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)

from ofi._loader import load_library


def FI_VERSION(major, minor):
    return (major << 16) | minor


# -- Opaque pointer types for fid-based objects --

class fid(Structure):
    _fields_ = [
        ("fclass", c_size_t),
        ("context", c_void_p),
        ("ops", c_void_p),
    ]


class fid_fabric(Structure):
    _fields_ = [("fid", fid)]


class fid_domain(Structure):
    _fields_ = [("fid", fid)]


class fid_nic(Structure):
    pass


# -- Attribute structures --

class fi_tx_attr(Structure):
    _fields_ = [
        ("caps", c_uint64),
        ("mode", c_uint64),
        ("op_flags", c_uint64),
        ("msg_order", c_uint64),
        ("comp_order", c_uint64),
        ("inject_size", c_size_t),
        ("size", c_size_t),
        ("iov_limit", c_size_t),
        ("rma_iov_limit", c_size_t),
        ("tclass", c_uint32),
    ]


class fi_rx_attr(Structure):
    _fields_ = [
        ("caps", c_uint64),
        ("mode", c_uint64),
        ("op_flags", c_uint64),
        ("msg_order", c_uint64),
        ("comp_order", c_uint64),
        ("total_buffered_recv", c_size_t),
        ("size", c_size_t),
        ("iov_limit", c_size_t),
    ]


# fi_ep_type enum values
FI_EP_UNSPEC = 0
FI_EP_MSG = 1
FI_EP_DGRAM = 2
FI_EP_RDM = 3


class fi_ep_attr(Structure):
    _fields_ = [
        ("type", c_int),
        ("protocol", c_uint32),
        ("protocol_version", c_uint32),
        ("max_msg_size", c_size_t),
        ("msg_prefix_size", c_size_t),
        ("max_order_raw_size", c_size_t),
        ("max_order_war_size", c_size_t),
        ("max_order_waw_size", c_size_t),
        ("mem_tag_format", c_uint64),
        ("tx_ctx_cnt", c_size_t),
        ("rx_ctx_cnt", c_size_t),
        ("auth_key_size", c_size_t),
        ("auth_key", POINTER(c_uint8)),
    ]


class fi_domain_attr(Structure):
    _fields_ = [
        ("domain", POINTER(fid_domain)),
        ("name", c_char_p),
        ("threading", c_int),
        ("control_progress", c_int),
        ("data_progress", c_int),
        ("resource_mgmt", c_int),
        ("av_type", c_int),
        ("mr_mode", c_int),
        ("mr_key_size", c_size_t),
        ("cq_data_size", c_size_t),
        ("cq_cnt", c_size_t),
        ("ep_cnt", c_size_t),
        ("tx_ctx_cnt", c_size_t),
        ("rx_ctx_cnt", c_size_t),
        ("max_ep_tx_ctx", c_size_t),
        ("max_ep_rx_ctx", c_size_t),
        ("max_ep_stx_ctx", c_size_t),
        ("max_ep_srx_ctx", c_size_t),
        ("cntr_cnt", c_size_t),
        ("mr_iov_limit", c_size_t),
        ("caps", c_uint64),
        ("mode", c_uint64),
        ("auth_key", POINTER(c_uint8)),
        ("auth_key_size", c_size_t),
        ("max_err_data", c_size_t),
        ("mr_cnt", c_size_t),
        ("tclass", c_uint32),
        ("max_ep_auth_key", c_size_t),
        ("max_group_id", c_uint32),
        ("max_cntr_value", c_uint64),
        ("max_err_cntr_value", c_uint64),
    ]


class fi_fabric_attr(Structure):
    _fields_ = [
        ("fabric", POINTER(fid_fabric)),
        ("name", c_char_p),
        ("prov_name", c_char_p),
        ("prov_version", c_uint32),
        ("api_version", c_uint32),
    ]


class fi_info(Structure):
    pass


fi_info._fields_ = [
    ("next", POINTER(fi_info)),
    ("caps", c_uint64),
    ("mode", c_uint64),
    ("addr_format", c_uint32),
    ("src_addrlen", c_size_t),
    ("dest_addrlen", c_size_t),
    ("src_addr", c_void_p),
    ("dest_addr", c_void_p),
    ("handle", POINTER(fid)),
    ("tx_attr", POINTER(fi_tx_attr)),
    ("rx_attr", POINTER(fi_rx_attr)),
    ("ep_attr", POINTER(fi_ep_attr)),
    ("domain_attr", POINTER(fi_domain_attr)),
    ("fabric_attr", POINTER(fi_fabric_attr)),
    ("nic", POINTER(fid_nic)),
]


# -- Library loading and function binding --

_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = load_library("fabric")
        _lib.fi_getinfo.argtypes = [
            c_uint32, c_char_p, c_char_p, c_uint64,
            POINTER(fi_info), POINTER(POINTER(fi_info)),
        ]
        _lib.fi_getinfo.restype = c_int
        _lib.fi_freeinfo.argtypes = [POINTER(fi_info)]
        _lib.fi_freeinfo.restype = None
        _lib.fi_allocinfo.argtypes = []
        _lib.fi_allocinfo.restype = POINTER(fi_info)
        _lib.fi_version.argtypes = []
        _lib.fi_version.restype = c_uint32
    return _lib


# -- Public API --

def version():
    """Return the libfabric version as (major, minor) tuple."""
    lib = _get_lib()
    v = lib.fi_version()
    return (v >> 16, v & 0xFFFF)


def get_info(provider=None, ep_type=None, caps=0, node=None, service=None, flags=0):
    """
    Call fi_getinfo and return a list of fi_info results.

    Args:
        provider: Provider name filter (e.g. "efa", "tcp")
        ep_type: Endpoint type (FI_EP_RDM, FI_EP_DGRAM, etc.)
        caps: Capability flags
        node: Node address
        service: Service/port
        flags: fi_getinfo flags

    Returns:
        List of fi_info pointer objects. Caller must call free_info() when done.
    """
    lib = _get_lib()
    ver = lib.fi_version()

    hints = lib.fi_allocinfo()
    if not hints:
        raise MemoryError("fi_allocinfo failed")

    if caps:
        hints.contents.caps = caps

    if ep_type is not None:
        hints.contents.ep_attr.contents.type = ep_type

    if provider:
        prov_bytes = provider.encode() if isinstance(provider, str) else provider
        hints.contents.fabric_attr.contents.prov_name = prov_bytes

    node_b = node.encode() if isinstance(node, str) else node
    service_b = service.encode() if isinstance(service, str) else service

    info_ptr = POINTER(fi_info)()
    ret = lib.fi_getinfo(ver, node_b, service_b, flags, hints, ctypes.byref(info_ptr))

    lib.fi_freeinfo(hints)

    if ret != 0:
        return []

    results = []
    current = info_ptr
    while current:
        results.append(current)
        nxt = current.contents.next
        current = nxt if nxt else None

    return InfoList(info_ptr, results)


class InfoList:
    """
    Wrapper around fi_info linked list that supports iteration and auto-cleanup.
    """

    def __init__(self, head_ptr, items):
        self._head = head_ptr
        self._items = items
        self._freed = False

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]

    def __bool__(self):
        return len(self._items) > 0

    def free(self):
        if not self._freed and self._head:
            _get_lib().fi_freeinfo(self._head)
            self._freed = True

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.free()
