# OFI Python Bindings (`fabtests/pytest/ofi/`)

## Overview

This package provides Python ctypes bindings for libfabric and related
libraries (libibverbs, libefa). It allows pytest tests to call C library
functions directly from Python, eliminating the need to compile and ship
separate C test binaries for simple queries and checks.

## Modules

| Module | Description |
|--------|-------------|
| `_loader.py` | Shared library discovery and loading utility |
| `libfabric.py` | Bindings for libfabric core API (`fi_getinfo`, `fi_freeinfo`, `fi_version`) |
| `verbs.py` | Bindings for libibverbs and libefa (`ibv_get_device_list`, `efadv_query_device`) |
| `efa_rdma_checker.py` | Python replacement for `fi_efa_rdma_checker` C binary |

## Library Discovery

Libraries are located in the following order:

1. `OFI_LIB_PATH` environment variable (colon-separated directories)
2. `LD_LIBRARY_PATH` (handled by the dynamic linker)
3. System library paths (via `ctypes.util.find_library`)

Example:
```bash
export OFI_LIB_PATH=/opt/amazon/efa/lib64:/usr/lib64
```

## Usage Examples

### Check EFA RDMA capability (replaces fi_efa_rdma_checker)

**As a CLI tool (drop-in replacement for the C binary):**
```bash
python3 -m ofi.efa_rdma_checker -o read
python3 -m ofi.efa_rdma_checker -o write
python3 -m ofi.efa_rdma_checker -o writedata
```

**Directly from Python test code:**
```python
from ofi.verbs import check_rdma_capability

if check_rdma_capability("write"):
    # RDMA write is supported
    ...
```

### Query libfabric providers

```python
from ofi.libfabric import get_info, FI_EP_RDM, version

print(f"libfabric version: {version()}")

with get_info(provider="efa", ep_type=FI_EP_RDM) as infos:
    for info in infos:
        name = info.contents.domain_attr.contents.name
        print(f"  domain: {name.decode()}")
```

### Enumerate IB devices

```python
from ofi.verbs import DeviceList, query_efa_device

with DeviceList() as devices:
    for dev_ptr, name in devices:
        print(f"Device: {name}")
        attr = query_efa_device(dev_ptr)
        if attr:
            print(f"  max_rdma_size: {attr.max_rdma_size}")
            print(f"  device_caps: {attr.device_caps:#x}")
```

## Adding New Bindings

To wrap additional library functions:

1. Define ctypes structures matching the C headers
2. Use `load_library("name")` from `_loader.py` to load the `.so`
3. Set `.argtypes` and `.restype` on the function object
4. Provide a Pythonic wrapper function

## Notes

- The `efadv_device_attr` struct layout is based on rdma-core. If the struct
  changes in future rdma-core versions, update `verbs.py` accordingly.
- Device pointers from `DeviceList` are only valid while the `DeviceList`
  context manager is active (before `ibv_free_device_list` is called).
- On macOS, the libraries won't load (no EFA/verbs support) — this is
  expected. The bindings are designed for Linux hosts with EFA hardware.
