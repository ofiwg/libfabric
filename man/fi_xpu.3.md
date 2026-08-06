---
layout: page
title: fi_xpu(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_xpu - OFI Accelerator API

# SYNOPSIS

```c
#include <rdma/fi_xpu.h>
```

# OVERVIEW

The XPU capability allows for a specified XPU (or device) to access a given
libfabric allocated resource for data path operations, with control operations
being managed by host (CPU) software. A resource (endpoint, completion queue,
counter, etc.) which has been configured with this capability may only be
accessed by the given XPU for any data path operation, such as submitting data
transfers or reading completions.

As a general rule, control path operations may only be invoked by the host CPU,
while data path functions may only be accessed by a specified XPU. The following
objects may be exported to an XPU: AV, EP, CQ, counters, and locally accessed
MRs.

# fi_xpu_attr

The `fi_xpu_attr` structure is an input/output argument. The application
fills in the input fields (`iface`, `device`, and optionally the memory
callbacks) before passing it to a resource creation call. The provider reads
these inputs during resource creation and writes back output fields
(`xpu_key_size`) to indicate provider-specific parameters. The application
must not modify the structure while it is referenced by an open resource,
and should read output fields only after the creation call returns
successfully.

```c
struct fi_xpu_attr {
    enum fi_hmem_iface   iface;
    uint64_t             device;

    int (*alloc)(uint64_t device, uint64_t size,
                 uint64_t alignment, uint64_t flags,
                 void **addr, int *fd, uint64_t *offset);

    int (*import)(uint64_t device, void *host_addr,
                  uint64_t size, uint64_t flags,
                  void **dev_addr);

    void (*free)(uint64_t device, void *addr);

    size_t               xpu_key_size;
};
```

*iface*
: [Input] The heterogeneous memory interface type (e.g., FI_HMEM_CUDA, FI_HMEM_ZE).

*device*
: [Input] The accelerator device ordinal.

*xpu_key_size*
: [Output] Set by the provider during resource creation. Indicates the size
  in bytes of the raw key returned by `fi_av_get_xpu_addr` (per AV entry)
  or `fi_mr_get_xpu_desc` (per MR). All keys for a given object type have
  the same size. The value may differ between object types (e.g., AV key
  size may differ from MR descriptor size).

## Memory Callbacks

The following callbacks are optional input fields. If provided, the provider
uses them to manage accelerator memory. If not provided (NULL), the provider
manages device memory internally using the libfabric HMEM infrastructure.

*alloc*
: Allocate accelerator memory. The provider calls this when it needs
  device memory (e.g., for hardware counters or CQ buffers). If the
  FI_XPU_ALLOC_DMABUF flag is set, the consumer must also export a
  DMA-BUF file descriptor so the provider can DMA directly to the memory.

*import*
: Map a host virtual address into the accelerator address space. The
  provider calls this when it has a host-side address (BAR MMIO or host
  RAM) that the accelerator kernel needs to access. Flags indicate the
  memory type: FI_XPU_IMPORT_IOMEMORY for PCIe BAR MMIO regions,
  FI_XPU_IMPORT_DEVICEMAP for addresses that must be accessible from
  accelerator kernels.

*free*
: Release memory previously allocated via alloc.

## Flags

*FI_XPU_ALLOC_DMABUF*
: Passed to the alloc callback. Indicates the allocation must be
  exportable as a DMA-BUF fd for provider access.

*FI_XPU_IMPORT_IOMEMORY*
: Passed to the import callback. Indicates the host address points to
  PCIe BAR MMIO (device I/O memory).

*FI_XPU_IMPORT_DEVICEMAP*
: Passed to the import callback. Indicates the resulting pointer must be
  accessible from accelerator kernel code.

# PER-OBJECT SEMANTICS

To create an XPU-capable resource, the application provides a pointer to
`struct fi_xpu_attr` in the corresponding attribute structure and indicates
the FI_XPU capability through the appropriate mechanism for each object.
The provider uses the `fi_xpu_attr` to identify the target device and, if
memory callbacks are provided, to allocate or map device memory during
resource creation. After creation, the resource is bound, enabled, and then
exported or queried as described below.

## AV

Created by setting `FI_XPU` in `fi_av_attr->flags` and
`fi_av_attr->xpu_attr` before calling `fi_av_open`.

All APIs of an AV created with FI_XPU are available only to the host CPU.
However, the AV may only be bound to EPs configured with FI_XPU.

The AV metadata remains on the host or the provider. That is, creating an AV
with FI_XPU does not create an AV on the XPU. The process of mapping an
application-visible address (e.g., sockaddr) to a transport-specific format
is still the responsibility of the host. The provider returns a "raw address"
that is usable by the XPU when posting work requests. For some implementations,
the raw address may simply be an index.

The application queries `xpu_key_size` from `struct fi_xpu_attr`, allocates
a buffer, calls `fi_av_get_xpu_addr` for each entry to retrieve the raw
address, and copies the results to device-accessible memory.

## EP

Created by calling `fi_endpoint2` with `FI_XPU` in the flags parameter and
setting `fi_ep_attr->xpu_attr`.

Data transfer operations on an EP created with FI_XPU are only available to
the given XPU. This includes msg (fi_msg.3), rma (fi_rma.3), tagged
(fi_tagged.3), atomic (fi_atomic.3), and collective APIs. Control functions
(fi_ep_bind, fi_enable, fi_setopt, CM operations) are available only to the
host CPU.

There is no restriction that an EP created with FI_XPU must be bound to
other resources also created with FI_XPU. For example, an XPU EP may be
bound to a standard CQ that was not created with FI_XPU, allowing the
host CPU to poll completions.

Once bound and enabled, the EP is exported via `fi_ep_export_xpu`. The
provider maps HW queue geometry (SQ, doorbell, RQ) into the accelerator
address space and returns an opaque handle usable by device-side functions.

## CQ

Created by setting `FI_XPU` in `fi_cq_attr->flags` and
`fi_cq_attr->xpu_attr` before calling `fi_cq_open`.

Completion entries on a CQ created with FI_XPU can only be read using the
device-side functions (`fi_xpu_cq_read`, `fi_xpu_cq_readerr`). Host-side
read operations (fi_cq_read, fi_cq_sread) are not available. Host-side
control operations (fi_cq_open, fi_close, fi_control) remain CPU-only.

The CQ is exported via `fi_cq_export_xpu`.

## Counters

Created by setting `FI_XPU` in `fi_cntr_attr->flags` and
`fi_cntr_attr->xpu_attr` before calling `fi_cntr_open`.

Counter values reside in XPU-accessible memory. Typically, the provider
writes completions directly via DMA. A counter created with FI_XPU can
only be read or waited on using device-side functions (`fi_xpu_cntr_read`,
`fi_xpu_cntr_wait`, etc.). Host-side read operations (fi_cntr_read,
fi_cntr_wait) are not available. Control operations (fi_cntr_open,
fi_close) are CPU-only.

Counters are exported via `fi_cntr_export_xpu`.

## MR

Created by passing `FI_XPU` in the flags parameter of `fi_mr_regattr` and
setting `fi_mr_attr->xpu_attr`.

Creating a MR with FI_XPU indicates that the MR will be accessed as a local
buffer as part of a data transfer submitted to an EP configured with FI_XPU.
For example, the MR may be used as a send or receive buffer for a message
operation submitted by an XPU. Such MRs should only be accessed by the
specified XPU, as the backing physical pages may not be accessible to the
host CPU or other XPUs.

The provider returns a "raw descriptor" (e.g., the hardware lkey) that is
usable by the XPU when posting work requests. The metadata behind the MR
remains on the host or the provider (e.g., VA, size, permissions). The physical
pages backing the MR are likely to be local to the XPU, but that is a
separate feature.

The desc_key replaces the desc parameter in data transfer operations that
are initiated by the XPU. The application queries `xpu_key_size` from
`struct fi_xpu_attr`, allocates a buffer, calls `fi_mr_get_xpu_desc` to
retrieve the raw descriptor, and copies the result to device-accessible
memory.

# HOST-SIDE FUNCTIONS

Once objects are created with FI_XPU and xpu_attr, bound, and enabled, they
are made accessible to the XPU using the following functions:

## fi_ep_export_xpu

```c
int fi_ep_export_xpu(struct fid_ep *ep, uint64_t flags,
                     void **xpu_ep, size_t *size);
```

Export an enabled endpoint for accelerator access. Returns an opaque handle
usable by device-side data transfer functions.

## fi_cq_export_xpu

```c
int fi_cq_export_xpu(struct fid_cq *cq, uint64_t flags,
                     void **xpu_cq, size_t *size);
```

Export a CQ for accelerator-side completion polling. Returns an opaque
handle usable by device-side `fi_xpu_cq_read` and `fi_xpu_cq_readerr`.

## fi_cntr_export_xpu

```c
int fi_cntr_export_xpu(struct fid_cntr *cntr, uint64_t flags,
                       void **xpu_cntr, size_t *size);
```

Export a counter for accelerator access. Returns an opaque handle usable
by device-side `fi_xpu_cntr_read`, `fi_xpu_cntr_wait`, and related
functions.

## fi_av_get_xpu_addr

```c
int fi_av_get_xpu_addr(struct fid_av *av, fi_addr_t fi_addr,
                       void *buf, size_t *len);
```

Retrieve the provider-specific raw address for a single AV entry, suitable
for use by an XPU when posting work requests. This is conceptually similar
to `fi_av_lookup`, which returns the application-level address (e.g.,
sockaddr); `fi_av_get_xpu_addr` instead returns a transport-level
representation that the XPU can write directly into a work queue entry. A
provider may use the same underlying implementation for both.

The caller allocates `buf` with at least `xpu_key_size` bytes. On input,
`*len` is the size of `buf`; on output it is the number of bytes written.
The application is responsible for copying the result to device-accessible
memory.

## fi_mr_get_xpu_desc

```c
int fi_mr_get_xpu_desc(struct fid_mr *mr, void *buf, size_t *len);
```

Retrieve the provider-specific raw descriptor key for a single MR, suitable
for use by an XPU when posting work requests. This is conceptually similar
to `fi_mr_desc`, which returns an opaque host-side descriptor pointer;
`fi_mr_get_xpu_desc` instead returns the raw bytes (e.g., the hardware lkey)
that the XPU writes directly into a work queue entry. A provider may use
the same underlying value for both.

The caller allocates `buf` with at least `xpu_key_size` bytes. On input,
`*len` is the size of `buf`; on output it is the number of bytes written.
The application is responsible for copying the result to device-accessible
memory.

## Return Values

All host-side functions return 0 on success or a negative errno value on
failure. Common error codes:

*FI_SUCCESS (0)*
: Operation completed successfully.

*-FI_ENOSYS*
: The provider does not support this operation.

*-FI_EINVAL*
: Invalid argument (e.g., NULL buffer, unsupported flags, fi_addr not
  found in AV).

*-FI_ETOOSMALL*
: The provided buffer is too small. For `fi_av_get_xpu_addr` and
  `fi_mr_get_xpu_desc`, `*len` is updated to the required size.

*-FI_ENODATA*
: The requested entry does not exist or has not been resolved.

*-FI_EOPBADSTATE*
: The object is not in the correct state (e.g., EP not enabled before
  export).

# DEVICE-SIDE API

The device-side API provides communication functions callable from accelerator
kernels. Include the following header:

```c
#include <rdma/fi_xpu_device.h>
```

## Scope

The scope parameter specifies the concurrency of the operation. All threads
in the scope are required to issue the same operation. This allows the
implementation to optimize.

| Scope | CUDA | SYCL |
|-------|------|------|
| FI_XPU_WORK_ITEM | Thread | Work item |
| FI_XPU_SUBGROUP | Warp | Subgroup |
| FI_XPU_WORK_GROUP | Thread block | Work group |
| FI_XPU_DEVICE | Device | Device |

## Data Transfer Operations

All data transfer functions below are callable only from XPU kernel code.
They correspond to the standard libfabric data transfer APIs (fi_msg, fi_rma,
fi_tagged, fi_atomic) but use opaque handles obtained via the host-side
functions.

```c
/* RMA */
int fi_xpu_write(void *xpu_ep, const void *buf, void *desc, uint64_t size,
                 uint64_t data, void *peer, uint64_t raddr, uint64_t rkey,
                 void *ctxt, int scope, uint64_t flags);
int fi_xpu_read(void *xpu_ep, void *buf, void *desc, uint64_t size,
                void *peer, uint64_t raddr, uint64_t rkey,
                void *ctxt, int scope, uint64_t flags);

/* Message */
int fi_xpu_send(void *xpu_ep, const void *buf, uint64_t size, void *desc,
                uint64_t data, void *peer, void *ctxt,
                int scope, uint64_t flags);
int fi_xpu_recv(void *xpu_ep, void *buf, void *desc, uint64_t size,
                void *peer, void *ctxt, int scope, uint64_t flags);

/* Tagged */
int fi_xpu_tsend(void *xpu_ep, const void *buf, uint64_t size, void *desc,
                 uint64_t data, void *peer, uint64_t tag, void *ctxt,
                 int scope, uint64_t flags);
int fi_xpu_trecv(void *xpu_ep, void *buf, uint64_t size, void *desc,
                 void *peer, uint64_t tag, uint64_t ignore, void *ctxt,
                 int scope, uint64_t flags);

/* Atomic */
int fi_xpu_atomic(void *xpu_ep, const void *buf, size_t count,
                  void *desc, void *peer, uint64_t addr, uint64_t key,
                  int datatype, int op, void *ctxt,
                  int scope, uint64_t flags);
int fi_xpu_fetch_atomic(void *xpu_ep, const void *buf, size_t count,
                        void *desc, void *result, void *result_desc,
                        void *peer, uint64_t addr, uint64_t key,
                        int datatype, int op, void *ctxt,
                        int scope, uint64_t flags);
int fi_xpu_compare_atomic(void *xpu_ep, const void *buf, size_t count,
                          void *desc, const void *compare,
                          void *compare_desc, void *result,
                          void *result_desc, void *peer, uint64_t addr,
                          uint64_t key, int datatype, int op, void *ctxt,
                          int scope, uint64_t flags);
```

## Completion Functions — Counter

```c
uint64_t fi_xpu_cntr_read(void *xpu_cntr, int scope);
uint64_t fi_xpu_cntr_readerr(void *xpu_cntr, int scope);
void fi_xpu_cntr_wait(void *xpu_cntr, uint64_t threshold, int timeout, int scope);
int fi_xpu_cntr_add(void *xpu_cntr, uint64_t value, int scope);
int fi_xpu_cntr_set(void *xpu_cntr, uint64_t value, int scope);
int fi_xpu_cntr_adderr(void *xpu_cntr, uint64_t value, int scope);
int fi_xpu_cntr_seterr(void *xpu_cntr, uint64_t value, int scope);
```

## Completion Functions — CQ

```c
int64_t fi_xpu_cq_read(void *xpu_cq, void *buf, size_t count, int scope);
int64_t fi_xpu_cq_readfrom(void *xpu_cq, void *buf, size_t count,
                            void *src_addr, int scope);
int64_t fi_xpu_cq_readerr(void *xpu_cq, void *buf, uint64_t flags, int scope);
int64_t fi_xpu_cq_sread(void *xpu_cq, void *buf, size_t count,
                         uint64_t threshold, int scope);
int64_t fi_xpu_cq_sreadfrom(void *xpu_cq, void *buf, size_t count,
                             void *src_addr, uint64_t threshold, int scope);
```

# EXAMPLE

The following illustrates the typical host-side setup flow for XPU-initiated
RDMA write:

```c
struct fi_xpu_attr xpu_attr = {
    .iface = FI_HMEM_CUDA,
    .device = 0,
    .alloc = my_cuda_alloc,
    .import = my_cuda_import,
    .free = my_cuda_free,
};

/* 1. Query provider for FI_XPU support */
struct fi_info *hints = fi_allocinfo();
hints->caps = FI_MSG | FI_RMA | FI_XPU;
hints->ep_attr->xpu_attr = &xpu_attr;
fi_getinfo(FI_VERSION(2,6), NULL, NULL, 0, hints, &info);

/* 2. Open domain */
fi_domain(fabric, info, &domain, NULL);

/* 3. Create XPU resources */
struct fi_av_attr av_attr = { .type = FI_AV_TABLE,
                              .flags = FI_XPU,
                              .xpu_attr = &xpu_attr };
fi_av_open(domain, &av_attr, &av, NULL);
/* av_attr.xpu_attr->xpu_key_size now set by provider (per AV entry) */
size_t av_key_size = av_attr.xpu_attr->xpu_key_size;

struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_DATA,
                              .flags = FI_XPU,
                              .xpu_attr = &xpu_attr };
fi_cq_open(domain, &cq_attr, &cq, NULL);

fi_endpoint2(domain, info, &ep, FI_XPU, NULL);

struct fi_xpu_attr mr_xpu_attr = xpu_attr;
struct fi_mr_attr mr_attr = { .mr_iov = &iov, .iov_count = 1,
                              .access = FI_SEND | FI_RECV,
                              .xpu_attr = &mr_xpu_attr };
fi_mr_regattr(domain, &mr_attr, FI_XPU, &mr);
/* mr_attr.xpu_attr->xpu_key_size now set by provider (per MR desc) */
size_t mr_key_size = mr_attr.xpu_attr->xpu_key_size;

/* 4. Insert peer address */
fi_av_insert(av, peer_addr, 1, &fi_addr, 0, NULL);

/* 5. Bind and enable EP */
fi_ep_bind(ep, &av->fid, 0);
fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
fi_enable(ep);

/* 6. Export EP/CQ for device-side use */
void *xpu_ep, *xpu_cq;
size_t ep_size, cq_size;
fi_ep_export_xpu(ep, 0, &xpu_ep, &ep_size);
fi_cq_export_xpu(cq, 0, &xpu_cq, &cq_size);

/* 7. Get raw AV addr and MR desc for device-side use */
void *raw_addr = malloc(av_key_size);
void *raw_desc = malloc(mr_key_size);
size_t len = av_key_size;
fi_av_get_xpu_addr(av, fi_addr, raw_addr, &len);
len = mr_key_size;
fi_mr_get_xpu_desc(mr, raw_desc, &len);

/* 8. Copy handles to device-accessible memory (method is device-specific) */
copy_to_device(gpu_ep,   xpu_ep,   ep_size);
copy_to_device(gpu_cq,   xpu_cq,   cq_size);
copy_to_device(gpu_addr, raw_addr,  av_key_size);
copy_to_device(gpu_desc, raw_desc,  mr_key_size);

/* 9. Launch device kernel — device code posts operations and polls
 *    completions using the exported handles:
 *
 *    void my_kernel(void *ep, void *cntr, void *peer, void *desc) {
 *        char buf[64] = "hello";
 *        uint64_t prev = fi_xpu_cntr_read(cntr, FI_XPU_WORK_ITEM);
 *
 *        fi_xpu_send(ep, buf, sizeof(buf), desc, 0, peer, NULL,
 *                    FI_XPU_WORK_ITEM, 0);
 *
 *        fi_xpu_cntr_wait(cntr, prev + 1, -1, FI_XPU_WORK_ITEM);
 *    }
 */
launch_kernel(my_kernel, gpu_ep, gpu_cntr, gpu_addr, gpu_desc);
```

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_mr`(3)](fi_mr.3.html),
[`fi_av`(3)](fi_av.3.html)
