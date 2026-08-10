---
layout: page
title: fi_xpu(3)
tagline: Libfabric Programmer's Manual
---
{% include JB/setup %}

# NAME

fi_xpu - OFI XPU API

# SYNOPSIS

```c
#include <rdma/fi_xpu.h>
```

# OVERVIEW

The FI_XPU capability allows for a specified XPU (or device) to access a given
libfabric allocated resource for data path operations, with control operations
being managed by host (CPU) software. A resource (endpoint, completion queue,
counter) which has been configured with this capability may only be
accessed by the given XPU for any data path operation, such as submitting data
transfers or reading completions.

As a general rule, control path operations may only be invoked by the host CPU,
while data path functions may only be accessed by a specified XPU. The following
objects may be exported to an XPU: EP, CQ, and counters. AV and MR lookups
produce raw data usable by the XPU but the objects themselves remain host-only.
Detailed XPU behavior is documented in the corresponding man pages
([`fi_av`(3)](fi_av.3.html), [`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_cq`(3)](fi_cq.3.html), [`fi_cntr`(3)](fi_cntr.3.html),
[`fi_mr`(3)](fi_mr.3.html)).

# XPU CONTEXT

Related EP, CQ, and counter resources created with FI_XPU must all target
the same XPU. The `fid_xpu_ctx` object groups these resources together and
binds them to a specific device. An EP created with FI_XPU may also be
bound to a host (non-XPU) CQ for error handling. It also provides the avenue
for querying provider-specific sizes needed to interact with AV and MR data
from the XPU.

An XPU context is created from a domain:

```c
int fi_xpu_ctx(struct fid_domain *domain, struct fi_xpu_attr *attr,
               struct fid_xpu_ctx **ctx, void *context);
```

The number of XPU contexts that a domain supports is indicated by the
`max_xpu_ctx_cnt` field in `fi_domain_attr`. A value of 0 indicates
no support for XPU-initiated communication. A value of 1 indicates a 1:1
mapping between a domain and an XPU.

The XPU context is passed to EP, CQ, and counter creation calls via a
pointer in their respective attribute structures (`fi_ep_attr->xpu_ctx`,
`fi_cq_attr->xpu_ctx`, `fi_cntr_attr->xpu_ctx`). XPU resources bound to the
same EP must share the same XPU context. A host CQ (without xpu_ctx set) may
also be bound to an XPU EP for error handling.

The XPU context is also passed to `fi_av_lookup2` and `fi_mr_get_xpu_desc` so
that the provider can return data in the appropriate format for the target XPU.
This allows AV and MR objects to be shared across multiple XPUs — the same AV
or MR can be queried with different XPU contexts to get device-specific
representations.

# fi_xpu_attr

The `fi_xpu_attr` structure is an input argument that identifies the target
XPU device and provides optional memory management callbacks. It is passed
to `fi_xpu_ctx()` during context creation.

```c
struct fi_xpu_ops {
    size_t size;
    int (*alloc)(uint64_t device, uint64_t size,
                 uint64_t alignment, uint64_t flags,
                 void **addr, int *fd, uint64_t *offset);
    int (*import)(uint64_t device, void *host_addr,
                  uint64_t size, uint64_t flags,
                  void **dev_addr);
    void (*free)(uint64_t device, void *addr);
};

struct fi_xpu_attr {
    int                  iface;    /* enum fi_hmem_iface */
    uint64_t             device;
    struct fi_xpu_ops    *ops;
};
```

*iface*
: The heterogeneous memory interface type (e.g., FI_HMEM_CUDA, FI_HMEM_ZE).

*device*
: The XPU device ordinal.

*ops*
: Optional pointer to a `fi_xpu_ops` structure containing memory management
  callbacks. If NULL, the provider uses its default mechanisms.

## fi_xpu_ops

The `fi_xpu_ops` structure groups memory management callbacks for XPU
device memory. The provider calls these when it needs to allocate, import,
or free device memory on behalf of the XPU context.

*size*
: Must be set to `sizeof(struct fi_xpu_ops)`. This allows future expansion
  of the structure while maintaining backward compatibility. The provider
  uses this field to determine which callback fields are present.

*alloc*
: Allocate XPU memory. The provider calls this when it needs device memory
  (e.g., for hardware queue buffers). If the `FI_XPU_ALLOC_DMABUF` flag is
  set, the consumer must also export a DMA-BUF file descriptor so the
  provider can DMA directly to the memory.

*import*
: Map a host virtual address into the XPU address space. The provider calls
  this when it has a host-side address (BAR MMIO or host RAM) that the XPU
  kernel needs to access. Flags indicate the memory type:
  `FI_XPU_IMPORT_IOMEMORY` for PCIe BAR MMIO regions,
  `FI_XPU_IMPORT_DEVICEMAP` for addresses that must be accessible from
  XPU kernels.

*free*
: Release memory previously allocated via alloc.

## Memory Callback Flags

*FI_XPU_ALLOC_DMABUF*
: Passed to the alloc callback. Indicates the allocation must be
  exportable as a DMA-BUF fd for provider access.

*FI_XPU_IMPORT_IOMEMORY*
: Passed to the import callback. Indicates the host address points to
  PCIe BAR MMIO (device I/O memory).

*FI_XPU_IMPORT_DEVICEMAP*
: Passed to the import callback. Indicates the resulting pointer must be
  accessible from XPU kernel code.

# fi_xpu_ctx_attr

The `fi_xpu_ctx_attr` structure is returned by `fi_xpu_ctx_query()`. It
contains provider-specific output parameters for the given XPU context.

```c
#define FI_XPU_CAP_EP      (1ULL << 0)
#define FI_XPU_CAP_CQ      (1ULL << 1)
#define FI_XPU_CAP_CNTR    (1ULL << 2)

struct fi_xpu_ctx_attr {
    uint64_t     caps;
    size_t       av_addr_size;
    size_t       mr_desc_size;
};
```

*caps*
: Bitmask of XPU capabilities supported by this provider. The application
  should check these flags before attempting to create XPU resources.
  Attempting to create an unsupported resource type will return -FI_ENOSYS.

  - `FI_XPU_CAP_EP`: Provider supports XPU endpoints. This includes:
    device-side data transfer operations (post/dispatch) for the
    capabilities returned by `fi_getinfo` (e.g., FI_MSG, FI_TAGGED, FI_RMA,
    FI_ATOMIC), `fi_ep_export_xpu` to export the endpoint for device access,
    `fi_av_lookup2` with FI_XPU to retrieve raw AV addresses, and
    `fi_mr_get_xpu_desc` with FI_XPU to retrieve raw MR descriptors.
  - `FI_XPU_CAP_CQ`: Provider supports XPU completion queues. This includes:
    `fi_cq_export_xpu` to export the CQ for device access, and the
    device-side CQ functions (`fi_xpu_cq_read`, `fi_xpu_cq_readfrom`,
    `fi_xpu_cq_readerr`, `fi_xpu_cq_sread`, `fi_xpu_cq_sreadfrom`).
  - `FI_XPU_CAP_CNTR`: Provider supports XPU counters. This includes:
    `fi_cntr_export_xpu` to export the counter for device access, and the
    device-side counter functions (`fi_xpu_cntr_read`, `fi_xpu_cntr_readerr`,
    `fi_xpu_cntr_wait`, `fi_xpu_cntr_add`, `fi_xpu_cntr_set`,
    `fi_xpu_cntr_adderr`, `fi_xpu_cntr_seterr`).

*av_addr_size*
: Size in bytes of the raw address returned by `fi_av_lookup2` (when called
  with `FI_XPU` flag) for each AV entry. All AV entries for a given context
  have the same size.

*mr_desc_size*
: Size in bytes of the raw descriptor returned by `fi_mr_get_xpu_desc` (when
  called with `FI_XPU` flag). All descriptors for a given context have the
  same size.

## fi_xpu_ctx_query

```c
int fi_xpu_ctx_query(struct fid_xpu_ctx *ctx,
                     struct fi_xpu_ctx_attr *attr);
```

Query the provider for XPU context parameters. The returned `caps` field
indicates which XPU objects the provider supports (FI_XPU_CAP_EP,
FI_XPU_CAP_CQ, FI_XPU_CAP_CNTR). The application should check these flags
before attempting to create XPU resources. The returned `av_addr_size` and
`mr_desc_size` fields are used to allocate appropriately sized buffers for
`fi_av_lookup2` and `fi_mr_get_xpu_desc` calls. Different XPU contexts (targeting
different devices) may report different sizes.

## EP

An XPU EP is created by calling `fi_endpoint2` with `FI_XPU` in the flags
parameter and setting `fi_ep_attr->xpu_ctx` to an open XPU context.

An EP created with FI_XPU may be bound to XPU CQs and XPU counters created
with the same XPU context. Binding to a host (non-XPU) CQ or counter is
allowed but behavior is provider-specific. Control functions such
as [`fi_ep_bind`](fi_ep_bind.3.html), [`fi_enable`](fi_endpoint.3.html),
[`fi_setopt`](fi_endpoint.3.html), and CM
operations are available only to the host CPU.

Once bound and enabled, `fi_ep_export_xpu` exports the EP for device access.
The caller provides a `struct fid_xpu_ep` which the provider fills with the
information needed for device-side data transfer functions.

Data transfer operations on the exported EP are only available to the given
XPU. This includes [`fi_msg`(3)](fi_msg.3.html),
[`fi_rma`(3)](fi_rma.3.html), [`fi_tagged`(3)](fi_tagged.3.html),
[`fi_atomic`(3)](fi_atomic.3.html). See
[`fi_endpoint`(3)](fi_endpoint.3.html) for details.

## CQ

An XPU CQ is created by setting `FI_XPU` in `fi_cq_attr->flags` and
`fi_cq_attr->xpu_ctx` to an open XPU context before calling `fi_cq_open`.
Control operations (`fi_cq_open`, `fi_close`, `fi_control`) remain CPU-only.

`fi_cq_export_xpu` exports the CQ for device access. The caller provides a
`struct fid_xpu_cq` which the provider fills with the information needed for
the device-side completion functions (`fi_xpu_cq_read`,
`fi_xpu_cq_readerr`). Host-side read operations (`fi_cq_read`, `fi_cq_sread`)
are not available on an exported CQ. See
[`fi_cq`(3)](fi_cq.3.html) for details.

## Counters

An XPU counter is created by setting `FI_XPU` in `fi_cntr_attr->flags` and
`fi_cntr_attr->xpu_ctx` to an open XPU context before calling `fi_cntr_open`.
Control operations (`fi_cntr_open`, `fi_close`) remain CPU-only.

`fi_cntr_export_xpu` exports the counter for device access. The caller provides
a `struct fid_xpu_cntr` which the provider fills with the information needed
for the device-side counter functions (`fi_xpu_cntr_read`,
`fi_xpu_cntr_wait`). Host-side read operations (`fi_cntr_read`,
`fi_cntr_wait`) are not available on an exported counter. See
[`fi_cntr`(3)](fi_cntr.3.html) for details.

## AV

The AV is not bound to an XPU context — it is a shared domain-level resource.

The application queries `av_addr_size` from `fi_xpu_ctx_query()`, allocates
a buffer of that size, calls `fi_av_lookup2` with `FI_XPU` flag and the
XPU context for each entry to retrieve the raw address, and copies the
results to device-accessible memory. The raw address is a provider-specific
representation usable by the XPU when posting work requests.

`fi_av_lookup2` is an extended version of `fi_av_lookup` that accepts flags
and an XPU context, allowing the same AV to be queried for different XPU
devices. See [`fi_av`(3)](fi_av.3.html) for details.

## MR

The MR is not bound to an XPU context — it is a shared domain-level resource.
The same MR can be queried with different XPU contexts.

The application queries `mr_desc_size` from `fi_xpu_ctx_query()`, allocates
a buffer, calls `fi_mr_get_xpu_desc` with the `FI_XPU` flag and the XPU context
to retrieve the raw descriptor (e.g., the hardware lkey), and copies the
result to device-accessible memory. The raw descriptor replaces the desc
parameter in data transfer operations initiated by the XPU.

`fi_mr_get_xpu_desc` is an extended descriptor query that accepts flags and an
XPU context, allowing the same MR to be queried for different XPU devices.
See [`fi_mr`(3)](fi_mr.3.html) for details.

# DEVICE-SIDE API

The device-side API provides communication functions callable from XPU
kernels. Include the following header:

```c
#include <rdma/fi_xpu_device.h>
```

This header is compiled with a single XPU kernel compiler at a time. It
supports XPU programming environments that provide a C/C++ interface:

- NVIDIA CUDA (nvcc)
- AMD ROCm HIP (hipcc)
- Intel oneAPI Level Zero / SYCL (icpx -fsycl)

The same header covers all of the above — the `FI_XPU_FUNC` macro adapts
the function qualifier (`__device__`, `static inline`, etc.) based on the
compiler detected at build time.

## Provider Dispatch Model

The device-side header uses a provider-identifier based dispatch model.
Each exported XPU handle (`fid_xpu_ep`, `fid_xpu_cq`, `fid_xpu_cntr`)
embeds `struct fid_xpu` as its first member. The provider populates
`fid_xpu.prov_id` during the export call (`fi_ep_export_xpu`,
`fi_cq_export_xpu`, `fi_cntr_export_xpu`) with its assigned value from
`enum fi_xpu_provider`:

```c
enum fi_xpu_provider {
    FI_XPU_PROV_EFA = 1,
};

struct fid_xpu {
    uint32_t fclass;    /* FI_CLASS_EP, _CQ, _CNTR */
    uint32_t prov_id;   /* enum fi_xpu_provider */
    uint64_t prov_ctx;  /* provider-internal context */
};

struct fid_xpu_ep {
    struct fid_xpu fid;
};

struct fid_xpu_cq {
    struct fid_xpu fid;
};

struct fid_xpu_cntr {
    struct fid_xpu fid;
};
```

The generic dispatch functions take the typed handle (`struct fid_xpu_ep *`,
`fid_xpu_cq *`, or `fid_xpu_cntr *`) and switch on `fid.prov_id` to route to
the appropriate provider-specific implementation. The `prov_ctx` field
provides a mechanism for the provider to locate all state associated with
the resource — the provider stores everything in device memory allocated
via the XPU ops callbacks, and `prov_ctx` holds the address where that
state resides.

The tight range of provider IDs (starting at 1) allows the compiler to
generate an efficient jump table rather than a chain of comparisons.

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

Device-side data transfer operations correspond to the standard libfabric
APIs. See the following man pages for function signatures and semantics:

- Message operations (fi_xpu_send, fi_xpu_recv):
  [`fi_msg`(3)](fi_msg.3.html)
- Tagged operations (fi_xpu_tsend, fi_xpu_trecv):
  [`fi_tagged`(3)](fi_tagged.3.html)
- RMA operations (fi_xpu_write, fi_xpu_read):
  [`fi_rma`(3)](fi_rma.3.html)
- Atomic operations (fi_xpu_atomic, fi_xpu_fetch_atomic,
  fi_xpu_compare_atomic):
  [`fi_atomic`(3)](fi_atomic.3.html)

## Completion Functions

Device-side completion functions operate on exported CQ and counter handles.
See the following man pages for function signatures and semantics:

- Counter operations (fi_xpu_cntr_read, fi_xpu_cntr_wait, etc.):
  [`fi_cntr`(3)](fi_cntr.3.html)
- CQ operations (fi_xpu_cq_read, fi_xpu_cq_readerr, etc.):
  [`fi_cq`(3)](fi_cq.3.html)

# EXAMPLE

The following illustrates the typical host-side setup flow for XPU-initiated
RDMA communication using the XPU context:

```c
/* 1. Query provider for FI_XPU support */
struct fi_info *hints = fi_allocinfo();
hints->caps = FI_MSG | FI_RMA | FI_XPU;
fi_getinfo(FI_VERSION(2,7), NULL, NULL, 0, hints, &info);
/* info->domain_attr->max_xpu_ctx_cnt > 0 confirms support */

/* 2. Open domain */
fi_domain(fabric, info, &domain, NULL);

/* 3. Create XPU context for GPU 0 (with memory callbacks) */
struct fi_xpu_ops my_ops = {
    .size = sizeof(struct fi_xpu_ops),
    .alloc = my_cuda_alloc,
    .import = my_cuda_import,
    .free = my_cuda_free,
};
struct fi_xpu_attr xpu_attr = {
    .iface = FI_HMEM_CUDA,
    .device = 0,
    .ops = &my_ops,
};
struct fid_xpu_ctx *xpu_ctx;
fi_xpu_ctx(domain, &xpu_attr, &xpu_ctx, NULL);

/* 4. Query context for sizes and capabilities */
struct fi_xpu_ctx_attr ctx_attr;
fi_xpu_ctx_query(xpu_ctx, &ctx_attr);
/* ctx_attr.caps indicates supported objects (FI_XPU_CAP_EP, etc.) */
size_t av_entry_size = ctx_attr.av_addr_size;
size_t desc_size = ctx_attr.mr_desc_size;

/* 5. Create AV (domain-level, no xpu_ctx binding) */
struct fi_av_attr av_attr = { .type = FI_AV_TABLE };
fi_av_open(domain, &av_attr, &av, NULL);
fi_av_insert(av, peer_addr, 1, &fi_addr, 0, NULL);

/* 6. Create CQ and counter with xpu_ctx */
struct fi_cq_attr cq_attr = { .format = FI_CQ_FORMAT_DATA,
                              .xpu_ctx = xpu_ctx };
fi_cq_open(domain, &cq_attr, &cq, NULL);

struct fi_cntr_attr cntr_attr = { .events = FI_CNTR_EVENTS_COMP,
                                  .xpu_ctx = xpu_ctx };
fi_cntr_open(domain, &cntr_attr, &cntr, NULL);

/* 7. Create EP with xpu_ctx */
info->ep_attr->xpu_ctx = xpu_ctx;
fi_endpoint2(domain, info, &ep, FI_XPU, NULL);

/* 8. Register MR (domain-level, no special flags) */
struct fi_mr_attr mr_attr = { .mr_iov = &iov, .iov_count = 1,
                              .access = FI_SEND | FI_RECV };
fi_mr_regattr(domain, &mr_attr, 0, &mr);

/* 9. Bind and enable EP */
fi_ep_bind(ep, &av->fid, 0);
fi_ep_bind(ep, &cq->fid, FI_TRANSMIT | FI_RECV);
fi_ep_bind(ep, &cntr->fid, FI_SEND);
fi_enable(ep);

/* 10. Export EP/CQ/counter for device-side use */
struct fid_xpu_ep xpu_ep;
struct fid_xpu_cq xpu_cq;
struct fid_xpu_cntr xpu_cntr;
fi_ep_export_xpu(ep, 0, &xpu_ep);
fi_cq_export_xpu(cq, 0, &xpu_cq);
fi_cntr_export_xpu(cntr, 0, &xpu_cntr);

/* 11. Get raw AV addr and MR desc for device-side use */
void *raw_addr = malloc(av_entry_size);
size_t len = av_entry_size;
fi_av_lookup2(av, fi_addr, raw_addr, &len, FI_XPU, xpu_ctx);

void *raw_desc = malloc(desc_size);
len = desc_size;
fi_mr_get_xpu_desc(mr, raw_desc, &len, FI_XPU, xpu_ctx);

/* 12. Copy raw AV addr, MR desc, and exported handles to device memory */
copy_to_device(gpu_addr, raw_addr, av_entry_size);
copy_to_device(gpu_desc, raw_desc, desc_size);
copy_to_device(gpu_xpu_ep, &xpu_ep, sizeof(xpu_ep));
copy_to_device(gpu_xpu_cntr, &xpu_cntr, sizeof(xpu_cntr));

/* 13. Launch device kernel — posts operations and polls completions
 *     using exported handles (already device-accessible):
 *
 *     __global__ void my_kernel(struct fid_xpu_ep *ep,
 *                               struct fid_xpu_cntr *cntr,
 *                               void *peer, void *desc, void *buf) {
 *         uint64_t prev = fi_xpu_cntr_read(cntr, FI_XPU_WORK_ITEM);
 *
 *         fi_xpu_send(ep, buf, 64, desc, 0, peer, NULL,
 *                     0, FI_XPU_WORK_ITEM);
 *
 *         fi_xpu_cntr_wait(cntr, prev + 1, -1, FI_XPU_WORK_ITEM);
 *     }
 */
launch_kernel(my_kernel, gpu_xpu_ep, gpu_xpu_cntr, gpu_addr, gpu_desc, gpu_buf);

/* 14. Cleanup */
fi_close(&ep->fid);
fi_close(&cq->fid);
fi_close(&cntr->fid);
fi_close(&mr->fid);
fi_close(&av->fid);
fi_close(&xpu_ctx->fid);
fi_close(&domain->fid);
```

# SEE ALSO

[`fi_getinfo`(3)](fi_getinfo.3.html),
[`fi_endpoint`(3)](fi_endpoint.3.html),
[`fi_cq`(3)](fi_cq.3.html),
[`fi_cntr`(3)](fi_cntr.3.html),
[`fi_mr`(3)](fi_mr.3.html),
[`fi_av`(3)](fi_av.3.html),
[`fi_set_ops`(3)](fi_set_ops.3.html)
