/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef FI_XPU_DEVICE_EFA_H
#define FI_XPU_DEVICE_EFA_H

/*
 * EFA provider - XPU device-side implementation.
 *
 * Inlined device functions for EFA hardware, implementing the OFI XPU device
 * API surface. Consumers include fi_xpu_device.h, which pulls in this header
 * and dispatches on prov_id.
 *
 * The host-side export calls fill in the endpoint, completion queue and
 * counter handles below and copy them to device memory; the device functions
 * read them back. The struct definitions exist so the compiler can inline the
 * functions. Consumers MUST NOT access struct fields directly - treat all
 * handles as opaque and only call fi_xpu_* dispatch functions.
 *
 * The peer and MR descriptor are what an XPU kernel builds EFA work requests
 * from. The host obtains each with an fi_*() call and copies it to device
 * memory, and the kernel passes it to the device-side data transfer calls; it
 * is never resolved back to an AV entry or an MR on the device. Both are
 * values, not handles, and their sizes are what fi_xpu_ctx_query() reports as
 * av_addr_size and mr_desc_size.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_efa_errno.h>
#include <rdma/fi_efa_io_defs.h>

/*
 * This header is included by fi_xpu_device.h which defines FI_XPU_FUNC.
 * Provide fallback for standalone compilation.
 */
#ifndef FI_XPU_FUNC
#define FI_XPU_FUNC static inline
#endif


/*
 * Device intrinsics
 *
 * The queues are shared by every thread that posts to them, so the cursors
 * that order the posting are read and written with device-scope atomics, and
 * the descriptors and doorbell the NIC reads are published with system-scope
 * fences. EFA_XPU_LOAD is a volatile load rather than an atomic one: the
 * cursors are naturally aligned 32-bit words, so the load cannot tear, and
 * volatile is what keeps the spin loops below re-reading memory.
 *
 * The subgroup primitives name the lowest active lane as the leader instead of
 * lane 0, so a partial subgroup - the last one of a work group whose size is
 * not a whole number of subgroups - still has a leader that is in the group.
 */
#if defined(__CUDACC__)
  #define EFA_XPU_DEVICE_COMPILE 1
  #define EFA_XPU_FENCE_SYSTEM() __threadfence_system()
  #define EFA_XPU_FENCE_DEVICE() __threadfence()
  #define EFA_XPU_FENCE_BLOCK()  __threadfence_block()
  #define EFA_XPU_ATOMIC_CAS(ptr, exp, des) \
	atomicCAS((unsigned int *)(ptr), (exp), (des))
  #define EFA_XPU_ATOMIC_EXCH(ptr, val) \
	atomicExch((unsigned int *)(ptr), (val))
  #define EFA_XPU_ATOMIC_ADD(ptr, val) \
	atomicAdd((unsigned int *)(ptr), (val))
  #define EFA_XPU_ATOMIC_ADD64(ptr, val) \
	atomicAdd((unsigned long long *)(ptr), (unsigned long long)(val))
  #define EFA_XPU_ATOMIC_MIN(ptr, val) \
	atomicMin((unsigned int *)(ptr), (val))
  #define EFA_XPU_LOAD(ptr)	(*(volatile unsigned int *)(ptr))
  #define EFA_XPU_STORE(ptr, val)	(*(volatile unsigned int *)(ptr) = (val))
  #define EFA_XPU_LOAD64(ptr)	(*(volatile unsigned long long *)(ptr))
  #define EFA_XPU_THREAD_RANK() \
	(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
  #define EFA_XPU_SUBGROUP_MASK()		__activemask()
  #define EFA_XPU_SUBGROUP_LEADER_LANE(mask) \
	(__ffs((unsigned int)(mask)) - 1)
  #define EFA_XPU_SUBGROUP_LANE()		(EFA_XPU_THREAD_RANK() & \
						 (warpSize - 1))
  #define EFA_XPU_SUBGROUP_POPC(mask) \
	__popcll((unsigned long long)(mask))
  #define EFA_XPU_SUBGROUP_SYNC(mask)		__syncwarp((unsigned int)(mask))
  #define EFA_XPU_SUBGROUP_BCAST(mask, val, lane) \
	__shfl_sync((unsigned int)(mask), (val), (lane))
  #define EFA_XPU_SUBGROUP_BALLOT(mask, pred) \
	__ballot_sync((unsigned int)(mask), (pred))
  #define EFA_XPU_WORK_GROUP_SYNC()		__syncthreads()
  #define EFA_XPU_WORK_GROUP_SIZE() \
	(blockDim.x * blockDim.y * blockDim.z)
  #define EFA_XPU_SHARED			__shared__
#elif defined(__HIPCC__)
  /*
   * Both passes, as for CUDA: the functions below are __device__ in the host
   * pass too (see FI_XPU_FUNC), so they need the intrinsics declared there.
   * hipcc does not include the HIP runtime header on its own, as nvcc does
   * for CUDA, and it is what declares them.
   */
  #include <hip/hip_runtime.h>
  /*
   * A wavefront runs in lockstep, so a wave barrier only has to stop the
   * compiler from moving memory operations across it.
   */
  #define EFA_XPU_DEVICE_COMPILE 1
  #define EFA_XPU_FENCE_SYSTEM() __threadfence_system()
  #define EFA_XPU_FENCE_DEVICE() __threadfence()
  #define EFA_XPU_FENCE_BLOCK()  __threadfence_block()
  #define EFA_XPU_ATOMIC_CAS(ptr, exp, des) \
	atomicCAS((unsigned int *)(ptr), (exp), (des))
  #define EFA_XPU_ATOMIC_EXCH(ptr, val) \
	atomicExch((unsigned int *)(ptr), (val))
  #define EFA_XPU_ATOMIC_ADD(ptr, val) \
	atomicAdd((unsigned int *)(ptr), (val))
  #define EFA_XPU_ATOMIC_ADD64(ptr, val) \
	atomicAdd((unsigned long long *)(ptr), (unsigned long long)(val))
  #define EFA_XPU_ATOMIC_MIN(ptr, val) \
	atomicMin((unsigned int *)(ptr), (val))
  #define EFA_XPU_LOAD(ptr)	(*(volatile unsigned int *)(ptr))
  #define EFA_XPU_STORE(ptr, val)	(*(volatile unsigned int *)(ptr) = (val))
  #define EFA_XPU_LOAD64(ptr)	(*(volatile unsigned long long *)(ptr))
  #define EFA_XPU_THREAD_RANK() \
	(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
  /*
   * The active lanes, as __activemask() reports them on CUDA. HIP only
   * declares __activemask() from ROCm 7 on, or with
   * HIP_ENABLE_WARP_SYNC_BUILTINS, and defines it as this ballot anyway.
   */
  #define EFA_XPU_SUBGROUP_MASK()		__ballot(1)
  #define EFA_XPU_SUBGROUP_LEADER_LANE(mask)	(__ffsll((unsigned long long) \
							 (mask)) - 1)
  #define EFA_XPU_SUBGROUP_LANE()		__lane_id()
  #define EFA_XPU_SUBGROUP_POPC(mask) \
	__popcll((unsigned long long)(mask))
  #define EFA_XPU_SUBGROUP_SYNC(mask) \
	do { (void)(mask); __builtin_amdgcn_wave_barrier(); } while (0)
  #define EFA_XPU_SUBGROUP_BCAST(mask, val, lane) \
	((void)(mask), __shfl((val), (lane)))
  #define EFA_XPU_SUBGROUP_BALLOT(mask, pred) \
	((unsigned long long)(mask) & __ballot(pred))
  #define EFA_XPU_WORK_GROUP_SYNC()		__syncthreads()
  #define EFA_XPU_WORK_GROUP_SIZE() \
	(blockDim.x * blockDim.y * blockDim.z)
  #define EFA_XPU_SHARED			__shared__
#else
  /*
   * Host builds, where every group is the calling thread alone. No kernel
   * runs these, but the gtests drive the header through them.
   */
  static inline int efa_xpu_host_popc(unsigned long long mask)
  {
	int n = 0;

	for (; mask; mask &= mask - 1)
		n++;
	return n;
  }
  #define EFA_XPU_FENCE_SYSTEM()		((void) 0)
  #define EFA_XPU_FENCE_DEVICE()		((void) 0)
  #define EFA_XPU_FENCE_BLOCK()		((void) 0)
  #define EFA_XPU_ATOMIC_CAS(ptr, exp, des) \
	(*(unsigned int *)(ptr) == (exp) ? \
		(*(unsigned int *)(ptr) = (des), (exp)) : \
		*(unsigned int *)(ptr))
  #define EFA_XPU_ATOMIC_EXCH(ptr, val) \
	({ unsigned int _o = *(unsigned int *)(ptr); \
	   *(unsigned int *)(ptr) = (val); _o; })
  #define EFA_XPU_ATOMIC_ADD(ptr, val) \
	({ unsigned int _o = *(unsigned int *)(ptr); \
	   *(unsigned int *)(ptr) = _o + (val); _o; })
  #define EFA_XPU_ATOMIC_ADD64(ptr, val) \
	({ unsigned long long _o = *(unsigned long long *)(ptr); \
	   *(unsigned long long *)(ptr) = _o + (val); _o; })
  #define EFA_XPU_ATOMIC_MIN(ptr, val) \
	({ unsigned int _o = *(unsigned int *)(ptr); \
	   if ((val) < _o) *(unsigned int *)(ptr) = (val); _o; })
  #define EFA_XPU_LOAD(ptr)	(*(volatile unsigned int *)(ptr))
  #define EFA_XPU_STORE(ptr, val)	(*(volatile unsigned int *)(ptr) = (val))
  #define EFA_XPU_LOAD64(ptr)	(*(volatile unsigned long long *)(ptr))
  #define EFA_XPU_THREAD_RANK()			0
  #define EFA_XPU_SUBGROUP_MASK()		1u
  #define EFA_XPU_SUBGROUP_LEADER_LANE(mask)	((void)(mask), 0)
  #define EFA_XPU_SUBGROUP_LANE()		0
  #define EFA_XPU_SUBGROUP_POPC(mask) \
	efa_xpu_host_popc((unsigned long long)(mask))
  #define EFA_XPU_SUBGROUP_SYNC(mask)		((void)(mask))
  #define EFA_XPU_SUBGROUP_BCAST(mask, val, lane) \
	((void)(mask), (void)(lane), (val))
  #define EFA_XPU_SUBGROUP_BALLOT(mask, pred)	((void)(mask), (pred) ? 1u : 0u)
  #define EFA_XPU_WORK_GROUP_SYNC()		((void) 0)
  #define EFA_XPU_WORK_GROUP_SIZE()		1
  #define EFA_XPU_SHARED
#endif

/*
 * Opaque device handle layouts — exported by the host-side functions
 */
/*
 * A work queue is produced by every thread that posts to the endpoint, so the
 * three cursors below are the whole of its concurrency control. They only ever
 * move forwards, and each is a 32-bit slot number that wraps with the queue:
 *
 *   pc       slots handed out. A poster claims one slot with an atomic add,
 *            which is what makes the claim unique without a lock.
 *   released slots handed off to the next poster, in slot order. A poster may
 *            ring the doorbell only while released names its own slot, so the
 *            doorbell never runs ahead of a descriptor that is still being
 *            written.
 *   db_rung  the value last written to the doorbell, which bounds how many
 *            written-but-unrung descriptors may accumulate.
 *
 * A fourth, done, is written by whoever reads the completion queue the work
 * queue reports to: the count of its descriptors whose completions have been
 * consumed, so the slots behind it are free again. The provider's host CQ
 * read publishes it into device memory, and a device-side CQ read advances it
 * in place; it is NULL only for a queue with no CQ to report to.
 * done_seen caches the last value read, so a poster only goes back to done
 * when its cached view says the queue is full.
 *
 * ctx is the queue's context table, one entry per slot, when the descriptor
 * has no room for the application's context: always on the receive queue,
 * whose descriptors have a 16-bit request ID, and on a send queue without the
 * 64-bit request ID. A post stores its context at ctx[slot & queue_mask] and
 * puts the slot in the request ID; a completion, read on the host or the
 * device, takes the context back out. It is NULL on a send queue whose
 * descriptors carry the context in their 64-bit request ID.
 *
 * slot_done goes with ctx: for each slot, the count of completions consumed
 * there, written by the same CQ reader as done. Completions can come back out
 * of order, so done says how many slots are free but not which, and a slot
 * whose completion is still outstanding could otherwise be handed out again
 * and have its context overwritten before the reader takes it out. A poster
 * waits on its own slot's count before storing a context (see
 * efa_xpu_wq_wait_slot()).
 *
 * init_phase is the phase bit the queue starts on; the bit for a given slot
 * follows from the slot number, so nothing has to track it.
 */
struct efa_xpu_wq {
	uint32_t  pc;
	uint32_t  released;
	uint32_t  db_rung;
	int32_t   init_phase;
	uint32_t  queue_mask;
	uint32_t  queue_size_shift;
	uint32_t  max_batch;
	uint32_t  entry_size;
	uint8_t  *buf;
	uint32_t *db;
	const uint32_t *done;	/* descriptors the CQ reader has consumed */
	uint32_t  depth;	/* descriptors the queue may hold at once */
	uint32_t  done_seen;	/* last value read through done */
	uint64_t *ctx;		/* context per slot, or NULL */
	const uint32_t *slot_done; /* completions consumed per slot, with ctx */
};

struct efa_xpu_ep {
	uint32_t		version;	/* FI_VERSION() of the exporter */
	uint32_t		pad0;
	struct efa_xpu_wq	sq;
	struct efa_xpu_wq	rq;
};

/*
 * The completion queue is consumed by every thread that reads it, and cc, the
 * consumer cursor, is the whole of its concurrency control: a read claims the
 * run of completions it takes by moving cc past them with a compare and swap,
 * and only the thread that moved it reports them. Reads that overlap, from
 * other threads, groups, blocks or kernels, take disjoint runs.
 */
/*
 * What a device-side CQ read needs to know about one QP reporting to the CQ: the
 * done cursors it advances for each queue (see struct efa_xpu_wq), and the
 * context tables where a queue still uses them. A completion names its QP by
 * number only, so the CQ carries a map of EFA_XPU_CQ_QP_MAP_SIZE rows indexed
 * by the QP number's low bits, with the row's qpn checked. fi_cq_export_xpu()
 * fills it from the endpoints bound to the CQ and exported by then, and refuses
 * a set of QPs two of which would share a row; a later binding needs the CQ
 * exported again.
 */
enum {
	EFA_XPU_CQ_QP_EMPTY,
	EFA_XPU_CQ_QP_LIVE,
};

#define EFA_XPU_CQ_QP_MAP_SIZE	256	/* rows; a power of two */

struct efa_xpu_cq_qp {
	uint32_t		qpn;
	uint32_t		state;		/* EFA_XPU_CQ_QP_* */
	uint32_t		*sq_done;	/* SQ's done cursor, or NULL */
	uint32_t		*rq_done;	/* RQ's done cursor, or NULL */
	const uint64_t		*sq_ctx;	/* or NULL: 64-bit request IDs */
	const uint64_t		*rq_ctx;
	uint32_t		*sq_slot_done;	/* slot_done, with sq_ctx */
	uint32_t		*rq_slot_done;	/* slot_done, with rq_ctx */
	uint32_t		sq_mask;
	uint32_t		rq_mask;
};

struct efa_xpu_cq {
	uint32_t		version;	/* FI_VERSION() of the exporter */
	uint32_t		pad0;
	uint32_t		cc;
	int32_t			init_phase;
	uint32_t		queue_mask;
	uint32_t		queue_size_shift;
	uint32_t		entry_size;	/* hardware completion size */
	uint32_t		format;		/* enum fi_cq_format */
	uint8_t			*buf;
	uint32_t		user_entry_size;/* one entry in that format */
	struct efa_xpu_cq_qp	*qp_map;	/* EFA_XPU_CQ_QP_MAP_SIZE rows */
};

struct efa_xpu_cntr {
	uint32_t		version;	/* FI_VERSION() of the exporter */
	uint32_t		pad0;
	volatile uint64_t	*value;
	volatile uint64_t	*err_value;
};

/*
 * A peer, as fi_av_lookup2() with FI_XPU returns it: the address handle
 * number and the remote QP number and QKEY an EFA send queue descriptor
 * addresses a peer with.
 */
struct efa_xpu_peer {
	uint16_t ahn;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
};

/*
 * A memory registration, as fi_mr_get_xpu_desc() with FI_XPU returns it: the
 * local key an EFA descriptor names the registered buffer with. The reserved
 * word is always zero; it keeps the descriptor 8 bytes, so it can carry more
 * without mr_desc_size changing.
 */
struct efa_xpu_desc {
	uint32_t lkey;
	uint32_t reserved;
};

/*
 * Handle resolution helpers.
 *
 * The pointer passed into each efa_xpu_*() function is the
 * device-resident copy of the small, fixed-size struct fid_xpu_ep /
 * fid_xpu_cq / fid_xpu_cntr that the host filled in via
 * fi_ep_export_xpu / fi_cq_export_xpu / fi_cntr_export_xpu and copied
 * to device memory. Its embedded struct fid_xpu.prov_ctx holds the
 * device-accessible address of the full EFA-specific state (WQ/RQ,
 * CQ ring, counter pointers, etc). Resolve it before touching any
 * EFA-specific fields.
 */
FI_XPU_FUNC struct efa_xpu_ep *
efa_xpu_ep_resolve(void *ep)
{
	struct fid_xpu *fid = (struct fid_xpu *)ep;

	return (struct efa_xpu_ep *)(uintptr_t)fid->prov_ctx;
}

FI_XPU_FUNC struct efa_xpu_cq *
efa_xpu_cq_resolve(void *cq)
{
	struct fid_xpu *fid = (struct fid_xpu *)cq;

	return (struct efa_xpu_cq *)(uintptr_t)fid->prov_ctx;
}

FI_XPU_FUNC struct efa_xpu_cntr *
efa_xpu_cntr_resolve(void *cntr)
{
	struct fid_xpu *fid = (struct fid_xpu *)cntr;

	return (struct efa_xpu_cntr *)(uintptr_t)fid->prov_ctx;
}

/*
 * Host/device version compatibility.
 *
 * A kernel is compiled against this header but runs against handles exported
 * by whatever libfabric the host process loaded, which may be older than the
 * header. Every exported handle carries the exporting library's version, so
 * the kernel can refuse a handle it cannot interpret instead of misreading a
 * struct whose layout it does not know. EFA_XPU_VERSION_MIN is the first
 * libfabric release that exported these layouts, so it must not be newer than
 * the FI_VERSION() of the library carrying this header; bump it only for an
 * incompatible change. A field added in a later release is read only when the
 * exporter is at least that new:
 *
 *   if (FI_VERSION_GE(e->version, FI_VERSION(2, 8)))
 *           use(e->new_field);
 *
 * A field that is optional *within* one release cannot be handled this way -
 * it needs its own validity flag, because the version alone does not say
 * whether the exporter populated it.
 */
#define EFA_XPU_VERSION_MIN	FI_VERSION(2, 7)

FI_XPU_FUNC int
efa_xpu_ep_compat(struct efa_xpu_ep *ep)
{
	return FI_VERSION_GE(ep->version, EFA_XPU_VERSION_MIN);
}

FI_XPU_FUNC int
efa_xpu_cq_compat(struct efa_xpu_cq *cq)
{
	return FI_VERSION_GE(cq->version, EFA_XPU_VERSION_MIN);
}

FI_XPU_FUNC int
efa_xpu_cntr_compat(struct efa_xpu_cntr *cntr)
{
	return FI_VERSION_GE(cntr->version, EFA_XPU_VERSION_MIN);
}

/*
 * Cooperative scopes
 *
 * A scope names the set of threads that make a call together, and a call at a
 * group scope is one operation for the group (see fi_xpu(3)): every thread
 * passes the same arguments, the group meets at a barrier, the leader performs
 * the operation, and the group leaves with the leader's result. A subgroup of
 * sixteen threads calling efa_xpu_send() posts one work request, not sixteen.
 * The barrier on the way in is what lets the operation see what the group
 * wrote before the call: the leader observes the other threads' writes through
 * it, and the fence it publishes its descriptor with orders them too, so a
 * group can fill a buffer together and send it with one call.
 *
 * Threads that each have an operation of their own call at FI_XPU_WORK_ITEM.
 * Those calls are separate operations, but the converged lanes of a subgroup
 * that make one together name the same endpoint, and they are coalesced into a
 * batch that claims its slots with one atomic add and rings one doorbell, with
 * every thread writing the descriptor it built from its own arguments. The
 * batch is whoever happens to be there, and a thread that arrives alone posts
 * alone.
 *
 * A group-scope call is a collective, so a thread of the group that does not
 * reach it leaves the ones that do waiting at a barrier. At FI_XPU_WORK_GROUP
 * that is every thread of the block, and rank 0 in particular, since it is the
 * leader. At FI_XPU_SUBGROUP it is every lane of the warp the block has,
 * whether or not they are converged when they reach the call: a warp that
 * arrives split meets at the barrier and makes one operation, and a warp some
 * of whose lanes never reach the call leaves the rest waiting.
 */
/*
 * How the threads of a group meet and pass values: not the scope a call was
 * made at, but what that scope, or a batch of work item posts, comes down to.
 */
enum efa_xpu_group_kind {
	EFA_XPU_GROUP_SELF,	/* one thread: meeting and passing are no-ops */
	EFA_XPU_GROUP_LANES,	/* the lanes of one warp in mask */
	EFA_XPU_GROUP_BLOCK,	/* every thread of the work group */
};

struct efa_xpu_group {
	enum efa_xpu_group_kind kind;
	int		   rank;	/* this thread's index in the group */
	int		   size;	/* threads taking part */
	int		   leader;	/* rank 0: does the group's bookkeeping */
	unsigned long long mask;	/* subgroup lanes taking part */
	int		   leader_lane;	/* the leader's lane in the subgroup */
	uint32_t	   bcast;	/* last value broadcast from the leader */
};

FI_XPU_FUNC void
efa_xpu_group_sync(struct efa_xpu_group *g)
{
	if (g->kind == EFA_XPU_GROUP_LANES)
		EFA_XPU_SUBGROUP_SYNC(g->mask);
	else if (g->kind == EFA_XPU_GROUP_BLOCK)
		EFA_XPU_WORK_GROUP_SYNC();
}

/*
 * Make the lanes in @mask, all of one warp, the group: the whole warp of a
 * subgroup call, or the lanes of a batch of work item posts.
 */
FI_XPU_FUNC void
efa_xpu_group_lanes(struct efa_xpu_group *g, unsigned long long mask)
{
	unsigned int lane = EFA_XPU_SUBGROUP_LANE();

	g->kind = EFA_XPU_GROUP_LANES;
	g->mask = mask;
	g->leader_lane = EFA_XPU_SUBGROUP_LEADER_LANE(mask);
	g->leader = (lane == (unsigned int) g->leader_lane);
	/*
	 * Rank within the group, not the lane number: the group is only the
	 * lanes in the mask, so a thread's place in it is how many of them lie
	 * below its own lane. That keeps the ranks dense, which is what makes
	 * the reserved slot range contiguous.
	 */
	g->size = EFA_XPU_SUBGROUP_POPC(mask);
	g->rank = EFA_XPU_SUBGROUP_POPC(mask & ((1ULL << lane) - 1ULL));
	g->bcast = 0;
}

/**
 * Every lane of this thread's warp, as a mask: the whole warp, or the lanes a
 * work group whose size is not a whole number of warps has left in its last
 * one. That is the subgroup a call at FI_XPU_SUBGROUP is made by, whether or
 * not the lanes are converged when they reach it (fi_xpu(3)), so a warp that
 * arrives split meets at the group's barrier and makes one operation, as a
 * work group does.
 */
FI_XPU_FUNC unsigned long long
efa_xpu_subgroup_warp_mask(void)
{
#ifdef EFA_XPU_DEVICE_COMPILE
	unsigned int width = (unsigned int) warpSize;
	unsigned int first = (unsigned int) EFA_XPU_THREAD_RANK() & ~(width - 1u);
	unsigned int lanes = (unsigned int) EFA_XPU_WORK_GROUP_SIZE() - first;

	if (lanes > width)
		lanes = width;
	return lanes >= 64u ? ~0ULL : (1ULL << lanes) - 1ULL;
#else
	return 1ULL;
#endif
}

FI_XPU_FUNC int
efa_xpu_group_enter(struct efa_xpu_group *g, int scope)
{
	g->kind = EFA_XPU_GROUP_SELF;
	g->mask = 0;
	g->leader_lane = 0;
	g->rank = 0;
	g->size = 1;
	g->bcast = 0;

	switch (scope) {
	case FI_XPU_WORK_ITEM:
		g->leader = 1;
		break;
	case FI_XPU_SUBGROUP:
		efa_xpu_group_lanes(g, efa_xpu_subgroup_warp_mask());
		break;
	case FI_XPU_WORK_GROUP:
		g->kind = EFA_XPU_GROUP_BLOCK;
		g->rank = (int) EFA_XPU_THREAD_RANK();
		g->size = (int) EFA_XPU_WORK_GROUP_SIZE();
		g->leader = (g->rank == 0);
		break;
	default:
		/* A scope this build does not know. */
		g->leader = 0;
		g->size = 0;
		return -FI_EOPNOTSUPP;
	}

	/*
	 * Hold the group here so the operation sees everything its members
	 * prepared for it, the way the barrier at the end of a group-scope
	 * call makes the result visible to all of them.
	 */
	efa_xpu_group_sync(g);
	return 0;
}

/**
 * Whether @pred holds in every thread of the group.
 *
 * A batch of work item posts carries one flags word per thread, so a
 * decision the batch takes once - leaving the doorbell to a later post - is
 * taken only if every thread asked for it.
 */
FI_XPU_FUNC int
efa_xpu_group_all(struct efa_xpu_group *g, int pred)
{
	if (g->kind != EFA_XPU_GROUP_LANES)
		return pred;

	return (EFA_XPU_SUBGROUP_BALLOT(g->mask, pred) & g->mask) == g->mask;
}

/**
 * Broadcast the leader's value to the whole group.
 *
 * Used for the base of the group's reserved slot range: only the leader claims
 * it, and every thread needs it to work out its own slot. The result lands in
 * the group, which is what a group of one - the work item scope, and a host
 * build where there is no group machinery at all - already has.
 */
FI_XPU_FUNC uint32_t
efa_xpu_group_bcast_u32(struct efa_xpu_group *g, uint32_t val)
{
	if (g->leader)
		g->bcast = val;

	if (g->kind == EFA_XPU_GROUP_LANES) {
		g->bcast = (uint32_t) EFA_XPU_SUBGROUP_BCAST(g->mask, val,
							     g->leader_lane);
	}
#ifdef EFA_XPU_DEVICE_COMPILE
	else if (g->kind == EFA_XPU_GROUP_BLOCK) {
		EFA_XPU_SHARED uint32_t group_val;

		if (g->leader)
			group_val = val;
		EFA_XPU_WORK_GROUP_SYNC();
		g->bcast = group_val;
		/* Hold the group until every thread has read it, so the next
		 * call through here can reuse the same storage. */
		EFA_XPU_WORK_GROUP_SYNC();
	}
#endif
	return g->bcast;
}

/**
 * The lowest rank in the group at which @pred holds, or the group's size if
 * it holds nowhere. Every thread of the group gets the same answer.
 *
 * Used by a group's completion queue read, where every thread looks at one
 * entry ahead of the consumer cursor and the read stops at the first entry
 * that is not a successful completion.
 */
FI_XPU_FUNC uint32_t
efa_xpu_group_first(struct efa_xpu_group *g, int pred)
{
	if (g->kind == EFA_XPU_GROUP_LANES) {
		unsigned long long hits = (unsigned long long)
			EFA_XPU_SUBGROUP_BALLOT(g->mask, pred) & g->mask;
		int lane;

		if (!hits)
			return (uint32_t) g->size;
		lane = EFA_XPU_SUBGROUP_LEADER_LANE(hits);
		return (uint32_t) EFA_XPU_SUBGROUP_POPC(g->mask &
							((1ULL << lane) - 1ULL));
	}
#ifdef EFA_XPU_DEVICE_COMPILE
	if (g->kind == EFA_XPU_GROUP_BLOCK) {
		EFA_XPU_SHARED uint32_t group_first;
		uint32_t first;

		if (g->leader)
			group_first = (uint32_t) g->size;
		EFA_XPU_WORK_GROUP_SYNC();
		if (pred)
			EFA_XPU_ATOMIC_MIN(&group_first, (uint32_t) g->rank);
		EFA_XPU_WORK_GROUP_SYNC();
		first = group_first;
		/* Hold the group until every thread has read it, so the next
		 * call through here can reuse the same storage. */
		EFA_XPU_WORK_GROUP_SYNC();
		return first;
	}
#endif
	return pred ? 0 : (uint32_t) g->size;
}

/**
 * The lanes of @mask whose @val equals this lane's, as a mask.
 *
 * On CUDA this is __match_any_sync(), one instruction from sm_70 on. Elsewhere
 * the lanes peel off one value at a time: the lowest lane left broadcasts its
 * value, the lanes holding it ballot, and those lanes leave. Every lane of
 * @mask runs the same number of rounds, one per distinct value, so a subgroup
 * whose lanes all hold the same value pays for one round.
 */
FI_XPU_FUNC unsigned long long
efa_xpu_subgroup_match(unsigned long long mask, uint64_t val)
{
#if defined(__CUDACC__)
	return (unsigned long long) __match_any_sync((unsigned int) mask,
						     (unsigned long long) val);
#else
	unsigned long long left = mask, mine = 0;

	while (left) {
		int lead = EFA_XPU_SUBGROUP_LEADER_LANE(left);
		uint32_t lo = (uint32_t) EFA_XPU_SUBGROUP_BCAST(left, (uint32_t) val,
								lead);
		uint32_t hi = (uint32_t) EFA_XPU_SUBGROUP_BCAST(left,
					(uint32_t) (val >> 32), lead);
		uint64_t v = ((uint64_t) hi << 32) | lo;
		unsigned long long same = (unsigned long long)
			EFA_XPU_SUBGROUP_BALLOT(left, val == v) & left;

		if (val == v)
			mine = same;
		left &= ~same;
	}
	return mine;
#endif
}

/**
 * Gather the threads posting at this moment into a batch for @ep.
 *
 * Called by every thread that is about to post one descriptor of its own. The
 * batch is the converged lanes of the subgroup that post to the same endpoint:
 * lanes that name different endpoints - a lane per QP, as a proxy warp that
 * drives one channel per lane does - are partitioned by endpoint, and each
 * part is a batch of its own, with its own leader, slot claim and doorbell on
 * its own queue. Lanes that all name one endpoint, the common case, are one
 * batch. A batch is never larger than a subgroup, and no thread waits for
 * another to arrive: one that is alone at the call is a batch of one.
 *
 * Returns @ep, which every lane of the batch names.
 */
FI_XPU_FUNC struct efa_xpu_ep *
efa_xpu_group_coalesce(struct efa_xpu_group *g, struct efa_xpu_ep *ep)
{
	unsigned long long mask = (unsigned long long) EFA_XPU_SUBGROUP_MASK();

	efa_xpu_group_lanes(g, efa_xpu_subgroup_match(mask,
						      (uint64_t) (uintptr_t) ep));
	return ep;
}

/**
 * Leave a group-scope call with the leader's result.
 *
 * Passing the result through the group is what lets every thread return the
 * same thing: the leader performed the group's one operation, and this is how
 * the others learn whether it was accepted, or what a read found. The barrier
 * it ends on also holds the group until the operation has been issued.
 */
FI_XPU_FUNC int
efa_xpu_group_leave(struct efa_xpu_group *g, int ret)
{
	if (g->kind == EFA_XPU_GROUP_LANES)
		return EFA_XPU_SUBGROUP_BCAST(g->mask, ret, g->leader_lane);

#ifdef EFA_XPU_DEVICE_COMPILE
	if (g->kind == EFA_XPU_GROUP_BLOCK) {
		EFA_XPU_SHARED int group_ret;

		if (g->leader)
			group_ret = ret;
		EFA_XPU_WORK_GROUP_SYNC();
		ret = group_ret;
		/*
		 * Hold the group until every thread has read the result, so the
		 * next call through here can reuse the same storage.
		 */
		EFA_XPU_WORK_GROUP_SYNC();
	}
#endif
	return ret;
}

/*
 * Work queue production
 *
 * Posting is lock free: a poster claims a run of slots with an atomic add,
 * writes its descriptors there, and then rings the doorbell only while the
 * release cursor names the first of them. Claiming is what keeps two posters
 * out of one slot, and ringing in slot order is what keeps the doorbell from
 * ever naming a descriptor that is still being written.
 *
 * A run is one slot per descriptor, so a thread posting alone claims one and a
 * batch of coalesced posts claims as many as it has members.
 */
FI_XPU_FUNC int
efa_xpu_wq_slot_phase(struct efa_xpu_wq *wq, uint32_t slot)
{
	return (wq->init_phase ^ (int)(slot >> wq->queue_size_shift)) & 1;
}

/* The hardware stages at most max_batch descriptors between doorbells. */
FI_XPU_FUNC uint32_t
efa_xpu_wq_batch(struct efa_xpu_wq *wq)
{
	return wq->max_batch ? wq->max_batch : 1u;
}

/**
 * Ring the doorbell up to, but not including, slot @target.
 *
 * Only the thread holding the turn may call this. The descriptors in
 * [wq->db_rung, target) were written by whichever posters held the turn before
 * it, each of which published its own before handing over. @fenced says the
 * caller has issued a system fence since it observed the turn, which already
 * does the work of the first fence below.
 */
FI_XPU_FUNC void
efa_xpu_wq_ring(struct efa_xpu_wq *wq, uint32_t target, int fenced)
{
	/*
	 * The acquire half of the handoff. The descriptors about to be rung
	 * were written by other threads, so this fence is not what publishes
	 * them - it is what keeps the doorbell write below from being
	 * reordered ahead of the read that observed the handoff.
	 */
	if (!fenced)
		EFA_XPU_FENCE_SYSTEM();

	*wq->db = target;
	EFA_XPU_FENCE_SYSTEM();	/* order the doorbell MMIO write */

	EFA_XPU_STORE(&wq->db_rung, target);
}

/**
 * Wait until the descriptors before slot @next have left the queue, so the
 * slots up to @next are free.
 *
 * The wait is on whoever reads the completion queue - typically a host
 * thread in fi_cq_read() - so a poster must never be the only reader of the
 * completions it waits for. It cannot wait on a descriptor nobody will ring:
 * efa_xpu_wq_wait_room() has rung every slot below next - max_batch before it
 * gets here, and the export only publishes done for a queue deeper than that.
 */
FI_XPU_FUNC void
efa_xpu_wq_wait_done(struct efa_xpu_wq *wq, uint32_t next)
{
	uint32_t seen = EFA_XPU_LOAD(&wq->done_seen);

	if (next - seen <= wq->depth)
		return;

	do {
		seen = EFA_XPU_LOAD(wq->done);
	} while (next - seen > wq->depth);

	/*
	 * A hint for the next poster, so a racing store of an older value only
	 * costs it one more read of done.
	 */
	EFA_XPU_STORE(&wq->done_seen, seen);

	/* The slots are free only once the read above is ordered before the
	 * stores that fill them. */
	EFA_XPU_FENCE_DEVICE();
}

/**
 * Wait until the previous occupant of @slot has completed and its completion
 * has been consumed, so the slot's context table entry may be written again.
 *
 * The occupant of a slot on pass p of the ring is its pth, so the slot is free
 * once p completions have been consumed there. The count is compared modulo
 * the number of passes a 32-bit slot number can name, which is all the
 * precision pass p has. Like efa_xpu_wq_wait_done(), the wait is on whoever
 * reads the completions, and the occupant waited for is a full ring behind,
 * older than anything the staging limit lets go unrung.
 */
FI_XPU_FUNC void
efa_xpu_wq_wait_slot(const struct efa_xpu_wq *wq, uint32_t slot)
{
	const uint32_t *done = &wq->slot_done[slot & wq->queue_mask];
	uint32_t pass = slot >> wq->queue_size_shift;

	while ((uint32_t)((EFA_XPU_LOAD(done) - pass) << wq->queue_size_shift))
		;

	/* The entry is free only once the read is ordered before its store. */
	EFA_XPU_FENCE_DEVICE();
}

/**
 * Wait until the @count slots starting at @base may be written.
 *
 * Two bounds apply. The run of written but unrung descriptors stays within
 * the hardware's staging limit, and, where the CQ reader publishes done, the
 * run of descriptors not yet completed stays within the queue depth, so a
 * post waits for the queue rather than overrunning it. A queue without done
 * leaves the depth to the application: see fi_efa(7).
 */
FI_XPU_FUNC void
efa_xpu_wq_wait_room(struct efa_xpu_wq *wq, uint32_t base, uint32_t count)
{
	uint32_t max_batch = efa_xpu_wq_batch(wq);
	uint32_t next = base + count;

	/*
	 * Keep the run of written but unrung descriptors within the staging
	 * limit. Make room by ringing rather than by waiting to be rung: a
	 * poster that deferred its doorbell has already handed the turn on and
	 * will not come back to ring it. Only the turn holder may ring, and it
	 * rings up to its own first slot, which every deferred descriptor below
	 * it has already been written into.
	 */
	while ((next - EFA_XPU_LOAD(&wq->db_rung)) > max_batch) {
		if (EFA_XPU_LOAD(&wq->released) == base) {
			uint32_t db_rung = EFA_XPU_LOAD(&wq->db_rung);

			if (db_rung != base)
				efa_xpu_wq_ring(wq, base, 0);
		}
	}

	if (wq->done)
		efa_xpu_wq_wait_done(wq, next);
}

/**
 * Hand the @count slots starting at @base on to the next poster, ringing the
 * doorbell for them first unless @defer asks for the ring to be left to a later
 * post.
 *
 * The descriptors must already be published: they were written by the group's
 * members, and a fence orders only the writes of the thread that issues it, so
 * no thread can publish another's on its behalf. The exception is @solo, a
 * batch of one, whose only descriptor is the caller's own: it is published
 * here, after the turn has been taken, so that the one fence both publishes it
 * and orders the doorbell after the handoff it observed.
 */
FI_XPU_FUNC void
efa_xpu_wq_handoff(struct efa_xpu_wq *wq, uint32_t base, uint32_t count,
		   int defer, int solo)
{
	uint32_t next = base + count;
	uint32_t db_rung;

	/* Take the turn in slot order, which is what keeps the doorbell sane. */
	while (EFA_XPU_LOAD(&wq->released) != base)
		;

	if (solo)
		EFA_XPU_FENCE_SYSTEM();

	db_rung = EFA_XPU_LOAD(&wq->db_rung);

	/*
	 * Ring past this run rather than just past the deferred one: holding
	 * the turn means every descriptor below is written, so one doorbell
	 * drains the whole batch.
	 */
	if (!defer || (next - db_rung) >= efa_xpu_wq_batch(wq))
		efa_xpu_wq_ring(wq, next, solo);

	EFA_XPU_STORE(&wq->released, next);
}

/*
 * Batch posting
 *
 * A batch is the work item posts that reach a queue together (see
 * efa_xpu_group_coalesce()), each carrying a descriptor of its own: the leader
 * reserves the whole contiguous run with a single atomic add and broadcasts its
 * base, every thread writes its own descriptor into base plus its rank, and the
 * leader rings one doorbell for the batch.
 *
 * The run is walked in chunks of at most max_batch, because that is the number
 * of descriptors the hardware will stage between doorbells and a batch can be
 * larger than it. Each chunk is bounded by the leader, written by its members,
 * and handed on by the leader before the next one opens:
 *
 *	struct efa_xpu_post p;
 *
 *	efa_xpu_post_begin(&p, wq, &g, flags);
 *	while (efa_xpu_post_next(&p))
 *		if (p.active)
 *			<write this thread's descriptor into slot p.slot>
 *
 * efa_xpu_post_next() closes the chunk the previous iteration wrote before it
 * opens the next, which is why the loop body carries no barrier of its own.
 * Every thread of the batch must run the loop to completion, including the ones
 * with nothing to write in the current chunk, because closing a chunk is a
 * barrier across the batch.
 */
struct efa_xpu_post {
	struct efa_xpu_wq	*wq;
	struct efa_xpu_group	*g;
	uint32_t		base;		/* the batch's first slot */
	uint32_t		next_rank;	/* rank the next chunk starts at */
	uint32_t		chunk_base;	/* the open chunk's first slot */
	uint32_t		chunk_size;
	uint32_t		slot;		/* this thread's slot, if active */
	int			active;		/* this thread writes this chunk */
	int			open;		/* a chunk is open */
	int			defer;		/* FI_MORE: leave the doorbell */
};

FI_XPU_FUNC void
efa_xpu_post_begin(struct efa_xpu_post *p, struct efa_xpu_wq *wq,
		   struct efa_xpu_group *g, uint64_t flags)
{
	uint32_t base = 0;

	p->wq = wq;
	p->g = g;
	p->next_rank = 0;
	p->chunk_base = 0;
	p->chunk_size = 0;
	p->slot = 0;
	p->active = 0;
	p->open = 0;
	p->defer = efa_xpu_group_all(g, !!(flags & FI_MORE));

	/* One atomic add for the batch, which is what makes the run contiguous. */
	if (g->leader)
		base = EFA_XPU_ATOMIC_ADD(&wq->pc, (uint32_t)g->size);
	p->base = efa_xpu_group_bcast_u32(g, base);
}

FI_XPU_FUNC void
efa_xpu_post_open(struct efa_xpu_post *p)
{
	struct efa_xpu_group *g = p->g;
	uint32_t max_batch = efa_xpu_wq_batch(p->wq);
	uint32_t first = p->next_rank;
	uint32_t size = (uint32_t)g->size - first;

	if (size > max_batch)
		size = max_batch;

	p->chunk_base = p->base + first;
	p->chunk_size = size;
	p->next_rank = first + size;

	/*
	 * The leader makes room for the whole chunk before any of it is written,
	 * so the batch waits once here instead of once per thread.
	 */
	if (g->leader)
		efa_xpu_wq_wait_room(p->wq, p->chunk_base, size);
	efa_xpu_group_sync(g);

	p->active = ((uint32_t)g->rank >= first &&
		     (uint32_t)g->rank < p->next_rank);
	p->slot = p->active ? p->chunk_base + ((uint32_t)g->rank - first) : 0;
	p->open = 1;
}

FI_XPU_FUNC void
efa_xpu_post_close(struct efa_xpu_post *p)
{
	struct efa_xpu_group *g = p->g;

	/*
	 * A batch of one - every group-scope post, and every work item post
	 * that found no others - has nobody else's descriptor to wait for. Its
	 * fence is taken after the turn instead (see efa_xpu_wq_handoff()), so
	 * one fence both publishes the descriptor and orders the doorbell, and
	 * there is no group to hold.
	 */
	if (g->size == 1) {
		efa_xpu_wq_handoff(p->wq, p->chunk_base, p->chunk_size,
				   p->defer, 1);
		p->open = 0;
		return;
	}

	/*
	 * Each thread publishes its own descriptor, because a fence orders only
	 * the writes of the thread that issues it and the leader cannot publish
	 * the group's on its behalf.
	 */
	EFA_XPU_FENCE_SYSTEM();
	efa_xpu_group_sync(g);

	if (g->leader)
		efa_xpu_wq_handoff(p->wq, p->chunk_base, p->chunk_size,
				   p->defer, 0);

	/*
	 * Hold the group until the chunk has been handed on, so the next chunk
	 * sees a release cursor that already covers this one. After the last
	 * chunk nothing follows, and the group may go.
	 */
	if (p->next_rank < (uint32_t) g->size)
		efa_xpu_group_sync(g);
	p->open = 0;
}

FI_XPU_FUNC int
efa_xpu_post_next(struct efa_xpu_post *p)
{
	if (p->open)
		efa_xpu_post_close(p);

	if (p->next_rank >= (uint32_t)p->g->size)
		return 0;

	efa_xpu_post_open(p);
	return 1;
}

/*
 * Write @wr_id into the WQE as its request ID: the application's context where
 * the send queue has 64-bit request IDs, or the slot when the context goes in
 * the queue's context table instead (see efa_xpu_post_wqe_batch()).
 */
FI_XPU_FUNC void
efa_xpu_set_wrid(struct efa_io_tx_meta_desc *md, void *context)
{
	uint64_t wr_id = (uint64_t)(uintptr_t)context;

	md->req_id = (uint16_t)wr_id;
	md->req_id_ex.w[0] = (uint16_t)(wr_id >> 16);
	md->req_id_ex.w[1] = (uint16_t)(wr_id >> 32);
	md->req_id_ex.w[2] = (uint16_t)(wr_id >> 48);
}

/**
 * Post one send queue descriptor per thread of the batch @g.
 *
 * The descriptor arrives already built by the thread that owns it, so all this
 * adds is what depends on the slot that thread ends up with: the slot's phase
 * bit, and the request ID. That is the application's @context itself where the
 * send queue has 64-bit request IDs; without them it is the slot, and the
 * context goes in the queue's context table at that slot, stored before the
 * fence that publishes the descriptor and only once the slot's previous
 * occupant has been completed.
 */
FI_XPU_FUNC int
efa_xpu_post_wqe_batch(struct efa_xpu_ep *ep, struct efa_xpu_group *g,
		       struct efa_io_tx_wqe *wqe, void *context,
		       uint64_t flags)
{
	struct efa_xpu_post p;

	efa_xpu_post_begin(&p, &ep->sq, g, flags);

	while (efa_xpu_post_next(&p)) {
		uint32_t sq_offset;
		uint64_t *dst;
		size_t i;

		if (!p.active)
			continue;

		/* Phase bit */
		wqe->meta.ctrl2 &= (uint8_t)~EFA_IO_TX_META_DESC_PHASE_MASK;
		if (efa_xpu_wq_slot_phase(&ep->sq, p.slot))
			wqe->meta.ctrl2 |= EFA_IO_TX_META_DESC_PHASE_MASK;

		if (ep->sq.ctx) {
			efa_xpu_wq_wait_slot(&ep->sq, p.slot);
			ep->sq.ctx[p.slot & ep->sq.queue_mask] =
				(uint64_t)(uintptr_t)context;
			efa_xpu_set_wrid(&wqe->meta,
					 (void *)(uintptr_t)(uint16_t)p.slot);
		} else {
			efa_xpu_set_wrid(&wqe->meta, context);
		}

		/*
		 * 64-byte write to the claimed SQ ring slot (BAR MMIO), as eight
		 * 8-byte stores. The descriptor was filled in field by field, so
		 * its bytes are gathered with memcpy rather than by reading the
		 * struct through a uint64_t pointer: a compiler is entitled to
		 * assume the two types never name the same memory and to drop
		 * the field stores that such a read appears not to depend on.
		 */
		sq_offset = (p.slot & ep->sq.queue_mask) *
			    (uint32_t)sizeof(*wqe);
		dst = (uint64_t *)(ep->sq.buf + sq_offset);
		for (i = 0; i < sizeof(*wqe) / sizeof(*dst); i++) {
			uint64_t word;

			memcpy(&word, (const uint8_t *)wqe + i * sizeof(word),
			       sizeof(word));
			dst[i] = word;
		}
	}

	return 0;
}

/*
 * Post this thread's send queue descriptor, in a batch with the converged lanes
 * posting with it.
 */
FI_XPU_FUNC int
efa_xpu_post_wqe(struct efa_xpu_ep *ep, struct efa_io_tx_wqe *wqe,
		 void *context, uint64_t flags)
{
	struct efa_xpu_group batch;

	ep = efa_xpu_group_coalesce(&batch, ep);
	return efa_xpu_post_wqe_batch(ep, &batch, wqe, context, flags);
}

/**
 * Post one receive descriptor per thread of the batch @g.
 *
 * The receive queue has no counter the NIC writes, so there is nothing to learn
 * about how far it has consumed and the staging limit is the only bound on the
 * queue here. A receive descriptor has no phase bit and only 16 bits of request
 * ID, which carry the receive queue slot; the @context goes in the queue's
 * context table at that slot, once the slot's previous occupant has been
 * completed.
 */
FI_XPU_FUNC int
efa_xpu_post_rqe_batch(struct efa_xpu_ep *ep, struct efa_xpu_group *g,
		       void *buf, size_t len, uint32_t lkey, void *context,
		       uint64_t flags)
{
	struct efa_xpu_post p;

	efa_xpu_post_begin(&p, &ep->rq, g, flags);

	while (efa_xpu_post_next(&p)) {
		struct efa_io_rx_desc rqe;
		uint32_t rq_offset;
		uint32_t *dst;
		size_t i;

		if (!p.active)
			continue;

		if (ep->rq.ctx) {
			efa_xpu_wq_wait_slot(&ep->rq, p.slot);
			ep->rq.ctx[p.slot & ep->rq.queue_mask] =
				(uint64_t)(uintptr_t)context;
		}

		rqe.buf_addr_lo = (uint32_t)((uint64_t)(uintptr_t)buf &
					     0xFFFFFFFF);
		rqe.buf_addr_hi = (uint32_t)((uint64_t)(uintptr_t)buf >> 32);
		rqe.req_id = (uint16_t)p.slot;
		rqe.length = (uint16_t)len;
		rqe.lkey_ctrl = (lkey & EFA_IO_RX_DESC_LKEY_MASK) |
				EFA_IO_RX_DESC_FIRST_MASK |
				EFA_IO_RX_DESC_LAST_MASK;

		/* Gathered with memcpy for the reason given in post_wqe_batch(). */
		rq_offset = (p.slot & ep->rq.queue_mask) *
			    (uint32_t)sizeof(rqe);
		dst = (uint32_t *)(ep->rq.buf + rq_offset);
		for (i = 0; i < sizeof(rqe) / sizeof(*dst); i++) {
			uint32_t word;

			memcpy(&word, (const uint8_t *)&rqe + i * sizeof(word),
			       sizeof(word));
			dst[i] = word;
		}
	}

	return 0;
}

/* Post this thread's receive descriptor, batched as on the send queue. */
FI_XPU_FUNC int
efa_xpu_post_rqe(struct efa_xpu_ep *ep, void *buf, size_t len, uint32_t lkey,
		 void *context, uint64_t flags)
{
	struct efa_xpu_group batch;

	ep = efa_xpu_group_coalesce(&batch, ep);
	return efa_xpu_post_rqe_batch(ep, &batch, buf, len, lkey, context, flags);
}

FI_XPU_FUNC void
efa_xpu_init_wqe(struct efa_io_tx_wqe *wqe,
		    uint32_t op_type, struct efa_xpu_peer *peer)
{
	memset(wqe, 0, sizeof(*wqe));

	wqe->meta.ctrl1 = (uint8_t)(EFA_IO_TX_META_DESC_META_DESC_MASK |
				    (op_type &
				     EFA_IO_TX_META_DESC_OP_TYPE_MASK));
	wqe->meta.ctrl2 = (uint8_t)(EFA_IO_TX_META_DESC_FIRST_MASK |
				    EFA_IO_TX_META_DESC_LAST_MASK |
				    EFA_IO_TX_META_DESC_COMP_REQ_MASK);
	wqe->meta.ah = peer->ahn;
	wqe->meta.dest_qp_num = peer->remote_qpn;
	wqe->meta.qkey = peer->remote_qkey;
}


FI_XPU_FUNC void
efa_xpu_set_remote(struct efa_io_remote_mem_addr *d,
		      uint32_t key, uint64_t addr, uint32_t len)
{
	d->length = len;
	d->rkey = key;
	d->buf_addr_lo = (uint32_t)(addr & 0xFFFFFFFF);
	d->buf_addr_hi = (uint32_t)(addr >> 32);
}

FI_XPU_FUNC void
efa_xpu_set_buf(struct efa_io_tx_buf_desc *d,
		   uint32_t key, uint64_t addr, uint32_t len)
{
	d->length = len;
	d->lkey = key & EFA_IO_TX_BUF_DESC_LKEY_MASK;
	d->buf_addr_lo = (uint32_t)(addr & 0xFFFFFFFF);
	d->buf_addr_hi = (uint32_t)(addr >> 32);
}

/*
 * Device-side data transfer operations
 */
FI_XPU_FUNC int
efa_xpu_write(void *ep, const void *buf, size_t len, void *desc,
		 uint64_t data, void *dest_addr, uint64_t addr,
		 uint64_t key, void *context, uint64_t flags, int scope)
{
	struct efa_xpu_ep *e = efa_xpu_ep_resolve(ep);
	struct efa_xpu_desc *d = (struct efa_xpu_desc *)desc;
	struct efa_xpu_peer *p = (struct efa_xpu_peer *)dest_addr;
	struct efa_xpu_group g;
	struct efa_io_tx_wqe wqe;
	int ret;

	if (!efa_xpu_ep_compat(e))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/*
	 * A group scope call is the group's one operation, so only the leader
	 * builds and posts it. At the work item scope every thread is its own
	 * leader and posts its own.
	 */
	if (g.leader) {
		efa_xpu_init_wqe(&wqe, EFA_IO_RDMA_WRITE, p);
		if (flags & FI_REMOTE_CQ_DATA) {
			wqe.meta.ctrl1 |= EFA_IO_TX_META_DESC_HAS_IMM_MASK;
			wqe.meta.immediate_data = (uint32_t)data;
		}
		efa_xpu_set_remote(&wqe.data.rdma_req.remote_mem,
				   (uint32_t)key, addr, (uint32_t)len);
		efa_xpu_set_buf(&wqe.data.rdma_req.local_mem[0], d->lkey,
				(uint64_t)(uintptr_t)buf, (uint32_t)len);
		wqe.meta.length = 1;

		ret = efa_xpu_post_wqe(e, &wqe, context, flags);
	}

	return efa_xpu_group_leave(&g, ret);
}

FI_XPU_FUNC int
efa_xpu_read(void *ep, void *buf, size_t len, void *desc,
		void *src_addr, uint64_t addr, uint64_t key,
		void *context, uint64_t flags, int scope)
{
	struct efa_xpu_ep *e = efa_xpu_ep_resolve(ep);
	struct efa_xpu_desc *d = (struct efa_xpu_desc *)desc;
	struct efa_xpu_peer *p = (struct efa_xpu_peer *)src_addr;
	struct efa_xpu_group g;
	struct efa_io_tx_wqe wqe;
	int ret;

	if (!efa_xpu_ep_compat(e))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/* The group's one operation, as for efa_xpu_write(). */
	if (g.leader) {
		efa_xpu_init_wqe(&wqe, EFA_IO_RDMA_READ, p);
		efa_xpu_set_remote(&wqe.data.rdma_req.remote_mem,
				   (uint32_t)key, addr, (uint32_t)len);
		efa_xpu_set_buf(&wqe.data.rdma_req.local_mem[0], d->lkey,
				(uint64_t)(uintptr_t)buf, (uint32_t)len);
		wqe.meta.length = 1;

		ret = efa_xpu_post_wqe(e, &wqe, context, flags);
	}

	return efa_xpu_group_leave(&g, ret);
}

FI_XPU_FUNC int
efa_xpu_send(void *ep, const void *buf, size_t len, void *desc,
		uint64_t data, void *dest_addr, void *context,
		uint64_t flags, int scope)
{
	struct efa_xpu_ep *e = efa_xpu_ep_resolve(ep);
	struct efa_xpu_desc *d = (struct efa_xpu_desc *)desc;
	struct efa_xpu_peer *p = (struct efa_xpu_peer *)dest_addr;
	struct efa_xpu_group g;
	struct efa_io_tx_wqe wqe;
	int ret;

	if (!efa_xpu_ep_compat(e))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/* The group's one operation, as for efa_xpu_write(). */
	if (g.leader) {
		efa_xpu_init_wqe(&wqe, EFA_IO_SEND, p);
		if (flags & FI_REMOTE_CQ_DATA) {
			wqe.meta.ctrl1 |= EFA_IO_TX_META_DESC_HAS_IMM_MASK;
			wqe.meta.immediate_data = (uint32_t)data;
		}
		efa_xpu_set_buf(&wqe.data.sgl[0], d->lkey,
				(uint64_t)(uintptr_t)buf, (uint32_t)len);
		wqe.meta.length = 1;

		ret = efa_xpu_post_wqe(e, &wqe, context, flags);
	}

	return efa_xpu_group_leave(&g, ret);
}

FI_XPU_FUNC int
efa_xpu_recv(void *ep, void *buf, size_t len, void *desc,
		void *src_addr, void *context, uint64_t flags, int scope)
{
	struct efa_xpu_ep *e = efa_xpu_ep_resolve(ep);
	struct efa_xpu_desc *d = (struct efa_xpu_desc *)desc;
	struct efa_xpu_group g;
	int ret;

	(void)src_addr;

	if (!efa_xpu_ep_compat(e))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/* The group's one receive, as on the send queue. */
	if (g.leader)
		ret = efa_xpu_post_rqe(e, buf, len, d->lkey, context, flags);

	return efa_xpu_group_leave(&g, ret);
}

/*
 * Counter operations
 *
 * A counter is only read here, never advanced, and a read is a single load of a
 * word the NIC writes. Every thread that loads it sees the same value, so no
 * counter call is collective: a read, an error read and a wait need no leader
 * and no agreement between threads, and the scope is ignored. That is also
 * what lets a thread use a counter wherever it is, inside a loop its lanes
 * leave one at a time, which a call carrying a barrier could not survive. efa_xpu_cntr_read and
 * efa_xpu_cntr_readerr return the counter value itself, so they have no way to
 * report an incompatible handle, and a kernel reaches them only after the
 * endpoint or completion queue it drives has already been accepted by
 * efa_xpu_ep_compat / efa_xpu_cq_compat.
 */
FI_XPU_FUNC uint64_t
efa_xpu_cntr_read(void *cntr, int scope)
{
	struct efa_xpu_cntr *c = efa_xpu_cntr_resolve(cntr);

	(void)scope;
	return *c->value;
}

FI_XPU_FUNC uint64_t
efa_xpu_cntr_readerr(void *cntr, int scope)
{
	struct efa_xpu_cntr *c = efa_xpu_cntr_resolve(cntr);

	(void)scope;
	if (!c->err_value)
		return 0;
	return *c->err_value;
}

FI_XPU_FUNC void
efa_xpu_cntr_wait(void *cntr, uint64_t threshold,
		     int timeout, int scope)
{
	struct efa_xpu_cntr *c = efa_xpu_cntr_resolve(cntr);

	(void)timeout;
	(void)scope;

	/* Never spin on a handle whose layout this kernel cannot interpret. */
	if (!efa_xpu_cntr_compat(c))
		return;

	/*
	 * Like a read, the wait needs no group: each thread spins on the word
	 * the NIC writes until it has seen the threshold itself, and leaves on
	 * its own. A kernel that wants its group in step after the wait meets
	 * at its next group call, which begins with a barrier.
	 */
	while (*c->value < threshold)
		;
}

/*
 * CQ operations
 */

/*
 * A successful completion. The EFA status codes live in the provider's private
 * efa_errno.h, so the one value the device needs is named here; it is the base
 * of that enumeration and cannot move without breaking the hardware interface.
 */
#define EFA_XPU_COMP_STATUS_OK	0

/**
 * Read a field out of a completion's flags byte.
 *
 * The shift comes from the mask rather than being written out, so the field
 * offsets stay wherever efa_io_defs.h puts them. Every caller passes a constant
 * mask, which folds the loop away.
 */
FI_XPU_FUNC uint32_t
efa_xpu_field_get(uint32_t mask, uint32_t reg)
{
	uint32_t shift = 0;

	while (mask && !(mask & 1u)) {
		mask >>= 1;
		shift++;
	}

	return (reg >> shift) & mask;
}

/**
 * The libfabric completion flags a hardware completion stands for.
 *
 * A send queue completion describes the operation this side posted; a receive
 * queue completion describes what arrived, which for RDMA is the remote side's
 * write landing here.
 */
FI_XPU_FUNC uint64_t
efa_xpu_cqe_flags(uint8_t cqe_flags)
{
	uint32_t q_type = efa_xpu_field_get(EFA_IO_CDESC_COMMON_Q_TYPE_MASK,
					    cqe_flags);
	uint32_t op_type = efa_xpu_field_get(EFA_IO_CDESC_COMMON_OP_TYPE_MASK,
					     cqe_flags);
	uint64_t flags = 0;

	if (q_type == EFA_IO_SEND_QUEUE) {
		switch (op_type) {
		case EFA_IO_RDMA_WRITE:
			flags = FI_RMA | FI_WRITE;
			break;
		case EFA_IO_RDMA_READ:
			flags = FI_RMA | FI_READ;
			break;
		default:
			flags = FI_MSG | FI_SEND;
			break;
		}
	} else {
		switch (op_type) {
		case EFA_IO_RDMA_WRITE:
			flags = FI_RMA | FI_REMOTE_WRITE;
			break;
		default:
			flags = FI_MSG | FI_RECV;
			break;
		}
		if (cqe_flags & EFA_IO_CDESC_COMMON_HAS_IMM_MASK)
			flags |= FI_REMOTE_CQ_DATA;
	}

	return flags;
}

/**
 * The 64-bit request ID a send queue completion carries: the application's
 * context, where the send queue has 64-bit request IDs.
 */
FI_XPU_FUNC uint64_t
efa_xpu_cqe_req_id(const struct efa_io_tx_cdesc *tx)
{
	return (uint64_t) tx->common.req_id |
	       ((uint64_t) tx->req_id_ex.w[0] << 16) |
	       ((uint64_t) tx->req_id_ex.w[1] << 32) |
	       ((uint64_t) tx->req_id_ex.w[2] << 48);
}

/**
 * The row of the QP a completion names, or NULL when the CQ's map has none for
 * it - a QP not bound to the CQ, or bound after the CQ was exported. See
 * struct efa_xpu_cq_qp.
 */
FI_XPU_FUNC const struct efa_xpu_cq_qp *
efa_xpu_cq_find_qp(struct efa_xpu_cq *c, uint32_t qpn)
{
	const struct efa_xpu_cq_qp *row;

	if (!c->qp_map)
		return NULL;

	row = &c->qp_map[qpn & (EFA_XPU_CQ_QP_MAP_SIZE - 1)];
	if (row->state != EFA_XPU_CQ_QP_LIVE || row->qpn != qpn)
		return NULL;
	return row;
}

/** The number of bytes a receive queue completion transferred. */
FI_XPU_FUNC uint64_t
efa_xpu_cqe_len(struct efa_xpu_cq *c, const struct efa_io_rx_cdesc *rx)
{
	uint32_t op_type = efa_xpu_field_get(EFA_IO_CDESC_COMMON_OP_TYPE_MASK,
					     rx->common.flags);
	uint64_t len = rx->length;

	/*
	 * An RDMA write can land more than 64KiB, so its length has 16 more
	 * bits in the extended completion - which is only there when the ring
	 * was created with entries big enough to hold it.
	 */
	if (op_type == EFA_IO_RDMA_WRITE &&
	    c->entry_size >= sizeof(struct efa_io_rx_cdesc_ex)) {
		const struct efa_io_rx_cdesc_ex *ex =
			(const struct efa_io_rx_cdesc_ex *) rx;

		len |= (uint64_t) ex->u.rdma_write.length_hi << 16;
	}

	return len;
}

/**
 * Fill one entry of the caller's buffer from a hardware completion.
 *
 * The entry is written in the format the CQ was opened with, so a kernel reads
 * the same struct out of fi_xpu_cq_read() that a host thread reads out of
 * fi_cq_read() on the same CQ.
 */
FI_XPU_FUNC void
efa_xpu_cq_fill(struct efa_xpu_cq *c, const struct efa_io_cdesc_common *cqe,
		void *entry)
{
	struct fi_cq_data_entry e;
	uint32_t q_type = efa_xpu_field_get(EFA_IO_CDESC_COMMON_Q_TYPE_MASK,
					    cqe->flags);
	const struct efa_xpu_cq_qp *qp = efa_xpu_cq_find_qp(c, cqe->qp_num);
	size_t len;

	e.flags = efa_xpu_cqe_flags(cqe->flags);
	e.buf = NULL;
	e.data = 0;

	/*
	 * The context is the request ID itself where the descriptor had room
	 * for it, and otherwise the entry the poster stored in the queue's
	 * context table at the slot the request ID names.
	 */
	if (q_type == EFA_IO_SEND_QUEUE) {
		if (qp && qp->sq_ctx)
			e.op_context = (void *) (uintptr_t)
				qp->sq_ctx[cqe->req_id & qp->sq_mask];
		else
			e.op_context = (void *) (uintptr_t) efa_xpu_cqe_req_id(
				(const struct efa_io_tx_cdesc *) cqe);
		e.len = 0;
	} else {
		const struct efa_io_rx_cdesc *rx =
			(const struct efa_io_rx_cdesc *) cqe;

		/* An unsolicited completion has no descriptor behind it at all. */
		if (cqe->flags & EFA_IO_CDESC_COMMON_UNSOLICITED_MASK)
			e.op_context = NULL;
		else if (qp && qp->rq_ctx)
			e.op_context = (void *) (uintptr_t)
				qp->rq_ctx[cqe->req_id & qp->rq_mask];
		else
			e.op_context = (void *) (uintptr_t) cqe->req_id;
		e.len = efa_xpu_cqe_len(c, rx);
		if (e.flags & FI_REMOTE_CQ_DATA)
			e.data = rx->imm;
	}

	/*
	 * The three formats are prefixes of one another, so the entry is built
	 * once and as much of it as the format asks for is copied out.
	 */
	switch (c->format) {
	case FI_CQ_FORMAT_CONTEXT:
		len = sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		len = sizeof(struct fi_cq_msg_entry);
		break;
	default:
		len = sizeof(struct fi_cq_data_entry);
		break;
	}

	memcpy(entry, &e, len);
}

/**
 * Count a consumed completion against the queue it frees a slot in: its done
 * cursor, which bounds the queue's posts as a host CQ read's does (see
 * efa_xpu_wq_wait_done()), and, where the queue's contexts go through a
 * context table, the slot it frees (see efa_xpu_wq_wait_slot()).
 *
 * Called once the completion's context has been read out, by a read or by
 * fi_xpu_cq_readerr(), so a poster the counts let go cannot overwrite the entry
 * first. An unsolicited completion took no receive
 * and frees nothing.
 */
FI_XPU_FUNC void
efa_xpu_cq_release_targets(struct efa_xpu_cq *c,
			   const struct efa_io_cdesc_common *cqe,
			   uint32_t **cursor, uint32_t **slot)
{
	const struct efa_xpu_cq_qp *qp = efa_xpu_cq_find_qp(c, cqe->qp_num);
	uint32_t q_type = efa_xpu_field_get(EFA_IO_CDESC_COMMON_Q_TYPE_MASK,
					    cqe->flags);

	*cursor = NULL;
	*slot = NULL;
	if (!qp)
		return;

	if (q_type == EFA_IO_SEND_QUEUE) {
		*cursor = qp->sq_done;
		if (qp->sq_slot_done)
			*slot = &qp->sq_slot_done[cqe->req_id & qp->sq_mask];
	} else if (!(cqe->flags & EFA_IO_CDESC_COMMON_UNSOLICITED_MASK)) {
		*cursor = qp->rq_done;
		if (qp->rq_slot_done)
			*slot = &qp->rq_slot_done[cqe->req_id & qp->rq_mask];
	}
}

FI_XPU_FUNC void
efa_xpu_cq_release(struct efa_xpu_cq *c, const struct efa_io_cdesc_common *cqe)
{
	uint32_t *cursor, *slot;

	efa_xpu_cq_release_targets(c, cqe, &cursor, &slot);
	if (!cursor && !slot)
		return;

	EFA_XPU_FENCE_DEVICE();
	if (slot)
		EFA_XPU_ATOMIC_ADD(slot, 1u);
	if (cursor)
		EFA_XPU_ATOMIC_ADD(cursor, 1u);
}

/**
 * Release a completion that a group's read took, as efa_xpu_cq_release() does,
 * with the converged lanes that release slots of one queue adding to its done
 * cursor once between them rather than once each.
 *
 * Every converged lane calls this, so that the lanes can find each other. The
 * per-slot counts stay per lane: each names a slot of its own, and it is that
 * count that lets a poster overwrite the slot's context, so each lane adds it
 * after its own fence. The done cursor only bounds how far posts run ahead,
 * and a lane adding on behalf of others orders nothing they read.
 */
FI_XPU_FUNC void
efa_xpu_cq_release_combined(struct efa_xpu_cq *c,
			    const struct efa_io_cdesc_common *cqe)
{
	uint32_t *cursor, *slot;

	efa_xpu_cq_release_targets(c, cqe, &cursor, &slot);
	EFA_XPU_FENCE_DEVICE();
	if (slot)
		EFA_XPU_ATOMIC_ADD(slot, 1u);
#ifdef EFA_XPU_DEVICE_COMPILE
	{
		unsigned long long same = efa_xpu_subgroup_match(
			(unsigned long long) EFA_XPU_SUBGROUP_MASK(),
			(uint64_t) (uintptr_t) cursor);

		if (cursor && EFA_XPU_SUBGROUP_LANE() ==
			      (unsigned int) EFA_XPU_SUBGROUP_LEADER_LANE(same))
			EFA_XPU_ATOMIC_ADD(cursor,
					   (uint32_t) EFA_XPU_SUBGROUP_POPC(same));
	}
#else
	if (cursor)
		EFA_XPU_ATOMIC_ADD(cursor, 1u);
#endif
}

/** The completion at position @pos of the ring. */
FI_XPU_FUNC const struct efa_io_cdesc_common *
efa_xpu_cq_at(const struct efa_xpu_cq *c, uint32_t pos)
{
	return (const struct efa_io_cdesc_common *)
		(c->buf + (pos & c->queue_mask) * c->entry_size);
}

/**
 * Whether the NIC has written the completion at position @pos on the pass of
 * the ring that @pos names. The expected phase bit follows from the position,
 * so nothing but the consumer cursor has to be kept.
 *
 * Nothing else in the completion may be read before its phase bit has been
 * seen, or a wrapped ring hands back the previous pass's contents, so a true
 * answer is followed by a fence. It is a system fence because what it orders
 * against is the NIC's DMA rather than another thread's writes.
 */
FI_XPU_FUNC int
efa_xpu_cq_ready(const struct efa_xpu_cq *c, uint32_t pos)
{
	const volatile uint8_t *flags = (const volatile uint8_t *)
		efa_xpu_cq_at(c, pos) +
		offsetof(struct efa_io_cdesc_common, flags);
	int expected = (c->init_phase ^ (int)(pos >> c->queue_size_shift)) & 1;
	int actual = *flags & EFA_IO_CDESC_COMMON_PHASE_MASK;

	if (actual != expected)
		return 0;

	EFA_XPU_FENCE_SYSTEM();
	return 1;
}

/**
 * Take a copy of the completion at @pos out of the ring.
 *
 * Every field is loaded with a volatile access, as the phase bit was: the ring
 * is memory the NIC writes by DMA, so an ordinary load may be served from a
 * cache line this thread read on an earlier pass of the ring and hand back the
 * completion that slot used to hold. Reading the copy also keeps the ring
 * itself read once, whatever a caller does with the fields afterwards.
 */
FI_XPU_FUNC void
efa_xpu_cq_fetch(const struct efa_xpu_cq *c, uint32_t pos,
		 struct efa_io_rx_cdesc_ex *out)
{
	const volatile uint32_t *src = (const volatile uint32_t *)
		efa_xpu_cq_at(c, pos);
	uint32_t *dst = (uint32_t *) out;
	uint32_t size = c->entry_size < (uint32_t) sizeof(*out) ?
			c->entry_size : (uint32_t) sizeof(*out);
	uint32_t words = size / (uint32_t) sizeof(uint32_t);

	for (uint32_t i = 0; i < words; i++)
		dst[i] = src[i];
}

/**
 * The status of the completion at position @pos, which the caller has seen
 * ready. A volatile load, like the phase bit's: the ring is memory the NIC
 * writes by DMA.
 */
FI_XPU_FUNC uint32_t
efa_xpu_cq_status(const struct efa_xpu_cq *c, uint32_t pos)
{
	const volatile uint8_t *st = (const volatile uint8_t *)
		efa_xpu_cq_at(c, pos) +
		offsetof(struct efa_io_cdesc_common, status);

	return *st;
}

/**
 * Claim the run of successful completions written from the consumer cursor
 * on, at most @count of them, as the batch read of efa-dp-direct does: look
 * ahead by position without consuming anything, then move the cursor past the
 * whole run with one compare and swap. The run stops before a failed
 * completion, which a read never takes: it stays at the head until
 * fi_xpu_cq_readerr() claims it, so every overlapping read sees it and none
 * can drop or skip it.
 *
 * Only the reader that moved the cursor reports the run; one that loses the
 * race looks again from where the cursor went. The claimed completions stay
 * in the ring - the NIC does not write a slot again until the ring has wrapped,
 * which a CQ sized for its queues never does - so they are read out after the
 * claim.
 *
 * Returns the length of the run and its first position in @base, or 0 with
 * @head_failed saying whether the read stopped at a failure. Both answers come
 * from the completions the look-ahead examined, never from a second look,
 * which the NIC could change under it.
 */
FI_XPU_FUNC uint32_t
efa_xpu_cq_claim(struct efa_xpu_cq *c, uint32_t count, uint32_t *base,
		 int *head_failed)
{
	for (;;) {
		uint32_t cc = EFA_XPU_LOAD(&c->cc), n = 0;
		int failed = 0;

		while (n < count && efa_xpu_cq_ready(c, cc + n)) {
			if (efa_xpu_cq_status(c, cc + n) !=
			    EFA_XPU_COMP_STATUS_OK) {
				failed = 1;
				break;
			}
			n++;
		}

		if (!n) {
			*head_failed = failed;
			return 0;
		}
		if (EFA_XPU_ATOMIC_CAS(&c->cc, cc, cc + n) == cc) {
			*base = cc;
			*head_failed = 0;
			return n;
		}
	}
}

/**
 * Read up to @count successful completions into @buf: claim a run, then read
 * each completion out of the ring, translate it into the CQ's format and
 * release its queue slot.
 *
 * Returns the number of completions read, -FI_EAVAIL if the completion at the
 * cursor failed, or -FI_EAGAIN if there is none.
 */
FI_XPU_FUNC int
efa_xpu_cq_poll(struct efa_xpu_cq *c, void *buf, uint32_t count)
{
	uint32_t base = 0, n, i;
	int head_failed = 0;

	n = efa_xpu_cq_claim(c, count, &base, &head_failed);
	for (i = 0; i < n; i++) {
		struct efa_io_rx_cdesc_ex copy;
		const struct efa_io_cdesc_common *cqe =
			(const struct efa_io_cdesc_common *) &copy;

		efa_xpu_cq_fetch(c, base + i, &copy);
		efa_xpu_cq_fill(c, cqe,
				(uint8_t *) buf + (size_t) i * c->user_entry_size);
		efa_xpu_cq_release(c, cqe);
	}

	if (n)
		return (int) n;
	return head_failed ? -FI_EAVAIL : -FI_EAGAIN;
}

/**
 * Read up to @count successful completions into @buf for the group @g, the
 * group dividing the work between its threads.
 *
 * The read claims what efa_xpu_cq_poll() would, in the same way: a run of
 * successful completions from the consumer cursor, moved past with one
 * compare and swap. The look-ahead goes a window of the group's size at a
 * time, each thread checking one entry, and stops at the first entry of a
 * window that is not a successful completion; whether the read stopped at a
 * failure is decided by that same look, never by a second one. Only the leader
 * moves the cursor, and once the run is claimed every thread reads out its
 * share of it - entries rank, rank + size, ... - into @buf, which the group
 * shares.
 *
 * Every decision the threads take is the group's, passed from the leader or
 * agreed by efa_xpu_group_first(), so the group stays together throughout.
 *
 * Returns what efa_xpu_cq_poll() does, the same in every thread of the group.
 */
FI_XPU_FUNC int
efa_xpu_cq_poll_group(struct efa_xpu_cq *c, struct efa_xpu_group *g,
		      void *buf, uint32_t count)
{
	uint32_t size = (uint32_t) g->size, rank = (uint32_t) g->rank;

	for (;;) {
		uint32_t cc = 0, n = 0, i;
		int failed = 0, won = 0;

		if (g->leader)
			cc = EFA_XPU_LOAD(&c->cc);
		cc = efa_xpu_group_bcast_u32(g, cc);

		while (n < count) {
			uint32_t want = count - n < size ? count - n : size;
			int bad = 0, bad_failed = 0;
			uint32_t first;

			if (rank < want) {
				uint32_t pos = cc + n + rank;

				if (!efa_xpu_cq_ready(c, pos))
					bad = 1;
				else if (efa_xpu_cq_status(c, pos) !=
					 EFA_XPU_COMP_STATUS_OK)
					bad = bad_failed = 1;
			}

			first = efa_xpu_group_first(g, bad);
			if (first < want) {
				failed = efa_xpu_group_first(g, bad_failed) ==
					 first;
				n += first;
				break;
			}
			n += want;
		}

		if (!n)
			return failed ? -FI_EAVAIL : -FI_EAGAIN;

		if (g->leader)
			won = EFA_XPU_ATOMIC_CAS(&c->cc, cc, cc + n) == cc;
		if (!efa_xpu_group_bcast_u32(g, (uint32_t) won))
			continue;	/* another reader moved the cursor */

		for (i = rank; i < n; i += size) {
			struct efa_io_rx_cdesc_ex copy;
			const struct efa_io_cdesc_common *cqe =
				(const struct efa_io_cdesc_common *) &copy;

			efa_xpu_cq_fetch(c, cc + i, &copy);
			efa_xpu_cq_fill(c, cqe, (uint8_t *) buf +
					(size_t) i * c->user_entry_size);
			efa_xpu_cq_release_combined(c, cqe);
		}

		/* The buffer is complete for every thread once the group has
		 * met: each thread wrote only its own entries. */
		efa_xpu_group_sync(g);
		return (int) n;
	}
}

/**
 * The message for an EFA completion status, as efa_strerror() gives it on the
 * host, so a kernel reading a failure gets the same text a host thread would.
 * The strings are constants the device can read, which is what lets
 * fi_xpu_cq_readerr() hand one out as provider-owned err_data.
 *
 * The strings come from EFA_IO_COMP_STATUSES in rdma/fi_efa_errno.h, the
 * same list efa_strerror() is built from.
 */
#define EFA_XPU_COMP_STATUS_CASE(code, suffix, ...) \
	case code: return #__VA_ARGS__;

FI_XPU_FUNC const char *
efa_xpu_comp_strerror(uint32_t status)
{
	switch (status) {
	EFA_IO_COMP_STATUSES(EFA_XPU_COMP_STATUS_CASE)
	default:
		return "Unknown error";
	}
}

#undef EFA_XPU_COMP_STATUS_CASE

/**
 * Return @msg as the error data of @e, following fi_cq_readerr(3): a caller
 * that passed a buffer (@err_data_size non-zero) gets up to that many bytes of
 * the message, NUL included where it fits, copied into it, and err_data_size
 * set to the bytes copied; otherwise err_data points at the provider's own
 * copy, which stays valid for as long as the kernel runs.
 */
FI_XPU_FUNC void
efa_xpu_cq_err_data(struct fi_cq_err_entry *e, void *err_data,
		    size_t err_data_size, const char *msg)
{
	size_t len = 0, i;
	uint32_t s;

	while (msg[len])
		len++;
	len++;	/* the NUL */

	/*
	 * An entry reused from an earlier read still holds the provider's
	 * string and its size, which would otherwise read as a buffer of the
	 * caller's; the strings are constants, so it is taken as asking for the
	 * provider's again, as a host provider's own buffer would be written.
	 */
	for (s = 0; err_data && s < EFA_IO_COMP_STATUS_MAX; s++) {
		if (err_data == (const void *) efa_xpu_comp_strerror(s)) {
			err_data_size = 0;
			break;
		}
	}
	if (err_data == (const void *) efa_xpu_comp_strerror(UINT32_MAX))
		err_data_size = 0;

	if (err_data_size && err_data) {
		if (len > err_data_size)
			len = err_data_size;
		for (i = 0; i < len; i++)
			((char *) err_data)[i] = msg[i];
		e->err_data = err_data;
		e->err_data_size = len;
	} else {
		e->err_data = (void *) msg;
		e->err_data_size = len;
	}
}

/**
 * Take the failed completion at the head of the queue.
 *
 * The failure is claimed with the same compare and swap a read uses, so of
 * readers racing for it exactly one takes it and the rest look again: none can
 * take it twice, and none can lose it.
 *
 * Returns 1 with @e filled, or -FI_EAGAIN if the completion at the cursor is
 * not a failure or there is none.
 */
FI_XPU_FUNC int
efa_xpu_cq_take_err(struct efa_xpu_cq *c, struct fi_cq_err_entry *e)
{
	/* What the caller passed in, before the entry is cleared. */
	void *err_data = e->err_data;
	size_t err_data_size = e->err_data_size;

	for (;;) {
		uint32_t cc = EFA_XPU_LOAD(&c->cc);
		struct efa_io_rx_cdesc_ex copy;
		const struct efa_io_cdesc_common *cqe =
			(const struct efa_io_cdesc_common *) &copy;
		struct fi_cq_data_entry d;

		if (!efa_xpu_cq_ready(c, cc) ||
		    efa_xpu_cq_status(c, cc) == EFA_XPU_COMP_STATUS_OK)
			return -FI_EAGAIN;
		if (EFA_XPU_ATOMIC_CAS(&c->cc, cc, cc + 1) != cc)
			continue;	/* another reader moved the cursor */

		efa_xpu_cq_fetch(c, cc, &copy);
		efa_xpu_cq_fill(c, cqe, &d);
		memset(e, 0, sizeof(*e));
		e->op_context = d.op_context;
		e->flags = d.flags;
		e->len = d.len;
		e->prov_errno = (int) cqe->status;
		e->err = FI_EIO;
		efa_xpu_cq_err_data(e, err_data, err_data_size,
				    efa_xpu_comp_strerror(cqe->status));
		efa_xpu_cq_release(c, cqe);
		return 1;
	}
}


FI_XPU_FUNC int64_t
efa_xpu_cq_read(void *cq, void *buf, size_t count, int scope)
{
	struct efa_xpu_group g;
	int ret;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/*
	 * A read is one poll for its group, with the CQ, buffer and count every
	 * thread of the group passed. A group of more than one thread divides
	 * the poll between its threads (efa_xpu_cq_poll_group()), and every
	 * thread of it returns the same result. At the work item scope the
	 * group is the calling thread: each thread's read is its own, with its
	 * own buffer and count, and reads that overlap take disjoint
	 * completions (efa_xpu_cq_claim()). The count read and every error code
	 * fit in an int, which is what the group can pass between its threads,
	 * so a count larger than that is read as the largest one that fits. A
	 * read with no buffer is refused rather than polled: a completion it
	 * consumed could not be reported.
	 */
	if (g.size > 1) {
		struct efa_xpu_cq *c = efa_xpu_cq_resolve(cq);

		/* Every thread passed the same arguments, so every thread
		 * reaches the same answer here without telling the others. */
		if (!efa_xpu_cq_compat(c))
			return -FI_EOPNOTSUPP;
		if (!buf)
			return -FI_EINVAL;
		return efa_xpu_cq_poll_group(c, &g, buf, count > INT32_MAX ?
					     (uint32_t) INT32_MAX :
					     (uint32_t) count);
	}

	if (g.leader) {
		struct efa_xpu_cq *c = efa_xpu_cq_resolve(cq);

		if (!efa_xpu_cq_compat(c))
			ret = -FI_EOPNOTSUPP;
		else if (!buf)
			ret = -FI_EINVAL;
		else
			ret = efa_xpu_cq_poll(c, buf, count > INT32_MAX ?
					      (uint32_t) INT32_MAX :
					      (uint32_t) count);
	}
	efa_xpu_group_sync(&g);

	return efa_xpu_group_leave(&g, ret);
}

/** Read the failed completion at the head of the queue, as a read's group. */
FI_XPU_FUNC int64_t
efa_xpu_cq_readerr(void *cq, void *buf, uint64_t flags, int scope)
{
	struct efa_xpu_group g;
	int ret;

	(void)flags;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	if (g.leader) {
		struct efa_xpu_cq *c = efa_xpu_cq_resolve(cq);

		if (!efa_xpu_cq_compat(c))
			ret = -FI_EOPNOTSUPP;
		else if (!buf)
			ret = -FI_EINVAL;
		else
			ret = efa_xpu_cq_take_err(c, (struct fi_cq_err_entry *) buf);
	}
	efa_xpu_group_sync(&g);

	return efa_xpu_group_leave(&g, ret);
}

#endif /* FI_XPU_DEVICE_EFA_H */
