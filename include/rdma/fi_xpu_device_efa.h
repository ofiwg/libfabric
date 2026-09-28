/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef FI_XPU_DEVICE_EFA_H
#define FI_XPU_DEVICE_EFA_H

/*
 * EFA Provider — XPU Device-Side Implementation
 *
 * Inlined device functions for EFA hardware, implementing the OFI XPU
 * device API surface. Consumers include fi_xpu_device.h which pulls in
 * this header and dispatches based on prov_id.
 *
 * The struct definitions below exist so the compiler can inline the
 * functions. Consumers MUST NOT access struct fields directly — treat
 * all handles as opaque and only call fi_xpu_* dispatch functions.
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_eq.h>
#include <rdma/efa_io_defs.h>

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
  #define EFA_XPU_WORK_GROUP_SYNC()		__syncthreads()
  #define EFA_XPU_WORK_GROUP_SIZE() \
	(blockDim.x * blockDim.y * blockDim.z)
  #define EFA_XPU_SHARED			__shared__
#elif defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
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
  #define EFA_XPU_LOAD(ptr)	(*(volatile unsigned int *)(ptr))
  #define EFA_XPU_STORE(ptr, val)	(*(volatile unsigned int *)(ptr) = (val))
  #define EFA_XPU_LOAD64(ptr)	(*(volatile unsigned long long *)(ptr))
  #define EFA_XPU_THREAD_RANK() \
	(threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z))
  #define EFA_XPU_SUBGROUP_MASK()		__activemask()
  #define EFA_XPU_SUBGROUP_LEADER_LANE(mask)	(__ffsll((unsigned long long) \
							 (mask)) - 1)
  #define EFA_XPU_SUBGROUP_LANE()		__lane_id()
  #define EFA_XPU_SUBGROUP_POPC(mask) \
	__popcll((unsigned long long)(mask))
  #define EFA_XPU_SUBGROUP_SYNC(mask) \
	do { (void)(mask); __builtin_amdgcn_wave_barrier(); } while (0)
  #define EFA_XPU_SUBGROUP_BCAST(mask, val, lane) \
	((void)(mask), __shfl((val), (lane)))
  #define EFA_XPU_WORK_GROUP_SYNC()		__syncthreads()
  #define EFA_XPU_WORK_GROUP_SIZE() \
	(blockDim.x * blockDim.y * blockDim.z)
  #define EFA_XPU_SHARED			__shared__
#else
  /* Host-side type checking only — never executed */
  #define EFA_XPU_FENCE_SYSTEM()
  #define EFA_XPU_FENCE_DEVICE()
  #define EFA_XPU_FENCE_BLOCK()
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
  #define EFA_XPU_LOAD(ptr)	(*(volatile unsigned int *)(ptr))
  #define EFA_XPU_STORE(ptr, val)	(*(volatile unsigned int *)(ptr) = (val))
  #define EFA_XPU_LOAD64(ptr)	(*(volatile unsigned long long *)(ptr))
  #define EFA_XPU_THREAD_RANK()			0
  #define EFA_XPU_SUBGROUP_MASK()		1u
  #define EFA_XPU_SUBGROUP_LEADER_LANE(mask)	((void)(mask), 0)
  #define EFA_XPU_SUBGROUP_LANE()		0
  #define EFA_XPU_SUBGROUP_POPC(mask)		((void)(mask), 1)
  #define EFA_XPU_SUBGROUP_SYNC(mask)		((void)(mask))
  #define EFA_XPU_SUBGROUP_BCAST(mask, val, lane) \
	((void)(mask), (void)(lane), (val))
  #define EFA_XPU_WORK_GROUP_SYNC()
  #define EFA_XPU_WORK_GROUP_SIZE()		1
  #define EFA_XPU_SHARED
#endif

/*
 * The NIC-written completion counter wraps at 2^31, so the distance between a
 * slot number and that counter is a 31-bit modular difference. The real number
 * of WQEs in flight is bounded by the queue size, far below 2^31, so masking
 * the subtraction to 31 bits gives the exact distance across either wrap.
 */
#define EFA_XPU_CNTR_MASK		0x7fffffffu

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
};

struct efa_xpu_ep {
	struct fid_xpu_ep	xpu_ep;		/* must be first */
	uint32_t		version;	/* FI_VERSION() of the exporter */
	uint32_t		pad0;
	struct efa_xpu_wq	sq;
	struct efa_xpu_wq	rq;
	uint32_t		pad1;
	uint32_t		sq_req_id_64_bit;
	uint64_t		submitted_count;
	uint32_t		sq_size;
	uint32_t		pad2;
	volatile uint64_t	*local_cntr;
};

/*
 * The completion queue is consumed by every thread that polls it, and cc is the
 * whole of its concurrency control: a poller claims a slot by moving cc past it
 * with a compare and swap, and only the thread that moved it may report what
 * that slot holds.
 *
 * format and user_entry_size describe the caller's buffer, not the ring: they
 * are the fi_cq_format the CQ was opened with and the size of one entry in that
 * format, so a device poll fills the same layout a host fi_cq_read() would.
 *
 * A failing completion is staged in err_* instead of being reported as a
 * successful entry, and err_pending is what makes the next poll report
 * -FI_EAVAIL until the staged error has been read out.
 */
struct efa_xpu_cq {
	struct fid_xpu_cq	xpu_cq;		/* must be first */
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
	uint32_t		err_pending;
	uint32_t		err_status;	/* enum efa_io_comp_status */
	uint32_t		err_dropped;
	uint64_t		err_op_context;
	uint64_t		err_flags;
	uint64_t		err_len;
};

struct efa_xpu_cntr {
	struct fid_xpu_cntr	xpu_cntr;	/* must be first */
	uint32_t		version;	/* FI_VERSION() of the exporter */
	uint32_t		pad0;
	volatile uint64_t	*value;
	volatile uint64_t	*err_value;
};

struct efa_xpu_peer {
	uint16_t ahn;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
};

struct efa_xpu_desc {
	uint32_t lkey;
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
 * A scope names the set of threads that issue an operation together, and every
 * thread of that set issues its own operation: a subgroup of sixteen threads
 * calling efa_xpu_send() posts sixteen work requests, exactly as sixteen
 * separate FI_XPU_WORK_ITEM calls would. What the scope buys is not fewer
 * operations but cheaper ones - the group claims its slots with one atomic add
 * and rings one doorbell for all of them - which is what the man page means by
 * "This allows the implementation to optimize".
 *
 * Reading a completion queue or waiting on a counter is the exception, because
 * there the group shares a single destination: one caller-supplied buffer and
 * one count. Those calls are served by the leader and the result is passed back
 * to the group, so what the others take is a description of the one read that
 * happened.
 *
 * A group-scope call is a collective either way, so a thread of the group that
 * does not reach it leaves the ones that do waiting at a barrier. At
 * FI_XPU_WORK_GROUP that is every thread of the block, and rank 0 in
 * particular, since it is the leader. At FI_XPU_SUBGROUP the group is only the
 * lanes converged at the call, so a lane that took another path is not part of
 * it and nothing waits for it - but that also means a divergent call site forms
 * more than one group, each posting its own batch.
 *
 * FI_XPU_DEVICE would need a barrier across the whole grid, which exists only
 * in a cooperatively launched kernel - a launch the caller would then be bound
 * to, with the grid no larger than the device can hold resident. It is refused
 * because it would buy nothing to pay that with: a block already claims its
 * slots with one atomic and rings its own doorbell, so a grid-wide scope would
 * only add grid barriers on the way to the same queue.
 */
struct efa_xpu_group {
	int		   scope;
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
	if (g->scope == FI_XPU_SUBGROUP)
		EFA_XPU_SUBGROUP_SYNC(g->mask);
	else if (g->scope == FI_XPU_WORK_GROUP)
		EFA_XPU_WORK_GROUP_SYNC();
}

FI_XPU_FUNC int
efa_xpu_group_enter(struct efa_xpu_group *g, int scope)
{
	g->scope = scope;
	g->mask = 0;
	g->leader_lane = 0;
	g->rank = 0;
	g->size = 1;
	g->bcast = 0;

	switch (scope) {
	case FI_XPU_WORK_ITEM:
		g->leader = 1;
		break;
	case FI_XPU_SUBGROUP: {
		unsigned int lane = EFA_XPU_SUBGROUP_LANE();

		g->mask = (unsigned long long) EFA_XPU_SUBGROUP_MASK();
		g->leader_lane = EFA_XPU_SUBGROUP_LEADER_LANE(g->mask);
		g->leader = (lane == (unsigned int) g->leader_lane);
		/*
		 * Rank within the group, not the lane number: the group is only
		 * the lanes in the mask, so a thread's place in it is how many
		 * of them lie below its own lane. That keeps the ranks dense,
		 * which is what makes the reserved slot range contiguous.
		 */
		g->size = EFA_XPU_SUBGROUP_POPC(g->mask);
		g->rank = EFA_XPU_SUBGROUP_POPC(g->mask &
						((1ULL << lane) - 1ULL));
		break;
	}
	case FI_XPU_WORK_GROUP:
		g->rank = (int) EFA_XPU_THREAD_RANK();
		g->size = (int) EFA_XPU_WORK_GROUP_SIZE();
		g->leader = (g->rank == 0);
		break;
	default:
		/* FI_XPU_DEVICE, and anything this build does not know. */
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

	if (g->scope == FI_XPU_SUBGROUP) {
		g->bcast = (uint32_t) EFA_XPU_SUBGROUP_BCAST(g->mask, val,
							     g->leader_lane);
	}
#ifdef EFA_XPU_DEVICE_COMPILE
	else if (g->scope == FI_XPU_WORK_GROUP) {
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
 * Leave a group-scope call with the leader's result.
 *
 * Passing the result through the group is what lets every thread return the
 * same thing. It is how the calls the leader serves on the group's behalf - the
 * completion queue reads - report the one read that happened; for the posting
 * calls, where every thread issued its own operation, it reports that the batch
 * as a whole was accepted.
 */
FI_XPU_FUNC int
efa_xpu_group_leave(struct efa_xpu_group *g, int ret)
{
	if (g->scope == FI_XPU_SUBGROUP)
		return EFA_XPU_SUBGROUP_BCAST(g->mask, ret, g->leader_lane);

#ifdef EFA_XPU_DEVICE_COMPILE
	if (g->scope == FI_XPU_WORK_GROUP) {
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
 * A run is one slot per posting thread, so a single thread claims one and a
 * group claims as many as it has members.
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
 * [db_rung, target) were written by whichever posters held the turn before it,
 * each of which published its own before handing over.
 */
FI_XPU_FUNC void
efa_xpu_wq_ring(struct efa_xpu_wq *wq, uint64_t *submitted_count,
		uint32_t db_rung, uint32_t target)
{
	/*
	 * The acquire half of the handoff. The descriptors about to be rung
	 * were written by other threads, so this fence is not what publishes
	 * them - it is what keeps the doorbell write below from being
	 * reordered ahead of the read that observed the handoff.
	 */
	EFA_XPU_FENCE_SYSTEM();

	*wq->db = target;
	EFA_XPU_FENCE_SYSTEM();	/* order the doorbell MMIO write */

	if (submitted_count)
		EFA_XPU_ATOMIC_ADD64(submitted_count,
				     (uint64_t)(target - db_rung));
	EFA_XPU_STORE(&wq->db_rung, target);
}

/**
 * Wait until the @count slots starting at @base are free to write.
 *
 * @local_cntr is the NIC's completion counter for this queue, or NULL when no
 * counter is bound, in which case there is nothing to learn about how far the
 * NIC has consumed and the queue is only bounded by the staging limit.
 */
FI_XPU_FUNC void
efa_xpu_wq_wait_room(struct efa_xpu_wq *wq, uint32_t base, uint32_t count,
		     volatile uint64_t *local_cntr, uint32_t queue_size,
		     uint64_t *submitted_count)
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
				efa_xpu_wq_ring(wq, submitted_count, db_rung,
						base);
		}
	}

	/*
	 * Wait for the NIC to consume enough of the queue for the highest of
	 * these slots to be free. The counter it writes wraps at 2^31, so the
	 * distance to it is a 31-bit modular difference.
	 */
	if (local_cntr) {
		while (((next - (uint32_t) EFA_XPU_LOAD64(local_cntr)) &
			EFA_XPU_CNTR_MASK) > queue_size)
			;
	}
}

/**
 * Hand the @count slots starting at @base on to the next poster, ringing the
 * doorbell for them first unless @defer asks for the ring to be left to a later
 * post.
 *
 * The descriptors must already be published: they were written by the group's
 * members, and a fence orders only the writes of the thread that issues it, so
 * no thread can publish another's on its behalf.
 */
FI_XPU_FUNC void
efa_xpu_wq_handoff(struct efa_xpu_wq *wq, uint32_t base, uint32_t count,
		   uint64_t *submitted_count, int defer)
{
	uint32_t next = base + count;
	uint32_t db_rung;

	/* Take the turn in slot order, which is what keeps the doorbell sane. */
	while (EFA_XPU_LOAD(&wq->released) != base)
		;

	db_rung = EFA_XPU_LOAD(&wq->db_rung);

	/*
	 * Ring past this run rather than just past the deferred one: holding
	 * the turn means every descriptor below is written, so one doorbell
	 * drains the whole batch.
	 */
	if (!defer || (next - db_rung) >= efa_xpu_wq_batch(wq))
		efa_xpu_wq_ring(wq, submitted_count, db_rung, next);

	EFA_XPU_STORE(&wq->released, next);
}

/*
 * Group posting
 *
 * A group issues one operation per thread, so a group-scope call claims one
 * slot per participating thread rather than one slot for the group: the leader
 * reserves the whole contiguous run with a single atomic add and broadcasts its
 * base, every thread writes its own descriptor into base plus its rank, and the
 * leader rings one doorbell for the batch.
 *
 * The run is walked in chunks of at most max_batch, because that is the number
 * of descriptors the hardware will stage between doorbells and a group can be
 * larger than it. Each chunk is bounded by the leader, written by its members,
 * and handed on by the leader before the next one opens:
 *
 *	struct efa_xpu_post p;
 *
 *	efa_xpu_post_begin(&p, wq, &g, cntr, queue_size, submitted, flags);
 *	while (efa_xpu_post_next(&p))
 *		if (p.active)
 *			<write this thread's descriptor into slot p.slot>
 *
 * efa_xpu_post_next() closes the chunk the previous iteration wrote before it
 * opens the next, which is why the loop body carries no barrier of its own.
 * Every thread of the group must run the loop to completion, including the ones
 * with nothing to write in the current chunk, because closing a chunk is a
 * group barrier.
 */
struct efa_xpu_post {
	struct efa_xpu_wq	*wq;
	struct efa_xpu_group	*g;
	volatile uint64_t	*local_cntr;
	uint64_t		*submitted_count;
	uint32_t		queue_size;
	uint32_t		base;		/* the group's first slot */
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
		   struct efa_xpu_group *g, volatile uint64_t *local_cntr,
		   uint32_t queue_size, uint64_t *submitted_count,
		   uint64_t flags)
{
	uint32_t base = 0;

	p->wq = wq;
	p->g = g;
	p->local_cntr = local_cntr;
	p->submitted_count = submitted_count;
	p->queue_size = queue_size;
	p->next_rank = 0;
	p->chunk_base = 0;
	p->chunk_size = 0;
	p->slot = 0;
	p->active = 0;
	p->open = 0;
	p->defer = !!(flags & FI_MORE);

	/* One atomic add for the group, which is what makes the run contiguous. */
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
	 * so the group waits once here instead of once per thread.
	 */
	if (g->leader)
		efa_xpu_wq_wait_room(p->wq, p->chunk_base, size, p->local_cntr,
				     p->queue_size, p->submitted_count);
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
	 * Each thread publishes its own descriptor, because a fence orders only
	 * the writes of the thread that issues it and the leader cannot publish
	 * the group's on its behalf.
	 */
	EFA_XPU_FENCE_SYSTEM();
	efa_xpu_group_sync(g);

	if (g->leader)
		efa_xpu_wq_handoff(p->wq, p->chunk_base, p->chunk_size,
				   p->submitted_count, p->defer);

	/*
	 * Hold the group until the chunk has been handed on, so the next chunk
	 * sees a release cursor that already covers this one.
	 */
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

/**
 * Post one send queue descriptor per thread of the group.
 *
 * The descriptor arrives already built by the thread that owns it, so all this
 * adds is what depends on the slot that thread ends up with: the slot's phase
 * bit, and - when the send queue has no 64-bit request ID for the application's
 * context to travel in - the slot number as the request ID, which is what a
 * device-side poll then reports back.
 */
FI_XPU_FUNC int
efa_xpu_post_wqe(struct efa_xpu_ep *ep, struct efa_xpu_group *g,
		 struct efa_io_tx_wqe *wqe, uint64_t flags)
{
	struct efa_xpu_post p;

	efa_xpu_post_begin(&p, &ep->sq, g, ep->local_cntr, ep->sq_size,
			   &ep->submitted_count, flags);

	while (efa_xpu_post_next(&p)) {
		uint32_t sq_offset;
		uint64_t *dst;
		size_t i;

		if (!p.active)
			continue;

		if (!ep->sq_req_id_64_bit)
			wqe->meta.req_id = (uint16_t)p.slot;

		/* Phase bit */
		wqe->meta.ctrl2 &= (uint8_t)~EFA_IO_TX_META_DESC_PHASE_MASK;
		if (efa_xpu_wq_slot_phase(&ep->sq, p.slot))
			wqe->meta.ctrl2 |= EFA_IO_TX_META_DESC_PHASE_MASK;

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

/**
 * Post one receive descriptor per thread of the group.
 *
 * The receive queue has no counter the NIC writes, so there is nothing to learn
 * about how far it has consumed and the staging limit is the only bound on the
 * queue here. A receive descriptor has no phase bit and only 16 bits of request
 * ID, which carry the receive queue slot.
 */
FI_XPU_FUNC int
efa_xpu_post_rqe(struct efa_xpu_ep *ep, struct efa_xpu_group *g, void *buf,
		 size_t len, uint32_t lkey, uint64_t flags)
{
	struct efa_xpu_post p;

	efa_xpu_post_begin(&p, &ep->rq, g, NULL, 0, NULL, flags);

	while (efa_xpu_post_next(&p)) {
		struct efa_io_rx_desc rqe;
		uint32_t rq_offset;
		uint32_t *dst;
		size_t i;

		if (!p.active)
			continue;

		rqe.buf_addr_lo = (uint32_t)((uint64_t)(uintptr_t)buf &
					     0xFFFFFFFF);
		rqe.buf_addr_hi = (uint32_t)((uint64_t)(uintptr_t)buf >> 32);
		rqe.req_id = (uint16_t)p.slot;
		rqe.length = (uint16_t)len;
		rqe.lkey_ctrl = (lkey & EFA_IO_RX_DESC_LKEY_MASK) |
				EFA_IO_RX_DESC_FIRST_MASK |
				EFA_IO_RX_DESC_LAST_MASK;

		/* Gathered with memcpy for the reason given in post_wqe(). */
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

/*
 * Write the application's context into the WQE as the request ID.
 *
 * The device has no wr_id pool to translate a 16-bit index through, so an
 * operation posted here can only be completed by a host-side fi_cq_read() if
 * the context travels in the descriptor itself. That needs the SQ's 64-bit
 * request ID support; without it the request ID is the send queue slot, filled
 * in by efa_xpu_post_wqe() once the slot is known, which is all a device-side
 * CQ poll needs.
 */
FI_XPU_FUNC void
efa_xpu_set_wrid(struct efa_xpu_ep *ep, struct efa_io_tx_meta_desc *md,
		    void *context)
{
	uint64_t wr_id = (uint64_t)(uintptr_t)context;

	if (!ep->sq_req_id_64_bit)
		return;

	md->req_id = (uint16_t)wr_id;
	md->req_id_ex.w[0] = (uint16_t)(wr_id >> 16);
	md->req_id_ex.w[1] = (uint16_t)(wr_id >> 32);
	md->req_id_ex.w[2] = (uint16_t)(wr_id >> 48);
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
	 * Every thread of the group builds and posts its own descriptor; only
	 * the slot it lands in is agreed between them.
	 */
	efa_xpu_init_wqe(&wqe, EFA_IO_RDMA_WRITE, p);
	efa_xpu_set_wrid(e, &wqe.meta, context);
	if (flags & FI_REMOTE_CQ_DATA) {
		wqe.meta.ctrl1 |= EFA_IO_TX_META_DESC_HAS_IMM_MASK;
		wqe.meta.immediate_data = (uint32_t)data;
	}
	efa_xpu_set_remote(&wqe.data.rdma_req.remote_mem,
			      (uint32_t)key, addr, (uint32_t)len);
	efa_xpu_set_buf(&wqe.data.rdma_req.local_mem[0], d->lkey,
			   (uint64_t)(uintptr_t)buf, (uint32_t)len);
	wqe.meta.length = 1;

	ret = efa_xpu_post_wqe(e, &g, &wqe, flags);

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

	efa_xpu_init_wqe(&wqe, EFA_IO_RDMA_READ, p);
	efa_xpu_set_wrid(e, &wqe.meta, context);
	efa_xpu_set_remote(&wqe.data.rdma_req.remote_mem,
			      (uint32_t)key, addr, (uint32_t)len);
	efa_xpu_set_buf(&wqe.data.rdma_req.local_mem[0], d->lkey,
			   (uint64_t)(uintptr_t)buf, (uint32_t)len);
	wqe.meta.length = 1;

	ret = efa_xpu_post_wqe(e, &g, &wqe, flags);

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

	efa_xpu_init_wqe(&wqe, EFA_IO_SEND, p);
	efa_xpu_set_wrid(e, &wqe.meta, context);
	if (flags & FI_REMOTE_CQ_DATA) {
		wqe.meta.ctrl1 |= EFA_IO_TX_META_DESC_HAS_IMM_MASK;
		wqe.meta.immediate_data = (uint32_t)data;
	}
	efa_xpu_set_buf(&wqe.data.sgl[0], d->lkey,
			   (uint64_t)(uintptr_t)buf, (uint32_t)len);
	wqe.meta.length = 1;

	ret = efa_xpu_post_wqe(e, &g, &wqe, flags);

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
	(void)context;

	if (!efa_xpu_ep_compat(e))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/* One receive descriptor per thread, as on the send queue. */
	ret = efa_xpu_post_rqe(e, &g, buf, len, d->lkey, flags);

	return efa_xpu_group_leave(&g, ret);
}

/*
 * Counter operations
 *
 * A counter is only read here, never advanced, so every thread that reads one
 * gets the same answer whatever scope it is called at and the reads need no
 * agreement between threads. efa_xpu_cntr_read and efa_xpu_cntr_readerr return
 * the counter value itself, so they also have no way to report an incompatible
 * handle or an unsupported scope; a kernel reaches them only after the endpoint
 * or completion queue it drives has already been accepted by
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
	struct efa_xpu_group g;

	(void)timeout;

	/* Never spin on a handle whose layout this kernel cannot interpret. */
	if (!efa_xpu_cntr_compat(c))
		return;

	if (efa_xpu_group_enter(&g, scope))
		return;

	/*
	 * Every thread sees the same counter, so the wait needs no leader. What
	 * the group scope adds is the barrier below: no thread of the group
	 * leaves the wait until all of them have seen the threshold reached.
	 */
	while (*c->value < threshold)
		;

	efa_xpu_group_sync(&g);
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
 * The request ID a send queue completion carries.
 *
 * A device-posted work request puts the application's context in the request ID
 * itself, because the device has no wr_id pool to translate an index through.
 * That needs the send queue's 64-bit request IDs, which fi_ep_export_xpu reports
 * in the endpoint handle: without them the upper words are not part of the
 * request ID and what comes back here is the send queue slot the descriptor was
 * posted to.
 */
FI_XPU_FUNC uint64_t
efa_xpu_cqe_req_id(const struct efa_io_tx_cdesc *tx)
{
	return (uint64_t) tx->common.req_id |
	       ((uint64_t) tx->req_id_ex.w[0] << 16) |
	       ((uint64_t) tx->req_id_ex.w[1] << 32) |
	       ((uint64_t) tx->req_id_ex.w[2] << 48);
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
	size_t len;

	e.flags = efa_xpu_cqe_flags(cqe->flags);
	e.buf = NULL;
	e.data = 0;

	if (q_type == EFA_IO_SEND_QUEUE) {
		e.op_context = (void *) (uintptr_t)
			efa_xpu_cqe_req_id((const struct efa_io_tx_cdesc *) cqe);
		e.len = 0;
	} else {
		const struct efa_io_rx_cdesc *rx =
			(const struct efa_io_rx_cdesc *) cqe;

		/*
		 * A receive descriptor has only 16 bits of request ID, so what
		 * comes back is the receive queue slot rather than a context.
		 * An unsolicited completion has no descriptor behind it at all.
		 */
		e.op_context = (cqe->flags & EFA_IO_CDESC_COMMON_UNSOLICITED_MASK) ?
			NULL : (void *) (uintptr_t) cqe->req_id;
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
 * Stage a failing completion for fi_xpu_cq_readerr().
 *
 * There is room for one, which is all a kernel that stops at the first error
 * needs; a second error while one is still staged is counted and dropped rather
 * than overwriting what has not been read yet.
 */
FI_XPU_FUNC void
efa_xpu_cq_stage_err(struct efa_xpu_cq *c,
		     const struct efa_io_cdesc_common *cqe)
{
	struct fi_cq_data_entry e;

	if (EFA_XPU_ATOMIC_CAS(&c->err_pending, 0u, 1u) != 0u) {
		EFA_XPU_ATOMIC_ADD(&c->err_dropped, 1u);
		return;
	}

	efa_xpu_cq_fill(c, cqe, &e);
	c->err_op_context = (uint64_t) (uintptr_t) e.op_context;
	c->err_flags = e.flags;
	c->err_len = e.len;
	EFA_XPU_STORE(&c->err_status, cqe->status);
}

/**
 * Take the next completion, if the NIC has written one.
 *
 * The consumer cursor is claimed with a compare and swap rather than an
 * unconditional add: a thread that finds no completion has to leave the cursor
 * where it was for the next caller, and only the thread that moves the cursor
 * off a slot may report the completion in it. Which phase bit a slot should
 * carry follows from the cursor, so the cursor is the only state to move.
 *
 * The slot stays readable after the claim - the NIC does not write it again
 * until the ring has wrapped all the way round - so the caller reads the
 * completion out of the ring rather than under the cursor.
 */
FI_XPU_FUNC const struct efa_io_cdesc_common *
efa_xpu_cq_consume(struct efa_xpu_cq *c)
{
	for (;;) {
		uint32_t cc = EFA_XPU_LOAD(&c->cc);
		uint32_t cq_offset = (cc & c->queue_mask) * c->entry_size;
		uint8_t *slot = c->buf + cq_offset;
		volatile uint8_t *flags = (volatile uint8_t *) slot +
			offsetof(struct efa_io_cdesc_common, flags);
		int expected = (c->init_phase ^
				(int)(cc >> c->queue_size_shift)) & 1;
		int actual = *flags & EFA_IO_CDESC_COMMON_PHASE_MASK;

		if (actual != expected)
			return NULL;	/* no completion ready */

		if (EFA_XPU_ATOMIC_CAS(&c->cc, cc, cc + 1) != cc)
			continue;	/* another thread took this one */

		/*
		 * Nothing else in the completion may be read before its phase
		 * bit has been seen, or a wrapped ring hands back the previous
		 * pass's contents.
		 */
		EFA_XPU_FENCE_BLOCK();
		return (const struct efa_io_cdesc_common *) slot;
	}
}

/**
 * Read up to @count completions into @buf.
 *
 * Pollers do not divide the queue between them ahead of time: each claims one
 * completion at a time, so any number of threads can poll the same CQ
 * concurrently and each fills its own buffer with the completions it claimed.
 * @buf may be NULL, which consumes the completions without reporting them - a
 * kernel that only needs to know how many operations finished.
 */
FI_XPU_FUNC int64_t
efa_xpu_cq_poll(struct efa_xpu_cq *c, void *buf, size_t count)
{
	int64_t n = 0;

	/* A staged error is reported before anything after it. */
	if (EFA_XPU_LOAD(&c->err_pending))
		return -FI_EAVAIL;

	while ((size_t) n < count) {
		const struct efa_io_cdesc_common *cqe = efa_xpu_cq_consume(c);

		if (!cqe)
			break;

		if (cqe->status != EFA_XPU_COMP_STATUS_OK) {
			efa_xpu_cq_stage_err(c, cqe);
			return n ? n : -FI_EAVAIL;
		}

		if (buf)
			efa_xpu_cq_fill(c, cqe,
					(uint8_t *) buf +
					(size_t) n * c->user_entry_size);
		n++;
	}

	return n ? n : -FI_EAGAIN;
}

FI_XPU_FUNC int64_t
efa_xpu_cq_read(void *cq, void *buf, size_t count, int scope)
{
	struct efa_xpu_cq *c = efa_xpu_cq_resolve(cq);
	struct efa_xpu_group g;
	int ret;

	if (!efa_xpu_cq_compat(c))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	/*
	 * A group polls once and shares the result rather than polling once per
	 * thread the way it posts once per thread: there is one buffer and one
	 * count between them, so the leader reads into the buffer the group
	 * passed and what the others take back is how many entries are in it.
	 * The count read and every error code fit in an int, which is what the
	 * group can pass between its threads.
	 */
	if (g.leader)
		ret = (int) efa_xpu_cq_poll(c, buf, count);

	return efa_xpu_group_leave(&g, ret);
}

/**
 * Read the staged error completion.
 *
 * Only the fields the hardware completion carries are filled; prov_errno is the
 * EFA completion status, which is what fi_cq_strerror() on the host turns into a
 * message.
 */
FI_XPU_FUNC int64_t
efa_xpu_cq_readerr(void *cq, void *buf, uint64_t flags, int scope)
{
	struct efa_xpu_cq *c = efa_xpu_cq_resolve(cq);
	struct efa_xpu_group g;
	int ret;

	(void)flags;

	if (!efa_xpu_cq_compat(c))
		return -FI_EOPNOTSUPP;

	ret = efa_xpu_group_enter(&g, scope);
	if (ret)
		return ret;

	if (g.leader) {
		if (!EFA_XPU_LOAD(&c->err_pending)) {
			ret = -FI_EAGAIN;
		} else {
			struct fi_cq_err_entry *e =
				(struct fi_cq_err_entry *) buf;

			if (e) {
				memset(e, 0, sizeof(*e));
				e->op_context =
					(void *) (uintptr_t) c->err_op_context;
				e->flags = c->err_flags;
				e->len = c->err_len;
				e->prov_errno = (int) EFA_XPU_LOAD(&c->err_status);
				e->err = FI_EIO;
			}
			EFA_XPU_FENCE_BLOCK();
			EFA_XPU_STORE(&c->err_pending, 0u);
			ret = 1;
		}
	}

	return efa_xpu_group_leave(&g, ret);
}

#endif /* FI_XPU_DEVICE_EFA_H */
