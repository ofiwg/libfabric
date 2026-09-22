/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef FI_XPU_DEVICE_EFA_H
#define FI_XPU_DEVICE_EFA_H

/*
 * EFA provider - XPU device-side layouts.
 *
 * The host-side export calls fill in the endpoint, completion queue and
 * counter handles below and copy them to device memory; the device-side
 * dispatch reads them back. They live in a public header so the compiler can
 * inline the device functions that walk them. Consumers MUST NOT access
 * struct fields directly - treat all handles as opaque and only call fi_xpu_*
 * dispatch functions.
 *
 * The peer and MR descriptor at the end are what an XPU kernel builds EFA work
 * requests from. The host obtains each with an fi_*() call and copies it to
 * device memory, and the kernel passes it to the device-side data transfer
 * calls; it is never resolved back to an AV entry or an MR on the device. Both
 * are values, not handles, and their sizes are what fi_xpu_ctx_query()
 * reports as av_addr_size and mr_desc_size.
 */

#include <stdint.h>
#include <stddef.h>
#include <rdma/fi_xpu.h>

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
 * read publishes it into device memory; it is NULL when the completions are
 * read on the device, and the queue is then bounded by the application alone.
 * done_seen caches the last value read, so a poster only goes back to done
 * when its cached view says the queue is full.
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
	uint32_t		sq_size;
	uint32_t		pad1;
};

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

#endif /* FI_XPU_DEVICE_EFA_H */
