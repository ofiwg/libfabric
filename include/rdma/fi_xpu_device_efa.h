/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef FI_XPU_DEVICE_EFA_H
#define FI_XPU_DEVICE_EFA_H

/*
 * EFA Provider — XPU device handle layouts
 *
 * The host-side export calls fill these in and copy them to device
 * memory; the device-side dispatch reads them back. They live in a
 * public header so the compiler can inline the device functions that
 * walk them. Consumers MUST NOT access struct fields directly — treat
 * all handles as opaque and only call fi_xpu_* dispatch functions.
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

#endif /* FI_XPU_DEVICE_EFA_H */
