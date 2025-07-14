/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef _EFA_CQDIRECT_STRUCTS_H
#define _EFA_CQDIRECT_STRUCTS_H

#include "config.h"

#if HAVE_EFA_CQ_DIRECT

#include "efa_cqdirect_efa_io_defs.h"
#include <asm/types.h>
// #include <infiniband/efadv.h>

/*
 * The contents of this file only make sense if we can query rdma-core for QP
 * and CQ information.
 */

/*** TODO: adjust CQ_INLINE_MODE and then refactor to remove
 * efa_cqdirect_internal.h CQ_INLINE_MODE=0: not even inline hints are given.
 * CQ_INLINE_MODE=1: typical "static inline" hint
 * CQ_INLINE_MODE=2: __attribute__((always_inline)) forces the issue even at
 * -O0. ALSO all CQ functions are inlined within libfabric, and defined before
 * the entry functions.
 *
 */
#define CQ_INLINE_MODE 2

#if CQ_INLINE_MODE == 0
#define MAYBE_INLINE static
#define ENTRY_FUN
#include "efa_cqdirect_internal.h"
#endif

#if CQ_INLINE_MODE == 1
#define MAYBE_INLINE static inline
#define ENTRY_FUN
#endif

#if CQ_INLINE_MODE == 2
#define MAYBE_INLINE __attribute__((always_inline)) static inline
#define ENTRY_FUN    static inline
#endif

struct efa_cqdirect_wq {
	/* see `struct efa_wq` in rdma-core/providers/efa/efa.h */

	uint64_t *wrid;
	/* wrid_idx_pool: Pool of free indexes in the wrid array, used to select
	 * the wrid entry to be used to hold the next tx packet's context. At
	 * init time, entry N will hold value N, as OOO tx-completions arrive,
	 * the value stored in a given entry might not equal the entry's index.
	 */
	uint32_t *wrid_idx_pool;
	uint32_t wqe_cnt;
	uint32_t wqe_size;
	uint32_t wqe_posted;
	uint32_t wqe_completed;
	uint16_t pc; /* Producer counter */
	uint16_t desc_mask;
	/* wrid_idx_pool_next: Index of the next entry to use in wrid_idx_pool.
	 */
	uint16_t wrid_idx_pool_next;
	int phase;
	struct ofi_genlock *wqlock;

	uint32_t *db;
};

struct efa_cqdirect_cq {
	/* combines fi_efa_cq_attr (public) with rdma-core's private efa_sub_cq
	 */

	uint8_t *buffer;
	uint32_t entry_size;
	uint32_t num_entries;

	struct efa_io_cdesc_common *cur_cqe;
	struct efa_qp *cur_qp;
	struct efa_cqdirect_wq *cur_wq;
	int phase;
	int qmask;
	uint16_t consumed_cnt;
};

struct efa_cqdirect_rq {
	/* see efa_rq in rdma-core/providers/efa/efa.h */
	struct efa_cqdirect_wq wq;
	uint8_t *buf;
	// size_t buf_size;
};

#define EFA_CQDIRECT_TX_WQE_MAX_CACHE 1
struct efa_cqdirect_sq {
	/* see efa_sq in rdma-core/providers/efa/efa.h */
	struct efa_cqdirect_wq wq;
	uint8_t *desc; // this is the "buf" for the sq.

	/* cqdirect change:  Number of WR entries we have accepted without
	   ringing doorbell, however we copy each wqe as soon as we finish
	   building it. */
	uint32_t num_wqe_pending;

	/* Current wqe being built. */
	struct efa_io_tx_wqe curr_tx_wqe;
};

struct efa_cqdirect_qp {
	struct efa_cqdirect_sq sq;
	struct efa_cqdirect_rq rq;
	int wr_session_err;
};

#endif /* HAVE_EFA_CQ_DIRECT */
#endif /* _EFA_CQDIRECT_STRUCTS_H */