/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/* C-linkage bridge for the work request (efa_wr.c) tests: flush, prepare,
 * queue_tx, and queue_recv. See efa_gtest_common_helpers.h for why this
 * exists. */

#ifndef EFA_GTEST_WR_UTILS_H
#define EFA_GTEST_WR_UTILS_H

#include <stddef.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_trigger.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Number of receive work requests staged on the endpoint but not yet
 * handed to the device, i.e. base_ep->recv_wr_index.
 */
size_t efa_test_ep_recv_wr_index(struct fid_ep *ep_fid);

/**
 * @brief Whether the endpoint holds transmit work that has been built but not
 * initiated. On the direct data path that is a non-empty send queue batch
 * (sq.num_wqe_pending); otherwise it is an open ibv_wr_start block
 * (base_ep->is_wr_started).
 */
int efa_test_ep_tx_wr_pending(struct fid_ep *ep_fid);

/**
 * @brief The provider's transmit work request size, i.e. efa_wr_tx_size().
 * Bridged so the C++ test need not link the C-linkage efa_wr.h symbol.
 */
size_t efa_test_wr_tx_size(void);

size_t efa_test_wr_rx_size(size_t num_sge);

uint32_t efa_test_mr_desc_lkey(void *desc);

int efa_test_av_ud_addr(struct fid_ep *ep_fid, fi_addr_t addr, uint16_t *ahn,
			uint16_t *qpn, uint32_t *qkey);

/**
 * @brief Whether this build and the endpoint's QP have the direct data path
 * enabled, i.e. whether efa_wr_queue_tx reaches a live send queue. Tests that
 * post to the send queue skip when this is false.
 */
int efa_test_ep_data_path_direct_enabled(struct fid_ep *ep_fid);

/**
 * @brief The count of send queue entries batched but not yet initiated on the
 * endpoint's QP (sq.num_wqe_pending). Only meaningful when
 * efa_test_ep_data_path_direct_enabled() is true.
 */
unsigned int efa_test_ep_tx_num_wqe_pending(struct fid_ep *ep_fid);

unsigned int efa_test_ep_rx_wqe_posted(struct fid_ep *ep_fid);

/**
 * @brief The op_type (enum efa_io_send_op_type: EFA_IO_SEND / EFA_IO_RDMA_READ
 * / EFA_IO_RDMA_WRITE) stored in a transmit work request efa_wr_prepare
 * formatted. @p wr is the opaque fi_wr handed to fi_wr_prepare. Lets a C++
 * test read the formatted op_type without including the device WQE struct.
 */
int efa_test_wr_tx_op_type(const void *wr);

/**
 * @brief The device op_type values a formatted transmit work request carries,
 * exposed so a C++ test can compare against efa_test_wr_tx_op_type() without
 * including the device headers that define the enum.
 */
int efa_test_wr_op_type_send(void);
int efa_test_wr_op_type_rdma_read(void);
int efa_test_wr_op_type_rdma_write(void);

int efa_test_wr_rx_desc_is_last(const void *wr, size_t index);

void efa_test_wr_tx_ud_addr(const void *wr, uint16_t *ahn, uint16_t *qpn,
			    uint32_t *qkey);

int efa_test_wr_tx_is_inline(const void *wr);

uint16_t efa_test_wr_tx_length(const void *wr);

int efa_test_wr_tx_has_imm(const void *wr);

uint32_t efa_test_wr_tx_imm_data(const void *wr);

int efa_test_wr_tx_processing_hints(const void *wr);

int efa_test_wr_processing_hint_high_pps(void);

void efa_test_wr_tx_sge(const void *wr, enum fi_op_type op_type, size_t index,
			uint64_t *addr, uint32_t *len, uint32_t *lkey);

const void *efa_test_wr_tx_inline_data(const void *wr,
				       enum fi_op_type op_type);

void efa_test_wr_tx_rdma_addr(const void *wr, uint64_t *addr, uint32_t *rkey,
			      uint32_t *len);

void efa_test_wr_rx_desc(const void *wr, size_t index, uint64_t *addr,
			 uint16_t *len, uint32_t *lkey);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_WR_UTILS_H */
