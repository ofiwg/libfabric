/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_XPU_UTILS_H
#define EFA_GTEST_XPU_UTILS_H

#include <stdbool.h>
#include <stdint.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_xpu.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Whether this build can export queues to an XPU kernel, which is what
 * the provider advertises FI_XPU on: 0 when it was built against an rdma-core
 * without the queue geometry queries. Whether a given device will serve an
 * export is answered by the export itself, not here.
 */
int efa_test_xpu_device_support(void);

/**
 * @brief Set efa_env.use_data_path_direct and return its previous value. FI_XPU
 * is only offered when the direct data path is on, and efa_env is not reachable
 * from C++.
 */
int efa_test_xpu_set_use_data_path_direct(int value);

/**
 * @brief Set data_path_direct_enabled on the QP behind @p ep and return its
 * previous value, so a test can present an endpoint that is not using the
 * direct data path to fi_ep_export_xpu(). Returns -1 when there is no QP.
 */
int efa_test_xpu_set_ep_data_path_direct(struct fid_ep *ep, int value);

/* Hardware queue geometry fed to the efadv_query_qp_wqs seam. */
struct efa_test_xpu_wq_geometry {
	uint32_t num_entries;
	uint32_t entry_size;
	uint32_t max_batch;
	void *buffer;
	void *doorbell;
	/* Report EFADV_WQ_CAPS_64_BIT_REQ_ID, which an XPU export requires. */
	int req_id_64_bit;
};

/**
 * @brief Fill a struct efadv_wq_attr pair from the given geometry.
 * Keeps the struct layout, which only newer rdma-core releases declare, out of
 * the C++ test.
 *
 * @return 0, or -FI_EOPNOTSUPP if this build has no efadv_query_qp_wqs.
 */
int efa_test_xpu_fill_wq_attrs(void *sq_attr, void *rq_attr,
			       const struct efa_test_xpu_wq_geometry *sq,
			       const struct efa_test_xpu_wq_geometry *rq);

/**
 * @brief Fill a struct efadv_cq_attr from the given ring geometry.
 *
 * @return 0, or -FI_EOPNOTSUPP if this build has no efadv_query_cq.
 */
int efa_test_xpu_fill_cq_attr(void *cq_attr, uint32_t num_entries,
			      uint32_t entry_size, void *buffer);

/**
 * @brief Attach XPU state to an open CQ, as efa_cq_open() does for a CQ opened
 * with FI_XPU, but with a caller-provided ring buffer. Lets a test reach
 * fi_cq_export_xpu() without the device memory allocation the real open
 * performs, which needs an XPU runtime.
 *
 * The state takes ownership of cq_buf_dev: closing the CQ releases it through
 * the XPU context's memory ops.
 */
int efa_test_xpu_install_cq_state(struct fid_cq *cq, struct fid_xpu_ctx *ctx,
				  void *cq_buf_dev);

/**
 * @brief How many XPU endpoint queues a read of this CQ publishes a done
 * cursor for.
 */
int efa_test_xpu_cq_publishers(struct fid_cq *cq);

/**
 * @brief The context table a host poll of the endpoint's QP reads for its send
 * queue (@rq false) or receive queue (@rq true), or NULL.
 */
const void *efa_test_xpu_qp_ctx(struct fid_ep *ep, bool rq);

/**
 * @brief Count one consumed completion against an exported endpoint's send
 * queue (@rq false) or receive queue (@rq true), as a host CQ read does.
 */
void efa_test_xpu_ep_note_done(struct fid_ep *ep, bool rq, uint16_t req_id);
void efa_test_xpu_cq_flush(struct fid_cq *cq);

/**
 * @brief Counter equivalent of efa_test_xpu_install_cq_state. The state takes
 * ownership of both buffers.
 */
int efa_test_xpu_install_cntr_state(struct fid_cntr *cntr,
				    struct fid_xpu_ctx *ctx, void *value_dev,
				    void *err_dev);

/**
 * @brief Insert a peer with the endpoint's own GID but a caller-chosen QP
 * number and qkey, so a test can assert that fi_av_lookup2 reports exactly what
 * was inserted.
 *
 * @return 1 on success, or a negative error code.
 */
int efa_test_xpu_av_insert_peer(struct fid_ep *ep, struct fid_av *av,
				uint16_t qpn, uint32_t qkey, fi_addr_t *addr);

/**
 * @brief The address handle number the AV holds for fi_addr, which is what
 * fi_av_lookup2 must report to the XPU. Returns -1 when there is no handle.
 */
int efa_test_xpu_av_ahn(struct fid_av *av, fi_addr_t fi_addr);

/**
 * @brief The ibv lkey cached at registration, which is what
 * fi_mr_get_xpu_desc must report to the XPU.
 */
uint32_t efa_test_xpu_mr_lkey(struct fid_mr *mr);

#ifdef __cplusplus
}
#endif

#endif /* EFA_GTEST_XPU_UTILS_H */
