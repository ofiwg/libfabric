/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_XPU_UTILS_H
#define EFA_GTEST_XPU_UTILS_H

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
 * @brief Whether the device can expose its queues to an XPU kernel, which is
 * what the provider advertises FI_XPU on. Returns 0 when the provider was built
 * without XPU support at all. Only meaningful once the provider has been
 * initialized, i.e. after the first fi_getinfo() in the process, because it
 * reads device state.
 */
int efa_test_xpu_device_support(void);

/* Hardware queue geometry fed to the efadv_query_qp_wqs seam. */
struct efa_test_xpu_wq_geometry {
	uint32_t num_entries;
	uint32_t entry_size;
	uint32_t max_batch;
	void *buffer;
	void *doorbell;
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
