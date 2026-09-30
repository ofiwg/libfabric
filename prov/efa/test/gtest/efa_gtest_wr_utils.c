/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_base_ep.h"
#include "efa_io_defs.h"
#include "efa_wr.h"
#include "efa_gtest_wr_utils.h"

static struct efa_base_ep *efa_test_wr_base_ep(struct fid_ep *ep_fid)
{
	return container_of(ep_fid, struct efa_base_ep, util_ep.ep_fid);
}

size_t efa_test_ep_recv_wr_index(struct fid_ep *ep_fid)
{
	return efa_test_wr_base_ep(ep_fid)->recv_wr_index;
}

int efa_test_ep_tx_wr_pending(struct fid_ep *ep_fid)
{
	struct efa_base_ep *base_ep = efa_test_wr_base_ep(ep_fid);

#if HAVE_EFA_DATA_PATH_DIRECT
	if (base_ep->qp->data_path_direct_enabled)
		return base_ep->qp->data_path_direct_qp.sq.num_wqe_pending != 0;
#endif
	return base_ep->is_wr_started;
}

int efa_test_ep_data_path_direct_enabled(struct fid_ep *ep_fid)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return efa_test_wr_base_ep(ep_fid)->qp->data_path_direct_enabled;
#else
	(void) ep_fid;
	return 0;
#endif
}

int efa_test_wr_supported(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return 1;
#else
	return 0;
#endif
}

size_t efa_test_wr_tx_size(void)
{
	return efa_wr_tx_size();
}

int efa_test_wr_tx_op_type(const void *wr)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	const struct efa_io_tx_wqe_128 *wqe = wr;

	return EFA_GET(&wqe->meta.ctrl1, EFA_IO_TX_META_DESC_OP_TYPE);
#else
	(void) wr;
	return -1;
#endif
}

int efa_test_wr_op_type_send(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return EFA_IO_SEND;
#else
	return -1;
#endif
}

int efa_test_wr_op_type_rdma_read(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return EFA_IO_RDMA_READ;
#else
	return -1;
#endif
}

int efa_test_wr_op_type_rdma_write(void)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return EFA_IO_RDMA_WRITE;
#else
	return -1;
#endif
}

unsigned int efa_test_ep_tx_num_wqe_pending(struct fid_ep *ep_fid)
{
#if HAVE_EFA_DATA_PATH_DIRECT
	return efa_test_wr_base_ep(ep_fid)->qp->data_path_direct_qp.sq.num_wqe_pending;
#else
	(void) ep_fid;
	return 0;
#endif
}
