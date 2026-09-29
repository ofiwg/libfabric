/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_WR_H
#define EFA_WR_H

#include <stddef.h>

#include <rdma/fabric.h>
#include <rdma/fi_wr.h>

/*
 * Flush calls.  These initiate work the provider deferred because the
 * operation was posted with FI_MORE, or because it was queued with
 * fi_wr_queue_tx or fi_wr_queue_recv, and are available whether or not the
 * endpoint reports FI_WR.  efa-direct has no tagged receive queue, so it does
 * not implement fi_trecv_flush.
 *
 * A receive queued with fi_wr_queue_recv is invisible to the device until
 * fi_recv_flush, so a send arriving in between finds no buffer.  Unlike
 * FI_MORE, which the next ordinary post resolves, this deferral lasts until
 * the application flushes.
 */
int efa_wr_tx_flush(struct fid_ep *ep_fid, uint64_t flags);
int efa_wr_rx_flush(struct fid_ep *ep_fid, uint64_t flags);

extern struct fi_ops_wr efa_wr_ops;

#endif /* EFA_WR_H */
