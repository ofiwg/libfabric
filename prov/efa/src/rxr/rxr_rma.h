/*
 * Copyright (c) 2019 Amazon.com, Inc. or its affiliates.
 * All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#ifndef _RXR_RMA_H_
#define _RXR_RMA_H_

#include <rdma/fi_rma.h>

int rxr_rma_verified_copy_iov(struct rxr_ep *ep, struct fi_rma_iov *rma,
			      size_t count, uint32_t flags, struct iovec *iov);

struct rxr_tx_entry *rxr_readrsp_tx_entry_init(struct rxr_ep *rxr_ep,
					       struct rxr_rx_entry *rx_entry);

ssize_t rxr_read(struct fid_ep *ep, void *buf, size_t len, void *desc,
		 fi_addr_t src_addr, uint64_t addr, uint64_t key,
		 void *context);

ssize_t rxr_readv(struct fid_ep *ep, const struct iovec *iov, void **desc,
		  size_t iov_count, fi_addr_t src_addr, uint64_t addr,
		  uint64_t key, void *context);

ssize_t rxr_readmsg(struct fid_ep *ep, const struct fi_msg_rma *msg, uint64_t flags);

ssize_t rxr_write(struct fid_ep *ep, const void *buf, size_t len, void *desc,
		  fi_addr_t dest_addr, uint64_t addr, uint64_t key,
		  void *context);

ssize_t rxr_writev(struct fid_ep *ep, const struct iovec *iov, void **desc,
		   size_t iov_count, fi_addr_t dest_addr, uint64_t addr,
		   uint64_t key, void *context);

ssize_t rxr_writemsg(struct fid_ep *ep, const struct fi_msg_rma *msg,
		     uint64_t flags);

ssize_t rxr_writedata(struct fid_ep *ep, const void *buf, size_t len,
		      void *desc, uint64_t data, fi_addr_t dest_addr,
		      uint64_t addr, uint64_t key, void *context);

ssize_t rxr_inject(struct fid_ep *ep, const void *buf, size_t len,
		   fi_addr_t dest_addr, uint64_t addr, uint64_t key);

ssize_t rxr_inject_data(struct fid_ep *ep, const void *buf, size_t len,
			uint64_t data, fi_addr_t dest_addr, uint64_t addr,
			uint64_t key);

extern struct fi_ops_rma rxr_ops_rma;

#endif
