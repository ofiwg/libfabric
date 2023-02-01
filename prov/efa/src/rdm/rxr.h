/*
 * Copyright (c) 2019-2022 Amazon.com, Inc. or its affiliates.
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

#ifndef _RXR_H_
#define _RXR_H_

#include <pthread.h>
#include <rdma/fabric.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_ext.h>

#include <ofi.h>
#include <ofi_iov.h>
#include <ofi_proto.h>
#include <ofi_enosys.h>
#include <ofi_rbuf.h>
#include <ofi_list.h>
#include <ofi_util.h>
#include <ofi_tree.h>
#include <uthash.h>
#include <ofi_recvwin.h>
#include <ofi_perf.h>
#include <ofi_hmem.h>
#include "efa_prov.h"
#include "efa_base_ep.h"
#include "rxr_pkt_type.h"
#include "rxr_op_entry.h"
#include "rxr_env.h"
#include "rxr_ep.h"

#ifdef ENABLE_EFA_POISONING
static inline void rxr_poison_mem_region(void *ptr, size_t size)
{
	uint32_t rxr_poison_value = 0xdeadbeef;
	for (int i = 0; i < size / sizeof(rxr_poison_value); i++)
		memcpy((uint32_t *)ptr + i, &rxr_poison_value, sizeof(rxr_poison_value));
}

static inline void rxr_poison_pkt_entry(struct rxr_pkt_entry *pkt_entry, size_t wiredata_size)
{
	rxr_poison_mem_region(pkt_entry->wiredata, wiredata_size);
	/*
	 * Don't poison pkt_entry->wiredata, which is the last element in rxr_pkt_entry
	 * pkt_entry->wiredata is released when the pkt_entry is released
	 */
	rxr_poison_mem_region(pkt_entry, sizeof(struct rxr_pkt_entry) - sizeof(char *));
}
#endif

/*
 * Set alignment to x86 cache line size.
 */
#define RXR_BUF_POOL_ALIGNMENT	(64)

/*
 * will add following parameters to env variable for tuning
 */
#define RXR_DEF_CQ_SIZE			(8192)

/* the default value for rxr_env.rnr_backoff_wait_time_cap */
#define RXR_DEFAULT_RNR_BACKOFF_WAIT_TIME_CAP	(1000000)

/*
 * the maximum value for rxr_env.rnr_backoff_wait_time_cap
 * Because the backoff wait time is multiplied by 2 when
 * RNR is encountered, its value must be < INT_MAX/2.
 * Therefore, its cap must be < INT_MAX/2 too.
 */
#define RXR_MAX_RNR_BACKOFF_WAIT_TIME_CAP	(INT_MAX/2 - 1)

/* bounds for random RNR backoff timeout */
#define RXR_RAND_MIN_TIMEOUT		(40)
#define RXR_RAND_MAX_TIMEOUT		(120)

/* bounds for flow control */
#define RXR_DEF_MIN_TX_CREDITS		(32)

/*
 * maximum time (microseconds) we will allow available_bufs for large msgs to
 * be exhausted
 */
#define RXR_AVAILABLE_DATA_BUFS_TIMEOUT	(5000000)

/*
 * Based on size of tx_id and rx_id in headers, can be arbitrary once those are
 * removed.
 */
#define RXR_MAX_RX_QUEUE_SIZE (UINT32_MAX)
#define RXR_MAX_TX_QUEUE_SIZE (UINT32_MAX)

/*
 * The maximum supported source address length in bytes
 */
#define RXR_MAX_NAME_LENGTH	(32)

/*
 * TODO: In future we will send RECV_CANCEL signal to sender,
 * to stop transmitting large message, this flag is also
 * used for fi_discard which has similar behavior.
 */
#define RXR_RECV_CANCEL		BIT_ULL(3)

/*
 * Flags to tell if the rx_entry is tracking FI_MULTI_RECV buffers
 */
#define RXR_MULTI_RECV_POSTED	BIT_ULL(4)
#define RXR_MULTI_RECV_CONSUMER	BIT_ULL(5)

/*
 * Flag to tell if the transmission is using FI_DELIVERY_COMPLETE
 * protocols
 */

#define RXR_DELIVERY_COMPLETE_REQUESTED	BIT_ULL(6)

#define RXR_OP_ENTRY_QUEUED_RNR BIT_ULL(9)

/*
 * Flag to indicate an rx_entry has an EOR
 * in flight (the EOR has been sent or queued,
 * and has not got send completion)
 * hence the rx_entry cannot be released
 */
#define RXR_EOR_IN_FLIGHT BIT_ULL(10)

/*
 * Flag to indicate a tx_entry has already
 * written an cq error entry for RNR
 */
#define RXR_TX_ENTRY_WRITTEN_RNR_CQ_ERR_ENTRY BIT_ULL(10)

/*
 * Flag to indicate an op_entry has queued ctrl packet,
 * and is on ep->op_entry_queued_ctrl_list
 */
#define RXR_OP_ENTRY_QUEUED_CTRL BIT_ULL(11)

/*
 * OFI flags
 * The 64-bit flag field is used as follows:
 * 1-grow up    common (usable with multiple operations)
 * 59-grow down operation specific (used for single call/class)
 * 60 - 63      provider specific
 */
#define RXR_NO_COMPLETION	BIT_ULL(60)
#define RXR_NO_COUNTER		BIT_ULL(61)

#define RXR_MTU_MAX_LIMIT	BIT_ULL(15)

void rxr_convert_desc_for_shm(int numdesc, void **desc);

void rxr_prepare_desc_send(struct efa_domain *efa_domain,
			   struct rxr_op_entry *tx_entry);

/* Aborts if unable to write to the eq */
static inline void efa_eq_write_error(struct util_ep *ep, ssize_t err,
				      ssize_t prov_errno)
{
	struct fi_eq_err_entry err_entry;
	int ret = -FI_ENOEQ;

	EFA_WARN(FI_LOG_EQ,
		"Writing error to EQ: err: %s (%zd) prov_errno: %s (%zd)\n",
		fi_strerror(err), err,
		efa_strerror(prov_errno), prov_errno);
	if (ep->eq) {
		memset(&err_entry, 0, sizeof(err_entry));
		err_entry.err = err;
		err_entry.prov_errno = prov_errno;
		ret = fi_eq_write(&ep->eq->eq_fid, FI_NOTIFY,
				  &err_entry, sizeof(err_entry),
				  UTIL_FLAG_ERROR);

		if (ret == sizeof(err_entry))
			return;
	}

	EFA_WARN(FI_LOG_EQ, "Unable to write to EQ\n");
	fprintf(stderr,
		"Libfabric EFA provider has encounterd an internal error:\n\n"
		"Libfabric error: (%zd) %s\n"
		"EFA internal error: (%zd) %s\n\n"
		"Your application will now abort().\n",
		err, fi_strerror(err),
		prov_errno, efa_strerror(prov_errno));
	abort();
}

#endif
