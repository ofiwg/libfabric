/*
 * Copyright (c) 2022 Amazon.com, Inc. or its affiliates.
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

#include "efa.h"

/**
 * @brief try to fill the desc field of a tx_entry
 *
 * The desc field of tx_entry contains the memory descriptors of
 * the user's data buffer.
 *
 * For EFA provider, a data buffer's memory descriptor is a pointer to an
 * efa_mr object, which contains the memory registration information
 * of the data buffer.
 *
 * EFA provider does not require user to provide a descriptor, when
 * user's data buffer is on host memory (Though user can register
 * its buffer, and provide its descriptor as an optimization).
 *
 * The EFA device requires send buffer to be registered.
 *
 * For a user that did not provide descriptors for the buffer,
 * EFA provider need to bridge the gap. It has 2 solutions for
 * this issue:
 *
 * First, EFA provider can copy the user data to a pre-registered bounce
 * buffer, then send data from bounce buffer.
 *
 * Second, EFA provider can register the user's buffer and fill tx_entry->desc
 * (by calling this function). then send directly from send buffer.
 *
 * Because of the high cost of memory registration, this function
 * check the availibity of MR cache, and only register memory
 * when MR cache is available.
 *
 * Also memory registration may fail due to limited resources, in which
 * case tx_entry->desc will not be filled either.
 *
 * Because this function is not guaranteed to fill tx_entry->desc,
 * caller is responsible to ensure the message transfer
 * can continue if tx_entry->desc is not available, which usually
 * means to use the 1st solution (bounce buffer).
 *
 * @param[in,out]	tx_entry	contains the inforation of a TX operation
 * @param[in]		efa_domain	where memory regstration function operates from
 * @param[in]		mr_iov_start	the IOV index to start generating descriptors
 * @param[in]		access		the access flag for the memory registation.
 *
 */
void rxr_tx_entry_try_fill_desc(struct rxr_tx_entry *tx_entry,
				struct efa_domain *efa_domain,
				int mr_iov_start, uint64_t access)
{
	int i, err;

	if (!efa_is_cache_available(efa_domain))
		return;

	for (i = mr_iov_start; i < tx_entry->iov_count; ++i) {
		if (tx_entry->desc[i])
			continue;

		if (tx_entry->iov[i].iov_len <= rxr_env.max_memcpy_size) {
			assert(!tx_entry->mr[i]);
			continue;
		}

		err = fi_mr_reg(&efa_domain->util_domain.domain_fid,
				tx_entry->iov[i].iov_base,
				tx_entry->iov[i].iov_len,
				access, 0, 0, 0,
				&tx_entry->mr[i], NULL);
		if (err) {
			FI_WARN(&rxr_prov, FI_LOG_EP_CTRL,
				"fi_mr_reg failed! buf: %p len: %ld access: %lx",
				tx_entry->iov[i].iov_base, tx_entry->iov[i].iov_len,
				access);

			tx_entry->mr[i] = NULL;
		} else {
			tx_entry->desc[i] = fi_mr_desc(tx_entry->mr[i]);
		}
	}
}

