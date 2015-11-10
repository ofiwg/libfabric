/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
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

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_nic.h"
#include "gnix_cm_nic.h"
#include "gnix_vc.h"
#include "gnix_mbox_allocator.h"

static int gnix_nics_per_ptag[GNI_PTAG_USER_END];
static DLIST_HEAD(gnix_nic_list);
static pthread_mutex_t gnix_nic_list_lock = PTHREAD_MUTEX_INITIALIZER;

/*
 * globals
 */

uint32_t gnix_max_nics_per_ptag = GNIX_DEF_MAX_NICS_PER_PTAG;

/*
 * local variables
 */

static struct gnix_nic_attr default_attr = {
		.gni_cdm_hndl        = NULL,
		.gni_nic_hndl        = NULL
};

/*******************************************************************************
 * Helper functions.
 ******************************************************************************/

/*
 * place holder for better attributes checker
 */
static int __gnix_nic_check_attr_sanity(struct gnix_nic_attr *attr)
{
	return FI_SUCCESS;
}

static inline struct gnix_tx_descriptor *
__desc_lkup_by_id(struct gnix_nic *nic, int desc_id)
{
	struct gnix_tx_descriptor *tx_desc;

	assert((desc_id >= 0) && (desc_id < nic->max_tx_desc_id));
	tx_desc = &nic->tx_desc_base[desc_id];
	return tx_desc;
}

static int __nic_rx_overrun(struct gnix_nic *nic)
{
	int i, max_id, ret;
	struct gnix_vc *vc;
	gni_return_t status;
	gni_cq_entry_t cqe;

	GNIX_WARN(FI_LOG_EP_DATA, "\n");

	/* clear out the CQ */
	while ((status = GNI_CqGetEvent(nic->rx_cq, &cqe)) == GNI_RC_SUCCESS);
	assert(status == GNI_RC_NOT_DONE);

	fastlock_acquire(&nic->vc_id_lock);
	max_id = nic->vc_id_table_count;
	fastlock_release(&nic->vc_id_lock);
	/*
	 * TODO: optimization would
	 * be to keep track of last time
	 * this happened and where smsg msgs.
	 * were found.
	 */
	for (i = 0; i < max_id; i++) {
		ret = _gnix_test_bit(&nic->vc_id_bitmap, i);
		if (ret) {
			vc = __gnix_nic_elem_by_rem_id(nic, i);
			if (unlikely(vc->conn_state != GNIX_VC_CONNECTED))
				_gnix_set_bit(&vc->flags,
					      GNIX_VC_FLAG_RX_PENDING);
			ret = _gnix_vc_dequeue_smsg(vc);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_DATA,
					  "_gnix_vc_dqueue_smsg returned %d\n",
					  ret);
			}
		}
	}

	return FI_SUCCESS;
}

static int process_rx_cqe(struct gnix_nic *nic, gni_cq_entry_t cqe)
{
	int ret = FI_SUCCESS, vc_id = 0;
	struct gnix_vc *vc;

	vc_id =  GNI_CQ_GET_INST_ID(cqe);
	vc = __gnix_nic_elem_by_rem_id(nic, vc_id);

#if 0 /* Process RX inline with arrival of an RX CQE. */
	if (unlikely(vc->conn_state != GNIX_VC_CONNECTED)) {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Scheduling VC for RX processing (%p)\n",
			  vc);
		_gnix_set_bit(&vc->flags, GNIX_VC_FLAG_RX_PENDING);
		ret = _gnix_vc_schedule(vc);
		assert(ret == FI_SUCCESS);
	} else {
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Processing VC RX (%p)\n",
			  vc);
		ret = _gnix_vc_dequeue_smsg(vc);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA,
					"_gnix_vc_dqueue_smsg returned %d\n",
					ret);
		}

		/* The RX CQE could be associated with an SMSG credit return.
		 * Schedule the VC just in case. */
		ret = _gnix_vc_schedule(vc);
		assert(ret == FI_SUCCESS);
	}
#else /* Defer RX processing until after the RX CQ is cleared. */
	_gnix_set_bit(&vc->flags, GNIX_VC_FLAG_RX_PENDING);
	ret = _gnix_vc_schedule(vc);
	assert(ret == FI_SUCCESS);
#endif

	return ret;
}

static int __nic_rx_progress(struct gnix_nic *nic)
{
	int ret = FI_SUCCESS;
	gni_return_t status = GNI_RC_NOT_DONE;
	gni_cq_entry_t cqe;

	status = GNI_CqTestEvent(nic->rx_cq);
	if (status == GNI_RC_NOT_DONE)
		return FI_SUCCESS;

	fastlock_acquire(&nic->lock);

	do {
		status = GNI_CqGetEvent(nic->rx_cq, &cqe);
		if (unlikely(status == GNI_RC_NOT_DONE)) {
			ret = FI_SUCCESS;
			break;
		}

		if (likely(status == GNI_RC_SUCCESS)) {
			/* Find and schedule the associated VC. */
			ret = process_rx_cqe(nic, cqe);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_DATA,
					  "process_rx_cqe() failed: %d\n",
					  ret);
			}
		} else if (status == GNI_RC_ERROR_RESOURCE) {
			/* The remote CQ was overrun.  Events related to any VC
			 * could have been missed.  Schedule each VC to be sure
			 * all messages are processed. */
			assert(GNI_CQ_OVERRUN(cqe));
			__nic_rx_overrun(nic);
		} else {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "GNI_CqGetEvent returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			break;
		}
	} while (1);

	fastlock_release(&nic->lock);

	return ret;
}

void _gnix_nic_txd_err_inject(struct gnix_nic *nic,
			      struct gnix_tx_descriptor *txd)
{
	slist_insert_tail(&txd->err_list, &nic->err_txds);
}

static int __gnix_nic_txd_err_get(struct gnix_nic *nic,
				  struct gnix_tx_descriptor **txd)
{
	struct slist_entry *list_entry;
	struct gnix_tx_descriptor *txd_p;

	list_entry = slist_remove_head(&nic->err_txds);
	if (list_entry) {
		txd_p = container_of(list_entry,
				     struct gnix_tx_descriptor,
				     err_list);
		*txd = txd_p;
		return 1;
	}

	return 0;
}

static int __nic_get_completed_txd(struct gnix_nic *nic,
				   struct gnix_tx_descriptor **txd,
				   gni_return_t *tx_status)
{
	gni_post_descriptor_t *gni_desc;
	struct gnix_tx_descriptor *txd_p = NULL;
	gni_return_t status;
	int msg_id;
	gni_cq_entry_t cqe;
	uint32_t recov;

	if (__gnix_nic_txd_err_get(nic, &txd_p)) {
		*txd = txd_p;
		*tx_status = GNI_RC_TRANSACTION_ERROR;
		return 1;
	}

	status = GNI_CqGetEvent(nic->tx_cq, &cqe);
	if (status == GNI_RC_NOT_DONE) {
		return 0;
	}

	assert(status == GNI_RC_SUCCESS ||
	       status == GNI_RC_TRANSACTION_ERROR);

	if (status == GNI_RC_TRANSACTION_ERROR) {
		status = GNI_CqErrorRecoverable(cqe, &recov);
		if (status != GNI_RC_SUCCESS || !recov) {
			char ebuf[512];
			GNI_CqErrorStr(cqe, ebuf, sizeof(ebuf));
			GNIX_WARN(FI_LOG_EP_DATA,
					"CQ error status - %s\n",
					ebuf);
			/* fatal */
			assert(0);
		}
	}

	if (GNI_CQ_GET_TYPE(cqe) == GNI_CQ_EVENT_TYPE_POST) {
		status = GNI_GetCompleted(nic->tx_cq, cqe, &gni_desc);

		assert(status == GNI_RC_SUCCESS ||
		       status == GNI_RC_TRANSACTION_ERROR);

		txd_p = container_of(gni_desc,
				   struct gnix_tx_descriptor,
				   gni_desc);
	} else if (GNI_CQ_GET_TYPE(cqe) == GNI_CQ_EVENT_TYPE_SMSG) {
		msg_id = GNI_CQ_GET_MSG_ID(cqe);
		txd_p = __desc_lkup_by_id(nic, msg_id);
	}

	if (!txd_p) {
		GNIX_WARN(FI_LOG_EP_DATA, "Unexpected CQE: 0x%lx", cqe);
		/* fatal */
		assert(0);
	}

	*tx_status = status;
	*txd = txd_p;

	return 1;
}

/*
 * function to process GNI CQ TX CQES to progress a gnix_nic
 */

static int __nic_tx_progress(struct gnix_nic *nic)
{
	int ret = FI_SUCCESS;
	gni_return_t tx_status;
	struct gnix_tx_descriptor *txd = NULL;

	do {
		fastlock_acquire(&nic->lock);
		if (!__nic_get_completed_txd(nic, &txd, &tx_status)) {
			fastlock_release(&nic->lock);
			ret = FI_SUCCESS;
			break;
		}
		fastlock_release(&nic->lock);

		if (txd->completer_fn) {
			ret = txd->completer_fn(txd, tx_status);
			if (ret != FI_SUCCESS)
				GNIX_WARN(FI_LOG_EP_DATA,
					  "TXD completer failed: %d", ret);
		}
	} while (1);

	return ret;
}

int __nic_vc_progress(struct gnix_nic *nic)
{
	struct gnix_vc *vc;
	int ret;

	while ((vc = _gnix_nic_next_pending_vc(nic))) {
		ret = _gnix_vc_progress(vc);
		if (ret != FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Rescheduling VC (%p)\n", vc);
			ret = _gnix_vc_schedule(vc);
			assert(ret == FI_SUCCESS);
		}
	}

	return FI_SUCCESS;
}

int _gnix_nic_progress(struct gnix_nic *nic)
{
	int ret = FI_SUCCESS;

	ret =  __nic_tx_progress(nic);
	if (unlikely(ret != FI_SUCCESS))
		return ret;

	ret = __nic_rx_progress(nic);
	if (unlikely(ret != FI_SUCCESS))
		return ret;

	ret = __nic_vc_progress(nic);
	if (unlikely(ret != FI_SUCCESS))
		return ret;

	return ret;
}

int _gnix_nic_free_rem_id(struct gnix_nic *nic, int remote_id)
{
	assert(nic);

	if ((remote_id < 0) || (remote_id > nic->vc_id_table_count))
		return -FI_EINVAL;

	_gnix_clear_bit(&nic->vc_id_bitmap, remote_id);

	return FI_SUCCESS;
}

/*
 * this function is needed to allow for quick lookup of a vc based on
 * the contents of the GNI CQE coming off of the GNI RX CQ associated
 * with GNI nic being used by this VC.  Using a bitmap to expedite
 * scanning vc's in the case of a GNI CQ overrun.
 */

int _gnix_nic_get_rem_id(struct gnix_nic *nic, int *remote_id, void *entry)
{
	int ret = FI_SUCCESS;
	void **table_base;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	/*
	 * TODO:  really need to search bitmap for clear
	 * bit before resizing the table
	 */

	fastlock_acquire(&nic->vc_id_lock);
	if (nic->vc_id_table_capacity == nic->vc_id_table_count) {
		table_base = realloc(nic->vc_id_table,
				     2 * nic->vc_id_table_capacity *
				     sizeof(void *));
		if (table_base == NULL) {
			ret =  -FI_ENOMEM;
			goto err;
		}
		nic->vc_id_table_capacity *= 2;
		nic->vc_id_table = table_base;
	}

	nic->vc_id_table[nic->vc_id_table_count] = entry;
	*remote_id = nic->vc_id_table_count;

	/*
	 * set bit in the bitmap
	 */

	_gnix_set_bit(&nic->vc_id_bitmap, nic->vc_id_table_count);

	++(nic->vc_id_table_count);
err:
	fastlock_release(&nic->vc_id_lock);
	return ret;
}

/*
 * allocate a tx desc for this nic
 */

int _gnix_nic_tx_alloc(struct gnix_nic *nic,
		       struct gnix_tx_descriptor **desc)
{
	struct dlist_entry *entry;

	fastlock_acquire(&nic->tx_desc_lock);
	if (dlist_empty(&nic->tx_desc_free_list)) {
		fastlock_release(&nic->tx_desc_lock);
		return -FI_ENOSPC;
	}

	entry = nic->tx_desc_free_list.next;
	dlist_remove_init(entry);
	dlist_insert_head(entry, &nic->tx_desc_active_list);
	*desc = dlist_entry(entry, struct gnix_tx_descriptor, list);
	fastlock_release(&nic->tx_desc_lock);

	return FI_SUCCESS;
}

/*
 * free a tx desc for this nic - the nic is not embedded in the
 * descriptor to help keep it small
 */

int _gnix_nic_tx_free(struct gnix_nic *nic,
		      struct gnix_tx_descriptor *desc)
{
	fastlock_acquire(&nic->tx_desc_lock);
	dlist_remove_init(&desc->list);
	dlist_insert_head(&desc->list, &nic->tx_desc_free_list);
	fastlock_release(&nic->tx_desc_lock);

	return FI_SUCCESS;
}

/*
 * allocate a free list of tx descs for a gnix_nic struct.
 */

static int __gnix_nic_tx_freelist_init(struct gnix_nic *nic, int n_descs)
{
	int i, ret = FI_SUCCESS;
	struct gnix_tx_descriptor *desc_base, *desc_ptr;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	/*
	 * set up free list of tx descriptors.
	 */

	desc_base = calloc(n_descs, sizeof(struct gnix_tx_descriptor));
	if (desc_base == NULL) {
		ret = -FI_ENOMEM;
		goto err;
	}

	dlist_init(&nic->tx_desc_free_list);
	dlist_init(&nic->tx_desc_active_list);

	for (i = 0, desc_ptr = desc_base; i < n_descs; i++, desc_ptr++) {
		desc_ptr->id = i;
		dlist_insert_tail(&desc_ptr->list,
				  &nic->tx_desc_free_list);
	}

	nic->max_tx_desc_id = n_descs - 1;
	nic->tx_desc_base = desc_base;

	fastlock_init(&nic->tx_desc_lock);
err:
	return ret;

}

/*
 * clean up the tx descs free list
 */
static void __gnix_nic_tx_freelist_destroy(struct gnix_nic *nic)
{
	free(nic->tx_desc_base);
	fastlock_destroy(&nic->tx_desc_lock);
}

/*
 * free a gnix nic and associated resources if refcnt drops to 0
 */

static void __nic_destruct(void *obj)
{
	int ret = FI_SUCCESS;
	gni_return_t status = GNI_RC_SUCCESS;
	struct gnix_nic *nic = (struct gnix_nic *) obj;

	/* Must free mboxes first, because the MR has a pointer to the
	 * nic handles below */
	ret = _gnix_mbox_allocator_destroy(nic->mbox_hndl);
	if (ret != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "_gnix_mbox_allocator_destroy returned %d\n",
			  ret);

	ret = _gnix_mbox_allocator_destroy(nic->s_rdma_buf_hndl);
	if (ret != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "_gnix_mbox_allocator_destroy returned %d\n",
			  ret);

	ret = _gnix_mbox_allocator_destroy(nic->r_rdma_buf_hndl);
	if (ret != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "_gnix_mbox_allocator_destroy returned %d\n",
			  ret);

	if (!nic->gni_cdm_hndl) {
		GNIX_ERR(FI_LOG_EP_CTRL, "No CDM attached to nic, nic=%p");
	}

	assert(nic->gni_cdm_hndl != NULL);

	if (nic->rx_cq_blk != NULL) {
		status = GNI_CqDestroy(nic->rx_cq_blk);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqDestroy returned %s\n",
				 gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err;
		}
	}

	if (nic->rx_cq != NULL) {
		status = GNI_CqDestroy(nic->rx_cq);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqDestroy returned %s\n",
				 gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err;
		}
	}

	if (nic->tx_cq_blk != NULL) {
		status = GNI_CqDestroy(nic->tx_cq_blk);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqDestroy returned %s\n",
				 gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err;
		}
	}

	if (nic->tx_cq != NULL) {
		status = GNI_CqDestroy(nic->tx_cq);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqDestroy returned %s\n",
				 gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err;
		}
	}

	if (nic->allocd_gni_res & GNIX_NIC_CDM_ALLOCD) {
		status = GNI_CdmDestroy(nic->gni_cdm_hndl);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CdmDestroy returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err;
		}
	}

	if (nic->vc_id_table != NULL) {
		free(nic->vc_id_table);
	} else {
		GNIX_WARN(FI_LOG_EP_CTRL, "vc_id_table was NULL\n");
	}

	/*
	 * remove the nic from the linked lists
	 * for the domain and the global nic list
	 */

err:
	_gnix_free_bitmap(&nic->vc_id_bitmap);

	pthread_mutex_lock(&gnix_nic_list_lock);

	dlist_remove(&nic->gnix_nic_list);
	--gnix_nics_per_ptag[nic->ptag];
	dlist_remove(&nic->dom_nic_list);

	pthread_mutex_unlock(&gnix_nic_list_lock);

	__gnix_nic_tx_freelist_destroy(nic);
	free(nic);
}

int _gnix_nic_free(struct gnix_nic *nic)
{
	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if (nic == NULL)
		return -FI_EINVAL;

	_gnix_ref_put(nic);

	return FI_SUCCESS;
}

/*
 * allocate a gnix_nic struct using attributes of the domain
 */

int gnix_nic_alloc(struct gnix_fid_domain *domain,
		   struct gnix_nic_attr *attr,
		   struct gnix_nic **nic_ptr)
{
	int ret = FI_SUCCESS;
	struct gnix_nic *nic = NULL;
	uint32_t device_addr;
	gni_return_t status;
	uint32_t fake_cdm_id;
	gni_smsg_attr_t smsg_mbox_attr;
	struct gnix_nic_attr *nic_attr = &default_attr;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	*nic_ptr = NULL;

	if (attr) {
		ret = __gnix_nic_check_attr_sanity(attr);
		if (ret != FI_SUCCESS)
			return ret;
		nic_attr = attr;
	}

	/*
	 * If we've maxed out the number of nics for this domain/ptag,
	 * search the list of existing nics.  Take the gnix_nic_list_lock
	 * here since the gnix_nic_list will be manipulated whether or
	 * not we attach to an existing nic or create a new one.
	 *
	 * Should not matter much that this is a pretty fat critical section
	 * since endpoint setup for RDM type will typically occur near
	 * app startup, likely in a single threaded region, and for the
	 * case of MSG, where there will likely be many 100s of EPs, after
	 * a few initial slow times through this section when nics are created,
	 * max nic count for the ptag will be reached and only the first part
	 * of the critical section - iteration over existing nics - will be
	 * happening.
	 */

	pthread_mutex_lock(&gnix_nic_list_lock);

	if (gnix_nics_per_ptag[domain->ptag] >= gnix_max_nics_per_ptag) {
		assert(!dlist_empty(&domain->nic_list));

		nic = dlist_first_entry(&domain->nic_list, struct gnix_nic,
					dom_nic_list);
		dlist_remove(&nic->dom_nic_list);
		dlist_insert_tail(&nic->dom_nic_list, &domain->nic_list);
		_gnix_ref_get(nic);

		GNIX_INFO(FI_LOG_EP_CTRL, "Reusing NIC:%p\n", nic);
	}

	/*
	 * no nic found create a cdm and attach
	 */

	if (!nic) {

		nic = calloc(1, sizeof(struct gnix_nic));
		if (nic == NULL) {
			ret = -FI_ENOMEM;
			goto err;
		}

		ret = _gnix_get_new_cdm_id(domain, &fake_cdm_id);
		if (ret != FI_SUCCESS)
			goto err;

		if (nic_attr->gni_cdm_hndl == NULL) {
			status = GNI_CdmCreate(fake_cdm_id,
						domain->ptag,
						domain->cookie,
						gnix_cdm_modes,
						&nic->gni_cdm_hndl);
			if (status != GNI_RC_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL, "GNI_CdmCreate returned %s\n",
					 gni_err_str[status]);
				ret = gnixu_to_fi_errno(status);
				goto err1;
			}
			nic->allocd_gni_res |= GNIX_NIC_CDM_ALLOCD;
		} else
			nic->gni_cdm_hndl = nic_attr->gni_cdm_hndl;

		/*
		 * Okay, now go for the attach
		*/

		if (nic_attr->gni_nic_hndl == NULL) {
			status = GNI_CdmAttach(nic->gni_cdm_hndl,
						0,
						&device_addr,
						&nic->gni_nic_hndl);
			if (status != GNI_RC_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL, "GNI_CdmAttach returned %s\n",
					 gni_err_str[status]);
				ret = gnixu_to_fi_errno(status);
				goto err1;
			}
		} else
			nic->gni_nic_hndl = nic_attr->gni_nic_hndl;

		/*
		 * create TX CQs - first polling, then blocking
		 */

		status = GNI_CqCreate(nic->gni_nic_hndl,
					domain->gni_tx_cq_size,
					0,                  /* no delay count */
					GNI_CQ_NOBLOCK |
					domain->gni_cq_modes,
					NULL,              /* useless handler */
					NULL,               /* useless handler
								context */
					&nic->tx_cq);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqCreate returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err1;
		}

		status = GNI_CqCreate(nic->gni_nic_hndl,
					domain->gni_tx_cq_size,
					0,
					GNI_CQ_BLOCKING |
						domain->gni_cq_modes,
					NULL,
					NULL,
					&nic->tx_cq_blk);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqCreate returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err1;
		}

		/*
		 * create RX CQs - first polling, then blocking
		 */

		status = GNI_CqCreate(nic->gni_nic_hndl,
					domain->gni_rx_cq_size,
					0,
					GNI_CQ_NOBLOCK |
						domain->gni_cq_modes,
					NULL,
					NULL,
					&nic->rx_cq);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqCreate returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err1;
		}

		status = GNI_CqCreate(nic->gni_nic_hndl,
					domain->gni_rx_cq_size,
					0,
					GNI_CQ_BLOCKING |
					domain->gni_cq_modes,
					NULL,
					NULL,
					&nic->rx_cq_blk);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_CqCreate returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err1;
		}

		nic->device_addr = device_addr;
		nic->ptag = domain->ptag;
		nic->cookie = domain->cookie;

		nic->vc_id_table_capacity = domain->params.vc_id_table_capacity;
		nic->vc_id_table = malloc(sizeof(void *) *
					       nic->vc_id_table_capacity);
		if (nic->vc_id_table == NULL) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "malloc of vc_id_table failed\n");
			ret = -FI_ENOMEM;
			goto err1;
		}

		fastlock_init(&nic->lock);

		ret = __gnix_nic_tx_freelist_init(nic, domain->gni_tx_cq_size);
		if (ret != FI_SUCCESS)
			goto err1;

		fastlock_init(&nic->pending_vc_lock);
		dlist_init(&nic->pending_vcs);

		_gnix_ref_init(&nic->ref_cnt, 1, __nic_destruct);
		atomic_initialize(&nic->outstanding_fab_reqs_nic, 0);
		ret = _gnix_alloc_bitmap(&nic->vc_id_bitmap, 1000);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "alloc_bitmap returned %d\n", ret);
			goto err1;
		}
		fastlock_init(&nic->vc_id_lock);

		smsg_mbox_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
		smsg_mbox_attr.mbox_maxcredit = domain->params.mbox_maxcredit;
		smsg_mbox_attr.msg_maxsize =  domain->params.mbox_msg_maxsize;

		status = GNI_SmsgBufferSizeNeeded(&smsg_mbox_attr,
						  &nic->mem_per_mbox);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "GNI_SmsgBufferSizeNeeded returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			goto err1;
		}

		/*
		 * set up mailbox allocator for SMSG mailboxes
		 */

		ret = _gnix_mbox_allocator_create(nic,
					  nic->rx_cq,
					  domain->params.mbox_page_size,
					  (size_t)nic->mem_per_mbox,
					  domain->params.mbox_num_per_slab,
					  &nic->mbox_hndl);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_mbox_alloc returned %d\n", ret);
			goto err1;
		}

		/*
		 * use the mailbox allocator system to set up an
		 * pre-pinned RDMA bounce buffers for longer eager
		 * messages and other cases where zero-copy
		 * can't be safely used.
		 *
		 * One set of blocks is used for the send side.
		 * A second set of blocks is used for the receive
		 * side.  Both sets of blocks are registered against
		 * the blocking RX CQ for this nic.
		 *
		 * TODO: hardwired constants, uff
		 * TODO: better to use a buddy allocator or some other
		 * allocator
		 */

		ret = _gnix_mbox_allocator_create(nic,
						  nic->rx_cq_blk,
						  GNIX_PAGE_2MB,
						  65536,
						  512,
						  &nic->s_rdma_buf_hndl);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_mbox_alloc returned %d\n", ret);
			goto err1;
		}

		ret = _gnix_mbox_allocator_create(nic,
						  nic->rx_cq_blk,
						  GNIX_PAGE_2MB,
						  65536,
						  512,
						  &nic->r_rdma_buf_hndl);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_mbox_alloc returned %d\n", ret);
			goto err1;
		}

		dlist_insert_tail(&nic->gnix_nic_list, &gnix_nic_list);
		dlist_insert_tail(&nic->dom_nic_list, &domain->nic_list);

		++gnix_nics_per_ptag[domain->ptag];

		GNIX_INFO(FI_LOG_EP_CTRL, "Allocated NIC:%p\n", nic);
	}

	*nic_ptr = nic;
	goto out;

err1:
	atomic_dec(&gnix_id_counter);
err:
	if (nic != NULL) {
		if (nic->r_rdma_buf_hndl != NULL)
			_gnix_mbox_allocator_destroy(nic->r_rdma_buf_hndl);
		if (nic->s_rdma_buf_hndl != NULL)
			_gnix_mbox_allocator_destroy(nic->s_rdma_buf_hndl);
		if (nic->mbox_hndl != NULL)
			_gnix_mbox_allocator_destroy(nic->mbox_hndl);
		if (nic->rx_cq_blk != NULL)
			GNI_CqDestroy(nic->rx_cq_blk);
		if (nic->rx_cq != NULL)
			GNI_CqDestroy(nic->rx_cq);
		if (nic->tx_cq_blk != NULL)
			GNI_CqDestroy(nic->tx_cq_blk);
		if (nic->tx_cq != NULL)
			GNI_CqDestroy(nic->tx_cq);
		if ((nic->gni_cdm_hndl != NULL) && (nic->allocd_gni_res &
		    GNIX_NIC_CDM_ALLOCD))
			GNI_CdmDestroy(nic->gni_cdm_hndl);
		free(nic);
	}

out:
	pthread_mutex_unlock(&gnix_nic_list_lock);
	return ret;
}
