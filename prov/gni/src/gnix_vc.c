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

/*
 * code for managing VC's
 */

#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_vc.h"
#include "gnix_util.h"
#include "gnix_datagram.h"
#include "gnix_cm_nic.h"
#include "gnix_nic.h"
#include "gnix_ep.h"
#include "gnix_mbox_allocator.h"
#include "gnix_hashtable.h"
#include "gnix_av.h"

/*
 * forward declarations and local struct defs.
 */

struct wq_hndl_conn_req {
	gni_smsg_attr_t src_smsg_attr;
	int src_vc_id;
	struct gnix_vc *vc;
	uint64_t src_vc_ptr;
};

static int __gnix_vc_conn_ack_prog_fn(void *data, int *complete_ptr);
static int __gnix_vc_conn_ack_comp_fn(void *data);

/*******************************************************************************
 * Helper functions
 ******************************************************************************/

/*******************************************************************************
 * connection request /response message pack/unpack functions
 ******************************************************************************/

/*
 * pack a connection request. Contents:
 * - target_addr (the addr of the targeted EP for the conn req)
 * - src_addr (the address of the EP originating the conn req)
 * - src_vc_id (the vc id the mbox the originating EP allocated to
 *              build this connection)
 * - src_vc_vaddr (virt. address of the vc struct allocated at the originating
 *                 EP to build this connection)
 * - src_smsg_attr (smsg attributes of the mbox allocated at the
 *                  originating EP for this connection)
 */
static void __gnix_vc_pack_conn_req(char *sbuf,
				    struct gnix_address *target_addr,
				    struct gnix_address *src_addr,
				    int src_vc_id,
				    uint64_t src_vc_vaddr,
				    gni_smsg_attr_t *src_smsg_attr)
{
	size_t __attribute__((unused)) len;
	char *cptr = sbuf;
	uint8_t rtype = GNIX_VC_CONN_REQ;

	/*
	 * sanity checks
	 */

	assert(sbuf != NULL);

	len = sizeof(uint8_t) + sizeof(struct gnix_address) * 2 + sizeof(int)
		+ sizeof(gni_smsg_attr_t);
	assert(len <= GNIX_CM_NIC_MAX_MSG_SIZE);

	memcpy(cptr, &rtype, sizeof(rtype));
	cptr += sizeof(rtype);
	memcpy(cptr, target_addr, sizeof(struct gnix_address));
	cptr += sizeof(struct gnix_address);
	memcpy(cptr, src_addr, sizeof(struct gnix_address));
	cptr += sizeof(struct gnix_address);
	memcpy(cptr, &src_vc_id, sizeof(int));
	cptr += sizeof(int);
	memcpy(cptr, &src_vc_vaddr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(cptr, src_smsg_attr, sizeof(gni_smsg_attr_t));
}

/*
 * unpack a connection request message
 */
static void __gnix_vc_unpack_conn_req(char *rbuf,
				      struct gnix_address *target_addr,
				      struct gnix_address *src_addr,
				      int *src_vc_id,
				      uint64_t *src_vc_vaddr,
				      gni_smsg_attr_t *src_smsg_attr)
{
	size_t __attribute__((unused)) len;
	uint8_t rtype;
	char *cptr = rbuf;

	/*
	 * sanity checks
	 */

	assert(rbuf);

	len = sizeof(rtype) + sizeof(struct gnix_address) * 2 + sizeof(int)
		+ sizeof(gni_smsg_attr_t);
	assert(len <= GNIX_CM_NIC_MAX_MSG_SIZE);

	cptr += sizeof(uint8_t);
	memcpy(target_addr, cptr, sizeof(struct gnix_address));
	cptr += sizeof(struct gnix_address);
	memcpy(src_addr, cptr, sizeof(struct gnix_address));
	cptr += sizeof(struct gnix_address);
	memcpy(src_vc_id, cptr, sizeof(int));
	cptr += sizeof(int);
	memcpy(src_vc_vaddr, cptr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(src_smsg_attr, cptr, sizeof(gni_smsg_attr_t));
}

/*
 * pack a connection response. Contents:
 * - src_vc_vaddr (vaddr of the vc struct allocated at the originating
 *                EP to build this connection)
 * - resp_vc_id (the vc id of the mbox the responding EP allocated to
 *          build this connection)
 * - resp_smsg_attr (smsg attributes of the mbox allocated at the
 *                   responding EP for this connection)
 */

static void __gnix_vc_pack_conn_resp(char *sbuf,
				     uint64_t src_vc_vaddr,
				     uint64_t resp_vc_vaddr,
				     int resp_vc_id,
				     gni_smsg_attr_t *resp_smsg_attr)
{
	char *cptr = sbuf;
	uint8_t rtype = GNIX_VC_CONN_RESP;

	/*
	 * sanity checks
	 */

	assert(sbuf != NULL);

	memcpy(cptr, &rtype, sizeof(rtype));
	cptr += sizeof(rtype);
	memcpy(cptr, &src_vc_vaddr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(cptr, &resp_vc_vaddr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(cptr, &resp_vc_id, sizeof(int));
	cptr += sizeof(int);
	memcpy(cptr, resp_smsg_attr, sizeof(gni_smsg_attr_t));
}

/*
 * unpack a connection request response
 */
static void __gnix_vc_unpack_resp(char *rbuf,
				  uint64_t *src_vc_vaddr,
				  uint64_t *resp_vc_vaddr,
				  int *resp_vc_id,
				  gni_smsg_attr_t *resp_smsg_attr)
{
	char *cptr = rbuf;

	cptr += sizeof(uint8_t);

	memcpy(src_vc_vaddr, cptr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(resp_vc_vaddr, cptr, sizeof(uint64_t));
	cptr += sizeof(uint64_t);
	memcpy(resp_vc_id, cptr, sizeof(int));
	cptr += sizeof(int);
	memcpy(resp_smsg_attr, cptr, sizeof(gni_smsg_attr_t));
}

static void __gnix_vc_get_msg_type(char *rbuf,
				  uint8_t *rtype)
{
	assert(rtype);
	memcpy(rtype, rbuf, sizeof(uint8_t));
}

/*
 * helper function to initialize an SMSG connection
 */
static int __gnix_vc_smsg_init(struct gnix_vc *vc,
				int peer_id,
				gni_smsg_attr_t *peer_smsg_attr)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep;
	struct gnix_fid_domain *dom;
	struct gnix_mbox *mbox = NULL;
	gni_smsg_attr_t local_smsg_attr;
	gni_return_t __attribute__((unused)) status;
	ssize_t __attribute__((unused)) len;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	assert(vc);

	ep = vc->ep;
	assert(ep);

	dom = ep->domain;
	if (dom == NULL)
		return -FI_EINVAL;

	mbox = vc->smsg_mbox;
	assert (mbox);

	local_smsg_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
	local_smsg_attr.msg_buffer = mbox->base;
	local_smsg_attr.buff_size =  vc->ep->nic->mem_per_mbox;
	local_smsg_attr.mem_hndl = *mbox->memory_handle;
	local_smsg_attr.mbox_offset = (uint64_t)mbox->offset;
	local_smsg_attr.mbox_maxcredit = dom->params.mbox_maxcredit;
	local_smsg_attr.msg_maxsize = dom->params.mbox_msg_maxsize;

	/*
	 *  now build the SMSG connection
	 */

	fastlock_acquire(&ep->nic->lock);

	status = GNI_EpCreate(ep->nic->gni_nic_hndl,
			      ep->nic->tx_cq,
			      &vc->gni_ep);
	if (status != GNI_RC_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			"GNI_EpCreate returned %s\n", gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
		goto err;
	}

	status = GNI_EpBind(vc->gni_ep,
			    vc->peer_addr.device_addr,
			    vc->peer_addr.cdm_id);
	if (status != GNI_RC_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "GNI_EpBind returned %s\n", gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
		goto err1;
	}

	status = GNI_SmsgInit(vc->gni_ep,
			      &local_smsg_attr,
			      peer_smsg_attr);
	if (status != GNI_RC_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			"GNI_SmsgInit returned %s\n", gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
		goto err1;
	}

	status = GNI_EpSetEventData(vc->gni_ep,
				    vc->vc_id,
				    peer_id);
	if (status != GNI_RC_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "GNI_EpSetEventData returned %s\n",
			   gni_err_str[status]);
		ret = gnixu_to_fi_errno(status);
		goto err1;
	}

	fastlock_release(&ep->nic->lock);
	return ret;
err1:
	GNI_EpDestroy(vc->gni_ep);
err:
	fastlock_release(&ep->nic->lock);
	return ret;
}

/*
 * connect to self, since we use a lock here
 * the only case we need to deal with is one
 * vc connect request with the other not yet initiated
 */
static int __gnix_vc_connect_to_self(struct gnix_vc *vc)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_domain *dom = NULL;
	struct gnix_fid_ep *ep = NULL;
	struct gnix_fid_ep *ep_peer = NULL;
	struct gnix_cm_nic *cm_nic = NULL;
	struct gnix_mbox *mbox = NULL, *mbox_peer = NULL;
	struct gnix_vc *vc_peer;
	gni_smsg_attr_t smsg_mbox_attr;
	gni_smsg_attr_t smsg_mbox_attr_peer;
	gnix_ht_key_t *key_ptr;
	struct gnix_av_addr_entry entry;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if ((vc->conn_state == GNIX_VC_CONNECTING) ||
	    (vc->conn_state == GNIX_VC_CONNECTED)) {
		return FI_SUCCESS;
	}

	ep = vc->ep;
	if (ep == NULL)
		return -FI_EINVAL;

	fastlock_acquire(&ep->vc_ht_lock);

	if ((vc->conn_state == GNIX_VC_CONNECTING) ||
	    (vc->conn_state == GNIX_VC_CONNECTED)) {
		fastlock_release(&ep->vc_ht_lock);
		return FI_SUCCESS;
	}

	cm_nic = ep->cm_nic;
	if (cm_nic == NULL) {
		ret = -FI_EINVAL;
		goto err;
	}

	dom = ep->domain;
	if (dom == NULL) {
		ret = -FI_EINVAL;
		goto err;
	}

	vc->conn_state = GNIX_VC_CONNECTING;
	GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connecting\n", vc);

	if (vc->smsg_mbox == NULL) {
		ret = _gnix_mbox_alloc(vc->ep->nic->mbox_hndl,
				       &mbox);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_mbox_alloc returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}
		vc->smsg_mbox = mbox;
	} else
		mbox = vc->smsg_mbox;

	smsg_mbox_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
	smsg_mbox_attr.msg_buffer = mbox->base;
	smsg_mbox_attr.buff_size =  vc->ep->nic->mem_per_mbox;
	smsg_mbox_attr.mem_hndl = *mbox->memory_handle;
	smsg_mbox_attr.mbox_offset = (uint64_t)mbox->offset;
	smsg_mbox_attr.mbox_maxcredit = dom->params.mbox_maxcredit;
	smsg_mbox_attr.msg_maxsize = dom->params.mbox_msg_maxsize;

	key_ptr = (gnix_ht_key_t *)&vc->peer_addr;
	ep_peer = (struct gnix_fid_ep *)_gnix_ht_lookup(cm_nic->addr_to_ep_ht,
						   *key_ptr);
	if (ep_peer == NULL) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "_gnix_ht_lookup addr_to_ep failed\n");
		ret = -FI_ENOENT;
		goto err;
	}

	key_ptr = (gnix_ht_key_t *)&ep->my_name.gnix_addr;

	vc_peer = (struct gnix_vc *)_gnix_ht_lookup(ep_peer->vc_ht,
						   *key_ptr);
	if ((vc_peer != NULL) &&
	    (vc_peer->conn_state != GNIX_VC_CONN_NONE)) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "_gnix_vc_connect self, vc_peer in inconsistent state\n");
		ret = -FI_ENOSPC;
		goto err;
	}

	if (vc_peer == NULL) {
		entry.gnix_addr = ep->my_name.gnix_addr;
		entry.cm_nic_cdm_id = ep->my_name.cm_nic_cdm_id;
		ret = _gnix_vc_alloc(ep_peer,
				     &entry,
				     &vc_peer);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				"_gnix_vc_alloc returned %s\n",
				fi_strerror(-ret));
			goto err;
		}

		ret = _gnix_ht_insert(ep_peer->vc_ht,
				      *key_ptr,
				      vc_peer);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_ht_insert returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}
		vc_peer->modes |= GNIX_VC_MODE_IN_HT;
	}

	vc_peer->conn_state = GNIX_VC_CONNECTING;
	GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connecting\n",
		   vc_peer);

	if (vc_peer->smsg_mbox == NULL) {
		ret = _gnix_mbox_alloc(vc_peer->ep->nic->mbox_hndl,
				       &mbox_peer);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_mbox_alloc returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}
		vc_peer->smsg_mbox = mbox_peer;
	} else 
		mbox_peer = vc_peer->smsg_mbox;

	smsg_mbox_attr_peer.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
	smsg_mbox_attr_peer.msg_buffer = mbox_peer->base;
	smsg_mbox_attr_peer.buff_size =  vc_peer->ep->nic->mem_per_mbox;
	smsg_mbox_attr_peer.mem_hndl = *mbox_peer->memory_handle;
	smsg_mbox_attr_peer.mbox_offset = (uint64_t)mbox_peer->offset;
	smsg_mbox_attr_peer.mbox_maxcredit = dom->params.mbox_maxcredit;
	smsg_mbox_attr_peer.msg_maxsize = dom->params.mbox_msg_maxsize;

	ret = __gnix_vc_smsg_init(vc, vc_peer->vc_id, &smsg_mbox_attr_peer);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "_gnix_vc_smsg_init returned %s\n",
			  fi_strerror(-ret));
		goto err;
	}

	ret = __gnix_vc_smsg_init(vc_peer, vc->vc_id, &smsg_mbox_attr);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "_gnix_vc_smsg_init returned %s\n",
			  fi_strerror(-ret));
		goto err;
	}

	vc->conn_state = GNIX_VC_CONNECTED;
	GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connected\n",
		   vc);
	vc_peer->conn_state = GNIX_VC_CONNECTED;
	GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connected\n",
		   vc_peer);

err:
	fastlock_release(&ep->vc_ht_lock);
	return ret;
}

/*******************************************************************************
 * functions for handling incoming connection request/response messages
 ******************************************************************************/

static int __gnix_vc_hndl_conn_resp(struct gnix_cm_nic *cm_nic,
				    char *msg_buffer,
				    struct gnix_address src_cm_nic_addr)
{
	int ret = FI_SUCCESS;
	int peer_id;
	struct gnix_vc *vc = NULL;
	uint64_t peer_vc_addr;
	struct gnix_fid_ep *ep;
	gni_smsg_attr_t peer_smsg_attr;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	/*
	 * unpack the message
	 */

	__gnix_vc_unpack_resp(msg_buffer,
			      (uint64_t *)&vc,
			      &peer_vc_addr,
			      &peer_id,
			      &peer_smsg_attr);

	GNIX_DEBUG(FI_LOG_EP_CTRL,
		"resp rx: (From Aries 0x%x Id %d src vc %p peer vc addr 0x%lx)\n",
		 src_cm_nic_addr.device_addr,
		 src_cm_nic_addr.cdm_id,
		 vc,
		 peer_vc_addr);

	ep = vc->ep;
	assert(ep != NULL);

	fastlock_acquire(&ep->vc_ht_lock);

	/*
	 * at this point vc should be in connecting state
	 */
	if (vc->conn_state != GNIX_VC_CONNECTING) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			  "vc %p not in connecting state, rather %d\n",
			  vc, vc->conn_state);
		ret = -FI_EINVAL;
		goto err;
	}

	/*
	 * build the SMSG connection
	 */

	ret = __gnix_vc_smsg_init(vc, peer_id, &peer_smsg_attr);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			"__gnix_vc_smsg_init returned %s\n",
			fi_strerror(-ret));
		goto err;
	}

	/*
	 * transition the VC to connected
	 * put in to the nic's work queue for
	 * further processing
	 */

	vc->conn_state = GNIX_VC_CONNECTED;
	GNIX_DEBUG(FI_LOG_EP_CTRL,
		   " moving vc %p to state connected\n",vc);

	fastlock_release(&ep->vc_ht_lock);

	ret = _gnix_vc_schedule(vc);
	if (ret == FI_SUCCESS)
		ret = _gnix_nic_progress(ep->nic);
	else
		GNIX_WARN(FI_LOG_EP_CTRL,
			"_gnix_vc_schedule returned %s\n",
			fi_strerror(-ret));

	return ret;
err:
	vc->conn_state = GNIX_VC_CONN_ERROR;
	fastlock_release(&ep->vc_ht_lock);
	return ret;
}

static int __gnix_vc_hndl_conn_req(struct gnix_cm_nic *cm_nic,
				   char *msg_buffer,
				   struct gnix_address src_cm_nic_addr)
{
	int ret = FI_SUCCESS;
	gni_return_t __attribute__((unused)) status;
	struct gnix_fid_ep *ep = NULL;
	gnix_ht_key_t *key_ptr;
	struct gnix_av_addr_entry entry;
	struct gnix_address src_addr, target_addr;
	struct gnix_vc *vc = NULL;
	struct gnix_vc *vc_try = NULL;
	struct gnix_work_req *work_req;
	int src_vc_id;
	gni_smsg_attr_t src_smsg_attr;
	uint64_t src_vc_ptr;
	struct wq_hndl_conn_req *data = NULL;

	ssize_t __attribute__((unused)) len;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	/*
	 * unpack the message
	 */

	__gnix_vc_unpack_conn_req(msg_buffer,
				  &target_addr,
				  &src_addr,
				  &src_vc_id,
				  &src_vc_ptr,
				  &src_smsg_attr);


	GNIX_DEBUG(FI_LOG_EP_CTRL,
		"conn req rx: (From Aries addr 0x%x Id %d to Aries 0x%x Id %d src vc 0x%lx )\n",
		 src_addr.device_addr,
		 src_addr.cdm_id,
		 target_addr.device_addr,
		 target_addr.cdm_id,
		 src_vc_ptr);

	/*
	 * lookup the ep from the addr_to_ep_ht using the target_addr
	 * in the datagram
	 */

	key_ptr = (gnix_ht_key_t *)&target_addr;

	ep = (struct gnix_fid_ep *)_gnix_ht_lookup(cm_nic->addr_to_ep_ht,
						   *key_ptr);
	if (ep == NULL) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "_gnix_ht_lookup addr_to_ep failed\n");
		ret = -FI_ENOENT;
		goto err;
	}

	/*
	 * look to see if there is a VC already for the
	 * address of the connecting EP.
	 */

	key_ptr = (gnix_ht_key_t *)&src_addr;

	fastlock_acquire(&ep->vc_ht_lock);
	vc = (struct gnix_vc *)_gnix_ht_lookup(ep->vc_ht,
					       *key_ptr);

	/*
 	 * if there is no corresponding vc in the hash,
 	 * or there is an entry and its not in connecting state
 	 * go down the conn req ack route.
 	 */

	if ((vc == NULL)  ||
	    (vc->conn_state == GNIX_VC_CONN_NONE)) {
		if (vc == NULL) {
			entry.gnix_addr = src_addr;
			entry.cm_nic_cdm_id = src_cm_nic_addr.cdm_id;
			ret = _gnix_vc_alloc(ep,
					     &entry,
					     &vc_try);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_CTRL,
					  "_gnix_vc_alloc returned %s\n",
					  fi_strerror(-ret));
				goto err;
			}

			vc_try->conn_state = GNIX_VC_CONNECTING;
			ret = _gnix_ht_insert(ep->vc_ht,
					      *key_ptr,
					      vc_try);
			if (likely(ret == FI_SUCCESS)) {
				vc = vc_try;
				vc->modes |= GNIX_VC_MODE_IN_HT;
			} else if (ret == -FI_ENOSPC) {
				_gnix_vc_destroy(vc_try);
			} else {
				GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_ht_insert returned %s\n",
				   fi_strerror(-ret));
				goto err;
			}
		} else
			vc->conn_state = GNIX_VC_CONNECTING;

		/*
		 * prepare a work request to
		 * initiate an request response
		 */

		work_req = calloc(1, sizeof(*work_req));
		if (work_req == NULL) {
			ret = -FI_ENOMEM;
			goto err;
		}

		data = calloc(1, sizeof(struct wq_hndl_conn_req));
		if (data == NULL) {
			ret = -FI_ENOMEM;
			goto err;
		}
		memcpy(&data->src_smsg_attr,
		       &src_smsg_attr,
		       sizeof(src_smsg_attr));
		data->vc = vc;
		data->src_vc_id = src_vc_id;
		data->src_vc_ptr = src_vc_ptr;

		work_req->progress_fn = __gnix_vc_conn_ack_prog_fn;
		work_req->data = data;
		work_req->completer_fn = __gnix_vc_conn_ack_comp_fn;
		work_req->completer_data = data;

		/*
		 * add the work request to the tail of the
		 * cm_nic's work queue, progress the cm_nic.
		 */


		fastlock_acquire(&cm_nic->wq_lock);
		dlist_insert_before(&work_req->list, &cm_nic->cm_nic_wq);
		fastlock_release(&cm_nic->wq_lock);

		fastlock_release(&ep->vc_ht_lock);

		ret = _gnix_vc_schedule(vc);
		ret = _gnix_cm_nic_progress(cm_nic);

	} else {

		/*
		 * we can only be in connecting state if we
		 * reach here.  We have all the informatinon,
		 * and the other side will get the information
		 * at some point, so go ahead and build SMSG connection.
		 */
		if (vc->conn_state != GNIX_VC_CONNECTING) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				 "vc %p not in connecting state nor in cm wq\n",
				  vc, vc->conn_state);
			ret = -FI_EINVAL;
			goto err;
		}

		ret = __gnix_vc_smsg_init(vc, src_vc_id,
					  &src_smsg_attr);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_vc_smsg_init returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}

		vc->conn_state = GNIX_VC_CONNECTED;
		GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connected\n",
			vc);

		fastlock_release(&ep->vc_ht_lock);

		ret = _gnix_vc_schedule(vc);
		if (ret == FI_SUCCESS)
			ret = _gnix_nic_progress(ep->nic);
		else
			GNIX_WARN(FI_LOG_EP_CTRL,
				"_gnix_vc_schedule returned %s\n",
				fi_strerror(-ret));
	}
err:
	return ret;
}

/*
 * callback function to process incoming messages
 */
int __gnix_vc_recv_fn(struct gnix_cm_nic *cm_nic,
		      char *msg_buffer,
		      struct gnix_address src_cm_nic_addr)
{
	int ret = FI_SUCCESS;
	uint8_t mtype;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	__gnix_vc_get_msg_type(msg_buffer, &mtype);

	GNIX_DEBUG(FI_LOG_EP_CTRL, "got a message of type %d\n", mtype);

	switch (mtype) {
	case GNIX_VC_CONN_REQ:
		ret = __gnix_vc_hndl_conn_req(cm_nic,
					      msg_buffer,
					      src_cm_nic_addr);
		break;
	case GNIX_VC_CONN_RESP:
		ret = __gnix_vc_hndl_conn_resp(cm_nic,
					       msg_buffer,
					       src_cm_nic_addr);
		break;
	default:
		GNIX_WARN(FI_LOG_EP_CTRL,
			"unknown cm_nic message type %d\n", mtype);
		assert(0);
	}

	return ret;
}

/*
 * progress function for progressing a connection
 * ACK.
 */

static int __gnix_vc_conn_ack_prog_fn(void *data, int *complete_ptr)
{
	int ret = FI_SUCCESS;
	int complete = 0;
	struct wq_hndl_conn_req *work_req_data;
	struct gnix_vc *vc;
	struct gnix_mbox *mbox = NULL;
	gni_smsg_attr_t smsg_mbox_attr;
	struct gnix_fid_ep *ep = NULL;
	struct gnix_fid_domain *dom = NULL;
	struct gnix_cm_nic *cm_nic = NULL;
	char sbuf[GNIX_CM_NIC_MAX_MSG_SIZE] = {0};

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");


	work_req_data = (struct wq_hndl_conn_req *)data;

	vc = work_req_data->vc;
	if (vc == NULL)
		return -FI_EINVAL;

	ep = vc->ep;
	if (ep == NULL)
		return -FI_EINVAL;

	dom = ep->domain;
	if (dom == NULL)
		return -FI_EINVAL;

	cm_nic = ep->cm_nic;
	if (cm_nic == NULL)
		return -FI_EINVAL;

	fastlock_acquire(&ep->vc_ht_lock);

	/*
	 * we may have already been moved to connecting or
	 * connected, if so early exit.
	 */
	if(vc->conn_state == GNIX_VC_CONNECTED) {
		complete = 1;
		goto exit;
	}

	/*
	 * first see if we still need a mailbox
	 */

	if (vc->smsg_mbox == NULL) {
		ret = _gnix_mbox_alloc(ep->nic->mbox_hndl,
				       &mbox);
		if (ret == FI_SUCCESS)
			vc->smsg_mbox = mbox;
		else
			goto exit;
	}

	mbox = vc->smsg_mbox;

	/*
	 * prep the smsg_mbox_attr
¬        */

	smsg_mbox_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
	smsg_mbox_attr.msg_buffer = mbox->base;
	smsg_mbox_attr.buff_size =  ep->nic->mem_per_mbox;
	smsg_mbox_attr.mem_hndl = *mbox->memory_handle;
	smsg_mbox_attr.mbox_offset = (uint64_t)mbox->offset;
	smsg_mbox_attr.mbox_maxcredit = dom->params.mbox_maxcredit;
	smsg_mbox_attr.msg_maxsize = dom->params.mbox_msg_maxsize;

	/*
	 * serialize the resp message in the buffer
	 */

	__gnix_vc_pack_conn_resp(sbuf,
				 work_req_data->src_vc_ptr,
				 (uint64_t)vc,
				 vc->vc_id,
				 &smsg_mbox_attr);

	/*
	 * try to send the message, if it succeeds,
	 * initialize mailbox and move vc to connected
	 * state.
	 */

	ret = _gnix_cm_nic_send(cm_nic,
				sbuf,
				GNIX_CM_NIC_MAX_MSG_SIZE,
				vc->peer_cm_nic_addr);
	if (ret == FI_SUCCESS) {
		ret = __gnix_vc_smsg_init(vc,
					  work_req_data->src_vc_id,
					  &work_req_data->src_smsg_attr);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_vc_smsg_init returned %s\n",
				  fi_strerror(-ret));
			goto exit;
		}
		complete = 1;
		vc->conn_state = GNIX_VC_CONNECTED;
		GNIX_DEBUG(FI_LOG_EP_CTRL,
			   "moving vc %p to connected\n",vc);
	} else if (ret == -FI_EAGAIN) {
		ret = _gnix_vc_schedule(vc);
		ret = FI_SUCCESS;
	} else
		assert(0);

exit:
	fastlock_release(&ep->vc_ht_lock);

	*complete_ptr = complete;
	return ret;
}

static int __gnix_vc_conn_req_prog_fn(void *data, int *complete_ptr)
{
	int ret = FI_SUCCESS;
	int complete = 0;
	struct gnix_vc *vc = (struct gnix_vc *)data;
	struct gnix_mbox *mbox = NULL;
	gni_smsg_attr_t smsg_mbox_attr;
	struct gnix_fid_ep *ep = NULL;
	struct gnix_fid_domain *dom = NULL;
	struct gnix_cm_nic *cm_nic = NULL;
	char sbuf[GNIX_CM_NIC_MAX_MSG_SIZE] = {0};

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	ep = vc->ep;
	if (ep == NULL)
		return -FI_EINVAL;

	dom = ep->domain;
	if (dom == NULL)
		return -FI_EINVAL;

	cm_nic = ep->cm_nic;
	if (cm_nic == NULL)
		return -FI_EINVAL;

	fastlock_acquire(&ep->vc_ht_lock);

	if ((vc->conn_state == GNIX_VC_CONNECTING) ||
		(vc->conn_state == GNIX_VC_CONNECTED)) {
			complete = 1;
			goto err;
	}

	/*
	 * sanity check that the vc is in the hash table
	 */

	if (!(vc->modes & GNIX_VC_MODE_IN_HT)) {
		ret = -FI_EINVAL;
		goto err;
	}

	/*
	 * first see if we still need a mailbox
	 */

	if (vc->smsg_mbox == NULL) {
		ret = _gnix_mbox_alloc(vc->ep->nic->mbox_hndl,
				       &mbox);
		if (ret == FI_SUCCESS)
			vc->smsg_mbox = mbox;
		else
			goto err;
	}

	mbox = vc->smsg_mbox;

	/*
	 * prep the smsg_mbox_attr
¬        */

	smsg_mbox_attr.msg_type = GNI_SMSG_TYPE_MBOX_AUTO_RETRANSMIT;
	smsg_mbox_attr.msg_buffer = mbox->base;
	smsg_mbox_attr.buff_size =  vc->ep->nic->mem_per_mbox;
	smsg_mbox_attr.mem_hndl = *mbox->memory_handle;
	smsg_mbox_attr.mbox_offset = (uint64_t)mbox->offset;
	smsg_mbox_attr.mbox_maxcredit = dom->params.mbox_maxcredit;
	smsg_mbox_attr.msg_maxsize = dom->params.mbox_msg_maxsize;

	/*
	 * serialize the message in the buffer
	 */

	GNIX_DEBUG(FI_LOG_EP_CTRL,
		"conn req tx: (From Aries addr 0x%x Id %d to Aries 0x%x Id %d CM NIC Id %d vc %p)\n",
		 ep->my_name.gnix_addr.device_addr,
		 ep->my_name.gnix_addr.cdm_id,
		 vc->peer_addr.device_addr,
		 vc->peer_addr.cdm_id,
		 vc->peer_cm_nic_addr.cdm_id,
		 vc);

	__gnix_vc_pack_conn_req(sbuf,
				&vc->peer_addr,
				&ep->my_name.gnix_addr,
				vc->vc_id,
				(uint64_t)vc,
				&smsg_mbox_attr);

	/*
	 * try to send the message, if -FI_EAGAIN is returned, okay,
	 * just don't mark complete.
	 */

	ret = _gnix_cm_nic_send(cm_nic,
				sbuf,
				GNIX_CM_NIC_MAX_MSG_SIZE,
				vc->peer_cm_nic_addr);
	if (ret == FI_SUCCESS) {
		complete = 1;
		vc->conn_state = GNIX_VC_CONNECTING;
		GNIX_DEBUG(FI_LOG_EP_CTRL, "moving vc %p state to connecting\n",
			vc);
	} else if (ret == -FI_EAGAIN) {
		ret = _gnix_vc_schedule(vc);
		ret = FI_SUCCESS;
	}

err:
	fastlock_release(&ep->vc_ht_lock);
	*complete_ptr = complete;
	return ret;
}

/*
 * conn ack completer function for work queue element,
 * free the previously allocated wq_hndl_conn_req
 * data struct
 */
static int __gnix_vc_conn_ack_comp_fn(void *data)
{
	free(data);
	return FI_SUCCESS;
}

/*
 * connect completer function for work queue element,
 * sort of a NO-OP for now.
 */
static int __gnix_vc_conn_req_comp_fn(void *data)
{
	return FI_SUCCESS;
}

/*******************************************************************************
 * Internal API functions
 ******************************************************************************/

int _gnix_vc_alloc(struct gnix_fid_ep *ep_priv,
		   struct gnix_av_addr_entry *entry, struct gnix_vc **vc)

{
	int ret = FI_SUCCESS;
	int remote_id;
	struct gnix_vc *vc_ptr = NULL;
	struct gnix_cm_nic *cm_nic = NULL;
	struct gnix_nic *nic = NULL;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	nic = ep_priv->nic;
	if (nic == NULL)
		return -FI_EINVAL;

	cm_nic = ep_priv->cm_nic;
	if (cm_nic == NULL)
		return -FI_EINVAL;

	vc_ptr = calloc(1, sizeof(*vc_ptr));
	if (!vc_ptr)
		return -FI_ENOMEM;

	vc_ptr->conn_state = GNIX_VC_CONN_NONE;
	if (entry) {
		memcpy(&vc_ptr->peer_addr,
			&entry->gnix_addr,
			sizeof(struct gnix_address));
		vc_ptr->peer_cm_nic_addr.device_addr =
			entry->gnix_addr.device_addr;
		vc_ptr->peer_cm_nic_addr.cdm_id =
			entry->cm_nic_cdm_id;
	} else {
		vc_ptr->peer_addr.device_addr = -1;
		vc_ptr->peer_addr.cdm_id = -1;
		vc_ptr->peer_cm_nic_addr.device_addr = -1;
		vc_ptr->peer_cm_nic_addr.cdm_id = -1;
	}
	vc_ptr->ep = ep_priv;

	slist_init(&vc_ptr->work_queue);
	fastlock_init(&vc_ptr->work_queue_lock);
	slist_init(&vc_ptr->tx_queue);
	fastlock_init(&vc_ptr->tx_queue_lock);
	dlist_init(&vc_ptr->rx_list);
	dlist_init(&vc_ptr->work_list);
	dlist_init(&vc_ptr->tx_list);
	vc_ptr->peer_fi_addr = FI_ADDR_NOTAVAIL;

	atomic_initialize(&vc_ptr->outstanding_tx_reqs, 0);
	ret = _gnix_alloc_bitmap(&vc_ptr->flags, 1);
	assert(!ret);

	/*
	 * we need an id for the vc to allow for quick lookup
	 * based on GNI_CQ_GET_INST_ID
	 */

	ret = _gnix_nic_get_rem_id(nic, &remote_id, vc_ptr);
	if (ret != FI_SUCCESS)
		goto err;
	vc_ptr->vc_id = remote_id;

	*vc = vc_ptr;

	return ret;

err:
	if (vc_ptr)
		free(vc_ptr);
	return ret;
}

static void __gnix_vc_cancel(struct gnix_vc *vc)
{
	struct gnix_nic *nic = vc->ep->nic;

	fastlock_acquire(&nic->rx_vc_lock);
	if (!dlist_empty(&vc->rx_list))
		dlist_remove(&vc->rx_list);
	fastlock_release(&nic->rx_vc_lock);

	fastlock_acquire(&nic->work_vc_lock);
	if (!dlist_empty(&vc->work_list))
		dlist_remove(&vc->work_list);
	fastlock_release(&nic->work_vc_lock);

	fastlock_acquire(&nic->tx_vc_lock);
	if (!dlist_empty(&vc->tx_list))
		dlist_remove(&vc->tx_list);
	fastlock_release(&nic->tx_vc_lock);
}

/* Destroy an unconnected VC.  More Support is needed to shutdown and destroy
 * an active VC. */
int _gnix_vc_destroy(struct gnix_vc *vc)
{
	int ret = FI_SUCCESS;
	struct gnix_nic *nic = NULL;
	gni_return_t status;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if (vc->ep == NULL) {
		GNIX_WARN(FI_LOG_EP_CTRL, "ep null\n");
		return -FI_EINVAL;
	}

	nic = vc->ep->nic;
	if (nic == NULL) {
		GNIX_WARN(FI_LOG_EP_CTRL, "ep nic null for vc %p\n", vc);
		return -FI_EINVAL;
	}

	/*
	 * if the vc is in a nic's work queue, remove it
	 */
	__gnix_vc_cancel(vc);

	/*
	 * We may eventually want to check the state of the VC, if we
	 * implement true VC shutdown.

	if ((vc->conn_state != GNIX_VC_CONN_NONE)
		&& (vc->conn_state != GNIX_VC_CONN_TERMINATED)) {
		GNIX_WARN(FI_LOG_EP_CTRL,
			      "vc conn state  %d\n",
			       vc->conn_state);
		GNIX_WARN(FI_LOG_EP_CTRL, "vc conn state error\n");
		return -FI_EBUSY;
	}
	 */

	/*
	 * if send_q not empty, return -FI_EBUSY
	 * Note for FI_EP_MSG type eps, this behavior
	 * may not be correct for handling fi_shutdown.
	 */

	if (!slist_empty(&vc->tx_queue)) {
		GNIX_WARN(FI_LOG_EP_CTRL, "vc sendqueue not empty\n");
		return -FI_EBUSY;
	}

	fastlock_destroy(&vc->tx_queue_lock);

	if (vc->gni_ep != NULL) {
		fastlock_acquire(&nic->lock);
		status = GNI_EpDestroy(vc->gni_ep);
		if (status != GNI_RC_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_CTRL, "GNI_EpDestroy returned %s\n",
				  gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			fastlock_release(&nic->lock);
			return ret;
		}
		fastlock_release(&nic->lock);
	}

	if (vc->smsg_mbox != NULL) {
		ret = _gnix_mbox_free(vc->smsg_mbox);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_CTRL,
			      "_gnix_mbox_free returned %s\n",
			      fi_strerror(-ret));
		vc->smsg_mbox = NULL;
	}

	if (vc->dgram != NULL) {
		ret = _gnix_dgram_free(vc->dgram);
		if (ret != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_CTRL,
			      "_gnix_dgram_free returned %s\n",
			      fi_strerror(-ret));
		vc->dgram = NULL;
	}

	ret = _gnix_nic_free_rem_id(nic, vc->vc_id);
	if (ret != FI_SUCCESS)
		GNIX_WARN(FI_LOG_EP_CTRL,
		      "__gnix_vc_free_id returned %s\n",
		      fi_strerror(-ret));

	_gnix_free_bitmap(&vc->flags);

	free(vc);

	return ret;
}

int _gnix_vc_connect(struct gnix_vc *vc)
{
	int ret = FI_SUCCESS;
	struct gnix_fid_ep *ep = NULL;
	struct gnix_cm_nic *cm_nic = NULL;
	struct gnix_work_req *work_req;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	/*
	 * can happen that we are already connecting, or
	 * are connected
	 */

	if ((vc->conn_state == GNIX_VC_CONNECTING) ||
		(vc->conn_state == GNIX_VC_CONNECTED)) {
		return FI_SUCCESS;
	}

	ep = vc->ep;
	if (ep == NULL)
		return -FI_EINVAL;

	cm_nic = ep->cm_nic;
	if (cm_nic == NULL)
		return -FI_EINVAL;

	/*
	 * only endpoints of type FI_EP_RDM use this
	 * connection method
	 */
	if (ep->type != FI_EP_RDM)
		return -FI_EINVAL;

	/*
	 * have to do something special for
	 * connect to self
	 */

	if (!memcmp(&vc->peer_cm_nic_addr,
		   &cm_nic->my_name.gnix_addr,
		   sizeof(struct gnix_address))) {
		return  __gnix_vc_connect_to_self(vc);
	}

	/*
	 * allocate a work request and try to
	 * run the progress function once.  If it
	 * doesn't succeed, put it on the cm_nic work queue.
	 */

	work_req = calloc(1, sizeof(*work_req));
	if (work_req == NULL)
		return -FI_ENOMEM;

	work_req->progress_fn = __gnix_vc_conn_req_prog_fn;
	work_req->data = vc;
	work_req->completer_fn = __gnix_vc_conn_req_comp_fn;
	work_req->completer_data = vc;

	/*
	 * add the work request to the tail of the
	 * cm_nic's work queue, progress the cm_nic.
	 */

	fastlock_acquire(&cm_nic->wq_lock);
	dlist_insert_before(&work_req->list, &cm_nic->cm_nic_wq);
	fastlock_release(&cm_nic->wq_lock);

	ret = _gnix_cm_nic_progress(cm_nic);

	return ret;
}

/*
 * TODO: this is very simple right now and will need more
 * work to propertly disconnect
 */

int _gnix_vc_disconnect(struct gnix_vc *vc)
{
	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	vc->conn_state = GNIX_VC_CONN_TERMINATED;
	return FI_SUCCESS;
}

/* Return 0 if VC is connected.  Progress VC CM if not. */
static int __gnix_vc_connected(struct gnix_vc *vc)
{
	struct gnix_cm_nic *cm_nic;
	int ret;

	if (unlikely(vc->conn_state < GNIX_VC_CONNECTED)) {
		cm_nic = vc->ep->cm_nic;
		ret = _gnix_cm_nic_progress(cm_nic);
		if ((ret != FI_SUCCESS) && (ret != -FI_EAGAIN))
			GNIX_WARN(FI_LOG_EP_CTRL,
				  "_gnix_cm_nic_progress() failed: %s\n",
				   fi_strerror(-ret));
		/* waiting to connect, check back later */
		return -FI_EAGAIN;
	}

	return 0;
}


/******************************************************************************
 *
 * VC RX progress
 *
 *****************************************************************************/

/* Schedule the VC for RX progress. */
int _gnix_vc_rx_schedule(struct gnix_vc *vc)
{
	struct gnix_nic *nic = vc->ep->nic;

	if (!_gnix_test_and_set_bit(&vc->flags, GNIX_VC_FLAG_RX_SCHEDULED)) {
		fastlock_acquire(&nic->rx_vc_lock);
		dlist_insert_tail(&vc->rx_list, &nic->rx_vcs);
		fastlock_release(&nic->rx_vc_lock);
		GNIX_INFO(FI_LOG_EP_CTRL, "Scheduled RX VC (%p)\n", vc);
	}

	return FI_SUCCESS;
}

/* Process a VC's SMSG mailbox. */
int _gnix_vc_dequeue_smsg(struct gnix_vc *vc)
{
	int ret = FI_SUCCESS;
	struct gnix_nic *nic;
	gni_return_t status;
	void *msg_ptr;
	uint8_t tag;

	GNIX_TRACE(FI_LOG_EP_DATA, "\n");

	assert(vc->gni_ep != NULL);
	assert(vc->conn_state == GNIX_VC_CONNECTED);

	nic = vc->ep->nic;
	assert(nic != NULL);

	do {
		tag = GNI_SMSG_ANY_TAG;
		status = GNI_SmsgGetNextWTag(vc->gni_ep,
					     &msg_ptr,
					     &tag);

		if (status == GNI_RC_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA, "Found RX (%p)\n", vc);
			ret = nic->smsg_callbacks[tag](vc, msg_ptr);
			if (ret != FI_SUCCESS) {
				/* Stalled, reschedule */
				break;
			}
		} else if (status == GNI_RC_NOT_DONE) {
			/* No more work. */
			ret = FI_SUCCESS;
			break;
		} else {
			GNIX_WARN(FI_LOG_EP_DATA,
				"GNI_SmsgGetNextWTag returned %s\n",
				gni_err_str[status]);
			ret = gnixu_to_fi_errno(status);
			break;
		}
	} while (1);

	return ret;
}

/* Progress VC RXs.  Reschedule VC if more there is more work. */
static int __gnix_vc_rx_progress(struct gnix_vc *vc)
{
	int ret;

	ret = __gnix_vc_connected(vc);
	if (ret) {
		/* The CM will schedule the VC when the connection is complete.
		 * Return success to allow continued VC RX processing. */
		_gnix_vc_rx_schedule(vc);
		return FI_SUCCESS;
	}

	/* Process pending RXs */
	fastlock_acquire(&vc->ep->nic->lock);
	ret = _gnix_vc_dequeue_smsg(vc);
	fastlock_release(&vc->ep->nic->lock);

	if (ret != FI_SUCCESS) {
		/* We didn't finish processing RXs.  Low memory likely.
		 * Try again later.  Return error to abort processing
		 * other VCs. */
		_gnix_vc_rx_schedule(vc);
		return -FI_EAGAIN;
	}

	/* Return success to continue processing other VCs */
	return FI_SUCCESS;
}

static struct gnix_vc *__gnix_nic_next_pending_rx_vc(struct gnix_nic *nic)
{
	struct gnix_vc *vc = NULL;

	fastlock_acquire(&nic->rx_vc_lock);
	vc = dlist_first_entry(&nic->rx_vcs, struct gnix_vc, rx_list);
	if (vc)
		dlist_remove_init(&vc->rx_list);
	fastlock_release(&nic->rx_vc_lock);

	if (vc) {
		GNIX_INFO(FI_LOG_EP_CTRL, "Dequeued RX VC (%p)\n", vc);
		_gnix_clear_bit(&vc->flags, GNIX_VC_FLAG_RX_SCHEDULED);
	}

	return vc;
}

/* Progress VC RXs.  Exit when all VCs are empty or if an error is encountered
 * during progress.  Failure to process an RX on any VC likely indicates an
 * inability to progress other VCs (due to low memory). */
static int __gnix_nic_vc_rx_progress(struct gnix_nic *nic)
{
	struct gnix_vc *vc;
	int ret;

	while ((vc = __gnix_nic_next_pending_rx_vc(nic))) {
		ret = __gnix_vc_rx_progress(vc);
		if (ret != FI_SUCCESS)
			break;
	}

	return FI_SUCCESS;
}


/******************************************************************************
 *
 * VC work progress
 *
 *****************************************************************************/

/* Schedule the VC for work progress. */
static int __gnix_vc_work_schedule(struct gnix_vc *vc)
{
	struct gnix_nic *nic = vc->ep->nic;

	/* Don't bother scheduling if there's no work to do. */
	if (slist_empty(&vc->work_queue))
		return FI_SUCCESS;

	if (!_gnix_test_and_set_bit(&vc->flags, GNIX_VC_FLAG_WORK_SCHEDULED)) {
		fastlock_acquire(&nic->work_vc_lock);
		dlist_insert_tail(&vc->work_list, &nic->work_vcs);
		fastlock_release(&nic->work_vc_lock);
		GNIX_INFO(FI_LOG_EP_CTRL, "Scheduled work VC (%p)\n", vc);
	}

	return FI_SUCCESS;
}

/* Schedule deferred request processing.  Usually used in RX completers. */
int _gnix_vc_queue_work_req(struct gnix_fab_req *req)
{
	struct gnix_vc *vc = req->vc;

	fastlock_acquire(&vc->work_queue_lock);
	slist_insert_tail(&req->slist, &vc->work_queue);
	__gnix_vc_work_schedule(vc);
	fastlock_release(&vc->work_queue_lock);

	return FI_SUCCESS;
}

/* Process deferred request work on the VC. */
static int __gnix_vc_push_work_reqs(struct gnix_vc *vc)
{
	int ret, fi_rc = FI_SUCCESS;
	struct slist *list;
	struct slist_entry *item;
	struct gnix_fab_req *req;

	fastlock_acquire(&vc->work_queue_lock);

	list = &vc->work_queue;
	item = list->head;
	while (item != NULL) {
		req = (struct gnix_fab_req *)container_of(item,
							  struct gnix_fab_req,
							  slist);
		ret = req->work_fn(req);
		if (ret == FI_SUCCESS) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Request processed: %p\n", req);
		} else {
			/* Work failed.  Reschedule to put this VC back on the
			 * end of the list. */
			__gnix_vc_work_schedule(vc);

			/* FI_ENOSPC is reserved to indicate a lack of TXDs,
			 * which are shared by all VCs on the NIC.  Return
			 * error to stall processing of VCs in this case.  The
			 * other likely error is a lack of SMSG credits, which
			 * only halts this VC. */
			if (ret == -FI_ENOSPC) {
				fi_rc = -FI_EAGAIN;
			} else if (ret != -FI_EAGAIN) {
				/* TODO report error? */
				GNIX_WARN(FI_LOG_EP_DATA,
					  "Failed to push request %p: %s\n",
					  req, fi_strerror(-ret));
			} /* else return success to keep processing TX VCs */
			break;
		}

		slist_remove_head(&vc->work_queue);
		item = list->head;
	}

	fastlock_release(&vc->work_queue_lock);

	return fi_rc;
}

static struct gnix_vc *__gnix_nic_next_pending_work_vc(struct gnix_nic *nic)
{
	struct gnix_vc *vc = NULL;

	fastlock_acquire(&nic->work_vc_lock);
	vc = dlist_first_entry(&nic->work_vcs, struct gnix_vc, work_list);
	if (vc)
		dlist_remove_init(&vc->work_list);
	fastlock_release(&nic->work_vc_lock);

	if (vc) {
		GNIX_INFO(FI_LOG_EP_CTRL, "Dequeued work VC (%p)\n", vc);
		_gnix_clear_bit(&vc->flags, GNIX_VC_FLAG_WORK_SCHEDULED);
	}

	return vc;
}

/* Progress VCs with deferred request work. */
static int __gnix_nic_vc_work_progress(struct gnix_nic *nic)
{
	int ret;
	struct gnix_vc *first_vc = NULL, *vc;

	while ((vc = __gnix_nic_next_pending_work_vc(nic))) {
		ret = __gnix_vc_push_work_reqs(vc);
		if (ret != FI_SUCCESS)
			break;

		if (!first_vc) {
			/* Record first VC processed. */
			first_vc = vc;
		} else if (vc == first_vc) {
			/* VCs can self reschedule or be rescheduled in other
			 * threads.  Exit if we loop back to the first VC. */
			break;
		}
	}

	return FI_SUCCESS;
}

/******************************************************************************
 *
 * VC TX progress
 *
 *****************************************************************************/

/* Schedule the VC for TX progress. */
int _gnix_vc_tx_schedule(struct gnix_vc *vc)
{
	struct gnix_nic *nic = vc->ep->nic;

	/* Don't bother scheduling if there's no work to do. */
	if (slist_empty(&vc->tx_queue))
		return FI_SUCCESS;

	if (!_gnix_test_and_set_bit(&vc->flags, GNIX_VC_FLAG_TX_SCHEDULED)) {
		fastlock_acquire(&nic->tx_vc_lock);
		dlist_insert_tail(&vc->tx_list, &nic->tx_vcs);
		fastlock_release(&nic->tx_vc_lock);
		GNIX_INFO(FI_LOG_EP_CTRL, "Scheduled TX VC (%p)\n", vc);
	}

	return FI_SUCCESS;
}

/* Attempt to initiate a TX request.  If the TX queue is blocked (due to low
 * resources or a FI_FENCE request), schedule the request to be sent later. */
int _gnix_vc_queue_tx_req(struct gnix_fab_req *req)
{
	int rc, queue_tx = 0;
	struct gnix_vc *vc = req->vc;
	int connected;

	connected = !__gnix_vc_connected(vc); /* 0 on success */

	fastlock_acquire(&vc->tx_queue_lock);

	if ((req->flags & FI_FENCE) && atomic_get(&vc->outstanding_tx_reqs)) {
		/* Fence request must be queued until all outstanding TX
		 * requests are completed.  Subsequent requests will be queued
		 * due to non-empty tx_queue. */
		queue_tx = 1;
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Queued FI_FENCE request (%p) on VC\n",
			  req);
	} else if (connected && slist_empty(&vc->tx_queue)) {
		/* try to initiate request */
		rc = req->work_fn(req);
		if (rc != FI_SUCCESS) {
			queue_tx = 1;
			GNIX_INFO(FI_LOG_EP_DATA,
				  "Queued request (%p) on full VC\n",
				  req);
		} else {
			atomic_inc(&vc->outstanding_tx_reqs);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "TX request processed: %p (OTX: %d)\n",
				  req, atomic_get(&vc->outstanding_tx_reqs));
		}
	} else {
		queue_tx = 1;
		GNIX_INFO(FI_LOG_EP_DATA,
			  "Queued request (%p) on busy VC\n",
			  req);
	}

	if (unlikely(queue_tx)) {
		slist_insert_tail(&req->slist, &vc->tx_queue);
		_gnix_vc_tx_schedule(vc);
	}

	fastlock_release(&vc->tx_queue_lock);

	return FI_SUCCESS;
}

/* Push TX requests queued on the VC. */
static int __gnix_vc_push_tx_reqs(struct gnix_vc *vc)
{
	int ret, fi_rc = FI_SUCCESS;
	struct slist *list;
	struct slist_entry *item;
	struct gnix_fab_req *req;

	ret = __gnix_vc_connected(vc);
	if (ret) {
		/* The CM will schedule the VC when the connection is complete.
		 * Return success to allow continued VC TX processing. */
		_gnix_vc_tx_schedule(vc);
		return FI_SUCCESS;
	}

	fastlock_acquire(&vc->tx_queue_lock);

	list = &vc->tx_queue;
	item = list->head;
	while (item != NULL) {
		req = (struct gnix_fab_req *)container_of(item,
							  struct gnix_fab_req,
							  slist);

		if ((req->flags & FI_FENCE) &&
		    atomic_get(&vc->outstanding_tx_reqs)) {
			GNIX_INFO(FI_LOG_EP_DATA,
				  "TX request queue stalled on FI_FENCE request: %p (%d)\n",
				  req, atomic_get(&vc->outstanding_tx_reqs));
			/* Success is returned to allow processing of more VCs.
			 * This VC will be rescheduled when the fence request
			 * is completed. */
			break;
		}

		ret = req->work_fn(req);
		if (ret == FI_SUCCESS) {
			atomic_inc(&vc->outstanding_tx_reqs);
			GNIX_INFO(FI_LOG_EP_DATA,
				  "TX request processed: %p (OTX: %d)\n",
				  req, atomic_get(&vc->outstanding_tx_reqs));
		} else {
			/* Work failed.  Reschedule to put this VC back on the
			 * end of the list. */
			_gnix_vc_tx_schedule(vc);

			GNIX_INFO(FI_LOG_EP_DATA,
				  "Failed to push TX request %p: %s\n",
				  req, fi_strerror(-ret));

			/* FI_ENOSPC is reserved to indicate a lack of TXDs,
			 * which are shared by all VCs on the NIC.  Return
			 * error to stall processing of VCs in this case.  The
			 * other likely error is a lack of SMSG credits, which
			 * only halts this VC. */
			if (ret == -FI_ENOSPC) {
				fi_rc = -FI_EAGAIN;
			} else if (ret != -FI_EAGAIN) {
				/* TODO report error? */
				GNIX_WARN(FI_LOG_EP_DATA,
					  "Failed to push TX request %p: %s\n",
					  req, fi_strerror(-ret));
			} /* else return success to keep processing TX VCs */
			break;
		}

		slist_remove_head(&vc->tx_queue);
		item = list->head;

		/* Return success if the queue is emptied. */
	}

	fastlock_release(&vc->tx_queue_lock);

	return fi_rc;
}

static struct gnix_vc *__gnix_nic_next_pending_tx_vc(struct gnix_nic *nic)
{
	struct gnix_vc *vc = NULL;

	fastlock_acquire(&nic->tx_vc_lock);
	vc = dlist_first_entry(&nic->tx_vcs, struct gnix_vc, tx_list);
	if (vc)
		dlist_remove_init(&vc->tx_list);
	fastlock_release(&nic->tx_vc_lock);

	if (vc) {
		GNIX_INFO(FI_LOG_EP_CTRL, "Dequeued TX VC (%p)\n", vc);
		_gnix_clear_bit(&vc->flags, GNIX_VC_FLAG_TX_SCHEDULED);
	}

	return vc;
}

/* Progress VC TXs.  Exit when all VCs TX queues are empty or stalled. */
static int __gnix_nic_vc_tx_progress(struct gnix_nic *nic)
{
	int ret;
	struct gnix_vc *first_vc = NULL, *vc;

	while ((vc = __gnix_nic_next_pending_tx_vc(nic))) {
		ret = __gnix_vc_push_tx_reqs(vc);
		if (ret != FI_SUCCESS)
			break;

		if (!first_vc) {
			/* Record first VC processed. */
			first_vc = vc;
		} else if (vc == first_vc) {
			/* VCs can self reschedule or be rescheduled in other
			 * threads.  Exit if we loop back to the first VC. */
			break;
		}
	}

	return FI_SUCCESS;
}

/* Progress all NIC VCs needing work. */
int _gnix_nic_vc_progress(struct gnix_nic *nic)
{
	/* Process VCs with RX traffic pending */
	__gnix_nic_vc_rx_progress(nic);

	/* Process deferred request work (deferred RX processing, etc.) */
	__gnix_nic_vc_work_progress(nic);

	/* Process VCs with TX traffic pending */
	__gnix_nic_vc_tx_progress(nic);

	return FI_SUCCESS;
}

/* Schedule VC to have all work queues processed.  This should only be needed
 * for newly connected VCs.
 *
 * TODO This is currently used to advance CM state.
 */
int _gnix_vc_schedule(struct gnix_vc *vc)
{
	_gnix_vc_rx_schedule(vc);
	__gnix_vc_work_schedule(vc);
	_gnix_vc_tx_schedule(vc);

	return FI_SUCCESS;
}

static int __gnix_ep_rdm_get_vc(struct gnix_fid_ep *ep, fi_addr_t dest_addr,
			    struct gnix_vc **vc_ptr)
{
	int ret = FI_SUCCESS;
	struct gnix_vc *vc = NULL, *vc_tmp;
	struct gnix_fid_av *av;
	struct gnix_av_addr_entry *av_entry;
	gnix_ht_key_t key;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	av = ep->av;
	assert(av != NULL);

	ret = _gnix_av_lookup(av, dest_addr, &av_entry);
	if (ret != FI_SUCCESS) {
		GNIX_WARN(FI_LOG_EP_DATA,
			  "_gnix_av_lookup returned %s\n",
			  fi_strerror(-ret));
		goto err;
	}
	GNIX_INFO(FI_LOG_EP_CTRL, "fi_addr_t: 0x%llx gnix_addr: 0x%llx\n",
		  dest_addr, av_entry->gnix_addr);

	memcpy(&key, &av_entry->gnix_addr, sizeof(gnix_ht_key_t));

	fastlock_acquire(&ep->vc_ht_lock);
	vc = (struct gnix_vc *)_gnix_ht_lookup(ep->vc_ht,
						key);
	if (vc == NULL) {
		ret = _gnix_vc_alloc(ep,
				     av_entry,
				     &vc_tmp);
		if (ret != FI_SUCCESS) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_vc_alloc returned %s\n",
				  fi_strerror(-ret));
			goto err;
		}
		ret = _gnix_ht_insert(ep->vc_ht, key,
					vc_tmp);
		fastlock_release(&ep->vc_ht_lock);
		if (likely(ret == FI_SUCCESS)) {
			vc = vc_tmp;
			vc->modes |= GNIX_VC_MODE_IN_HT;
			ret = _gnix_vc_connect(vc);
			if (ret != FI_SUCCESS) {
				GNIX_WARN(FI_LOG_EP_DATA,
					"_gnix_vc_connect returned %s\n",
					   fi_strerror(-ret));
				goto err;
			}
		} else if (ret == -FI_ENOSPC) {
			_gnix_vc_destroy(vc_tmp);
			fastlock_acquire(&ep->vc_ht_lock);
			vc = _gnix_ht_lookup(ep->vc_ht, key);
			fastlock_release(&ep->vc_ht_lock);
			assert(vc != NULL);
			assert(vc->modes & GNIX_VC_MODE_IN_HT);
			ret = FI_SUCCESS;
		} else {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_ht_insert returned %s\n",
				   fi_strerror(-ret));
			goto err;
		}
	}
	*vc_ptr = vc;
	fastlock_release(&ep->vc_ht_lock);
	return ret;
err:
	if (vc != NULL)
		_gnix_vc_destroy(vc);
	return ret;
}

int _gnix_ep_get_vc(struct gnix_fid_ep *ep, fi_addr_t dest_addr,
			struct gnix_vc **vc_ptr)
{
	int ret;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	if (ep->type == FI_EP_RDM) {
		ret = __gnix_ep_rdm_get_vc(ep, dest_addr, vc_ptr);
		if (unlikely(ret != FI_SUCCESS)) {
			GNIX_WARN(FI_LOG_EP_DATA,
				  "__gnix_ep_get_vc returned %s\n",
				   fi_strerror(-ret));
			return ret;
		}
	} else if (ep->type == FI_EP_MSG) {
		*vc_ptr = ep->vc;
	} else {
		GNIX_WARN(FI_LOG_EP_DATA, "Invalid endpoint type: %d\n",
			  ep->type);
		return -FI_EINVAL;
	}

	return FI_SUCCESS;
}

fi_addr_t _gnix_vc_peer_fi_addr(struct gnix_vc *vc)
{
	int rc;

	/* If FI_SOURCE capability was requested, do a reverse lookup of a VC's
	 * FI address once.  Skip translation on connected EPs (no AV). */
	if (vc->ep->caps & FI_SOURCE &&
	    vc->ep->av &&
	    vc->peer_fi_addr == FI_ADDR_NOTAVAIL) {
		rc = _gnix_av_reverse_lookup(vc->ep->av,
					     vc->peer_addr,
					     &vc->peer_fi_addr);
		if (rc != FI_SUCCESS)
			GNIX_WARN(FI_LOG_EP_DATA,
				  "_gnix_av_reverse_lookup() failed: %d\n",
				  rc);
	}

	return vc->peer_fi_addr;
}

int _gnix_vc_cm_init(struct gnix_cm_nic *cm_nic)
{
	int ret = FI_SUCCESS;
	gnix_cm_nic_rcv_cb_func *ofunc = NULL;

	GNIX_TRACE(FI_LOG_EP_CTRL, "\n");

	fastlock_acquire(&cm_nic->lock);
	ret = _gnix_cm_nic_reg_recv_fn(cm_nic,
					__gnix_vc_recv_fn,
					&ofunc);
	if ((ofunc != NULL) &&
	    (ofunc != __gnix_vc_recv_fn)) {
		GNIX_WARN(FI_LOG_EP_DATA, "callback reg failed: %s\n",
			  fi_strerror(-ret));
	}

	fastlock_release(&cm_nic->lock);

	return ret;
}

