/*
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
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

#ifndef _GNIX_VC_H_
#define _GNIX_VC_H_

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */

#include "gnix.h"
#include "gnix_bitmap.h"
#include "gnix_av.h"

/*
 * mode bits
 */
#define GNIX_VC_MODE_IN_WQ		(1U)
#define GNIX_VC_MODE_IN_HT		(1U << 1)
#define GNIX_VC_MODE_DG_POSTED		(1U << 2)
#define GNIX_VC_MODE_PENDING_MSGS	(1U << 3)
#define GNIX_VC_MODE_PEER_CONNECTED	(1U << 4)

/* VC flags */
#define GNIX_VC_FLAG_SCHEDULED		0
#define GNIX_VC_FLAG_RX_PENDING		1

/*
 * defines for connection state for gnix VC
 */
enum gnix_vc_conn_state {
	GNIX_VC_CONN_NONE = 1,
	GNIX_VC_CONNECTING,
	GNIX_VC_CONNECTED,
	GNIX_VC_CONN_TERMINATING,
	GNIX_VC_CONN_TERMINATED,
	GNIX_VC_CONN_ERROR
};

enum gnix_vc_conn_req_type {
	GNIX_VC_CONN_REQ = 1,
	GNIX_VC_CONN_RESP
};

#define LOCAL_MBOX_SENT (1UL)
#define REMOTE_MBOX_RCVD (1UL << 1)

/**
 * Virual Connection (VC) struct
 *
 * @var tx_queue             linked list of pending send requests to be
 *                           delivered to peer_address
 * @var tx_queue_lock        lock for serializing access to vc's tx_queue
 * @var entry                used internally for managing linked lists
 *                           of vc structs that require O(1) insertion/removal
 * @var peer_addr            address of peer with which this VC is connected
 * @var peer_cm_nic_addr     address of the cm_nic being used by peer, this
 *                           is the address to which GNI datagrams must be
 *                           posted
 * @var ep                   libfabric endpoint with which this VC is
 *                           associated
 * @var smsg_mbox            pointer to GNI SMSG mailbox used by this VC
 *                           to exchange SMSG messages with its peer
 * @var dgram                pointer to dgram - used in connection setup
 * @var gni_ep               GNI endpoint for this VC
 * @var outstanding_fab_reqs Count of outstanding libfabric level requests
 *                           associated with this endpoint.
 * @var conn_state           Connection state of this VC
 * @var vc_id                ID of this vc. Allows for rapid O(1) lookup
 *                           of the VC when using GNI_CQ_GET_INST_ID to get
 *                           the inst_id of a GNI CQE.
 * @var modes                Used internally to track current state of
 *                           the VC not pertaining to the connection state.
 */
struct gnix_vc {
	struct slist tx_queue;
	fastlock_t tx_queue_lock;
	struct slist req_queue;
	fastlock_t req_queue_lock;
	struct dlist_entry entry;
	struct gnix_address peer_addr;
	struct gnix_address peer_cm_nic_addr;
	struct gnix_fid_ep *ep;
	void *smsg_mbox;
	struct gnix_datagram *dgram;
	gni_ep_handle_t gni_ep;
	atomic_t outstanding_tx_reqs;
	atomic_t outstanding_reqs;
	enum gnix_vc_conn_state conn_state;
	uint32_t post_state;
	int vc_id;
	int modes;
	struct dlist_entry pending_list;
	gnix_bitmap_t flags; /* We're missing regular bit ops */
};

/*
 * prototypes
 */

/**
 * @brief Allocates a virtual channel(vc) struct
 *
 * @param[in]  ep_priv    pointer to previously allocated gnix_fid_ep object
 * @param[in]  entry      av entry for remote peer for this VC.  Can be NULL
 *                        for accepting VCs.
 * @param[out] vc         location in which the address of the allocated vc
 *                        struct is to be returned.
 * @return FI_SUCCESS on success, -FI_ENOMEM if allocation of vc struct fails,
 */
int _gnix_vc_alloc(struct gnix_fid_ep *ep_priv,
		   struct gnix_av_addr_entry *entry, struct gnix_vc **vc);

/**
 * @brief Initiates non-blocking connect of a vc with its peer
 *
 * @param[in]  vc   pointer to previously allocated vc struct
 *
 * @return FI_SUCCESS on success, -FI_EINVAL if an invalid field in the vc
 *         struct is encountered, -ENOMEM if insufficient memory to initiate
 *         connection request.
 */
int _gnix_vc_connect(struct gnix_vc *vc);


/**
 * @brief Sets up an accepting vc - one accepting vc can accept a
 *        single incoming connection request
 *
 * @param[in]  vc   pointer to previously allocated vc struct with
 *                  FI_ADDR_UNSPEC value supplied for the fi_addr_t
 *                  argument
 *
 * @return FI_SUCCESS on success, -FI_EINVAL if an invalid field in the vc
 *         struct is encountered, -ENOMEM if insufficient memory to initiate
 *         accept request.
 */
int _gnix_vc_accept(struct gnix_vc *vc);

/**
 * @brief Initiates a non-blocking disconnect of a vc from its peer
 *
 * @param[in]  vc   pointer to previously allocated and connected vc struct
 *
 * @return FI_SUCCESS on success, -FI_EINVAL if an invalid field in the vc
 *         struct is encountered, -ENOMEM if insufficient memory to initiate
 *         connection request.
 */
int _gnix_vc_disconnect(struct gnix_vc *vc);


/**
 * @brief Destroys a previously allocated vc and cleans up resources
 *        associated with the vc
 *
 * @param[in]  vc   pointer to previously allocated vc struct
 *
 * @return FI_SUCCESS on success, -FI_EINVAL if an invalid field in the vc
 *         struct is encountered.
 */
int _gnix_vc_destroy(struct gnix_vc *vc);

/**
 * @brief Add a vc to the work queue of its associated nic
 *
 * @param[in] vc  pointer to previously allocated vc struct
 *
 * @return FI_SUCCESS on success, -ENOMEM if insufficient memory
 * 	   allocate memory to enqueue work request
 */
int _gnix_vc_add_to_wq(struct gnix_vc *vc);

/*
 * inline functions
 */

/**
 * @brief Return connection state of a vc
 *
 * @param[in]  vc     pointer to previously allocated vc struct
 * @return connection state of vc
 */
static inline enum gnix_vc_conn_state _gnix_vc_state(struct gnix_vc *vc)
{
	assert(vc);
	return vc->conn_state;
}


int _gnix_vc_schedule(struct gnix_vc *vc);
int _gnix_vc_schedule_reqs(struct gnix_vc *vc);
struct gnix_vc *_gnix_nic_next_pending_vc(struct gnix_nic *nic);
int _gnix_vc_dequeue_smsg(struct gnix_vc *vc);
int _gnix_vc_progress(struct gnix_vc *vc);
int _gnix_vc_queue_tx_req(struct gnix_fab_req *req);
int _gnix_vc_force_queue_req(struct gnix_fab_req *req);
int _gnix_vc_queue_req(struct gnix_fab_req *req);

/**
 * @brief  return vc associated with a given ep/dest address, or the ep in the
 *         case of FI_EP_MSG endpoint type.  For FI_EP_RDM type, a vc may be
 *         allocated and a connection initiated if no vc is associated with
 *         ep/dest_addr.
 *
 * @param[in] ep        pointer to a previously allocated endpoint
 * @param[in] dest_addr for FI_EP_RDM endpoints, used to look up vc associated
 *                      with this target address
 * @param[out] vc_ptr   address in which to store pointer to returned vc
 * @return              FI_SUCCESS on success, -FI_ENOMEM insufficient
 *                      memory to allocate vc, -FI_EINVAL if an invalid
 *                      argument was supplied
 */
int _gnix_ep_get_vc(struct gnix_fid_ep *ep, fi_addr_t dest_addr,
		    struct gnix_vc **vc_ptr);

int _gnix_vc_cm_init(struct gnix_cm_nic *cm_nic);

#endif /* _GNIX_VC_H_ */
