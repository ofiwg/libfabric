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

#ifndef _GNIX_NIC_H_
#define _GNIX_NIC_H_

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif /* HAVE_CONFIG_H */
#include <fi_list.h>
#include <assert.h>

#include "gnix.h"
#include "gnix_bitmap.h"
#include "gnix_mbox_allocator.h"
#include "gnix_util.h"

#define GNIX_DEF_MAX_NICS_PER_PTAG	4

extern uint32_t gnix_max_nics_per_ptag;

/*
 * allocation flags for cleaning up GNI resources
 * when closing a gnix_nic - needed since these
 * can be dup'd from another structure.
 */

#define GNIX_NIC_CDM_ALLOCD	(1ULL << 1)
#define GNIX_NIC_TX_CQ_ALLOCD	(1ULL << 2)
#define GNIX_NIC_RX_CQ_ALLOCD	(1ULL << 3)

/*
 * typedefs for callbacks for handling
 * receipt of SMSG messages at the target
 */
typedef int (*smsg_callback_fn_t)(void  *ptr, void *msg);

/*
 * typedef for completer functions invoked
 * at initiator when local CQE (tx) is processed
 */
typedef int (*smsg_completer_fn_t)(void  *desc, gni_return_t);

/**
 * Set of attributes that can be passed to the gnix_alloc_nic.
 *
 * @var gni_cdm_hndl         optional previously allocated gni_cdm_hndl to
 *                           use for allocating GNI resources (GNI CQs) for
 *                           this nic.
 * @var gni_nic_hndl         optional previously allocated gni_nic_hndl to
 *                           use for allocating GNI resources (GNI CQs) for
 *                           this nic
 */
struct gnix_nic_attr {
	gni_cdm_handle_t gni_cdm_hndl;
	gni_nic_handle_t gni_nic_hndl;
};

/**
 * GNIX nic struct
 *
 * @var gnix_nic_list        list element used for global NIC list
 * @var dom_nic_list         list element used for nic linked list associated
 *                           with a given gnix_fid_domain
 * @var lock                 lock used for serializing access to
 *                           gni_nic_hndl, rx_cq, and tx_cq
 * @var gni_cdm_hndl         handle for the GNI communication domain (CDM)
 *                           this nic is bound to.
 * @var gni_nic_hndl         handle for the GNI nic to which this GNIX nic is bound
 * @var rx_cq                GNI rx cq (non-blocking) bound to this nic
 * @var rx_cq_blk            GNI rx cq (blocking) bound to this nic
 * @var tx_cq                GNI tx cq (non-blocking) bound to this nic
 * @var tx_cq_blk            GNI tx cq (blocking) bound to this nic
 * @var progress_thread      thread id of progress thread for this nic
 * @var tx_desc_active_list  linked list of active tx descriptors associated
 *                           with this nic
 * @var tx_desc_free_list    linked list of free tx descriptors associated
 *                           with this nic
 * @var tx_desc_base         base address for the block of memory from which
 *                           tx descriptors were allocated
 * @var wq_lock              lock for serializing access to the nic's work queue
 * @var nic_wq               head of linked list of work queue elements
 *                           associated with this nic
 * @var ptag                 ptag of the GNI CDM this nic is bound to
 * @var cookie               cookie of the GNI CDM this nic is bound to
 *                           of the VC when using GNI_CQ_GET_INST_ID to get
 * @var device_id            device id of the GNI nic this nic is bound to (always 0,
 *                           unless ever need to support multiple GNI nics/node)
 * @var device_addr          address (L2) of the GNI nic to which this nic is bound
 * @var max_tx_desc_id       max tx descriptor id for this nic
 * @var vc_id_lock           lock for serializing access to the vc_id_table for
 *                           this nic
 * @var vc_id_table          base address of the nic's vc_id_table
 * @var vc_id_table_capacity current capacity of the nic's vc_id_table
 * @var vc_id_table_count    current location of the next unoccupied entry in the
 *                           vc_id_table - note there may be unused entries
 *                           below this entry.
 * @var vc_id_bitmap         bitmap indicating which entries in the vc_id_table are
 *                           currently in use (1 - used, 0 - unused)
 * @var mem_per_mbox         number of bytes consumed per GNI SMSG mailbox associated
 *                           with this nic's vd_id_table
 * @var mbox_hndl            handle for the mailbox allocator bound to this nic
 * @var s_rdma_buf_hndl      handle for send side rdma buffer allocator bound to this nic
 * @var r_rdma_buf_hndl      handle for recv side rdma buffer allocator bound to this nic
 * @var ref_cnt              ref cnt for this nid
 * @var smsg_callbacks       pointer to table of GNI SMSG callback functions used
 *                           by this nic for processing incoming GNI SMS
 *                           messages
 * @var err_txds             slist of error'd tx descriptors
 * @var tx_cq_blk_post_cnt   count of outstanding tx desc's posted using tx_cq_blk
 *                           GNI CQ.
 * @var irq_mem_hndl         gni_mem_handle_t for mmap region registered with
 *                           gni hw cq handle used for GNI_PostCqWrite
 * @var irq_mmap_addr        base address of mmap associated with irq_dma_hndl
 * @var irq_mmap_len         length of the mmap in bytes
 */
struct gnix_nic {
	struct dlist_entry gnix_nic_list; /* global NIC list */
	struct dlist_entry dom_nic_list;  /* domain NIC list */
	fastlock_t lock;
	uint32_t allocd_gni_res;
	gni_cdm_handle_t gni_cdm_hndl;
	gni_nic_handle_t gni_nic_hndl;
	gni_cq_handle_t rx_cq;
	gni_cq_handle_t rx_cq_blk;
	gni_cq_handle_t tx_cq;
	gni_cq_handle_t tx_cq_blk;
	pthread_t progress_thread;
	fastlock_t tx_desc_lock;
	struct dlist_entry tx_desc_active_list;
	struct dlist_entry tx_desc_free_list;
	struct gnix_tx_descriptor *tx_desc_base;
	fastlock_t rx_vc_lock;
	struct dlist_entry rx_vcs;
	fastlock_t work_vc_lock;
	struct dlist_entry work_vcs;
	fastlock_t tx_vc_lock;
	struct dlist_entry tx_vcs;
	uint8_t ptag;
	uint32_t cookie;
	uint32_t device_id;
	uint32_t device_addr;
	int max_tx_desc_id;
	fastlock_t vc_id_lock;
	void **vc_id_table;
	int vc_id_table_capacity;
	int vc_id_table_count;
	gnix_bitmap_t vc_id_bitmap;
	uint32_t mem_per_mbox;
	struct gnix_mbox_alloc_handle *mbox_hndl;
	struct gnix_mbox_alloc_handle *s_rdma_buf_hndl;
	struct gnix_mbox_alloc_handle *r_rdma_buf_hndl;
	struct gnix_reference ref_cnt;
	smsg_callback_fn_t const *smsg_callbacks;
	struct slist err_txds;
	void *int_bufs;
	gni_mem_handle_t int_bufs_mdh;
	gni_mem_handle_t irq_mem_hndl;
	void *irq_mmap_addr;
	size_t irq_mmap_len;
};


/**
 * gnix_smsg_eager_hdr  - first part of an eager send SMSG message
 *
 * @var flags      flag bits from send side that are needed at
 *                 rcv side (e.g. FI_REMOTE_CQ_DATA)
 * @var imm        immediate data associated with this message
 * @var msg_tag    libfabric tag associated with this message
 * @var len        length in bytes of the incoming message
 */
struct gnix_smsg_eager_hdr {
	uint64_t flags;
	uint64_t imm;
	uint64_t msg_tag;
	size_t len;
};

/**
 * gnix_smsg_rndzv_start_hdr  - first part of a rendezvous send start SMSG
 *                              message
 *
 * @var flags      flag bits from send side that are needed at
 *                 rcv side (e.g. FI_REMOTE_CQ_DATA)
 * @var imm        immediate data associated with this message
 * @var msg_tag    libfabric tag associated with this message
 * @var mdh        MDH for the rendezvous send buffer
 * @var addr       address of the rendezvous send buffer
 * @var len        length in bytes of the send buffer
 * @var req_addr   local request address
 * @var head       unaligned data at the head of a rendezvous send
 * @var tail       unaligned data at the tail of a rendezvous send
 */
struct gnix_smsg_rndzv_start_hdr {
	uint64_t flags;
	uint64_t imm;
	uint64_t msg_tag;
	gni_mem_handle_t mdh;
	uint64_t addr;
	size_t len;
	uint64_t req_addr;
	uint32_t head;
	uint32_t tail;
};

/**
 * gnix_smsg_rndzv_fin_hdr  - first part of a rendezvous send fin SMSG message
 *
 * @var req_addr   returned local request address
 */
struct gnix_smsg_rndzv_fin_hdr {
	uint64_t req_addr;
};

/**
 * gnix_smsg_rma_data_hdr  - RMA remote data message
 *
 * @var flags  remote CQ RMA flags
 * @var data   remote CQ immediate data
 */
struct gnix_smsg_rma_data_hdr {
	uint64_t flags;
	uint64_t data;
};

/**
 * gni_tx_descriptor - full tx descriptor used to to track GNI SMSG
 *                     and Post operations
 *
 * @var list             list element
 * @var gni_desc         embedded GNI post descriptor
 * @var gnix_smsg_desc   embedded gnix SMSG descriptor
 * @var req              pointer to fab request associated with this descriptor
 * @var id               the id of this descriptor - the value returned
 *                       from GNI_CQ_MSG_ID
 * @var err_list         Error TXD list entry
 * @var int_buf          Intermediate buffer for landing unaligned data, etc.
 */
struct gnix_tx_descriptor {
	struct dlist_entry          list;
	union {
		struct {
			gni_post_descriptor_t        gni_desc;
			gni_ct_get_post_descriptor_t gni_ct_descs[2];
		};
		struct gnix_smsg_eager_hdr       eager_hdr;
		struct gnix_smsg_rndzv_start_hdr rndzv_start_hdr;
		struct gnix_smsg_rndzv_fin_hdr   rndzv_fin_hdr;
		struct gnix_smsg_rma_data_hdr    rma_data_hdr;
	};
	struct gnix_fab_req *req;
	int  (*completer_fn)(void *, gni_return_t);
	int id;
	struct slist_entry err_list;
	void *int_buf;
};

/*
 * globals
 */

extern uint32_t gnix_def_max_nics_per_ptag;

/*
 * prototypes
 */

/**
 * @brief allocate a tx descriptor to use for GNI Post, SMSG ops
 *
 * @param[in] nic      pointer to previously allocated gnix_nic struct
 * @param[in] tdesc    pointer to address where allocated tx descriptor
 *                     is to be stored
 * @return             FI_SUCCESS on success, -FI_ENOSPC no free tx descriptors
 */
int _gnix_nic_tx_alloc(struct gnix_nic *nic, struct gnix_tx_descriptor **tdesc);

/**
 * @brief frees a previously allocated tx descriptor
 *
 * @param[in] nic      pointer to previously allocated gnix_nic struct used
 *                     when allocating the tx descriptor to be freed
 * @param[in] tdesc    pointer to previously allocated tx descriptor
 * @return             FI_SUCCESS on success
 */
int _gnix_nic_tx_free(struct gnix_nic *nic, struct gnix_tx_descriptor *tdesc);

/**
 * @brief allocate a gnix_nic struct
 *
 * @param[in] domain   pointer to previously allocated gnix_fid_domain struct
 * @param[in] attrs    optional pointer to an attributes argument.  NULL
 *                     can be supplied if no attributes are required
 * @param[out] nic_ptr pointer to address where address of allocated nic is
 *                     to be returned
 * @return             FI_SUCCESS on success, -FI_ENOMEM if insufficient memory
 *                     to allocate nic struct, -FI_EINVAL if an invalid domain
 *                     struct was supplied, -FI_EBUSY if insufficient resources
 *                     are available to allocate the nic struct, -FI_EACCESS
 *                     if a permissions access error occurred while allocating
 *                     the nic struct, -FI_EAGAIN if an invalid state
 *                     prevents creation or an interrupt was received while
 *                     allocating kernel related resources for the nic.
 */
int gnix_nic_alloc(struct gnix_fid_domain *domain,
		   struct gnix_nic_attr *attrs,
		   struct gnix_nic **nic_ptr);

/**
 * @brief frees a previously allocated gnix_nic struct
 *
 * @param[in] nic      pointer to previously allocated gnix_nic struct
 * @return             FI_SUCCESS on success, -FI_ENOSPC no free tx descriptors
 */
int _gnix_nic_free(struct gnix_nic *nic);

/**
 * @brief progresses control/data operations associated with the nic
 *
 * @param[in] nic      pointer to previously allocated gnix_nic struct
 * @return             FI_SUCCESS on success, -FI_EINVAL if an invalid
 *                     nic struct was supplied. TODO: a lot more error
 *                     values can be returned.
 */
int _gnix_nic_progress(struct gnix_nic *nic);

/**
 * @brief allocate a remote id for an object, used for looking up an object
 *        in O(1) based on returned value of GNI_CQ_INST_ID applied to a GNI
 *        CQE
 *
 * @param[in] nic             pointer to previously allocated gnix_nic struct
 * @param[out] remote_id      address where allocate remote_id is returned
 * @param[in] entry           pointer to object to be associated with the
 *                            remote id

 * @return             FI_SUCCESS on success, -FI_ENOMEM if insufficient
 *                     memory to allocate remote_id
 */
int _gnix_nic_get_rem_id(struct gnix_nic *nic, int *remote_id, void *entry);

/**
 * @brief release a previously allocated remote_id
 *
 * @param[in] nic            pointer to previously allocated gnix_nic struct
 * @param[in] remote_id      previously allocated remote_id

 * @return             FI_SUCCESS on success, -FI_EINVAL if an invalid
 *                     argument was provided.
 */
int _gnix_nic_free_rem_id(struct gnix_nic *nic, int remote_id);

/**
 * @brief Look up an element by id
 *
 * @param[in] nic    pointer to gni nic with which the vc is associated
 * @param[in] rem_id rem_id of the object being looked up
 *                   rem_id comes from GNI_CQ_GET_INST_ID on a GNI CQE
 *
 * @return           pointer to vc with the given vc_id
 *
 * This function is only here because its used for criterion tests,
 * otherwise it would be a static function within gnix_nic.c
 */
static inline void *__gnix_nic_elem_by_rem_id(struct gnix_nic *nic, int rem_id)
{
	assert(nic);
	assert(rem_id <= nic->vc_id_table_count);
	return nic->vc_id_table[rem_id];
	return 0;
}

void _gnix_nic_txd_err_inject(struct gnix_nic *nic,
			      struct gnix_tx_descriptor *txd);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* _GNIX_NIC_H_ */
