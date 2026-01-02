/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_ZBCOLL_H_
#define _CXIP_ZBCOLL_H_


#include <stdint.h>
#include <stdbool.h>
#include <ofi_list.h>
#include <ofi_atom.h>
#include <ofi_lock.h>

/* Forward declarations */
struct cxip_addr;
struct cxip_ep_obj;

/* Type definitions */
struct cxip_zbcoll_cb_obj {
	zbcomplete_t usrfunc;		// callback function
	void *usrptr;			// callback data
};

struct cxip_zbcoll_state {
	struct cxip_zbcoll_obj *zb;	// backpointer to zbcoll_obj
	uint64_t *dataptr;		// user-supplied target
	uint64_t dataval;		// collective data
	int num_relatives;		// number of nearest relatives
	int *relatives;			// nearest relative indices
	int contribs;			// contribution count
	int grp_rank;			// local rank within group
};

struct cxip_zbcoll_obj {
	struct dlist_entry ready_link;	// link to zb_coll ready_list
	struct cxip_ep_obj *ep_obj;	// backpointer to endpoint
	struct cxip_zbcoll_state *state;// state array
	struct cxip_addr *caddrs;	// cxip addresses in collective
	int num_caddrs;			// number of cxip addresses
	zbcomplete_t userfunc;		// completion callback function
	void *userptr;			// completion callback data
	uint64_t *grpmskp;		// pointer to global group mask
	uint32_t *shuffle;		// TEST shuffle array
	int simcount;			// TEST count of states
	int simrank;			// TEST simulated rank
	int simref;			// TEST zb0 reference count
	int busy;			// serialize collectives in zb
	int grpid;			// zb collective grpid
	int error;			// error code
	int reduce;			// set to report reduction data
};

struct cxip_ep_zbcoll_obj {
	struct dlist_entry ready_list;	// zbcoll ops ready to advance
	struct cxip_zbcoll_obj **grptbl;// group lookup table
	uint64_t grpmsk;		// mask of used grptbl entries
	int refcnt;			// grptbl reference count
	bool disable;			// low level tests
	ofi_spin_t lock;		// group ID negotiation lock
	ofi_atomic32_t dsc_count;	// cumulative RCV discard count
	ofi_atomic32_t err_count;	// cumulative ACK error count
	ofi_atomic32_t ack_count;	// cumulative ACK success count
	ofi_atomic32_t rcv_count;	// cumulative RCV success count
};

/* Function declarations */
void cxip_tree_rowcol(int radix, int nodeidx, int *row, int *col, int *siz);

void cxip_tree_nodeidx(int radix, int row, int col, int *nodeidx);

int cxip_tree_relatives(int radix, int nodeidx, int maxnodes, int *rels);

int cxip_zbcoll_recv_cb(struct cxip_ep_obj *ep_obj, uint32_t init_nic,
			uint32_t init_pid, uint64_t mbv, uint64_t data);

void cxip_zbcoll_send(struct cxip_zbcoll_obj *zb, int srcidx, int dstidx,
		      uint64_t payload);

void cxip_zbcoll_free(struct cxip_zbcoll_obj *zb);

int cxip_zbcoll_alloc(struct cxip_ep_obj *ep_obj, int num_addrs,
		      fi_addr_t *fiaddrs, int simrank,
		      struct cxip_zbcoll_obj **zbp);

int cxip_zbcoll_simlink(struct cxip_zbcoll_obj *zb0,
			struct cxip_zbcoll_obj *zb);

void cxip_zbcoll_set_user_cb(struct cxip_zbcoll_obj *zb,
			     zbcomplete_t userfunc, void *userptr);

int cxip_zbcoll_max_grps(bool sim);

int cxip_zbcoll_getgroup(struct cxip_zbcoll_obj *zb);

void cxip_zbcoll_rlsgroup(struct cxip_zbcoll_obj *zb);

int cxip_zbcoll_broadcast(struct cxip_zbcoll_obj *zb, uint64_t *dataptr);

int cxip_zbcoll_reduce(struct cxip_zbcoll_obj *zb, uint64_t *dataptr);

int cxip_zbcoll_barrier(struct cxip_zbcoll_obj *zb);

void cxip_ep_zbcoll_progress(struct cxip_ep_obj *ep_obj);

void cxip_zbcoll_reset_counters(struct cxip_ep_obj *ep_obj);

void cxip_zbcoll_get_counters(struct cxip_ep_obj *ep_obj, uint32_t *dsc,
			      uint32_t *err, uint32_t *ack, uint32_t *rcv);

void cxip_zbcoll_fini(struct cxip_ep_obj *ep_obj);

int cxip_zbcoll_init(struct cxip_ep_obj *ep_obj);

#endif /* _CXIP_ZBCOLL_H_ */
