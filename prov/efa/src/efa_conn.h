/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_CONN_H
#define EFA_CONN_H

#include "ofi_util.h"
#include "rdm/efa_rdm_peer.h"
#include "efa_thread_annotations.h"

struct efa_conn {
	struct efa_ah *ah;
	struct efa_ep_addr *ep_addr;
	struct efa_av *av;
	fi_addr_t		implicit_fi_addr;
	fi_addr_t		fi_addr;
	fi_addr_t		shm_fi_addr;
	struct dlist_entry	implicit_av_lru_entry;
	struct dlist_entry ah_implicit_conn_list_entry OFI_TSA_GUARDED_BY(efa_util_domain_lock_sym);
};

int efa_conn_rdm_insert_shm_av(struct efa_av *av, struct efa_conn *conn);

void efa_conn_rdm_deinit(struct efa_av *av, struct efa_conn *conn);

struct efa_conn *efa_conn_alloc_explicit(struct efa_av *av, struct efa_ep_addr *raw_addr,
					uint64_t flags, void *context, bool insert_shm_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_conn *efa_conn_alloc_implicit(struct efa_av *av, struct efa_ep_addr *raw_addr,
					 uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_explicit(struct efa_av *av, struct efa_conn *conn)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_implicit(struct efa_av *av, struct efa_conn *conn)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_implicit_ah_unsafe(struct efa_av *av, struct efa_conn *conn)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif