/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_CONN_H
#define EFA_CONN_H

#include "ofi_util.h"
#include "efa_av.h"
#include "rdm/efa_rdm_peer.h"
#include "efa_thread_annotations.h"

int efa_conn_rdm_insert_shm_av(struct efa_av *av, struct efa_rdm_av_entry *av_entry);

void efa_conn_rdm_deinit(struct efa_av *av, struct efa_rdm_av_entry *av_entry);

struct efa_av_entry *efa_conn_alloc_explicit(struct efa_av *av, struct efa_ep_addr *raw_addr,
					uint64_t flags, void *context, bool insert_shm_av)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

struct efa_rdm_av_entry *efa_conn_alloc_implicit(struct efa_av *av, struct efa_ep_addr *raw_addr,
					 uint64_t flags, void *context)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_explicit(struct efa_av *av, struct efa_av_entry *entry)
	OFI_TSA_REQUIRES(efa_util_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_implicit(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

void efa_conn_release_implicit_ah_unsafe(struct efa_av *av, struct efa_rdm_av_entry *av_entry)
	OFI_TSA_REQUIRES(efa_implicit_av_lock_sym)
	OFI_TSA_REQUIRES(efa_util_domain_lock_sym);

#endif
