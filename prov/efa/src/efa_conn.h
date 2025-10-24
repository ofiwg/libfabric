/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_CONN_H
#define EFA_CONN_H

#include "ofi_util.h"

struct efa_conn {
	struct efa_ah *ah;
	struct efa_ep_addr *ep_addr;
	struct efa_av *av;
	fi_addr_t		implicit_fi_addr;
	fi_addr_t		fi_addr;
	fi_addr_t		shm_fi_addr;
	struct dlist_entry	implicit_av_lru_entry;
	struct dlist_entry ah_implicit_conn_list_entry;
};

int efa_conn_rdm_insert_shm_av(struct efa_av *av, struct efa_conn *conn);

void efa_conn_rdm_deinit(struct efa_av *av, struct efa_conn *conn);

struct efa_conn *efa_conn_alloc(struct efa_av *av, struct efa_ep_addr *raw_addr,
				uint64_t flags, void *context, bool insert_shm_av, bool insert_implicit_av);

void efa_conn_release_reverse_av(struct efa_av *av, struct efa_conn *conn,
				 bool release_from_implicit_av);

void efa_conn_release_util_av(struct efa_av *av, struct efa_conn *conn,
			      bool release_from_implicit_av);

void efa_conn_release(struct efa_av *av, struct efa_conn *conn,
		      bool release_from_implicit_av);

void efa_conn_release_ah_unsafe(struct efa_av *av, struct efa_conn *conn,
				bool release_from_implicit_av);

#endif