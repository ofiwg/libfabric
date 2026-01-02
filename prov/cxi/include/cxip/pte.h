/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_PTE_H_
#define _CXIP_PTE_H_


#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <ofi_list.h>

/* Forward declarations */
struct cxip_cmdq;
struct cxip_cntr;
struct cxip_evtq;
struct cxip_if;
struct cxip_portals_table;

/* Macros */
#define CXIP_PTE_IGNORE_DROPS		((1 << 24) - 1)

/* Type definitions */
struct cxip_pte_map_entry {
        struct dlist_entry entry;
        struct cxil_pte_map *map;
};

struct cxip_pte {
	struct dlist_entry pte_entry;
	struct cxip_portals_table *ptable;
	struct cxil_pte *pte;
	enum c_ptlte_state state;
	struct dlist_entry map_list;

	void (*state_change_cb)(struct cxip_pte *pte,
				const union c_event *event);
	void *ctx;
};

/* Function declarations */
int cxip_pte_set_state(struct cxip_pte *pte, struct cxip_cmdq *cmdq,
		       enum c_ptlte_state new_state, uint32_t drop_count);

int cxip_pte_set_state_wait(struct cxip_pte *pte, struct cxip_cmdq *cmdq,
			    struct cxip_evtq *evtq,
			    enum c_ptlte_state new_state, uint32_t drop_count);

int cxip_pte_append(struct cxip_pte *pte, uint64_t iova, size_t len,
		    unsigned int lac, enum c_ptl_list list,
		    uint32_t buffer_id, uint64_t match_bits,
		    uint64_t ignore_bits, uint32_t match_id,
		    uint64_t min_free, uint32_t flags,
		    struct cxip_cntr *cntr, struct cxip_cmdq *cmdq,
		    bool ring);

int cxip_pte_unlink(struct cxip_pte *pte, enum c_ptl_list list,
		    int buffer_id, struct cxip_cmdq *cmdq);

int cxip_pte_map(struct cxip_pte *pte, uint64_t pid_idx, bool is_multicast);

int cxip_pte_alloc_nomap(struct cxip_portals_table *ptable, struct cxi_eq *evtq,
			 struct cxi_pt_alloc_opts *opts,
			 void (*state_change_cb)(struct cxip_pte *pte,
						 const union c_event *event),
			 void *ctx, struct cxip_pte **pte);

int cxip_pte_alloc(struct cxip_portals_table *ptable, struct cxi_eq *evtq,
		   uint64_t pid_idx, bool is_multicast,
		   struct cxi_pt_alloc_opts *opts,
		   void (*state_change_cb)(struct cxip_pte *pte,
					   const union c_event *event),
		   void *ctx, struct cxip_pte **pte);

void cxip_pte_free(struct cxip_pte *pte);

int cxip_pte_state_change(struct cxip_if *dev_if, const union c_event *event);

#endif /* _CXIP_PTE_H_ */
