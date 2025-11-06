/* SPDX-License-Identifier: GPL-2.0-only */
/* Copyright 2023 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_RXTX_PROFILE_LIST_H_
#define _CXI_RXTX_PROFILE_LIST_H_

#include <linux/xarray.h>

#define RX_PROFILE_ID_MIN        (1)
#define RX_PROFILE_ID_MAX        (UINT_MAX)
#define RX_PROFILE_XARRAY_FLAGS  (XA_FLAGS_ALLOC1)
#define RX_PROFILE_GFP_OPTS      (GFP_KERNEL)

#define TX_PROFILE_ID_MIN        (1)
#define TX_PROFILE_ID_MAX        (UINT_MAX)
#define TX_PROFILE_XARRAY_FLAGS  (XA_FLAGS_ALLOC1)
#define TX_PROFILE_GFP_OPTS      (GFP_KERNEL)

struct cxi_rxtx_profile;

void cxi_rxtx_profile_list_init(struct cxi_rxtx_profile_list *list,
				struct xa_limit *limits,
				gfp_t flags,
				gfp_t gfp_opts);

void cxi_rxtx_profile_list_destroy(struct cxi_rxtx_profile_list *list,
				   void (*cleanup)(struct cxi_rxtx_profile *rxtx_profile,
						   void *user_args),
				   void *user_args);

void cxi_rxtx_profile_list_lock(struct cxi_rxtx_profile_list *list);
void cxi_rxtx_profile_list_unlock(struct cxi_rxtx_profile_list *list);

int cxi_rxtx_profile_list_iterate(struct cxi_rxtx_profile_list *list,
				int (*operator)(struct cxi_rxtx_profile *rxtx_profile,
						void *user_arg),
				void *user_arg);

int cxi_rxtx_profile_list_insert(struct cxi_rxtx_profile_list *list,
				 struct cxi_rxtx_profile *rxtx_profile,
				 unsigned int *rxtx_profile_id);
int cxi_rxtx_profile_list_remove(struct cxi_rxtx_profile_list *list,
				 unsigned int rxtx_profile_id);
int cxi_rxtx_profile_list_retrieve(struct cxi_rxtx_profile_list *list,
				   unsigned int id,
				   struct cxi_rxtx_profile **rxtx_profile);

int cxi_rxtx_profile_list_get_ids(struct cxi_rxtx_profile_list *list,
				  size_t max_ids,
				  unsigned int *ids,
				  size_t *num_ids);

#endif /* _CXI_RXTX_PROFILE_LIST_H_ */
