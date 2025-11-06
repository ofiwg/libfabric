/* SPDX-License-Identifier: LGPL-2.1-or-later */
/* Copyright 2018,2024 Hewlett Packard Enterprise Development LP */

/* User space-CXI device interaction */

#ifndef __LIBCXI_PRIV_H__
#define __LIBCXI_PRIV_H__

#include "libcxi.h"

#include <stddef.h>
#ifndef container_of
#define container_of(ptr, type, field) \
	((type *) ((char *)ptr - offsetof(type, field)))
#endif

/* Userspace CXI Device structure */
struct cxil_dev_priv {
	struct cxil_dev dev;
	int fd;

	/* CSRs mapping */
	void *mapped_csrs;
	size_t mapped_csrs_size;
};

/* Userspace CXI LNI (Logical Network Interface) structure */
struct cxil_lni_priv {
	struct cxil_dev_priv *dev;
	struct cxil_lni lni;
};

/* Userspace CXI Domain structure */
struct cxil_domain_priv {
	struct cxil_domain domain;
	struct cxil_lni_priv *lni_priv;
	unsigned int domain_hndl;
};

/* Userspace CXI communication profile structure */
struct cxil_cp_priv {
	struct cxi_cp cp;
	struct cxil_lni_priv *lni_priv;
	unsigned int cp_hndl;
};

/* Userspace CXI CMDQ structure */
struct cxil_cq {
	struct cxil_lni_priv *lni_priv;
	int cmdq_hndl;
	unsigned int size_req;
	void *cmds;
	size_t cmds_len;
	void *csr;
	size_t csr_len;
	struct cxi_cq hw;
	unsigned int flags;
};

/* Userspace CXI Event Queue structure */
struct cxil_eq {
	struct cxil_lni_priv *lni_priv;
	int evtq_hndl;
	void *evts;
	size_t evts_len;
	struct cxil_md_priv *evts_md;
	void *csr;
	size_t csr_len;
	struct cxi_eq hw;

	/* Resize attributes */
	void *resized_evts;
	size_t resized_evts_len;
	struct cxil_md_priv *resized_evts_md;
};

/* Userspace CXI counting event structure */
struct cxil_ct {
	struct cxil_lni_priv *lni_priv;
	unsigned int ct_hndl;
	void *doorbell;
	size_t doorbell_len;
	struct cxi_ct ct;
};

/* Userspace CXI Memory Descriptor private structure */
struct cxil_md_priv {
	struct cxil_lni_priv *lni_priv;
	unsigned int md_hndl;
	struct cxi_md md;
	uint32_t flags;
};

/* Userspace CXI PTE structure */
struct cxil_pte_priv {
	struct cxil_pte pte;
	struct cxil_lni_priv *lni_priv;
};

/* Userspace CXI PTE map index structure */
struct cxil_pte_map {
	struct cxil_lni_priv *lni_priv;
	unsigned int pte_index;
};

struct cxil_wait_obj {
	struct cxil_lni_priv *lni_priv;
	unsigned int wait;
	int fd;
};

/* Flags to support cxil_sbus_op_compat.
 * TODO: remove with cxil_sbus_op_compat
 */
enum {
	SBL_FLAG_DELAY_3US       =  1<<0,
	SBL_FLAG_DELAY_4US       =  1<<1,
	SBL_FLAG_DELAY_5US       =  1<<2,
	SBL_FLAG_DELAY_10US      =  1<<3,
	SBL_FLAG_DELAY_20US      =  1<<4,
	SBL_FLAG_DELAY_50US      =  1<<5,
	SBL_FLAG_DELAY_100US     =  1<<6,
	SBL_FLAG_INTERVAL_1MS    =  1<<7,
	SBL_FLAG_INTERVAL_10MS   =  1<<8,
	SBL_FLAG_INTERVAL_100MS  =  1<<9,
	SBL_FLAG_INTERVAL_1S     =  1<<10,
};

int cxil_query_devinfo(uint32_t dev_id, struct cxil_dev *dev);
int cxil_fork_init(void);
int cxil_dofork_range(void *base, size_t size);
int cxil_dontfork_range(void *base, size_t size);
int read_sysfs_var(char *fname, char *var_fmt, void *var);
int cxil_get_dev_info(struct cxil_dev *dev_in,
		      struct cxi_dev_info_use *devinfo);

#endif /* __LIBCXI_PRIV_H__ */
