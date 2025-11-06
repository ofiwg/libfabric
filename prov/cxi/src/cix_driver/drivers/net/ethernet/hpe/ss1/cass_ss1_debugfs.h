/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */

#ifndef __CASS_DEBUGFS_H__
#define __CASS_DEBUGFS_H__
#include <linux/dcache.h>
#include <linux/debugfs.h>
#include <linux/netdevice.h>

#ifdef CONFIG_DEBUG_FS

void cass_port_debugfs_init(struct cass_dev *hw);
void cass_sbl_counters_debugfs_init(struct cass_dev *hw);
void domain_debugfs_create(int vni, int pid, struct cass_dev *hw,
				struct cxi_domain_priv *domain_priv);
void pt_debugfs_create(int id, struct cxi_pte_priv *pt, struct cass_dev *hw,
				struct cxi_lni_priv *lni_priv);
void lni_debugfs_create(int id, struct cass_dev *hw, struct cxi_lni_priv *lni_priv);
void eq_debugfs_create(int id, struct cxi_eq_priv *eq, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv);
void cass_dmac_debugfs_init(struct cass_dev *hw);
void cass_probe_debugfs_init(struct cass_dev *hw);
void cq_debugfs_create(int id, struct cxi_cq_priv *cq, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv);
void ct_debugfs_setup(int id, struct cxi_ct_priv *ct_priv, struct cass_dev *hw,
			struct cxi_lni_priv *lni_priv);
void ac_debugfs_create(int id, struct cass_ac *cac, struct cass_dev *hw,
	       struct cxi_lni_priv *lni_priv);
void svc_service_debugfs_create(struct cass_dev *hw);
#else
static inline void cass_port_debugfs_init(struct cass_dev *hw)
{
}
static inline void cass_sbl_counters_debugfs_init(struct cass_dev *hw)
{
}
static inline void domain_debugfs_create(int vni, int pid, struct cass_dev *hw,
					struct cxi_domain_priv *domain_priv)
{
}
static inline void pt_debugfs_create(int id, struct cxi_pte_priv *pt, struct cass_dev *hw,
					struct cxi_lni_priv *lni_priv)
{
}
static inline void lni_debugfs_create(int id, struct cass_dev *hw,
					struct cxi_lni_priv *lni_priv)
{
}
static inline void eq_debugfs_create(int id, struct cxi_eq_priv *eq, struct cass_dev *hw,
					struct cxi_lni_priv *lni_priv)
{
}
static inline void cass_dmac_debugfs_init(struct cass_dev *hw)
{
}
static inline void cass_probe_debugfs_init(struct cass_dev *hw)
{
}
static inline void cq_debugfs_create(int id, struct cxi_cq_priv *cq,
			struct cass_dev *hw, struct cxi_lni_priv *lni_priv)
{
}
static inline void ct_debugfs_setup(int id, struct cxi_ct_priv *ct_priv, struct cass_dev *hw,
				struct cxi_lni_priv *lni_priv)
{
}
static inline void ac_debugfs_create(int id, struct cass_ac *cac, struct cass_dev *hw,
					struct cxi_lni_priv *lni_priv)
{
}
static inline void svc_service_debugfs_create(struct cass_dev *hw)
{
}
#endif
#endif
