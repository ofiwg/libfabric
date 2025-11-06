/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2018 Hewlett Packard Enterprise Development LP */
/* cxi-eth debug fs*/

#ifndef __CXI_ETH_DEBUGFS_H__
#define __CXI_ETH_DEBUGFS_H__
#include <linux/debugfs.h>
#include <linux/netdevice.h>

#ifdef CONFIG_DEBUG_FS
void device_debugfs_create(char *name, struct cxi_eth *dev,
			struct dentry *cxieth_debug_dir);
struct dentry *device_eth_debugfs_init(void);

#else
static inline void device_debugfs_create(char *name, struct cxi_eth *dev,
			struct dentry *cxieth_debug_dir)
{
}
static inline struct dentry *device_eth_debugfs_init(void)
{
	return NULL;
}
#endif
#endif
