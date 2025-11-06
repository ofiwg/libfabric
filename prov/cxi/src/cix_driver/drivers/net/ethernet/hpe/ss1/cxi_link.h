/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2023,2024 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_LINK_H
#define _CXI_LINK_H

struct cxi_dev;
struct cass_dev;

/**
 * struct cxi_link_info - Attributes for link configuration
 *
 * @flags: CXI private flags (CXI_ETH_PF_xxx)
 * @speed: ethtool speed setting (SPEED_xxx)
 * @autoneg: ethtool autonegotiation setting (AUTONEG_ENABLE / AUTONEG_DISABLE)
 * @port_type: ethtool; port type (PORT_FIBRE, PORT_DA, ...)
 *
 * These attributes must be configured for a link.
 * Different setting will be required for fabric or Ethernet links
 */
struct cxi_link_info {
	u32 flags;		/* or'ed CXI_ETH_PF_xxx */
	int speed;		/* SPEED_xxx */
	u32 autoneg;		/* AUTONEG_ENABLE / AUTONEG_DISABLE */
	u32 port_type;		/* PORT_FIBRE, PORT_DA, ... */
};

void cxi_link_mode_get(struct cxi_dev *cxi_dev, struct cxi_link_info *link_info);
void cxi_link_mode_set(struct cxi_dev *cxi_dev, const struct cxi_link_info *link_info);
void cxi_link_flags_get(struct cxi_dev *cxi_dev, u32 *flags);
void cxi_link_flags_set(struct cxi_dev *cxi_dev, u32 clr_flags, u32 set_flags);
void cxi_link_use_unsupported_cable(struct cxi_dev *cxi_dev, bool use);
void cxi_link_use_supported_ss200_cable(struct cxi_dev *cxi_dev, bool use);
void cxi_link_ignore_media_error(struct cxi_dev *cxi_dev, bool ignore);
void cxi_link_auto_lane_degrade(struct cxi_dev *cxi_dev, bool enable);
void cxi_link_fec_monitor(struct cxi_dev *cxi_dev, bool on);
void cxi_pml_recovery_set(struct cxi_dev *cxi_dev, bool set);

struct cxi_link_ops {
	int (*init)(struct cass_dev *hw);
	int (*link_start)(struct cass_dev *hw);
	void (*link_fini)(struct cass_dev *hw);
	void (*mode_get)(struct cass_dev *hw, struct cxi_link_info *link_info);
	void (*mode_set)(struct cass_dev *hw, const struct cxi_link_info *link_info);
	void (*flags_get)(struct cass_dev *hw, u32 *flags);
	void (*flags_set)(struct cass_dev *hw, u32 clr_flags, u32 set_flags);
	int  (*link_up)(struct cass_dev *hw);
	int  (*link_down)(struct cass_dev *hw);
	bool (*is_pcs_aligned)(struct cass_dev *hw);
	int  (*media_config)(struct cass_dev *hw, void *attr);
	int  (*media_unconfig)(struct cass_dev *hw);
	int  (*link_config)(struct cass_dev *hw);
	int  (*link_reset)(struct cass_dev *hw);
	void (*link_exit)(struct cass_dev *hw);
	void (*pml_recovery_set)(struct cass_dev *hw, bool set);
	bool (*is_link_up)(struct cass_dev *hw);
	int (*eth_name_set)(struct cass_dev *hw, const char *name);
};

void cxi_set_link_ops(struct cass_dev *hw);
#endif
