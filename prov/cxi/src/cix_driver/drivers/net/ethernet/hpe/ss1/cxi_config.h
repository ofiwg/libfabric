/* SPDX-License-Identifier: GPL-2.0 */

/* Copyright 2025 Hewlett Packard Enterprise Development LP */

#ifndef _CXI_CONFIGFS_MGMT_H
#define _CXI_CONFIGFS_MGMT_H

#include <linux/hpe/cxi/cxi.h>
#include <linux/configfs.h>
#include <linux/slab.h>
#include <linux/workqueue.h>
#include <linux/ctype.h>

#include "cass_core.h"

#define PE_LE_MAX (C_LPE_STS_LIST_ENTRIES_ENTRIES / C_PE_COUNT)
#define CONFIGFS_DIR_LEN 64
#define CONFIGFS_RES_TYPE_LEN 16

struct cxi_cfg_group_item {
	struct config_group group;
	struct cxi_cfg_rgroup_dir *rgp_dir;
	struct cxi_cfg_rx_prof_dir *rx_prof_dir;
	struct cxi_cfg_tx_prof_dir *tx_prof_dir;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_dir {
	struct cxi_dev *cdev;
	struct cxi_rgroup *rgroup;
	struct config_group group;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup {
	struct cxi_dev *cdev;
	struct cxi_rgroup *rgroup;
	struct cxi_cfg_rgroup_attr *rgp_attr;
	struct cxi_cfg_rgroup_res *rgp_res;
	struct cxi_cfg_rgroup_state *rgp_state;
	struct cxi_cfg_rgroup_ac_entry *ac_entry;
	struct config_group group;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_ac_entry {
	struct config_group group;
	struct cxi_rgroup *rgroup;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_ac_entry_sub {
	struct config_group group;
	struct config_item item;
	struct cxi_rgroup *rgroup;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_attr {
	struct config_group group;
	struct config_item item;
	struct cxi_dev *cdev;
	struct cxi_rgroup *rgroup;
	char name[CONFIGFS_DIR_LEN];
	int c_p_id;
};

struct cxi_cfg_rgroup_res {
	struct cxi_dev *cdev;
	struct cxi_rgroup *rgroup;
	struct config_group group;
	struct cxi_svc_desc *desc;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_state {
	struct config_group group;
	struct config_item item;
	struct cxi_dev *cdev;
	struct cxi_rgroup *rgroup;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rgroup_res_type {
	struct config_group group;
	struct config_item item;
	struct config_group *parent_group;
	struct cxi_rgroup *rgroup;
	struct cxi_dev *cdev;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
	char res_type[CONFIGFS_RES_TYPE_LEN];
	int reserved;
	int max;
	int in_use;
};

struct cxi_cfg_tx_prof_dir {
	struct config_group group;
	struct cxi_dev *cdev;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof_state {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_tx_profile *tx_profile;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof_ac_entry {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_tx_profile *tx_profile;
	bool enable;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof_ac_entry_sub {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_tx_profile *tx_profile;
	bool enable;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof_attr {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_tx_profile *tx_profile;
	uint16_t match;
	uint16_t ignore;
	char tx_attr_name[CXI_VNI_NAME_LEN];
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof_tc {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_tx_profile *tx_profile;
	bool dedicated;
	bool low_latency;
	bool bulk_data;
	bool best_effort;
	bool eth;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_tx_prof {
	struct config_group group;
	struct cxi_dev *cdev;
	unsigned int id;
	struct cxi_tx_profile *tx_profile;
	struct cxi_cfg_tx_prof_ac_entry *tx_prof_ac_entry;
	struct cxi_cfg_tx_prof_attr *tx_prof_attr;
	struct cxi_cfg_tx_prof_state *tx_prof_state;
	struct cxi_cfg_tx_prof_tc *tx_prof_tc;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof_dir {
	struct config_group group;
	struct cxi_dev *cdev;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof {
	struct config_group group;
	struct cxi_dev *cdev;
	unsigned int id;
	struct cxi_rx_profile *rx_profile;
	struct cxi_cfg_rx_prof_ac_entry *rx_prof_ac_entry;
	struct cxi_cfg_rx_prof_attr *rx_prof_attr;
	struct cxi_cfg_rx_prof_state *rx_prof_state;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof_state {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_rx_profile *rx_profile;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof_ac_entry {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_rx_profile *rx_profile;
	bool enable;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof_ac_entry_sub {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_rx_profile *rx_profile;
	bool enable;
	enum cxi_ac_type ac_type;
	union cxi_ac_data ac_data;
	unsigned int id;
	char name[CONFIGFS_DIR_LEN];
};

struct cxi_cfg_rx_prof_attr {
	struct config_group group;
	struct cxi_dev *cdev;
	struct cxi_rx_profile *rx_profile;
	uint16_t match;
	uint16_t ignore;
	char rx_attr_name[CXI_VNI_NAME_LEN];
	char name[CONFIGFS_DIR_LEN];
};

int cxi_configfs_subsys_init(void);
void cxi_configfs_cleanup(struct cass_dev *hw);
int cxi_configfs_device_init(struct cass_dev *hw);
void cxi_configfs_fini(void);
void cxi_configfs_exit(void);

#endif /* _CXI_CONFIGFS_MGMT_H */
