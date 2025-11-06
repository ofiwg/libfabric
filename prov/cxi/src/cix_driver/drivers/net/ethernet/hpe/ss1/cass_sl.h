/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2022,2023,2024,2025 Hewlett Packard Enterprise Development LP */

#ifndef _CASS_SL_H_
#define _CASS_SL_H_

#include <linux/hpe/sl/sl_media.h>
#include <linux/hpe/sl/sl_link.h>
#include <linux/hpe/sl/sl_llr.h>
#include <linux/hpe/sl/sl_mac.h>
#include <linux/hpe/sl/sl_lgrp.h>
#include <linux/hpe/sl/sl_ldev.h>
#include <linux/hpe/sl/sl_fec.h>

struct cass_sl_dev {
	bool                         is_initialized;

	struct sl_ops                ops;
	struct sl_uc_ops             uc_ops;
	struct sl_accessors          accessors;
	struct sl_uc_accessor        uc_accessor;

	struct sl_hw_attr            hw_attr;

	struct sl_ldev              *ldev;
	struct sl_ldev_attr          ldev_config;

	struct sl_lgrp              *lgrp;
	struct sl_lgrp_config        lgrp_config;

	struct sl_link              *link;
	struct sl_link_config        link_config;
	struct sl_link_policy        link_policy;
	u32                          link_state;
	bool                         ck_speed;
	bool                         is_fw_loaded;

	struct sl_mac               *mac;

	struct sl_llr               *llr;
	struct sl_llr_config         llr_config;
	bool                         enable_llr;
	u32                          llr_state;
	struct sl_llr_data           llr_data;

	struct sl_media_attr         media_attr;
	bool                         has_cable;

	struct list_head             intr_list;

	struct mutex                 pmi_lock;
	struct mutex                 sbus_lock;

	struct completion            step_complete;

	u32                          old_an_mode;
	u32                          old_lt_mode;

	bool                         is_canceled;

	struct {
		void       *cntrs;
		size_t      cntrs_size;
		int         dma_id;
		dma_addr_t  dma_addr;
	} fec;
};

bool cass_sl_is_pcs_aligned(struct cass_dev *cass_dev);

void cass_sl_mode_get(struct cass_dev *cass_dev, struct cxi_link_info *link_info);
void cass_sl_mode_set(struct cass_dev *cass_dev, const struct cxi_link_info *link_info);
void cass_sl_flags_get(struct cass_dev *cass_dev, u32 *flags);
void cass_sl_flags_set(struct cass_dev *cass_dev, u32 clr_flags, u32 set_flags);

void cass_sl_pml_recovery_set(struct cass_dev *cass_dev, bool set);

int  cass_sl_init(struct cass_dev *cass_dev);
int  cass_sl_connect_id_set(struct cass_dev *cass_dev, const char *connect_id);
int  cass_sl_media_config(struct cass_dev *cass_dev, void *media_attr);
int  cass_sl_link_config(struct cass_dev *cass_dev);
bool cass_sl_is_link_up(struct cass_dev *cass_dev);
int  cass_sl_link_up(struct cass_dev *cass_dev);
int  cass_sl_link_down(struct cass_dev *cass_dev);
void cass_sl_exit(struct cass_dev *cass_dev);
int  cass_sl_media_unconfig(struct cass_dev *cass_dev);
void cass_sl_link_fini(struct cass_dev *cass_dev);

#endif /* _CASS_SL_H_ */
