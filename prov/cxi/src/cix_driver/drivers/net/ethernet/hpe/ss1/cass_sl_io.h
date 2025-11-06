/* SPDX-License-Identifier: GPL-2.0 */
/* Copyright 2025 Hewlett Packard Enterprise Development LP */

#ifndef _CASS_SL_IO_H_
#define _CASS_SL_IO_H_

int cass_sl_uc_read8(void *uc_accessor, u32 offset, u32 page, u8 *data);
int cass_sl_uc_write8(void *uc_accessor, u8 page, u8 addr, u8 data);
u64 cass_sl_read64(void *pci_accessor, long addr);
void cass_sl_write64(void *pci_accessor, long addr, u64 data64);
int cass_sl_sbus_rd(void *sbus_accessor, u32 sbus_addr, u8 reg_addr, u32 *rd_data);
int cass_sl_sbus_wr(void *sbus_accessor, u32 sbus_addr, u8 reg_addr, u32 req_data);
int cass_sl_sbus_cmd(void *sbus_accessor, int ring, u32 req_data, u8 reg_addr,
	u8 dev_addr, u8 op, u32 *rsp_data, u8 *result_code,
	u8 *overrun, u8 *error, int timeout_ms, u32 flags);
int cass_sl_pmi_rd(void *pmi_accessor, u8 lgrp_num, u32 addr, u16 *data);
int cass_sl_pmi_wr(void *pmi_accessor, u8 lgrp_num, u32 addr, u16 data, u16 mask);

#endif /* _CASS_SL_IO_H_ */
