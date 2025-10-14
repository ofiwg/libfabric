// SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
/*
 * Platform-specific definitions of SBL_* macros.
 * Copyright 2020 Cray Inc. All rights reserved
 */
#ifndef _SBL_MB_H_
#define _SBL_MB_H_

#include "cassini_user_defs.h"
#include "sbl_csr_common.h"

/* BASE defines */
#define SBL_MB_MB_BASE  C_MB_BASE

/* OFFSET defines */
#define SBL_MB_DBG_CFG_SBUS_MASTER_OFFSET(a)     C_MB_DBG_CFG_SBUS_MASTER_OFFSET
#define SBL_MB_DBG_STS_SBUS_MASTER_OFFSET(a)     C_MB_DBG_STS_SBUS_MASTER_OFFSET

/* UPDATE defines */

/* DFLT defines */

/* SET defines */
#define SBL_MB_DBG_STS_SBUS_MASTER_OVERRUN_SET(a)                             C_SET(a, c_mb_dbg_sts_sbus_master, overrun)
#define SBL_MB_DBG_STS_SBUS_MASTER_WRITE_ERROR_SET(a)                         C_SET(a, c_mb_dbg_sts_sbus_master, write_error)
#define SBL_MB_DBG_CFG_SBUS_MASTER_DATA_SET(a)                                C_SET(a, c_mb_dbg_cfg_sbus_master, data)
#define SBL_MB_DBG_CFG_SBUS_MASTER_DATA_ADDRESS_SET(a)                        C_SET(a, c_mb_dbg_cfg_sbus_master, data_address)
#define SBL_MB_DBG_CFG_SBUS_MASTER_RECEIVER_ADDRESS_SET(a)                    C_SET(a, c_mb_dbg_cfg_sbus_master, receiver_address)
#define SBL_MB_DBG_CFG_SBUS_MASTER_COMMAND_SET(a)                             C_SET(a, c_mb_dbg_cfg_sbus_master, command)
#define SBL_MB_DBG_CFG_SBUS_MASTER_STOP_ON_WRITE_ERROR_SET(a)                 C_SET(a, c_mb_dbg_cfg_sbus_master, stop_on_write_error)
#define SBL_MB_DBG_CFG_SBUS_MASTER_STOP_ON_OVERRUN_SET(a)                     C_SET(a, c_mb_dbg_cfg_sbus_master, stop_on_overrun)
#define SBL_MB_DBG_CFG_SBUS_MASTER_MODE_SET(a)                                C_SET(a, c_mb_dbg_cfg_sbus_master, mode)
#define SBL_MB_DBG_CFG_SBUS_MASTER_RCV_DATA_VALID_SEL_SET(a)                  C_SET(a, c_mb_dbg_cfg_sbus_master, rcv_data_valid_sel)
#define SBL_MB_DBG_CFG_SBUS_MASTER_EXECUTE_SET(a)                             C_SET(a, c_mb_dbg_cfg_sbus_master, execute)
#define SBL_MB_DBG_CFG_SBUS_MASTER_SET(DATA,DATA_ADDRESS,RECEIVER_ADDRESS,COMMAND,STOP_ON_WRITE_ERROR,STOP_ON_OVERRUN,MODE,RCV_DATA_VALID_SEL,EXECUTE) \
    (SBL_MB_DBG_CFG_SBUS_MASTER_DATA_SET(DATA)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_DATA_ADDRESS_SET(DATA_ADDRESS)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_RECEIVER_ADDRESS_SET(RECEIVER_ADDRESS)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_COMMAND_SET(COMMAND)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_STOP_ON_WRITE_ERROR_SET(STOP_ON_WRITE_ERROR)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_STOP_ON_OVERRUN_SET(STOP_ON_OVERRUN)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_MODE_SET(MODE)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_RCV_DATA_VALID_SEL_SET(RCV_DATA_VALID_SEL)| \
     SBL_MB_DBG_CFG_SBUS_MASTER_EXECUTE_SET(EXECUTE))

/* GET defines */
#define SBL_MB_DBG_STS_SBUS_MASTER_DATA_GET(a)                   C_GET(a, c_mb_dbg_sts_sbus_master, data)
#define SBL_MB_DBG_STS_SBUS_MASTER_RESULT_CODE_GET(a)            C_GET(a, c_mb_dbg_sts_sbus_master, result_code)
#define SBL_MB_DBG_STS_SBUS_MASTER_OVERRUN_GET(a)                C_GET(a, c_mb_dbg_sts_sbus_master, overrun)
#define SBL_MB_DBG_STS_SBUS_MASTER_RCV_DATA_VALID_GET(a)         C_GET(a, c_mb_dbg_sts_sbus_master, rcv_data_valid)
#define SBL_MB_DBG_STS_SBUS_MASTER_WRITE_ERROR_GET(a)            C_GET(a, c_mb_dbg_sts_sbus_master, write_error)

/* Other defines */
#define SBL_MB_DBG_CFG_SBUS_MASTER_ENTRIES   1

#endif /* _SBL_MB_H_ */
