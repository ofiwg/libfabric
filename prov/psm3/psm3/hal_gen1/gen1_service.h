#ifdef PSM_OPA
/*

  This file is provided under a dual BSD/GPLv2 license.  When using or
  redistributing this file, you may do so under either license.

  GPL LICENSE SUMMARY

  Copyright(c) 2015 Intel Corporation.

  This program is free software; you can redistribute it and/or modify
  it under the terms of version 2 of the GNU General Public License as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  General Public License for more details.

  Contact Information:
  Intel Corporation, www.intel.com

  BSD LICENSE

  Copyright(c) 2015 Intel Corporation.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions
  are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.
    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#ifndef PSM_HAL_GEN1_SERVICE_H
#define PSM_HAL_GEN1_SERVICE_H

/* This file contains all the lowest level routines calling into sysfs */
/* and qib driver. All other calls are based on these routines. */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif
#include <sched.h>              /* cpu_set_t and CPU_* MACROs */
#include <libgen.h>

#include "utils_user.h"
#include "gen1_types.h"
#include "gen1_common.h"
#include "psm_netutils.h"

/* HAL specific upper and lower bounds for NIC port numbers */
#define HFI_MIN_PORT 1
#define HFI_MAX_PORT 1
#ifndef HFI_NUM_PORTS_GEN1
#define HFI_NUM_PORTS_GEN1 (HFI_MAX_PORT - HFI_MIN_PORT + 1)
#endif

/* base name of path (without unit #) for qib driver */
#ifndef HFI_DEVICE_PATH_GEN1
#define HFI_DEVICE_PATH_GEN1 "/dev/hfi1"
#endif

#ifdef PSM_CUDA
#define GDR_DEVICE_PATH "/dev/hfi1_gdr"
#endif

/* The major and minor versions of driver that support non-DW multiple SDMA */
#define HFI1_USER_SWMAJOR_NON_DW_MUL_MSG_SIZE_ALLOWED 6
#define HFI1_USER_SWMINOR_NON_DW_MUL_MSG_SIZE_ALLOWED 2

/* Commands used to communicate with driver. */
enum PSMI_HFI_CMD {
    PSMI_HFI_CMD_ASSIGN_CTXT = 0,   /* allocate HFI and context */
    PSMI_HFI_CMD_CTXT_INFO,         /* find out what resources we got */
    PSMI_HFI_CMD_USER_INFO,         /* set up userspace */
    PSMI_HFI_CMD_TID_UPDATE,        /* update expected TID entries */
    PSMI_HFI_CMD_TID_FREE,          /* free expected TID entries */
    PSMI_HFI_CMD_CREDIT_UPD,        /* force an update of PIO credit */
    PSMI_HFI_CMD_RECV_CTRL,         /* control receipt of packets */
    PSMI_HFI_CMD_POLL_TYPE,         /* set the kind of polling we want */
    PSMI_HFI_CMD_ACK_EVENT,         /* ack & clear user status bits */
    PSMI_HFI_CMD_SET_PKEY,          /* set context's pkey */
    PSMI_HFI_CMD_CTXT_RESET,        /* reset context's HW send context */
    PSMI_HFI_CMD_TID_INVAL_READ,    /* read TID cache invalidations */
    PSMI_HFI_CMD_GET_VERS,          /* get the version of the user cdev */

#ifdef PSM_CUDA
    PSMI_HFI_CMD_TID_UPDATE_V2 = 28,
#endif
    PSMI_HFI_CMD_LAST,
};

/* Legacy commands used to communicate with driver using 'write' */
enum LEGACY_HFI1_CMD {
    LEGACY_HFI1_CMD_ASSIGN_CTXT     = 1,     /* allocate HFI and context */
    LEGACY_HFI1_CMD_CTXT_INFO       = 2,     /* find out what resources we got */
    LEGACY_HFI1_CMD_USER_INFO       = 3,     /* set up userspace */
    LEGACY_HFI1_CMD_TID_UPDATE      = 4,     /* update expected TID entries */
    LEGACY_HFI1_CMD_TID_FREE        = 5,     /* free expected TID entries */
    LEGACY_HFI1_CMD_CREDIT_UPD      = 6,     /* force an update of PIO credit */

    LEGACY_HFI1_CMD_RECV_CTRL       = 8,     /* control receipt of packets */
    LEGACY_HFI1_CMD_POLL_TYPE       = 9,     /* set the kind of polling we want */
    LEGACY_HFI1_CMD_ACK_EVENT       = 10,    /* ack & clear user status bits */
    LEGACY_HFI1_CMD_SET_PKEY        = 11,    /* set context's pkey */
    LEGACY_HFI1_CMD_CTXT_RESET      = 12,    /* reset context's HW send context */
    LEGACY_HFI1_CMD_TID_INVAL_READ  = 13,    /* read TID cache invalidations */
    LEGACY_HFI1_CMD_GET_VERS        = 14    /* get the version of the user cdev */
};

/* Given a unit number and port number, returns 1 if the unit and port are active.
   returns 0 if the unit and port are not active. returns -1 when an error occurred. */
int psm3_hfp_gen1_get_port_active(int, int);


/* Given the unit number, port and addr_index, */
/*  return an error, or the corresponding LID */
/* Returns an int, so -1 indicates a general error.  -2 indicates that the unit/port
   are not active.  0 indicates that the unit is valid, but no LID has been assigned. */
enum gen1_init_max_speed { GEN1_NOFILTER, GEN1_FILTER, GEN1_FINDMAX };
int psm3_gen1_get_port_lid(int, int, int, enum gen1_init_max_speed init_max_speed);

/* Given the unit number, port and addr_index, return an error, or the corresponding */
/* subnet, addr and gid.  For ethernet uses 1st IPv4 RoCE gid. */
/* For IB/OPA uses 1st valid gid */
/* Returns an int, so -1 indicates an error. */
int psm3_hfp_gen1_get_port_subnet(int unit, int port, int addr_index,
	psmi_subnet128_t *subnet, psmi_naddr128_t *addr,
	int *idx, psmi_gid128_t *gid);

/* Given a unit and port umber, return an error, or the corresponding speed in bps. */
/* Returns an int, so -1 indicates an error. 0 on success */
int psm3_hfp_gen1_get_port_speed(int unit, int port, uint64_t *speed);

/* Given the unit number, return an error, or the corresponding LMC value
   for the port */
/* Returns an int, so -1 indicates an error.  0 */
int psm3_gen1_get_port_lmc(int unit, int port);

/* Given a unit, port and SL, return an error, or the corresponding SC for the
   SL as programmed by the SM */
/* Returns an int, so -1 indicates an error. */
int psm3_gen1_get_port_sl2sc(int unit, int port, int sl);

/* Given a unit, port and SC, return an error, or the corresponding VL for the
   SC as programmed by the SM */
/* Returns an int, so -1 indicates an error. */
int psm3_gen1_get_port_sc2vl(int unit, int port, int sc);

/* Given a unit, port and VL, return an error, or the corresponding MTU for the
   VL as programmed by the SM */
/* Returns an int, so -1 indicates an error. */
int psm3_gen1_get_port_vl2mtu(int unit, int port, int vl);

/* Given a unit, port and index, return an error, or the corresponding pkey for
   the index as programmed by the SM */
/* Returns an int, so -1 indicates an error. */
int psm3_gen1_get_port_index2pkey(int unit, int port, int index);

/* Get the number of units supported by the driver.  Does not guarantee
   that a working chip has been found for each possible unit #.
   Returns -1 with errno set, or number of units >=0 (0 means none found). */
int psm3_hfp_gen1_get_num_units();

/* Given a unit number, returns 1 if any port on the unit is active.
   returns <=0 if no port on the unit is active. */
int psm3_gen1_get_unit_active(int unit, enum gen1_init_max_speed init_max_speed);

/* Given a number of units, returns 1 if any port on the units is active
   returns <= 0 if no port on any of the units is active. */
int psm3_hfp_gen1_have_active_unit(int num_units);

/* get the number of contexts from the unit id. */
int psm3_hfp_gen1_get_num_contexts(int unit);

/* Open hfi device file, return -1 on error. */
int psm3_gen1_nic_context_open_ex(int unit, int port, uint64_t open_timeout,
		     char *dev_name,size_t dev_name_len);

uint32_t psm3_gen1_check_non_dw_mul_sdma(void);

void psm3_gen1_nic_context_close(int fd);

/* psm3_gen1_get_user_major_version() returns the major version of the driver
   that should be used for this session of psm. Valid only after
   psm3_gen1_nic_context_open_ex has been called. */
uint16_t psm3_gen1_get_user_major_version(void);

/* psm3_gen1_get_user_minor_version() return the minor version of the driver */
uint16_t psm3_gen1_get_user_minor_version(void);

void psm3_gen1_set_user_version(uint32_t version);
void psm3_gen1_set_user_major_version(uint16_t major_version);

int psm3_gen1_nic_cmd_write(int fd, struct hfi1_cmd *, size_t count);

int psm3_gen1_nic_cmd_writev(int fd, const struct iovec *iov, int iovcnt);

/* psm3_gen1_get_cc_settings_bin() returns less than or equal to 0 on failure,
   returns greater than 0 on success. */
 int psm3_gen1_get_cc_settings_bin(int unit, int port, char *ccabuf, size_t len_ccabuf);
int psm3_gen1_get_cc_table_bin(int unit, int port, uint16_t **cctp);

/* We use mmap64() because we compile in both 32 and 64 bit mode,
   and we have to map physical addresses that are > 32 bits long.
   While linux implements mmap64, it doesn't have a man page,
   and isn't declared in any header file, so we declare it here ourselves. */

/* We'd like to just use -D_LARGEFILE64_SOURCE, to make off_t 64 bits and
   redirects mmap to mmap64 for us, but at least through suse10 and fc4,
   it doesn't work when the address being mapped is > 32 bits.  It chips
   off bits 32 and above.   So we stay with mmap64. */
extern void *mmap64(void *, size_t, int, int, int, __off64_t);
void *psm3_gen1_mmap64(void *, size_t, int, int, int, __off64_t);

/* Statistics maintained by the driver */
int psm3_gen1_get_stats(uint64_t *, int);
int psm3_gen1_get_stats_names(char **namep);
int psm3_gen1_get_stats_names_count(void);
const char *psm3_gen1_get_next_name(char **names);
void psm3_gen1_release_names(char *namep);
/* Counters maintained in the chip, globally, and per-prot */
int psm3_gen1_get_ctrs_unit(int unitno, uint64_t *, int);
int psm3_gen1_get_ctrs_unit_names(int unitno, char **namep);
int psm3_gen1_get_ctrs_unit_names_count(int unitno);
int psm3_gen1_get_ctrs_port(int unitno, int port, uint64_t *, int);
int psm3_gen1_get_ctrs_port_names(int unitno, char **namep);
int psm3_gen1_get_ctrs_port_names_count(int unitno);
uint64_t psm3_gen1_get_single_unitctr(int unit, const char *attr, uint64_t *s);
int psm3_gen1_get_single_portctr(int unit, int port, const char *attr, uint64_t *c);

int psm3_gen1_cmd_wait_for_packet(int fd);

#endif /* PSM_HAL_GEN1_SERVICE_H */
#endif /* PSM_OPA */
