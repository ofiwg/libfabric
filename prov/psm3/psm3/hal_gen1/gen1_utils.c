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

/* This file contains hfi service routine interface used by the low */
/* level hfi protocol code. */

#include <sys/poll.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <malloc.h>
#include <time.h>

#include "gen1_user.h"

/* touch the pages, with a 32 bit read */
void psm3_gen1_touch_mmap(void *m, size_t bytes)
{
	volatile uint32_t *b = (volatile uint32_t *)m, c;
	size_t i;		/* m is always page aligned, so pgcnt exact */
	int __hfi_pg_sz;

	/* First get the page size */
	__hfi_pg_sz = sysconf(_SC_PAGESIZE);

	_HFI_VDBG("Touch %lu mmap'ed pages starting at %p\n",
		  (unsigned long)bytes / __hfi_pg_sz, m);
	bytes /= sizeof(c);
	for (i = 0; i < bytes; i += __hfi_pg_sz / sizeof(c))
		c = b[i];
}

/* ack event bits, and clear them.  Usage is check *spi_sendbuf_status,
   pass bits you are prepared to handle to psm3_gen1_event_ack(), perform the
   appropriate actions for bits that were set, and then (if appropriate)
   check the bits again. */
int psm3_gen1_event_ack(struct _hfi_ctrl *ctrl, __u64 ackbits)
{
	struct hfi1_cmd cmd;

	cmd.type = PSMI_HFI_CMD_ACK_EVENT;
	cmd.len = 0;
	cmd.addr = ackbits;

	if (psm3_gen1_nic_cmd_write(ctrl->fd, &cmd, sizeof(cmd)) == -1) {
		if (errno != EINVAL)	/* not implemented in driver. */
			_HFI_DBG("event ack failed: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

/* Tell the driver to change the way packets can generate interrupts.

 HFI1_POLL_TYPE_URGENT: Generate interrupt only when packet sets
 HFI_KPF_INTR
 HFI1_POLL_TYPE_ANYRCV: wakeup on any rcv packet (when polled on).

 PSM: Uses TYPE_URGENT in ips protocol
*/
int psm3_gen1_poll_type(struct _hfi_ctrl *ctrl, uint16_t poll_type)
{
	struct hfi1_cmd cmd;

	cmd.type = PSMI_HFI_CMD_POLL_TYPE;
	cmd.len = 0;
	cmd.addr = (uint64_t) poll_type;

	if (psm3_gen1_nic_cmd_write(ctrl->fd, &cmd, sizeof(cmd)) == -1) {
		if (errno != EINVAL)	/* not implemented in driver */
			_HFI_INFO("poll type failed: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

/* set the send context pkey to check BTH pkey in each packet.
   driver should check its pkey table to see if it can find
   this pkey, if not, driver should return error. */
int psm3_gen1_set_pkey(struct _hfi_ctrl *ctrl, uint16_t pkey)
{
	struct hfi1_cmd cmd;
	struct hfi1_base_info tbinfo;

	cmd.type = PSMI_HFI_CMD_SET_PKEY;
	cmd.len = 0;
	cmd.addr = (uint64_t) pkey;

	_HFI_VDBG("Setting context pkey to 0x%04x.\n", pkey);
	if (psm3_gen1_nic_cmd_write(ctrl->fd, &cmd, sizeof(cmd)) == -1) {
		_HFI_INFO("Setting context pkey to 0x%04x failed: %s\n",
			  pkey, strerror(errno));
		return -1;
	} else {
		_HFI_VDBG("Successfully set context pkey to 0x%04x.\n", pkey);
	}

        if (getenv("PSM3_SELINUX")) {
		/*
		 * If SELinux is in use the kernel may have changed our JKey based on
		 * what we supply for the PKey so go ahead and interrogate the user info
		 * again and update our saved copy. In the future there may be a new
		 * IOCTL to get the JKey only. For now, this temporary workaround works.
		 */
		cmd.type = PSMI_HFI_CMD_USER_INFO;
		cmd.len = sizeof(tbinfo);
		cmd.addr = (uint64_t) &tbinfo;

		if (psm3_gen1_nic_cmd_write(ctrl->fd, &cmd, sizeof(cmd)) == -1) {
			_HFI_VDBG("BASE_INFO command failed in setpkey: %s\n",
				  strerror(errno));
			return -1;
		}
		_HFI_VDBG("PSM3_SELINUX is set, updating jkey to 0x%04x\n", tbinfo.jkey);
		ctrl->base_info.jkey = tbinfo.jkey;
	}
	return 0;
}

/* Tell the driver to reset the send context. if the send context
   if halted, reset it, if not, return error back to caller.
   After context reset, the credit return should be reset to
   zero by a hardware credit return DMA.
   Driver will return ENOLCK if the reset is timeout, in this
   case PSM needs to re-call again. */
int psm3_gen1_nic_reset_context(struct _hfi_ctrl *ctrl)
{
	struct hfi1_cmd cmd;

	cmd.type = PSMI_HFI_CMD_CTXT_RESET;
	cmd.len = 0;
	cmd.addr = 0;

retry:
	if (psm3_gen1_nic_cmd_write(ctrl->fd, &cmd, sizeof(cmd)) == -1) {
		if (errno == ENOLCK)
			goto retry;

		if (errno != EINVAL)
			_HFI_INFO("reset ctxt failed: %s\n", strerror(errno));
		return -1;
	}
	return 0;
}

/* wait for a received packet for our context
   This allows us to not busy wait, if nothing has happened for a
   while, which allows better measurements of cpu utilization, and
   in some cases, slightly better performance.  Called where we would
   otherwise call sched_yield().  It is not guaranteed that a packet
   has arrived, so the normal checking loop(s) should be done.

   PSM: not used as is, PSM has it's own use of polling for interrupt-only
   packets (sets psm3_gen1_poll_type to TYPE_URGENT) */
int psm3_gen1_wait_for_packet(struct _hfi_ctrl *ctrl)
{
	return psm3_gen1_cmd_wait_for_packet(ctrl->fd);
}

const char *psm3_gen1_get_next_name(char **names)
{
	char *p, *start;

	p = start = *names;
	while (*p != '\0' && *p != '\n') {
		p++;
	}
	if (*p == '\n') {
		*p = '\0';
		p++;
		*names = p;
		return start;
	} else
		return NULL;
}

void psm3_gen1_release_names(char *namep)
{
	/* names are allocated when hfi_hfifs_read() is called. Allocation
	 * for names is done only once at init time. Should we eventually
	 * have an "stats_type_unregister" type of routine to explicitly
	 * deallocate memory and free resources ?
	 */
#if 0
	if (namep != NULL)
		psm3_hfifs_free(namep);
#endif
}

/* These have been fixed to read the values, but they are not
 * compatible with the hfi driver, they return new info with
 * the qib driver
 */
static int psm3_gen1_count_names(const char *namep)
{
	int n = 0;
	while (*namep != '\0') {
		if (*namep == '\n')
			n++;
		namep++;
	}
	return n;
}

static int psm3_gen1_lookup_stat(const char *attr, char *namep, uint64_t *stats,
		    uint64_t *s)
{
	const char *p;
	int i, ret = -1, len = strlen(attr);
	int nelem = psm3_gen1_count_names(namep);

	for (i = 0; i < nelem; i++) {
		p = psm3_gen1_get_next_name(&namep);
		if (p == NULL)
			break;
		if (strncasecmp(p, attr, len + 1) == 0) {
			ret = i;
			*s = stats[i];
		}
	}
	return ret;
}

int psm3_gen1_get_single_portctr(int unit, int port, const char *attr, uint64_t *s)
{
	int nelem, n = 0, ret = -1;
	char *namep = NULL;
	uint64_t *stats = NULL;

	nelem = psm3_gen1_get_ctrs_port_names(unit, &namep);
	if (nelem == -1 || namep == NULL)
		goto bail;
	stats = calloc(nelem, sizeof(uint64_t));
	if (stats == NULL)
		goto bail;
	n = psm3_gen1_get_ctrs_port(unit, port, stats, nelem);
	if (n != nelem)
		goto bail;
	ret = psm3_gen1_lookup_stat(attr, namep, stats, s);
bail:
	if (namep != NULL)
		psm3_hfifs_free(namep);
	if (stats != NULL)
		free(stats);
	return ret;
}

int psm3_gen1_get_stats_names_count()
{
	char *namep;
	int c;

	c = psm3_gen1_get_stats_names(&namep);
	psm3_hfifs_free(namep);
	return c;
}

int psm3_gen1_get_ctrs_unit_names_count(int unitno)
{
	char *namep;
	int c;

	c = psm3_gen1_get_ctrs_unit_names(unitno, &namep);
	psm3_hfifs_free(namep);
	return c;
}

int psm3_gen1_get_ctrs_port_names_count(int unitno)
{
	char *namep;
	int c;

	c = psm3_gen1_get_ctrs_port_names(unitno, &namep);
	psm3_hfifs_free(namep);
	return c;
}

/* These have been fixed to read the values, but they are not
 * compatible with the hfi driver, they return new info with
 * the qib driver
 */
int psm3_gen1_get_ctrs_unit_names(int unitno, char **namep)
{
	int i;
	i = psm3_hfifs_unit_read(unitno, "counter_names", namep);
	if (i < 0)
		return -1;
	else
		return psm3_gen1_count_names(*namep);
}

int psm3_gen1_get_ctrs_unit(int unitno, uint64_t *c, int nelem)
{
	int i;
	i = psm3_hfifs_unit_rd(unitno, "counters", c, nelem * sizeof(*c));
	if (i < 0)
		return -1;
	else
		return i / sizeof(*c);
}

int psm3_gen1_get_ctrs_port_names(int unitno, char **namep)
{
	int i;
	i = psm3_hfifs_unit_read(unitno, "portcounter_names", namep);
	if (i < 0)
		return -1;
	else
		return psm3_gen1_count_names(*namep);
}

int psm3_gen1_get_ctrs_port(int unitno, int port, uint64_t *c, int nelem)
{
	int i;
	char buf[32];
	snprintf(buf, sizeof(buf), "port%dcounters", port);
	i = psm3_hfifs_unit_rd(unitno, buf, c, nelem * sizeof(*c));
	if (i < 0)
		return -1;
	else
		return i / sizeof(*c);
}

int psm3_gen1_get_stats_names(char **namep)
{
	int i;
	i = psm3_hfifs_read("driver_stats_names", namep);
	if (i < 0)
		return -1;
	else
		return psm3_gen1_count_names(*namep);
}

int psm3_gen1_get_stats(uint64_t *s, int nelem)
{
	int i;
	i = psm3_hfifs_rd("driver_stats", s, nelem * sizeof(*s));
	if (i < 0)
		return -1;
	else
		return i / sizeof(*s);
}
#endif /* PSM_OPA */
