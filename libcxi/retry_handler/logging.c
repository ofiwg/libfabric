/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */
#define _GNU_SOURCE

#include <stdio.h>
#include <search.h>
#include <time.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "rh.h"

/* Debug functions for Cassini retry handler */

/* Dump an SPT. If unordered only, only dump it if it not attached to
 * an SCT.
 *
 * Values for 'which':
 *    0 -> any SPT, 1 -> unordered only, 2 ->ordered only.
 */
static void dump_spt(struct retry_handler *rh,
		     struct spt_entry *spt,
		     int which)
{
	if (which != 0) {
		if (list_empty(&spt->list)) {
			if (which == 2)
				return;
		} else {
			if (which == 1)
				return;
		}
	}

	rh_printf(rh, LOG_WARNING, "  spt=%u DUMP START\n", spt->spt_idx);
	rh_printf(rh, LOG_WARNING,
		  "    status=%u, opcode_valid=%u, opcode=%u, to_retries=%u, nack_retries=%u\n",
		  spt->status, spt->opcode_valid, spt->opcode,
		  spt->to_retries, spt->nack_retries);
	rh_printf(rh, LOG_WARNING, "    clr_nxt=%u\n", spt->ram1.clr_nxt);
	rh_printf(rh, LOG_WARNING, "  spt=%u DUMP END\n", spt->spt_idx);
}

static void dump_spt_tw(const void *nodep, const VISIT which,
			    const int depth)
{
	struct spt_entry *spt;
	struct retry_handler *rh;

	rh = container_of(nodep, struct retry_handler, spt_tree);

	if (which != postorder && which != leaf)
		return;

	spt = *(struct spt_entry **)nodep;
	dump_spt(rh, spt, 1);
}

static void dump_sct(struct retry_handler *rh, struct sct_entry *sct)
{
	struct spt_entry *spt;

	rh_printf(rh, LOG_WARNING, "sct=%u state\n", sct->sct_idx);
	rh_printf(rh, LOG_WARNING,
		  "  close_retries=%d, cancel_spts=%d, do_force_close=%d\n",
		  sct->close_retries, sct->cancel_spts, sct->do_force_close);
	rh_printf(rh, LOG_WARNING,
		  "  num_entries=%d, spt_status_known=%d, spt_completed=%d\n",
		  sct->num_entries, sct->spt_status_known, sct->spt_completed);
	rh_printf(rh, LOG_WARNING,
		  "  clr_head=%u, sw_recycle=%u\n",
		  sct->head, sct->ram1.sw_recycle);

	list_for_each_entry(spt, &sct->spt_list, list) {
		dump_spt(rh, spt, 0);
	}
}

static void dump_sct_tw(const void *nodep, const VISIT which, const int depth)
{
	struct sct_entry *sct;
	struct retry_handler *rh;

	rh = container_of(nodep, struct retry_handler, sct_tree);
	if (which != postorder && which != leaf)
		return;

	sct = *(struct sct_entry **)nodep;
	dump_sct(rh, sct);
}

static void dump_timeout_list(const struct retry_handler *rh)
{
	struct timer_list *timer;
	struct timespec nowns;
	struct timeval now;

	clock_gettime(CLOCK_MONOTONIC_RAW, &nowns);
	TIMESPEC_TO_TIMEVAL(&now, &nowns);

	rh_printf(rh, LOG_WARNING,
		  "timeout list DUMP START. now is %lu.%06lu\n",
		  now.tv_sec, now.tv_usec);
	list_for_each_entry(timer, &rh->timeout_list.list, list)
		rh_printf(rh, LOG_WARNING, "  ts=%lu.%04lu, fn=%p\n",
			  timer->timeout_ms / 1000, timer->timeout_ms % 1000,
			  timer->func);

	rh_printf(rh, LOG_WARNING, "timeout list DUMP END\n");
}

/* On the HUP signal, dump the known SCTs and their SPTs, plus the
 * unordered SPTs.
 */
void dump_rh_state(const struct retry_handler *rh)
{
	rh_printf(rh, LOG_WARNING, "SCT STATE DUMP BEGIN\n");
	twalk(rh->sct_tree, dump_sct_tw);
	rh_printf(rh, LOG_WARNING, "SCT STATE DUMP END\n");

	rh_printf(rh, LOG_WARNING, "SPT STATE DUMP BEGIN\n");
	twalk(rh->spt_tree, dump_spt_tw);
	rh_printf(rh, LOG_WARNING, "SPT STATE DUMP END\n");

	dump_timeout_list(rh);

	rh_printf(rh, LOG_WARNING, "STATS BEGIN\n");
	rh_printf(rh, LOG_WARNING,
		  "spt_alloc=%u, spt_freed=%u, spt_released=%u, spt_free_deferred=%u\n",
		  rh->stats.spt_alloc, rh->stats.spt_freed,
		  rh->stats.spt_released, rh->stats.spt_free_deferred);
	rh_printf(rh, LOG_WARNING, "sct_alloc=%u, sct_freed=%u\n",
		  rh->stats.sct_alloc, rh->stats.sct_freed);
	rh_printf(rh, LOG_WARNING,
		  "Events nack=%u, spt_timeout=%u, sct_timeout=%u\n",
		  rh->stats.event_nack, rh->stats.event_spt_timeout,
		  rh->stats.event_sct_timeout);
	rh_printf(rh, LOG_WARNING,
		  "Events tct_timeout=%u, accel_close_complete=%u, retry_complete=%u\n",
		  rh->stats.event_tct_timeout,
		  rh->stats.event_accel_close_complete,
		  rh->stats.event_retry_complete);
	rh_printf(rh, LOG_WARNING, "STATS END\n");
}

#define GET_CSRS(name) \
	cxil_read_csr(rh->dev, name(0), &buf[name(0)], name##_SIZE)

void dump_csrs(const struct retry_handler *rh)
{
	char path[1024];
	char *buf;
	int fd;
	int rc;

	rh_printf(rh, LOG_WARNING, "dumping CSRs\n");

	sprintf(path, "/tmp/cxi-rh-debug-%u-%s-%lu-csrs.bin",
		getpid(), rh->dev->info.device_name, time(NULL));
	fd = open(path, O_CREAT | O_RDWR, 0600);
	if (fd == -1)
		return;

	rc = ftruncate(fd, C_MEMORG_CSR_SIZE);
	if (rc == -1)
		return;

	buf = mmap(NULL, C_MEMORG_CSR_SIZE, PROT_READ | PROT_WRITE,
		   MAP_SHARED, fd, 0);
	if (buf == MAP_FAILED)
		return;

	GET_CSRS(C_PCT_CFG_SPT_RAM0);
	GET_CSRS(C_PCT_CFG_SPT_RAM1);
	GET_CSRS(C_PCT_CFG_SPT_RAM2);
	if (rh->is_c1)
		GET_CSRS(C1_PCT_CFG_SPT_MISC_INFO);
	else
		GET_CSRS(C2_PCT_CFG_SPT_MISC_INFO);

	GET_CSRS(C_PCT_CFG_SCT_RAM0);
	GET_CSRS(C_PCT_CFG_SCT_RAM1);
	GET_CSRS(C_PCT_CFG_SCT_RAM2);
	GET_CSRS(C_PCT_CFG_SCT_RAM3);
	GET_CSRS(C_PCT_CFG_SCT_RAM4);
	if (rh->is_c1)
		GET_CSRS(C1_PCT_CFG_SCT_MISC_INFO);
	else
		GET_CSRS(C2_PCT_CFG_SCT_MISC_INFO);
	GET_CSRS(C_PCT_CFG_SCT_CAM);

	GET_CSRS(C_PCT_CFG_TCT_RAM);
	if (rh->is_c1)
		GET_CSRS(C1_PCT_CFG_TCT_MISC_INFO);
	else
		GET_CSRS(C2_PCT_CFG_TCT_MISC_INFO);
	GET_CSRS(C_PCT_CFG_TCT_CAM);

	GET_CSRS(C_PCT_CFG_SMT_RAM0);
	GET_CSRS(C_PCT_CFG_SMT_RAM1);

	if (rh->is_c1) {
		GET_CSRS(C1_PCT_CFG_TRS_RAM0);
		GET_CSRS(C1_PCT_CFG_TRS_RAM1);
		GET_CSRS(C1_PCT_CFG_TRS_CAM);

		GET_CSRS(C1_PCT_CFG_MST_CAM);
	} else {
		GET_CSRS(C2_PCT_CFG_TRS_RAM0);
		GET_CSRS(C2_PCT_CFG_TRS_RAM1);
		GET_CSRS(C2_PCT_CFG_TRS_CAM);

		GET_CSRS(C2_PCT_CFG_MST_CAM);
	}

	msync(buf, C_MEMORG_CSR_SIZE, MS_SYNC);

	munmap(buf, C_MEMORG_CSR_SIZE);
	close(fd);

	rh_printf(rh, LOG_WARNING, "CSRs dumped in %s\n", path);
}
