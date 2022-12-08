/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * (c) Copyright 2022-2023 Hewlett Packard Enterprise Development LP
 *
 * Validation test for the pmi_frmwk implementation.
 *
 * Launch using: srun -N4 ./test_frmwk
 *
 * This can be used as a prototype for test applications.
 *
 * This activates libfabric, populates the AV, and then frees the libfabric
 * instance.
 */

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <malloc.h>
#include <time.h>
#include <cxip.h>
#include <ofi.h>
#include "multinode_frmwk.h"

int main(int argc, char **argv)
{
	fi_addr_t *fiaddr = NULL;
	struct cxip_av *av;
	size_t size = 0;
	int i, j, ret;

	frmwk_init();
	printf("[%d|%d] initialized\n", frmwk_rank, frmwk_numranks);

	ret = frmwk_gather_nics();
	for (i = 0; i < frmwk_numranks; i++) {
		printf("[%d|%d] rank %d HSNS [", frmwk_rank, frmwk_numranks, i);
		for (j = 0; j < frmwk_nics_per_rank; j++)
			printf(" %05x", frmwk_nic_addr(i, j));
		printf("]\n");
	}

	frmwk_barrier();

	ret = frmwk_init_libfabric();
	if (frmwk_errmsg(ret, "frmwk_init_libfabric()\n"))
		return ret;
	printf("libfabric initialized\n");

	ret = frmwk_populate_av(&fiaddr, &size);
	if (frmwk_errmsg(ret, "frmwk_populate_av()\n"))
		return ret;

	av = container_of(cxit_av, struct cxip_av, av_fid.fid);
	printf("[%d|%d] fiaddrs\n", frmwk_rank, frmwk_numranks);
	for (i = 0; i < size; i++) {
		printf("[%d|%d] %ld=%05x\n", frmwk_rank, frmwk_numranks,
			fiaddr[i], av->table[i].nic);
	}

	cxit_trace_enable(true);
	CXIP_TRACE("Trace message test %d\n", 0);
	CXIP_TRACE("Trace message test %d\n", 1);
	cxit_trace_enable(false);
	CXIP_TRACE("This message should not appear\n");

	frmwk_free_libfabric();
	free(fiaddr);

	frmwk_term();
	return ret;
}