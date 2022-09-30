/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/**
 * Simplest test of the framework itself.
 *
 * This can be used as a prototype for more real test applications.
 *
 * This activates libfabric, populates the AV, and then frees the
 * libfabric instance.
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
#include <pmi_frmwk.h>

int main(int argc, char **argv)
{
	fi_addr_t *fiaddr = NULL;
	size_t size = 0;
	int i, ret;

	ret = pmi_init_libfabric();
	if (pmi_errmsg(ret, "pmi_init_libfabric()\n"))
		return ret;

	ret = pmi_populate_av(&fiaddr, &size);
	if (pmi_errmsg(ret, "pmi_populate_av()\n"))
		return ret;

	printf("%s: rank %2d of %2d fiaddrs [",
		pmi_hostname, pmi_rank, pmi_numranks);
	for (i = 0; i < size; i++)
		printf(" %ld", fiaddr[i]);
	printf("]\n");

	cxit_trace_enable(true);
	CXIP_TRACE("Trace message test %d\n", 0);
	CXIP_TRACE("Trace message test %d\n", 1);
	cxit_trace_enable(false);
	CXIP_TRACE("This message should not appear\n");

	pmi_free_libfabric();
	free(fiaddr);
	return 0;
}