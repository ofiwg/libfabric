/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/**
 * Simplest test of the framework itself.
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
#include <pmi_frmwk.h>

#define RETURN_ERROR(ret, txt) \
	if (ret) { \
		printf("FAILED %s = %d\n", txt, ret); \
		return ret; \
	}

int main(int argc, char **argv)
{
	int ret;

	ret = pmi_init_libfabric();
	RETURN_ERROR(ret, "pmi_init_libfabric()");

	ret = pmi_populate_av();
	RETURN_ERROR(ret, "pmi_populate_av()");
	printf("rank %2d of %2d address %08x\n",
		pmi_rank, pmi_numranks, pmi_nids[pmi_rank]);

	ret = pmi_enable_libfabric();
	RETURN_ERROR(ret, "pmi_enable_libfabric()");

	pmi_free_libfabric();
	return 0;
}