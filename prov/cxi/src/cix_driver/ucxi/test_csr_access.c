// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018 Cray Inc. All rights reserved */

/* Map CSRs from userspace, and check by reading C_MB_STS_REV. That CSR
 * is in the same location in both Cassini 1, 2 and 3.
 */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>

#include "test_ucxi_common.h"
#include "cassini_user_defs.h"

static int test_init(struct cass_dev **dev)
{
	*dev = open_device("cxi0");
	if (*dev == NULL) {
		fprintf(stderr, "cannot open cxi0\n");
		return -1;
	}

	return 0;
}

int main(int argc, char **argv)
{
	struct cass_dev *dev;
	union c_mb_sts_rev rev = {};
	int rc;

	rc = test_init(&dev);
	if (rc)
		return 1;

	rc = map_csr(dev);
	if (rc)
		return 1;

	rc = read_csr(dev, C_MB_STS_REV,
		      &rev, sizeof(rev));
	if (rc)
		return 1;

	printf("vendor: %x\n", rev.vendor_id);
	printf("device: %x\n", rev.device_id);
	printf("platform: %x\n", rev.platform);

	if (rev.vendor_id != 0x1590 && rev.vendor_id != 0x17db) {
		printf("invalid vendor\n");
		return 1;
	}

	close_device(dev);

	printf("success\n");

	return 0;
}
