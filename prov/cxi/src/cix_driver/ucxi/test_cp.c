// SPDX-License-Identifier: GPL-2.0
/* Copyright 2025 Hewlett Packard Enterprise Development LP */

/* CP test - allocate and free several CP per LNIs. */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <sys/errno.h>
#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdint.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdbool.h>
#include <linux/types.h>

#include "test_ucxi_common.h"

#define LPR 3
#define TEST_UID 789
#define TEST_GID 1789
#define NUM_LNI 6
#define NUM_CP_PER_LNI 8

int main(void)
{
	int lni[NUM_LNI];
	int i;
	int j;
	struct ucxi_cp *cp[NUM_LNI][NUM_CP_PER_LNI];
	struct cass_dev *dev;
	int rc;
	struct cxi_svc_desc svc_desc = {
		.enable = 1,
		.is_system_svc = 1,
		.resource_limits = 1,
		.limits.type[CXI_RSRC_TYPE_PTE].max = 100,
		.limits.type[CXI_RSRC_TYPE_PTE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TXQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_TGQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].max = 100,
		.limits.type[CXI_RSRC_TYPE_EQ].res = 100,
		.limits.type[CXI_RSRC_TYPE_CT].max = 100,
		.limits.type[CXI_RSRC_TYPE_CT].res = 100,
		.limits.type[CXI_RSRC_TYPE_LE].max = 100,
		.limits.type[CXI_RSRC_TYPE_LE].res = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].max = 100,
		.limits.type[CXI_RSRC_TYPE_TLE].res = 100,
		.limits.type[CXI_RSRC_TYPE_AC].max = 4,
		.limits.type[CXI_RSRC_TYPE_AC].res = 4,
		.restricted_vnis = 1,
		.num_vld_vnis = 2,
		.vnis[0] = 16,
		.vnis[1] = 17,
		.vnis[2] = 18,
		.vnis[3] = 19,
		.restricted_members = 1,
		.members[0].type = CXI_SVC_MEMBER_UID,
		.members[0].svc_member.uid = TEST_UID,
		.members[1].type = CXI_SVC_MEMBER_GID,
		.members[1].svc_member.gid = TEST_GID,
		.tcs[CXI_TC_BEST_EFFORT] = true,
	};

	/* TODO: find device list. Maybe need to be an API function,
	 * looking through /sys */
	dev = open_device("cxi0");
	if (dev == NULL) {
		fprintf(stderr, "cannot open cxi0\n");
		return 1;
	}

	/* Get a Service */
	rc = svc_alloc(dev, &svc_desc);
	if (rc <= 0) {
		fprintf(stderr, "cannot get a SVC. rc: %d\n", rc);
		return 1;
	}
	printf("SVC Allocated: %d\n", rc);
	svc_desc.svc_id = rc;

	rc = set_svc_lpr(dev, svc_desc.svc_id, LPR);
	if (rc <= 0) {
		fprintf(stderr, "cannot set lnis_per_rgid rc: %d\n", rc);
		return 1;
	}
	printf("Set lnis_per_rgid success\n");

	rc = seteuid(TEST_UID);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	for (i = 0; i < NUM_LNI; i++) {
		/* Get an LNI */
		lni[i] = alloc_lni(dev, svc_desc.svc_id);
		if (lni[i] < 0) {
			fprintf(stderr, "cannot get an LNI %d\n", lni[i]);
			return 1;
		}
		printf("LNI allocated: %d\n", lni[i]);

		for (j = 0; j < NUM_CP_PER_LNI; j++) {
			cp[i][j] = alloc_cp(dev, lni[i], svc_desc.vnis[0],
					    CXI_TC_BEST_EFFORT);
			if (!cp[i][j]) {
				fprintf(stderr, "cannot get a CP\n");
				return 1;
			}
		}
	}

	/* Free resources */
	for (i = 0; i < NUM_LNI; i++) {
		for (j = 0; j < NUM_CP_PER_LNI; j++)
			destroy_cp(dev, cp[i][j]);

		destroy_lni(dev, lni[i]);
	}

	rc = seteuid(0);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	svc_destroy(dev, svc_desc.svc_id);

	close_device(dev);

	printf("good\n");

	return 0;
}
