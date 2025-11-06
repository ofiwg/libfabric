// SPDX-License-Identifier: GPL-2.0
/* Copyright 2018-2020,2024 Hewlett Packard Enterprise Development LP */

/* User space ATU test */

#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <getopt.h>
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

static void wait_cb(void *data)
{
	printf("Got an event\n");
}

void dump_services(char *device)
{
	FILE *fp;
	char filename[256];

	snprintf(filename, sizeof(filename), "/sys/kernel/debug/cxi/%s/services", device);
	fp = fopen(filename, "r");
	if (fp) {
		char *line = NULL;
		size_t len = 0;
		ssize_t read;

		while ((read = getline(&line, &len, fp)) != -1)
		       printf("%s", line);

		free(line);
		fclose(fp);
	}
}

int test_main(char *device, int service)
{
	int lni;
	int domain;
	struct ucxi_cp *cp;
	struct ucxi_cq *transmit_cq;
	struct ucxi_cq *target_cq;
	struct ucxi_eq *eq;
	struct ucxi_ct *ct;
	int pte_number;
	int pte_index;
	int mc_pte_number1;
	int mc_pte_number2;
	int mc_pte_index1;
	int mc_pte_index2;
	struct cass_dev *dev;
	int retry;
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

	struct ucxi_wait *wait;
	unsigned int ack_counter;
	int reserved_fc;

	/* TODO: find device list. Maybe need to be an API function,
	 * looking through /sys */
	dev = open_device(device);
	if (dev == NULL) {
		fprintf(stderr, "cannot open %s\n", device);
		return 1;
	}

	if (service) {
		rc = svc_get(dev, service, &svc_desc);
		if (rc) {
			fprintf(stderr, "could not get svc: %d\n", rc);
			return 1;
		}
		dump_services(device);
	} else {
		/* Get a Service */
		rc = svc_alloc(dev, &svc_desc);
		if (rc <= 0) {
			fprintf(stderr, "cannot get a SVC. rc: %d\n", rc);
			return 1;
		}
		printf("SVC Allocated: %d\n", rc);
		svc_desc.svc_id = rc;
		dump_services(device);

		rc = set_svc_lpr(dev, svc_desc.svc_id, LPR);
		if (rc <= 0) {
			fprintf(stderr, "cannot set lnis_per_rgid rc: %d\n", rc);
			return 1;
		}
		printf("Set lnis_per_rgid success\n");

		rc = get_svc_lpr(dev, svc_desc.svc_id);
		if (rc != LPR) {
			fprintf(stderr, "cannot get lnis_per_rgid rc: %d\n", rc);
			return 1;
		}
		printf("Get lnis_per_rgid success\n");
	}

	lni = alloc_lni(dev, svc_desc.svc_id);
	if (!lni) {
		fprintf(stderr, "alloc_lni should fail %d\n", lni);
		return 1;
	}

	rc = seteuid(TEST_UID);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	/* Get an LNI */
	lni = alloc_lni(dev, svc_desc.svc_id);
	if (lni < 0) {
		fprintf(stderr, "cannot get an LNI %d\n", lni);
		return 1;
	}
	printf("LNI allocated: %d\n", lni);

	/* Get a domain */
	domain = alloc_domain(dev, lni, svc_desc.vnis[0], 40, 1024);
	if (domain < 0) {
		fprintf(stderr, "cannot get a domain\n");
		return 1;
	}
	printf("Domain allocated: %d\n", domain);

	rc = seteuid(0);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	cp = alloc_cp(dev, lni, svc_desc.vnis[0], CXI_TC_BEST_EFFORT);
	if (cp) {
		fprintf(stderr, "get CP should fail %p\n", cp);
		return 1;
	}

	rc = seteuid(TEST_UID);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	cp = alloc_cp(dev, lni, svc_desc.vnis[0], CXI_TC_BEST_EFFORT);
	if (!cp) {
		fprintf(stderr, "cannot get a CP\n");
		return 1;
	}

	/* Get a counting event */
	ct = alloc_ct(dev, lni);
	if (!ct) {
		fprintf(stderr, "cannot get counting event\n");
		return 1;
	}
	printf("Counting event allocated: %d\n", ct->ctn);

	transmit_cq = create_cq(dev, lni, true, cp->lcid);
	if (transmit_cq == NULL) {
		fprintf(stderr, "Transmit CQ cannot be created\n");
		return 1;
	}
	printf("Transmit CQ allocated: %d\n", transmit_cq->cq);

	rc = cq_get_ack_counter(dev, transmit_cq, &ack_counter);
	if (rc) {
		fprintf(stderr, "Failed to get transmit CQ ack counter %d\n",
			rc);
		return 1;
	}
	printf("Transmit CQ ack counter %u\n", ack_counter);

	target_cq = create_cq(dev, lni, false, 0);
	if (target_cq == NULL) {
		fprintf(stderr, "Transmit CQ cannot be created\n");
		return 1;
	}
	printf("Target CQ allocated: %d\n", target_cq->cq);

	rc = cq_get_ack_counter(dev, target_cq, &ack_counter);
	if (rc) {
		fprintf(stderr, "Failed to get target CQ ack counter %d\n", rc);
		return 1;
	}
	printf("Target CQ ack counter %u\n", ack_counter);

	wait = create_wait_obj(dev, lni, wait_cb);
	if (!wait) {
		fprintf(stderr, "Wait object cannot be created\n");
		return 1;
	}

	/* Invalid reserved slots. */
	eq = create_eq(dev, lni, wait, 16382);
	if (eq) {
		fprintf(stderr,
			"Event queue created with bad reserved slots\n");
		return 1;
	}

	eq = create_eq(dev, lni, wait, 0);
	if (eq == NULL) {
		fprintf(stderr, "Event queue cannot be created\n");
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, (1U << 14));
	if (rc != -EINVAL) {
		fprintf(stderr, "%d: Bad adjust_eq_reserved_fq rc: %d\n",
			__LINE__, rc);
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, -1 * (int)(1U << 14));
	if (rc != -EINVAL) {
		fprintf(stderr, "%d: Bad adjust_eq_reserved_fq rc: %d\n",
			__LINE__, rc);
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, (1U << 14) - 1);
	if (rc != -ENOSPC) {
		fprintf(stderr, "%d: Bad adjust_eq_reserved_fq rc: %d\n",
			__LINE__, rc);
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, -1 * (int)((1U << 14) - 1));
	if (rc != -EINVAL) {
		fprintf(stderr, "%d: Bad adjust_eq_reserved_fq rc: %d\n",
			__LINE__, rc);
		return 1;
	}

	reserved_fc = 10;
	rc = adjust_eq_reserved_fq(dev, eq, reserved_fc);
	if (rc != reserved_fc) {
		fprintf(stderr,
			"%d: Bad adjust_eq_reserved_fq rc: expected=%d got=%d\n",
			__LINE__, reserved_fc, rc);
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, -1);
	reserved_fc -= 1;
	if (rc != reserved_fc) {
		fprintf(stderr,
			"%d: Bad adjust_eq_reserved_fq rc: expected=%d got=%d\n",
			__LINE__, reserved_fc, rc);
		return 1;
	}

	rc = adjust_eq_reserved_fq(dev, eq, -9);
	reserved_fc = 0;
	if (rc != reserved_fc) {
		fprintf(stderr,
			"%d: Bad adjust_eq_reserved_fq rc: expected=%d got=%d\n",
			__LINE__, reserved_fc, rc);
		return 1;
	}

	/* Create PTE and map to domain, pte_idx == 0 */
	pte_number = create_pte(dev, lni, eq->eq);
	if (pte_number < 0) {
		fprintf(stderr, "PtlTE cannot be created\n");
		return 1;
	}

	pte_index = map_pte(dev, lni, pte_number, domain);
	if (pte_index < 0) {
		fprintf(stderr, "PtlTE cannot be mapped to domain\n");
		return 1;
	}

	/* Create PTE and map to multicast */
	mc_pte_number1 = create_pte(dev, lni, eq->eq);
	if (mc_pte_number1 < 0) {
		fprintf(stderr, "Multicast PtlTE cannot be created\n");
		return 1;
	}

	mc_pte_index1 = multicast_map_pte(dev, lni, mc_pte_number1, domain, 999, 15);
	if (mc_pte_index1 < 0) {
		fprintf(stderr, "Multicast PtlTE cannot be mapped to domain\n");
		return 1;
	}

	/* Attempt to re-map with a different address, should fail */
	fprintf(stderr, "Attempt illegal multicast pte re-mapping\n");
	mc_pte_index2 = multicast_map_pte(dev, lni, mc_pte_number1, domain, 999, 14);
	if (mc_pte_index2 >= 0) {
		fprintf(stderr, "Multicast PtlTE remap should have failed\n");
		return 1;
	}
	fprintf(stderr, "Attempt illegal multicast mapping passed\n");

	/* Unmap, remap with same address */
	unmap_pte(dev, mc_pte_index1);

	mc_pte_index1 = multicast_map_pte(dev, lni, mc_pte_number1, domain, 999, 15);
	if (mc_pte_index1 < 0) {
		fprintf(stderr, "Multicast PtlTE cannot be mapped after release\n");
		return 1;
	}

	/* Create new PTE, attempt to reuse address already in-use */
	mc_pte_number2 = create_pte(dev, lni, eq->eq);
	if (mc_pte_number2 < 0) {
		fprintf(stderr, "Multicast PtlTE cannot be created\n");
		return 1;
	}

	fprintf(stderr, "Attempt illegal multicast address reuse\n");
	mc_pte_index2 = multicast_map_pte(dev, lni, mc_pte_number2, domain, 999, 15);
	if (mc_pte_index2 >= 0) {
		fprintf(stderr, "Multicast address reuse should have failed\n");
		return 1;
	}
	fprintf(stderr, "Attempt illegal multicast address reuse passed\n");

	/* Set an LCID */
	rc = cxi_cq_emit_cq_lcid(&transmit_cq->cmdq, 0);
	if (rc) {
		fprintf(stderr, "Command emit failure\n");
		return 1;
	}
	cxi_cq_ring(&transmit_cq->cmdq);

	/* Wait for the adapter to consume the command */
	retry = 200;
	while (retry-- &&
	       transmit_cq->cmdq.status->rd_ptr != C_CQ_FIRST_WR_PTR + 2) {
		usleep(10000);
		__sync_synchronize();
	}

	if (transmit_cq->cmdq.status->rd_ptr != C_CQ_FIRST_WR_PTR + 2) {
		fprintf(stderr, "Command ptr hasn't moved\n");
		return 1;
	}
	/* Free resources */
	unmap_pte(dev, mc_pte_index1);
	unmap_pte(dev, pte_index);
	destroy_pte(dev, mc_pte_number2);
	destroy_pte(dev, mc_pte_number1);
	destroy_pte(dev, pte_number);
	destroy_eq(dev, eq);
	destroy_wait_obj(dev, wait);
	destroy_cq(dev, transmit_cq);
	destroy_cq(dev, target_cq);
	destroy_cp(dev, cp);
	destroy_domain(dev, domain);
	free_ct(dev, ct);
	destroy_lni(dev, lni);

	rc = seteuid(0);
	if (rc) {
		fprintf(stderr, "cannot seteuid rc:%d\n", rc);
		return 1;
	}

	svc_destroy(dev, svc_desc.svc_id);

	close_device(dev);
	dump_services(device);

	printf("good\n");

	return 0;
}

void usage(void)
{
	fprintf(stderr, "Usage: test_ucxi [options]\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -d, --device <dev>   Specify the device (default: cxi0)\n");
	fprintf(stderr, "  -s, --service <svc>  Use an existing service (default: create new)\n");
	fprintf(stderr, "  -h, --help           Show this help message\n");
}

int main(int argc, char **argv)
{
	int opt, option_index;
	int service = 0;
	char device[80] = "cxi0";

	static const char shortopts[] = "d:s:h";
	static const struct option longopts[] = {
		{"device", required_argument, NULL, 'd'},
		{"service", required_argument, NULL, 's'},
		{"help", no_argument, NULL, 'h'},
		{0, 0, 0, 0},
	};

	while ((opt = getopt_long(argc, argv, shortopts, longopts, &option_index)) != -1) {
		switch (opt) {
		case 'd':
			strncpy(device, optarg, sizeof(device) - 1);
			break;
		case 's':
			service = atoi(optarg);
			break;
		case 'h':
		default:
			usage();
			return 0;
		}
	}

	return test_main(device, service);
}
