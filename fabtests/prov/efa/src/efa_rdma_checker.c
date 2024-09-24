/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */


#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <shared.h>

enum rdma_op {
	READ,
	WRITE,
};

/*
 * Check whether rdma read/write is enabled on the instance by querying the rdma device.
 * Return 0 if rdma read/write is enabled, otherwise return -1.
 */
int main(int argc, char *argv[])
{
	struct ibv_device **device_list;
	struct ibv_context *ibv_ctx;
	struct ibv_device_attr_ex ibv_dev_attr = {0};
	struct efadv_device_attr efadv_attr = {0};
	int dev_cnt;
	int err, opt;
	enum rdma_op op = READ;

	while ((opt = getopt(argc, argv, "ho:")) != -1) {
		switch (opt) {
		case 'o':
			if (!strcasecmp(optarg, "read")) {
				op = READ;
			} else if (!strcasecmp(optarg, "write")) {
				op = WRITE;
			} else {
				fprintf(stderr, "Unknown operation '%s. Allowed: read | write'\n", optarg);
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
		default:
			fprintf(stderr, "Usage:\n");
			FT_PRINT_OPTS_USAGE("fi_efa_rdma_checker -o <op>", "rdma operation type: read|write");
			return EXIT_FAILURE;
        }
	}

	device_list = ibv_get_device_list(&dev_cnt);
	if (dev_cnt <= 0) {
		fprintf(stderr, "No ibv device found!\n");
		return -ENODEV;
	}

	ibv_ctx = ibv_open_device(device_list[0]);
	if (!ibv_ctx) {
		fprintf(stderr, "cannot open device %d\n", 0);
		return EXIT_FAILURE;
	}

	err = ibv_query_device_ex(ibv_ctx, NULL, &ibv_dev_attr);
	if (!err) {
		fprintf(stdout, "ibv_dev_attr.device_cap_flags_ex: %lx\n", ibv_dev_attr.device_cap_flags_ex);
	}

	err = efadv_query_device(ibv_ctx, (struct efadv_device_attr *)&efadv_attr, sizeof(efadv_attr));
	ibv_close_device(ibv_ctx);
	if (err) {
		fprintf(stderr, "cannot query device\n");
		goto out;
	}

	if (efadv_attr.max_rdma_size == 0) {
		fprintf(stderr, "rdma is not enabled \n");
		err = EXIT_FAILURE;
		goto out;
	}
	fprintf(stdout, "rdma read is enabled \n");
	fprintf(stdout, "efa_dev_attr.max_rdma_size: %d\n", efadv_attr.max_rdma_size);

	if (op == READ)
		goto out;

	if (efadv_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE) {
		fprintf(stdout, "rdma write is enabled \n");
	} else {
		fprintf(stderr, "rdma write is NOT enabled \n");
		err = op == WRITE ? 1 : 0;
	}

out:
	ibv_free_device_list(device_list);
	return err;
}
