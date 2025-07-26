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
	UNSOLICITED_WRITE_RECV,
};

/*
 * Check whether rdma read/write is enabled on the instance by querying the rdma device.
 * Return 0 if rdma read/write is enabled
 */
int main(int argc, char *argv[])
{
	struct ibv_device **device_list;
	struct ibv_context *ibv_ctx;
	struct efadv_device_attr efadv_attr = {0};
	int dev_cnt;
	int err, opt, i;
	enum rdma_op op = READ;

	while ((opt = getopt(argc, argv, "ho:")) != -1) {
		switch (opt) {
		case 'o':
			if (!strcasecmp(optarg, "read")) {
				op = READ;
			} else if (!strcasecmp(optarg, "write")) {
				op = WRITE;
			} else if (!strcasecmp(optarg, "writedata")) {
				op = UNSOLICITED_WRITE_RECV;
			} else {
				fprintf(stderr, "Unknown operation '%s. Allowed: read | write | writedata '\n", optarg);
				return EXIT_FAILURE;
			}
			break;
		case '?':
		case 'h':
		default:
			fprintf(stderr, "Usage:\n");
			FT_PRINT_OPTS_USAGE("fi_efa_rdma_checker -o <op>", "rdma operation type: read | write | writedata");
			return EXIT_FAILURE;
        }
	}

	device_list = ibv_get_device_list(&dev_cnt);
	if (dev_cnt <= 0) {
		fprintf(stderr, "No ibv device found!\n");
		return -FI_ENODEV;
	}

	for (i = 0; i < dev_cnt; i++) {
		ibv_ctx = ibv_open_device(device_list[i]);
		if (!ibv_ctx) {
			fprintf(stderr, "cannot open device %d\n", i);
			return EXIT_FAILURE;
		}

		memset(&efadv_attr, 0, sizeof(efadv_attr));
		err = efadv_query_device(ibv_ctx, (struct efadv_device_attr *)&efadv_attr, sizeof(efadv_attr));
		ibv_close_device(ibv_ctx);
		if (err) {
			if (err == EOPNOTSUPP) {
				fprintf(stdout, "Not an EFA device. Continue to check the next device.\n");
				continue;
			} else {
				fprintf(stderr, "cannot query device %d, err: %d\n", i, -err);
				goto out;
			}
		}

		fprintf(stdout, "Checking device %d: %s\n", i, ibv_get_device_name(device_list[i]));
		if (efadv_attr.max_rdma_size == 0) {
			fprintf(stderr, "rdma is not enabled \n");
			err = EXIT_FAILURE;
			goto out;
		}

		fprintf(stdout, "efa_dev_attr.max_rdma_size: %d\n", efadv_attr.max_rdma_size);

		if (op == READ)
			fprintf(stdout, "rdma read is enabled \n");

		if (op == WRITE) {
			if (efadv_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_RDMA_WRITE) {
				fprintf(stdout, "rdma write is enabled \n");
			} else {
				fprintf(stderr, "rdma write is NOT enabled \n");
				err = 1;
			}
		}

		if (op == UNSOLICITED_WRITE_RECV) {
			if (efadv_attr.device_caps & EFADV_DEVICE_ATTR_CAPS_UNSOLICITED_WRITE_RECV) {
				fprintf(stdout,
					"rdma unsolicited write recv is enabled \n");
			} else {
				fprintf(stderr, "rdma unsolicited write recv is NOT "
						"enabled \n");
				err = 1;
			}
		}

		goto out;
	}

	fprintf(stderr, "No EFA device found!\n");
	err = -FI_ENODEV;

out:
	ibv_free_device_list(device_list);
	return err;
}
