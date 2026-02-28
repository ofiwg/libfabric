/*
 * Copyright (c) 2021-2022 Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/*
 *	fi-mr-reg-xe.c
 *
 *	This is simple libfabric memory registration test for buffers allocated
 *	via oneAPI L0 functions.
 *
 *	Register memory allocted with malloc():
 *
 *	    ./fi_xe_mr_reg -m malloc
 *
 *	Register memory allocated with zeMemAllocHost():
 *
 *	    ./fi_xe_mr_reg -m host
 *
 *	Register memory allocated with zeMemAllocDevice() on device 0
 *
 *	    ./fi_xe_mr_reg -m device -d 0
 *
 *	For more options:
 *
 *	    ./fi_xe_mr_reg -h
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_errno.h>
#include <level_zero/ze_api.h>
#include "shared.h"
#include "util.h"
#include "xe.h"
#include "ofi_ctx_pool.h"

static void set_default_options(void)
{
	options.max_ranks = 1;
	options.num_mappings = 0;
	peers = calloc(options.max_ranks, sizeof(struct business_card));
}

static void parse_mapping(char *gpu_dev_num, char *domain_name)
{
	char *s;
	char *saveptr;

	if (!gpu_dev_num || !domain_name)
		return;

	options.num_mappings = 1;
	nics[0].mapping.domain_name = domain_name;
	s = strtok_r(gpu_dev_num, ".", &saveptr);
	nics[0].mapping.gpu.dev_num = atoi(s);
	s = strtok_r(NULL, "\0", &saveptr);
	if (s)
		nics[0].mapping.gpu.subdev_num = atoi(s);
	else
		nics[0].mapping.gpu.subdev_num = -1;
}

static void print_opts()
{
	size_t len = 16;
	char *str = malloc(len);

	printf("\nOPTIONS:\n");
	printf("\tDomain: %s\n", nics[0].mapping.domain_name);
	printf("\tGPU: %d", nics[0].mapping.gpu.dev_num);
	if (nics[0].mapping.gpu.subdev_num > 0)
		printf(".%d", nics[0].mapping.gpu.subdev_num);
	printf("\n");
	printf("\tEndpoint type: %s\n", options.ep_type == FI_EP_RDM ? "RDM" :
	       "MSG");
	buf_location_str(options.loc1, str, len);
	printf("\tBuffer location: %s\n", str);
	printf("\tProvider name: %s\n", options.prov_name ? options.prov_name :
	       "default");
	printf("\tBuffer size: %zd\n", options.msg_size);
	free(str);
}

static void usage(char *prog)
{
	printf("Usage: %s [options]\n", prog);
	printf("Options:\n");
	printf("\t-D <domain_name> Open OFI domain named as <domain_name>, "
				   "default: automatic\n");
	printf("\t-d <x>[.<y>]     Use the GPU device <x>, optionally "
				   "subdevice <y>, default: 0\n");
	printf("\t-e <ep_type>     Set the endpoint type, can be 'rdm' or "
				   "'msg', default: rdm\n");
	printf("\t-m <location>    Where to allocate the buffer, can be "
				   "'malloc', 'host', 'device' or 'shared', "
				   "default: malloc\n");
	printf("\t-p <prov_name>   Use the OFI provider named as <prov_name>, "
				   "default: the first one\n");
	printf("\t-S <size>        Set the buffer size, default: 65536\n");
	printf("\t-h               Print this message\n");
}

static void parse_opts(int argc, char **argv)
{
	int op;
	char *gpu_dev_nums = NULL;
	char *domain_name = NULL;

	while ((op = getopt(argc, argv, "d:D:e:p:m:S:h")) != -1) {
		switch (op) {
		case 'd':
			gpu_dev_nums = strdup(optarg);
			break;
		case 'D':
			domain_name = strdup(optarg);
			break;
		case 'e':
			if (!strcmp(optarg, "rdm"))
				options.ep_type = FI_EP_RDM;
			else if (!strcmp(optarg, "msg"))
				options.ep_type = FI_EP_MSG;
			else
				printf("Invalid ep type %s, use default\n",
				       optarg);
			break;
		case 'p':
			options.prov_name = strdup(optarg);
			break;
		case 'm':
			if (!strcmp(optarg, "malloc"))
				options.buf_location = MALLOC;
			else if (!strcmp(optarg, "host"))
				options.buf_location = HOST;
			else if (!strcmp(optarg, "device"))
				options.buf_location = DEVICE;
			else if (!strcmp(optarg, "shared"))
				options.buf_location = SHARED;
			else
				printf("Invalid buffer location %s, use "
				       "default\n", optarg);
			break;
		case 'S':
			options.msg_size = atoi(optarg);
			break;
		default:
			usage(argv[0]);
			exit(-1);
			break;
		}
	}

	parse_mapping(gpu_dev_nums, domain_name);
}

int main(int argc, char *argv[])
{
	int i;

	set_default_options();
	parse_opts(argc, argv);

	if (options.verbose)
		print_opts();

	for (i = 0; i < options.num_mappings; i++) {
		if (options.verbose)
			printf("Init NIC %s with GPU %d<.%d>.\n",
			       nics[i].mapping.domain_name,
			       nics[i].mapping.gpu.dev_num,
			       nics[i].mapping.gpu.subdev_num);

		if(xe_init(&nics[i].mapping.gpu))
			goto err_out;

		if (options.verbose)
			show_xe_resources(&nics[i].mapping.gpu);
	}


	CHECK_ERROR(init_ofi());

	if (options.verbose)
		print_nic_info();
err_out:
	finalize_ofi();
	free_buf();
	for (i = 0; i < options.num_mappings; i++)
		xe_cleanup_gpu(&nics[i].mapping.gpu);

	xe_cleanup();
	if (peers)
		free(peers);
        return 0;
}
