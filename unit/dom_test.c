/*
 * Copyright (c) 2013-2014 Intel Corporation.  All rights reserved.
 * Copyright (c) 2014 Cisco Systems, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AWV
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <inttypes.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "unit_common.h"

#define MAX_ADDR 256

struct fi_info *hints = NULL;
static struct fi_fabric_attr fabric_hints;

static struct fi_info *fi = NULL;
static struct fid_fabric *fabric = NULL;
static struct fid_domain **domain_vec = NULL;

/*
 * Tests:
 * - test open and close of a domain
 */

int main(int argc, char **argv)
{
	int i;
	int op, ret, num_domains = 1;

	hints = fi_allocinfo();
	if (hints == NULL)
		exit(EXIT_FAILURE);

	while ((op = getopt(argc, argv, "f:p:n:")) != -1) {
		switch (op) {
		case 'f':
			fabric_hints.name = strdup(optarg);
			break;
		case 'n':
			num_domains = atoi(optarg);
			break;
		case 'p':
			fabric_hints.prov_name = strdup(optarg);
			break;
		default:
			printf("usage: %s\n", argv[0]);
			printf("\t[-f fabric_name]\n");
			printf("\t[-p provider_name]\n");
			printf("\t[-n num domains to open]\n");
			exit(EXIT_FAILURE);
		}
	}

	hints->fabric_attr = &fabric_hints;
	hints->mode = ~0;

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	if (ret != 0) {
		printf("fi_getinfo %s\n", fi_strerror(-ret));
		goto err;
	}

	ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
	if (ret != 0) {
		printf("fi_fabric %s\n", fi_strerror(-ret));
		goto err;
	}

	domain_vec = calloc(num_domains,sizeof (struct fid_domain *));
	if (domain_vec == NULL) {
		perror("malloc");
		goto err;
	}

	for (i = 0; i < num_domains; i++) {
		ret = fi_domain(fabric, fi, &domain_vec[i], NULL);
		if (ret != FI_SUCCESS) {
			printf("fi_domain num %d %s\n", i, fi_strerror(-ret));
			goto err;
		}
	}

	for (i = 0; i < num_domains; i++) {
		ret = fi_close(&domain_vec[i]->fid);
		if (ret != FI_SUCCESS) {
			printf("Error %d closing domain num %d: %s\n", ret,
				i, fi_strerror(-ret));
			goto err;
		}
		domain_vec[i] = NULL;
	}
	free(domain_vec);
	domain_vec = NULL;

	ret = fi_close(&fabric->fid);
	if (ret != FI_SUCCESS) {
		printf("Error %d closing fabric: %s\n", ret, fi_strerror(-ret));
		exit(EXIT_FAILURE);
	}

	return ret;
err:
	if (domain_vec != NULL) {
		for (i=0;i<num_domains;i++) {
			if (domain_vec[i] != NULL) {
				ret = fi_close(&domain_vec[i]->fid);
				if (ret != FI_SUCCESS) {
					printf("Error in cleanup %d closing domain num %d: %s\n",
					       ret, i, fi_strerror(-ret));
				}
				domain_vec[i] = NULL;
			}
		}
		free(domain_vec);
		domain_vec = NULL;
	}

	if (fabric != NULL) {
		ret = fi_close(&fabric->fid);
		if (ret != FI_SUCCESS) {
			printf("Error in cleanup %d closing fabric: %s\n", ret,
			       fi_strerror(-ret));
		}
	}
	exit(EXIT_FAILURE);
}
