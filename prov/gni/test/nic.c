/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
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

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>

#include "gnix.h"
#include "gnix_nic.h"

#include <criterion/criterion.h>

static struct fi_info *hints;
static struct fi_info *fi;
static struct fid_fabric *fab;
static struct fid_domain *dom;

static void setup(void)
{
	int ret;

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	ret = fi_domain(fab, fi, &dom, NULL);
	cr_assert(!ret, "fi_domain");

}

static void teardown(void)
{
	int ret;

	ret = fi_close(&dom->fid);
	cr_assert(!ret, "fi_close domain");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "fi_close fabric");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
}

TestSuite(nic, .init = setup, .fini = teardown);

Test(nic, alloc_free)
{
	int i, ret;
	const int num_nics = 79;
	struct gnix_nic *nics[num_nics];

	for (i = 0; i < num_nics; i++) {
		ret = gnix_nic_alloc(container_of(dom, struct gnix_fid_domain,
						  domain_fid),
						  NULL,
						  &nics[i]);
		cr_assert_eq(ret, FI_SUCCESS, "Could not allocate nic");
	}

	for (i = 0; i < num_nics; i++) {
		ret = _gnix_nic_free(nics[i]);
		cr_assert_eq(ret, FI_SUCCESS, "Could not free nic");
	}
}

