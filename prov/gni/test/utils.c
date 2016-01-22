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
#include <linux/limits.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <criterion/criterion.h>
#include "gnix_util.h"
#include "gnix.h"

struct gnix_reference_tester {
	struct gnix_reference ref_cnt;
	int destructed;
};

Test(utils, proc)
{
	int rc;

	rc = _gnix_task_is_not_app();
	cr_expect(rc == 0);

	/* *_unassigned_cpus flags don't work on tiger */
	rc = _gnix_job_enable_unassigned_cpus();
	cr_expect(rc != 0);

	rc = _gnix_job_disable_unassigned_cpus();
	cr_expect(rc != 0);

	rc = _gnix_job_enable_affinity_apply();
	cr_expect(rc == 0);

	rc = _gnix_job_disable_affinity_apply();
	cr_expect(rc == 0);

}

Test(utils, alps)
{
	int rc;
	uint8_t ptag;
	uint32_t cookie, fmas, cqs, npes, npr;
	void *addr = NULL;

	_gnix_alps_cleanup();

	rc = gnixu_get_rdma_credentials(addr, &ptag, &cookie);
	cr_expect(!rc);

	rc = _gnix_job_fma_limit(0, ptag, &fmas);
	cr_expect(!rc);

	rc = _gnix_job_cq_limit(0, ptag, &cqs);
	cr_expect(!rc);

	rc = _gnix_pes_on_node(&npes);
	cr_expect(!rc);

	rc = _gnix_nics_per_rank(&npr);
	cr_expect(!rc);

	cqs /= GNIX_CQS_PER_EP;
	cr_expect(((fmas > cqs ? cqs : fmas) / npes) == npr);

	_gnix_alps_cleanup();
}

static void test_destruct(void *obj)
{
	struct gnix_reference_tester *t = (struct gnix_reference_tester *) obj;

	t->destructed = 1;
}

Test(utils, references)
{
	int refs;
	struct gnix_reference_tester test;

	/* initialize test structure */
	_gnix_ref_init(&test.ref_cnt, 1, test_destruct);
	test.destructed = 0;

	/* check for validity */
	cr_assert(atomic_get(&test.ref_cnt.references) == 1);
	cr_assert(test.destructed == 0);

	/* increment refs and check */
	refs = _gnix_ref_get(&test);
	cr_assert(refs == 2);
	cr_assert(atomic_get(&test.ref_cnt.references) == 2);
	cr_assert(test.destructed == 0);

	/* decrement refs and check */
	refs = _gnix_ref_put(&test);
	cr_assert(refs == 1);
	cr_assert(atomic_get(&test.ref_cnt.references) == 1);
	cr_assert(test.destructed == 0);

	/* decrement and destruct, check for validity */
	refs = _gnix_ref_put(&test);
	cr_assert(refs == 0);
	cr_assert(atomic_get(&test.ref_cnt.references) == 0);
	cr_assert(test.destructed == 1);
}

