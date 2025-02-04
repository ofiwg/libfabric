/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2020 Hewlett Packard Enterprise Development LP
 */

/* Notes:
 *
 * This test is perfunctory at present. A fuller set of tests is available:
 *
 * virtualize.sh fabtests/unit/fi_eq_test
 *
 * TODO: current implementation does not support wait states.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(eq, .init = cxit_setup_eq, .fini = cxit_teardown_eq,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic EQ creation */
Test(eq, simple)
{
	cxit_create_eq();
	cr_assert(cxit_eq != NULL);
	cxit_destroy_eq();
}

void setup_eq_wait_obj(enum fi_wait_obj wait_obj, bool pass)
{
	struct fi_eq_attr attr = {
		.size = 32,
		.flags = FI_WRITE,
		.wait_obj = wait_obj,
	};
	int ret;

	ret = fi_eq_open(cxit_fabric, &attr, &cxit_eq, NULL);
	if (pass) {
		cr_assert(ret == FI_SUCCESS,
			  "fi_eq_open wait_obj %d, unexpected err %d",
			  wait_obj, ret);
		cr_assert(cxit_eq != NULL, "cxit_eq NULL on good wait_obj");
		fi_close(&cxit_eq->fid);
	} else {
		cr_assert(ret == -FI_ENOSYS,
			  "fi_eq_open wait_obj %d, unexpected success %d",
			  wait_obj, ret);
		cr_assert(cxit_eq == NULL, "cxit_eq not NULL on bad wait_obj");
	}
}

Test(eq, good_wait_obj_none)
{
	setup_eq_wait_obj(FI_WAIT_NONE, true);
}

Test(eq, good_wait_obj_unspec)
{
	setup_eq_wait_obj(FI_WAIT_UNSPEC, true);
}

Test(eq, good_wait_obj_wait_yield)
{
	setup_eq_wait_obj(FI_WAIT_YIELD, true);
}

Test(eq, bad_wait_obj_wait_fd)
{
	setup_eq_wait_obj(FI_WAIT_FD, false);
}

Test(eq, bad_wait_obj_wait_set)
{
	setup_eq_wait_obj(FI_WAIT_SET, false);
}

TestSuite(eq_wait, .init = cxit_setup_enabled_ep_eq_yield,
	  .fini = cxit_teardown_enabled_ep, .timeout = CXIT_DEFAULT_TIMEOUT);

Test(eq_wait, timeout)
{
	uint64_t end_ms;
	uint64_t start_ms = ofi_gettime_ms();
	struct fi_eq_err_entry eqe = {};
	uint32_t event;
	int ret;
	int timeout = 200;

	ret = fi_eq_sread(cxit_eq, &event, &eqe, sizeof(eqe), timeout, 0);
	end_ms = ofi_gettime_ms();
	cr_assert(ret == -FI_EAGAIN, "Unexpected return value %s",
		  fi_strerror(-ret));
	cr_assert(end_ms >= start_ms + timeout,
		  "Timeout too short %ld ms asked for %d ms",
		  end_ms - start_ms, timeout);
}

struct eq_worker_data {
	uint32_t event;
	void *context;
	uint64_t data;
};

static void *eq_worker(void *data)
{
	struct eq_worker_data *args = (struct eq_worker_data *) data;
	struct fi_eq_err_entry eqe = {};
	uint32_t event;
	ssize_t ret;
	int timeout = 2000;

	ret = fi_eq_sread(cxit_eq, &event, &eqe, sizeof(eqe), timeout, 0);
	cr_assert(ret >= 0, "Unexpected EQ read failure %s", fi_strerror(-ret));
	cr_assert(args->event == event, "Unexpected EQ event %d", event);
	cr_assert(args->context == eqe.context, "Unexpected EQ context %p",
		  eqe.context);
	cr_assert(args->data == eqe.data, "Unexpected EQ data %ld", eqe.data);

	pthread_exit(NULL);
}

Test(eq_wait, yield_write)
{
	struct fi_eq_entry entry = {
		.context = (void *) 0x1,
		.data = 1ULL  << 63
	};
	struct eq_worker_data parms = {
		.event = FI_JOIN_COMPLETE,
		.context = entry.context,
		.data = entry.data
	};
	pthread_t eq_read_thread;
	pthread_attr_t attr = {};
	int ret;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	ret = pthread_create(&eq_read_thread, &attr, eq_worker, (void *)&parms);
	cr_assert(ret == 0, "Unexpected pthread_create error %d", ret);

	/* Make sure worker in fi_cq_sread() */
	sleep(1);

	ret = fi_eq_write(cxit_eq, FI_JOIN_COMPLETE, &entry, sizeof(entry), 0);
	cr_assert(ret == sizeof(entry), "Bad return for eq_write %d", ret);

	pthread_join(eq_read_thread, NULL);
}
