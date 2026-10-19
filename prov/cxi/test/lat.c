/*
 * Copyright (c) 2018-2021 Hewlett Packard Enterprise Development LP
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "cxip.h"
#include "cxip_test_common.h"

void *buf;

void do_tsend(size_t len)
{
	int ret;

	ret = fi_tsend(cxit_ep, buf, len, NULL, cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "ret is %d\n", ret);
}
void do_tsend_0(void) { do_tsend(0); }
void do_tsend_8(void) { do_tsend(8); }
void do_tsend_256(void) { do_tsend(256); }

void do_trecv(size_t len)
{
	int ret;

	ret = fi_trecv(cxit_ep, buf, len, NULL, FI_ADDR_UNSPEC, 0, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "ret is %d\n", ret);
}
void do_trecv_0(void) { do_trecv(0); }
void do_trecv_8(void) { do_trecv(8); }
void do_trecv_256(void) { do_trecv(256); }

void do_tsend_more(size_t len)
{
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = len,
	};
	struct fi_msg_tagged msg = {
		.msg_iov = &iov,
		.iov_count = 1,
		.addr = cxit_ep_fi_addr,
	};
	int ret;

	ret = fi_tsendmsg(cxit_ep, &msg, FI_MORE);
	cr_assert_eq(ret, FI_SUCCESS, "ret is %d\n", ret);
}
void do_tsend_more_8(void) { do_tsend_more(8); }
void do_tsend_more_256(void) { do_tsend_more(256); }

void do_trecv_more(size_t len)
{
	struct iovec iov = {
		.iov_base = buf,
		.iov_len = len,
	};
	struct fi_msg_tagged msg = {
		.msg_iov = &iov,
		.iov_count = 1,
		.addr = FI_ADDR_UNSPEC,
	};
	int ret;

	ret = fi_trecvmsg(cxit_ep, &msg, FI_MORE);
	cr_assert_eq(ret, FI_SUCCESS, "ret is %d\n", ret);
}
void do_trecv_more_8(void) { do_trecv_more(8); }
void do_trecv_more_256(void) { do_trecv_more(256); }

TestSuite(latency, .init = cxit_setup_tagged, .fini = cxit_teardown_tagged,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

enum latency_api {
	LAT_TSEND_0,
	LAT_TRECV_0,
	LAT_TSEND_8,
	LAT_TRECV_8,
	LAT_TSEND_256,
	LAT_TRECV_256,
	LAT_TSEND_MORE_8,
	LAT_TRECV_MORE_8,
	LAT_TSEND_MORE_256,
	LAT_TRECV_MORE_256,
};

static const char *const latency_api_name[] = {
	[LAT_TSEND_0] = "tsend (0-byte)",
	[LAT_TRECV_0] = "trecv (0-byte)",
	[LAT_TSEND_8] = "tsend (8-byte)",
	[LAT_TRECV_8] = "trecv (8-byte)",
	[LAT_TSEND_256] = "tsend (256-byte)",
	[LAT_TRECV_256] = "trecv (256-byte)",
	[LAT_TSEND_MORE_8] = "tsend_more (8b, no doorbell)",
	[LAT_TRECV_MORE_8] = "trecv_more (8b, no doorbell)",
	[LAT_TSEND_MORE_256] = "tsend_more (256b, no doorbell)",
	[LAT_TRECV_MORE_256] = "trecv_more (256b, no doorbell)",
};

static void (*const latency_api_func[])(void) = {
	[LAT_TSEND_0] = do_tsend_0,
	[LAT_TRECV_0] = do_trecv_0,
	[LAT_TSEND_8] = do_tsend_8,
	[LAT_TRECV_8] = do_trecv_8,
	[LAT_TSEND_256] = do_tsend_256,
	[LAT_TRECV_256] = do_trecv_256,
	[LAT_TSEND_MORE_8] = do_tsend_more_8,
	[LAT_TRECV_MORE_8] = do_trecv_more_8,
	[LAT_TSEND_MORE_256] = do_tsend_more_256,
	[LAT_TRECV_MORE_256] = do_trecv_more_256,
};

struct latency_params {
	enum latency_api api;
	bool flush_send;
};

ParameterizedTestParameters(latency, basic)
{
	size_t param_sz;

	static struct latency_params params[] = {
		{
			.api = LAT_TSEND_0,
			.flush_send = false,
		},
		{
			.api = LAT_TRECV_0,
			.flush_send = false,
		},
		{
			.api = LAT_TSEND_8,
			.flush_send = false,
		},
		{
			.api = LAT_TRECV_8,
			.flush_send = false,
		},
		{
			.api = LAT_TSEND_256,
			.flush_send = false,
		},
		{
			.api = LAT_TRECV_256,
			.flush_send = false,
		},
		{
			.api = LAT_TSEND_MORE_8,
			.flush_send = true,
		},
		{
			.api = LAT_TRECV_MORE_8,
			.flush_send = false,
		},
		{
			.api = LAT_TSEND_MORE_256,
			.flush_send = true,
		},
		{
			.api = LAT_TRECV_MORE_256,
			.flush_send = false,
		},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct latency_params, params, param_sz);
}

/* Test API latency */
ParameterizedTest(struct latency_params *params, latency, basic)
{
	int warmup = 10;
	uint64_t loops = 200;
	void (*func)(void);
	int i;
	uint64_t start;
	uint64_t end;

	cr_assert_lt((size_t)params->api, ARRAY_SIZE(latency_api_func));
	func = latency_api_func[params->api];

	buf = malloc(0x1000);
	cr_assert(buf);

	for (i = 0; i < warmup; i++)
		func();

	start = ofi_gettime_ns();

	for (i = 0; i < loops; i++)
		func();

	end = ofi_gettime_ns();

	printf("%s latency: %lu ns\n", latency_api_name[params->api],
	       (end - start) / loops);

	/* Cleanup all outstanding more sends. */
	if (params->flush_send) {
		do_tsend_0();
		sleep(1);
		fi_cq_read(cxit_tx_cq, NULL, 0);
	}

	free(buf);
}
