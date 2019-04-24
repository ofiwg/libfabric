/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(msg, .init = cxit_setup_msg, .fini = cxit_teardown_msg,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic send/recv */
Test(msg, ping)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	ret = fi_recv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_send(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_MSG | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test basic sendv/recvv */
Test(msg, vping)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	ret = fi_recvv(cxit_ep, &riovec, NULL, 1, FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	ret = fi_sendv(cxit_ep, &siovec, NULL, 1, cxit_ep_fi_addr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_MSG | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test basic sendmsg/recvmsg */
Test(msg, msgping)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg rmsg = {};
	struct fi_msg smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.context = NULL;

	ret = fi_recvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.context = NULL;

	ret = fi_sendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_MSG | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test send/recv sizes small to large */
Test(msg, sizes)
{
	int i, j, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64*1024; /* 128k fails */
	int send_len = 64*1024;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	for (i = 1; i <= recv_len; i <<= 1) {
		/* Post RX buffer */
		ret = fi_recv(cxit_ep, recv_buf, i, NULL, FI_ADDR_UNSPEC,
			      NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

		/* Send 64 bytes to self */
		ret = fi_send(cxit_ep, send_buf, i, NULL, cxit_ep_fi_addr,
			      NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

		/* Wait for async event indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		} while (ret == -FI_EAGAIN);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		/* Validate RX event fields */
		cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
		cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
			  "RX CQE flags mismatch");
		cr_assert(rx_cqe.len == i, "Invalid RX CQE length");
		cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
		cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
		cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		/* Validate TX event fields */
		cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
		cr_assert(tx_cqe.flags == (FI_MSG | FI_SEND),
			  "TX CQE flags mismatch");
		cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
		cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
		cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
		cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

		/* Validate sent data */
		for (j = 0; j < i; j++) {
			cr_expect_eq(recv_buf[j], send_buf[j],
				     "data mismatch, element[%d], exp=%d saw=%d, size:%d err=%d\n",
				     j, send_buf[j], recv_buf[j], i, err++);
		}
	}

	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test send/recv interoperability with tagged messaging */
Test(msg, tagged_interop)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	uint8_t *trecv_buf,
		*tsend_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	trecv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(trecv_buf);
	memset(trecv_buf, 0, recv_len);

	tsend_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(tsend_buf);

	for (i = 0; i < send_len; i++)
		tsend_buf[i] = i + 0xc1;

	/* Post tagged RX buffer */
	ret = fi_trecv(cxit_ep, trecv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Post RX buffer */
	ret = fi_recv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_send(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Send 64 byte tagged message to self */
	ret = fi_tsend(cxit_ep, tsend_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_MSG | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_TAGGED | FI_RECV),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
	cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate TX event fields */
	cr_assert(tx_cqe.op_context == NULL, "TX CQE Context mismatch");
	cr_assert(tx_cqe.flags == (FI_TAGGED | FI_SEND),
		  "TX CQE flags mismatch");
	cr_assert(tx_cqe.len == 0, "Invalid TX CQE length");
	cr_assert(tx_cqe.buf == 0, "Invalid TX CQE address");
	cr_assert(tx_cqe.data == 0, "Invalid TX CQE data");
	cr_assert(tx_cqe.tag == 0, "Invalid TX CQE tag");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(trecv_buf[i], tsend_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, tsend_buf[i], trecv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(tsend_buf);
	free(trecv_buf);

	free(send_buf);
	free(recv_buf);
}
