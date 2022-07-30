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

/* Test basic send/recv with data */
Test(msg, pingdata)
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
	uint64_t data = 0xabcdabcdabcdabcd;

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
	ret = fi_senddata(cxit_ep, send_buf, send_len, NULL, data,
			  cxit_ep_fi_addr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == data, "Invalid RX CQE data");
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

/* Test basic inject send */
Test(msg, inject_ping)
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
	ret = fi_inject(cxit_ep, send_buf, send_len, cxit_ep_fi_addr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_inject failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_MSG | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Make sure a TX event wasn't delivered */
	ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	free(send_buf);
	free(recv_buf);
}

/* Test basic injectdata */
Test(msg, injectdata_ping)
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
	uint64_t data = 0xabcdabcdabcdabcd;

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
	ret = fi_injectdata(cxit_ep, send_buf, send_len, data,
			    cxit_ep_fi_addr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_inject failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len,
			  FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA, NULL, data, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Make sure a TX event wasn't delivered */
	ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

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

/* Test basic sendmsg/recvmsg with data */
Test(msg, msgping_wdata)
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
	uint64_t data = 0xabcdabcdabcdabcd;

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
	smsg.data = data;

	ret = fi_sendmsg(cxit_ep, &smsg, FI_REMOTE_CQ_DATA);
	cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	/* Validate RX event fields */
	cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
	cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV | FI_REMOTE_CQ_DATA),
		  "RX CQE flags mismatch");
	cr_assert(rx_cqe.len == send_len, "Invalid RX CQE length");
	cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
	cr_assert(rx_cqe.data == data, "Invalid RX CQE data");
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

/* Test basic injectmsg */
Test(msg, inject_msgping)
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

	ret = fi_sendmsg(cxit_ep, &smsg, FI_INJECT);
	cr_assert_eq(ret, FI_SUCCESS, "fi_sendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_MSG | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_MSG | FI_SEND, NULL);

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
	bool sent;
	bool recved;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	for (i = 0; i <= recv_len; i = (i ? i << 1 : 1)) {
		recved = sent = false;

		/* Post RX buffer */
		ret = fi_recv(cxit_ep, i ? recv_buf : NULL, i, NULL,
			      FI_ADDR_UNSPEC, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

		/* Send to self */
		ret = fi_send(cxit_ep, i ? send_buf : NULL, i, NULL,
			      cxit_ep_fi_addr, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

		/* Gather both events, ensure progress on both sides. */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			if (ret == 1) {
				cr_assert_eq(recved, false);
				recved = true;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1) {
				cr_assert_eq(sent, false);
				sent = true;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}
		} while (!(sent && recved));

		/* Validate RX event fields */
		cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
		cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
			  "RX CQE flags mismatch");
		cr_assert(rx_cqe.len == i, "Invalid RX CQE length");
		cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
		cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
		cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

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

/* Test send/recv sizes large to small (this exercises MR caching) */
Test(msg, sizes_desc)
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
	bool sent;
	bool recved;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	for (i = recv_len; i >= 1; i >>= 1) {
		recved = sent = false;

		/* Post RX buffer */
		ret = fi_recv(cxit_ep, recv_buf, i, NULL, FI_ADDR_UNSPEC,
			      NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

		/* Send 64 bytes to self */
		ret = fi_send(cxit_ep, send_buf, i, NULL, cxit_ep_fi_addr,
			      NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d", ret);

		/* Gather both events, ensure progress on both sides. */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			if (ret == 1) {
				cr_assert_eq(recved, false);
				recved = true;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1) {
				cr_assert_eq(sent, false);
				sent = true;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}
		} while (!(sent && recved));

		/* Validate RX event fields */
		cr_assert(rx_cqe.op_context == NULL, "RX CQE Context mismatch");
		cr_assert(rx_cqe.flags == (FI_MSG | FI_RECV),
			  "RX CQE flags mismatch");
		cr_assert(rx_cqe.len == i, "Invalid RX CQE length");
		cr_assert(rx_cqe.buf == 0, "Invalid RX CQE address");
		cr_assert(rx_cqe.data == 0, "Invalid RX CQE data");
		cr_assert(rx_cqe.tag == 0, "Invalid RX CQE tag");
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

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

/* Test software posted receives greater than hardware limits */
Test(msg, sw_max_recv, .timeout = CXIT_DEFAULT_TIMEOUT)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	fi_addr_t from;
	char *rx_mode;

	/* Test is only valid in software only matching */
	rx_mode = getenv("FI_CXI_RX_MATCH_MODE");
	if (!rx_mode || strcmp(rx_mode, "software")) {
		cr_assert(1);
		return;
	}

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	/* Only 64K buffer IDs are available */
	for (i = 0; i < 68000; i++) {
		ret = fi_recv(cxit_ep, recv_buf, recv_len, NULL,
			      FI_ADDR_UNSPEC, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);
	}

	/* Send 64 bytes to self */
	for (i = 0; i < 68000; i++) {
		ret = fi_send(cxit_ep, send_buf, send_len, NULL,
			      cxit_ep_fi_addr, NULL);
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
	}
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

#define RECV_CTX ((void *)0xabc0000000000000)
#define SEND_CTX ((void *)0xdef0000000000000)

void do_multi_recv(uint8_t *send_buf, size_t send_len,
		   uint8_t *recv_buf, size_t recv_len,
		   bool send_first, size_t sends, size_t olen)
{
	int i, j, ret;
	int err = 0;
	fi_addr_t from;
	struct fi_msg rmsg = {};
	struct fi_msg smsg = {};
	struct iovec riovec;
	struct iovec siovec;
	uint64_t rxe_flags;
	uint64_t txe_flags;
	size_t sent = 0;
	size_t recved = 0;
	size_t err_recved = 0;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cq_err_entry err_cqe = {};
	size_t recved_len = 0;
	bool dequeued = false;

	if (!sends)
		sends = recv_len / send_len;

	memset(recv_buf, 0, recv_len);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.context = RECV_CTX;

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.context = SEND_CTX;

	if (send_first) {
		for (i = 0; i < sends; i++) {
			ret = fi_sendmsg(cxit_ep, &smsg, 0);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_sendmsg failed %d", ret);
		}

		/* Progress send to ensure it arrives unexpected */
		i = 0;
		do {
			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1) {
				sent = true;
				break;
			}
			cr_assert_eq(ret, -FI_EAGAIN,
				     "send failed %d", ret);
		} while (i++ < 100000);
	}

	ret = fi_recvmsg(cxit_ep, &rmsg, FI_MULTI_RECV);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recvmsg failed %d", ret);

	if (!send_first) {
		sleep(1);
		for (i = 0; i < sends; i++) {
			ret = fi_sendmsg(cxit_ep, &smsg, 0);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_sendmsg failed %d", ret);
		}
	}

	/* Gather both events, ensure progress on both sides. */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		if (ret == 1) {
			rxe_flags = FI_MSG | FI_RECV;

			validate_multi_recv_rx_event(&rx_cqe, RECV_CTX,
						     send_len, rxe_flags,
						     0, 0);
			cr_assert(from == cxit_ep_fi_addr,
				  "Invalid source address");

			if (rx_cqe.flags & FI_MULTI_RECV) {
				cr_assert(!dequeued);
				dequeued = true;
			}

			recved_len = rx_cqe.len;

			/* Validate sent data */
			uint8_t *rbuf = rx_cqe.buf;

			for (j = 0; j < recved_len; j++) {
				cr_expect_eq(rbuf[j], send_buf[j],
					     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
					     j, send_buf[j], rbuf[j],
					     err++);
				cr_assert(err < 10);
			}
			cr_assert_eq(err, 0, "Data errors seen\n");

			recved++;
		} else if (ret == -FI_EAVAIL) {
			ret = fi_cq_readerr(cxit_rx_cq, &err_cqe, 0);
			cr_assert_eq(ret, 1);

			recved_len = err_cqe.len;
			uint8_t *rbuf = recv_buf + ((sends-1) * send_len);

			/* The truncated transfer is always the last, which
			 * dequeued the multi-recv buffer.
			 */
			rxe_flags = FI_MSG | FI_RECV;

			cr_assert(err_cqe.op_context == RECV_CTX,
				  "Error RX CQE Context mismatch");
			cr_assert((err_cqe.flags & ~FI_MULTI_RECV) == rxe_flags,
				  "Error RX CQE flags mismatch");
			cr_assert(err_cqe.len == send_len - olen,
				  "Invalid Error RX CQE length, got: %ld exp: %ld",
				  err_cqe.len, recv_len);
			cr_assert(err_cqe.buf == rbuf,
				  "Invalid Error RX CQE address (%p %p)",
				  err_cqe.buf, rbuf);
			cr_assert(err_cqe.data == 0,
				  "Invalid Error RX CQE data");
			cr_assert(err_cqe.tag == 0,
				  "Invalid Error RX CQE tag");
			cr_assert(err_cqe.olen == olen,
				  "Invalid Error RX CQE olen, got: %ld exp: %ld",
				  err_cqe.olen, olen);
			cr_assert(err_cqe.err == FI_EMSGSIZE,
				  "Invalid Error RX CQE code\n");
			cr_assert(err_cqe.prov_errno == C_RC_OK,
				  "Invalid Error RX CQE errno");
			cr_assert(err_cqe.err_data == NULL);
			cr_assert(err_cqe.err_data_size == 0);

			if (err_cqe.flags & FI_MULTI_RECV) {
				cr_assert(!dequeued);
				dequeued = true;
			}

			/* Validate sent data */
			for (j = 0; j < recved_len; j++) {
				cr_expect_eq(rbuf[j], send_buf[j],
					     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
					     j, send_buf[j], rbuf[j],
					     err++);
				cr_assert(err < 10);
			}
			cr_assert_eq(err, 0, "Data errors seen\n");

			err_recved++;
		} else {
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		}

		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		if (ret == 1) {
			txe_flags = FI_MSG | FI_SEND;
			sent++;
			validate_tx_event(&tx_cqe, txe_flags, SEND_CTX);
		} else {
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		}
	} while (sent < sends || (recved + err_recved) < sends);
}

struct msg_multi_recv_params {
	size_t send_len;
	size_t recv_len;
	bool ux;
	size_t sends;
	size_t olen;
};

#define SHORT_SEND_LEN 128
#define SHORT_SENDS 200
#define LONG_SEND_LEN 4096
#define LONG_SENDS 20
#define SHORT_OLEN (3*1024)
#define LONG_OLEN 1024

static struct msg_multi_recv_params params[] = {
#if 1
	/* expected/unexp eager */
	{.send_len = SHORT_SEND_LEN,
	 .recv_len = SHORT_SENDS * SHORT_SEND_LEN,
	 .ux = false},
	{.send_len = SHORT_SEND_LEN,
	 .recv_len = SHORT_SENDS * SHORT_SEND_LEN,
	 .ux = true},

	/* exp/unexp long */
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN,
	 .ux = false},
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN,
	 .ux = true},
#endif

#if 1
	/* exp/unexp overflow */
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN + (LONG_SEND_LEN - LONG_OLEN),
	 .ux = false,
	 .sends = LONG_SENDS+1,
	 .olen = LONG_OLEN},
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN + (LONG_SEND_LEN - LONG_OLEN),
	 .ux = true,
	 .sends = LONG_SENDS+1,
	 .olen = LONG_OLEN},
#endif

#if 1
	/* exp/unexp overflow */
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN + (LONG_SEND_LEN - SHORT_OLEN),
	 .ux = false,
	 .sends = LONG_SENDS+1,
	 .olen = SHORT_OLEN},
	{.send_len = LONG_SEND_LEN,
	 .recv_len = LONG_SENDS*LONG_SEND_LEN + (LONG_SEND_LEN - SHORT_OLEN),
	 .ux = true,
	 .sends = LONG_SENDS+1,
	 .olen = SHORT_OLEN},
#endif
};

ParameterizedTestParameters(msg, multi_recv)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct msg_multi_recv_params, params,
				   param_sz);
}

/* Test multi-recv messaging */
ParameterizedTest(struct msg_multi_recv_params *param, msg, multi_recv)
{
	void *recv_buf;
	void *send_buf;

	recv_buf = aligned_alloc(C_PAGE_SIZE, param->recv_len);
	cr_assert(recv_buf);

	send_buf = aligned_alloc(C_PAGE_SIZE, param->send_len);
	cr_assert(send_buf);

	do_multi_recv(send_buf, param->send_len, recv_buf,
		      param->recv_len, param->ux, param->sends,
		      param->olen);

	free(send_buf);
	free(recv_buf);
}

/* Test multi-recv cancel */
Test(msg, multi_recv_cancel)
{
	int i, ret;
	uint8_t *recv_buf;
	int recv_len = 0x1000;
	int recvs = 5;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cq_err_entry err_cqe;
	struct fi_msg rmsg = {};
	struct iovec riovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	/* Post RX buffer */
	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.context = NULL;

	for (i = 0; i < recvs; i++) {
		ret = fi_recvmsg(cxit_ep, &rmsg, FI_MULTI_RECV);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);
	}

	for (i = 0; i < recvs; i++) {
		ret = fi_cancel(&cxit_ep->fid, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_cancel failed %d", ret);
	}

	for (i = 0; i < recvs; i++) {
		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			if (ret == -FI_EAVAIL)
				break;

			cr_assert_eq(ret, -FI_EAGAIN,
				     "unexpected event %d", ret);
		} while (1);

		ret = fi_cq_readerr(cxit_rx_cq, &err_cqe, 0);
		cr_assert_eq(ret, 1);

		cr_assert(err_cqe.op_context == NULL,
			  "Error RX CQE Context mismatch");
		cr_assert(err_cqe.flags == (FI_MSG | FI_RECV | FI_MULTI_RECV),
			  "Error RX CQE flags mismatch");
		cr_assert(err_cqe.err == FI_ECANCELED,
			  "Invalid Error RX CQE code\n");
		cr_assert(err_cqe.prov_errno == 0,
			  "Invalid Error RX CQE errno");
	}
}

/* Test out-of-order multi-receive transaction completion */
Test(msg, multi_recv_ooo)
{
	int i, j, ret;
	int err = 0;
	fi_addr_t from;
	struct fi_msg rmsg = {};
	struct fi_msg smsg = {};
	struct iovec riovec;
	struct iovec siovec;
	uint64_t rxe_flags;
	int bytes_sent = 0;
	uint8_t *recv_buf;
	uint8_t *send_buf;
	size_t send_len = 8*1024;
	int sends = 10;
	size_t recv_len = send_len * 5 + 64 * 5;
	int sent = 0;
	int recved = 0;
	struct fi_cq_tagged_entry tx_cqe[sends];
	struct fi_cq_tagged_entry rx_cqe[sends];

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

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.context = NULL;

	ret = fi_recvmsg(cxit_ep, &rmsg, FI_MULTI_RECV);
	cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);

	sleep(1);
	for (i = 0; i < sends; i++) {
		/* Interleave long and short sends. They will complete in a
		 * different order than they were sent or received.
		 */
		if (i % 2)
			siovec.iov_len = 64;
		else
			siovec.iov_len = 8*1024;

		ret = fi_sendmsg(cxit_ep, &smsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_send failed %d",
			     ret);
	}

	for (i = 0; i < sends; i++) {
		/* Gather both events, ensure progress on both sides. */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe[recved], 1,
					     &from);
			if (ret == 1) {
				recved++;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe[sent], 1);
			if (ret == 1) {
				sent++;
			} else {
				cr_assert_eq(ret, -FI_EAGAIN,
					     "fi_cq_read unexpected value %d",
					     ret);
			}
		} while (!(sent == sends && recved == sends));
	}

	for (i = 0; i < sends; i++) {
		bytes_sent += rx_cqe[i].len;
		rxe_flags = FI_MSG | FI_RECV;
		if (bytes_sent > (recv_len - CXIP_EP_MIN_MULTI_RECV))
			rxe_flags |= FI_MULTI_RECV;

		cr_assert(rx_cqe[i].flags == rxe_flags, "CQE flags mismatch");
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

		validate_tx_event(&tx_cqe[i], FI_MSG | FI_SEND, NULL);

		/* Validate sent data */
		uint8_t *rbuf = rx_cqe[i].buf;

		for (j = 0; j < rx_cqe[i].len; j++) {
			cr_expect_eq(rbuf[j], send_buf[j],
				  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				  j, send_buf[j], recv_buf[j], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");
	}

	free(send_buf);
	free(recv_buf);
}

Test(msg, fc_multi_recv, .timeout = 30)
{
	int i, j, k, ret, tx_ret;
	uint8_t *send_bufs;
	uint8_t *send_buf;
	int send_len = 64;
	uint8_t *recv_buf;
	int recv_len = 64;
	int mrecv_msgs = 10;
	struct fi_msg rmsg = {};
	struct iovec riovec;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int nsends_concurrent = 3; /* must be less than the LE pool min. */
	int nsends = 20;
	int sends = 0;
	fi_addr_t from;

	cr_assert(!(nsends % mrecv_msgs));

	send_bufs = aligned_alloc(C_PAGE_SIZE, send_len * nsends_concurrent);
	cr_assert(send_bufs);

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len * mrecv_msgs);
	cr_assert(recv_buf);

	for (i = 0; i < nsends_concurrent - 1; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		tx_ret = fi_send(cxit_ep, send_buf, send_len, NULL,
				 cxit_ep_fi_addr, NULL);
	}

	for (i = nsends_concurrent - 1; i < nsends; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		do {
			tx_ret = fi_send(cxit_ep, send_buf, send_len, NULL,
					 cxit_ep_fi_addr, NULL);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d", ret);

			/* Just progress */
			fi_cq_read(cxit_tx_cq, NULL, 0);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, FI_SUCCESS, "fi_tsend failed %d", tx_ret);

		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d", ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_MSG | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	for (i = 0; i < nsends_concurrent - 1; i++) {
		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d", ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_MSG | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len * mrecv_msgs;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.context = NULL;

	for (i = 0; i < nsends / mrecv_msgs; i++) {
		memset(recv_buf, 0, recv_len * mrecv_msgs);
		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 0);
			assert(ret == FI_SUCCESS || ret == -FI_EAGAIN);

			ret = fi_recvmsg(cxit_ep, &rmsg, FI_MULTI_RECV);
			cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d",
				     ret);
		} while (ret == -FI_EAGAIN);

		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

		for (k = 0; k < mrecv_msgs; k++) {
			do {
				ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1,
						     &from);
			} while (ret == -FI_EAGAIN);

			cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d",
				     ret);

			validate_multi_recv_rx_event(&rx_cqe, NULL, recv_len,
						     FI_MSG | FI_RECV, 0, 0);
			cr_assert(from == cxit_ep_fi_addr,
				  "Invalid source address");
			bool last_msg = (k == (mrecv_msgs - 1));
			bool dequeued = rx_cqe.flags & FI_MULTI_RECV;

			cr_assert(!(last_msg ^ dequeued));

			for (j = 0; j < recv_len; j++) {
				cr_assert_eq(recv_buf[k * recv_len + j],
					     (uint8_t)i * mrecv_msgs + k,
					     "data mismatch, recv: %d,%d element[%d], exp=%d saw=%d\n",
					     i, k, j,
					     (uint8_t)i * mrecv_msgs + k,
					     recv_buf[k * recv_len + j]);
			}
		}
	}

	free(send_bufs);
	free(recv_buf);
}

static void test_fc_multi_recv(size_t xfer_len, bool progress_before_post)
{
	int ret;
	char *recv_buf;
	char *send_buf;
	int i;
	struct fi_msg rmsg = {};
	struct iovec riovec;
	unsigned int send_events = 0;
	unsigned int recv_events = 0;
	struct fi_cq_tagged_entry cqe;
	size_t min_mrecv = 0;
	size_t opt_len = sizeof(size_t);
	bool unlinked = false;

	/* Needs to exceed available LEs. */
	unsigned int num_xfers = 100;

	ret = fi_setopt(&cxit_ep->fid, FI_OPT_ENDPOINT, FI_OPT_MIN_MULTI_RECV,
			&min_mrecv, opt_len);
	cr_assert(ret == FI_SUCCESS);

	recv_buf = calloc(num_xfers, xfer_len);
	cr_assert(recv_buf);

	send_buf = calloc(num_xfers, xfer_len);
	cr_assert(send_buf);

	for (i = 0; i < (num_xfers * xfer_len); i++)
		send_buf[i] = (char)(rand() % 256);

	/* Fire off all the unexpected sends expect 1. Last send will be sent
	 * expectedly to verify that hardware has updates the manage local LE
	 * start and length fields accordingly.
	 */
	for (i = 0; i < num_xfers - 1; i++) {
		do {
			ret = fi_send(cxit_ep, &send_buf[i * xfer_len],
				      xfer_len, NULL, cxit_ep_fi_addr, NULL);
			if (ret == -FI_EAGAIN) {
				fi_cq_read(cxit_rx_cq, &cqe, 0);
				fi_cq_read(cxit_tx_cq, &cqe, 0);
			}
		} while (ret == -FI_EAGAIN);
		cr_assert(ret == FI_SUCCESS);
	}

	/* Progress before post will cause all ULEs to be onloaded before the
	 * append occurs.
	 */
	if (progress_before_post)
		fi_cq_read(cxit_rx_cq, &cqe, 0);

	/* Append late multi-recv buffer. */
	riovec.iov_base = recv_buf;
	riovec.iov_len = num_xfers * xfer_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = cxit_ep_fi_addr;
	rmsg.context = NULL;

	do {
		ret = fi_recvmsg(cxit_ep, &rmsg, FI_MULTI_RECV);
		if (ret == -FI_EAGAIN) {
			fi_cq_read(cxit_tx_cq, NULL, 0);
			fi_cq_read(cxit_rx_cq, NULL, 0);
		}
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for all send events. Since this test can be run with or without
	 * flow control, progressing the RX CQ may be required.
	 */
	while (send_events != (num_xfers - 1)) {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN || ret == 1);
		if (ret == 1)
			send_events++;

		/* Progress RXC. */
		fi_cq_read(cxit_rx_cq, &cqe, 0);
	}

	/* Wait for all receive events. */
	while (recv_events != (num_xfers - 1)) {
		ret = fi_cq_read(cxit_rx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN || ret == 1);
		if (ret == 1 && cqe.flags & FI_RECV)
			recv_events++;
	}

	ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	ret = fi_cq_read(cxit_rx_cq, &cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	/* Make last send expected. This ensures that hardware and/or software
	 * has correctly updated the LE start and length fields correctly.
	 */
	do {
		ret = fi_send(cxit_ep, &send_buf[(num_xfers - 1) * xfer_len],
			      xfer_len, NULL, cxit_ep_fi_addr, NULL);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for all send events. Since this test can be run with or without
	 * flow control, progressing the RX CQ may be required.
	 */
	while (send_events != num_xfers) {
		ret = fi_cq_read(cxit_tx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN || ret == 1);
		if (ret == 1)
			send_events++;

		/* Progress RXC. */
		fi_cq_read(cxit_rx_cq, &cqe, 0);
	}

	/* Process the last receive event and the multi-receive event signaling
	 * the provider is no longer using the buffer.
	 */
	while (recv_events != num_xfers && !unlinked) {
		ret = fi_cq_read(cxit_rx_cq, &cqe, 1);
		cr_assert(ret == -FI_EAGAIN || ret == 1);
		if (ret == 1) {
			if (cqe.flags & FI_RECV)
				recv_events++;
			if (cqe.flags & FI_MULTI_RECV)
				unlinked = true;
		}
	}

	/* Data integrity check. If hardware/software mismanaged the multi-recv
	 * start and/or length fields on the expected send, data will be
	 * corrupted.
	 */
	for (i = 0; i < (num_xfers * xfer_len); i++)
		cr_assert_eq(send_buf[i], recv_buf[i],
			     "Data miscompare: byte=%u", i);

	free(send_buf);
	free(recv_buf);
}

Test(msg, fc_multi_recv_rdzv, .timeout = 10)
{
	/* Transfer size needs to be large enough to trigger rendezvous. */
	test_fc_multi_recv(16384, false);
}

Test(msg, fc_multi_recv_rdzv_onload_ules, .timeout = 10)
{
	/* Transfer size needs to be large enough to trigger rendezvous. */
	test_fc_multi_recv(16384, true);
}

Test(msg, fc_no_eq_space_expected_multi_recv, .timeout = 10)
{
	test_fc_multi_recv(1, false);
}

Test(msg, fc_no_eq_space_expected_multi_recv_onload_ules, .timeout = 10)
{
	test_fc_multi_recv(1, false);
}
