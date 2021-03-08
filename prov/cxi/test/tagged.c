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

TestSuite(tagged, .init = cxit_setup_tagged, .fini = cxit_teardown_tagged,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

/* Test basic send/recv */
Test(tagged, ping)
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
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Try invalid lengths */
	ret = fi_tsend(cxit_ep, send_buf, cxit_fi->ep_attr->max_msg_size+1,
		       NULL, cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, -FI_EMSGSIZE, "fi_tsend failed %d", ret);

	free(send_buf);
	free(recv_buf);
}

/* Test basic zero-byte send/recv */
Test(tagged, zbr)
{
	int ret;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	fi_addr_t from;

	ret = fi_trecv(cxit_ep, NULL, 0, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	ret = fi_tsend(cxit_ep, NULL, 0, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, 0, FI_TAGGED | FI_RECV, NULL, 0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Try an unexpected send */
	ret = fi_tsend(cxit_ep, NULL, 0, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	sleep(1);

	ret = fi_trecv(cxit_ep, NULL, 0, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, 0, FI_TAGGED | FI_RECV, NULL, 0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);
}

/* Test basic rendezvous send */
Test(tagged, rdzv)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 4096;
	int send_len = 4096;
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
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Try invalid lengths */
	ret = fi_tsend(cxit_ep, send_buf, cxit_fi->ep_attr->max_msg_size+1,
		       NULL, cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, -FI_EMSGSIZE, "fi_tsend failed %d", ret);

	free(send_buf);
	free(recv_buf);
}

/* Test basic send/recv w/data */
Test(tagged, pingdata)
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
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tsenddata(cxit_ep, send_buf, send_len, NULL, data,
			   cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsenddata failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  data, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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
Test(tagged, inject_ping)
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
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tinject(cxit_ep, send_buf, send_len, cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tinject failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
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

	/* Try invalid lengths */
	ret = fi_tinject(cxit_ep, send_buf, cxit_fi->tx_attr->inject_size+1,
			 cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, -FI_EMSGSIZE, "fi_tinject failed %d", ret);

	ret = fi_tinject(cxit_ep, send_buf, 4*1024*1024,
			 cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, -FI_EMSGSIZE, "fi_tinject failed %d", ret);

	ret = fi_tinject(cxit_ep, send_buf, cxit_fi->ep_attr->max_msg_size+1,
			 cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, -FI_EMSGSIZE, "fi_tinject failed %d", ret);

	free(send_buf);
	free(recv_buf);
}

/* Test basic injectdata */
Test(tagged, injectdata_ping)
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
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

	/* Send 64 bytes to self */
	ret = fi_tinjectdata(cxit_ep, send_buf, send_len, data,
			     cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tinject failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  data, 0);
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
Test(tagged, vping)
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
	ret = fi_trecvv(cxit_ep, &riovec, NULL, 1, FI_ADDR_UNSPEC, 0, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvv failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	ret = fi_tsendv(cxit_ep, &siovec, NULL, 1, cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendv failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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
Test(tagged, msgping)
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
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
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
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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

/* Test FI_FENCE */
Test(tagged, fence)
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
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, C_PAGE_SIZE);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, C_PAGE_SIZE);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post RX buffer */
	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	ret = fi_tsendmsg(cxit_ep, &smsg, FI_FENCE);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Test rendezvous fence */
	send_len = recv_len = C_PAGE_SIZE;
	siovec.iov_len = send_len;
	riovec.iov_len = recv_len;

	for (i = 0; i < send_len; i++) {
		recv_buf[i] = 0;
		send_buf[i] = i + 0xa0;
	}

	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	ret = fi_tsendmsg(cxit_ep, &smsg, FI_FENCE);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);

		/* progress */
		fi_cq_read(cxit_tx_cq, &tx_cqe, 0);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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
Test(tagged, msgping_wdata)
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
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
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
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;
	smsg.data = data;

	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  data, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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
Test(tagged, inject_msgping)
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
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
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
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Send 64 bytes to self */
	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	ret = fi_tsendmsg(cxit_ep, &smsg, FI_INJECT);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

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

/* Test unexpected send/recv */
Test(tagged, ux_ping)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	fi_addr_t from;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Send 64 bytes to self */
	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Give some time for the message to move */
	sleep(1);

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Wait for async event indicating data has been sent */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert(ret == 1);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_assert(recv_buf[i] == send_buf[i],
			  "data mismatch, element: %d\n", i);
	}

	free(send_buf);
	free(recv_buf);
}

/* Test DIRECTED_RECV send/recv */
void directed_recv(bool logical)
{
	int i, ret;
	uint8_t *recv_buf,
		*fake_recv_buf,
		*send_buf;
	int recv_len = 0x1000;
	int send_len = 0x1000;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
#define N_FAKE_ADDRS 3
	struct cxip_addr fake_ep_addrs[N_FAKE_ADDRS+1];
	fi_addr_t from;

	if (logical)
		cxit_av_attr.flags = FI_SYMMETRIC;
	cxit_setup_enabled_ep();

	/* Create multiple logical names for the local EP address */
	for (i = 0; i < N_FAKE_ADDRS; i++) {
		fake_ep_addrs[i].nic = i + 0x41c;
		fake_ep_addrs[i].pid = i + 0x21;
	}

	ret = fi_av_insert(cxit_av, (void *)fake_ep_addrs, 3, NULL, 0, NULL);
	cr_assert(ret == 3);

	ret = fi_av_insert(cxit_av, (void *)&cxit_ep_addr, 1, NULL, 0, NULL);
	cr_assert(ret == 1);

	recv_buf = calloc(recv_len, 1);
	cr_assert(recv_buf);

	fake_recv_buf = calloc(recv_len, 1);
	cr_assert(fake_recv_buf);

	send_buf = malloc(send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	/* Post an RX buffer matching each EP name that won't be targeted */
	for (i = 0; i < N_FAKE_ADDRS; i++) {
		ret = fi_trecv(cxit_ep, fake_recv_buf, recv_len, NULL, i, 0, 0,
			       NULL);
		cr_assert(ret == FI_SUCCESS);
	}

	if (!logical) {
		/* Test bad source addr (not valid for logical matching) */
		ret = fi_trecv(cxit_ep, fake_recv_buf, recv_len, NULL, 100, 0,
			       0, NULL);
		cr_assert(ret == -FI_EINVAL);
	}

	/* Post short RX buffer matching EP name 3 */
	ret = fi_trecv(cxit_ep, recv_buf, 64, NULL, 3, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Post long RX buffer matching EP name 3 */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, 3, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Send short message to self (FI address 3)  */
	send_len = 64;

	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 3, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == 3, "Invalid source address, exp: 3 got: %lu", from);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			     i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(fake_recv_buf[i], 0,
			     "fake data corrupted, element[%d] err=%d\n",
			     i, err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Send long message to self (FI address 3)  */
	memset(recv_buf, 0, recv_len);
	send_len = 0x1000;

	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 3, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == 3, "Invalid source address, exp: 3 got: %lu", from);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			     i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(fake_recv_buf[i], 0,
			     "fake data corrupted, element[%d] err=%d\n",
			     i, err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	/* Send long UX message to self (FI address 3)  */
	memset(recv_buf, 0, recv_len);
	send_len = 0x1000;

	ret = fi_tsend(cxit_ep, send_buf, send_len, NULL, 3, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	sleep(1);

	/* Post long RX buffer matching EP name 3 */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL, 3, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);

		/* Progress */
		fi_cq_read(cxit_tx_cq, &tx_cqe, 0);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == 3, "Invalid source address, exp: 3 got: %lu", from);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			     i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(fake_recv_buf[i], 0,
			     "fake data corrupted, element[%d] err=%d\n",
			     i, err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(fake_recv_buf);
	free(recv_buf);

	cxit_teardown_tagged();
}

Test(tagged_directed, directed)
{
	directed_recv(false);
}

Test(tagged_directed, directed_logical)
{
	directed_recv(true);
}

/* Test unexpected send/recv */
#define RDZV_TAG (46)

struct tagged_thread_args {
	uint8_t *buf;
	size_t len;
	struct fi_cq_tagged_entry *cqe;
	fi_addr_t src_addr;
	size_t io_num;
	size_t tag;
};

static void *tsend_worker(void *data)
{
	int ret;
	struct tagged_thread_args *args;
	uint64_t tag;

	args = (struct tagged_thread_args *)data;
	tag = args->tag;

	/* Send 64 bytes to FI address 0 (self) */
	ret = fi_tsend(cxit_ep, args->buf, args->len, NULL, cxit_ep_fi_addr,
		       tag, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	/* Wait for async event indicating data has been sent */
	do {
		ret = fi_cq_read(cxit_tx_cq, args->cqe, 1);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	pthread_exit(NULL);
}

static void *trecv_worker(void *data)
{
	int ret;
	struct tagged_thread_args *args;
	uint64_t tag;

	args = (struct tagged_thread_args *)data;
	tag = args->tag;

	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, args->buf, args->len, NULL, FI_ADDR_UNSPEC, tag,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, args->cqe, 1, &args->src_addr);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "%s %ld: unexpected ret %d", __func__,
		     args->io_num, ret);

	pthread_exit(NULL);
}

Test(tagged, ux_sw_rdzv)
{
	size_t i;
	int ret;
	uint8_t *recv_buf, *send_buf;
	size_t buf_len = 2 * 1024 * 1024;
	int recv_len = 4 * 1024;
	int send_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe, tx_cqe;
	pthread_t threads[2];
	struct tagged_thread_args args[2];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, buf_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	for (i = 0; i < buf_len; i++)
		send_buf[i] = i + 0xa0;

	args[0].buf = send_buf;
	args[0].len = send_len;
	args[0].cqe = &tx_cqe;
	args[0].io_num = 0;
	args[0].tag = RDZV_TAG;
	args[1].buf = recv_buf;
	args[1].len = recv_len;
	args[1].cqe = &rx_cqe;
	args[1].io_num = 1;
	args[1].tag = RDZV_TAG;

	/* Give some time for the message to move */
	cr_assert_arr_neq(recv_buf, send_buf, buf_len);

	/* start tsend thread */
	ret = pthread_create(&threads[0], &attr, tsend_worker,
			     (void *)&args[0]);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	sleep(1);

	/* start trecv thread */
	ret = pthread_create(&threads[1], &attr, trecv_worker,
			     (void *)&args[1]);
	cr_assert_eq(ret, 0, "Recv thread create failed %d", ret);

	/* Wait for the threads to complete */
	ret = pthread_join(threads[0], NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);
	ret = pthread_join(threads[1], NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	pthread_attr_destroy(&attr);

	/* Validate sent data */
	cr_expect_arr_eq(recv_buf, send_buf, recv_len);
	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
	validate_rx_event(&rx_cqe, NULL, recv_len, FI_TAGGED | FI_RECV, NULL,
			  0, args[0].tag);
	cr_assert_eq(args[1].src_addr, cxit_ep_fi_addr,
		     "Invalid source address");

	free(send_buf);
	free(recv_buf);
}

Test(tagged, expected_sw_rdzv)
{
	size_t i;
	int ret;
	uint8_t *recv_buf, *send_buf;
	size_t buf_len = 2 * 1024 * 1024;
	int recv_len = 4 * 1024;
	int send_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe, tx_cqe;
	pthread_t threads[2];
	struct tagged_thread_args args[2];
	pthread_attr_t attr;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, buf_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	for (i = 0; i < buf_len; i++)
		send_buf[i] = i + 0xa0;

	args[0].buf = send_buf;
	args[0].len = send_len;
	args[0].cqe = &tx_cqe;
	args[0].io_num = 0;
	args[0].tag = RDZV_TAG;
	args[1].buf = recv_buf;
	args[1].len = recv_len;
	args[1].cqe = &rx_cqe;
	args[1].io_num = 1;
	args[1].tag = RDZV_TAG;

	/* Give some time for the message to move */
	cr_assert_arr_neq(recv_buf, send_buf, buf_len);

	/* Start trecv thread first so the buffer is ready when the data is sent
	 */
	ret = pthread_create(&threads[1], &attr, trecv_worker,
			     (void *)&args[1]);
	cr_assert_eq(ret, 0, "Recv thread create failed %d", ret);

	sleep(1);

	/* Start tsend thread to send the data into the ready buffer */
	ret = pthread_create(&threads[0], &attr, tsend_worker,
			     (void *)&args[0]);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	/* Wait for the threads to complete */
	ret = pthread_join(threads[0], NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);
	ret = pthread_join(threads[1], NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	pthread_attr_destroy(&attr);

	/* Validate sent data */
	cr_expect_arr_eq(recv_buf, send_buf, recv_len);
	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
	validate_rx_event(&rx_cqe, NULL, recv_len, FI_TAGGED | FI_RECV, NULL,
			  0, args[0].tag);
	cr_assert_eq(args[1].src_addr, cxit_ep_fi_addr,
		     "Invalid source address");

	free(send_buf);
	free(recv_buf);
}

#define NUM_IOS (12)

struct tagged_event_args {
	struct fid_cq *cq;
	struct fi_cq_tagged_entry *cqe;
	size_t io_num;
};

static void *tagged_evt_worker(void *data)
{
	int ret;
	struct tagged_event_args *args;

	args = (struct tagged_event_args *)data;

	for (size_t i = 0; i < args->io_num; i++) {
		/* Wait for async event indicating data has been sent */
		do {
			ret = fi_cq_read(args->cq, &args->cqe[i], 1);
		} while (ret == -FI_EAGAIN);
		cr_assert_eq(ret, 1, "%ld: unexpected ret %d", i,
			     ret);
	}

	pthread_exit(NULL);
}

Test(tagged, multitudes_sw_rdzv, .timeout=60)
{
	int ret;
	size_t buf_len = 4 * 1024;
	struct fi_cq_tagged_entry rx_cqe[NUM_IOS];
	struct fi_cq_tagged_entry tx_cqe[NUM_IOS];
	struct tagged_thread_args tx_args[NUM_IOS];
	struct tagged_thread_args rx_args[NUM_IOS];
	pthread_t tx_thread;
	pthread_t rx_thread;
	pthread_attr_t attr;
	struct tagged_event_args tx_evt_args = {
		.cq = cxit_tx_cq,
		.cqe = tx_cqe,
		.io_num = NUM_IOS,
	};
	struct tagged_event_args rx_evt_args = {
		.cq = cxit_rx_cq,
		.cqe = rx_cqe,
		.io_num = NUM_IOS,
	};

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	/* Issue the Sends */
	for (size_t tx_io = 0; tx_io < NUM_IOS; tx_io++) {
		tx_args[tx_io].len = buf_len;
		tx_args[tx_io].tag = tx_io;
		tx_args[tx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(tx_args[tx_io].buf);
		for (size_t i = 0; i < buf_len; i++)
			tx_args[tx_io].buf[i] = i + 0xa0 + tx_io;

		ret = fi_tsend(cxit_ep, tx_args[tx_io].buf, tx_args[tx_io].len,
			       NULL, cxit_ep_fi_addr, tx_args[tx_io].tag,
			       NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend %ld: unexpected ret %d",
			     tx_io, ret);
	}

	/* Start processing Send events */
	ret = pthread_create(&tx_thread, &attr, tagged_evt_worker,
				(void *)&tx_evt_args);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	sleep(1);

	/* Issue the Receives */
	for (size_t rx_io = 0; rx_io < NUM_IOS; rx_io++) {
		rx_args[rx_io].len = buf_len;
		rx_args[rx_io].tag = rx_io;
		rx_args[rx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(rx_args[rx_io].buf);
		memset(rx_args[rx_io].buf, 0, buf_len);

		ret = fi_trecv(cxit_ep, rx_args[rx_io].buf, rx_args[rx_io].len,
			       NULL, FI_ADDR_UNSPEC, rx_args[rx_io].tag,
			       0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv %ld: unexpected ret %d",
			     rx_io, ret);
	}

	/* Start processing Receive events */
	ret = pthread_create(&rx_thread, &attr, tagged_evt_worker,
			     (void *)&rx_evt_args);
	cr_assert_eq(ret, 0, "Receive thread create failed %d", ret);

	/* Wait for the RX/TX event threads to complete */
	ret = pthread_join(tx_thread, NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);

	ret = pthread_join(rx_thread, NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	/* Validate results */
	for (size_t io = 0; io < NUM_IOS; io++) {
		/* Validate sent data */
		cr_expect_arr_eq(rx_args[io].buf, tx_args[io].buf, buf_len);
		validate_tx_event(&tx_cqe[io], FI_TAGGED | FI_SEND, NULL);
		validate_rx_event(&rx_cqe[io], NULL, buf_len,
				  FI_TAGGED | FI_RECV, NULL,
				  0, tx_args[rx_cqe[io].tag].tag);

		free(tx_args[io].buf);
		free(rx_args[io].buf);
	}

	pthread_attr_destroy(&attr);
}

struct multitudes_params {
	size_t length;
	size_t num_ios;
};

/* This is a parameterized test to execute an arbitrary set of tagged send/recv
 * operations. The test is configurable in two parameters, the length value is
 * the size of the data to be transferred. The num_ios will set the number of
 * matching send/recv that are launched in each test.
 *
 * The test will first execute the fi_tsend() for `num_ios` number of buffers.
 * A background thread is launched to start processing the Cassini events for
 * the Send operations. The test will then pause for 1 second. After the pause,
 * The test will execute the fi_trecv() to receive the buffers that were
 * previously sent. Another background thread is then launched to process the
 * receive events. When all send and receive operations have completed, the
 * threads exit and the results are compared to ensure the expected data was
 * returned.
 *
 * Based on the test's length parameter it will change the processing of the
 * send and receive operation. 2kiB and below lengths will cause the eager
 * data path to be used. Larger than 2kiB buffers will use the SW Rendezvous
 * data path to be used.
 */
ParameterizedTestParameters(tagged, multitudes)
{
	size_t param_sz;

	static struct multitudes_params params[] = {
		{.length = 1024,	/* Eager */
		 .num_ios = 10},
		{.length = 2 * 1024,	/* Eager */
		 .num_ios = 15},
		{.length = 4 * 1024,	/* Rendezvous */
		 .num_ios = 12},
		{.length = 128 * 1024,	/* Rendezvous */
		 .num_ios = 25},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct multitudes_params, params,
				   param_sz);
}

ParameterizedTest(struct multitudes_params *param, tagged, multitudes, .timeout=60)
{
	int ret;
	size_t buf_len = param->length;
	struct fi_cq_tagged_entry *rx_cqe;
	struct fi_cq_tagged_entry *tx_cqe;
	struct tagged_thread_args *tx_args;
	struct tagged_thread_args *rx_args;
	pthread_t tx_thread;
	pthread_t rx_thread;
	pthread_attr_t attr;
	struct tagged_event_args tx_evt_args = {
		.cq = cxit_tx_cq,
		.io_num = param->num_ios,
	};
	struct tagged_event_args rx_evt_args = {
		.cq = cxit_rx_cq,
		.io_num = param->num_ios,
	};

	tx_cqe = calloc(param->num_ios, sizeof(struct fi_cq_tagged_entry));
	cr_assert_not_null(tx_cqe);

	rx_cqe = calloc(param->num_ios, sizeof(struct fi_cq_tagged_entry));
	cr_assert_not_null(rx_cqe);

	tx_args = calloc(param->num_ios, sizeof(struct tagged_thread_args));
	cr_assert_not_null(tx_args);

	rx_args = calloc(param->num_ios, sizeof(struct tagged_thread_args));
	cr_assert_not_null(rx_args);

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	tx_evt_args.cqe = tx_cqe;
	rx_evt_args.cqe = rx_cqe;

	/* Issue the Sends */
	for (size_t tx_io = 0; tx_io < param->num_ios; tx_io++) {
		tx_args[tx_io].len = buf_len;
		tx_args[tx_io].tag = tx_io;
		tx_args[tx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(tx_args[tx_io].buf);
		for (size_t i = 0; i < buf_len; i++)
			tx_args[tx_io].buf[i] = i + 0xa0 + tx_io;

		ret = fi_tsend(cxit_ep, tx_args[tx_io].buf, tx_args[tx_io].len,
			       NULL, cxit_ep_fi_addr, tx_args[tx_io].tag,
			       NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend %ld: unexpected ret %d",
			     tx_io, ret);
	}

	/* Start processing Send events */
	ret = pthread_create(&tx_thread, &attr, tagged_evt_worker,
				(void *)&tx_evt_args);
	cr_assert_eq(ret, 0, "Send thread create failed %d", ret);

	sleep(1);

	/* Issue the Receives */
	for (size_t rx_io = 0; rx_io < param->num_ios; rx_io++) {
		rx_args[rx_io].len = buf_len;
		rx_args[rx_io].tag = rx_io;
		rx_args[rx_io].buf = aligned_alloc(C_PAGE_SIZE, buf_len);
		cr_assert_not_null(rx_args[rx_io].buf);
		memset(rx_args[rx_io].buf, 0, buf_len);

		ret = fi_trecv(cxit_ep, rx_args[rx_io].buf, rx_args[rx_io].len,
			       NULL, FI_ADDR_UNSPEC, rx_args[rx_io].tag,
			       0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv %ld: unexpected ret %d",
			     rx_io, ret);
	}

	/* Start processing Receive events */
	ret = pthread_create(&rx_thread, &attr, tagged_evt_worker,
			     (void *)&rx_evt_args);
	cr_assert_eq(ret, 0, "Receive thread create failed %d", ret);

	/* Wait for the RX/TX event threads to complete */
	ret = pthread_join(tx_thread, NULL);
	cr_assert_eq(ret, 0, "Send thread join failed %d", ret);

	ret = pthread_join(rx_thread, NULL);
	cr_assert_eq(ret, 0, "Recv thread join failed %d", ret);

	/* Validate results */
	for (size_t io = 0; io < param->num_ios; io++) {
		/* Validate sent data */
		cr_expect_arr_eq(rx_args[io].buf, tx_args[io].buf, buf_len);

		validate_tx_event(&tx_cqe[io], FI_TAGGED | FI_SEND, NULL);

		validate_rx_event(&rx_cqe[io], NULL, buf_len,
				  FI_TAGGED | FI_RECV, NULL,
				  0, tx_args[rx_cqe[io].tag].tag);

		free(tx_args[io].buf);
		free(rx_args[io].buf);
	}

	pthread_attr_destroy(&attr);
	free(rx_cqe);
	free(tx_cqe);
	free(tx_args);
	free(rx_args);
}

#define RECV_INIT 0x77
#define SEND_INIT ~RECV_INIT

void do_msg(uint8_t *send_buf, size_t send_len, uint64_t send_tag,
	    uint8_t *recv_buf, size_t recv_len, uint64_t recv_tag,
	    uint64_t recv_ignore, bool send_first, size_t buf_size,
	    bool tagged, bool wdata, uint64_t data, bool match_complete)
{
	int i, ret;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	bool sent = false,
	     recved = false,
	     truncated = false;
	struct fi_cq_err_entry err_cqe = {};
	size_t recved_len;
	static int send_cnt;
	static int recv_cnt;
	static int recv_errcnt;

	struct fi_msg_tagged tsmsg = {};
	struct fi_msg smsg = {};
	struct iovec siovec;
	uint64_t send_flags = 0;

	memset(recv_buf, RECV_INIT, buf_size);

	for (i = 0; i < buf_size; i++) {
		if (i < send_len)
			send_buf[i] = i + 0xa0;
		else
			send_buf[i] = SEND_INIT;
	}

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;

	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.context = NULL;
	smsg.data = data;

	tsmsg.msg_iov = &siovec;
	tsmsg.iov_count = 1;
	tsmsg.addr = cxit_ep_fi_addr;
	tsmsg.tag = send_tag;
	tsmsg.ignore = 0;
	tsmsg.context = NULL;
	tsmsg.data = data;

	/* FI_REMOTE_CQ_DATA flag is not strictly necessary. */
	if (wdata)
		send_flags |= FI_REMOTE_CQ_DATA;
	if (match_complete)
		send_flags |= FI_MATCH_COMPLETE;

	if (send_first) {
		if (tagged) {
			ret = fi_tsendmsg(cxit_ep, &tsmsg, send_flags);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_tsendmsg failed %d", ret);
		} else {
			ret = fi_sendmsg(cxit_ep, &smsg, send_flags);
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

	/* Post RX buffer */

	if (tagged) {
		ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
			       FI_ADDR_UNSPEC, recv_tag, recv_ignore, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
	} else {
		ret = fi_recv(cxit_ep, recv_buf, recv_len, NULL,
			      FI_ADDR_UNSPEC, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_recv failed %d", ret);
	}

	if (!send_first) {
		if (tagged) {
			ret = fi_tsendmsg(cxit_ep, &tsmsg, send_flags);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_tsendmsg failed %d", ret);
		} else {
			ret = fi_sendmsg(cxit_ep, &smsg, send_flags);
			cr_assert_eq(ret, FI_SUCCESS,
				     "fi_sendmsg failed %d", ret);
		}
	}

	/* Gather both events, ensure progress on both sides. */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		if (ret == 1) {
			cr_assert_eq(recved, false);
			recved = true;
		} else if (ret == -FI_EAVAIL) {
			cr_assert_eq(recved, false);
			recved = true;
			truncated = true;

			ret = fi_cq_readerr(cxit_rx_cq, &err_cqe, 0);
			cr_assert_eq(ret, 1);
		} else {
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d", ret);
		}

		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		if (ret == 1) {
			cr_assert_eq(sent, false);
			sent = true;
		} else {
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d", ret);
		}
	} while (!(sent && recved));

	if (truncated) {
		cr_assert(err_cqe.op_context == NULL,
			  "Error RX CQE Context mismatch");
		cr_assert(err_cqe.flags ==
			  ((tagged ? FI_TAGGED : FI_MSG) | FI_RECV),
			  "Error RX CQE flags mismatch");
		cr_assert(err_cqe.len == recv_len,
			  "Invalid Error RX CQE length, got: %ld exp: %ld",
			  err_cqe.len, recv_len);
		cr_assert(err_cqe.buf == 0, "Invalid Error RX CQE address");
		cr_assert(err_cqe.data == data, "Invalid Error RX CQE data");
		cr_assert(err_cqe.tag == send_tag, "Invalid Error RX CQE tag");
		cr_assert(err_cqe.olen == (send_len - recv_len),
			  "Invalid Error RX CQE olen, got: %ld exp: %ld",
			  err_cqe.olen, send_len - recv_len);
		cr_assert(err_cqe.err == FI_EMSGSIZE,
			  "Invalid Error RX CQE code\n");
		cr_assert(err_cqe.prov_errno == C_RC_OK,
			  "Invalid Error RX CQE errno");
		cr_assert(err_cqe.err_data == NULL);
		cr_assert(err_cqe.err_data_size == 0);
		recved_len = err_cqe.len;
	} else {
		validate_rx_event(&rx_cqe, NULL, send_len,
				  (tagged ? FI_TAGGED : FI_MSG) | FI_RECV,
				  NULL, data, send_tag);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");
		recved_len = rx_cqe.len;
	}

	validate_tx_event(&tx_cqe, (tagged ? FI_TAGGED : FI_MSG) | FI_SEND,
			  NULL);

	/* Validate sent data */
	for (i = 0; i < buf_size; i++) {
		uint8_t cmp = RECV_INIT;
		if (i < recved_len)
			cmp = send_buf[i];

		cr_expect_eq(recv_buf[i], cmp,
			     "data mismatch, len: %ld, element[%d], exp=0x%x saw=0x%x, err=%d\n",
			     recv_len, i, cmp, recv_buf[i], err++);
		if (err >= 10)
			break;
	}
	cr_assert_eq(err, 0, "%d data errors seen\n", err);

	/* Check counters */
	send_cnt++;

	if (truncated)
		recv_errcnt++;
	else
		recv_cnt++;

	while (fi_cntr_read(cxit_send_cntr) != send_cnt)
		sched_yield();
	while (fi_cntr_read(cxit_recv_cntr) != recv_cnt)
		sched_yield();
	while (fi_cntr_readerr(cxit_recv_cntr) != recv_errcnt)
		sched_yield();

	/* Error count is 7 bits */
	if (recv_errcnt == 127) {
		recv_errcnt = 0;
		fi_cntr_seterr(cxit_recv_cntr, 0);
	}
}

#define BUF_SIZE (8*1024)
#define SEND_MIN 64
#define SEND_INC 64
#define TAG 0x333333333333
#define IGNORE_ALL (-1ULL & CXIP_TAG_MASK)
#define HDR_DATA 0xabcdabcdabcdabcd

struct tagged_rx_params {
	size_t buf_size;
	size_t send_min;
	size_t send_inc;
	uint64_t send_tag;
	int recv_len_off;
	uint64_t recv_tag;
	uint64_t ignore;
	bool ux;
	bool tagged;
	bool wdata;
	uint64_t data;
};

static struct tagged_rx_params params[] = {
	{.buf_size = BUF_SIZE, /* equal length no data */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = true},

	/* Use CQ data */

	{.buf_size = BUF_SIZE, /* truncate */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = -8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = -8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = 0,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = 0,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = true},
	{.buf_size = BUF_SIZE, /* excess */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = 8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = TAG,
	 .recv_len_off = 8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},

	/* Un-tagged variants */

	{.buf_size = BUF_SIZE, /* equal length no data */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = false},

	/* Use CQ data */

	{.buf_size = BUF_SIZE, /* truncate */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = true,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* truncate ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = -8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* equal length ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 0,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = false,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = 0,
	 .ignore = 0,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess ignore */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = false,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
	{.buf_size = BUF_SIZE, /* excess ignore UX */
	 .send_min = SEND_MIN,
	 .send_inc = SEND_INC,
	 .send_tag = 0,
	 .recv_len_off = 8,
	 .recv_tag = ~TAG & CXIP_TAG_MASK,
	 .ignore = IGNORE_ALL,
	 .ux = true,
	 .tagged = false,
	 .wdata = true,
	 .data = HDR_DATA},
};

ParameterizedTestParameters(tagged, rx)
{
	size_t param_sz;

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct tagged_rx_params, params,
				   param_sz);
}

ParameterizedTest(struct tagged_rx_params *param, tagged, rx, .timeout=30)
{
	uint8_t *recv_buf,
		*send_buf;
	size_t send_len;

	recv_buf = aligned_alloc(C_PAGE_SIZE, param->buf_size);
	cr_assert(recv_buf);

	send_buf = aligned_alloc(C_PAGE_SIZE, param->buf_size);
	cr_assert(send_buf);

	for (send_len = param->send_min;
	     send_len <= param->buf_size;
	     send_len += param->send_inc) {
		do_msg(send_buf, send_len, param->send_tag,
		       recv_buf, send_len + param->recv_len_off,
		       param->recv_tag, param->ignore, param->ux,
		       param->buf_size, param->tagged,
		       param->wdata, param->data, false);
		do_msg(send_buf, send_len, param->send_tag,
		       recv_buf, send_len + param->recv_len_off,
		       param->recv_tag, param->ignore, param->ux,
		       param->buf_size, param->tagged,
		       param->wdata, param->data, true);
	}

	free(send_buf);
	free(recv_buf);
}

#define GB 1024*1024*1024
Test(tagged, rput_abort, .disabled=true)
{
	size_t recv_len = GB;
	size_t send_len = GB;
	void *recv_buf;
	void *send_buf;
	int ret;
	uint64_t val __attribute__((unused));

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
		       FI_ADDR_UNSPEC, 0, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
	sleep(1);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	ret = fi_tsend(cxit_ep, send_buf, send_len,
		       NULL, cxit_ep_fi_addr, 0, NULL);
	cr_assert_eq(ret, FI_SUCCESS,
		     "fi_tsend failed %d", ret);

	sleep(1);
	val = *(uint64_t *)0;
}


Test(tagged, oflow_replenish, .timeout=180)
{
	uint8_t *recv_buf,
		*send_buf;
	size_t send_len = 1024;
	int i;

	recv_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(recv_buf);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < 6*1024+1; i++) {
		do_msg(send_buf, send_len, 0,
		       recv_buf, send_len, 0, 0,
		       true, send_len, true, false, 0, false);
	}

	free(send_buf);
	free(recv_buf);
}

/* Test outstanding send cleanup */
Test(tagged, cleanup_sends)
{
	int i, ret;
	uint8_t *send_buf;
	int send_len = 64;
	int sends = 5;

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	/* Send 64 bytes to self */
	for (i = 0; i < sends; i++) {
		ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);
	}

	/* Close Endpoint with outstanding Sends */
}

/* Test UX cleanup */
Test(tagged, ux_cleanup)
{
	int i, ret;
	uint8_t *send_buf;
	int send_len = 64;
	struct fi_cq_tagged_entry cqe;
	int sends = 5;

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	/* Send 64 bytes to self */
	for (i = 0; i < sends; i++) {
		ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);
	}

	validate_tx_event(&cqe, FI_TAGGED | FI_SEND, NULL);

	/* Wait for async event indicating data has been received */
	for (i = 0 ; i < 1000; i++)
		fi_cq_readfrom(cxit_rx_cq, &cqe, 1, NULL);

	free(send_buf);

	/* Close Endpoint with UX sends on the RX Queue */
}

/* Test outstanding recv cleanup */
Test(tagged, cleanup_recvs)
{
	int i, ret;
	uint8_t *recv_buf;
	int recv_len = 64;
	int recvs = 5;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	for (i = 0; i < recvs; i++) {
		ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
			       FI_ADDR_UNSPEC, 0x0, 0x0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);
	}

	/* Close Endpoint with outstanding Receives */
}

/* Test outstanding recv cancel */
Test(tagged, cancel_recvs)
{
	int i, ret;
	uint8_t *recv_buf;
	int recv_len = 64;
	int recvs = 5;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	for (i = 0; i < recvs; i++) {
		ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
			       FI_ADDR_UNSPEC, 0x0, 0x0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);
	}

	for (i = 0; i < recvs; i++) {
		ret = fi_cancel(&cxit_ep->fid, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_cancel failed %d", ret);
	}
}

/* Test cancel with no matching recv */
Test(tagged, cancel_nomatch)
{
	int ret;

	ret = fi_cancel(&cxit_ep->fid, NULL);
	cr_assert_eq(ret, -FI_ENOENT, "fi_cancel failed to fail %d", ret);
}

/* Test outstanding recv cancel events */
Test(tagged, cancel_recvs_sync)
{
	int i, ret;
	uint8_t *recv_buf;
	int recv_len = 64;
	int recvs = 5;
	struct fi_cq_tagged_entry rx_cqe;
	struct fi_cq_err_entry err_cqe;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	for (i = 0; i < recvs; i++) {
		ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
			       FI_ADDR_UNSPEC, 0x0, 0x0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);
	}

	for (i = 0; i < recvs; i++) {
		ret = fi_cancel(&cxit_ep->fid, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_cancel failed %d", ret);
	}

	ret = fi_cancel(&cxit_ep->fid, NULL);
	cr_assert_eq(ret, -FI_ENOENT, "fi_cancel failed to fail %d", ret);

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
		cr_assert(err_cqe.flags == (FI_TAGGED | FI_RECV),
			  "Error RX CQE flags mismatch");
		cr_assert(err_cqe.err == FI_ECANCELED,
			  "Invalid Error RX CQE code\n");
		cr_assert(err_cqe.prov_errno == 0,
			  "Invalid Error RX CQE errno");
	}
}

void cxit_setup_selective_completion(void)
{
	cxit_tx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;
	cxit_rx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;

	cxit_setup_getinfo();
	cxit_fi_hints->tx_attr->op_flags = FI_COMPLETION;
	cxit_fi_hints->rx_attr->op_flags = FI_COMPLETION;
	cxit_setup_tagged();
}

/* Test selective completion behavior with RMA. */
Test(tagged_sel, selective_completion,
     .init = cxit_setup_selective_completion,
     .fini = cxit_teardown_tagged)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int buf_len = 0x1000;
	int send_len;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged smsg = {};
	struct fi_msg_tagged rmsg = {};
	struct iovec siovec;
	struct iovec riovec;
	int recv_cnt = 0;

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);

	riovec.iov_base = recv_buf;
	riovec.iov_len = buf_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	siovec.iov_base = send_buf;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	/* Normal writes generate completions */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		bool sent = false;
		bool rcved = false;

		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		ret = fi_trecv(cxit_ep, recv_buf, send_len, NULL,
			       FI_ADDR_UNSPEC, 0, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
		recv_cnt++;

		/* Send to self */
		ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			if (ret == 1)
				rcved = true;

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1)
				sent = true;
		} while (!(sent && rcved));

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");
		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");
	}

	/* Request completions from fi_writemsg */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		bool sent = false;
		bool rcved = false;

		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		ret = fi_trecvmsg(cxit_ep, &rmsg, FI_COMPLETION);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);
		recv_cnt++;

		/* Send to self */
		siovec.iov_len = send_len;
		ret = fi_tsendmsg(cxit_ep, &smsg, FI_COMPLETION);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			if (ret == 1)
				rcved = true;

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1)
				sent = true;
		} while (!(sent && rcved));

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");
		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");
	}

	/* Suppress completions using fi_writemsg */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		riovec.iov_len = send_len;
		ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);
		recv_cnt++;

		/* Send to self */
		siovec.iov_len = send_len;
		ret = fi_tsendmsg(cxit_ep, &smsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			cr_assert(ret == -FI_EAGAIN);

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			cr_assert(ret == -FI_EAGAIN);
		} while (fi_cntr_read(cxit_recv_cntr) != recv_cnt);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);

		ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Inject never generates an event */

	send_len = 8;
	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, send_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
	recv_cnt++;

	/* Send 64 bytes to self */
	ret = fi_tinject(cxit_ep, send_buf, send_len, cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async event indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
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

void cxit_setup_selective_completion_suppress(void)
{
	cxit_tx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;
	cxit_rx_cq_bind_flags |= FI_SELECTIVE_COMPLETION;

	cxit_setup_getinfo();
	cxit_fi_hints->tx_attr->op_flags = 0;
	cxit_fi_hints->rx_attr->op_flags = 0;
	cxit_setup_tagged();
}

/* Test selective completion behavior with RMA. */
Test(tagged_sel, selective_completion_suppress,
     .init = cxit_setup_selective_completion_suppress,
     .fini = cxit_teardown_tagged)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int buf_len = 0x1000;
	int send_len;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged smsg = {};
	struct fi_msg_tagged rmsg = {};
	struct iovec siovec;
	struct iovec riovec;
	int recv_cnt = 0;

	recv_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(recv_buf);

	riovec.iov_base = recv_buf;
	riovec.iov_len = buf_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	send_buf = aligned_alloc(C_PAGE_SIZE, buf_len);
	cr_assert(send_buf);

	siovec.iov_base = send_buf;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	/* Normal writes do not generate completions */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		ret = fi_trecv(cxit_ep, recv_buf, send_len, NULL,
			       FI_ADDR_UNSPEC, 0, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
		recv_cnt++;

		/* Send to self */
		ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 0, NULL);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			cr_assert(ret == -FI_EAGAIN);

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			cr_assert(ret == -FI_EAGAIN);
		} while (fi_cntr_read(cxit_recv_cntr) != recv_cnt);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);

		ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Request completions from fi_writemsg */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		bool sent = false;
		bool rcved = false;

		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		riovec.iov_len = send_len;
		ret = fi_trecvmsg(cxit_ep, &rmsg, FI_COMPLETION);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);
		recv_cnt++;

		/* Send to self */
		siovec.iov_len = send_len;
		ret = fi_tsendmsg(cxit_ep, &smsg, FI_COMPLETION);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			if (ret == 1)
				rcved = true;

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			if (ret == 1)
				sent = true;
		} while (!(sent && rcved));

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");
		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");
	}

	/* Suppress completions using fi_writemsg */
	for (send_len = 1; send_len <= buf_len; send_len <<= 1) {
		memset(recv_buf, 0, send_len);
		for (i = 0; i < send_len; i++)
			send_buf[i] = i + 0xa0;

		/* Post RX buffer */
		riovec.iov_len = send_len;
		ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);
		recv_cnt++;

		/* Send to self */
		siovec.iov_len = send_len;
		ret = fi_tsendmsg(cxit_ep, &smsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Wait for async events indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
			cr_assert(ret == -FI_EAGAIN);

			ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
			cr_assert(ret == -FI_EAGAIN);
		} while (fi_cntr_read(cxit_recv_cntr) != recv_cnt);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				     "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				     i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");

		/* Ensure no events were generated */
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);

		ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	}

	/* Inject never generates an event */

	send_len = 8;
	/* Post RX buffer */
	ret = fi_trecv(cxit_ep, recv_buf, send_len, NULL, FI_ADDR_UNSPEC, 0,
		       0, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);
	recv_cnt++;

	/* Send 64 bytes to self */
	ret = fi_tinject(cxit_ep, send_buf, send_len, cxit_ep_fi_addr, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsend failed %d", ret);

	/* Wait for async events indicating data has been received */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		cr_assert(ret == -FI_EAGAIN);

		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);
	} while (fi_cntr_read(cxit_recv_cntr) != recv_cnt);

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

	ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
	cr_assert(ret == -FI_EAGAIN);

	free(send_buf);
	free(recv_buf);
}

/* Test match complete */
Test(tagged, match_comp)
{
	int i, j, ret;
	uint8_t *recv_buf,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	for (j = 0; j < 100; j++) {
		ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

		ret = fi_tsendmsg(cxit_ep, &smsg, FI_MATCH_COMPLETE);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Wait for async event indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		} while (ret == -FI_EAGAIN);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		/* Validate sent data */
		for (i = 0; i < send_len; i++) {
			cr_expect_eq(recv_buf[i], send_buf[i],
				  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
				  i, send_buf[i], recv_buf[i], err++);
		}
		cr_assert_eq(err, 0, "Data errors seen\n");

		/* UX */

		ret = fi_tsendmsg(cxit_ep, &smsg, FI_MATCH_COMPLETE);
		cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

		/* Ensure no TX event is generated */
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert(ret == -FI_EAGAIN);

		ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
		cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

		/* Wait for async event indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		} while (ret == -FI_EAGAIN);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

		/* Wait for async event indicating data has been sent */
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
	}

	free(send_buf);
	free(recv_buf);
}

/* Test eager Send with FI_MORE */
Test(tagged, esend_more)
{
	int i, ret;
	uint8_t *recv_buf,
		*recv_buf2,
		*send_buf;
	int recv_len = 64;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	recv_buf2 = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf2);
	memset(recv_buf2, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	/* Post two Receives */
	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	riovec.iov_base = recv_buf2;
	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	ret = fi_tsendmsg(cxit_ep, &smsg, FI_MORE);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Ensure no completion before the doorbell ring */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN,
			     "write failed %d", ret);
	} while (i++ < 100000);

	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Gather 2 Receive events */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Gather 2 Send events */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(recv_buf2[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf2[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test rendezvous Send with FI_MORE */
Test(tagged, rsend_more)
{
	int i, ret;
	uint8_t *recv_buf,
		*recv_buf2,
		*send_buf;
	int recv_len = 0x1000;
	int send_len = 0x1000;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	recv_buf2 = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf2);
	memset(recv_buf2, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	/* Post two Receives */
	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	riovec.iov_base = recv_buf2;
	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	ret = fi_tsendmsg(cxit_ep, &smsg, FI_MORE);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Ensure no completion before the doorbell ring */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN,
			     "write failed %d", ret);
	} while (i++ < 100000);

	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Gather 2 Receive events */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Gather 2 Send events */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(recv_buf2[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf2[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test Receive with FI_MORE */
Test(tagged, recv_more)
{
	int i, ret;
	uint8_t *recv_buf,
		*recv_buf2,
		*send_buf;
	int recv_len = 0x2000;
	int send_len = 0x2000;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	int err = 0;
	fi_addr_t from;
	struct fi_msg_tagged rmsg = {};
	struct fi_msg_tagged smsg = {};
	struct iovec riovec;
	struct iovec siovec;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	recv_buf2 = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf2);
	memset(recv_buf2, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < send_len; i++)
		send_buf[i] = i + 0xa0;

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len;
	rmsg.msg_iov = &riovec;
	rmsg.iov_count = 1;
	rmsg.addr = FI_ADDR_UNSPEC;
	rmsg.tag = 0;
	rmsg.ignore = 0;
	rmsg.context = NULL;

	siovec.iov_base = send_buf;
	siovec.iov_len = send_len;
	smsg.msg_iov = &siovec;
	smsg.iov_count = 1;
	smsg.addr = cxit_ep_fi_addr;
	smsg.tag = 0;
	smsg.ignore = 0;
	smsg.context = NULL;

	/* Perform 2 Sends */
	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	ret = fi_tsendmsg(cxit_ep, &smsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_tsendmsg failed %d", ret);

	/* Post two Receives */
	ret = fi_trecvmsg(cxit_ep, &rmsg, FI_MORE);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Ensure no completion before the doorbell ring */
	do {
		ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN,
			     "write failed %d", ret);
	} while (i++ < 100000);

	riovec.iov_base = recv_buf2;
	ret = fi_trecvmsg(cxit_ep, &rmsg, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_trecvmsg failed %d", ret);

	/* Gather 2 Receive events */
	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV, NULL,
			  0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	/* Gather 2 Send events */
	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
	cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

	validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

	/* Validate sent data */
	for (i = 0; i < send_len; i++) {
		cr_expect_eq(recv_buf[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf[i], err++);
		cr_expect_eq(recv_buf2[i], send_buf[i],
			  "data mismatch, element[%d], exp=%d saw=%d, err=%d\n",
			  i, send_buf[i], recv_buf2[i], err++);
	}
	cr_assert_eq(err, 0, "Data errors seen\n");

	free(send_buf);
	free(recv_buf);
}

/* Test flow control.
 *
 * Perform enough Sends to overwhelm target LEs. Flow control recovery is
 * transparent.
 *
 * Post matching Receives and check data to validate correct ordering amid flow
 * control recovery.
 */
Test(tagged, fc, .timeout = 180)
{
	int i, j, ret, tx_ret;
	uint8_t *send_bufs;
	uint8_t *send_buf;
	int send_len = 64;
	uint8_t *recv_buf;
	int recv_len = 64;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int nsends_concurrent = 3; /* must be less than the LE pool min. */
	int nsends = 14000;
	int sends = 0;
	uint64_t tag = 0xbeef;
	fi_addr_t from;

	send_bufs = aligned_alloc(C_PAGE_SIZE, send_len * nsends_concurrent);
	cr_assert(send_bufs);

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	for (i = 0; i < nsends_concurrent - 1; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		tx_ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, tag, NULL);
	}

	for (i = nsends_concurrent - 1; i < nsends; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		do {
			tx_ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
				       cxit_ep_fi_addr, tag, NULL);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);

			/* Just progress */
			fi_cq_read(cxit_tx_cq, NULL, 0);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, FI_SUCCESS, "fi_tsend failed %d", tx_ret);

		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	for (i = 0; i < nsends_concurrent - 1; i++) {
		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	for (i = 0; i < nsends; i++) {
		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			assert(ret == -FI_EAGAIN);

			ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
				       FI_ADDR_UNSPEC, tag, 0, NULL);
		} while (ret == -FI_EAGAIN);

		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		} while (ret == -FI_EAGAIN);

		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_rx_event(&rx_cqe, NULL, recv_len, FI_TAGGED | FI_RECV,
				  NULL, 0, tag);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

		for (j = 0; j < recv_len; j++) {
			cr_assert_eq(recv_buf[j], (uint8_t)i,
				     "data mismatch, recv: %d element[%d], exp=%d saw=%d\n",
				     i, j, (uint8_t)i, recv_buf[j]);
		}
	}

	free(send_bufs);
	free(recv_buf);
}

Test(tagged, fc_multi_recv, .timeout = 30)
{
	int i, j, k, ret, tx_ret;
	uint8_t *send_bufs;
	uint8_t *send_buf;
	int send_len = 64;
	uint8_t *recv_buf;
	int recv_len = 64;
	int mrecv_msgs = 10;
	struct fi_msg_tagged trmsg = {};
	struct iovec riovec;
	struct fi_cq_tagged_entry tx_cqe;
	struct fi_cq_tagged_entry rx_cqe;
	int nsends_concurrent = 3; /* must be less than the LE pool min. */
	int nsends = 20;
	int sends = 0;
	uint64_t tag = 0xbeef;
	fi_addr_t from;

	cr_assert(!(nsends % mrecv_msgs));

	send_bufs = aligned_alloc(C_PAGE_SIZE, send_len * nsends_concurrent);
	cr_assert(send_bufs);

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len * mrecv_msgs);
	cr_assert(recv_buf);

	for (i = 0; i < nsends_concurrent - 1; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		tx_ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, tag, NULL);
	}

	for (i = nsends_concurrent - 1; i < nsends; i++) {
		send_buf = send_bufs + (i % nsends_concurrent) * send_len;
		memset(send_buf, i, send_len);

		do {
			tx_ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
				       cxit_ep_fi_addr, tag, NULL);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);

			/* Just progress */
			fi_cq_read(cxit_tx_cq, NULL, 0);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, FI_SUCCESS, "fi_tsend failed %d", tx_ret);

		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	for (i = 0; i < nsends_concurrent - 1; i++) {
		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

			/* Progress RX to avoid EQ drops */
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			cr_assert_eq(ret, -FI_EAGAIN,
				     "fi_cq_read unexpected value %d",
				     ret);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);

		if (!(++sends % 1000))
			printf("%u Sends complete.\n", sends);
	}

	riovec.iov_base = recv_buf;
	riovec.iov_len = recv_len * mrecv_msgs;
	trmsg.msg_iov = &riovec;
	trmsg.iov_count = 1;
	trmsg.addr = FI_ADDR_UNSPEC;
	trmsg.context = NULL;
	trmsg.tag = tag;

	for (i = 0; i < nsends / mrecv_msgs; i++) {
		memset(recv_buf, 0, recv_len * mrecv_msgs);
		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 0);
			assert(ret == FI_SUCCESS || ret == -FI_EAGAIN);

			ret = fi_trecvmsg(cxit_ep, &trmsg, FI_MULTI_RECV);
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
						     FI_TAGGED | FI_RECV, 0,
						     tag);
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

Test(tagged, fc_multi_recv_rdzv, .timeout = 10)
{
	int ret;
	char *recv_buf;
	char *send_buf;
	int i;
	uint64_t tag = 0xbeef;
	struct fi_msg_tagged trmsg = {};
	struct iovec riovec;
	unsigned int send_events = 0;
	unsigned int recv_events = 0;
	struct fi_cq_tagged_entry cqe;
	size_t min_mrecv = 0;
	size_t opt_len = sizeof(size_t);
	bool unlinked = false;

	/* Transfer size needs to be large enough to trigger rendezvous. */
	unsigned int xfer_len = 16384;

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
		ret = fi_tsend(cxit_ep, &send_buf[i * xfer_len], xfer_len, NULL,
			       cxit_ep_fi_addr, tag, NULL);
		cr_assert(ret == FI_SUCCESS);
	}

	/* Append late multi-recv buffer. */
	riovec.iov_base = recv_buf;
	riovec.iov_len = num_xfers * xfer_len;
	trmsg.msg_iov = &riovec;
	trmsg.iov_count = 1;
	trmsg.addr = cxit_ep_fi_addr;
	trmsg.context = NULL;
	trmsg.tag = tag;
	ret = fi_trecvmsg(cxit_ep, &trmsg, FI_MULTI_RECV);
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

	/* Make last send expected. This ensures that hardware and/or software
	 * has correctly updated the LE start and length fields correctly.
	 */
	ret = fi_tsend(cxit_ep, &send_buf[(num_xfers - 1) * xfer_len], xfer_len,
		       NULL, cxit_ep_fi_addr, tag, NULL);
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

#define FC_TRANS 100

static void *fc_sender(void *data)
{
	int i, tx_ret;
	uint8_t *send_buf;
	int send_len = 64;
	struct fi_cq_tagged_entry tx_cqe;

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	for (i = 0; i < FC_TRANS; i++) {
		memset(send_buf, i, send_len);

		/* Send 64 bytes to self */
		do {
			tx_ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
				       cxit_ep_fi_addr, 0xa, NULL);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, FI_SUCCESS, "fi_tsend failed %d", tx_ret);

		do {
			tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);
		} while (tx_ret == -FI_EAGAIN);

		cr_assert_eq(tx_ret, 1, "fi_cq_read unexpected value %d",
			     tx_ret);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
	}

	free(send_buf);

	pthread_exit(NULL);
}

static void *fc_recver(void *data)
{
	int i, j, ret;
	uint8_t *recv_buf;
	int recv_len = 64;
	struct fi_cq_tagged_entry rx_cqe;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);

	for (i = 0; i < 5; i++) {
		sleep(1);

		/* Progress RX to avoid EQ drops */
		ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		cr_assert_eq(ret, -FI_EAGAIN,
			     "fi_cq_read unexpected value %d",
			     ret);
	}

	for (i = 0; i < FC_TRANS; i++) {
		memset(recv_buf, 0, recv_len);

		/* Send 64 bytes to self */

		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
			assert(ret == -FI_EAGAIN);

			ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
				       FI_ADDR_UNSPEC, 0xa, 0, NULL);
		} while (ret == -FI_EAGAIN);

		cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d", ret);

		do {
			ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);
		} while (ret == -FI_EAGAIN);

		cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d", ret);

		validate_rx_event(&rx_cqe, NULL, recv_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 0xa);

		for (j = 0; j < recv_len; j++) {
			cr_assert_eq(recv_buf[j], i,
				     "data mismatch, element[%d], exp=%d saw=%d\n",
				     j, i, recv_buf[j]);
		}
	}

	free(recv_buf);

	pthread_exit(NULL);
}

/*
 * Multi-threaded flow control test.
 *
 * Run sender and receiver threads. Start sender first to allow it to overwhelm
 * target LEs (set artificially low). Software matching is exercised while the
 * receiver catches up. Matching is a hybrid of SW/HW as threads race to
 * finish.
 *
 * Run with driver le_pool_max set below FC_TRANS.
 */
Test(tagged, fc_mt)
{
	pthread_t send_thread;
	pthread_t recv_thread;
	pthread_attr_t attr;
	int ret;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	ret = pthread_create(&send_thread, &attr, fc_sender, NULL);
	cr_assert_eq(ret, 0);

	ret = pthread_create(&recv_thread, &attr, fc_recver, NULL);
	cr_assert_eq(ret, 0);

	ret = pthread_join(recv_thread, NULL);
	cr_assert_eq(ret, 0);

	ret = pthread_join(send_thread, NULL);
	cr_assert_eq(ret, 0);

	pthread_attr_destroy(&attr);
}

#define RDZV_FC_ITERS 100
#define RDZV_FC_BATCH 5

static void *rdzv_fc_sender(void *data)
{
	int i, j, tx_ret;
	int send_id;
	uint8_t *send_bufs;
	uint8_t *send_buf;
	long send_len = (long)data;
	struct fi_cq_tagged_entry tx_cqe;
	int batch_size = RDZV_FC_BATCH;

	send_bufs = aligned_alloc(C_PAGE_SIZE, send_len * batch_size);
	cr_assert(send_bufs);

	for (i = 0; i < RDZV_FC_ITERS; i++) {
		for (j = 0; j < batch_size; j++) {
			send_id = i * batch_size + j;
			send_buf = &send_bufs[j * send_len];
			memset(send_buf, send_id, send_len);

			do {
				tx_ret = fi_tsend(cxit_ep, send_buf, send_len,
						  NULL, cxit_ep_fi_addr,
						  send_id, NULL);

				if (tx_ret == -FI_EAGAIN) {
					fi_cq_read(cxit_tx_cq, &tx_cqe, 0);
					sched_yield();
				}
			} while (tx_ret == -FI_EAGAIN);

			cr_assert_eq(tx_ret, FI_SUCCESS, "fi_tsend failed %d",
				     tx_ret);
		}

		for (j = 0; j < batch_size; j++) {
			do {
				tx_ret = fi_cq_read(cxit_tx_cq, &tx_cqe, 1);

				if (tx_ret == -FI_EAGAIN)
					sched_yield();
			} while (tx_ret == -FI_EAGAIN);

			cr_assert_eq(tx_ret, 1,
				     "fi_cq_read unexpected value %d",
				     tx_ret);

			validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
		}
	}

	free(send_bufs);

	pthread_exit(NULL);
}

static void *rdzv_fc_recver(void *data)
{
	int i, j, k, ret;
	int recv_id;
	uint8_t *recv_bufs;
	uint8_t *recv_buf;
	long recv_len = (long)data;
	struct fi_cq_tagged_entry rx_cqe;
	int batch_size = RDZV_FC_BATCH;

	recv_bufs = aligned_alloc(C_PAGE_SIZE, recv_len * batch_size);
	cr_assert(recv_bufs);

	/* Let Sender get ahead and land some UX messages */
	sleep(1);

	for (i = 0; i < RDZV_FC_ITERS; i++) {

		for (j = 0; j < batch_size; j++) {
			recv_id = i * batch_size + j;
			recv_buf = &recv_bufs[j * recv_len];
			memset(recv_buf, -1, recv_len);

			do {
				ret = fi_trecv(cxit_ep, recv_buf, recv_len,
					       NULL, FI_ADDR_UNSPEC, recv_id,
					       0, NULL);

				if (ret == -FI_EAGAIN) {
					fi_cq_read(cxit_rx_cq, &rx_cqe, 0);
					sched_yield();
				}
			} while (ret == -FI_EAGAIN);

			cr_assert_eq(ret, FI_SUCCESS, "fi_trecv failed %d",
				     ret);

			do {
				ret = fi_cq_read(cxit_rx_cq, &rx_cqe, 1);

				if (ret == -FI_EAGAIN)
					sched_yield();
			} while (ret == -FI_EAGAIN);

			cr_assert_eq(ret, 1, "fi_cq_read unexpected value %d",
				     ret);

			validate_rx_event(&rx_cqe, NULL, recv_len,
					  FI_TAGGED | FI_RECV,
					  NULL, 0, rx_cqe.tag);

			recv_id = rx_cqe.tag % batch_size;
			recv_buf = &recv_bufs[recv_id * recv_len];
			for (k = 0; k < recv_len; k++) {
				cr_assert_eq(recv_buf[k], (uint8_t)rx_cqe.tag,
					     "data mismatch, element[%d], exp=%d saw=%d\n",
					     k, (uint8_t)rx_cqe.tag,
					     recv_buf[k]);
			}
		}
	}

	free(recv_bufs);

	pthread_exit(NULL);
}

/*
 * Rendezvous Send multi-threaded flow control test.
 *
 * Run with driver le_pool_max set just above RDZV_FC_BATCH.
 */
Test(tagged, rdzv_fc_mt, .timeout = 60)
{
	pthread_t send_thread;
	pthread_t recv_thread;
	pthread_attr_t attr;
	int ret;
	long xfer_len;

	pthread_attr_init(&attr);
	pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

	for (xfer_len = 64; xfer_len <= 4*1024; xfer_len <<= 2) {
		ret = pthread_create(&send_thread, &attr, rdzv_fc_sender,
				     (void *)xfer_len);
		cr_assert_eq(ret, 0);

		ret = pthread_create(&recv_thread, &attr, rdzv_fc_recver,
				     (void *)xfer_len);
		cr_assert_eq(ret, 0);

		ret = pthread_join(recv_thread, NULL);
		cr_assert_eq(ret, 0);

		ret = pthread_join(send_thread, NULL);
		cr_assert_eq(ret, 0);

		printf("%ld byte Sends complete\n", xfer_len);
	}

	pthread_attr_destroy(&attr);
}

Test(tagged, NC2192)
{
	int i, ret;
	uint8_t *recv_buf,
		*send_buf;
	int send_len = CXIP_RDZV_THRESHOLD - 1;
	int recv_len = send_len;
	struct fi_cq_tagged_entry tx_cqe,
				  rx_cqe;
	fi_addr_t from;
	int sends = (CXIP_OFLOW_BUF_SIZE - CXIP_RDZV_THRESHOLD) / send_len + 1;

	recv_buf = aligned_alloc(C_PAGE_SIZE, recv_len);
	cr_assert(recv_buf);
	memset(recv_buf, 0, recv_len);

	send_buf = aligned_alloc(C_PAGE_SIZE, send_len);
	cr_assert(send_buf);

	/* Consume 1 oflow byte */
	ret = fi_tsend(cxit_ep, send_buf, 1, NULL, cxit_ep_fi_addr, 0,
		       NULL);
	cr_assert(ret == FI_SUCCESS);

	for (i = 0; i < sends; i++) {
		ret = fi_tsend(cxit_ep, send_buf, send_len, NULL,
			       cxit_ep_fi_addr, 1, NULL);
		cr_assert(ret == FI_SUCCESS);
	}

	for (i = 0; i < sends + 1; i++) {
		ret = cxit_await_completion(cxit_tx_cq, &tx_cqe);
		cr_assert(ret == 1);

		validate_tx_event(&tx_cqe, FI_TAGGED | FI_SEND, NULL);
	}

	for (i = 0; i < sends; i++) {
		ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
			       FI_ADDR_UNSPEC, 1, 0, NULL);
		cr_assert(ret == FI_SUCCESS);
	}

	for (i = 0; i < sends; i++) {
		/* Wait for async event indicating data has been received */
		do {
			ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
		} while (ret == -FI_EAGAIN);
		cr_assert(ret == 1);

		validate_rx_event(&rx_cqe, NULL, send_len, FI_TAGGED | FI_RECV,
				  NULL, 0, 1);
		cr_assert(from == cxit_ep_fi_addr, "Invalid source address");
	}

	/* Match the 1 byte Send */
	ret = fi_trecv(cxit_ep, recv_buf, recv_len, NULL,
		       FI_ADDR_UNSPEC, 0, 0, NULL);
	cr_assert(ret == FI_SUCCESS);

	do {
		ret = fi_cq_readfrom(cxit_rx_cq, &rx_cqe, 1, &from);
	} while (ret == -FI_EAGAIN);
	cr_assert(ret == 1);

	validate_rx_event(&rx_cqe, NULL, 1, FI_TAGGED | FI_RECV, NULL, 0, 0);
	cr_assert(from == cxit_ep_fi_addr, "Invalid source address");

	free(send_buf);
	free(recv_buf);
}
