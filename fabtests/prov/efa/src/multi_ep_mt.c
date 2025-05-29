/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

/**
 * Multi-ep multi-threaded test
 * The client creates N+1 (specified by -c)
 * threads. The first N threads have their
 * own EP/AV created, do some
 * sends with given number of iterations and
 * size, and destroy EP/AV without waiting for
 * completions. Each thread can join and
 * exit in random timeline, realized by
 * the random nanosleep at the beginning
 * and the end of the thread function.
 * The last (N+1) thread polls a persistent
 * CQ that are bound to the EPs in the first
 * N threads, in the given timeout (-T)
 *
 * The server has 1 thread that posts recv
 * and posts CQ in the given timeout (-T).
 */

#include <getopt.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include "hmem.h"
#include "shared.h"
#include <pthread.h>

static struct fid_ep **eps;
static char **send_bufs, **recv_bufs;
static struct fid_mr **send_mrs, **recv_mrs;
static void **send_descs, **recv_descs;
static struct fi_context2 *recv_ctx;
static struct fi_context2 *send_ctx;
static struct fid_av **avs;
static fi_addr_t *remote_fiaddr;
static size_t num_eps = 3;
char remote_raw_addr[FT_MAX_CTRL_MSG];

static bool shared_av = false;
int num_avs;

void close_client(int i);
int open_client(int i);

#define RANDOM_MIN 0
#define RANDOM_MAX 999999999

struct thread_context {
	int idx;
	pthread_t thread;
};

struct thread_context *contexts_ep;
struct thread_context context_cq;

static void free_ep_res()
{
	int i;

	for (i = 0; i < num_eps; i++) {
		FT_CLOSE_FID(send_mrs[i]);
		FT_CLOSE_FID(recv_mrs[i]);
		FT_CLOSE_FID(eps[i]);

		(void) ft_hmem_free(opts.iface, (void *) send_bufs[i]);
		(void) ft_hmem_free(opts.iface, (void *) recv_bufs[i]);
	}

	for (i = 0; i < num_avs; i++) {
		FT_CLOSE_FID(avs[i]);
	}

	free(send_bufs);
	free(recv_bufs);
	free(send_mrs);
	free(recv_mrs);
	free(send_descs);
	free(recv_descs);
	free(send_ctx);
	free(recv_ctx);
	free(remote_fiaddr);
	free(eps);
	free(avs);
}

static int reg_mrs(void)
{
	int i, ret;

	for (i = 0; i < num_eps; i++) {
		ret = ft_reg_mr(fi, send_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 1) * (i + 1), opts.iface,
				opts.device, &send_mrs[i], &send_descs[i]);
		if (ret)
			return ret;

		ret = ft_reg_mr(fi, recv_bufs[i], opts.transfer_size,
				ft_info_to_mr_access(fi),
				(FT_MR_KEY + 2) * (i + 2), opts.iface,
				opts.device, &recv_mrs[i], &recv_descs[i]);
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

static int alloc_multi_ep_res()
{
	int i, ret;
	size_t alloc_size;
	struct fi_av_attr av_attr = {0};

	eps = calloc(num_eps, sizeof(*eps));
	remote_fiaddr = calloc(num_eps, sizeof(*remote_fiaddr));
	send_mrs = calloc(num_eps, sizeof(*send_mrs));
	recv_mrs = calloc(num_eps, sizeof(*recv_mrs));
	send_descs = calloc(num_eps, sizeof(*send_descs));
	recv_descs = calloc(num_eps, sizeof(*recv_descs));
	send_ctx = calloc(num_eps, sizeof(*send_ctx));
	recv_ctx = calloc(num_eps, sizeof(*recv_ctx));
	send_bufs = calloc(num_eps, opts.transfer_size);
	recv_bufs = calloc(num_eps, opts.transfer_size);

	avs = calloc(num_avs, sizeof(*avs));

	/* Open shared AV */
	if (shared_av) {
		ret = fi_av_open(domain, &av_attr, &avs[0], NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			return ret;
		}
	}

	if (!eps || !remote_fiaddr || !send_bufs || !recv_bufs || !send_ctx ||
	    !recv_ctx || !send_bufs || !recv_bufs || !send_mrs || !recv_mrs ||
	    !send_descs || !recv_descs)
		return -FI_ENOMEM;

	alloc_size = opts.transfer_size < FT_MAX_CTRL_MSG ? FT_MAX_CTRL_MSG : opts.transfer_size;
	for (i = 0; i < num_eps; i++) {
		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &send_bufs[i],
				    alloc_size);
		if (ret)
			return ret;

		ret = ft_hmem_alloc(opts.iface, opts.device,
				    (void **) &recv_bufs[i],
				    alloc_size);
		if (ret)
			return ret;
	}

	return 0;
}

static int get_one_comp(struct fid_cq *cq)
{
	struct fi_cq_err_entry comp;
	struct fi_cq_err_entry cq_err;

	memset(&cq_err, 0, sizeof(cq_err));
	int ret;

	do {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0)
			break;

		if (ret < 0) {
			if (ret != -FI_EAGAIN) {
				printf("fi_cq_read returns error %d\n", ret);
				(void) fi_cq_readerr(cq, &cq_err, 0);
			}
			return ret;
		}
	} while (1);

	return FI_SUCCESS;
}

static void *post_sends(void *context)
{
	int idx, ret, i, j;
	size_t len;
	// the range of the sleep time (in nanoseconds)
	int sleep_time;
	struct timespec ts;
	int num_transient_eps = 10;

	srand(time(NULL));
	sleep_time = (rand() % (RANDOM_MAX - RANDOM_MIN + 1)) + RANDOM_MIN;
	idx = ((struct thread_context *) context)->idx;
	ts.tv_nsec = sleep_time;

	nanosleep(&ts, NULL);
	len = opts.transfer_size;
	for (j = 0; j < num_transient_eps; j++) {

		printf("Thread %d: opening client \n", idx);
		ret = open_client(idx);
		if (ret) {
			FT_PRINTERR("open client failed!\n", ret);
			return NULL;
		}

		for (i = 0; i < opts.iterations / num_transient_eps; i++) {
			printf("Thread %d: post send for ep %d \n", idx, idx);
			ret = ft_post_tx_buf(eps[idx], remote_fiaddr[idx], len, NO_CQ_DATA, &send_ctx[idx], send_bufs[idx], send_descs[idx], ft_tag);
			if (ret) {
				FT_PRINTERR("ft_post_tx_buf", ret);
				return NULL;
			}
		}

		sleep_time = (rand() % (RANDOM_MAX - RANDOM_MIN + 1)) + RANDOM_MIN;
		ts.tv_nsec = sleep_time;
		nanosleep(&ts, NULL);

		// exit
		printf("Thread %d: closing client\n", idx);
		close_client(idx);
	}
	return NULL;
}

static void *poll_tx_cq(void *context)
{
	int i, ret;
	int num_cqes = 0;
	struct timespec a, b;

	i = ((struct thread_context *) context)->idx;

	clock_gettime(CLOCK_MONOTONIC, &a);
	while (true) {
		clock_gettime(CLOCK_MONOTONIC, &b);
		if ((b.tv_sec - a.tv_sec) > timeout) {
			printf("%ds timeout expired, exiting \n", timeout);
			break;
		}
		ret = get_one_comp(txcq);
		if (ret)
			continue;
		num_cqes++;
		printf("Client: thread %d get %d completion from tx cq \n", i,
		       num_cqes);
		// This is the maximal number of sends client will do
		if (num_cqes == num_eps * opts.iterations)
			break;
	}

	return NULL;
}

static int run_server(void)
{
	int i, ret;
	int num_cqes = 0;

	// posting enough recv buffers for each ep
	// so the sent pkts can at least find a match
	for (i = 0; i < opts.iterations * num_eps; i++) {
		printf("Server: posting recv\n");
		ret = ft_post_rx_buf(ep, FI_ADDR_UNSPEC, opts.transfer_size, &rx_ctx, rx_buf, mr_desc, ft_tag);
		if (ret) {
			FT_PRINTERR("ft_post_rx_buf", ret);
			return ret;
		}
	}

	printf("Server: wait for completions\n");
	struct timespec a, b;

	clock_gettime(CLOCK_MONOTONIC, &a);
	while (true) {
		clock_gettime(CLOCK_MONOTONIC, &b);
		if ((b.tv_sec - a.tv_sec) > timeout) {
			printf("%ds timeout expired, exiting...\n", timeout);
			break;
		}
		ret = get_one_comp(rxcq);
		// ignore cq error
		if (ret)
			continue;
		num_cqes++;
		printf("Server: Get %d completions from rx cq\n", num_cqes);
		// This is the maximal number of sends client will do
		if (num_cqes == num_eps * opts.iterations)
			break;
	}

	printf("Server: PASSED multi ep recvs\n");
	return FI_SUCCESS;
}

static int run_client(void)
{
	char temp[FT_MAX_CTRL_MSG];
	struct fi_rma_iov *rma_iov = (struct fi_rma_iov *) temp;
	int i, ret;
	size_t key_size, len;

	len = opts.transfer_size;

	for (i = 0; i < num_eps; i++) {
		ret = ft_fill_rma_info(recv_mrs[i], recv_bufs[i], rma_iov,
				       &key_size, &len);
		if (ret)
			return ret;

		ret = ft_hmem_copy_to(opts.iface, opts.device, send_bufs[i],
				      rma_iov, len);
		if (ret)
			return ret;
	}

	contexts_ep = calloc(num_eps, sizeof(*contexts_ep));
	for (i = 0; i < num_eps; i++) {
		contexts_ep[i].idx = i;
	}

	context_cq.idx = num_eps;

	pthread_create(&context_cq.thread, NULL, poll_tx_cq, &context_cq);

	for (i = 0; i < num_eps; i++) {
		ret = pthread_create(&contexts_ep[i].thread, NULL, post_sends,
				     &contexts_ep[i]);
		if (ret)
			printf("Client: thread %d post_sends create failed: "
			       "%d\n",
			       i, ret);
	}

	for (i = 0; i < num_eps; i++)
		pthread_join(contexts_ep[i].thread, NULL);

	pthread_join(context_cq.thread, NULL);

	/* For a shared AV, close it at the end */
	if (shared_av)
		FT_CLOSE_FID(avs[0]);

	printf("Client: PASSED multi ep sends\n");
	free(contexts_ep);
	return 0;
}

int open_client(int idx)
{
	int ret;
	struct fid_av *av;

	if (opts.av_name) {
		av_attr.name = opts.av_name;
	}
	av_attr.count = opts.av_size;

	ret = fi_endpoint(domain, fi, &eps[idx], NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	/* ft_enable_ep bind the ep with cq and av before enabling */
	if (shared_av) {
		av = avs[0];
	} else {
		av = avs[idx];

		ret = fi_av_open(domain, &av_attr, &av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			return ret;
		}
	}

	ret = ft_enable_ep(eps[idx], eq, av, txcq, rxcq, NULL, NULL, NULL);
	if (ret)
		return ret;

	/* Use the same remote addr we got from the persistent receiver ep */
	ret = ft_av_insert(av, (void *)remote_raw_addr, 1, &remote_fiaddr[idx], 0, NULL);
	if (ret)
		return ret;

	return 0;
}

void close_client(int i)
{
	printf("closing ep %d, av %d\n", i, i);
	FT_CLOSE_FID(eps[i]);

	/* Close AV when it's 1 AV per client thread */
	if (!shared_av)
		FT_CLOSE_FID(avs[i]);
}

int exchange_addresses_oob(struct fid_av *av_ptr, struct fid_ep *ep_ptr,
			   fi_addr_t *remote_addr, void *addr)
{
	int ret;
	size_t addrlen = FT_MAX_CTRL_MSG;

	ret = fi_getname(&ep_ptr->fid, addr, &addrlen);
	if (ret) {
		FT_PRINTERR("fi_getname", ret);
		return ret;
	}

	ret = ft_sock_send(oob_sock, addr, FT_MAX_CTRL_MSG);
	if (ret)
		return ret;

	ret = ft_sock_recv(oob_sock, addr, FT_MAX_CTRL_MSG);
	if (ret)
		return ret;

	ret = ft_av_insert(av_ptr, addr, 1, remote_addr, 0, NULL);
	if (ret)
		return ret;

	return 0;
}

int init_fabric(void)
{
	int ret;

	ret = ft_init();
	if (ret)
		return ret;

	ret = ft_init_oob();
	if (ret)
		return ret;

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_enable_ep_recv();
	if (ret)
		return ret;

	/* We want to get the remote raw addr so we use our own OOB exchange
	 * function */
	ret = exchange_addresses_oob(av, ep, &remote_fi_addr, remote_raw_addr);
	if (ret)
		return ret;

	return 0;
}

static int run_test(void)
{
	int ret;

	if (shared_av)
		num_avs = 1;
	else
		num_avs = num_eps;

	opts.av_size = num_eps + 1;
	ret = init_fabric();
	if (ret)
		return ret;

	ret = alloc_multi_ep_res();
	if (ret)
		return ret;

	ret = reg_mrs();
	if (ret)
		goto out;

	if (opts.dst_addr)
		ret = run_client();
	else
		ret = run_server();

	if (ret)
		goto out;

out:
	free_ep_res();
	return ret;
}

int main(int argc, char **argv)
{
	int op;
	int ret = 0;

	opts = INIT_OPTS;
	opts.transfer_size = 256;
	opts.options |= FT_OPT_OOB_ADDR_EXCH;
	timeout = 10;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv,
				 "c:hT:A" ADDR_OPTS INFO_OPTS CS_OPTS, long_opts,
				 &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		case 'c':
			num_eps = atoi(optarg);
			break;
		case 'T':
			timeout = atoi(optarg);
			break;
		case 'A':
			shared_av = true;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0],
				 "Multi-threading Multi endpoint test");
			FT_PRINT_OPTS_USAGE("-c <int>",
					    "number of endpoints to create and "
					    "test (def 3)");
			FT_PRINT_OPTS_USAGE("-T <timeout>",
					    "seconds for the cq poll on both "
					    "sides (default 5)");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	opts.threading = FI_THREAD_SAFE;
	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	ret = run_test();

	ft_free_res();
	return ft_exit_code(ret);
}
