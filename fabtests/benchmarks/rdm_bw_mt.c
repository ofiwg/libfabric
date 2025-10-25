/*
* Copyright (c) Intel Corporation. All rights reserved.
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

/*
 * rdm_bw_mt.c
 * This is a mostly stand-alone test.
 * It does not use any common fabtests init or destroy code but is intended to
 * be run in the fabtest suite. In order to make this test use common code it
 * must be refactored to remove the global fi resources and put them into
 * objects. These objects can be struct ft_fabric_resources which will contain
 * all resources that can be opened under a single fabtest and
 * struct ft_buffer_resouces for all resources associated with a buffer.
 * For the case of this test it will be:
 *
 * 				fabric
 *			 _________|_________
 * 			/	  |	    \
 * 		    thread_0   thread_1   thread_n
 *                 ____|____
 *		  /	    \
 *	fabric_resources buffer_resources
 *         _____|_____	        _____|_____
 * 	  /           \        /           \
 * 	domain, ep, cq...     buf, mr_desc, mr...
 *
 * Each thread must have its own domain, ep, cq, av, cntr, buffer_reources, etc.
 * The buffer_resources will contain the buffer, memory registration, and any
 * other miscellaneous resources any buffer might need.
 * There is an implicit 1:1 ratio where each thread's fabric resources are only
 * associated with that thread's buffer resources. There are no resources other
 * than fabric that are shared between threads.
 * This test comes with a TODO to refactor the common code to support more tests
 * of this type and enable easier development of future tests with more
 * compatible objects instead of global variables.
 * WARNING: Not all options are supported in this test!
 */

#include <pthread.h>
#include <rdma/fi_cm.h>

#include "shared.h"
#include "benchmark_shared.h"
#include "hmem.h"

#define BUFFER_SIZE 1024
static char oob_buffer[BUFFER_SIZE];

static size_t num_eps = 1;
static bool bidir = false;
static ssize_t xfer_size = 1;
pthread_barrier_t barrier;

struct thread_args {
	pthread_t thread;
	fi_addr_t fiaddr;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_cq *cq;
	struct fid_av *av;
	struct fid_mr *tx_mr;
	struct fid_mr *rx_mr;
	void *tx_mr_desc;
	void *rx_mr_desc;
	struct fi_context2 send_ctx;
	struct fi_context2 recv_ctx;
	char *tx_buf;
	char *rx_buf;
	int id;
	int ret;
};

static struct thread_args *targs = NULL;

static void cleanup_ofi(void)
{
	int ret;
	int i;

	for (i = 0; i < num_eps; i++) {
		if (targs[i].ep) {
			ret = fi_close(&targs[i].ep->fid);
			if (ret)
				printf("fi_close(ep[%d]) failed: %d\n", i, ret);
		}

		if (targs[i].cq) {
			ret = fi_close(&targs[i].cq->fid);
			if (ret)
				printf("fi_close(cq[%d]) failed: %d\n", i, ret);
		}

		if (targs[i].av) {
			ret = fi_close(&targs[i].av->fid);
			if (ret)
				printf("fi_close(av[%d]) failed: %d\n", i, ret);
		}

		if (targs[i].domain) {
			ret = fi_close(&targs[i].domain->fid);
			if (ret)
				printf("fi_close(domain[%d]) failed: %d\n", i,
					ret);
		}
	}

	if (fabric) {
		ret = fi_close(&fabric->fid);
		if (ret)
			printf("fi_close(fabric) failed: %d\n", ret);
	}

	if (fi)
		fi_freeinfo(fi);
	if (hints)
		fi_freeinfo(hints);
	if (targs)
		free(targs);
}

static int init_av(int i)
{
	int ret;
	size_t len = BUFFER_SIZE;
	char print_buf[BUFFER_SIZE];

	memset(print_buf, 0, BUFFER_SIZE);

	ret = fi_getname(&targs[i].ep->fid, oob_buffer, &len);
	if (ret) {
		printf("fi_getname failed: %d\n", ret);
		return ret;
	}

	len = BUFFER_SIZE;

	ret = ft_sock_send(oob_sock, oob_buffer, len);
	if (ret) {
		printf("ft_sock_send failed: %d\n", ret);
		return ret;
	}

	ret = ft_sock_recv(oob_sock, oob_buffer, len);
	if (ret) {
		printf("ft_sock_recv failed: %d\n", ret);
		return ret;
	}

	ret = fi_av_insert(targs[i].av, oob_buffer, 1, &targs[i].fiaddr, 0, NULL);
	if (ret != 1) {
		printf("fi_av_insert failed: %d\n", ret);
		return ret;
	}

	return 0;
}

static int init_ofi(void)
{
	struct fi_cq_attr cq_attr;
	struct fi_av_attr av_attr;
	struct fi_cntr_attr cntr_attr;
	int ret = FI_SUCCESS, i;

        ret = fi_fabric(fi->fabric_attr, &fabric, NULL);
        if (ret) {
		printf("fi_fabric failed: %d\n", ret);
                return ret;
	}

	targs = calloc(num_eps, sizeof(*targs));
	if (!targs) {
		printf("thread_args calloc failed\n");
		return -FI_ENOMEM;
	}

	for (i = 0; i < num_eps; i++) {
		memset(&cq_attr, 0, sizeof(cq_attr));
		memset(&av_attr, 0, sizeof(av_attr));
		memset(&cntr_attr, 0, sizeof(cntr_attr));

		ret = fi_domain(fabric, fi, &targs[i].domain, NULL);
		if (ret) {
			printf("fi_domain failed ep[%d]: %d\n", i, ret);
			return ret;
		}

		ret = fi_endpoint(targs[i].domain, fi, &targs[i].ep, NULL);
		if (ret) {
			printf("fi_endpoint failed: %d\n", ret);
			return ret;
		}

		cq_attr.size = 128;
		cq_attr.format = FI_CQ_FORMAT_CONTEXT;
		ret = fi_cq_open(targs[i].domain, &cq_attr, &targs[i].cq, NULL);
		if (ret) {
			printf("fi_cq_open failed: %d\n", ret);
			return ret;
		}

		av_attr.type = FI_AV_UNSPEC;
		av_attr.count = 1;
		ret = fi_av_open(targs[i].domain, &av_attr, &targs[i].av, NULL);
		if (ret) {
			printf("fi_av_open failed: %d\n", ret);
			return ret;
		}

		ret = fi_ep_bind(targs[i].ep, &targs[i].av->fid, 0);
		if (ret) {
			printf("fi_ep_bind av failed: %d\n", ret);
			return ret;
		}

		ret = fi_ep_bind(targs[i].ep, &targs[i].cq->fid, FI_TRANSMIT | FI_RECV);
		if (ret) {
			printf("fi_ep_bind cq failed: %d\n", ret);
			return ret;
		}

		ret = fi_enable(targs[i].ep);
		if (ret) {
			printf("fi_enable failed: %d\n", ret);
			return ret;
		}
	}

	return ret;
}


static int mt_reg_mr(struct fi_info *fi, void *buf, size_t size,
		     uint64_t access, uint64_t key, enum fi_hmem_iface iface,
		     uint64_t device, struct fid_domain *dom,
		     struct fid_ep *endpoint, struct fid_mr **mr, void **desc)
{
	struct fi_mr_attr attr = {0};
	struct iovec iov = {0};
	int ret;
	uint64_t flags;
	int dmabuf_fd;
	uint64_t dmabuf_offset;
	struct fi_mr_dmabuf dmabuf = {0};

	if (!ft_need_mr_reg(fi))
		return 0;

	iov.iov_base = buf;
	iov.iov_len = size;

	flags = (iface) ? FI_HMEM_DEVICE_ONLY : 0;

	if (opts.options & FT_OPT_REG_DMABUF_MR) {
		ret = ft_hmem_get_dmabuf_fd(iface, buf, size,
					    &dmabuf_fd, &dmabuf_offset);
		if (ret)
			return ret;

		dmabuf.fd = dmabuf_fd;
		dmabuf.offset = dmabuf_offset;
		dmabuf.len = size;
		dmabuf.base_addr = (void *)((uintptr_t) buf - dmabuf_offset);
		flags |= FI_MR_DMABUF;
	}

	ft_fill_mr_attr(&iov, &dmabuf, 1, access, key, iface, device, &attr,
			flags);
	ret = fi_mr_regattr(dom, &attr, flags, mr);
	if (opts.options & FT_OPT_REG_DMABUF_MR)
		ft_hmem_put_dmabuf_fd(iface, dmabuf_fd);
	if (ret)
		return ret;

	if (desc)
		*desc = fi_mr_desc(*mr);

	if (fi->domain_attr->mr_mode & FI_MR_ENDPOINT) {
		ret = fi_mr_bind(*mr, &endpoint->fid, 0);
		if (ret)
			return ret;

		ret = fi_mr_enable(*mr);
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

static int reg_mrs(struct thread_args *targs)
{
	int ret = FI_SUCCESS;

	ret = ft_hmem_alloc(opts.iface, opts.device, (void **) &(targs->tx_buf),
			    xfer_size);
	if (ret) {
		printf("ft_hmem_alloc tx %d failed: %d\n", targs->id, ret);
		return ret;
	}

	ret = ft_hmem_alloc(opts.iface, opts.device, (void **) &(targs->rx_buf),
			    xfer_size);
	if (ret) {
		printf("ft_hmem_alloc rx %d failed: %d\n", targs->id, ret);
		return ret;
	}

	ret = mt_reg_mr(fi, (void *) targs->tx_buf, xfer_size, FI_SEND,
			targs->id, opts.iface, opts.device,
			targs->domain, targs->ep,
			&targs->tx_mr, &targs->tx_mr_desc);
	if (ret) {
		printf("fi_mr_reg tx %d failed: %d\n", targs->id, ret);
		return ret;
	}

	ret = mt_reg_mr(fi, (void *) targs->rx_buf, xfer_size, FI_RECV,
			targs->id + 0xDAD, opts.iface, opts.device,
			targs->domain, targs->ep,
			&targs->rx_mr, &targs->rx_mr_desc);
	if (ret)
		printf("fi_mr_reg rx %d failed: %d\n", targs->id, ret);

	return ret;
}

static void force_progress(struct fid_cq *cq)
{
	(void) fi_cq_read(cq, NULL, 0);
}

static int read_cq(struct fid_cq *cqueue)
{
	struct fi_cq_entry cq_entry;
	int ret;

	do {
		ret = fi_cq_read(cqueue, &cq_entry, 1);
		if (ret < 0 && ret != -FI_EAGAIN)
			return ret;
		if (ret == 1)
			return 0;
	} while (1);
}

static int post_send(void *context)
{
	int ret;
	struct thread_args *targs = context;

	do {
		ret = fi_send(targs->ep, targs->tx_buf, xfer_size,
			      targs->tx_mr, targs->fiaddr, &targs->send_ctx);
		if (ret != -FI_EAGAIN)
			return ret;

		force_progress(targs->cq);
	} while (1);
}

static int post_recv(void *context)
{
	int ret;
	struct thread_args *targs = context;

	do {
		ret = fi_recv(targs->ep, targs->rx_buf, xfer_size,
			      targs->rx_mr, targs->fiaddr, &targs->recv_ctx);
		if (ret != -FI_EAGAIN)
			return ret;

		force_progress(targs->cq);
	} while (1);
}

static int bw_send(void *context)
{
	int ret;
	struct thread_args *targs = context;

	ret = post_send(context);
	if (ret)
		return ret;

	ret = read_cq(targs->cq);
	if (ret) {
		printf("send read_cq error: %d\n", ret);
		return ret;
	}
	return 0;
}

static int bw_recv(void *context)
{
	int ret = FI_SUCCESS;
	struct thread_args *targs = context;

	ret = post_recv(context);
	if (ret)
		return ret;

	ret = read_cq(targs->cq);
	if (ret) {
		printf("recv read_cq error: %d\n", ret);
		return ret;
	}

	return 0;
}

static void *uni_bandwidth(void *context)
{
	int i, ret;
	const struct thread_args *targs = context;

	pthread_barrier_wait(&barrier);
	for (i = 0; i < opts.warmup_iterations; i++) {
		ret = opts.dst_addr ? bw_send(context) : bw_recv(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			printf("ep[%d] warmup failed iter %d\n", targs->id, i);
			break;
		}
	}

	pthread_barrier_wait(&barrier);
	if (targs->id == 0)
		ft_start();
	for (i = 0; i < opts.iterations; i++) {
		ret = opts.dst_addr ? bw_send(context) : bw_recv(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			break;
		}
	}
	pthread_barrier_wait(&barrier);
	if (targs->id == 0)
		ft_stop();

	return  NULL;
}

static void *bi_bandwidth(void *context)
{
	int i, ret;
	struct thread_args *targs = context;

	pthread_barrier_wait(&barrier);
	for (i = 0; i < opts.warmup_iterations; i++) {
		ret = opts.dst_addr ? bw_send(context) : bw_recv(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			printf("%d warmup failed iter %d\n", targs->id, i);
			break;
		}
		ret = opts.dst_addr ? bw_recv(context) : bw_send(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			printf("%d warmup failed iter %d\n", targs->id, i);
			break;
		}
	}

	pthread_barrier_wait(&barrier);
	if (targs->id == 0)
		ft_start();

	for (i = 0; i < opts.iterations; i++) {
		ret = opts.dst_addr ? bw_send(context) : bw_recv(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			printf("%d warmup failed iter %d\n", targs->id, i);
			break;
		}
		ret = opts.dst_addr ? bw_recv(context) : bw_send(context);
		if (ret) {
			((struct thread_args *) context)->ret = ret;
			printf("%d warmup failed iter %d\n", targs->id, i);
			break;
		}
	}
	pthread_barrier_wait(&barrier);
	if (targs->id == 0)
		ft_stop();

	return NULL;
}

static int run_size(void)
{
	int i, err, ret = FI_SUCCESS;

	for (i = 0; i < num_eps; i++) {
		targs[i].id = i;
		targs[i].ret = FI_SUCCESS;
		ret = reg_mrs(&targs[i]);
		if (ret)
			goto out;
	}

	for (i = 0; i < num_eps; i++) {
		ret = pthread_create(&targs[i].thread, NULL,
				     bidir ? bi_bandwidth : uni_bandwidth,
				     &targs[i]);
		if (ret) {
			printf("pthread_create failed: %d\n", ret);
			return ret;
		}
	}
	for (i = 0; i < num_eps; i++) {
		pthread_join(targs[i].thread, NULL);
		if (targs[i].ret) {
			ret = targs[i].ret;
			goto out;
		}
	}

	show_perf(NULL, xfer_size, opts.iterations, &start, &end, num_eps);

out:
	for (i = 0; i < num_eps; i++) {
		if (targs[i].tx_mr) {
			err = fi_close(&targs[i].tx_mr->fid);
			if (err)
				printf("fi_close(targs[%d].tx_mr) failed: %d\n",
					i, err);
		}
		if (targs[i].rx_mr) {
			err = fi_close(&targs[i].rx_mr->fid);
			if (err)
				printf("fi_close(targs[%d].rx_mr) failed: %d\n",
					i, err);
		}
		if (targs[i].tx_buf) {
			err = ft_hmem_free(opts.iface, targs[i].tx_buf);
			if (err)
				printf("ft_hmem_free tx %d failed: %d\n",
				       i, err);
		}
		if (targs[i].rx_buf) {
			err = ft_hmem_free(opts.iface, targs[i].rx_buf);
			if (err)
				printf("ft_hmem_free rx %d failed: %d\n",
				       i, err);
		}
	}

	return ret;
}

static int run_test(void)
{
	int i, ret;

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			xfer_size = test_size[i].size;
			ret = run_size();
			if (ret)
				return ret;
		}
	} else {
		xfer_size = opts.transfer_size;
		ret = run_size();
		if (ret)
			return ret;
	}

	return FI_SUCCESS;
}

static void usage(void)
{
	fprintf(stderr, "\nrdm_bw_mt test options:\n");
	FT_PRINT_OPTS_USAGE("-g", "enable bidirectional");
	FT_PRINT_OPTS_USAGE("-n <num endpoints>",
			    "number of endpoints (threads) to use");
	FT_PRINT_OPTS_USAGE("-U", "enable FI_DELIVERY_COMPLETE");
	fprintf(stderr, "Notice to user: Not all fabtests options are supported"
		" by this test. If something isn't working check if the option"
		" is supported before reporting a bug.\n");
}

int main(int argc, char **argv)
{
        int ret, op, i;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_CTRL;

        hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv, "gn:Uh" CS_OPTS INFO_OPTS API_OPTS
		BENCHMARK_OPTS, long_opts, &lopt_idx)) != -1) {
		switch (op) {
		default:
			if (!ft_parse_long_opts(op, optarg))
				continue;
			ft_parse_benchmark_opts(op, optarg);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ft_parse_api_opts(op, optarg, hints, &opts);
			break;
		case 'g':
			bidir = true;
			break;
		case 'n':
			num_eps = atoi(optarg);
			break;
		case 'U':
			hints->tx_attr->op_flags |= FI_DELIVERY_COMPLETE;
			break;
		case '?':
		case 'h':
			ft_csusage(argv[0], "Multi-Threaded Bandwidth test for "
				   "RDM endpoints.");
			ft_benchmark_usage();
			ft_longopts_usage();
			usage();
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->domain_attr->threading = FI_THREAD_DOMAIN;
	hints->caps = FI_MSG;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = opts.mr_mode;
	hints->addr_format = opts.address_format;

	if (opts.options & FT_OPT_ENABLE_HMEM) {
		hints->caps |= FI_HMEM;
		hints->domain_attr->mr_mode |= FI_MR_HMEM;
	}

        ret = ft_init_oob();
        if (ret)
                goto out;

	if (oob_sock >= 0 && opts.dst_addr) {
		ret = ft_sock_sync(oob_sock, 0);
		if (ret)
			return ret;
	}

	ret = ft_hmem_init(opts.iface);
	if (ret)
		FT_PRINTERR("ft_hmem_init", ret);

	ret = fi_getinfo(FT_FIVERSION, NULL, NULL, 0, hints, &fi);
	if (ret) {
		printf("fi_getinfo() failed: %d\n", ret);
		goto out;
	}

	ret = init_ofi();
	if (ret) {
		printf("init ofi failed\n");
                goto out;
	}

	if (oob_sock >= 0 && !opts.dst_addr) {
		ret = ft_sock_sync(oob_sock, 0);
		if (ret)
			return ret;
	}

	for (i = 0; i < num_eps; i++) {
		ret = init_av(i);
		if (ret) {
			printf("init_av[%d] failed\n", i);
			goto out;
		}
	}

	ret = pthread_barrier_init(&barrier, NULL, num_eps);
	if (ret)
		goto out;

	ret = run_test();

	pthread_barrier_destroy(&barrier);

out:
	cleanup_ofi();
	ft_close_oob();
        return ret;
}
