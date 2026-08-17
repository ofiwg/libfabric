/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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
 *
 * XPU GPU Direct Async (GDA) Test
 *
 * This test exercises the fi_xpu API for GPU-initiated fabric
 * operations. It exports EP, CQ, CNTR, AV addresses and MR
 * descriptors to device memory, then launches CUDA kernels that
 * call fi_xpu_send/fi_xpu_write/fi_xpu_read directly from the GPU.
 */

#include "hmem.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <rdma/fi_xpu.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <shared.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "efa_xpu_kernels.h"

/* XPU context and exported handles */
static struct fid_xpu_ctx *xpu_ctx;
static struct fid_xpu_ep xpu_ep;
static struct fid_xpu_cq xpu_send_cq;
static struct fid_xpu_cq xpu_recv_cq;
static struct fid_xpu_cntr xpu_send_cntr;
static struct fid_xpu_cntr xpu_recv_cntr;

/* Device-side copies of exported handles */
static struct fid_xpu_ep *dev_xpu_ep;
static struct fid_xpu_cq *dev_xpu_send_cq;
static struct fid_xpu_cq *dev_xpu_recv_cq;
static struct fid_xpu_cntr *dev_xpu_send_cntr;
static struct fid_xpu_cntr *dev_xpu_recv_cntr;

/* Device-side AV address and MR descriptor */
static void *dev_av_addr;
static size_t av_addr_size;
static void *dev_mr_desc;
static size_t mr_desc_size;

/* XPU context query attributes */
static struct fi_xpu_ctx_attr xpu_ctx_attr;

/* Test mode */
static int gda_op_is_write;
static int gda_op_is_read;
/* Device-side RMA opcode passed to the XPU kernel:
 * 0=send, 1=write, 2=writedata (write with imm), 3=read */
static int gda_op;
static bool use_cntr;

/*
 * Cooperative scope the kernels issue at, and the block size they run with.
 * Every thread of a scope issues its own operation, so the block size alone
 * decides how many operations a call produces and the scope decides how they are
 * batched: independently at FI_XPU_WORK_ITEM, or claimed and doorbelled together
 * by a warp at FI_XPU_SUBGROUP or by the whole block at FI_XPU_WORK_GROUP.
 */
static int gda_scope = FI_XPU_WORK_ITEM;
static int gda_threads = 1;

/*
 * --lat: run the ping-pong kernel instead of the pipelined one. The two measure
 * different things, so they are separate runs rather than two phases of one: the
 * bandwidth path keeps a window of operations in flight, while this one has a
 * single operation outstanding and waits for the peer's reply before posting
 * again, which is what makes the round trip the thing being timed.
 */
static bool run_latency;

/* A scope no build can make collective, used to check the refusal path. */
#define GDA_SCOPE_UNKNOWN 0x7fff

enum {
	LONG_OPT_USE_CNTR = 1000,
	LONG_OPT_SCOPE,
	LONG_OPT_THREADS,
	LONG_OPT_LAT,
};

#define XPU_LOG(fmt, ...)						\
	do {								\
		printf("[fi_efa_xpu] " fmt "\n", ##__VA_ARGS__);	\
		fflush(stdout);						\
	} while (0)

static int create_xpu_ctx(void)
{
	int ret;
	struct fi_xpu_attr xpu_attr = {0};

	xpu_attr.iface = FI_HMEM_CUDA;
	xpu_attr.device = opts.device;
	/* ops = NULL means provider manages memory */
	xpu_attr.ops = NULL;

	ret = fi_xpu_ctx(domain, &xpu_attr, &xpu_ctx, NULL);
	if (ret) {
		FT_PRINTERR("fi_xpu_ctx", -ret);
		return ret;
	}

	ret = fi_xpu_ctx_query(xpu_ctx, &xpu_ctx_attr);
	if (ret) {
		FT_PRINTERR("fi_xpu_ctx_query", -ret);
		return ret;
	}

	av_addr_size = xpu_ctx_attr.av_addr_size;
	mr_desc_size = xpu_ctx_attr.mr_desc_size;

	FT_DEBUG("XPU ctx caps=0x%lx av_addr_size=%zu mr_desc_size=%zu\n",
		 xpu_ctx_attr.caps, av_addr_size, mr_desc_size);

	return 0;
}

static int export_ep(void)
{
	int ret;

	ret = fi_ep_export_xpu(ep, 0, &xpu_ep);
	if (ret) {
		FT_PRINTERR("fi_ep_export_xpu", -ret);
		return ret;
	}

	/* Copy exported handle to device memory */
	if (ft_cuda_alloc(opts.device, (void **)&dev_xpu_ep, sizeof(xpu_ep)) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for xpu_ep failed\n");
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_xpu_ep, &xpu_ep, sizeof(xpu_ep)) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for xpu_ep failed\n");
		return -FI_EIO;
	}

	XPU_LOG("  dev_xpu_ep=%p", (void *) dev_xpu_ep);
	return 0;
}

static int export_cqs(void)
{
	int ret;

	ret = fi_cq_export_xpu(txcq, FI_XPU, &xpu_send_cq);
	if (ret) {
		FT_PRINTERR("fi_cq_export_xpu (send)", -ret);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, (void **)&dev_xpu_send_cq, sizeof(xpu_send_cq)) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for xpu_send_cq failed\n");
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_xpu_send_cq, &xpu_send_cq, sizeof(xpu_send_cq)) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for xpu_send_cq failed\n");
		return -FI_EIO;
	}

	ret = fi_cq_export_xpu(rxcq, FI_XPU, &xpu_recv_cq);
	if (ret) {
		FT_PRINTERR("fi_cq_export_xpu (recv)", -ret);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, (void **)&dev_xpu_recv_cq, sizeof(xpu_recv_cq)) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for xpu_recv_cq failed\n");
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_xpu_recv_cq, &xpu_recv_cq, sizeof(xpu_recv_cq)) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for xpu_recv_cq failed\n");
		return -FI_EIO;
	}

	XPU_LOG("  dev_xpu_send_cq=%p dev_xpu_recv_cq=%p",
		(void *) dev_xpu_send_cq, (void *) dev_xpu_recv_cq);
	return 0;
}

static int export_cntrs(void)
{
	int ret;

	ret = fi_cntr_export_xpu(txcntr, FI_XPU, &xpu_send_cntr);
	if (ret) {
		FT_PRINTERR("fi_cntr_export_xpu (send)", -ret);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, (void **)&dev_xpu_send_cntr, sizeof(xpu_send_cntr)) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for xpu_send_cntr failed\n");
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_xpu_send_cntr, &xpu_send_cntr, sizeof(xpu_send_cntr)) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for xpu_send_cntr failed\n");
		return -FI_EIO;
	}

	ret = fi_cntr_export_xpu(rxcntr, FI_XPU, &xpu_recv_cntr);
	if (ret) {
		FT_PRINTERR("fi_cntr_export_xpu (recv)", -ret);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, (void **)&dev_xpu_recv_cntr, sizeof(xpu_recv_cntr)) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for xpu_recv_cntr failed\n");
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_xpu_recv_cntr, &xpu_recv_cntr, sizeof(xpu_recv_cntr)) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for xpu_recv_cntr failed\n");
		return -FI_EIO;
	}

	XPU_LOG("  dev_xpu_send_cntr=%p dev_xpu_recv_cntr=%p",
		(void *) dev_xpu_send_cntr, (void *) dev_xpu_recv_cntr);
	return 0;
}

static int export_av_addr(void)
{
	int ret;
	void *addr_buf;
	size_t len;

	addr_buf = calloc(1, av_addr_size);
	if (!addr_buf)
		return -FI_ENOMEM;

	len = av_addr_size;
	ret = fi_av_lookup2(av, remote_fi_addr, addr_buf, &len, FI_XPU,
			    xpu_ctx);
	if (ret) {
		FT_PRINTERR("fi_av_lookup2", -ret);
		free(addr_buf);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, &dev_av_addr, av_addr_size) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for av_addr failed\n");
		free(addr_buf);
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_av_addr, addr_buf, av_addr_size) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for av_addr failed\n");
		free(addr_buf);
		return -FI_EIO;
	}

	free(addr_buf);
	XPU_LOG("  dev_av_addr=%p size=%zu", dev_av_addr, av_addr_size);
	return 0;
}

static int export_mr_desc(void)
{
	int ret;
	void *desc_buf;
	size_t len;

	desc_buf = calloc(1, mr_desc_size);
	if (!desc_buf)
		return -FI_ENOMEM;

	len = mr_desc_size;
	ret = fi_mr_get_xpu_desc(mr, desc_buf, &len, FI_XPU, xpu_ctx);
	if (ret) {
		FT_PRINTERR("fi_mr_get_xpu_desc", -ret);
		free(desc_buf);
		return ret;
	}

	if (ft_cuda_alloc(opts.device, &dev_mr_desc, mr_desc_size) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for mr_desc failed\n");
		free(desc_buf);
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_mr_desc, desc_buf, mr_desc_size) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for mr_desc failed\n");
		free(desc_buf);
		return -FI_EIO;
	}

	free(desc_buf);
	XPU_LOG("  dev_mr_desc=%p size=%zu mr_key=%#lx", dev_mr_desc,
		mr_desc_size, fi_mr_key(mr));
	return 0;
}

static const char *scope_str(int scope)
{
	switch (scope) {
	case FI_XPU_WORK_ITEM:
		return "work_item";
	case FI_XPU_SUBGROUP:
		return "subgroup";
	case FI_XPU_WORK_GROUP:
		return "work_group";
	case FI_XPU_DEVICE:
		return "device";
	default:
		return "unknown";
	}
}

/*
 * FI_XPU_DEVICE asks for a barrier across the whole grid, which exists only in a
 * cooperatively launched kernel and would bound the grid to what the device
 * holds resident, so EFA refuses it. Check that the refusal reaches the caller,
 * for that scope and for one the build has never heard of. Nothing is
 * transferred, so this runs without the peer's participation.
 */
static int run_scope_reject(void)
{
	cudaStream_t stream;
	int ret;

	ret = ft_cuda_stream_create((void **) &stream);
	if (ret) {
		FT_PRINTERR("ft_cuda_stream_create", -ret);
		return ret;
	}

	/* The refused calls never touch the buffer, so any length will do. */
	ret = ft_efa_xpu_run_scope_reject(dev_xpu_ep, dev_xpu_send_cq, tx_buf, 64,
				      dev_mr_desc, dev_av_addr, FI_XPU_DEVICE,
				      stream);
	if (!ret)
		ret = ft_efa_xpu_run_scope_reject(dev_xpu_ep, dev_xpu_send_cq,
					      tx_buf, 64, dev_mr_desc,
					      dev_av_addr, GDA_SCOPE_UNKNOWN,
					      stream);

	if (ret)
		FT_ERR("an unsupported scope was not refused");
	else
		printf("FI_XPU_DEVICE and unknown scopes refused with "
		       "-FI_EOPNOTSUPP\n");

	ft_cuda_stream_destroy((void *) stream);

	return ret;
}

/*
 * Data verification (-v).
 *
 * The data only ever moves on the device, so the host checks it at the edges:
 * the side that supplies the bytes fills its buffer before the run and the side
 * that ends up with them checks afterwards. ft_fill_buf() and ft_check_buf()
 * stage through a host buffer for an FI_HMEM_CUDA iface, so they work on the
 * device-only buffers this test registers.
 *
 * Which buffer that is depends on the op. A send, a write and a write with
 * immediate all push the initiator's tx_buf into the target's rx_buf. A read
 * pulls the other way, and the address the target advertises is its rx_buf, so
 * for a read it is the target that fills and the initiator that checks. Either
 * way the receiving side never fills its own buffer, so a check that passes
 * means the bytes arrived rather than that they were already there.
 *
 * Every iteration carries the same pattern into the same buffer, so one check
 * after the run covers the run.
 */
static void *xpu_verify_fill_buf(int is_client)
{
	if (!(opts.options & FT_OPT_VERIFY_DATA))
		return NULL;

	if (gda_op == 3)
		return is_client ? NULL : rx_buf;

	return is_client ? tx_buf : NULL;
}

static void *xpu_verify_check_buf(int is_client)
{
	if (!(opts.options & FT_OPT_VERIFY_DATA))
		return NULL;

	if (gda_op == 3)
		return is_client ? rx_buf : NULL;

	return is_client ? NULL : rx_buf;
}

/*
 * Round the iteration count to something the block can divide.
 *
 * A block of gda_threads threads issues gda_threads operations per call, so it
 * walks the count in steps of that: round it down to a multiple of the block
 * size, since a group-scope call is a collective and every thread has to make
 * the same number of calls. Both sides round the same way, and show_perf()
 * reports what was run.
 *
 * This runs per transfer size rather than once in main(), because init_test()
 * picks an iteration count for each size.
 */
static void xpu_round_iterations(void)
{
	if (gda_threads <= 1)
		return;

	if (opts.iterations < gda_threads)
		opts.iterations = gda_threads;
	else
		opts.iterations -= opts.iterations % gda_threads;
}

/*
 * The largest transfer the op under test can carry.
 *
 * An XPU operation is one work request, so it is bounded by what the device
 * carries in one: the MTU for a send, the RDMA limit for a write or a read.
 * efa-direct reports ep_attr->max_msg_size as the larger of the two - which is
 * what ft_use_size() bounds the size sweep by - so ask the endpoint for the
 * limit that applies to this op instead.
 */
static size_t xpu_max_op_size(void)
{
	int optname = gda_op ? FI_OPT_MAX_RMA_SIZE : FI_OPT_MAX_MSG_SIZE;
	size_t max_size;
	size_t len = sizeof(max_size);

	if (fi_getopt(&ep->fid, FI_OPT_ENDPOINT, optname, &max_size, &len))
		return fi->ep_attr->max_msg_size;

	return max_size;
}

/*
 * Ping-pong latency over send/recv.
 *
 * Both sides run the same kernel; is_client decides who sends first, and from
 * there each side waits for the peer's message before sending its own. The
 * kernel posts rx_depth receives up front and reposts as it drains them, so the
 * receive queue never runs dry mid-run, but only one message is ever in flight
 * in each direction.
 */
static int run_lat(void)
{
	int ret = 0;
	cudaStream_t stream;
	int is_client;
	int rx_depth;

	ret = ft_sync();
	if (ret) {
		FT_PRINTERR("ft_sync", -ret);
		return ret;
	}

	ret = ft_cuda_stream_create((void **) &stream);
	if (ret) {
		FT_PRINTERR("ft_cuda_stream_create", -ret);
		return ret;
	}

	is_client = opts.dst_addr ? 1 : 0;

	xpu_round_iterations();

	/*
	 * Both sides send, so both fill. The ping-pong carries the same pattern
	 * each way, which is what lets one check at the end stand for the run.
	 */
	if (opts.options & FT_OPT_VERIFY_DATA) {
		ret = ft_fill_buf(tx_buf, opts.transfer_size);
		if (ret) {
			FT_PRINTERR("ft_fill_buf", -ret);
			goto out;
		}
	}

	/*
	 * Never pre-post more receives than the run will consume: a receive the
	 * peer never sends to is one the kernel would wait out its final counter
	 * threshold for.
	 */
	rx_depth = opts.window_size;
	if (rx_depth > opts.iterations)
		rx_depth = opts.iterations;

	/*
	 * Warm up before timing. The first exchange of a run pays for things a
	 * latency number should not carry - the first touch of each queue and
	 * the device's first walk of these descriptors - and at these iteration
	 * counts that lands almost entirely on whichever transfer size runs
	 * first, which is why it showed up as the smallest size being the
	 * slowest. Both sides warm up with the same count, so the exchange stays
	 * matched.
	 */
	if (opts.warmup_iterations > 0) {
		int saved_iters = opts.iterations;
		int warmup_depth;

		opts.iterations = opts.warmup_iterations;
		xpu_round_iterations();
		warmup_depth = opts.window_size;
		if (warmup_depth > opts.iterations)
			warmup_depth = opts.iterations;

		ret = ft_efa_xpu_run_lat_send(
			dev_xpu_ep, dev_xpu_send_cq, dev_xpu_recv_cq,
			use_cntr ? dev_xpu_send_cntr : NULL,
			use_cntr ? dev_xpu_recv_cntr : NULL,
			dev_av_addr, av_addr_size,
			rx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
			tx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
			opts.iterations, warmup_depth, is_client, gda_scope,
			gda_threads, stream);

		opts.iterations = saved_iters;
		if (ret) {
			FT_PRINTERR("ft_efa_xpu_run_lat_send (warmup)", -ret);
			goto out;
		}

		/* Start the timed run from a queue both sides have drained. */
		ret = ft_sync();
		if (ret) {
			FT_PRINTERR("ft_sync", -ret);
			goto out;
		}
	}

	ft_start();

	ret = ft_efa_xpu_run_lat_send(
		dev_xpu_ep, dev_xpu_send_cq, dev_xpu_recv_cq,
		use_cntr ? dev_xpu_send_cntr : NULL,
		use_cntr ? dev_xpu_recv_cntr : NULL,
		dev_av_addr, av_addr_size,
		rx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
		tx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
		opts.iterations, rx_depth, is_client, gda_scope, gda_threads,
		stream);

	ft_stop();

	if (ret) {
		FT_PRINTERR("ft_efa_xpu_run_lat_send", -ret);
		goto out;
	}

	if (opts.options & FT_OPT_VERIFY_DATA) {
		ret = ft_check_buf(rx_buf, opts.transfer_size);
		if (ret) {
			FT_PRINTERR("ft_check_buf", -ret);
			goto out;
		}
	}

	/*
	 * A factor of 2: an iteration is a message each way, so the elapsed time
	 * covers two transfers and show_perf() halves it to report the one-way
	 * latency.
	 */
	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 2);

out:
	ft_cuda_stream_destroy((void *) stream);

	return ret;
}

static int run_bw(void)
{
	int ret = 0;
	cudaStream_t stream;
	int is_client;
	int tx_depth;
	int rx_depth;
	void *fill_buf, *check_buf;

	is_client = opts.dst_addr ? 1 : 0;

	/*
	 * Fill before the barrier below, so the bytes are in place before the
	 * peer touches them. That matters for a read in particular, where the
	 * initiator pulls from a buffer the target never writes again.
	 */
	fill_buf = xpu_verify_fill_buf(is_client);
	check_buf = xpu_verify_check_buf(is_client);
	if (fill_buf) {
		ret = ft_fill_buf(fill_buf, opts.transfer_size);
		if (ret) {
			FT_PRINTERR("ft_fill_buf", -ret);
			return ret;
		}
	}

	ret = ft_sync();
	if (ret) {
		FT_PRINTERR("ft_sync", -ret);
		return ret;
	}

	ret = ft_cuda_stream_create((void **) &stream);
	if (ret) {
		FT_PRINTERR("ft_cuda_stream_create", -ret);
		return ret;
	}

	xpu_round_iterations();

	/* Number of in-flight WQEs is the window size (-W). */
	tx_depth = opts.window_size;
	rx_depth = opts.window_size;
	if (rx_depth > opts.iterations)
		rx_depth = opts.iterations;

	ft_start();

	if (is_client) {
		/*
		 * Client. For RMA ops (write/writedata/read) exchange the
		 * peer's target iov first; send (op 0) needs no remote iov.
		 */
		struct fi_rma_iov remote_iov = {0};

		if (gda_op != 0) {
			struct fi_rma_iov my_iov = {
				.addr = (gda_op == 3) ? (uintptr_t) tx_buf :
							(uintptr_t) rx_buf,
				.key = fi_mr_key(mr),
			};

			ret = ft_sock_send(oob_sock, &my_iov, sizeof(my_iov));
			if (ret)
				goto out;
			ret = ft_sock_recv(oob_sock, &remote_iov,
					   sizeof(remote_iov));
			if (ret)
				goto out;
		}

		ret = ft_efa_xpu_run_bw(
			dev_xpu_ep, dev_xpu_send_cq,
			use_cntr ? dev_xpu_send_cntr : NULL,
			gda_op,
			gda_op == 3 ? rx_buf : tx_buf,
			opts.transfer_size,
			dev_mr_desc, mr_desc_size,
			dev_av_addr, av_addr_size,
			remote_iov.addr, remote_iov.key,
			opts.iterations, tx_depth, gda_scope, gda_threads,
			stream);

		/*
		 * write (1) and read (3) are silent at the target, so this sync
		 * is all the server has to go on: it says this size's operations
		 * have landed, which is what keeps the server from moving on to
		 * the next size - or tearing its endpoint down after the last
		 * one - while they are still in flight. The server waits for it
		 * whether or not the run succeeded, so send it either way.
		 */
		if (gda_op == 1 || gda_op == 3) {
			int sync_ret = ft_sync();

			if (!ret)
				ret = sync_ret;
		}
	} else {
		/*
		 * Server. send (0) and writedata (2) generate receive
		 * completions, so post receives and drain. write (1) and
		 * read (3) are silent at the target; just sync.
		 */
		if (gda_op == 0 || gda_op == 2) {
			if (gda_op != 0) {
				/* writedata: exchange iov so the client has our
				 * target address/key. */
				struct fi_rma_iov remote_iov = {0};
				struct fi_rma_iov my_iov = {
					.addr = (uintptr_t) rx_buf,
					.key = fi_mr_key(mr),
				};

				ret = ft_sock_recv(oob_sock, &remote_iov,
						   sizeof(remote_iov));
				if (ret)
					goto out;
				ret = ft_sock_send(oob_sock, &my_iov,
						   sizeof(my_iov));
				if (ret)
					goto out;
			}

			ret = ft_efa_xpu_run_bw_recv(
				dev_xpu_ep, dev_xpu_recv_cq,
				use_cntr ? dev_xpu_recv_cntr : NULL,
				rx_buf, opts.transfer_size,
				dev_mr_desc, mr_desc_size,
				opts.iterations, rx_depth, gda_scope,
				gda_threads, stream);
		} else {
			/* Plain write / read: server just waits. */
			struct fi_rma_iov remote_iov = {0};
			struct fi_rma_iov my_iov = {
				.addr = (uintptr_t) rx_buf,
				.key = fi_mr_key(mr),
			};

			ret = ft_sock_recv(oob_sock, &remote_iov,
					   sizeof(remote_iov));
			if (ret)
				goto out;
			ret = ft_sock_send(oob_sock, &my_iov, sizeof(my_iov));
			if (ret)
				goto out;
			ft_sync();
		}
	}

	ft_stop();

	if (ret) {
		FT_PRINTERR("ft_efa_xpu_run_bw", -ret);
		goto out;
	}

	/*
	 * The run is over on both sides by now - a receive kernel that returned
	 * has drained its completions, and the silent ops have been synced for -
	 * so whatever arrived has arrived.
	 */
	if (check_buf) {
		ret = ft_check_buf(check_buf, opts.transfer_size);
		if (ret) {
			FT_PRINTERR("ft_check_buf", -ret);
			goto out;
		}
	}

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 1);

out:
	ft_cuda_stream_destroy((void *) stream);

	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, i, cleanup_ret;
	size_t max_op_size;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_SYNC;
	opts.iface = FI_HMEM_CUDA;
	/* Register the data MR ourselves (device-only), like fi_acc. */
	opts.options |= FT_OPT_SKIP_REG_MR;

	timeout = 5;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv,
			    "vhW:" ADDR_OPTS INFO_OPTS CS_OPTS API_OPTS,
			    (struct option[]){
				{"use-cntr", no_argument, NULL,
				 LONG_OPT_USE_CNTR},
				{"scope", required_argument, NULL,
				 LONG_OPT_SCOPE},
				{"threads", required_argument, NULL,
				 LONG_OPT_THREADS},
				{"lat", no_argument, NULL, LONG_OPT_LAT},
				{0, 0, 0, 0}
			    }, NULL)) != -1) {
		switch (op) {
		case LONG_OPT_USE_CNTR:
			use_cntr = true;
			break;
		case LONG_OPT_SCOPE:
			if (!strcasecmp(optarg, "work_item"))
				gda_scope = FI_XPU_WORK_ITEM;
			else if (!strcasecmp(optarg, "subgroup"))
				gda_scope = FI_XPU_SUBGROUP;
			else if (!strcasecmp(optarg, "work_group"))
				gda_scope = FI_XPU_WORK_GROUP;
			else if (!strcasecmp(optarg, "device"))
				gda_scope = FI_XPU_DEVICE;
			else {
				FT_ERR("unknown scope '%s'", optarg);
				return EXIT_FAILURE;
			}
			break;
		case LONG_OPT_THREADS:
			gda_threads = atoi(optarg);
			break;
		case LONG_OPT_LAT:
			run_latency = true;
			break;
		case 'o':
			/*
			 * Determine the device-side opcode directly from the
			 * -o string (like fi_acc), since opts.rma_op defaults
			 * to FT_RMA_WRITE and "msg" would not reset it.
			 */
			if (!strcasecmp(optarg, "write"))
				gda_op = 1;
			else if (!strcasecmp(optarg, "writedata"))
				gda_op = 2;
			else if (!strcasecmp(optarg, "read"))
				gda_op = 3;
			else
				gda_op = 0; /* msg / send */
			/* Also let the standard parser set hints->caps etc. */
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ret = ft_parse_api_opts(op, optarg, hints, &opts);
			if (ret)
				return ret;
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA;
			break;
		case 'W':
			opts.window_size = atoi(optarg);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "XPU GPU Direct Async test");
			FT_PRINT_OPTS_USAGE("-o <op>",
				"op: msg, write, writedata, read\n");
			FT_PRINT_OPTS_USAGE("-W <window>",
				"in-flight WQE / tx-rx depth (default 64)");
			FT_PRINT_OPTS_USAGE("-v", "Enable data verification");
			FT_PRINT_OPTS_USAGE("--use-cntr",
				"Use XPU counters instead of CQ polling");
			FT_PRINT_OPTS_USAGE("--scope <scope>",
				"cooperative scope the kernel issues at: "
				"work_item (default), subgroup, work_group, "
				"device");
			FT_PRINT_OPTS_USAGE("--lat",
				"ping-pong latency instead of bandwidth; one "
				"operation in flight, -o msg only");
			FT_PRINT_OPTS_USAGE("--threads <n>",
				"threads per block; every thread issues its own "
				"operation, so iterations and window size are "
				"split between them (default 1)");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (gda_threads < 1 || gda_threads > 1024) {
		FT_ERR("--threads must be between 1 and 1024");
		return EXIT_FAILURE;
	}

	/*
	 * Every thread of the block issues its own operation, so the block walks
	 * the window in steps of gda_threads and each thread needs a share of it
	 * to keep in flight. The iteration count is rounded the same way, but in
	 * run_bw(), because init_test() picks one per transfer size.
	 */
	if (gda_threads > 1) {
		int window = opts.window_size - opts.window_size % gda_threads;

		if (!window) {
			FT_ERR("--threads %d needs a window (-W) of at least "
			       "that many entries", gda_threads);
			return EXIT_FAILURE;
		}
		if (window != opts.window_size)
			FT_WARN("window rounded down to %d for %d threads per "
				"block", window, gda_threads);
		opts.window_size = window;
	}

	/*
	 * The ping-pong kernel is send/recv both ways, so it has no meaning for
	 * the RMA ops: a write or a read completes without the target issuing
	 * anything, which is the opposite of a round trip.
	 */
	if (run_latency && gda_op != 0) {
		FT_ERR("--lat runs over -o msg; write, writedata and read have "
		       "no reply to wait for");
		return EXIT_FAILURE;
	}

	/* Track write/read for buffer/verify helpers. */
	switch (gda_op) {
	case 1: /* write */
	case 2: /* writedata */
		gda_op_is_write = 1;
		break;
	case 3: /* read */
		gda_op_is_read = 1;
		break;
	default: /* 0 = send */
		break;
	}

	/*
	 * The bandwidth path is the default; FT_OPT_BW is what makes init_test()
	 * pick an iteration count for a windowed run and show_perf() report
	 * bandwidth. The latency path wants neither.
	 */
	if (!run_latency)
		opts.options |= FT_OPT_BW;

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_RMA | FI_HMEM | FI_XPU;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL |
				      FI_MR_VIRT_ADDR | FI_MR_PROV_KEY |
				      FI_MR_HMEM | FI_MR_XPU_DESC;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;

	/*
	 * HW counters require API >= 2.5: the provider only negotiates
	 * max_cntr_value down to the hardware limit for 2.5+. With an older
	 * version max_cntr_value stays UINT64_MAX and hw counter creation is
	 * rejected with -FI_EOPNOTSUPP.
	 */
	ft_fiversion = FI_VERSION(2, 6);

	XPU_LOG("step 1: ft_init + ft_init_oob");
	ret = ft_init();
	if (ret) {
		FT_PRINTERR("ft_init", -ret);
		return ret;
	}
	ret = ft_init_oob();
	if (ret) {
		FT_PRINTERR("ft_init_oob", -ret);
		return ret;
	}

	XPU_LOG("step 2: ft_getinfo (API %u.%u)",
		FI_MAJOR(ft_fiversion), FI_MINOR(ft_fiversion));
	ret = ft_getinfo(hints, &fi);
	if (ret) {
		FT_PRINTERR("ft_getinfo", -ret);
		return ret;
	}
	XPU_LOG("  fabric=%s domain=%s", fi->fabric_attr->name,
		fi->domain_attr->name);
	XPU_LOG("  max_cntr_value=%#lx max_err_cntr_value=%#lx",
		fi->domain_attr->max_cntr_value,
		fi->domain_attr->max_err_cntr_value);

	if (use_cntr && fi->domain_attr->max_cntr_value == UINT64_MAX) {
		FT_ERR("device does not support hw counters "
		       "(max_cntr_value not negotiated)");
		ret = -FI_ENOSYS;
		goto out;
	}

	XPU_LOG("step 3: ft_open_fabric_res (fabric/eq/domain)");
	ret = ft_open_fabric_res();
	if (ret)
		goto out;

	/* Create XPU context */
	XPU_LOG("step 4: fi_xpu_ctx (iface=%d device=%" PRIu64 ")",
		FI_HMEM_CUDA, opts.device);
	ret = create_xpu_ctx();
	if (ret)
		goto out;
	XPU_LOG("  xpu_ctx=%p av_addr_size=%zu mr_desc_size=%zu",
		(void *) xpu_ctx, av_addr_size, mr_desc_size);

	/* Create CQs with XPU context (device-resident ring). */
	XPU_LOG("step 5: fi_cq_open tx/rx (FI_XPU, device-resident CQ ring)");
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.flags = FI_XPU;
	cq_attr.xpu_ctx = xpu_ctx;

	cq_attr.size = fi->tx_attr->size;
	ret = fi_cq_open(domain, &cq_attr, &txcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open (tx)", -ret);
		goto out;
	}
	XPU_LOG("  txcq=%p size=%zu", (void *) txcq, cq_attr.size);

	cq_attr.size = fi->rx_attr->size;
	ret = fi_cq_open(domain, &cq_attr, &rxcq, NULL);
	if (ret) {
		FT_PRINTERR("fi_cq_open (rx)", -ret);
		goto out;
	}
	XPU_LOG("  rxcq=%p size=%zu", (void *) rxcq, cq_attr.size);

	/* Create counters if requested. */
	if (use_cntr) {
		struct fi_cntr_attr cntr_attr = {0};
		cntr_attr.events = FI_CNTR_EVENTS_COMP;
		cntr_attr.wait_obj = FI_WAIT_NONE;
		cntr_attr.flags = FI_XPU;
		cntr_attr.xpu_ctx = xpu_ctx;

		XPU_LOG("step 6: fi_cntr_open tx/rx (FI_XPU, hw cntr in device HBM)");
		ret = fi_cntr_open(domain, &cntr_attr, &txcntr, NULL);
		if (ret) {
			FT_PRINTERR("fi_cntr_open (tx)", -ret);
			goto out;
		}
		XPU_LOG("  txcntr=%p", (void *) txcntr);

		ret = fi_cntr_open(domain, &cntr_attr, &rxcntr, NULL);
		if (ret) {
			FT_PRINTERR("fi_cntr_open (rx)", -ret);
			goto out;
		}
		XPU_LOG("  rxcntr=%p", (void *) rxcntr);
	}

	/* Open AV. */
	XPU_LOG("step 7: fi_av_open");
	if (fi->domain_attr->av_type != FI_AV_UNSPEC)
		av_attr.type = fi->domain_attr->av_type;
	av_attr.count = opts.av_size;
	ret = fi_av_open(domain, &av_attr, &av, NULL);
	if (ret) {
		FT_PRINTERR("fi_av_open", -ret);
		goto out;
	}

	/* Endpoint with xpu_ctx set on ep_attr for later fi_ep_export_xpu(). */
	XPU_LOG("step 8: fi_endpoint2 FI_XPU (ep_attr->xpu_ctx=%p)",
		(void *) xpu_ctx);
	fi->ep_attr->xpu_ctx = xpu_ctx;
	ret = fi_endpoint2(domain, fi, &ep, FI_XPU, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint2", -ret);
		goto out;
	}

	/* Bind av/cq/cntr + enable (no initial recv posted). */
	XPU_LOG("step 9: bind av/cq/cntr + fi_enable");
	ret = fi_ep_bind(ep, &av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind(av)", -ret);
		goto out;
	}
	ret = fi_ep_bind(ep, &txcq->fid, FI_TRANSMIT | FI_SELECTIVE_COMPLETION);
	if (ret) {
		FT_PRINTERR("fi_ep_bind(txcq)", -ret);
		goto out;
	}
	ret = fi_ep_bind(ep, &rxcq->fid, FI_RECV | FI_SELECTIVE_COMPLETION);
	if (ret) {
		FT_PRINTERR("fi_ep_bind(rxcq)", -ret);
		goto out;
	}
	if (use_cntr) {
		uint64_t cntr_flags;

		cntr_flags = FI_SEND | FI_WRITE | FI_READ;
		ret = fi_ep_bind(ep, &txcntr->fid, cntr_flags);
		if (ret) {
			FT_PRINTERR("fi_ep_bind(txcntr)", -ret);
			goto out;
		}
		cntr_flags = (gda_op == 2) ? FI_REMOTE_WRITE : FI_RECV;
		ret = fi_ep_bind(ep, &rxcntr->fid, cntr_flags);
		if (ret) {
			FT_PRINTERR("fi_ep_bind(rxcntr)", -ret);
			goto out;
		}
	}
	ret = fi_enable(ep);
	if (ret) {
		FT_PRINTERR("fi_enable", -ret);
		goto out;
	}

	/* Allocate buffers only (device memory); we register the MR below. */
	XPU_LOG("step 10: ft_alloc_msgs (buffers only, MR registered separately)");
	opts.options |= FT_OPT_SKIP_REG_MR;
	ret = ft_alloc_msgs();
	if (ret)
		goto out;

	/*
	 * Register the data MR ourselves (device-only), mirroring fi_acc. The
	 * XPU descriptor is obtained later via fi_mr_get_xpu_desc() (the new-API
	 * equivalent of fi_acc's mr export) — the only difference from fi_acc.
	 */
	XPU_LOG("step 10b: fi_mr_regattr (FI_HMEM_DEVICE_ONLY)");
	{
		struct iovec iov = {
			.iov_base = rx_buf,
			.iov_len = buf_size,
		};
		struct fi_mr_attr mr_attr = {
			.mr_iov = &iov,
			.iov_count = 1,
			.access = ft_info_to_mr_access(fi),
			.requested_key = FT_MR_KEY,
			.iface = opts.iface,
		};
		mr_attr.device.cuda = opts.device;

		ret = fi_mr_regattr(domain, &mr_attr, FI_HMEM_DEVICE_ONLY, &mr);
		if (ret) {
			FT_PRINTERR("fi_mr_regattr", -ret);
			goto out;
		}
		mr_desc = fi_mr_desc(mr);
		XPU_LOG("  mr=%p key=%#lx", (void *) mr, fi_mr_key(mr));
	}

	XPU_LOG("step 11: ft_init_av_dst_addr (address exchange)");
	ret = ft_init_av_dst_addr(av, ep, &remote_fi_addr);
	if (ret)
		goto out;
	XPU_LOG("  peer inserted, remote_fi_addr=%#lx", remote_fi_addr);

	/* Export XPU handles */
	XPU_LOG("step 12a: export EP to device handle");
	ret = export_ep();
	if (ret)
		goto out;

	XPU_LOG("step 12b: export tx/rx CQ to device handles");
	ret = export_cqs();
	if (ret)
		goto out;

	if (use_cntr) {
		XPU_LOG("step 12c: export tx/rx CNTR to device handles");
		ret = export_cntrs();
		if (ret)
			goto out;
	}

	XPU_LOG("step 12d: export AV address to device handle");
	ret = export_av_addr();
	if (ret)
		goto out;

	XPU_LOG("step 12e: export MR descriptor to device handle");
	ret = export_mr_desc();
	if (ret)
		goto out;

	printf("EFA XPU GPU Direct Async Test\n");
	printf("  Operation: %s\n",
	       gda_op == 0 ? "send" : gda_op == 1 ? "write" :
	       gda_op == 2 ? "writedata" : "read");
	printf("  XPU counter: %s\n", use_cntr ? "yes" : "no");
	printf("  Scope: %s, threads/block: %d\n", scope_str(gda_scope),
	       gda_threads);
	fflush(stdout);

	XPU_LOG("step 13: launch GPU kernel (iters=%d)", opts.iterations);

	/*
	 * A scope EFA cannot support has no data path to run, so the only thing
	 * to check is that asking for it fails.
	 */
	if (gda_scope == FI_XPU_DEVICE) {
		ret = run_scope_reject();
		goto out;
	}

	/* Run the test */
	max_op_size = xpu_max_op_size();
	if (opts.options & FT_OPT_SIZE && opts.transfer_size > max_op_size) {
		FT_ERR("transfer size %zu is larger than the %zu this op can "
		       "carry in one work request", opts.transfer_size,
		       max_op_size);
		ret = -FI_EMSGSIZE;
		goto out;
	}
	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			if (test_size[i].size > max_op_size)
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = run_latency ? run_lat() : run_bw();
			if (ret)
				break;
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = run_latency ? run_lat() : run_bw();
	}

out:
	/*
	 * Match the fi_accelerator GDA test teardown: rely on ft_free_res()
	 * (which closes ep/cq/cntr/mr/av/domain/fabric in the fabtests-managed
	 * order) rather than explicitly closing the exported objects here.
	 * The XPU objects reference device-resident memory that the hardware
	 * still holds until the managed teardown releases the EP/CQ, so closing
	 * them out of order (or deregistering the MR early) fails with EINVAL.
	 */
	if (xpu_ctx)
		fi_close(&xpu_ctx->fid);

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}
