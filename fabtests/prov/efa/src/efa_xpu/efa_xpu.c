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
#include <inttypes.h>
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
 * Cooperative scope the kernels post at, and the block size they run with.
 * A call at a group scope is one operation for the group, so the scope
 * decides how many operations a call by the grid makes: one per thread at
 * FI_XPU_WORK_ITEM, one per warp at FI_XPU_SUBGROUP, one per block at
 * FI_XPU_WORK_GROUP (xpu_issuers()). Completions and counters are read at the
 * same scope, as one read for the group.
 */
static int gda_scope = FI_XPU_WORK_ITEM;
static int gda_threads = 1;

/*
 * --blocks: how many blocks the kernels launch. The totals are split across the
 * issuers of every block, and blocks post into the same queues concurrently -
 * which nothing in a one-block run exercises, since a block's threads can only
 * race each other.
 */
static int gda_blocks = 1;

/*
 * The issuers of the grid: what makes one operation per call, so how many
 * operations one call by the grid makes. The kernels count them the same way
 * (ft_efa_xpu_issuers()).
 */
static int xpu_issuers(void)
{
	switch (gda_scope) {
	case FI_XPU_SUBGROUP:
		return (gda_threads + 31) / 32 * gda_blocks;
	case FI_XPU_WORK_GROUP:
		return gda_blocks;
	default:
		return gda_threads * gda_blocks;
	}
}

/*
 * --lat: run the ping-pong kernel instead of the pipelined one. The two measure
 * different things, so they are separate runs rather than two phases of one: the
 * bandwidth path keeps a window of operations in flight, while this one has a
 * single operation outstanding and waits for the peer's reply before posting
 * again, which is what makes the round trip the thing being timed.
 */
static bool run_latency;

/*
 * --host-cq: open ordinary CQs, read on the host, instead of exporting XPU
 * CQs to the kernel. The kernel then only posts, many times the queue's depth,
 * and the host drains the completions with fi_cq_read() as they arrive - the
 * shape of an application with a host progress thread. Nothing in the kernel
 * bounds how far it runs ahead, so the run is what shows that a post waits for
 * the queue to drain rather than overrunning it.
 */
static bool host_cq;

/* A scope no build can make collective, used to check the refusal path. */
#define GDA_SCOPE_UNKNOWN 0x7fff

enum {
	LONG_OPT_USE_CNTR = 1000,
	LONG_OPT_SCOPE,
	LONG_OPT_THREADS,
	LONG_OPT_LAT,
	LONG_OPT_BLOCKS,
	LONG_OPT_HOST_CQ,
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
	int ret, i;
	void *addr_buf;
	size_t len, total;

	/*
	 * One copy per issuer, since every issuer of a kernel passes its own
	 * target address and indexes this array with its issuer index.
	 *
	 * The copies name the same peer, because the test has one. So what the
	 * array establishes is that each issuer's own argument is the one read -
	 * issuers reaching different peers needs more peers than this test has.
	 */
	addr_buf = calloc(xpu_issuers(), av_addr_size);
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

	for (i = 1; i < xpu_issuers(); i++)
		memcpy((char *) addr_buf + (size_t) i * av_addr_size, addr_buf,
		       av_addr_size);

	total = av_addr_size * (size_t) xpu_issuers();
	if (ft_cuda_alloc(opts.device, &dev_av_addr, total) != FI_SUCCESS) {
		FT_ERR("cudaMalloc for av_addr failed\n");
		free(addr_buf);
		return -FI_ENOMEM;
	}
	if (ft_cuda_copy_to_hmem(opts.device, dev_av_addr, addr_buf, total) != FI_SUCCESS) {
		FT_ERR("cudaMemcpy for av_addr failed\n");
		free(addr_buf);
		return -FI_EIO;
	}

	free(addr_buf);
	XPU_LOG("  dev_av_addr=%p size=%zu x %d", dev_av_addr, av_addr_size,
		xpu_issuers());
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

static const char *xpu_op_str(void)
{
	switch (gda_op) {
	case 1:
		return "write";
	case 2:
		return "writedata";
	case 3:
		return "read";
	default:
		return "send";
	}
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
 * that ends up with them checks afterwards. Every issuer moves its own chunk,
 * so the buffers carry one chunk per issuer of the grid, each with a pattern
 * that names its issuer (xpu_pt_fill(), xpu_pt_check()).
 *
 * For a send, a write and a write with immediate the pattern is written on the
 * device instead, by every thread of the issuing group before its first call,
 * into a buffer the host has zeroed. The call is made by the group's leader
 * alone, so a chunk that arrives whole is also the check that a group-scope
 * operation sees what the group wrote before the call, as fi_xpu(3) promises.
 *
 * Which buffer that is depends on the op. A send, a write and a write with
 * immediate all push the initiator's tx_buf into the target's rx_buf. A read
 * pulls the other way, and the address the target advertises is its rx_buf, so
 * for a read it is the target that fills and the initiator that checks. Either
 * way the receiving side never fills its own buffer, so a check that passes
 * means the bytes arrived rather than that they were already there.
 *
 * What a check can insist on depends on how many rounds landed. After one
 * round - every issuer one operation - each chunk has to hold exactly its
 * issuer's bytes, in place for an RMA op and as a set for a send, every issuer
 * exactly once; that is the check that each work item's own arguments were the
 * ones posted, and that a group made one operation and not one per thread.
 * After a run of many rounds an RMA chunk is still in place, but a receive
 * lands in whichever posted buffer the fabric picks, so all a send's chunk can
 * still be held to is being some issuer's. A verified run
 * therefore starts with one round on its own, checked strictly
 * (xpu_verify_round()).
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
 * Round the iteration count to something the grid can divide.
 *
 * Every issuer of the grid makes one operation per call, so the grid walks the
 * count in steps of its issuer count: round it down to a multiple of that,
 * since a group-scope call is a collective and every issuer has to make the
 * same number of calls. Both sides round the same way, and show_perf()
 * reports what was run.
 *
 * This runs per transfer size rather than once in main(), because init_test()
 * picks an iteration count for each size.
 */
static void xpu_round_iterations(void)
{
	int round = xpu_issuers();

	if (round <= 1)
		return;

	if (opts.iterations < round)
		opts.iterations = round;
	else
		opts.iterations -= opts.iterations % round;
}

/*
 * Swap the buffer each side's RMA targets over the OOB socket. A target
 * advertises its rx_buf; for a read the initiator advertises its tx_buf, which
 * the target never uses, but the exchange is symmetric so the two sides cannot
 * disagree about who sends first.
 */
static int xpu_exchange_iov(int is_client, struct fi_rma_iov *remote_iov)
{
	struct fi_rma_iov my_iov = {
		.addr = (uintptr_t) ((is_client && gda_op == 3) ?
					     tx_buf : rx_buf),
		.key = fi_mr_key(mr),
	};
	int ret;

	if (is_client) {
		ret = ft_sock_send(oob_sock, &my_iov, sizeof(my_iov));
		if (!ret)
			ret = ft_sock_recv(oob_sock, remote_iov,
					   sizeof(*remote_iov));
	} else {
		ret = ft_sock_recv(oob_sock, remote_iov, sizeof(*remote_iov));
		if (!ret)
			ret = ft_sock_send(oob_sock, &my_iov, sizeof(my_iov));
	}

	return ret;
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
static int xpu_pt_fill(void *dev_buf, size_t chunk);
static int xpu_pt_check(void *dev_buf, size_t chunk, int unordered, int once);
static int xpu_pt_zero(void *dev_buf, size_t chunk);

/*
 * One pass of the ping-pong at @iters iterations, @depth receives deep. With
 * -v the kernel writes the send pattern itself (see Data verification above).
 */
static int xpu_lat_pass(int is_client, int iters, int depth,
			cudaStream_t stream)
{
	int fill = !!(opts.options & FT_OPT_VERIFY_DATA);

	return ft_efa_xpu_run_lat_send(
		dev_xpu_ep, dev_xpu_send_cq, dev_xpu_recv_cq,
		use_cntr ? dev_xpu_send_cntr : NULL,
		use_cntr ? dev_xpu_recv_cntr : NULL,
		dev_av_addr, av_addr_size,
		rx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
		tx_buf, opts.transfer_size, dev_mr_desc, mr_desc_size,
		iters, depth, is_client, gda_scope, gda_threads, gda_blocks,
		fill, stream);
}

static void xpu_report_round(void)
{
	printf("one operation per issuer, %s at %s: %d blocks x %d threads, "
	       "%d operations x %zu bytes: PASS\n", xpu_op_str(),
	       scope_str(gda_scope), gda_blocks, gda_threads, xpu_issuers(),
	       opts.transfer_size);
}

static int run_lat(void)
{
	bool verify = !!(opts.options & FT_OPT_VERIFY_DATA);
	int grid = xpu_issuers();
	cudaStream_t stream;
	int ret, sync_ret;
	int is_client;
	int rx_depth;

	is_client = opts.dst_addr ? 1 : 0;

	/*
	 * Both sides send and both receive, so both zero both buffers - the
	 * kernel writes the send pattern itself - before the barrier, so that
	 * nothing lands in a buffer its owner is still preparing.
	 */
	if (verify) {
		ret = xpu_pt_zero(tx_buf, opts.transfer_size);
		if (!ret)
			ret = xpu_pt_zero(rx_buf, opts.transfer_size);
		if (ret)
			return ret;
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

	/*
	 * Never pre-post more receives than the run will consume: a receive the
	 * peer never sends to is one the kernel would wait out its final counter
	 * threshold for.
	 */
	rx_depth = MIN(opts.window_size, opts.iterations);

	/*
	 * One round on its own first: every issuer one message each way, so
	 * every issuer's chunk has to arrive exactly once (see Data verification
	 * above). Zero again afterwards, and agree on it before the next pass
	 * writes into the buffers.
	 */
	if (verify) {
		ret = xpu_lat_pass(is_client, grid, grid, stream);
		if (!ret)
			ret = xpu_pt_check(rx_buf, opts.transfer_size, 1, 1);
		if (!ret)
			ret = xpu_pt_zero(rx_buf, opts.transfer_size);
		sync_ret = ft_sync();
		if (!ret)
			ret = sync_ret;
		if (ret)
			goto out;
		xpu_report_round();
	}

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

		opts.iterations = opts.warmup_iterations;
		xpu_round_iterations();
		ret = xpu_lat_pass(is_client, opts.iterations,
				   MIN(opts.window_size, opts.iterations), stream);
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
	ret = xpu_lat_pass(is_client, opts.iterations, rx_depth, stream);
	ft_stop();

	if (ret) {
		FT_PRINTERR("ft_efa_xpu_run_lat_send", -ret);
		goto out;
	}

	if (verify) {
		ret = xpu_pt_check(rx_buf, opts.transfer_size, 1, 0);
		if (ret)
			goto out;
	}

	/*
	 * Report per call, not per message. A call by the grid puts one message
	 * per issuer in flight at once, so dividing by messages would credit
	 * every one of them with a round trip it shared, and report a latency
	 * that falls as issuers are added. A call is a round of the grid's
	 * messages, which is what each issuer waits out before its next; it
	 * carries all of their bytes. With one issuer the two are the same.
	 *
	 * A factor of 2: an iteration is a message each way, so the elapsed time
	 * covers two transfers and show_perf() halves it to report the one-way
	 * latency.
	 */
	show_perf(NULL, opts.transfer_size * (size_t) grid,
		  opts.iterations / grid, &start, &end, 2);

out:
	ft_cuda_stream_destroy((void *) stream);

	return ret;
}

/*
 * One pass of the bandwidth exchange, at the current iteration count: the
 * initiator's kernel, the target's half of it, and - for a write or a read,
 * which are silent at the target - the sync that tells the target the pass is
 * over. The same pass serves the warmup and the timed run.
 */
static int xpu_bw_pass(int is_client, const struct fi_rma_iov *remote_iov,
		       cudaStream_t stream)
{
	/* Number of in-flight WQEs is the window size (-W). */
	int tx_depth = opts.window_size;
	int rx_depth = MIN(opts.window_size, opts.iterations);
	/* A read pulls the target's bytes, which the host fills instead. */
	int fill = (opts.options & FT_OPT_VERIFY_DATA) && gda_op != 3;
	int ret = 0, sync_ret;

	if (is_client) {
		ret = ft_efa_xpu_run_bw(
			dev_xpu_ep, dev_xpu_send_cq,
			use_cntr ? dev_xpu_send_cntr : NULL,
			gda_op,
			gda_op == 3 ? rx_buf : tx_buf,
			opts.transfer_size,
			dev_mr_desc, mr_desc_size,
			dev_av_addr, av_addr_size,
			remote_iov->addr, remote_iov->key,
			opts.iterations, tx_depth, gda_scope, gda_threads,
			gda_blocks, fill, stream);
	} else if (gda_op == 0 || gda_op == 2) {
		/*
		 * A send and a write with immediate data generate receive
		 * completions at the target, so it posts receives and drains.
		 */
		ret = ft_efa_xpu_run_bw_recv(
			dev_xpu_ep, dev_xpu_recv_cq,
			use_cntr ? dev_xpu_recv_cntr : NULL,
			rx_buf, opts.transfer_size,
			dev_mr_desc, mr_desc_size,
			opts.iterations, rx_depth, gda_scope,
			gda_threads, gda_blocks, gda_op == 2, stream);
	}

	/*
	 * write (1) and read (3) are silent at the target, so this sync is all
	 * the target has to go on: it says the pass's operations have landed,
	 * which is what keeps the target from moving on to the next size - or
	 * tearing its endpoint down after the last one - while they are still
	 * in flight. Both sides reach it whether or not the run succeeded.
	 */
	if (gda_op == 1 || gda_op == 3) {
		sync_ret = ft_sync();
		if (!ret)
			ret = sync_ret;
	}

	return ret;
}

static int run_bw(void)
{
	bool verify = !!(opts.options & FT_OPT_VERIFY_DATA);
	cudaStream_t stream;
	int ret, sync_ret;
	int is_client;
	void *fill_buf, *check_buf;
	struct fi_rma_iov remote_iov = {0};

	is_client = opts.dst_addr ? 1 : 0;

	/*
	 * Fill before the barrier below, so the bytes are in place before the
	 * peer touches them. That matters for a read in particular, where the
	 * initiator pulls from a buffer the target never writes again.
	 */
	fill_buf = xpu_verify_fill_buf(is_client);
	check_buf = xpu_verify_check_buf(is_client);
	if (fill_buf) {
		/* Only a read's pattern is the host's to write; see above. */
		ret = gda_op == 3 ? xpu_pt_fill(fill_buf, opts.transfer_size) :
				    xpu_pt_zero(fill_buf, opts.transfer_size);
		if (ret)
			return ret;
	}
	if (check_buf) {
		ret = xpu_pt_zero(check_buf, opts.transfer_size);
		if (ret)
			return ret;
	}

	ret = ft_sync();
	if (ret) {
		FT_PRINTERR("ft_sync", -ret);
		return ret;
	}

	/*
	 * The RMA ops need the peer's target buffer; a send does not. Swap it
	 * before the clock starts, so the bandwidth is the fabric's and not the
	 * socket's.
	 */
	if (gda_op != 0) {
		ret = xpu_exchange_iov(is_client, &remote_iov);
		if (ret)
			return ret;
	}

	ret = ft_cuda_stream_create((void **) &stream);
	if (ret) {
		FT_PRINTERR("ft_cuda_stream_create", -ret);
		return ret;
	}

	xpu_round_iterations();

	/*
	 * One round on its own first: every issuer one operation, checked
	 * strictly (see Data verification above). Zero again afterwards, and
	 * agree on it before the next pass writes into the buffer.
	 */
	if (verify) {
		int saved_iters = opts.iterations;

		opts.iterations = xpu_issuers();
		ret = xpu_bw_pass(is_client, &remote_iov, stream);
		opts.iterations = saved_iters;
		if (!ret && check_buf)
			ret = xpu_pt_check(check_buf, opts.transfer_size,
					   gda_op == 0, 1);
		if (!ret && check_buf)
			ret = xpu_pt_zero(check_buf, opts.transfer_size);
		sync_ret = ft_sync();
		if (!ret)
			ret = sync_ret;
		if (ret)
			goto out;
		if (check_buf)
			xpu_report_round();
	}

	/*
	 * Warm up before timing, as the latency path does: the first pass of a
	 * run pays for the first touch of each queue and the device's first walk
	 * of these descriptors, which a bandwidth number should not carry. Both
	 * sides warm up with the same count, so a target posting receives posts
	 * as many as the initiator sends.
	 */
	if (opts.warmup_iterations > 0) {
		int saved_iters = opts.iterations;

		opts.iterations = opts.warmup_iterations;
		xpu_round_iterations();
		ret = xpu_bw_pass(is_client, &remote_iov, stream);
		opts.iterations = saved_iters;
		if (ret) {
			FT_PRINTERR("xpu_bw_pass (warmup)", -ret);
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
	ret = xpu_bw_pass(is_client, &remote_iov, stream);
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
		ret = xpu_pt_check(check_buf, opts.transfer_size, gda_op == 0, 0);
		if (ret)
			goto out;
	}

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 1);

out:
	ft_cuda_stream_destroy((void *) stream);

	return ret;
}

/*
 * Issuer tid's chunk carries a pattern of its own: the first byte names the
 * issuer that wrote it and the rest walks away from there. Naming the issuer in
 * the bytes is what lets a received chunk be checked, since a receive buffer
 * holds whichever message the fabric delivered into it rather than the one
 * posted by the issuer that owns it. For the RMA ops, where the initiator names
 * the offset itself, a chunk has to be exactly where its issuer put it. The
 * kernels write the same pattern (ft_efa_xpu_fill()).
 */
static uint8_t xpu_pt_byte(int tid, size_t off)
{
	return (uint8_t) ((size_t) tid + off * 31);
}

/*
 * How many bytes of one buffer the issuers have between them.
 *
 * ft_alloc_msgs() lays rx_buf and tx_buf out back to back in one allocation,
 * sized for one message per window entry. Every issuer's chunk has to fit at
 * once, so the issuers are bounded by the smaller of the two spans; the window
 * is at least one entry per issuer, which is what makes the usual sizes fit.
 */
static size_t xpu_pt_capacity(void)
{
	size_t rx_span = (size_t) (tx_buf - rx_buf);

	return MIN(rx_span, buf_size - rx_span);
}

static int xpu_pt_fill(void *dev_buf, size_t chunk)
{
	size_t total = chunk * (size_t) xpu_issuers();
	uint8_t *host;
	int tid, ret;

	host = malloc(total);
	if (!host)
		return -FI_ENOMEM;

	for (tid = 0; tid < xpu_issuers(); tid++) {
		size_t off;

		for (off = 0; off < chunk; off++)
			host[(size_t) tid * chunk + off] = xpu_pt_byte(tid, off);
	}

	ret = ft_cuda_copy_to_hmem(opts.device, dev_buf, host, total);
	if (ret)
		FT_PRINTERR("ft_cuda_copy_to_hmem", -ret);

	free(host);
	return ret;
}

/*
 * Check the chunks that arrived. @unordered says the transfer was a send, whose
 * messages land in whichever posted buffer the fabric picked: then a chunk is
 * located by the issuer its first byte names, and with @once every issuer has
 * to appear exactly once - which only holds after a single round. An RMA op
 * carries its own destination offset, so there the chunk in slot tid must be
 * issuer tid's.
 */
static int xpu_pt_check(void *dev_buf, size_t chunk, int unordered, int once)
{
	int grid = xpu_issuers();
	size_t total = chunk * (size_t) grid;
	uint8_t *host, *seen;
	int slot, ret;

	host = malloc(total);
	seen = calloc(grid, 1);
	if (!host || !seen) {
		free(host);
		free(seen);
		return -FI_ENOMEM;
	}

	ret = ft_cuda_copy_from_hmem(opts.device, host, dev_buf, total);
	if (ret) {
		FT_PRINTERR("ft_cuda_copy_from_hmem", -ret);
		goto out;
	}

	for (slot = 0; slot < grid; slot++) {
		const uint8_t *p = host + (size_t) slot * chunk;
		int tid = unordered ? p[0] : slot;
		size_t off;

		if (tid >= grid) {
			FT_ERR("slot %d holds no issuer's chunk: first byte %#x",
			       slot, p[0]);
			ret = -FI_EIO;
			goto out;
		}
		if (once && seen[tid]) {
			FT_ERR("issuer %d's chunk arrived twice, at slot %d",
			       tid, slot);
			ret = -FI_EIO;
			goto out;
		}
		seen[tid] = 1;

		for (off = 0; off < chunk; off++) {
			if (p[off] == xpu_pt_byte(tid, off))
				continue;
			FT_ERR("slot %d (issuer %d) byte %zu: got %#x, expected %#x",
			       slot, tid, off, p[off], xpu_pt_byte(tid, off));
			ret = -FI_EIO;
			goto out;
		}
	}

out:
	free(host);
	free(seen);
	return ret;
}

/* Zero a check buffer, so a check that passes means the bytes arrived. */
static int xpu_pt_zero(void *dev_buf, size_t chunk)
{
	int ret;

	ret = ft_cuda_memset(opts.device, dev_buf, 0,
			     chunk * (size_t) xpu_issuers());
	if (ret)
		FT_PRINTERR("ft_cuda_memset", -ret);
	return ret;
}

/*
 * Drain @expected completions from a host CQ, failing on an error completion or
 * on a stretch with no progress at all - which is what a kernel stuck behind a
 * queue it overran, or one that never posted, looks like from here - and then
 * check that nothing more arrives.
 */
/*
 * Account for one completion's context, which the kernel encoded as its issuer
 * and its sequence among that issuer's operations, each plus one (see
 * FT_EFA_XPU_CTX() in efa_xpu_kernels.cu). Every context has to arrive exactly
 * once: one that is no post's, or one seen twice, is a context the provider
 * did not hand back as it was given. An unsolicited write with immediate data
 * completes with no context at all, which @null_ok allows.
 */
static int xpu_ctx_account(void *ctx, uint8_t *seen, int per_issuer,
			   bool null_ok, const char *what)
{
	uint64_t v = (uint64_t) (uintptr_t) ctx;
	uint64_t issuer = v >> 32, seq = v & 0xffffffffu;
	size_t idx;

	if (!ctx && null_ok)
		return 0;
	if (!issuer || issuer > (uint64_t) xpu_issuers() || !seq ||
	    seq > (uint64_t) per_issuer) {
		FT_ERR("%s: completion context %#" PRIx64 " is no post's",
		       what, v);
		return -FI_EIO;
	}
	idx = (size_t) (issuer - 1) * per_issuer + (seq - 1);
	if (seen[idx]) {
		FT_ERR("%s: completion context %#" PRIx64 " arrived twice",
		       what, v);
		return -FI_EIO;
	}
	seen[idx] = 1;
	return 0;
}

static int xpu_host_drain(struct fid_cq *cq, int expected, bool null_ok,
			  const char *what)
{
	struct fi_cq_msg_entry comp[64];
	struct fi_cq_err_entry err = {0};
	uint64_t idle_since = ft_gettime_ms();
	int per_issuer = expected / xpu_issuers();
	uint8_t *seen;
	int got = 0;
	ssize_t ret;

	seen = calloc(expected, 1);
	if (!seen)
		return -FI_ENOMEM;

	while (got < expected) {
		ret = fi_cq_read(cq, comp, MIN(64, expected - got));
		if (ret > 0) {
			for (ssize_t k = 0; k < ret; k++) {
				int cret = xpu_ctx_account(comp[k].op_context,
							   seen, per_issuer,
							   null_ok, what);

				if (cret) {
					free(seen);
					return cret;
				}
			}
			got += (int) ret;
			idle_since = ft_gettime_ms();
			continue;
		}
		if (ret == -FI_EAVAIL) {
			fi_cq_readerr(cq, &err, 0);
			FT_ERR("%s: error completion after %d of %d: %s", what,
			       got, expected,
			       fi_cq_strerror(cq, err.prov_errno, err.err_data,
					      NULL, 0));
			free(seen);
			return -FI_EIO;
		}
		if (ret != -FI_EAGAIN) {
			FT_PRINTERR("fi_cq_read", -ret);
			free(seen);
			return (int) ret;
		}
		if (ft_gettime_ms() - idle_since > 10000) {
			FT_ERR("%s: no completion for 10 s after %d of %d",
			       what, got, expected);
			free(seen);
			return -FI_ETIMEDOUT;
		}
	}
	free(seen);

	/* A post that made more operations than it was asked to shows here. */
	for (idle_since = ft_gettime_ms();
	     ft_gettime_ms() - idle_since < 100;) {
		ret = fi_cq_read(cq, comp, 1);
		if (ret != -FI_EAGAIN) {
			FT_ERR("%s: completion left after %d: %zd", what,
			       expected, ret);
			return -FI_EIO;
		}
	}

	return 0;
}

/*
 * The --host-cq bandwidth run: the initiator's kernel posts every iteration
 * without reading a completion and the host drains its CQ meanwhile; for a
 * send or a write with immediate the target does the same with receives. The
 * iteration count is many times the queues' depth, so the kernels keep up only
 * if a post waits for the host's CQ reads to free a slot.
 */
static int run_host_cq(void)
{
	struct fi_rma_iov remote_iov = {0};
	cudaStream_t stream;
	void *run = NULL;
	int is_client = opts.dst_addr ? 1 : 0;
	int ret, fin;

	ret = ft_sync();
	if (ret)
		return ret;

	if (gda_op != 0) {
		ret = xpu_exchange_iov(is_client, &remote_iov);
		if (ret)
			return ret;
	}

	ret = ft_cuda_stream_create((void **) &stream);
	if (ret)
		return ret;

	xpu_round_iterations();

	if (!is_client && gda_op != 0 && gda_op != 2) {
		/* A write or a read is silent at the target. */
		ret = ft_sync();
		goto out;
	}

	if (is_client)
		ft_start();
	ret = ft_efa_xpu_post_launch(
		dev_xpu_ep, is_client ? gda_op : FT_EFA_XPU_OP_RECV,
		is_client ? (gda_op == 3 ? rx_buf : tx_buf) : rx_buf,
		opts.transfer_size, dev_mr_desc, dev_av_addr, av_addr_size,
		remote_iov.addr, remote_iov.key, opts.iterations, gda_scope,
		gda_threads, gda_blocks, stream, &run);
	if (ret)
		goto out;

	/*
	 * Read nothing for a while, so the kernel runs as far ahead as the
	 * queue lets it. A slot is free once the NIC has taken its descriptor,
	 * which it does at once, so what fills is the completions nobody has
	 * read: a post that did not wait for them would overflow the CQ here.
	 */
	usleep(500 * 1000);

	ret = xpu_host_drain(is_client ? txcq : rxcq, opts.iterations,
			     !is_client && gda_op == 2,
			     is_client ? "send queue" : "receive queue");
	fin = ft_efa_xpu_post_finish(run, stream);
	if (!ret)
		ret = fin;
	if (is_client)
		ft_stop();

	/* As in xpu_bw_pass(): a write or a read tells the target it is over. */
	if (gda_op == 1 || gda_op == 3) {
		fin = ft_sync();
		if (!ret)
			ret = fin;
	}

	if (!ret && is_client)
		show_perf(NULL, opts.transfer_size, opts.iterations, &start,
			  &end, 1);
out:
	ft_cuda_stream_destroy((void *) stream);
	return ret;
}

static int xpu_run_one(void)
{
	if (host_cq)
		return run_host_cq();
	return run_latency ? run_lat() : run_bw();
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
				{"blocks", required_argument, NULL,
				 LONG_OPT_BLOCKS},
				{"host-cq", no_argument, NULL,
				 LONG_OPT_HOST_CQ},
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
		case LONG_OPT_BLOCKS:
			gda_blocks = atoi(optarg);
			break;
		case LONG_OPT_HOST_CQ:
			host_cq = true;
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
			FT_PRINT_OPTS_USAGE("-v", "Enable data verification, "
				"including that every issuer's own arguments were "
				"the ones posted, once each");
			FT_PRINT_OPTS_USAGE("--use-cntr",
				"Use XPU counters instead of CQ polling");
			FT_PRINT_OPTS_USAGE("--host-cq",
				"read completions on the host while the kernel "
				"only posts, many times the queue depth");
			FT_PRINT_OPTS_USAGE("--scope <scope>",
				"cooperative scope the kernel posts at; a call "
				"makes one operation per thread at work_item "
				"(default), per warp at subgroup, per block at "
				"work_group; device is refused");
			FT_PRINT_OPTS_USAGE("--lat",
				"ping-pong latency instead of bandwidth; one "
				"operation in flight, -o msg only");
			FT_PRINT_OPTS_USAGE("--threads <n>",
				"threads per block; iterations and window size "
				"are split between the issuers of the scope "
				"(default 1)");
			FT_PRINT_OPTS_USAGE("--blocks <n>",
				"blocks per kernel launch; the split is across "
				"the issuers of every block, and the blocks "
				"post into the same queues (default 1)");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (gda_threads < 1 || gda_threads > 1024) {
		FT_ERR("--threads must be between 1 and 1024");
		return EXIT_FAILURE;
	}
	if (gda_blocks < 1 || gda_blocks > 1024) {
		FT_ERR("--blocks must be between 1 and 1024");
		return EXIT_FAILURE;
	}

	/*
	 * Every issuer of the grid makes its own operation, so the grid walks
	 * the window in steps of its issuer count and each issuer needs a share
	 * of it to keep in flight. The iteration count is rounded the same way,
	 * but per run, because init_test() picks one per transfer size.
	 */
	if (xpu_issuers() > 1) {
		int round = xpu_issuers();
		int window = opts.window_size - opts.window_size % round;

		if (!window) {
			FT_ERR("%d blocks of %d threads at %s need a window "
			       "(-W) of at least %d entries", gda_blocks,
			       gda_threads, scope_str(gda_scope), round);
			return EXIT_FAILURE;
		}
		if (window != opts.window_size)
			FT_WARN("window rounded down to %d for %d issuers in "
				"the grid", window, round);
		opts.window_size = window;
	}

	/*
	 * The first byte of a verified chunk names the issuer that wrote it, so
	 * that a received chunk can be located at all.
	 */
	if ((opts.options & FT_OPT_VERIFY_DATA) && xpu_issuers() > 256) {
		FT_ERR("-v runs with at most 256 issuers in the grid: a chunk "
		       "names its issuer in one byte");
		return EXIT_FAILURE;
	}

	/*
	 * A host-cq run posts far more than any buffer holds a chunk of per
	 * operation, and it has no completions on the device to time a round
	 * trip or a counter by.
	 */
	if (host_cq && (run_latency || use_cntr ||
			(opts.options & FT_OPT_VERIFY_DATA))) {
		FT_ERR("--host-cq runs bandwidth only, without --lat, "
		       "--use-cntr or -v");
		return EXIT_FAILURE;
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
	 * FI_XPU is only offered from API 2.7 on, which is the first version
	 * that can state an efa-direct endpoint requiring no FI_CONTEXT2 and
	 * offering no inject. (2.5 is separately the floor for hw counters: the
	 * provider only negotiates max_cntr_value down to the hardware limit
	 * from there, and below it counter creation is refused.)
	 */
	ft_fiversion = FI_VERSION(2, 7);

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
	if (!host_cq) {
		cq_attr.flags = FI_XPU;
		cq_attr.xpu_ctx = xpu_ctx;
	}

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
	/*
	 * Plain FI_TRANSMIT / FI_RECV, not selective completion: the XPU data
	 * path completes every operation it posts, which is what the kernels
	 * count on, and an XPU endpoint refuses the flag for exactly that
	 * reason.
	 */
	ret = fi_ep_bind(ep, &txcq->fid, FI_TRANSMIT);
	if (ret) {
		FT_PRINTERR("fi_ep_bind(txcq)", -ret);
		goto out;
	}
	ret = fi_ep_bind(ep, &rxcq->fid, FI_RECV);
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

	if (!host_cq) {
		XPU_LOG("step 12b: export tx/rx CQ to device handles");
		ret = export_cqs();
		if (ret)
			goto out;
	}

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
	printf("  Operation: %s\n", xpu_op_str());
	printf("  Mode: %s\n", run_latency ? "latency" : "bandwidth");
	printf("  XPU counter: %s\n", use_cntr ? "yes" : "no");
	printf("  Completions read on: %s\n", host_cq ? "host" : "device");
	printf("  Scope: %s, threads/block: %d, blocks: %d\n",
	       scope_str(gda_scope), gda_threads, gda_blocks);
	if (run_latency && xpu_issuers() > 1)
		printf("  Latency is per call: %d messages in flight per call, "
		       "bytes are all of them\n", xpu_issuers());
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
			/*
			 * Every issuer has a chunk of the buffers to itself, so
			 * a size that does not leave room for all of them is
			 * skipped rather than run short.
			 */
			if (test_size[i].size * (size_t) xpu_issuers() >
			    xpu_pt_capacity())
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = xpu_run_one();
			if (ret)
				break;
		}
	} else {
		if (opts.transfer_size * (size_t) xpu_issuers() >
		    xpu_pt_capacity()) {
			FT_ERR("%d issuers of %zu bytes need %zu bytes of "
			       "buffer, more than the %zu allocated; raise the "
			       "window (-W) or lower the size",
			       xpu_issuers(), opts.transfer_size,
			       opts.transfer_size * (size_t) xpu_issuers(),
			       xpu_pt_capacity());
			ret = -FI_EMSGSIZE;
			goto out;
		}
		init_test(&opts, test_name, sizeof(test_name));
		ret = xpu_run_one();
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
