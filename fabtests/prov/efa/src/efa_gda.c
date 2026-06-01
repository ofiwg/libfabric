/*
 * Copyright (c) 2025, Amazon.com, Inc.  All rights reserved.
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
 */

#include "hmem.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <getopt.h>
#include <rdma/fi_ext.h>
#include <rdma/fi_ext_efa.h>
#include <shared.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef HAVE_EFAGDA
#include <efa_cuda_dp.h>
#include <efa_gda_kernels.h>

struct fi_efa_ops_gda *efa_gda_ops;
struct efa_cuda_cq *gda_send_cq, *gda_recv_cq;
struct fid_cq *txcq_ext, *rxcq_ext;
void *send_cq_buffer, *recv_cq_buffer;
struct fid_ep *gda_ep;
struct efa_cuda_qp *gda_qp;
static enum ibv_wr_opcode gda_op = IBV_WR_SEND;
static bool use_hw_cntr;
static volatile uint64_t *hw_send_cntr_ptr;
static volatile uint64_t *hw_recv_cntr_ptr;

enum {
	LONG_OPT_USE_HW_CNTR,
};

static int create_hw_cntr(struct fid_cntr **cntr,
			  volatile uint64_t **cntr_ptr)
{
	int ret, dmabuf_fd;
	uint64_t dmabuf_offset;
	void *cuda_buf;
	struct fi_cntr_attr attr = {0};
	struct fi_efa_comp_cntr_init_attr efa_attr = {0};

	ret = ft_hmem_alloc(opts.iface, opts.device, &cuda_buf,
			    sizeof(uint64_t));
	if (ret)
		return ret;

	ret = ft_hmem_get_dmabuf_fd(opts.iface, cuda_buf, sizeof(uint64_t),
				    &dmabuf_fd, &dmabuf_offset);
	if (ret) {
		ft_hmem_free(opts.iface, cuda_buf);
		return ret;
	}

	attr.events = FI_CNTR_EVENTS_COMP;
	attr.wait_obj = FI_WAIT_UNSPEC;

	efa_attr.flags = FI_EFA_COMP_CNTR_INIT_WITH_COMP_EXTERNAL_MEM;
	efa_attr.comp_cntr_ext_mem.type = FI_EFA_MEMORY_LOCATION_DMABUF;
	efa_attr.comp_cntr_ext_mem.dmabuf.fd = dmabuf_fd;
	efa_attr.comp_cntr_ext_mem.dmabuf.offset = dmabuf_offset;

	ret = efa_gda_ops->cntr_open_ext(domain, &attr, cntr, NULL, &efa_attr);
	if (ret) {
		FT_WARN("hw cntr open failed (%s)\n", fi_strerror(-ret));
		ft_hmem_free(opts.iface, cuda_buf);
		return ret;
	}

	*cntr_ptr = (volatile uint64_t *)cuda_buf;
	return 0;
}

int create_ext_cq(void **cq_buffer, struct fid_cq **cq_ext,
		  struct efa_cuda_cq **gda_cq)
{
	int ret;
	int dmabuf_fd;
	uint64_t dmabuf_offset;
	uint32_t cq_entries, entry_size, additional_space, buf_size;
	struct fi_efa_cq_attr cq_ext_attr = {0};

	cq_entries = cq_attr.size;
	entry_size = 32;
	additional_space = 4096;
	buf_size = cq_entries * entry_size + additional_space;

	ret = ft_hmem_alloc(opts.iface, opts.device, cq_buffer, buf_size);
	if (ret) {
		FT_PRINTERR("ft_hmem_alloc", -ret);
		return ret;
	}

	ret = ft_hmem_get_dmabuf_fd(opts.iface, *cq_buffer, buf_size,
				    &dmabuf_fd, &dmabuf_offset);
	if (ret) {
		FT_PRINTERR("ft_hmem_get_dmabuf_fd", -ret);
		goto free_buf;
	}

	struct fi_efa_cq_init_attr efa_cq_init_attr = {0};
	efa_cq_init_attr.flags = FI_EFA_CQ_INIT_FLAGS_EXT_MEM_DMABUF;
	efa_cq_init_attr.ext_mem_dmabuf.length = buf_size;
	efa_cq_init_attr.ext_mem_dmabuf.offset = dmabuf_offset;
	efa_cq_init_attr.ext_mem_dmabuf.fd = dmabuf_fd;

	ret = efa_gda_ops->cq_open_ext(domain, &cq_attr, &efa_cq_init_attr,
				       cq_ext, NULL);
	if (ret) {
		FT_PRINTERR("cq_open_ext", -ret);
		goto free_buf;
	}

	ret = efa_gda_ops->query_cq(*cq_ext, &cq_ext_attr);
	if (ret) {
		FT_PRINTERR("query_cq", -ret);
		goto close_cq;
	}

	FT_DEBUG("cq_ext %p, cq_ext_attr: buffer %p, entry_size %u, "
		 "num_entries %u, cq_entries %u\n",
		 *cq_ext, cq_ext_attr.buffer, cq_ext_attr.entry_size,
		 cq_ext_attr.num_entries, cq_entries);

	struct efa_cuda_cq_attrs cq_attrs = {0};
	cq_attrs.buffer = (uint8_t *)*cq_buffer;
	cq_attrs.num_entries = cq_entries;
	cq_attrs.entry_size = cq_ext_attr.entry_size;

	*gda_cq = efa_cuda_create_cq(&cq_attrs, sizeof(cq_attrs));
	if (!*gda_cq) {
		ret = FI_EINVAL;
		FT_PRINTERR("efa_cuda_create_cq", -ret);
	}

	return ret;

close_cq:
	fi_close(&(*cq_ext)->fid);
free_buf:
	ft_hmem_free(opts.iface, *cq_buffer);
	*cq_buffer = NULL;

	return ret;
}

int create_gda_qp()
{
	int ret, status;
	void *sq_ptr = 0;
	void *rq_ptr = 0;
	uint32_t *sq_db = 0;
	uint32_t *rq_db = 0;
	struct fi_efa_wq_attr sq_attr = {0};
	struct fi_efa_wq_attr rq_attr = {0};

	ret = efa_gda_ops->query_qp_wqs(gda_ep, &sq_attr, &rq_attr);
	if (ret) {
		FT_PRINTERR("query_qp_wqs", -ret);
		return ret;
	}

	FT_DEBUG("sq_attr: buffer %p, entry_size %u, num_entries %u, doorbell "
		 "%p\n",
		 sq_attr.buffer, sq_attr.entry_size, sq_attr.num_entries,
		 sq_attr.doorbell);
	FT_DEBUG("rq_attr: buffer %p, entry_size %u, num_entries %u, doorbell "
		 "%p\n",
		 rq_attr.buffer, rq_attr.entry_size, rq_attr.num_entries,
		 rq_attr.doorbell);

	// Require PeerMappingOverride=1 in NVIDIA kernel module to enable
	// mapping of external memory
	status = cuMemHostRegister(
		sq_attr.buffer, sq_attr.num_entries * sq_attr.entry_size,
		CU_MEMHOSTREGISTER_IOMEMORY | CU_MEMHOSTREGISTER_DEVICEMAP);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("SQ buffer cuMemHostRegister", status);
	}

	status = cuMemHostGetDevicePointer((CUdeviceptr *) &sq_ptr,
					   sq_attr.buffer, 0);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("SQ buffer cuMemHostGetDevicePointer", status);
	}

	status = cuMemHostRegister(sq_attr.doorbell, 4,
				   CU_MEMHOSTREGISTER_IOMEMORY |
					   CU_MEMHOSTREGISTER_DEVICEMAP);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("SQ doorbell cudaHostRegister", status);
	}

	status = cuMemHostGetDevicePointer((CUdeviceptr *) &sq_db,
					   sq_attr.doorbell, 0);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("SQ doorbell cuMemHostGetDevicePointer", status);
	}

	status = cuMemHostRegister(rq_attr.buffer,
				   rq_attr.num_entries * rq_attr.entry_size,
				   CU_MEMHOSTREGISTER_DEVICEMAP);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("RQ buffer cudaHostRegister", status);
	}

	status = cuMemHostGetDevicePointer((CUdeviceptr *) &rq_ptr,
					   rq_attr.buffer, 0);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("RQ buffer cuMemHostGetDevicePointer", status);
	}

	status = cuMemHostRegister(rq_attr.doorbell, 4,
				   CU_MEMHOSTREGISTER_IOMEMORY |
					   CU_MEMHOSTREGISTER_DEVICEMAP);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("RQ doorbell cudaHostRegister", status);
	}

	status = cuMemHostGetDevicePointer((CUdeviceptr *) &rq_db,
					   rq_attr.doorbell, 0);
	if (status != CUDA_SUCCESS) {
		FT_PRINTERR("RQ doorbell cuMemHostGetDevicePointer", status);
	}

	// initialize sq and rq on device
	struct efa_cuda_qp_attrs qp_attrs = {0};
	qp_attrs.sq_buffer = (uint8_t *)sq_ptr;
	qp_attrs.rq_buffer = (uint8_t *)rq_ptr;
	qp_attrs.sq_doorbell = sq_db;
	qp_attrs.rq_doorbell = rq_db;
	qp_attrs.sq_num_entries = sq_attr.num_entries;
	qp_attrs.sq_entry_size = sq_attr.entry_size;
	qp_attrs.sq_max_batch = sq_attr.max_batch;
	qp_attrs.rq_num_entries = rq_attr.num_entries;
	qp_attrs.rq_entry_size = rq_attr.entry_size;

	gda_qp = efa_cuda_create_qp(&qp_attrs, sizeof(qp_attrs));
	if (!gda_qp) {
		FT_PRINTERR("efa_cuda_create_qp", -ret);
	}
	return ret;
}

static int run_bw(struct fi_rma_iov *remote_iov)
{
	int ret;
	uint16_t ah;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
	uint64_t lkey;
	cudaStream_t stream;
	int is_client;
	int tx_depth;
	int rx_depth;

	ret = ft_sync();
	if (ret) {
		FT_PRINTERR("ft_sync", -ret);
		return ret;
	}

	lkey = efa_gda_ops->get_mr_lkey(mr);
	if (lkey == FI_KEY_NOTAVAIL) {
		FT_PRINTERR("get_mr_lkey", FI_KEY_NOTAVAIL);
		return -FI_ENODATA;
	}

	cudaStreamCreate(&stream);

	ret = efa_gda_ops->query_addr(gda_ep, remote_fi_addr, &ah,
				      &remote_qpn, &remote_qkey);
	if (ret) {
		FT_PRINTERR("query_addr", -ret);
		goto out;
	}

	is_client = opts.dst_addr ? 1 : 0;
	tx_depth = fi->tx_attr->size / 2;
	rx_depth = fi->rx_attr->size / 2;
	if (rx_depth > opts.iterations)
		rx_depth = opts.iterations;

	if (ft_check_opts(FT_OPT_VERIFY_DATA)) {
		if (gda_op == IBV_WR_RDMA_READ) {
			ret = ft_fill_buf((char *) tx_buf, opts.transfer_size);
		} else if (is_client) {
			ret = ft_fill_buf((char *) tx_buf, opts.transfer_size);
		}
		if (ret)
			goto out;
		ft_sync();
	}

	ft_start();
	if (is_client) {
		/* Client: post writes/writedata/read */
		ret = efagda_run_bw(gda_qp, gda_send_cq,
				    use_hw_cntr ? hw_send_cntr_ptr : NULL,
				    gda_op,
				    (uintptr_t) (gda_op == IBV_WR_RDMA_READ ?
						 rx_buf : tx_buf),
				    opts.transfer_size,
				    lkey, ah, remote_qpn, remote_qkey,
				    remote_iov->addr, remote_iov->key,
				    opts.iterations, tx_depth, stream);
	} else {
		/* Server: for writedata/send, poll recv CQ for completions */
		if (gda_op == IBV_WR_RDMA_WRITE_WITH_IMM ||
		    gda_op == IBV_WR_SEND) {
			ret = efagda_run_bw_recv(gda_qp, gda_recv_cq,
						 use_hw_cntr ? hw_recv_cntr_ptr : NULL,
						 (uintptr_t) rx_buf,
						 opts.transfer_size, lkey,
						 opts.iterations, rx_depth,
						 stream);
		}
		/* For plain write, server just waits for sync */
	}
	ft_stop();

	if (ret) {
		FT_PRINTERR("efagda_run_bw", -ret);
		goto out;
	}

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 1);

out:
	cudaStreamDestroy(stream);
	ft_sync();

	if (!ret && ft_check_opts(FT_OPT_VERIFY_DATA)) {
		if ((gda_op == IBV_WR_RDMA_READ && is_client) ||
		    (gda_op != IBV_WR_RDMA_READ && !is_client)) {
			ret = ft_check_buf((char *) rx_buf, opts.transfer_size);
		}
	}

	return ret;
}

static int run()
{
	int ret;
	uint16_t ah;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
	uint64_t lkey;
	cudaStream_t stream;
	int is_client;
	int rx_depth;

	ret = ft_sync();
	if (ret) {
		FT_PRINTERR("ft_sync", -ret);
		return ret;
	}

	lkey = efa_gda_ops->get_mr_lkey(mr);
	if (lkey == FI_KEY_NOTAVAIL) {
		FT_PRINTERR("get_mr_lkey", FI_KEY_NOTAVAIL);
		return -FI_ENODATA;
	}

	cudaStreamCreate(&stream);

	ret = efa_gda_ops->query_addr(gda_ep, remote_fi_addr, &ah,
				      &remote_qpn, &remote_qkey);
	if (ret) {
		FT_PRINTERR("query_addr", -ret);
		goto out;
	}

	is_client = opts.dst_addr ? 1 : 0;
	rx_depth = fi->rx_attr->size / 2;
	if (rx_depth > opts.iterations)
		rx_depth = opts.iterations;

	if (is_client && ft_check_opts(FT_OPT_VERIFY_DATA)) {
		ret = ft_fill_buf((char *) tx_buf + ft_tx_prefix_size(),
				  opts.transfer_size);
		if (ret)
			goto out;
	}

	ft_start();
	ret = efagda_run_lat_send(gda_qp, gda_send_cq, gda_recv_cq,
			  use_hw_cntr ? hw_send_cntr_ptr : NULL,
			  use_hw_cntr ? hw_recv_cntr_ptr : NULL,
			  ah, remote_qpn, remote_qkey,
			  (uintptr_t) rx_buf, opts.transfer_size, lkey,
			  (uintptr_t) tx_buf, opts.transfer_size, lkey,
			  opts.iterations, rx_depth, is_client, stream);
	ft_stop();

	if (ret) {
		FT_PRINTERR("efagda_run_lat_send", -ret);
		goto out;
	}

	if (!is_client && ft_check_opts(FT_OPT_VERIFY_DATA)) {
		ret = ft_check_buf((char *) rx_buf, opts.transfer_size);
		if (ret)
			goto out;
	}

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 2);

out:
	cudaStreamDestroy(stream);
	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, i, cleanup_ret;
	struct fi_rma_iov remote_iov = {0};

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_SYNC;

	timeout = 5;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt_long(argc, argv,
			    "vh" ADDR_OPTS INFO_OPTS CS_OPTS API_OPTS,
			    (struct option[]){
				{"use-hw-cntr", no_argument, NULL,
				 LONG_OPT_USE_HW_CNTR},
				{0, 0, 0, 0}
			    }, NULL)) != -1) {
		switch (op) {
		case LONG_OPT_USE_HW_CNTR:
			use_hw_cntr = true;
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
		case '?':
		case 'h':
			ft_usage(argv[0], "GPU Direct Async test");
			FT_PRINT_OPTS_USAGE("-o <op>",
				"op: msg, write, writedata, read\n");
			FT_PRINT_OPTS_USAGE("-v", "Enable data verification");
			FT_PRINT_OPTS_USAGE("--use-hw-cntr",
				"Use hardware counter instead of send CQ");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	switch (opts.rma_op) {
	case FT_RMA_WRITE:
		gda_op = IBV_WR_RDMA_WRITE;
		break;
	case FT_RMA_WRITEDATA:
		gda_op = IBV_WR_RDMA_WRITE_WITH_IMM;
		break;
	case FT_RMA_READ:
		gda_op = IBV_WR_RDMA_READ;
		break;
	default:
		gda_op = IBV_WR_SEND;
		break;
	}

	if (gda_op != IBV_WR_SEND)
		opts.options |= FT_OPT_BW;

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_RMA | FI_HMEM;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL |
				      FI_MR_VIRT_ADDR | FI_MR_PROV_KEY |
				      FI_MR_HMEM;
	hints->domain_attr->progress = FI_PROGRESS_MANUAL;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;

	if (use_hw_cntr)
		ft_fiversion = FI_VERSION(2, 5);

	ret = ft_init_fabric();
	if (ret) {
		FT_PRINTERR("ft_init_fabric", -ret);
		return ret;
	}

	// open efa gda domain ops
	ret = fi_open_ops(&domain->fid, FI_EFA_GDA_OPS, 0,
			  (void **) &efa_gda_ops, NULL);
	if (ret) {
		FT_PRINTERR("fi_open_ops", -ret);
		return ret;
	}

	// create extended CQs
	cq_attr.format = FI_CQ_FORMAT_MSG;
	cq_attr.wait_obj = FI_WAIT_NONE;
	cq_attr.size = fi->tx_attr->size;
	ret = create_ext_cq(&send_cq_buffer, &txcq_ext, &gda_send_cq);
	if (ret) {
		FT_PRINTERR("create_ext_cq send", -ret);
		return ret;
	}

	cq_attr.size = fi->rx_attr->size;
	ret = create_ext_cq(&recv_cq_buffer, &rxcq_ext, &gda_recv_cq);
	if (ret) {
		FT_PRINTERR("create_ext_cq recv", -ret);
		return ret;
	}

	ret = fi_endpoint(domain, fi, &gda_ep, NULL);
	if (ret) {
		return ret;
	}

	if (use_hw_cntr) {
		ret = create_hw_cntr(&txcntr, &hw_send_cntr_ptr);
		if (ret) {
			FT_PRINTERR("create_hw_cntr send", -ret);
			return ret;
		}
		ret = create_hw_cntr(&rxcntr, &hw_recv_cntr_ptr);
		if (ret) {
			FT_PRINTERR("create_hw_cntr recv", -ret);
			return ret;
		}
		ret = fi_ep_bind(gda_ep, &rxcntr->fid,
				 (gda_op == IBV_WR_RDMA_WRITE_WITH_IMM) ?
					 FI_REMOTE_WRITE :
					 FI_RECV);
		if (ret) {
			FT_PRINTERR("fi_ep_bind rxcntr", -ret);
			return ret;
		}
	}

	ret = ft_enable_ep(gda_ep, eq, av, txcq_ext, rxcq_ext, txcntr, NULL,
			   rma_cntr);
	if (ret)
		return ret;

	ret = ft_init_av_dst_addr(av, gda_ep, &remote_fi_addr);
	if (ret)
		return ret;

	ret = create_gda_qp();
	if (ret) {
		FT_PRINTERR("create_gda_qp", -ret);
		return ret;
	}

	/* Exchange RMA keys for write/writedata/read ops */
	if (gda_op != IBV_WR_SEND) {
		struct fi_rma_iov my_iov = {0};
		my_iov.addr = (gda_op == IBV_WR_RDMA_READ) ?
			      (uintptr_t) tx_buf : (uintptr_t) rx_buf;
		my_iov.key = fi_mr_key(mr);

		ret = ft_sock_send(oob_sock, &my_iov, sizeof(my_iov));
		if (ret) {
			FT_PRINTERR("ft_sock_send rma_iov", -ret);
			return ret;
		}
		ret = ft_sock_recv(oob_sock, &remote_iov, sizeof(remote_iov));
		if (ret) {
			FT_PRINTERR("ft_sock_recv rma_iov", -ret);
			return ret;
		}
	}

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			if (gda_op == IBV_WR_SEND)
				ret = run();
			else
				ret = run_bw(&remote_iov);
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		if (gda_op == IBV_WR_SEND)
			ret = run();
		else
			ret = run_bw(&remote_iov);
	}

	if (send_cq_buffer)
		efa_cuda_destroy_cq(gda_send_cq);
	if (recv_cq_buffer)
		efa_cuda_destroy_cq(gda_recv_cq);
	if (gda_qp)
		efa_cuda_destroy_qp(gda_qp);
	// qp need to be destroyed before closing cq
	FT_CLOSE_FID(gda_ep);
	if (txcq_ext)
		fi_close(&txcq_ext->fid);
	if (rxcq_ext)
		fi_close(&rxcq_ext->fid);

	cleanup_ret = ft_free_res();
	return ft_exit_code(ret ? ret : cleanup_ret);
}

#endif /* HAVE_EFAGDA */
