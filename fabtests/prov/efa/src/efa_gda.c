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
#ifdef HAVE_EFAGDA
#include <efagda.h>

struct fi_efa_ops_gda *efa_gda_ops;
struct efa_cq *gda_send_cq, *gda_recv_cq;
struct fid_cq *txcq_ext, *rxcq_ext;
void *send_cq_buffer, *recv_cq_buffer;
struct fid_ep *gda_ep;
struct efa_qp *gda_qp;

int create_ext_cq(void **cq_buffer, struct fid_cq **cq_ext,
		  struct efa_cq **gda_cq)
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
		return ret;
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
		return ret;
	}

	ret = efa_gda_ops->query_cq(*cq_ext, &cq_ext_attr);
	if (ret) {
		FT_PRINTERR("query_cq", -ret);
		return ret;
	}

	FT_DEBUG("cq_ext %p, cq_ext_attr: buffer %p, entry_size %u, "
		 "num_entries %u, cq_entries %u\n",
		 *cq_ext, cq_ext_attr.buffer, cq_ext_attr.entry_size,
		 cq_ext_attr.num_entries, cq_entries);

	*gda_cq = efagda_create_cuda_cq(*cq_buffer, 1, cq_entries,
					cq_ext_attr.entry_size);
	if (!*gda_cq) {
		ret = FI_EINVAL;
		FT_PRINTERR("efagda_create_cuda_cq", -ret);
	}

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
	gda_qp = efagda_create_cuda_qp(sq_ptr, sq_attr.num_entries, sq_db,
				       sq_attr.max_batch, rq_ptr,
				       rq_attr.num_entries, rq_db);
	if (!gda_qp) {
		FT_PRINTERR("efagda_create_cuda_qp", -ret);
	}
	return ret;
}

static int run()
{
	int ret;
	uint16_t ah;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
	struct ibv_wc wc;
	uint64_t lkey;

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

	cuda_stream_t stream = cuda_create_stream();
	if (opts.dst_addr) {
		ret = efa_gda_ops->query_addr(gda_ep, remote_fi_addr, &ah,
					      &remote_qpn, &remote_qkey);
		if (ret) {
			FT_PRINTERR("query_addr", -ret);
			return ret;
		}

		if (ft_check_opts(FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE)) {
			ret = ft_fill_buf((char *) tx_buf + ft_tx_prefix_size(),
					  opts.transfer_size);
			if (ret)
				return ret;
		}

		ft_start();
		for (int i = 0; i < opts.iterations; i++) {
			ret = efagda_post_send(gda_qp, ah, remote_qpn,
					       remote_qkey, (uintptr_t) tx_buf,
					       opts.transfer_size, lkey,
					       stream);
			FT_DEBUG("efagda_post_send: gda_qp = %p, ah = %u, "
				 "remote_qpn = %u, remote_qkey = %u, addr %p, "
				 "len %lu, lkey %lu\n",
				 gda_qp, ah, remote_qpn, remote_qkey,
				 (void *) (uintptr_t) tx_buf,
				 opts.transfer_size, lkey);
			if (ret) {
				FT_PRINTERR("efagda_post_send", -ret);
				return ret;
			}

			do {
				ret = efagda_poll_cq(gda_send_cq, 1, &wc, stream);
				if (ret > 0) {
					FT_DEBUG("client gets %d CQ entry successfully\n", ret);
					ret = 0;
					break;
				}
				if (ret < 0) {
					FT_PRINTERR("efagda_poll_cq", ret);
					return ret;
				}

				ft_stop();
				if ((end.tv_sec - start.tv_sec) > timeout) {
					fprintf(stderr,
						"client %ds timeout expired\n",
						timeout);
					return -FI_ENODATA;
				}
			} while (ret == 0);
		}
		ft_stop();
	} else {
		ft_start();
		for (int i = 0; i < opts.iterations; i++) {
			ret = efagda_post_recv(gda_qp, (uintptr_t) rx_buf,
					       opts.transfer_size, lkey, stream);
			FT_DEBUG("efagda_post_recv gda_qp = %p, addr = %p, "
				 "length = %zu, lkey = %lu\n",
				 gda_qp, (void *) (uintptr_t) rx_buf,
				 opts.transfer_size, lkey);
			if (ret) {
				FT_PRINTERR("efagda_post_recv", -ret);
				return ret;
			}

			do {
				ret = efagda_poll_cq(gda_recv_cq, 1, &wc,
						     stream);
				if (ret > 0) {
					FT_DEBUG("server gets %d CQ entry successfully\n", ret);
					ret = 0;
					goto verify_data;
				}
				if (ret < 0) {
					FT_PRINTERR("efagda_poll_cq", ret);
					return ret;
				}

				ft_stop();
				if ((end.tv_sec - start.tv_sec) > timeout) {
					fprintf(stderr,
						"server %ds timeout expired\n",
						timeout);
					return -FI_ENODATA;
				}
			} while (ret == 0);

		verify_data:
			if (ft_check_opts(FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE)) {
				ret = ft_check_buf((char *) rx_buf,
						   opts.transfer_size);
				if (ret)
					return ret;
			}
		}
		ft_stop();
	}

	show_perf(NULL, opts.transfer_size, opts.iterations, &start, &end, 2);

	return ret;
}

int main(int argc, char **argv)
{
	int op, ret, i;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_OOB_SYNC;

	timeout = 5;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	while ((op = getopt(argc, argv,
			    "vh" ADDR_OPTS INFO_OPTS CS_OPTS API_OPTS)) != -1) {
		switch (op) {
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			ft_parse_api_opts(op, optarg, hints, &opts);
			break;
		case 'v':
			opts.options |= FT_OPT_VERIFY_DATA | FT_OPT_ACTIVE;
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "GPU Direct Async test");
			FT_PRINT_OPTS_USAGE("-o <op>", "op: msg.\n");
			FT_PRINT_OPTS_USAGE("-v", "Enable data verification");
			return EXIT_FAILURE;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	hints->ep_attr->type = FI_EP_RDM;
	hints->caps |= FI_MSG | FI_RMA | FI_HMEM;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL |
				      FI_MR_VIRT_ADDR | FI_MR_PROV_KEY |
				      FI_MR_HMEM;
	hints->domain_attr->progress = FI_PROGRESS_MANUAL;
	hints->mode |= FI_CONTEXT | FI_CONTEXT2;

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

	ret = ft_enable_ep(gda_ep, eq, av, txcq_ext, rxcq_ext, txcntr, rxcntr,
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

	if (!(opts.options & FT_OPT_SIZE)) {
		for (i = 0; i < TEST_CNT; i++) {
			if (!ft_use_size(i, opts.sizes_enabled))
				continue;
			opts.transfer_size = test_size[i].size;
			init_test(&opts, test_name, sizeof(test_name));
			ret = run();
		}
	} else {
		init_test(&opts, test_name, sizeof(test_name));
		ret = run();
	}

	if (send_cq_buffer)
		efagda_destroy_cuda_cq(send_cq_buffer);
	if (recv_cq_buffer)
		efagda_destroy_cuda_cq(recv_cq_buffer);
	if (gda_qp)
		efagda_destroy_cuda_qp(gda_qp);
	// qp need to be destroyed before closing cq
	FT_CLOSE_FID(gda_ep);
	if (txcq_ext)
		fi_close(&txcq_ext->fid);
	if (rxcq_ext)
		fi_close(&rxcq_ext->fid);

	ft_free_res();
	return ft_exit_code(ret);
}

#endif /* HAVE_EFAGDA */
