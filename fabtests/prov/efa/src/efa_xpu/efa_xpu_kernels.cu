/*
 * Copyright (c) 2026, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

#include <cstdio>
#include <rdma/fi_xpu_device.h>
#include "efa_xpu_kernels.h"
#include <assert.h>

#define CNTR_TIMEOUT -1 /* infinite wait */

/*
 * One group's slice of a block-wide total, counted in operations.
 *
 * Every thread of a scope issues its own operation, so one call by a block of
 * blockDim.x threads produces blockDim.x operations whatever the scope; what the
 * scope changes is how they are batched, not how many there are. A block of
 * blockDim.x threads therefore walks @total in rounds of blockDim.x, and a group
 * of @gsize threads owns @gsize of the operations in each round.
 *
 * The result is the group's total and not one thread's share, which is what the
 * counters built from it track: the threads of a group each hold their own copy
 * of those counters, but every copy holds the same value, because a group posts
 * and completes as one. Only at FI_XPU_WORK_ITEM, where @gsize is 1 and a thread
 * is its own group, do they count what a single thread did.
 *
 * The split is even rather than by rank because a group-scope call is a
 * collective: a thread that made fewer calls than the rest would leave them
 * waiting at a barrier it never reaches. The host rounds every total down to a
 * multiple of the block size, so nothing is lost to the division and the block
 * issues exactly the total it was given.
 */
__device__ static int ft_efa_xpu_group_share(int total, int gsize)
{
	return total / (int) blockDim.x * gsize;
}

/*
 * How many threads issue an operation together, which is how many operations one
 * call makes: the calling thread alone at FI_XPU_WORK_ITEM, the converged lanes
 * at FI_XPU_SUBGROUP, the whole block at FI_XPU_WORK_GROUP. Counting operations
 * rather than calls is what keeps the completions balanced, because a group
 * claims completions once for all of its threads - see ft_efa_xpu_claim().
 *
 * The mask is what the provider forms its subgroup from, so taking it here gives
 * the same group, including the smaller one a block that is not a whole number
 * of warps ends with.
 */
__device__ static int ft_efa_xpu_group_size(int scope)
{
	switch (scope) {
	case FI_XPU_SUBGROUP:
		return __popc(__activemask());
	case FI_XPU_WORK_GROUP:
		return (int) blockDim.x;
	default:
		return 1;
	}
}

/*
 * Claim @count completions for the group.
 *
 * Reading a completion queue is the one call a group does not repeat per thread:
 * the group shares one buffer and one count, so the leader reads for all of them
 * and every thread takes back the same number. @count is therefore the group's
 * size rather than one, and the read is repeated - for what is left of @count,
 * never more - until they are all in. Asking for more than the group is owed
 * would take another group's completions and leave it waiting, since a
 * completion queue does not say whose operation finished.
 *
 * Returns @count, or the error the queue reported.
 */
__device__ static int64_t ft_efa_xpu_claim(struct fid_xpu_cq *cq, int count,
				      int scope)
{
	int got = 0;

	while (got < count) {
		int64_t cq_ret = fi_xpu_cq_read(cq, NULL, count - got, scope);

		if (cq_ret > 0)
			got += (int) cq_ret;
		else if (cq_ret != -FI_EAGAIN)
			return cq_ret;
	}

	return got;
}

__global__ void ft_efa_xpu_lat_send_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_send_cq,
	struct fid_xpu_cq *xpu_recv_cq,
	struct fid_xpu_cntr *xpu_send_cntr,
	struct fid_xpu_cntr *xpu_recv_cntr,
	void *dest_addr, size_t dest_addr_size,
	void *recv_buf, size_t recv_len,
	void *recv_desc, size_t recv_desc_size,
	void *send_buf, size_t send_len,
	void *send_desc, size_t send_desc_size,
	int iters, int rx_depth, int is_client, int scope)
{
	int gsize = ft_efa_xpu_group_size(scope);
	int group_ops, group_depth;
	/* Operations the group has sent and received, not calls it has made. */
	int scnt = 0;
	int rcnt = 0;
	int ret;
	uint64_t send_cntr_base = 0;
	uint64_t recv_cntr_base = 0;

	group_ops = ft_efa_xpu_group_share(iters, gsize);
	group_depth = ft_efa_xpu_group_share(rx_depth, gsize);

	if (xpu_send_cntr)
		send_cntr_base = fi_xpu_cntr_read(xpu_send_cntr, scope);
	if (xpu_recv_cntr)
		recv_cntr_base = fi_xpu_cntr_read(xpu_recv_cntr, scope);
	/*
	 * The threads share the counters, so they must all measure from the same
	 * starting point: none of them may post until every one has read the
	 * base.
	 */
	if (blockDim.x > 1)
		__syncthreads();

	/*
	 * Post initial receives batched: defer the RQ doorbell with FI_MORE
	 * on all but the last post, which rings it for the whole batch.
	 * (Our XPU API has no explicit flush; a post without FI_MORE flushes.)
	 */
	for (int i = 0; i < group_depth; i += gsize) {
		uint64_t rflags = (i + gsize < group_depth) ? FI_MORE : 0;

		ret = fi_xpu_recv(xpu_ep, recv_buf, recv_len, recv_desc,
				  NULL, NULL, rflags, scope);
		if (ret) {
			printf("fi_xpu_recv post failed: %d\n", ret);
			return;
		}
	}

	while (scnt < group_ops || rcnt < group_ops) {
		/* Poll for receive completion (except for first client send) */
		if (rcnt < group_ops && !(scnt < 1 && is_client == 1)) {
			if (xpu_recv_cntr) {
				fi_xpu_cntr_wait(xpu_recv_cntr,
						 recv_cntr_base + rcnt + gsize,
						 CNTR_TIMEOUT, scope);
			} else {
				int64_t cq_ret = ft_efa_xpu_claim(xpu_recv_cq,
							      gsize, scope);
				if (cq_ret < 0) {
					printf("ft_efa_xpu_lat: recv cq_read failed: %lld\n",
					       (long long) cq_ret);
					return;
				}
			}

			rcnt += gsize;

			/* Repost receive */
			if (rcnt + group_depth <= group_ops) {
				ret = fi_xpu_recv(xpu_ep, recv_buf, recv_len,
						  recv_desc, NULL, NULL, 0,
						  scope);
				if (ret) {
					printf("fi_xpu_recv repost failed: %d\n", ret);
					return;
				}
			}
		}

		/* Send */
		if (scnt < group_ops) {
			ret = fi_xpu_send(xpu_ep, send_buf, send_len,
					  send_desc, 0, dest_addr, NULL, 0,
					  scope);
			if (ret) {
				printf("fi_xpu_send failed: %d\n", ret);
				return;
			}
			scnt += gsize;

			/* Wait for send completion */
			if (xpu_send_cntr) {
				fi_xpu_cntr_wait(xpu_send_cntr,
						 send_cntr_base + scnt,
						 CNTR_TIMEOUT, scope);
			} else {
				int64_t cq_ret = ft_efa_xpu_claim(xpu_send_cq,
							      gsize, scope);
				if (cq_ret < 0) {
					printf("ft_efa_xpu_lat: send cq_read failed: %lld\n",
					       (long long) cq_ret);
					return;
				}
			}
		}
	}

	/*
	 * A counter counts every thread's completions, so a per-thread threshold
	 * only bounds the block's progress from below. Wait for the whole
	 * block's operations before leaving, so the host does not tear the
	 * endpoint down with any of them still in flight.
	 */
	if (blockDim.x > 1) {
		if (xpu_send_cntr)
			fi_xpu_cntr_wait(xpu_send_cntr, send_cntr_base + iters,
					 CNTR_TIMEOUT, scope);
		if (xpu_recv_cntr)
			fi_xpu_cntr_wait(xpu_recv_cntr, recv_cntr_base + iters,
					 CNTR_TIMEOUT, scope);
	}
}

int ft_efa_xpu_run_lat_send(struct fid_xpu_ep *xpu_ep,
			struct fid_xpu_cq *xpu_send_cq,
			struct fid_xpu_cq *xpu_recv_cq,
			struct fid_xpu_cntr *xpu_send_cntr,
			struct fid_xpu_cntr *xpu_recv_cntr,
			void *dest_addr, size_t dest_addr_size,
			void *recv_buf, size_t recv_len,
			void *recv_desc, size_t recv_desc_size,
			void *send_buf, size_t send_len,
			void *send_desc, size_t send_desc_size,
			int iters, int rx_depth, int is_client,
			int scope, int threads,
			cudaStream_t stream)
{
	cudaError_t err;

	ft_efa_xpu_lat_send_kernel<<<1, threads, 0, stream>>>(
		xpu_ep, xpu_send_cq, xpu_recv_cq,
		xpu_send_cntr, xpu_recv_cntr,
		dest_addr, dest_addr_size,
		recv_buf, recv_len, recv_desc, recv_desc_size,
		send_buf, send_len, send_desc, send_desc_size,
		iters, rx_depth, is_client, scope);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_lat_send: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_lat_send: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

__global__ void ft_efa_xpu_bw_tx_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_send_cq,
	struct fid_xpu_cntr *xpu_send_cntr,
	int opcode,
	void *send_buf, size_t send_len,
	void *send_desc, size_t send_desc_size,
	void *dest_addr, size_t dest_addr_size,
	uint64_t remote_addr, uint64_t remote_key,
	int iters, int tx_depth, int scope)
{
	int gsize = ft_efa_xpu_group_size(scope);
	int group_ops, group_depth;
	/* Operations the group has posted and completed, not calls it has made. */
	int scnt = 0;
	int ccnt = 0;
	int ret;
	uint64_t send_cntr_base = 0;

	group_ops = ft_efa_xpu_group_share(iters, gsize);
	group_depth = ft_efa_xpu_group_share(tx_depth, gsize);

	if (xpu_send_cntr) {
		send_cntr_base = fi_xpu_cntr_read(xpu_send_cntr, scope);
		/*
		 * The threads share the counter, so they must all measure from
		 * the same starting point: none of them may post until every one
		 * has read the base.
		 */
		if (blockDim.x > 1)
			__syncthreads();
	}

	while (scnt < group_ops || ccnt < group_ops) {
		/* Post operations up to this group's share of tx_depth */
		while (scnt < group_ops && (scnt - ccnt) < group_depth) {
			switch (opcode) {
			case 1: /* write */
				ret = fi_xpu_write(xpu_ep, send_buf, send_len,
						   send_desc, 0, dest_addr,
						   remote_addr, remote_key,
						   NULL, 0, scope);
				break;
			case 2: /* writedata (write with imm) */
				ret = fi_xpu_write(xpu_ep, send_buf, send_len,
						   send_desc, 0x12345678,
						   dest_addr, remote_addr,
						   remote_key, NULL,
						   FI_REMOTE_CQ_DATA, scope);
				break;
			case 3: /* read */
				ret = fi_xpu_read(xpu_ep, send_buf, send_len,
						  send_desc, dest_addr,
						  remote_addr, remote_key,
						  NULL, 0, scope);
				break;
			default: /* send */
				ret = fi_xpu_send(xpu_ep, send_buf, send_len,
						  send_desc, 0, dest_addr,
						  NULL, 0, scope);
				break;
			}
			if (ret) {
				printf("ft_efa_xpu_bw: post failed: %d at scnt=%d\n",
				       ret, scnt);
				return;
			}
			scnt += gsize;
		}

		/*
		 * Claim completions - a group claims them for all its threads,
		 * and never more than it is owed: a completion queue does not
		 * say which group's operation finished, so a group that asked
		 * for a whole group's worth when it was owed less would take
		 * another group's completions and leave it waiting.
		 */
		while (ccnt < scnt && (scnt == group_ops ||
		       (scnt - ccnt) >= group_depth)) {
			int owed = scnt - ccnt < gsize ? scnt - ccnt : gsize;

			if (xpu_send_cntr) {
				fi_xpu_cntr_wait(xpu_send_cntr,
						 send_cntr_base + ccnt + owed,
						 CNTR_TIMEOUT, scope);
				ccnt += owed;
			} else {
				int64_t cq_ret;
				cq_ret = fi_xpu_cq_read(xpu_send_cq, NULL,
							owed, scope);
				if (cq_ret > 0) {
					ccnt += (int) cq_ret;
				} else if (cq_ret != -FI_EAGAIN) {
					printf("ft_efa_xpu_bw: cq_read failed: %lld at ccnt=%d\n",
					       (long long) cq_ret, ccnt);
					return;
				}
			}
		}
	}

	/*
	 * A counter counts every thread's completions, so a per-thread threshold
	 * only bounds the block's progress from below. Wait for the whole
	 * block's operations before leaving, so the host neither reports a
	 * bandwidth for work still in flight nor tears the endpoint down under
	 * it. The completion queue needs no such wait: the threads consume one
	 * entry per operation between them, so every operation has completed
	 * once they have all finished counting.
	 */
	if (blockDim.x > 1 && xpu_send_cntr)
		fi_xpu_cntr_wait(xpu_send_cntr, send_cntr_base + iters,
				 CNTR_TIMEOUT, scope);
}

int ft_efa_xpu_run_bw(struct fid_xpu_ep *xpu_ep,
		  struct fid_xpu_cq *xpu_send_cq,
		  struct fid_xpu_cntr *xpu_send_cntr,
		  int opcode,
		  void *send_buf, size_t send_len,
		  void *send_desc, size_t send_desc_size,
		  void *dest_addr, size_t dest_addr_size,
		  uint64_t remote_addr, uint64_t remote_key,
		  int iters, int tx_depth, int scope, int threads,
		  cudaStream_t stream)
{
	cudaError_t err;

	ft_efa_xpu_bw_tx_kernel<<<1, threads, 0, stream>>>(
		xpu_ep, xpu_send_cq, xpu_send_cntr, opcode,
		send_buf, send_len, send_desc, send_desc_size,
		dest_addr, dest_addr_size,
		remote_addr, remote_key,
		iters, tx_depth, scope);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_bw: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_bw: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

__global__ void ft_efa_xpu_bw_rx_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_recv_cq,
	struct fid_xpu_cntr *xpu_recv_cntr,
	void *recv_buf, size_t recv_len,
	void *recv_desc, size_t recv_desc_size,
	int iters, int rx_depth, int scope)
{
	int gsize = ft_efa_xpu_group_size(scope);
	int group_ops, group_depth;
	/* Operations the group has received, not calls it has made. */
	int rcnt = 0;
	int ret;
	uint64_t recv_cntr_base = 0;

	group_ops = ft_efa_xpu_group_share(iters, gsize);
	group_depth = ft_efa_xpu_group_share(rx_depth, gsize);

	if (xpu_recv_cntr) {
		recv_cntr_base = fi_xpu_cntr_read(xpu_recv_cntr, scope);
		/*
		 * The threads share the counter, so they must all measure from
		 * the same starting point: none of them may post until every one
		 * has read the base.
		 */
		if (blockDim.x > 1)
			__syncthreads();
	}

	/*
	 * Post initial receives batched: defer the RQ doorbell with FI_MORE
	 * on all but the last post, which rings it for the whole batch.
	 */
	for (int i = 0; i < group_depth; i += gsize) {
		uint64_t rflags = (i + gsize < group_depth) ? FI_MORE : 0;

		ret = fi_xpu_recv(xpu_ep, recv_buf, recv_len, recv_desc,
				  NULL, NULL, rflags, scope);
		if (ret) {
			printf("ft_efa_xpu_bw_recv: post failed: %d\n", ret);
			return;
		}
	}

	while (rcnt < group_ops) {
		if (xpu_recv_cntr) {
			fi_xpu_cntr_wait(xpu_recv_cntr,
					 recv_cntr_base + rcnt + gsize,
					 CNTR_TIMEOUT, scope);
		} else {
			int64_t cq_ret = ft_efa_xpu_claim(xpu_recv_cq, gsize, scope);

			if (cq_ret < 0) {
				printf("ft_efa_xpu_bw_recv: cq_read failed: %lld\n",
				       (long long) cq_ret);
				return;
			}
		}

		rcnt += gsize;

		/* Repost receive */
		if (rcnt + group_depth <= group_ops) {
			ret = fi_xpu_recv(xpu_ep, recv_buf, recv_len,
					  recv_desc, NULL, NULL, 0, scope);
			if (ret) {
				printf("ft_efa_xpu_bw_recv: repost failed: %d\n", ret);
				return;
			}
		}
	}

	/* The counter counts every thread's completions - see the tx kernel. */
	if (blockDim.x > 1 && xpu_recv_cntr)
		fi_xpu_cntr_wait(xpu_recv_cntr, recv_cntr_base + iters,
				 CNTR_TIMEOUT, scope);
}

int ft_efa_xpu_run_bw_recv(struct fid_xpu_ep *xpu_ep,
		       struct fid_xpu_cq *xpu_recv_cq,
		       struct fid_xpu_cntr *xpu_recv_cntr,
		       void *recv_buf, size_t recv_len,
		       void *recv_desc, size_t recv_desc_size,
		       int iters, int rx_depth, int scope, int threads,
		       cudaStream_t stream)
{
	cudaError_t err;

	ft_efa_xpu_bw_rx_kernel<<<1, threads, 0, stream>>>(
		xpu_ep, xpu_recv_cq, xpu_recv_cntr,
		recv_buf, recv_len, recv_desc, recv_desc_size,
		iters, rx_depth, scope);

	err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_bw_recv: launch failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaStreamSynchronize(stream);
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_bw_recv: kernel failed: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return 0;
}

/*
 * A scope the provider does not implement - FI_XPU_DEVICE, which would need a
 * barrier across the whole grid, and anything the build has never heard of - is
 * refused before anything is posted, so an application gets an error rather
 * than a queue it has half committed to. Each entry point has to refuse on its
 * own, so each one is checked.
 */
__global__ void ft_efa_xpu_scope_reject_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_cq,
	void *buf, size_t len, void *desc,
	void *dest_addr, int scope, int *failures)
{
	int fails = 0;
	int64_t cq_ret;
	int ret;

	ret = fi_xpu_send(xpu_ep, buf, len, desc, 0, dest_addr, NULL, 0, scope);
	if (ret != -FI_EOPNOTSUPP) {
		printf("fi_xpu_send at scope %d: %d, expected %d\n", scope, ret,
		       -FI_EOPNOTSUPP);
		fails++;
	}

	ret = fi_xpu_recv(xpu_ep, buf, len, desc, NULL, NULL, 0, scope);
	if (ret != -FI_EOPNOTSUPP) {
		printf("fi_xpu_recv at scope %d: %d, expected %d\n", scope, ret,
		       -FI_EOPNOTSUPP);
		fails++;
	}

	ret = fi_xpu_write(xpu_ep, buf, len, desc, 0, dest_addr, 0, 0, NULL, 0,
			   scope);
	if (ret != -FI_EOPNOTSUPP) {
		printf("fi_xpu_write at scope %d: %d, expected %d\n", scope, ret,
		       -FI_EOPNOTSUPP);
		fails++;
	}

	ret = fi_xpu_read(xpu_ep, buf, len, desc, dest_addr, 0, 0, NULL, 0,
			  scope);
	if (ret != -FI_EOPNOTSUPP) {
		printf("fi_xpu_read at scope %d: %d, expected %d\n", scope, ret,
		       -FI_EOPNOTSUPP);
		fails++;
	}

	cq_ret = fi_xpu_cq_read(xpu_cq, NULL, 1, scope);
	if (cq_ret != -FI_EOPNOTSUPP) {
		printf("fi_xpu_cq_read at scope %d: %lld, expected %d\n", scope,
		       (long long) cq_ret, -FI_EOPNOTSUPP);
		fails++;
	}

	*failures = fails;
}

int ft_efa_xpu_run_scope_reject(struct fid_xpu_ep *xpu_ep,
			    struct fid_xpu_cq *xpu_cq,
			    void *buf, size_t len, void *desc,
			    void *dest_addr, int scope,
			    cudaStream_t stream)
{
	cudaError_t err;
	int *dev_failures;
	int failures = -1;

	err = cudaMalloc((void **) &dev_failures, sizeof(*dev_failures));
	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_scope_reject: cudaMalloc: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	/* One thread: an unsupported scope is refused without any cooperation. */
	ft_efa_xpu_scope_reject_kernel<<<1, 1, 0, stream>>>(
		xpu_ep, xpu_cq, buf, len, desc, dest_addr, scope,
		dev_failures);

	err = cudaGetLastError();
	if (err == cudaSuccess)
		err = cudaMemcpyAsync(&failures, dev_failures,
				      sizeof(failures), cudaMemcpyDeviceToHost,
				      stream);
	if (err == cudaSuccess)
		err = cudaStreamSynchronize(stream);

	cudaFree(dev_failures);

	if (err != cudaSuccess) {
		fprintf(stderr, "ft_efa_xpu_run_scope_reject: %s\n",
			cudaGetErrorString(err));
		return -1;
	}

	return failures ? -1 : 0;
}
