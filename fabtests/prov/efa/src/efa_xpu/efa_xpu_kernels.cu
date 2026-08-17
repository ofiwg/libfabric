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
 * What a run hands back to the host, and what it measures its counters from.
 *
 * @failures counts the device-side errors of the run - a refused post, a failed
 * completion read, an error counted on a counter - so the host reports a failure
 * rather than a pass when one happens. The kernels print each one too.
 *
 * The counter bases are read before the kernel starts, by a kernel of their own
 * (see ft_efa_xpu_begin()). The threads of a run share the counters, so they
 * must all measure from the same starting point, and with more than one block
 * no kernel can arrange that itself: a block cannot wait for another, so a late
 * block reading the base would read one that the other blocks had already
 * moved.
 */
/* The most issuers a run may have: 256 threads in each of 32 blocks. */
#define FT_EFA_XPU_MAX_ISSUERS	8192

struct ft_efa_xpu_status {
	int		failures;
	int		pad;
	uint64_t	cntr_base[2];
	uint64_t	err_base[2];
	/*
	 * What the run's poller (ft_efa_xpu_poller()) has read off each
	 * completion queue: how many completions in all, how many for each
	 * issuer - its context says whose - and how many with no context to
	 * say, which an issuer that is owed one may claim.
	 */
	uint32_t	total[2];
	uint32_t	anon[2];
	uint32_t	done[2][FT_EFA_XPU_MAX_ISSUERS];
};

enum {
	FT_EFA_XPU_SEND = 0,
	FT_EFA_XPU_RECV = 1,
};

/*
 * How many blocks at the end of the grid are not issuers: the poller's, in a
 * run whose completions the device reads, or none. Set by ft_efa_xpu_begin().
 */
__device__ int ft_efa_xpu_poller_blocks;

/*
 * How many endpoints the run posts on (--eps): the handles the kernels are
 * given are an array of this many, and issuer i posts on endpoint i % eps, so
 * the lanes of a warp posting at the work item scope reach different QPs.
 */
__device__ int ft_efa_xpu_eps = 1;

/*
 * --readers: the issuers read the CQ themselves, concurrently, instead of the
 * run's one poller; see ft_efa_xpu_await().
 */
static int ft_efa_xpu_readers;

void ft_efa_xpu_set_readers(int readers)
{
	ft_efa_xpu_readers = readers;
}

int ft_efa_xpu_set_eps(int eps)
{
	return cudaMemcpyToSymbol(ft_efa_xpu_eps, &eps, sizeof(eps)) ==
	       cudaSuccess ? 0 : -1;
}

/*
 * The issuers of a call.
 *
 * A call at a group scope is one operation for the group (fi_xpu(3)), and a
 * call at FI_XPU_WORK_ITEM one for its thread, so what a call by the grid makes
 * is one operation per issuer: per thread at the work item scope, per warp at
 * the subgroup scope, per block at the work group scope. A block that is not a
 * whole number of warps ends in a smaller warp, which is an issuer like the
 * rest. The host counts them the same way (xpu_issuers()).
 */
__device__ static int ft_efa_xpu_block_issuers(int scope)
{
	switch (scope) {
	case FI_XPU_SUBGROUP:
		return (int) ((blockDim.x + warpSize - 1) / warpSize);
	case FI_XPU_WORK_GROUP:
		return 1;
	default:
		return (int) blockDim.x;
	}
}

/* Every issuer of the grid, which is how many operations one call makes. */
__device__ static int ft_efa_xpu_issuers(int scope)
{
	return ft_efa_xpu_block_issuers(scope) *
	       ((int) gridDim.x - ft_efa_xpu_poller_blocks);
}

/* Lanes in this thread's warp: a full warp, or what the block has left. */
__device__ static int ft_efa_xpu_warp_lanes(void)
{
	int first = (int) (threadIdx.x / warpSize) * warpSize;
	int left = (int) blockDim.x - first;

	return left < warpSize ? left : warpSize;
}

/* The lanes of this thread's warp that the block has, as a mask. */
__device__ static unsigned int ft_efa_xpu_warp_mask(void)
{
	int lanes = ft_efa_xpu_warp_lanes();

	return lanes == 32 ? 0xffffffffu : (1u << lanes) - 1u;
}

/*
 * The issuer this thread belongs to, as an index across the grid. Every thread
 * of a group computes the same one, which is what makes the arguments derived
 * from it identical across the group, as a group-scope call requires.
 */
__device__ static int ft_efa_xpu_issuer(int scope)
{
	int in_block;

	switch (scope) {
	case FI_XPU_SUBGROUP:
		in_block = (int) (threadIdx.x / warpSize);
		break;
	case FI_XPU_WORK_GROUP:
		in_block = 0;
		break;
	default:
		in_block = (int) threadIdx.x;
		break;
	}

	return (int) blockIdx.x * ft_efa_xpu_block_issuers(scope) + in_block;
}

/*
 * Bring a warp back together before a subgroup-scope call.
 *
 * The subgroup a call is made by is the lanes converged at it, so a warp that
 * arrived split would make one call per piece - and with it more operations
 * than the warp is meant to - instead of the one this test counts on.
 */
__device__ static void ft_efa_xpu_converge(int scope)
{
	if (scope == FI_XPU_SUBGROUP) {
		int lanes = ft_efa_xpu_warp_lanes();

		__syncwarp(lanes == 32 ? 0xffffffffu : (1u << lanes) - 1u);
	}
}

/*
 * One issuer's slice of a grid-wide total, counted in operations.
 *
 * Every call by the grid makes one operation per issuer, so the grid walks
 * @total in rounds of its issuer count and each issuer owns one operation of
 * every round. The split is even because a group-scope call is a collective:
 * a group that made fewer calls than the rest would leave its threads waiting
 * at a barrier the others never reach. The host rounds every total down to a
 * multiple of the round, so nothing is lost to the division.
 */
__device__ static int ft_efa_xpu_issuer_share(int total, int scope)
{
	return total / ft_efa_xpu_issuers(scope);
}

/*
 * Fill this issuer's @chunk with the pattern the host checks it against
 * (xpu_pt_byte() in efa_xpu.c), every thread of the group writing a share of
 * the bytes. The group's call is made by its leader alone, so a chunk that
 * arrives whole is what shows fi_xpu(3)'s promise kept: what the threads of a
 * scope wrote before a call is visible to its operation.
 */
__device__ static void ft_efa_xpu_fill(void *chunk, size_t len, int issuer,
				       int scope)
{
	size_t rank, size;

	switch (scope) {
	case FI_XPU_SUBGROUP:
		rank = threadIdx.x % warpSize;
		size = (size_t) ft_efa_xpu_warp_lanes();
		break;
	case FI_XPU_WORK_GROUP:
		rank = threadIdx.x;
		size = blockDim.x;
		break;
	default:
		rank = 0;
		size = 1;
		break;
	}

	/*
	 * Every warp but the leader's writes late, so that a call which let the
	 * leader post before the group had finished would send the chunk short
	 * of their bytes, rather than winning the race by a margin no check
	 * could see. Not the leader's own warp: it reconverges after the delay,
	 * which would hold the leader back with it.
	 */
	if (rank >= (size_t) warpSize)
		__nanosleep(100000);

	for (size_t off = rank; off < len; off += size)
		((uint8_t *) chunk)[off] = (uint8_t) ((size_t) issuer + off * 31);
}

__device__ static void ft_efa_xpu_fail(struct ft_efa_xpu_status *st);

/*
 * The context every operation carries: its issuer and its sequence among that
 * issuer's operations, each plus one so that no context is NULL. A completion
 * hands it back, which is what checks that the provider returns the context a
 * post was given - from the descriptor's 64-bit request ID, or from the
 * context table a 16-bit one indexes - rather than anything standing in for it.
 * The host decodes the same encoding (xpu_ctx_decode() in efa_xpu.c).
 */
#define FT_EFA_XPU_CTX(issuer, seq) \
	((void *) (uintptr_t) ((((uint64_t) (issuer) + 1) << 32) | \
			       ((uint64_t) (seq) + 1)))


/*
 * Check that a completion's context is one a post of this run was given, by
 * any issuer of up to @max_seq operations. An unsolicited write with immediate
 * data completes with no receive behind it, and so with no context, which
 * @null_ok allows.
 */
__device__ static int ft_efa_xpu_check_ctx(const void *ctx, int scope,
					   int max_seq, bool null_ok,
					   struct ft_efa_xpu_status *st)
{
	uint64_t v = (uint64_t) (uintptr_t) ctx;
	uint64_t issuer = v >> 32;
	uint64_t seq = v & 0xffffffffu;

	if (!ctx && null_ok)
		return 0;
	if (issuer && issuer <= (uint64_t) ft_efa_xpu_issuers(scope) &&
	    seq && seq <= (uint64_t) max_seq)
		return 0;

	printf("ft_efa_xpu: completion context %#llx is no post's\n",
	       (unsigned long long) v);
	ft_efa_xpu_fail(st);
	return -1;
}


/*
 * Wait for @cntr to reach @threshold, or for an error to be counted on it.
 *
 * fi_xpu_cntr_wait() spins until the threshold is met, which after a failed
 * operation it never is: the failure is counted on the error counter instead. So
 * watch both, and hand over to fi_xpu_cntr_wait() once the threshold is known
 * to be met, so the call is still exercised. No counter call is collective, so
 * every thread waits for itself and the scope only says who is calling.
 *
 * Returns 0, or -FI_EIO if an error was counted since the run began.
 */
__device__ static int ft_efa_xpu_cntr_wait(struct fid_xpu_cntr *cntr,
					   uint64_t threshold, uint64_t err_base,
					   int scope)
{
	while (fi_xpu_cntr_read(cntr, FI_XPU_WORK_ITEM) < threshold) {
		if (fi_xpu_cntr_readerr(cntr, FI_XPU_WORK_ITEM) != err_base)
			return -FI_EIO;
	}

	fi_xpu_cntr_wait(cntr, threshold, CNTR_TIMEOUT, scope);
	return 0;
}


__device__ static void ft_efa_xpu_fail(struct ft_efa_xpu_status *st)
{
	atomicAdd(&st->failures, 1);
}

__device__ static bool ft_efa_xpu_failed(struct ft_efa_xpu_status *st)
{
	return *(volatile int *) &st->failures != 0;
}

/* How many completions one poll asks for. */
#define FT_EFA_XPU_POLL_BATCH	16
/* Completions a group reads at once with --readers at a group scope. */
#define FT_EFA_XPU_GROUP_READ_BATCH	8

/*
 * The one thread that reads the run's completion queues.
 *
 * A completion queue has at most one poller at a time (fi_efa(7)), so the
 * issuers never read it: the run's last block is the poller's, and its first
 * thread reads each queue until it has taken @want[q] completions, checking
 * every context and crediting each completion to the issuer it names. That
 * holds in counter mode too, where the issuers wait on counters: every CQ
 * bound to the endpoint still has to be read, since on EFA a queue whose
 * contexts go through a context table reuses a slot only once the completion
 * of its previous occupant has been read, and posts stall once the ring wraps
 * if nothing reads them. A failure anywhere ends the poll, so a run that fails
 * still finishes.
 */
__device__ static void ft_efa_xpu_poller(struct fid_xpu_cq *cq[2],
					 const uint32_t want[2], int scope,
					 int max_seq, bool null_ok,
					 struct ft_efa_xpu_status *st)
{
	struct fi_cq_msg_entry e[FT_EFA_XPU_POLL_BATCH];

	for (;;) {
		bool busy = false;

		if (ft_efa_xpu_failed(st))
			return;

		for (int q = 0; q < 2; q++) {
			int64_t n;

			if (!cq[q] || st->total[q] >= want[q])
				continue;
			busy = true;

			n = fi_xpu_cq_read(cq[q], e, FT_EFA_XPU_POLL_BATCH,
					   FI_XPU_WORK_ITEM);
			if (n == -FI_EAGAIN)
				continue;
			if (n < 0) {
				printf("ft_efa_xpu_poller: cq_read failed: %lld "
				       "after %u\n", (long long) n,
				       st->total[q]);
				ft_efa_xpu_fail(st);
				return;
			}

			for (int64_t i = 0; i < n; i++) {
				uint64_t v = (uint64_t) (uintptr_t)
					     e[i].op_context;

				if (ft_efa_xpu_check_ctx(e[i].op_context,
							 scope, max_seq,
							 null_ok, st))
					return;
				if (v)
					atomicAdd(&st->done[q][(v >> 32) - 1],
						  1u);
				else
					atomicAdd(&st->anon[q], 1u);
			}
			st->total[q] += (uint32_t) n;
		}

		if (!busy)
			return;
	}
}

/*
 * Wait until the poller has read @n completions for issuer @me on queue @q.
 *
 * A completion with no context - an unsolicited write with immediate data -
 * says no issuer's name, so an issuer still owed one may take it instead.
 * One thread of the group waits and the group then meets, so every thread
 * leaves with the same answer.
 *
 * The lanes meet on the warp's own mask rather than on whatever __activemask()
 * reports: the waiter spins while the rest are already here, so the lanes are
 * divergent, and a mask they do not all agree on would leave them that way -
 * after which a group-scope post would see a split warp and post once for each
 * piece.
 *
 * Returns 0, or -1 if the run failed while waiting.
 */
__device__ static int ft_efa_xpu_await(int q, int me, uint32_t n, int scope,
				       uint32_t *anon_taken,
				       struct fid_xpu_cq *cq, int max_seq,
				       bool null_ok,
				       struct ft_efa_xpu_status *st)
{
	__shared__ int group_ret;
	__shared__ struct fi_cq_msg_entry
		group_entry[32][FT_EFA_XPU_GROUP_READ_BATCH];
	bool waiter;
	int ret = 0;

	/*
	 * Without a poller (--readers), the issuers read the CQ themselves,
	 * crediting each completion to the issuer its context names exactly as
	 * the poller would, until their own credit is in. At a group scope the
	 * group reads up to FT_EFA_XPU_GROUP_READ_BATCH completions together
	 * into a shared buffer, which the provider may divide among the group's
	 * threads; at the work item scope each thread reads one completion into
	 * an entry of its own, so the lanes of a warp are separate reads racing
	 * for the queue, and each takes its own completions.
	 */
	if (!ft_efa_xpu_poller_blocks && cq) {
		for (;;) {
			struct fi_cq_msg_entry own = {};
			struct fi_cq_msg_entry *e;
			size_t count = FT_EFA_XPU_GROUP_READ_BATCH;
			bool owed, checker;
			int64_t r;

			switch (scope) {
			case FI_XPU_SUBGROUP:
				e = group_entry[threadIdx.x / warpSize];
				checker = threadIdx.x % warpSize == 0;
				break;
			case FI_XPU_WORK_GROUP:
				e = group_entry[0];
				checker = threadIdx.x == 0;
				break;
			default:
				e = &own;
				count = 1;
				checker = true;
				break;
			}

			owed = *(volatile uint32_t *) &st->done[q][me] +
			       *anon_taken < n;
			if (scope == FI_XPU_SUBGROUP)
				owed = __shfl_sync(ft_efa_xpu_warp_mask(),
						   owed, 0);
			else if (scope == FI_XPU_WORK_GROUP)
				owed = __syncthreads_or(owed && checker);
			if (!owed || ft_efa_xpu_failed(st))
				return ft_efa_xpu_failed(st) ? -1 : 0;

			/*
			 * Every thread of the group gets the same r, so the
			 * group leaves together; a bad context is recorded
			 * by the checker alone, and the group sees it as a
			 * failed run at the top of the next pass.
			 */
			/*
			 * A subgroup-scope read is made by the lanes converged
			 * at the call. The checker's credit loop above leaves
			 * the warp diverged, and a read made by part of it
			 * would take completions no checker credits, so the
			 * warp meets first and reads as one group.
			 */
			ft_efa_xpu_converge(scope);
			r = fi_xpu_cq_read(cq, e, count, scope);
			if (r > 0 && checker) {
				for (int64_t i = 0; i < r; i++) {
					uint64_t v = (uint64_t) (uintptr_t)
						     e[i].op_context;

					if (ft_efa_xpu_check_ctx(e[i].op_context,
								 scope, max_seq,
								 null_ok, st))
						continue;
					if (v)
						atomicAdd(&st->done[q][(v >> 32) - 1],
							  1u);
					else
						(*anon_taken)++;
				}
			} else if (r < 0 && r != -FI_EAGAIN) {
				if (checker) {
					printf("ft_efa_xpu: cq_read failed: "
					       "%lld\n", (long long) r);
					ft_efa_xpu_fail(st);
				}
				return -1;
			}
			if (scope == FI_XPU_WORK_GROUP)
				__syncthreads();
		}
	}

	switch (scope) {
	case FI_XPU_SUBGROUP:
		waiter = threadIdx.x % warpSize == 0;
		break;
	case FI_XPU_WORK_GROUP:
		waiter = threadIdx.x == 0;
		break;
	default:
		waiter = true;
		break;
	}

	if (waiter) {
		volatile uint32_t *done = &st->done[q][me];
		volatile uint32_t *anon = &st->anon[q];

		while (*done + *anon_taken < n) {
			uint32_t a = *anon;

			if (ft_efa_xpu_failed(st)) {
				ret = -1;
				break;
			}
			if (a && atomicCAS((unsigned int *) anon, a, a - 1) == a)
				(*anon_taken)++;
		}
	}

	switch (scope) {
	case FI_XPU_SUBGROUP: {
		unsigned int mask = ft_efa_xpu_warp_mask();

		__syncwarp(mask);
		ret = __shfl_sync(mask, ret, 0);
		break;
	}
	case FI_XPU_WORK_GROUP:
		if (waiter)
			group_ret = ret;
		__syncthreads();
		ret = group_ret;
		__syncthreads();
		break;
	default:
		break;
	}
	return ret;
}

/* Read the counter bases a run measures from - see struct ft_efa_xpu_status. */
__global__ void ft_efa_xpu_cntr_base_kernel(struct fid_xpu_cntr *send_cntr,
					    struct fid_xpu_cntr *recv_cntr,
					    struct ft_efa_xpu_status *st)
{
	if (send_cntr) {
		st->cntr_base[FT_EFA_XPU_SEND] =
			fi_xpu_cntr_read(send_cntr, FI_XPU_WORK_ITEM);
		st->err_base[FT_EFA_XPU_SEND] =
			fi_xpu_cntr_readerr(send_cntr, FI_XPU_WORK_ITEM);
	}
	if (recv_cntr) {
		st->cntr_base[FT_EFA_XPU_RECV] =
			fi_xpu_cntr_read(recv_cntr, FI_XPU_WORK_ITEM);
		st->err_base[FT_EFA_XPU_RECV] =
			fi_xpu_cntr_readerr(recv_cntr, FI_XPU_WORK_ITEM);
	}
}

/*
 * Set up a run: allocate the status block the kernel reports into and, when the
 * run uses counters, fill in their bases ahead of it on the same stream.
 */
static int ft_efa_xpu_begin(const char *who, struct fid_xpu_cntr *send_cntr,
			    struct fid_xpu_cntr *recv_cntr, int poller_blocks,
			    struct ft_efa_xpu_status **st, cudaStream_t stream)
{
	cudaError_t err;

	err = cudaMalloc((void **) st, sizeof(**st));
	if (err != cudaSuccess) {
		fprintf(stderr, "%s: cudaMalloc: %s\n", who,
			cudaGetErrorString(err));
		return -1;
	}

	err = cudaMemsetAsync(*st, 0, sizeof(**st), stream);
	if (err == cudaSuccess)
		err = cudaMemcpyToSymbolAsync(ft_efa_xpu_poller_blocks,
					      &poller_blocks,
					      sizeof(poller_blocks), 0,
					      cudaMemcpyHostToDevice, stream);
	if (err == cudaSuccess)
		err = cudaStreamSynchronize(stream);
	if (err == cudaSuccess && (send_cntr || recv_cntr)) {
		ft_efa_xpu_cntr_base_kernel<<<1, 1, 0, stream>>>(send_cntr,
								 recv_cntr,
								 *st);
		err = cudaGetLastError();
	}
	if (err != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", who, cudaGetErrorString(err));
		cudaFree(*st);
		return -1;
	}

	return 0;
}

/*
 * After a run, check that it completed exactly the operations it was owed and
 * no more: @cq stays empty, or @cntr stays at its base plus @ops. A group
 * call that made an operation per thread instead of one for the group leaves
 * completions behind that the run never claimed; one that made fewer never
 * lets the run finish. They
 * arrive with the rest, so a short watch is enough to see them.
 */
#define FT_EFA_XPU_EXACT_SPINS (1 << 16)

__global__ void ft_efa_xpu_exact_kernel(struct fid_xpu_cq *cq,
					struct fid_xpu_cntr *cntr, int which,
					uint64_t ops,
					struct ft_efa_xpu_status *st)
{
	uint64_t expected = st->cntr_base[which] + ops;

	for (int i = 0; i < FT_EFA_XPU_EXACT_SPINS; i++) {
		if (cntr) {
			uint64_t v = fi_xpu_cntr_read(cntr, FI_XPU_WORK_ITEM);

			if (v != expected) {
				printf("ft_efa_xpu_exact: counter at %llu, "
				       "expected %llu\n", (unsigned long long) v,
				       (unsigned long long) expected);
				ft_efa_xpu_fail(st);
				return;
			}
		} else {
			struct fi_cq_data_entry e;
			int64_t r = fi_xpu_cq_read(cq, &e, 1,
						   FI_XPU_WORK_ITEM);

			if (r != -FI_EAGAIN) {
				printf("ft_efa_xpu_exact: completion left "
				       "after the run: %lld\n", (long long) r);
				ft_efa_xpu_fail(st);
				return;
			}
		}
	}
}

/*
 * Finish a run whose kernel was just launched: wait for it and report whether
 * it, or anything it did on the device, failed.
 */
static int ft_efa_xpu_end(const char *who, struct ft_efa_xpu_status *st,
			  cudaStream_t stream)
{
	cudaError_t err;
	int failures = -1;

	err = cudaGetLastError();
	if (err == cudaSuccess)
		err = cudaMemcpyAsync(&failures, &st->failures,
				      sizeof(failures), cudaMemcpyDeviceToHost,
				      stream);
	if (err == cudaSuccess)
		err = cudaStreamSynchronize(stream);

	cudaFree(st);

	if (err != cudaSuccess) {
		fprintf(stderr, "%s: %s\n", who, cudaGetErrorString(err));
		return -1;
	}
	if (failures)
		fprintf(stderr, "%s: %d device-side failure(s)\n", who,
			failures);

	return failures ? -1 : 0;
}

/* Whether this thread's block is the run's poller rather than an issuer's. */
__device__ static bool ft_efa_xpu_is_poller_block(void)
{
	return (int) blockIdx.x >= (int) gridDim.x - ft_efa_xpu_poller_blocks;
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
	int iters, int rx_depth, int is_client, int scope, int fill,
	struct ft_efa_xpu_status *st)
{
	int my_ops, my_depth;
	/* Operations this issuer has sent and received. */
	int scnt = 0;
	int rcnt = 0;
	int ret;
	uint64_t send_cntr_base = st->cntr_base[FT_EFA_XPU_SEND];
	uint64_t recv_cntr_base = st->cntr_base[FT_EFA_XPU_RECV];
	/*
	 * Every issuer posts from arguments of its own - its own chunk of the
	 * buffers and its own copy of the peer's address - which every thread
	 * of a group derives from the same issuer index, so a group passes
	 * identical arguments to its one call.
	 */
	int me = ft_efa_xpu_issuer(scope);
	/* This issuer's endpoint - see ft_efa_xpu_eps. */
	struct fid_xpu_ep *my_ep = xpu_ep + me % ft_efa_xpu_eps;
	void *my_recv = (char *) recv_buf + (size_t) me * recv_len;
	void *my_send = (char *) send_buf + (size_t) me * send_len;
	void *my_addr = (char *) dest_addr + (size_t) me * dest_addr_size;
	uint32_t anon_taken[2] = { 0, 0 };

	my_ops = ft_efa_xpu_issuer_share(iters, scope);
	my_depth = ft_efa_xpu_issuer_share(rx_depth, scope);

	if (ft_efa_xpu_is_poller_block()) {
		struct fid_xpu_cq *cq[2] = { xpu_send_cq, xpu_recv_cq };
		const uint32_t want[2] = { (uint32_t) iters, (uint32_t) iters };

		if (threadIdx.x == 0)
			ft_efa_xpu_poller(cq, want, scope, my_ops, false, st);
		return;
	}

	if (fill)
		ft_efa_xpu_fill(my_send, send_len, me, scope);

	/*
	 * Post initial receives batched: defer the RQ doorbell with FI_MORE
	 * on all but the last post, which rings it for the whole batch.
	 * (Our XPU API has no explicit flush; a post without FI_MORE flushes.)
	 */
	for (int i = 0; i < my_depth; i++) {
		uint64_t rflags = (i + 1 < my_depth) ? FI_MORE : 0;

		ft_efa_xpu_converge(scope);
		ret = fi_xpu_recv(my_ep, my_recv, recv_len, recv_desc,
				  NULL, FT_EFA_XPU_CTX(me, i), rflags, scope);
		if (ret) {
			printf("fi_xpu_recv post failed: %d\n", ret);
			ft_efa_xpu_fail(st);
			return;
		}
	}

	while (scnt < my_ops || rcnt < my_ops) {
		/* Poll for receive completion (except for first client send) */
		if (rcnt < my_ops && !(scnt < 1 && is_client == 1)) {
			if (xpu_recv_cntr) {
				ret = ft_efa_xpu_cntr_wait(
					xpu_recv_cntr,
					recv_cntr_base + rcnt + 1,
					st->err_base[FT_EFA_XPU_RECV], scope);
				if (ret) {
					printf("ft_efa_xpu_lat: recv cntr error\n");
					ft_efa_xpu_fail(st);
					return;
				}
			} else if (ft_efa_xpu_await(FT_EFA_XPU_RECV, me,
						    (uint32_t) rcnt + 1, scope,
						    &anon_taken[FT_EFA_XPU_RECV],
						    xpu_recv_cq, my_ops, false,
						    st)) {
				return;
			}

			rcnt++;

			/* Repost receive */
			if (rcnt + my_depth <= my_ops) {
				ft_efa_xpu_converge(scope);
				ret = fi_xpu_recv(my_ep, my_recv, recv_len,
						  recv_desc, NULL,
						  FT_EFA_XPU_CTX(me, rcnt +
								 my_depth - 1),
						  0, scope);
				if (ret) {
					printf("fi_xpu_recv repost failed: %d\n", ret);
					ft_efa_xpu_fail(st);
					return;
				}
			}
		}

		/* Send */
		if (scnt < my_ops) {
			ft_efa_xpu_converge(scope);
			ret = fi_xpu_send(my_ep, my_send, send_len,
					  send_desc, 0, my_addr,
					  FT_EFA_XPU_CTX(me, scnt), 0, scope);
			if (ret) {
				printf("fi_xpu_send failed: %d\n", ret);
				ft_efa_xpu_fail(st);
				return;
			}
			scnt++;

			/* Wait for send completion */
			if (xpu_send_cntr) {
				ret = ft_efa_xpu_cntr_wait(
					xpu_send_cntr, send_cntr_base + scnt,
					st->err_base[FT_EFA_XPU_SEND], scope);
				if (ret) {
					printf("ft_efa_xpu_lat: send cntr error\n");
					ft_efa_xpu_fail(st);
					return;
				}
			} else if (ft_efa_xpu_await(FT_EFA_XPU_SEND, me,
						    (uint32_t) scnt, scope,
						    &anon_taken[FT_EFA_XPU_SEND],
						    xpu_send_cq, my_ops, false,
						    st)) {
				return;
			}
		}
	}

	/*
	 * A counter counts every issuer's completions, so a per-issuer threshold
	 * only bounds the grid's progress from below. Wait for the whole grid's
	 * operations before leaving, so the host does not tear the endpoint
	 * down with any of them still in flight.
	 */
	if (ft_efa_xpu_issuers(scope) > 1) {
		if (xpu_send_cntr &&
		    ft_efa_xpu_cntr_wait(xpu_send_cntr, send_cntr_base + iters,
					 st->err_base[FT_EFA_XPU_SEND], scope))
			ft_efa_xpu_fail(st);
		if (xpu_recv_cntr &&
		    ft_efa_xpu_cntr_wait(xpu_recv_cntr, recv_cntr_base + iters,
					 st->err_base[FT_EFA_XPU_RECV], scope))
			ft_efa_xpu_fail(st);
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
			int scope, int threads, int blocks, int fill,
			cudaStream_t stream)
{
	struct ft_efa_xpu_status *st;

	/* The run's one poller, unless the issuers read (--readers). */
	int pollers = ft_efa_xpu_readers ? 0 : 1;

	if (ft_efa_xpu_begin(__func__, xpu_send_cntr, xpu_recv_cntr, pollers, &st,
			     stream))
		return -1;

	/* One more block, the run's poller (ft_efa_xpu_poller()). */
	ft_efa_xpu_lat_send_kernel<<<blocks + pollers, threads, 0, stream>>>(
		xpu_ep, xpu_send_cq, xpu_recv_cq,
		xpu_send_cntr, xpu_recv_cntr,
		dest_addr, dest_addr_size,
		recv_buf, recv_len, recv_desc, recv_desc_size,
		send_buf, send_len, send_desc, send_desc_size,
		iters, rx_depth, is_client, scope, fill, st);

	return ft_efa_xpu_end(__func__, st, stream);
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
	int iters, int tx_depth, int scope, int fill,
	struct ft_efa_xpu_status *st)
{
	int my_ops, my_depth;
	/* Operations this issuer has posted and completed. */
	int scnt = 0;
	int ccnt = 0;
	int ret;
	uint64_t send_cntr_base = st->cntr_base[FT_EFA_XPU_SEND];
	/*
	 * Every issuer posts from arguments of its own: its own chunk of the
	 * buffer, its own copy of the peer's address and, for the RMA ops, its
	 * own chunk of the peer's buffer - identical across a group, as in the
	 * latency kernel.
	 */
	int me = ft_efa_xpu_issuer(scope);
	/* This issuer's endpoint - see ft_efa_xpu_eps. */
	struct fid_xpu_ep *my_ep = xpu_ep + me % ft_efa_xpu_eps;
	void *my_buf = (char *) send_buf + (size_t) me * send_len;
	void *my_addr = (char *) dest_addr + (size_t) me * dest_addr_size;
	uint64_t my_remote = remote_addr + (uint64_t) me * send_len;
	uint32_t anon_taken = 0;

	my_ops = ft_efa_xpu_issuer_share(iters, scope);
	my_depth = ft_efa_xpu_issuer_share(tx_depth, scope);

	if (ft_efa_xpu_is_poller_block()) {
		struct fid_xpu_cq *cq[2] = { xpu_send_cq, NULL };
		const uint32_t want[2] = { (uint32_t) iters, 0 };

		if (threadIdx.x == 0)
			ft_efa_xpu_poller(cq, want, scope, my_ops, false, st);
		return;
	}

	if (fill)
		ft_efa_xpu_fill(my_buf, send_len, me, scope);

	while (scnt < my_ops || ccnt < my_ops) {
		/* Post operations up to this issuer's share of tx_depth */
		while (scnt < my_ops && (scnt - ccnt) < my_depth) {
			ft_efa_xpu_converge(scope);
			switch (opcode) {
			case 1: /* write */
				ret = fi_xpu_write(my_ep, my_buf, send_len,
						   send_desc, 0, my_addr,
						   my_remote, remote_key,
						   FT_EFA_XPU_CTX(me, scnt), 0,
						   scope);
				break;
			case 2: /* writedata (write with imm) */
				ret = fi_xpu_write(my_ep, my_buf, send_len,
						   send_desc, 0x12345678,
						   my_addr, my_remote,
						   remote_key,
						   FT_EFA_XPU_CTX(me, scnt),
						   FI_REMOTE_CQ_DATA, scope);
				break;
			case 3: /* read */
				ret = fi_xpu_read(my_ep, my_buf, send_len,
						  send_desc, my_addr,
						  my_remote, remote_key,
						  FT_EFA_XPU_CTX(me, scnt), 0,
						  scope);
				break;
			default: /* send */
				ret = fi_xpu_send(my_ep, my_buf, send_len,
						  send_desc, 0, my_addr,
						  FT_EFA_XPU_CTX(me, scnt), 0,
						  scope);
				break;
			}
			if (ret) {
				printf("ft_efa_xpu_bw: post failed: %d at scnt=%d\n",
				       ret, scnt);
				ft_efa_xpu_fail(st);
				return;
			}
			scnt++;
		}

		/*
		 * Wait for completions, one at a time and only those this
		 * issuer is owed. The poller reads the queue for every issuer
		 * and credits each completion to the issuer its context names.
		 */
		while (ccnt < scnt && (scnt == my_ops ||
		       (scnt - ccnt) >= my_depth)) {
			if (xpu_send_cntr) {
				ret = ft_efa_xpu_cntr_wait(
					xpu_send_cntr,
					send_cntr_base + ccnt + 1,
					st->err_base[FT_EFA_XPU_SEND], scope);
				if (ret) {
					printf("ft_efa_xpu_bw: cntr error at ccnt=%d\n",
					       ccnt);
					ft_efa_xpu_fail(st);
					return;
				}
			} else if (ft_efa_xpu_await(FT_EFA_XPU_SEND, me,
						    (uint32_t) ccnt + 1, scope,
						    &anon_taken, xpu_send_cq,
						    my_ops, false, st)) {
				return;
			}
			ccnt++;
		}
	}

	/*
	 * A counter counts every issuer's completions, so a per-issuer threshold
	 * only bounds the grid's progress from below. Wait for the whole grid's
	 * operations before leaving, so the host neither reports a bandwidth for
	 * work still in flight nor tears the endpoint down under it. The
	 * completion queue needs no such wait: the issuers consume one entry per
	 * operation between them, so every operation has completed once they
	 * have all finished counting.
	 */
	if (ft_efa_xpu_issuers(scope) > 1 && xpu_send_cntr &&
	    ft_efa_xpu_cntr_wait(xpu_send_cntr, send_cntr_base + iters,
				 st->err_base[FT_EFA_XPU_SEND], scope))
		ft_efa_xpu_fail(st);
}

int ft_efa_xpu_run_bw(struct fid_xpu_ep *xpu_ep,
		  struct fid_xpu_cq *xpu_send_cq,
		  struct fid_xpu_cntr *xpu_send_cntr,
		  int opcode,
		  void *send_buf, size_t send_len,
		  void *send_desc, size_t send_desc_size,
		  void *dest_addr, size_t dest_addr_size,
		  uint64_t remote_addr, uint64_t remote_key,
		  int iters, int tx_depth, int scope, int threads, int blocks,
		  int fill, cudaStream_t stream)
{
	struct ft_efa_xpu_status *st;

	/* The run's one poller, unless the issuers read (--readers). */
	int pollers = ft_efa_xpu_readers ? 0 : 1;

	if (ft_efa_xpu_begin(__func__, xpu_send_cntr, NULL, pollers, &st, stream))
		return -1;

	ft_efa_xpu_bw_tx_kernel<<<blocks + pollers, threads, 0, stream>>>(
		xpu_ep, xpu_send_cq, xpu_send_cntr, opcode,
		send_buf, send_len, send_desc, send_desc_size,
		dest_addr, dest_addr_size,
		remote_addr, remote_key,
		iters, tx_depth, scope, fill, st);
	ft_efa_xpu_exact_kernel<<<1, 1, 0, stream>>>(xpu_send_cq, xpu_send_cntr,
						  FT_EFA_XPU_SEND, iters, st);

	return ft_efa_xpu_end(__func__, st, stream);
}

__global__ void ft_efa_xpu_bw_rx_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_recv_cq,
	struct fid_xpu_cntr *xpu_recv_cntr,
	void *recv_buf, size_t recv_len,
	void *recv_desc, size_t recv_desc_size,
	int iters, int rx_depth, int scope, bool ctx_null_ok,
	struct ft_efa_xpu_status *st)
{
	int my_ops, my_depth;
	/* Operations this issuer has received. */
	int rcnt = 0;
	int ret;
	uint64_t recv_cntr_base = st->cntr_base[FT_EFA_XPU_RECV];
	int me = ft_efa_xpu_issuer(scope);
	/* This issuer's endpoint - see ft_efa_xpu_eps. */
	struct fid_xpu_ep *my_ep = xpu_ep + me % ft_efa_xpu_eps;
	/* Every issuer receives into its own chunk - see the tx kernel. */
	void *my_buf = (char *) recv_buf + (size_t) me * recv_len;
	uint32_t anon_taken = 0;

	my_ops = ft_efa_xpu_issuer_share(iters, scope);
	my_depth = ft_efa_xpu_issuer_share(rx_depth, scope);

	if (ft_efa_xpu_is_poller_block()) {
		struct fid_xpu_cq *cq[2] = { NULL, xpu_recv_cq };
		const uint32_t want[2] = { 0, (uint32_t) iters };

		if (threadIdx.x == 0)
			ft_efa_xpu_poller(cq, want, scope, my_ops, ctx_null_ok,
					  st);
		return;
	}

	/*
	 * Post initial receives batched: defer the RQ doorbell with FI_MORE
	 * on all but the last post, which rings it for the whole batch.
	 */
	for (int i = 0; i < my_depth; i++) {
		uint64_t rflags = (i + 1 < my_depth) ? FI_MORE : 0;

		ft_efa_xpu_converge(scope);
		ret = fi_xpu_recv(my_ep, my_buf, recv_len, recv_desc,
				  NULL, FT_EFA_XPU_CTX(me, i), rflags, scope);
		if (ret) {
			printf("ft_efa_xpu_bw_recv: post failed: %d\n", ret);
			ft_efa_xpu_fail(st);
			return;
		}
	}

	while (rcnt < my_ops) {
		if (xpu_recv_cntr) {
			ret = ft_efa_xpu_cntr_wait(
				xpu_recv_cntr, recv_cntr_base + rcnt + 1,
				st->err_base[FT_EFA_XPU_RECV], scope);
			if (ret) {
				printf("ft_efa_xpu_bw_recv: cntr error\n");
				ft_efa_xpu_fail(st);
				return;
			}
		} else if (ft_efa_xpu_await(FT_EFA_XPU_RECV, me,
					    (uint32_t) rcnt + 1, scope,
					    &anon_taken, xpu_recv_cq, my_ops,
					    ctx_null_ok, st)) {
			return;
		}

		rcnt++;

		/* Repost receive */
		if (rcnt + my_depth <= my_ops) {
			ft_efa_xpu_converge(scope);
			ret = fi_xpu_recv(my_ep, my_buf, recv_len,
					  recv_desc, NULL,
					  FT_EFA_XPU_CTX(me, rcnt + my_depth - 1),
					  0, scope);
			if (ret) {
				printf("ft_efa_xpu_bw_recv: repost failed: %d\n", ret);
				ft_efa_xpu_fail(st);
				return;
			}
		}
	}

	/* The counter counts every issuer's completions - see the tx kernel. */
	if (ft_efa_xpu_issuers(scope) > 1 && xpu_recv_cntr &&
	    ft_efa_xpu_cntr_wait(xpu_recv_cntr, recv_cntr_base + iters,
				 st->err_base[FT_EFA_XPU_RECV], scope))
		ft_efa_xpu_fail(st);
}

int ft_efa_xpu_run_bw_recv(struct fid_xpu_ep *xpu_ep,
		       struct fid_xpu_cq *xpu_recv_cq,
		       struct fid_xpu_cntr *xpu_recv_cntr,
		       void *recv_buf, size_t recv_len,
		       void *recv_desc, size_t recv_desc_size,
		       int iters, int rx_depth, int scope, int threads,
		       int blocks, int ctx_null_ok, cudaStream_t stream)
{
	struct ft_efa_xpu_status *st;

	/* The run's one poller, unless the issuers read (--readers). */
	int pollers = ft_efa_xpu_readers ? 0 : 1;

	if (ft_efa_xpu_begin(__func__, NULL, xpu_recv_cntr, pollers, &st, stream))
		return -1;

	ft_efa_xpu_bw_rx_kernel<<<blocks + pollers, threads, 0, stream>>>(
		xpu_ep, xpu_recv_cq, xpu_recv_cntr,
		recv_buf, recv_len, recv_desc, recv_desc_size,
		iters, rx_depth, scope, !!ctx_null_ok, st);
	ft_efa_xpu_exact_kernel<<<1, 1, 0, stream>>>(xpu_recv_cq, xpu_recv_cntr,
						  FT_EFA_XPU_RECV, iters, st);

	return ft_efa_xpu_end(__func__, st, stream);
}

/*
 * Post @iters operations - or receives, for FT_EFA_XPU_OP_RECV - as fast as the
 * queue takes them, reading no completions at all. The host drains the CQ
 * (see --host-cq in efa_xpu.c), so how far ahead of it the kernel may run is
 * bounded by nothing but the provider: a run many times the queue's depth is
 * what shows a post waits for room rather than overrunning the queue.
 */
__global__ void ft_efa_xpu_post_kernel(
	struct fid_xpu_ep *xpu_ep, int opcode,
	void *buf, size_t len, void *desc,
	void *dest_addr, size_t dest_addr_size,
	uint64_t remote_addr, uint64_t remote_key,
	int iters, int scope, struct ft_efa_xpu_status *st)
{
	int me = ft_efa_xpu_issuer(scope);
	/* This issuer's endpoint - see ft_efa_xpu_eps. */
	struct fid_xpu_ep *my_ep = xpu_ep + me % ft_efa_xpu_eps;
	int my_ops = ft_efa_xpu_issuer_share(iters, scope);
	void *my_buf = (char *) buf + (size_t) me * len;
	void *my_addr = (char *) dest_addr + (size_t) me * dest_addr_size;
	uint64_t my_remote = remote_addr + (uint64_t) me * len;
	int ret;

	for (int i = 0; i < my_ops; i++) {
		ft_efa_xpu_converge(scope);
		switch (opcode) {
		case 1: /* write */
			ret = fi_xpu_write(my_ep, my_buf, len, desc, 0,
					   my_addr, my_remote, remote_key,
					   FT_EFA_XPU_CTX(me, i), 0, scope);
			break;
		case 2: /* writedata */
			ret = fi_xpu_write(my_ep, my_buf, len, desc,
					   0x12345678, my_addr, my_remote,
					   remote_key, FT_EFA_XPU_CTX(me, i),
					   FI_REMOTE_CQ_DATA, scope);
			break;
		case 3: /* read */
			ret = fi_xpu_read(my_ep, my_buf, len, desc, my_addr,
					  my_remote, remote_key,
					  FT_EFA_XPU_CTX(me, i), 0, scope);
			break;
		case FT_EFA_XPU_OP_RECV:
			ret = fi_xpu_recv(my_ep, my_buf, len, desc, NULL,
					  FT_EFA_XPU_CTX(me, i), 0, scope);
			break;
		default: /* send */
			ret = fi_xpu_send(my_ep, my_buf, len, desc, 0,
					  my_addr, FT_EFA_XPU_CTX(me, i), 0,
					  scope);
			break;
		}
		if (ret) {
			printf("ft_efa_xpu_post: post %d failed: %d\n", i, ret);
			ft_efa_xpu_fail(st);
			return;
		}
	}
}

int ft_efa_xpu_post_launch(struct fid_xpu_ep *xpu_ep, int opcode,
			   void *buf, size_t len, void *desc,
			   void *dest_addr, size_t dest_addr_size,
			   uint64_t remote_addr, uint64_t remote_key,
			   int iters, int scope, int threads, int blocks,
			   cudaStream_t stream, void **run)
{
	struct ft_efa_xpu_status *st;

	if (ft_efa_xpu_begin(__func__, NULL, NULL, 0, &st, stream))
		return -1;

	ft_efa_xpu_post_kernel<<<blocks, threads, 0, stream>>>(
		xpu_ep, opcode, buf, len, desc, dest_addr, dest_addr_size,
		remote_addr, remote_key, iters, scope, st);
	if (cudaGetLastError() != cudaSuccess) {
		cudaFree(st);
		return -1;
	}

	*run = st;
	return 0;
}

int ft_efa_xpu_post_finish(void *run, cudaStream_t stream)
{
	return ft_efa_xpu_end(__func__, (struct ft_efa_xpu_status *) run,
			      stream);
}

/*
 * A scope the build has never heard of is refused before anything is posted, so an application gets an error rather
 * than a queue it has half committed to. Each entry point has to refuse on its
 * own, so each one is checked.
 */
__global__ void ft_efa_xpu_scope_reject_kernel(
	struct fid_xpu_ep *xpu_ep,
	struct fid_xpu_cq *xpu_cq,
	void *buf, size_t len, void *desc,
	void *dest_addr, int scope, struct ft_efa_xpu_status *st)
{
	int cq_expect = -FI_EOPNOTSUPP;
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
	if (cq_ret != cq_expect) {
		printf("fi_xpu_cq_read at scope %d: %lld, expected %d\n", scope,
		       (long long) cq_ret, cq_expect);
		fails++;
	}

	st->failures = fails;
}

int ft_efa_xpu_run_scope_reject(struct fid_xpu_ep *xpu_ep,
			    struct fid_xpu_cq *xpu_cq,
			    void *buf, size_t len, void *desc,
			    void *dest_addr, int scope,
			    cudaStream_t stream)
{
	struct ft_efa_xpu_status *st;

	if (ft_efa_xpu_begin(__func__, NULL, NULL, 0, &st, stream))
		return -1;

	/* One thread: an unsupported scope is refused without any cooperation. */
	ft_efa_xpu_scope_reject_kernel<<<1, 1, 0, stream>>>(
		xpu_ep, xpu_cq, buf, len, desc, dest_addr, scope, st);

	return ft_efa_xpu_end(__func__, st, stream);
}

/*
 * The threads converged at a call that name the same object are one group for
 * it: the lanes of a warp that post at the work item scope are batched per
 * endpoint. A host build has
 * one lane, so this is where the partition across lanes is checked: each lane
 * names one of three objects by its lane number, first with the whole warp
 * converged and then with only its even lanes, and checks the group it lands
 * in - the lanes naming its object, ranked densely, led by the lowest of them.
 * The objects are never dereferenced: forming a group reads only their
 * addresses.
 */
__device__ static int ft_efa_xpu_check_group(const struct efa_xpu_group *g,
					     unsigned int active, int pick)
{
	unsigned int lane = threadIdx.x % warpSize;
	unsigned int mine = 0;
	int rank = 0;

	for (unsigned int l = 0; l < 32; l++) {
		if (!(active & (1u << l)) || (int) (l % 3) != pick)
			continue;
		if (l < lane)
			rank++;
		mine |= 1u << l;
	}
	if ((unsigned int) g->mask != mine || g->size != __popc(mine) ||
	    g->rank != rank || g->leader != (rank == 0) ||
	    g->leader_lane != __ffs(mine) - 1) {
		printf("ft_efa_xpu_partition: lane %u pick %d: mask %#x size %d "
		       "rank %d, expected %#x %d %d\n", lane, pick,
		       (unsigned int) g->mask, g->size, g->rank, mine,
		       __popc(mine), rank);
		return 1;
	}
	return 0;
}

__global__ void ft_efa_xpu_partition_kernel(struct ft_efa_xpu_status *st)
{
	__shared__ uint64_t objs[3];
	unsigned int lane = threadIdx.x % warpSize;
	int pick = (int) (lane % 3);
	void *obj = &objs[pick];
	struct efa_xpu_group g;
	int fails = 0;

	/* The whole warp: three batches, three CQ groups. */
	(void) efa_xpu_group_coalesce(&g, (struct efa_xpu_ep *) obj);
	fails += ft_efa_xpu_check_group(&g, 0xffffffffu, pick);
	__syncwarp();

	/* Only the even lanes: the partition is of the lanes that are here. */
	if (!(lane & 1)) {
		(void) efa_xpu_group_coalesce(&g, (struct efa_xpu_ep *) obj);
		fails += ft_efa_xpu_check_group(&g, 0x55555555u, pick);
	}
	__syncwarp();

	if (fails)
		atomicAdd(&st->failures, fails);
}

int ft_efa_xpu_run_partition(cudaStream_t stream)
{
	struct ft_efa_xpu_status *st;

	if (ft_efa_xpu_begin(__func__, NULL, NULL, 0, &st, stream))
		return -1;

	ft_efa_xpu_partition_kernel<<<1, 32, 0, stream>>>(st);
	return ft_efa_xpu_end(__func__, st, stream);
}
