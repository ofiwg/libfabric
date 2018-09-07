/*
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
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

#include <mpi.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#include <core/user.h>

static int num_ranks = 0;
static int our_rank = 0;

/*
 * We'd like to keep mpi dependencies out of core and libfabric dependencies
 * out of harness, so when core needs to exchange addresses it does so via
 * this callback routine passed into core's main loop.
 */
int address_exchange(void *my_address, void *addresses, int size, int count)
{
	int ret;

	if (count != num_ranks)
		return EXIT_FAILURE;

	ret = MPI_Allgather(my_address, size, MPI_BYTE,
			addresses, size, MPI_BYTE,
			MPI_COMM_WORLD);

	if (ret)
		fprintf(stderr, "address exchange failed, ret=%d\n", ret);

	return ret;
}

/*
 * Core uses this callback to synchronize job to force expected or unexpected.
 */
void barrier()
{
	MPI_Barrier(MPI_COMM_WORLD);
}

static int process_mgmt_init()
{
	int ret, our_ret;

	ret = MPI_Init(NULL, NULL);
	if (ret) {
		our_ret = ret;
		goto err_mpi_init;
	}

	ret = MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
	if (ret) {
		our_ret = EXIT_FAILURE;
		goto err_mpi_comm;
	}

	ret = MPI_Comm_rank(MPI_COMM_WORLD, &our_rank);
	if (ret != MPI_SUCCESS) {
		our_ret = EXIT_FAILURE;
		goto err_mpi_comm;
	}
	return FI_SUCCESS;

err_mpi_comm:
	ret = MPI_Finalize();
	if (ret) {
		our_ret = our_ret ? our_ret : ret;
	}
err_mpi_init:
	return our_ret;
}

int main(const int argc, char * const *argv)
{
	struct job job;
	int ret;

	ret = process_mgmt_init();
	if (ret)
		return ret;

	job = (struct job) {
		.address_exchange = address_exchange,
		.barrier = barrier,
		.ranks = num_ranks,
		.rank = our_rank
	};

	ret = core(argc, argv, &job);
	if (ret) {
		fprintf(stderr, "TEST FAILED\n");
		return EXIT_FAILURE;
	}
	fprintf(stderr, "TEST PASSED\n");
	return FI_SUCCESS;
}
