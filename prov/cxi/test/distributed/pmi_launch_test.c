/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Test application to validate pmi_launch, and to serve as example code.
 *
 * Initiate this as 'srun -N{numranks} ./pmi_launch ./pmi_launch_test'
 *
 * NOTE: Do not include in production code at this time, pending architectural
 * discussion.
 *
 * (c) Copyright 2022 Hewlett Packard Enterprise Development LP
 */

#include <stdio.h>
#include <stdlib.h>
#include "pmi_launch.h"

int main(int argc, char **argv)
{
	int num_ranks = pmi_num_ranks();
	int num_hsns = pmi_num_hsns();
	int myrank = pmi_rank();
	int hsn, rank;
	uint64_t addr;

	if (num_ranks < 0) {
		fprintf(stderr,
		        "Application must be launched with pmi_launch\n");
		return -1;
	}
	printf("RANK %d=======\n", myrank);
	for (hsn = 0; hsn < num_hsns; hsn++)
		for (rank = 0; rank < num_ranks; rank++) {
			addr = pmi_nic_addr(rank, hsn);
			printf("hsn=%d rank=%d addr=%012lx\n", hsn, rank, addr);
		}

	return 0;
}
