/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Acquire CXI HSN addresses, set environment variables, and launch an
 * application in a multi-node environment.
 *
 * This duplicates some SLURM/PBS information, but has a consistent API
 * regardless of WLM, and provides additional information, specifically the CXI
 * HSN addresses for each node (PMI rank) in the new environment variable
 * PMI_NIC_ADDRS. It also completely decouples the PMI functionality (which has
 * some ill-defined linking issues) from the test application.
 *
 * Caller supplies the number of NICs per node an environment variable
 * PMI_NUM_HSNS. This can be any value above zero. If unspecified, a value of 1
 * is assumed. Hardware typically limits the value to 4 on any system, but this
 * is not a hard limit.
 *
 * Each node will generate PMI_NUM_HSNS addresses as concatenated strings of the
 * form "XX:XX:XX:XX:XX:XX\n". If a NIC is missing for a node, the value will be
 * reported as the illegal address "FF:FF:FF:FF:FF:FF\n".
 *
 * For efficiency and clarity in C, this text string is interpreted into a
 * 48-bit value, and has 2 bits of hsn network (up to four networks), and 14
 * bits of rank information, resulting in a 64-bit value.
 *
 * Each 64-bit value is distributed through pmi_Allreduce to generate a complete
 * list of addresses across all endpoints in the reduction.
 *
 * NOTE: Do not include in production code at this time, pending architectural
 * discussion.
 *
 * (c) Copyright 2022 Hewlett Packard Enterprise Development LP
 */

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <malloc.h>
#include <sys/time.h>
#include <sys/types.h>
#include <dirent.h>
#include "pmi_utils.h"
#include "pmi_launch.h"

#define	NICSIZE	(sizeof(union nicaddr))

/* Display a nic */
__attribute__((__unused__))
static void show_nic(int rank, union nicaddr nic)
{
	printf("[%d] rank=%2d hsn=%d addr=%12lx\n", rank,
		nic.rank, nic.hsn, (uint64_t)nic.nic);
}

/* Set environment variable to string representation of int */
static void setenv_int(const char *name, int value)
{
	char buf[32];
	sprintf(buf, "%d", value);
	setenv(name, buf, 1);
}

/* Read /sys files to get the HSN nic addresses */
static void get_nic(int rank, int hsn, union nicaddr *nic)
{
	char fname[256];
	char text[256];
	char *ptr;
	FILE *fid;
	int i, n;

	/* default */
	strcpy(text, "FF:FF:FF:FF:FF:FF\n");
	/* read from file, if possible */
	snprintf(fname, sizeof(fname), "/sys/class/net/hsn%d/address", hsn);
	if ((fid = fopen(fname, "r"))) {
		n = fread(text, 1, sizeof(text), fid);
		fclose(fid);
		text[n] = 0;
	}
	/* parse "XX:XX:XX:XX:XX:XX\n" into 48-bit integer value */
	nic->value = 0L;
	ptr = text;
	for (i = 0; i < 6; i++) {
		nic->value <<= 8;
		nic->value |= strtol(ptr, &ptr, 16);
		ptr++;
	}
	nic->hsn = hsn;
	nic->rank = rank;
}

int main(int argc, char **argv)
{
	int num_hsns;
	int num_ranks;
	int rank;
	int hsn;
	union nicaddr *nics = NULL;
	union nicaddr *allrecv = NULL;
	union nicaddr *allnics = NULL;
	char *str = NULL;
	char *s;
	int blk, i;
	int err;

	/* until cleared, will report failure */
	err = -1;

	/* If no application specified, show usage */
	if (argc < 2) {
		fprintf(stderr, "Usage: %s appname [args...]\n", argv[0]);
		goto done;
	}
	/* argv[1] becomes the new application name on success */
	argv++;

	/* Caller can specify this value, defaults to 1 */
	str = getenv(PMI_NUM_HSNS_NAME);
	if (!str || (sscanf(str, "%d", &num_hsns) < 1))
		num_hsns = 1;

	/* This should accomodate at least 256 ranks in PMI */
	setenv_int("PMI_MAX_KVS_ENTRIES", 1024);

	/* Initialize PMI and acquire num_ranks and rank */
	pmi_Init();
	num_ranks = pmi_GetNumRanks();
	rank = pmi_GetRank();

	/* Allocate spaces */
	nics = calloc(num_hsns, NICSIZE);
	allnics = calloc(num_ranks*num_hsns, NICSIZE);
	allrecv = calloc(num_ranks, NICSIZE);
	str = calloc(1, num_ranks*num_hsns*17);
	if (!nics || !allnics || !allrecv || !str) {
		fprintf(stderr, "Out of memory\n");
		goto done;
	}

	/* Acquire all of our local nics from local /sys fs */
	for (hsn = 0; hsn < num_hsns; hsn++)
		get_nic(rank, hsn, &nics[hsn]);
	pmi_Barrier();

	/* Perform the allgather over one HSN at a time */
	blk = 0;
	for (hsn = 0; hsn < num_hsns; hsn++) {
		/* contibute one nic from this node, receive num_ranks nics */
		pmi_Allgather(&nics[hsn], NICSIZE, allrecv);
		/* copy num_ranks nics into allnics at offset blk */
		memcpy((void *)&allnics[blk], (void *)allrecv,
			num_ranks*NICSIZE);
		/* advance by num_ranks */
		blk += num_ranks;
		pmi_Barrier();
	}

	/* Create an environment variable with the list */
	for (i = 0, s = str; i < num_ranks*num_hsns; i++)
		s += sprintf(s, "%016lx,", allnics[i].value);
	*--s = 0;

	/* Set the env vars we use to pass info to the application */
	setenv(PMI_NIC_ADDRS_NAME, str, 1);
	setenv_int(PMI_NUM_HSNS_NAME, num_hsns);
	setenv_int(PMI_NUM_RANKS_NAME, num_ranks);
	setenv_int(PMI_RANK_NAME, rank);

	/* successful set-up */
	err = 0;

done:
	pmi_Finalize();
	free(allrecv);
	free(allnics);
	free(nics);
	free(str);

	if (err)
		return err;

	/* Launch the new application */
	return execvp(*argv, argv);
}
