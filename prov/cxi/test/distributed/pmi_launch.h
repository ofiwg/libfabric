/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Header for pmi_launch.c
 *
 * Include this in any application to be launched by pmi_launch.
 *
 * Functions are defined in the include so the application does not link to
 * pmi_utils, which can cause problems in some circumstances.
 *
 * Do not use _pmi_variables directly, since they may be uninitialized. Instead,
 * call the access functions, which will guarantee initialization.
 *
 * See pmi_launch_test.c as example code.
 *
 * NOTE: Do not include in production code at this time, pending architectural
 * discussion.
 *
 * (c) Copyright 2022 Hewlett Packard Enterprise Development LP
 */

#ifndef _PMI_LAUNCH_H_
#define _PMI_LAUNCH_H_

#include <stdint.h>

#define	PMI_NIC_ADDRS_NAME	"PMI_NIC_ADDRS"
#define	PMI_NUM_RANKS_NAME	"PMI_NUM_RANKS"
#define PMI_NUM_HSNS_NAME	"PMI_NUM_HSNS"
#define	PMI_RANK_NAME		"PMI_RANK"

/* Packed 64-bit structure */
union nicaddr {
	uint64_t value;
	struct {
		uint64_t nic:48;
		uint64_t hsn:2;
		uint64_t rank:14;
	} __attribute__((__packed__));
};
static int _pmi_init = 0;
static int _pmi_num_ranks = 0;
static int _pmi_num_hsns = 0;
static int _pmi_rank = 0;
static int _pmi_nic_count = 0;
static union nicaddr *_pmi_nic_addrs = NULL;

static inline int pmi_init(void)
{
	if (!_pmi_init) {
		char *numranks = getenv(PMI_NUM_RANKS_NAME);
		char *numhsns = getenv(PMI_NUM_HSNS_NAME);
		char *rank = getenv(PMI_RANK_NAME);
		char *addrs = getenv(PMI_NIC_ADDRS_NAME);
		char *c;
		int i;

		if (! numranks || !numhsns || !rank || !addrs)
			return -1;
		_pmi_num_ranks = atoi(numranks);
		_pmi_num_hsns = atoi(numhsns);
		_pmi_rank = atoi(rank);

		/* already tested for addrs==NULL, comma-delimited hex values */
		_pmi_nic_count = 1;
		for (c = addrs; *c; c++)
			if (*c == ',')
				_pmi_nic_count++;
		_pmi_nic_addrs = calloc(_pmi_nic_count, sizeof(union nicaddr));
		if (! _pmi_nic_addrs)
			return -1;
		c = addrs;
		for (i = 0; i < _pmi_nic_count; i++) {
			_pmi_nic_addrs[i].value = strtol(c, &c, 16);
			c++;
		}
		_pmi_init = 1;
	}
	return 0;
}

static inline int pmi_num_ranks(void)
{
	pmi_init();
	return _pmi_num_ranks;
}

static inline int pmi_num_hsns(void)
{
	pmi_init();
	return _pmi_num_hsns;
}

static inline int pmi_rank(void)
{
	pmi_init();
	return _pmi_rank;
}

static inline int pmi_nic_addr_count(void)
{
	pmi_init();
	return _pmi_nic_count;
}

/* return the address associated with rank and hsn */
static inline uint64_t pmi_nic_addr(int rank, int hsn)
{
	int i, count;

	pmi_init();
	count = pmi_nic_addr_count();
	for (i = 0; i < count; i++)
		if (_pmi_nic_addrs[i].rank == rank &&
		    _pmi_nic_addrs[i].hsn == hsn)
			return _pmi_nic_addrs[i].nic;
	return (uint64_t)-1;
}

#endif
