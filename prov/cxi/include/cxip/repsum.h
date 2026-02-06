/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_REPSUM_H_
#define _CXIP_REPSUM_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Type definitions */
union cxip_dbl_bits {
	struct {
		uint64_t mantissa : 52;
		uint64_t exponent : 11;
		uint64_t sign : 1;
	} __attribute__((__packed__));
	double dval;
	uint64_t ival;
};

struct cxip_repsum {
	int64_t T[4];
	int32_t M;
	int8_t overflow_id;
	bool inexact;
	bool overflow;
	bool invalid;
};

/* Function declarations */
void cxip_dbl_to_rep(struct cxip_repsum *x, double d);

void cxip_rep_to_dbl(double *d, const struct cxip_repsum *x);

void cxip_rep_add(struct cxip_repsum *x, const struct cxip_repsum *y);

double cxip_rep_add_dbl(double d1, double d2);

double cxip_rep_sum(size_t count, double *values);

#endif /* _CXIP_REPSUM_H_ */
