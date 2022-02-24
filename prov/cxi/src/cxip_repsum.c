/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2022 Cray Inc. All rights reserved.
 */

/* Notes:
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cxip.h"

/**
 * @brief REPRODUCIBLE SUM IMPLEMENATION
 *
 * - Reference:
 *   - https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.pdf
 *     Algorithm 7
 * - Example Code:
 *   - https://github.com/peterahrens/ReproBLAS.git
 *
 * This code supplies the software component of the RSDG Reproducible Sum
 * collective reduction operation.
 *
 * Conceptually, the 52-bit mantissa of a double precision IEEE floating point
 * value, extended to 53-bits to include the "hidden" bit, is placed in a
 * register containing 2048 bits (the full possible range of IEEE double
 * exponents) and shifted so that the MSBit of the mantissa is aligned with the
 * 11-bit exponent.
 *
 * This large register is then divided into numbered "bins" of W bits. Each bin
 * is then expanded by adding (64 - W) zero bits to the most-significant end of
 * each bin, and these 64-bit quantities are copied into an array of Kt 64-bit
 * registers, along with the bin number M in which the MSBit of the value is
 * located.
 *
 * The extra space in each bin allow us to sum without carry from bin-to-bin
 * until the end of the computation. With W=40, there are 24 bits of overflow,
 * allowing at least 2^24 summations to occur before overflow can occur.
 *
 * If overflow does occur, both Rosetta and this software set an overflow flag,
 * and the final result should be treated as invalid.
 *
 * Low order bits can be discarded in the process, and this will set an inexact
 * flag. The result should still be reproducible, and accurate to within
 * round-off error.
 */

#define W	40
#define Kt	4

/* special values of M for non-numbers */
#define	MNInf	125
#define	MInf	126
#define	MNaN	127

/**
 * @Description
 *
 * BIN() converts the exponent 'e' to a W-bit bin number.
 *
 * OFF() provides the offset of exponent 'e' within the W-bit bin.
 *
 * MSK() provides a bitmask for the W LSBits.
 */
#define BIN(e)	(((e) - 1023 + 1024*W)/W - 1024)
#define OFF(e)	((e) - 1023 - W*BIN(e))
#define	MSK(w)	((1ULL << w) - 1)

/**
 * @brief Decompose an IEEE value into its parts.
 *
 * @param d double precision input
 * @param s sign returned as 1 or -1
 * @param e biased exponent
 * @param m mantissa
 */
static inline void _decompose_ieee(double d, int *s, int *e, uint64_t *m)
{
	union {
		struct {
			uint64_t m:52;
			uint64_t e:11;
			uint64_t s:1;
		} __attribute__((__packed__));
		double d;
	} v;

	v.d = d;
	*s = (v.s) ? -1 : 1;
	*e = v.e;
	*m = v.m;
}

/**
 * @brief Convert double to repsum
 *
 * Rosetta expects T[0] to be the LSBits of the value, so we load from Kt-1
 * downward. Because W=40, T[0] will always be zero: 53 bits of mantissa cannot
 * span more than three 40-bit registers, regardless of alignment.
 *
 * @param x returned repsum object
 * @param d double to convert
 */
void cxip_dbl_to_rep(cxip_repsum_t *x, double d)
{
	uint64_t m;	// mantissa
	int s;		// sign
	int e;		// exponent
	int w;		// bin offset of MSbit
	int lsh;	// left-shift amount
	int rem;	// remaining bits to shift
	int siz;	// number of bits to keep
	int i;

	memset(x, 0, sizeof(*x));
	_decompose_ieee(d, &s, &e, &m);
	if (e == 0x7ff) {
		// NaN, +inf, -inf
		x->M = (m) ? MNaN : ((s < 0) ? MNInf : MInf);
		w = 0;
		m = 0;
	} else if (e) {
		// Normal values
		x->M = BIN(e);
		w = OFF(e);
		m |= 1ULL << 52;
	} else {
		// Subnormal values and zero
		e = 1;
		x->M = BIN(e);
		w = OFF(e);
 	}

	/**
	 * Copy the mantissa into the correct locations within T[].
	 *
	 * T[3] should contain the w+1 MSBits of m, aligned to bit 0.
	 * T[2] should contain the next W bits, aligned to bit W-1.
	 * T[1] should contain any remaining bits, aligned to bit W-1.
	 * T[0] will always be zero.
	 */
	rem = 53;	// number of bits to process
	siz = w + 1;	// bits to include in MSRegister
	lsh = 0;	// left-shift to align
	i = Kt;		// start with most significant
	while (rem) {
		x->T[--i] = s*((m >> (rem - siz)) << lsh);
		rem -= siz;	// siz MSBits consumed
		m &= MSK(rem);	// keep only rem LSBits
		siz = (rem < W) ? rem : W;
		lsh = W - siz;	// align to bit W-1
	}
	while (i)
		x->T[--i] = 0;	// clear remaining bins
}

/**
 * @brief Convert repsum back to double.
 *
 * Simply use scalbn() to scale the signed mantissas and add to the accumulator.
 *
 * @param x repsum object
 * @return double returned value
 */
void cxip_rep_to_dbl(double *d, const cxip_repsum_t *x)
{
	int i, m;

	*d = 0.0;
	switch (x->M) {
		case MNaN:
			*d = 0.0/0.0;
			return;
		case MNInf:
			*d = scalbn(-1.0, 1024);
			return;
		case MInf:
			*d = scalbn(1.0, 1024);
			return;
	}
	m = x->M;
	for (i = Kt-1; i >= 0; i--) {
		*d += scalbn(1.0*(int64_t)x->T[i], W*m);
		m--;
	}
}

/**
 * @brief Add two repsum objects, and return the result in x.
 *
 * @param x accumulator
 * @param y added to accumulator
 */
void cxip_rep_add(cxip_repsum_t *x, const cxip_repsum_t *y)
{
	cxip_repsum_t swap;
	int i, j;

	/* overflow propagates */
	if (y->overflow)
		x->overflow = true;
	/* overflow is fatal */
	if (x->overflow)
		return;
	/* swap x and y if y is has largest magnitude.
	 * NaN is largest, followed by +Inf, -Inf, and numbers
	 */
	if (y->M > x->M) {
		memcpy(&swap, x, sizeof(cxip_repsum_t));
		memcpy(x, y, sizeof(cxip_repsum_t));
		y = (const cxip_repsum_t *)&swap;
	}
	/* +Inf > -Inf, and if added, promote to NaN */
	if (x->M == MInf && y->M == MNInf)
		x->M = MNaN;
	/* Handle the not-numbers */
	if (x->M == MNaN || x->M == MInf || x->M == MNInf)
		return;
	/* inexact propagates */
	if (y->inexact)
		x->inexact = true;
	/* advance j until bins are aligned */
	for (j = 0; j < Kt && j + y->M < x->M; j++)
		if (y->T[j])
			x->inexact = true;
	/* Add remaining y to x in each aligned bin, check for overflow */
	for (i = 0; i < Kt && j < Kt; i++, j++) {
		int sgn0, sgn1;
		sgn0 = x->T[i] >> 63;
		x->T[i] += y->T[j];
		sgn1 = x->T[i] >> 63;
		/* sign change in wrong direction */
		if (sgn0 != sgn1 && sgn1 != y->T[j] >> 63)
			x->overflow = true;
	}
}

/**
 * @brief Add two doubles using the repsum method.
 *
 * @param d1 : operand 1
 * @param d2 : operand 2
 * @return double result
 */
double cxip_rep_add_dbl(double d1, double d2)
{
	cxip_repsum_t x, y;

	cxip_dbl_to_rep(&x, d1);
	cxip_dbl_to_rep(&y, d2);
	cxip_rep_add(&x, &y);
	cxip_rep_to_dbl(&d1, &x);

	return d1;
}

/**
 * @brief Sum over a list of values.
 *
 * @param count   : count of values
 * @param values  : array of values to sum
 * @return double result
 */
double cxip_rep_sum(size_t count, double *values)
{
	cxip_repsum_t x, y;
	double d;
	size_t i;

	if (count <= 0)
		return 0.0;
	if (count == 1)
		return values[0];

	cxip_dbl_to_rep(&x, values[0]);
	for (i = 1; i < count; i++) {
		cxip_dbl_to_rep(&y, values[i]);
		cxip_rep_add(&x, &y);
	}
	cxip_rep_to_dbl(&d, &x);
	return d;
}

