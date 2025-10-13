/*
 * Copyright (c) 2016 Intel Corp, Inc. All rights reserved.
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

#include "shared.h"

/* Provide external definitions for complex functions */
extern inline ofi_complex_float ofi_complex_sum_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline ofi_complex_float ofi_complex_prod_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline int ofi_complex_eq_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline ofi_complex_float ofi_complex_land_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline ofi_complex_float ofi_complex_lor_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline ofi_complex_float ofi_complex_lxor_float(ofi_complex_float v1, ofi_complex_float v2);
extern inline void ofi_complex_set_float(ofi_complex_float *v1, ofi_complex_float v2);
extern inline void ofi_complex_fill_float(ofi_complex_float *v1, float v2);

extern inline ofi_complex_double ofi_complex_sum_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline ofi_complex_double ofi_complex_prod_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline int ofi_complex_eq_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline ofi_complex_double ofi_complex_land_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline ofi_complex_double ofi_complex_lor_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline ofi_complex_double ofi_complex_lxor_double(ofi_complex_double v1, ofi_complex_double v2);
extern inline void ofi_complex_set_double(ofi_complex_double *v1, ofi_complex_double v2);
extern inline void ofi_complex_fill_double(ofi_complex_double *v1, double v2);

extern inline ofi_complex_long_double ofi_complex_sum_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline ofi_complex_long_double ofi_complex_prod_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline int ofi_complex_eq_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline ofi_complex_long_double ofi_complex_land_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline ofi_complex_long_double ofi_complex_lor_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline ofi_complex_long_double ofi_complex_lxor_long_double(ofi_complex_long_double v1, ofi_complex_long_double v2);
extern inline void ofi_complex_set_long_double(ofi_complex_long_double *v1, ofi_complex_long_double v2);
extern inline void ofi_complex_fill_long_double(ofi_complex_long_double *v1, long_double v2);
