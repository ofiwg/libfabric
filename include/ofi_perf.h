/*
 * Copyright (c) 2018 Intel Corporation. All rights reserved.
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

#ifndef _OFI_PERF_H_
#define _OFI_PERF_H_

#include "config.h"

#include <string.h>
#include <fi_osd.h>


#ifdef __cplusplus
extern "C" {
#endif


enum ofi_perf_domain {
	OFI_PMU_NONE,
};

struct ofi_perf_data {
	uint64_t	start;
	uint64_t	sum;
	uint64_t	events;
};


/*
 * Performance management unit:
 *
 * Access to a PMU is platform specific.  If an osd.h file provides methods
 * to access a PMU, it should define HAVE_OFI_PERF and provide implementations
 * for the following functions.  Platforms that do not support PMUs will
 * default to no-op definitions.
 */

#ifdef HAVE_OFI_PERF

struct ofi_perf_ctx;

int ofi_pmu_open(struct ofi_perf_ctx **ctx,
		 enum ofi_perf_domain domain, uint32_t cntr_id, uint32_t flags);
uint64_t ofi_pmu_read(struct ofi_perf_ctx *ctx);
void ofi_pmu_close(struct ofi_perf_ctx *ctx);

#else /* HAVE_OFI_PERF */

struct ofi_perf_ctx {
	int dummy;
};

static inline int ofi_pmu_open(struct ofi_perf_ctx **ctx,
			       enum ofi_perf_domain domain, uint32_t cntr_id,
			       uint32_t flags)
{
	*ctx = NULL;
	return 0;
}

static inline uint64_t ofi_pmu_read(struct ofi_perf_ctx *ctx)
{
	return 0;
}

static inline void ofi_pmu_close(struct ofi_perf_ctx *ctx)
{
}
#endif /* HAVE_OFI_PERF */



static inline void ofi_perf_reset(struct ofi_perf_data *data)
{
	memset(data, 0, sizeof *data);
}

static inline void ofi_perf_start(struct ofi_perf_ctx *ctx,
				  struct ofi_perf_data *data)
{
	data->start = ofi_pmu_read(ctx);
}

static inline void ofi_perf_end(struct ofi_perf_ctx *ctx,
				struct ofi_perf_data *data)
{
	data->sum += ofi_pmu_read(ctx) - data->start;
	data->events++;
}


#ifdef __cplusplus
}
#endif

#endif /* _OFI_PERF_H_ */
