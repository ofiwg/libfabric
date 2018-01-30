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


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <inttypes.h>

#include <rdma/fi_errno.h>
#include <ofi_perf.h>
#include <rdma/providers/fi_log.h>


int ofi_perfset_create(const struct fi_provider *prov,
		       struct ofi_perfset *set, size_t size,
		       enum ofi_perf_domain domain, uint32_t cntr_id,
		       uint32_t flags)
{
	int ret;

	ret = ofi_pmu_open(&set->ctx, domain, cntr_id, flags);
	if (ret) {
		FI_WARN(prov, FI_LOG_CORE, "Unable to open PMU %d (%s)\n",
			ret, fi_strerror(ret));
		return ret;
	}

	set->data = calloc(size, sizeof(*set->data) + sizeof(*set->names));
	if (!set->data) {
		ofi_pmu_close(set->ctx);
		return -FI_ENOMEM;
	}

	set->prov = prov;
	set->size = size;
	set->count = 0;
	set->names = (char **)(set->data + size);
	return 0;
}

void ofi_perfset_close(struct ofi_perfset *set)
{
	while (set->count--)
		free(set->names[set->count]);
	ofi_pmu_close(set->ctx);
	free(set->data);
}

struct ofi_perf_data *ofi_perfset_data(struct ofi_perfset *set,
				       const char *name)
{
	if (set->count == set->size)
		return NULL;

	if (name) {
		set->names[set->count] = strdup(name);
		if (!set->names[set->count])
			return NULL;
	}

	return &set->data[set->count++];
}

void ofi_perfset_log(struct ofi_perfset *set)
{
	size_t i;

	for (i = 0; i < set->count; i++) {
		if (!set->data[i].sum)
			continue;

		FI_INFO(set->prov, FI_LOG_CORE, "PERF (%s) "
			"events=%" PRIu64 " avg=%g\n",
			set->names[i] ? set->names[i] : "unknown",
			set->data[i].events,
			(double) set->data[i].sum / set->data[i].events);
	}
}
