/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_TELEMETRY_H_
#define _CXIP_TELEMETRY_H_


#include <ofi_list.h>

/* Forward declarations */
struct cxip_domain;
struct cxip_telemetry;

/* Type definitions */
struct cxip_telemetry_entry {
	struct cxip_telemetry *telemetry;
	struct dlist_entry telemetry_entry;

	/* Telemetry name. */
	char name[TELEMETRY_ENTRY_NAME_SIZE];

	/* Telemetry value. */
	unsigned long value;
};

/* Function declarations */
void cxip_telemetry_dump_delta(struct cxip_telemetry *telemetry);

void cxip_telemetry_free(struct cxip_telemetry *telemetry);

int cxip_telemetry_alloc(struct cxip_domain *dom,
			 struct cxip_telemetry **telemetry);

#endif /* _CXIP_TELEMETRY_H_ */
