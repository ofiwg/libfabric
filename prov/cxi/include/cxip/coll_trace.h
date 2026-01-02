/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_COLL_TRACE_H_
#define _CXIP_COLL_TRACE_H_

/* Forward declarations */
struct cxip_ep_obj;

/* Macros */
#define cxip_coll_trace_attr __attribute__((format(__printf__, 1, 2)))

#define CXIP_COLL_TRACE(mod, fmt, ...)                            \
	do {                                                      \
		if (cxip_coll_prod_trace_true())                  \
			cxip_coll_prod_trace(fmt, ##__VA_ARGS__); \
	} while (0)

/* Function declarations */
int cxip_coll_trace_attr cxip_coll_trace(const char *fmt, ...);

void cxip_coll_trace_flush(void);

void cxip_coll_trace_close(void);

void cxip_coll_trace_init(struct cxip_ep_obj *ep_obj);

#endif /* _CXIP_COLL_TRACE_H_ */
