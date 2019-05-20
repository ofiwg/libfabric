/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2019 Cray Inc. All rights reserved.
 */

/* Fault injection. */

struct cxip_fault {
	char *env;	/* Configuration env. var. name */
	int prop;	/* Proportion of rand() values */
	size_t count;	/* Count of injected faults */
};

extern struct cxip_fault dma_fault;
extern struct cxip_fault malloc_fault;

void cxip_fault_inject_fini(void);
void cxip_fault_inject_init(void);

#if ENABLE_DEBUG
#define INJECT_FAULT(fault) \
	((fault).prop && rand() < (fault).prop && (fault).count++)
#else
#define INJECT_FAULT(fault) 0
#endif

#define cxi_cq_emit_dma_f(...)			\
	(INJECT_FAULT(dma_fault) ? -ENOSPC :	\
	 cxi_cq_emit_dma(__VA_ARGS__))

#define issue_unlink_le_f(...)			\
	(INJECT_FAULT(dma_fault) ? -FI_EAGAIN :	\
	 issue_unlink_le(__VA_ARGS__))

#define malloc_f(...)				\
	(INJECT_FAULT(malloc_fault) ? NULL :	\
	 malloc(__VA_ARGS__))
