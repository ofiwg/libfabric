/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018,2020 Cray Inc. All rights reserved.
 */

#ifndef _CXIP_TEST_COMMON_PMI_H_
#define _CXIP_TEST_COMMON_PMI_H_

#include "cxip.h"

extern int cxit_ranks;
extern int cxit_rank;
extern uint32_t cxit_mcast_ref;
extern uint32_t cxit_mcast_id;

void cxit_setup_distributed(void);
void cxit_teardown_distributed(void);
void cxit_setup_multicast(void);
void cxit_teardown_multicast(void);

void cxit_LTU_barrier(void);
void cxit_LTU_create_universe(void);
void cxit_LTU_destroy_universe(void);
void cxit_LTU_create_coll_mcast(int hwrootidx, int timeout, uint32_t *mcast_ref, uint32_t *mcast_id);
void cxit_LTU_destroy_coll_mcast(uint32_t mcast_ref);
double _print_delay(struct timespec *ts0, struct timespec *ts1, const char *func, const char *tag);

#endif
