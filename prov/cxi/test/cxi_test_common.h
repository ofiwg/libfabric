/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#ifndef _CXI_TEST_COMMON_H_
#define _CXI_TEST_COMMON_H_

extern struct fi_info *cxit_fi_hints;
extern struct fid_fabric *cxit_fabric;
extern struct fi_info *cxit_fi;
extern char *cxit_node, *cxit_service;
extern uint64_t cxit_flags;
extern int cxit_n_ifs;

void cxit_create_fabric_info(void);
void cxit_destroy_fabric_info(void);
void cxit_create_fabric(void);
void cxit_destroy_fabric(void);
void cxit_fabric_test_init(void);
void cxit_setup_getinfo(void);
void cxit_teardown_getinfo(void);
void cxit_setup_fabric(void);
void cxit_teardown_fabric(void);

#endif
