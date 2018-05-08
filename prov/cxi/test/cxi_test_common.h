/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#ifndef _CXI_TEST_COMMON_H_
#define _CXI_TEST_COMMON_H_

extern struct fi_info *cxit_fi_hints;
extern struct fi_info *cxit_fi;
extern struct fid_fabric *cxit_fabric;
extern struct fid_domain *cxit_domain;
extern char *cxit_node, *cxit_service;
extern uint64_t cxit_flags;
extern int cxit_n_ifs;

void cxit_create_fabric_info(void);
void cxit_destroy_fabric_info(void);
void cxit_create_fabric(void);
void cxit_destroy_fabric(void);
void cxit_create_domain(void);
void cxit_destroy_domain(void);
void cxit_setup_getinfo(void);
void cxit_teardown_getinfo(void);
void cxit_setup_fabric(void);
void cxit_teardown_fabric(void);
void cxit_setup_domain(void);
void cxit_teardown_domain(void);

#endif
