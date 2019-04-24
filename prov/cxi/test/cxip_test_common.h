/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2018 Cray Inc. All rights reserved.
 */

#ifndef _CXIP_TEST_COMMON_H_
#define _CXIP_TEST_COMMON_H_

#include "cxip.h"

#define CXIT_DEFAULT_TIMEOUT 10

extern struct fi_info *cxit_fi_hints;
extern struct fi_info *cxit_fi;
extern struct fid_fabric *cxit_fabric;
extern struct fid_domain *cxit_domain;
extern struct fid_ep *cxit_ep;
extern struct cxip_addr cxit_ep_addr;
extern fi_addr_t cxit_ep_fi_addr;
extern struct fid_ep *cxit_sep;
extern struct fi_cq_attr cxit_tx_cq_attr, cxit_rx_cq_attr;
extern struct fid_cq *cxit_tx_cq, *cxit_rx_cq;
extern struct fi_av_attr cxit_av_attr;
extern struct fid_av *cxit_av;
extern char *cxit_node, *cxit_service;
extern uint64_t cxit_flags;
extern int cxit_n_ifs;

void cxit_dump_tx_attr(struct fi_tx_attr *tx_attr);
void cxit_dump_rx_attr(struct fi_rx_attr *rx_attr);
void cxit_dump_ep_attr(struct fi_ep_attr *ep_attr);
void cxit_dump_domain_attr(struct fi_domain_attr *dom_attr);
void cxit_dump_fabric_attr(struct fi_fabric_attr *fab_attr);
void cxit_dump_attr(struct fi_info *info);

void cxit_create_fabric_info(void);
void cxit_destroy_fabric_info(void);
void cxit_create_fabric(void);
void cxit_destroy_fabric(void);
void cxit_create_domain(void);
void cxit_destroy_domain(void);
void cxit_create_ep(void);
void cxit_destroy_ep(void);
void cxit_create_sep(void);
void cxit_destroy_sep(void);
void cxit_create_cqs(void);
void cxit_destroy_cqs(void);
void cxit_create_av(void);
void cxit_destroy_av(void);
void cxit_bind_av(void);
void cxit_bind_cqs(void);

void cxit_setup_getinfo(void);
void cxit_teardown_getinfo(void);
void cxit_setup_fabric(void);
void cxit_teardown_fabric(void);
void cxit_setup_domain(void);
void cxit_teardown_domain(void);
void cxit_setup_ep(void);
void cxit_teardown_ep(void);
#define cxit_setup_cq cxit_setup_ep
#define cxit_teardown_cq cxit_teardown_ep
#define cxit_setup_av cxit_setup_ep
#define cxit_teardown_av cxit_teardown_ep
void cxit_setup_enabled_ep(void);
void cxit_setup_rma(void);
#define cxit_setup_tagged cxit_setup_rma
#define cxit_setup_msg cxit_setup_rma
void cxit_setup_tagged_offload(void);
void cxit_teardown_rma(void);
#define cxit_teardown_tagged cxit_teardown_rma
#define cxit_teardown_msg cxit_teardown_rma
void cxit_teardown_tagged_offload(void);
int cxit_await_completion(struct fid_cq *cq, struct fi_cq_tagged_entry *cqe);

#endif
