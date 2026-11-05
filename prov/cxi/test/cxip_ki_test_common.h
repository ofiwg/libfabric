/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2026 Hewlett Packard Enterprise Development LP
 */

#ifndef CXIP_KI_TEST_COMMON_H
#define CXIP_KI_TEST_COMMON_H

#include "fi_cxi_gpu_ext.h"

struct mem_region;

/* Capabilities for an endpoint given a KI TX command queue buffer. The
 * provider rejects FI_MSG, FI_TAGGED and FI_COLLECTIVE on such an
 * endpoint because they let it emit on the shared TX ring without a host
 * operation.
 */
#define CXIT_KI_EP_CAPS (FI_RMA | FI_ATOMIC | FI_READ | FI_WRITE | \
			 FI_REMOTE_READ | FI_REMOTE_WRITE)

/* Page-aligned, zeroed buffer for command queues and MRs */
void *cxit_ki_page_alloc(size_t len);
void cxit_ki_page_free(void *buf);

/* Populate host-memory TX and triggered queue hints. A NULL buffer
 * leaves that queue unset.
 */
void cxit_ki_init_hints(struct fi_cxi_ki_ep_hints *hints, void *tx_buf,
			size_t tx_len, void *trig_buf, size_t trig_len);

/* Fixture building blocks. cxit_ki_setup_ep() runs getinfo with the KI
 * attributes and creates an un-enabled endpoint; force_prov_key=true
 * forces FI_MR_PROV_KEY, while false preserves the environment-selected
 * key mode. Provider keys are cached by default.
 * cxit_ki_create_enabled_objects() creates and
 * binds the EQ, CQs, optionally the counters, and the AV;
 * cxit_ki_destroy_enabled_objects() destroys whichever of them exist.
 */
void cxit_ki_setup_ep(bool force_prov_key);
void cxit_ki_create_enabled_objects(bool with_cntrs);
void cxit_ki_destroy_enabled_objects(void);

/* Shared KI (GPU kernel interface) fixture for ki_info.c and ki_rma.c: a
 * fully enabled endpoint with counters and AV bound, both TX and
 * triggered command-queue hints applied, cxit_ep_addr populated and the
 * KI ops opened into cxit_ki_ops. The client_key and prov_key variants
 * explicitly select the key mode regardless of CXIP_TEST_PROV_KEY.
 * The global cxit_prov_key still reflects the environment, not the
 * fixture's override; use the domain's mr_mode to check its key mode.
 */
void cxit_setup_ki_full(void);
void cxit_setup_ki_full_client_key(void);
void cxit_setup_ki_full_prov_key(void);
void cxit_teardown_ki_full(void);

extern struct fi_cxi_ki_ops *cxit_ki_ops;

/* Hint buffers passed at setup, so tests can check they are echoed back */
extern void *cxit_ki_full_tx_buf;
extern void *cxit_ki_full_trig_buf;

/* Create an MR and resolve its GPU-side remote description through
 * get_peer_info() and get_gpu_remote_mrs(). peer may be NULL.
 */
void cxit_ki_create_remote_mr(size_t len, uint64_t access, uint64_t *key,
			      struct mem_region *window,
			      struct fi_cxi_ki_peer_info *peer,
			      struct fi_cxi_gpu_remote_mr *remote);

#endif /* CXIP_KI_TEST_COMMON_H */
