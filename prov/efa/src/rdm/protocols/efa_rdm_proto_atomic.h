/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#ifndef _EFA_RDM_PROTO_ATOMIC_H
#define _EFA_RDM_PROTO_ATOMIC_H

#include "efa_rdm_pke.h"

struct efa_rdm_ope;

/**
 * @brief Atomic TX protocol interface.
 *
 * Atomic requests use fi_msg_atomic and carry RMA IOVs in their request
 * headers. Each protocol constructs exactly 1 complete request PKE and installs
 * the callback that owns its send-completion cleanup.
 *
 * Keeping this interface separate from the message protocol avoids coupling
 * the 2 paths. A TXE uses either proto or atomic_proto, never both.
 */
struct efa_rdm_atomic_proto {
	int req_pkt_type;
	int req_pkt_type_dc;
	int (*get_req_data_size)(struct efa_rdm_ope *txe, size_t *data_size);
	ssize_t (*init_req_pke)(struct efa_rdm_pke *pkt_entry,
				struct efa_rdm_ope *txe);
	efa_rdm_pke_callback handle_send_completion;
};

extern const struct efa_rdm_atomic_proto efa_rdm_atomic_write_proto;
extern const struct efa_rdm_atomic_proto efa_rdm_atomic_fetch_proto;
extern const struct efa_rdm_atomic_proto efa_rdm_atomic_compare_proto;

const struct efa_rdm_atomic_proto *efa_rdm_proto_atomic_select(uint32_t op);

ssize_t efa_rdm_proto_atomic_post(struct efa_rdm_ope *txe);

#endif /* _EFA_RDM_PROTO_ATOMIC_H */
