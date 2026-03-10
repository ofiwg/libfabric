/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa.h"
#include "efa_rdm_proto.h"

int efa_rdm_proto_select_send_protocol(struct efa_rdm_proto **proto,
				       struct efa_rdm_ep *ep,
				       struct efa_rdm_peer *peer,
				       const struct fi_msg *msg, uint32_t op,
				       uint64_t flags)
{
	return 0;
}
