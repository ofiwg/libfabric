/* Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#ifndef _EFA_RDM_PROTO_LONGCTS_H
#define _EFA_RDM_PROTO_LONGCTS_H

#include "efa_rdm_proto.h"

extern struct efa_rdm_proto efa_rdm_proto_longcts;

void efa_rdm_proto_longcts_handle_cts_recv(struct efa_rdm_pke *pkt_entry);
void efa_rdm_proto_longcts_handle_ctsdata_recv(struct efa_rdm_pke *pkt_entry);

#endif /* _EFA_RDM_PROTO_LONGCTS_H */
