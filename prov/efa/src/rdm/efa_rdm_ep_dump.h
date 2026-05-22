/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_RDM_EP_DUMP_H
#define EFA_RDM_EP_DUMP_H

#include <signal.h>
#include "efa_rdm_ep.h"

extern volatile sig_atomic_t g_efa_rdm_dump_requested;

void efa_rdm_dump_init(void);
void efa_rdm_dump_ep_state(struct efa_rdm_ep *ep);

#endif /* EFA_RDM_EP_DUMP_H */
