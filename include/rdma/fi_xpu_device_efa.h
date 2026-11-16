/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. */

#ifndef FI_XPU_DEVICE_EFA_H
#define FI_XPU_DEVICE_EFA_H

/*
 * EFA provider - XPU device-side layouts.
 *
 * What an XPU kernel builds EFA work requests from. The host obtains each
 * with an fi_*() call and copies it to device memory, and the kernel passes
 * it to the device-side data transfer calls; it is never resolved back to an
 * AV entry or an MR on the device. Both are values, not handles, and their
 * sizes are what fi_xpu_ctx_query() reports as av_addr_size and
 * mr_desc_size.
 */

#include <stdint.h>

/*
 * A peer, as fi_av_lookup2() with FI_XPU returns it: the address handle
 * number and the remote QP number and QKEY an EFA send queue descriptor
 * addresses a peer with.
 */
struct efa_xpu_peer {
	uint16_t ahn;
	uint16_t remote_qpn;
	uint32_t remote_qkey;
};

/*
 * A memory registration, as fi_mr_get_xpu_desc() with FI_XPU returns it: the
 * local key an EFA descriptor names the registered buffer with. The reserved
 * word is always zero; it keeps the descriptor 8 bytes, so it can carry more
 * without mr_desc_size changing.
 */
struct efa_xpu_desc {
	uint32_t lkey;
	uint32_t reserved;
};

#endif /* FI_XPU_DEVICE_EFA_H */
