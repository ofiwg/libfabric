/*
 * Copyright (C) 2026 Cornelis Networks.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef _FI_PROV_OPX_FLOW_CONTROL_SPECIALIZATION_H_
#define _FI_PROV_OPX_FLOW_CONTROL_SPECIALIZATION_H_

#define OPX_FLOW_CONTROL_RMA_SPECIALIZATION_CATALOG(APPLY)                                               \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, NONE)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, NONE)          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, NONE) \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, NONE)     \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, NONE)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, NONE)          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, NONE)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, NONE)          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, RCCC)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, RCCC)          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, RCCC) \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, RCCC)     \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, RCCC)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, RCCC)          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, RCCC)      \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, RCCC)

#define OPX_FLOW_CONTROL_ATOMIC_SPECIALIZATION_CATALOG(APPLY)                                                 \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, NONE)  \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)      \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF, NONE) \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)      \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)      \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, RCCC)  \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)       \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)      \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF, RCCC) \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)      \
	APPLY(OPX_LOCK, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)

#define OPX_FLOW_CONTROL_MSG_SPECIALIZATION_CATALOG(APPLY)                                                             \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)

#define OPX_FLOW_CONTROL_TAGGED_SPECIALIZATION_CATALOG(APPLY)                                                          \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, NONE)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, NONE)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      NONE)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, NONE)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0018000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0008000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON, \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_ON,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON,      \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_ON, RCCC)    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_WFR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B,                     \
	      OPX_CTX_SHARING_OFF, RCCC)                                                                               \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_MIXED_9B, OPX_CTX_SHARING_OFF,    \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_JKR, OPX_CTX_SHARING_OFF, RCCC)   \
	APPLY(FI_OPX_LOCK_NOT_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF,     \
	      RCCC)                                                                                                    \
	APPLY(FI_OPX_LOCK_REQUIRED, 0x0010000000000000ull, OPX_RELIABILITY, OPX_HFI1_CYR, OPX_CTX_SHARING_OFF, RCCC)

/*
 * CQ table order is mode -> format -> comms -> physical HFI slot -> context.
 * Slot 4 represents CYR hardware using the MIXED_9B protocol and therefore
 * carries the CYR mask while retaining the MIXED_9B specialization token.
 */
#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, \
							FLOW_CONTROL)                                                \
	APPLY(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, OPX_CTX_SHARING_OFF, FLOW_CONTROL)               \
	APPLY(FORMAT, RELIABILITY, CAPS, MASK, HFI_SLOT, HFI1_TYPE, OPX_CTX_SHARING_ON, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_HFI_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, STANDARD_MASK, CYR_MASK,       \
						    FLOW_CONTROL)                                                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, STANDARD_MASK, 0,          \
							OPX_HFI1_WFR, FLOW_CONTROL)                                  \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, STANDARD_MASK, 1,          \
							OPX_HFI1_MIXED_9B, FLOW_CONTROL)                             \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, STANDARD_MASK, 2,          \
							OPX_HFI1_JKR, FLOW_CONTROL)                                  \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, CYR_MASK, 3, OPX_HFI1_CYR, \
							FLOW_CONTROL)                                                \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, CYR_MASK, 4,               \
							OPX_HFI1_MIXED_9B, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FORMAT, RELIABILITY, STANDARD_MASK, CYR_MASK,          \
						      FLOW_CONTROL)                                                 \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_NONE, STANDARD_MASK,   \
						    CYR_MASK, FLOW_CONTROL)                                         \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_LOCAL, STANDARD_MASK,  \
						    CYR_MASK, FLOW_CONTROL)                                         \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_REMOTE, STANDARD_MASK, \
						    CYR_MASK, FLOW_CONTROL)                                         \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_LOCAL_REMOTE,          \
						    STANDARD_MASK, CYR_MASK, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_ROWS(APPLY, STANDARD_MASK, CYR_MASK, FLOW_CONTROL)              \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FI_CQ_FORMAT_UNSPEC, OFI_RELIABILITY_KIND_ONLOAD,  \
						      STANDARD_MASK, CYR_MASK, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FI_CQ_FORMAT_CONTEXT, OFI_RELIABILITY_KIND_ONLOAD, \
						      STANDARD_MASK, CYR_MASK, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FI_CQ_FORMAT_MSG, OFI_RELIABILITY_KIND_ONLOAD,     \
						      STANDARD_MASK, CYR_MASK, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FI_CQ_FORMAT_DATA, OFI_RELIABILITY_KIND_ONLOAD,    \
						      STANDARD_MASK, CYR_MASK, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_COMMS_ROWS(APPLY, FI_CQ_FORMAT_TAGGED, OFI_RELIABILITY_KIND_ONLOAD,  \
						      STANDARD_MASK, CYR_MASK, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CATALOG(APPLY, STANDARD_MASK, CYR_MASK)         \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_ROWS(APPLY, STANDARD_MASK, CYR_MASK, NONE) \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_ROWS(APPLY, STANDARD_MASK, CYR_MASK, RCCC)

/* Runtime slot 4 aliases the slot-1 MIXED_9B function because both use the runtime mask. */
#define OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_HFI_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, FI_OPX_HDRQ_MASK_RUNTIME, 0, \
							OPX_HFI1_WFR, FLOW_CONTROL)                                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, FI_OPX_HDRQ_MASK_RUNTIME, 1, \
							OPX_HFI1_MIXED_9B, FLOW_CONTROL)                               \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, FI_OPX_HDRQ_MASK_RUNTIME, 2, \
							OPX_HFI1_JKR, FLOW_CONTROL)                                    \
	OPX_FLOW_CONTROL_CQ_SPECIALIZATION_CONTEXT_ROWS(APPLY, FORMAT, RELIABILITY, CAPS, FI_OPX_HDRQ_MASK_RUNTIME, 3, \
							OPX_HFI1_CYR, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FORMAT, RELIABILITY, FLOW_CONTROL)                    \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_NONE, FLOW_CONTROL)   \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_LOCAL, FLOW_CONTROL)  \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_REMOTE, FLOW_CONTROL) \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_HFI_ROWS(APPLY, FORMAT, RELIABILITY, FI_OPX_COMMS_LOCAL_REMOTE, FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_MODE_ROWS(APPLY, FLOW_CONTROL)                                       \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FI_CQ_FORMAT_UNSPEC, OFI_RELIABILITY_KIND_ONLOAD,  \
						      FLOW_CONTROL)                                             \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FI_CQ_FORMAT_CONTEXT, OFI_RELIABILITY_KIND_ONLOAD, \
						      FLOW_CONTROL)                                             \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FI_CQ_FORMAT_MSG, OFI_RELIABILITY_KIND_ONLOAD,     \
						      FLOW_CONTROL)                                             \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FI_CQ_FORMAT_DATA, OFI_RELIABILITY_KIND_ONLOAD,    \
						      FLOW_CONTROL)                                             \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_COMMS_ROWS(APPLY, FI_CQ_FORMAT_TAGGED, OFI_RELIABILITY_KIND_ONLOAD,  \
						      FLOW_CONTROL)

#define OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_SPECIALIZATION_CATALOG(APPLY) \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_MODE_ROWS(APPLY, NONE)        \
	OPX_FLOW_CONTROL_CQ_RUNTIME_UNIQUE_MODE_ROWS(APPLY, RCCC)

#define OPX_FLOW_CONTROL_CQ_SPECIALIZATION_MODE_COUNT 2u

#endif
