/*
 * Copyright (C) 2016 by Argonne National Laboratory.
 * Copyright (C) 2021-2022,2024-2026 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_TAGGED_H_
#define _FI_PROV_OPX_TAGGED_H_

/* Macro indirection in order to support other macros as arguments
 * C requires another indirection for expanding macros since
 * operands of the token pasting operator are not expanded */

#define FI_OPX_TAGGED_RECV_GENERIC(FLOW_CONTROL)  FI_OPX_TAGGED_RECV_GENERIC_(FLOW_CONTROL)
#define FI_OPX_TAGGED_RECV_GENERIC_(FLOW_CONTROL) FI_OPX_TAGGED_RECV_GENERIC_##FLOW_CONTROL
#define FI_OPX_TAGGED_RECV_GENERIC_NONE		  fi_opx_recv_generic
#define FI_OPX_TAGGED_RECV_GENERIC_RCCC		  fi_opx_rccc_trecv_generic

#define FI_OPX_TAGGED_SPECIALIZED_FUNC(LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL) \
	FI_OPX_TAGGED_SPECIALIZED_FUNC_(LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL)

#define FI_OPX_TAGGED_SPECIALIZED_FUNC_(LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL)                 \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_tsend_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(           \
			struct fid_ep *ep, const void *buf, size_t len, void *desc, fi_addr_t dest_addr, uint64_t tag, \
			void *context)                                                                                 \
	{                                                                                                              \
		return fi_opx_ep_tx_send(ep, buf, len, desc, dest_addr, tag, context, 0, LOCK, /* lock_required */     \
					 1,						       /* is_contiguous */     \
					 0,						       /* override_flags */    \
					 0,						       /* flags */             \
					 CAPS | FI_TAGGED, RELIABILITY, OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL),     \
					 HFI1_TYPE, CTX_SHARING);                                                      \
	}                                                                                                              \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_trecv_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(           \
			struct fid_ep *ep, void *buf, size_t len, void *desc, fi_addr_t src_addr, uint64_t tag,        \
			uint64_t ignore, void *context)                                                                \
	{                                                                                                              \
		return FI_OPX_TAGGED_RECV_GENERIC(FLOW_CONTROL)(ep, buf, len, desc, src_addr, tag, ignore, context,    \
								LOCK, FI_TAGGED, RELIABILITY, HFI1_TYPE, CTX_SHARING); \
	}                                                                                                              \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_tinject_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(         \
			struct fid_ep *ep, const void *buf, size_t len, fi_addr_t dest_addr, uint64_t tag)             \
	{                                                                                                              \
		return fi_opx_ep_tx_inject(ep, buf, len, dest_addr, tag, 0, LOCK, /* lock_required */                  \
					   0,					  /* flags */                          \
					   CAPS | FI_TAGGED, RELIABILITY, OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL),   \
					   HFI1_TYPE, CTX_SHARING);                                                    \
	}                                                                                                              \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_tsenddata_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(       \
			struct fid_ep *ep, const void *buf, size_t len, void *desc, uint64_t data,                     \
			fi_addr_t dest_addr, uint64_t tag, void *context)                                              \
	{                                                                                                              \
		return fi_opx_ep_tx_send(ep, buf, len, desc, dest_addr, tag, context, data, LOCK, /* lock_required */  \
					 1,							  /* is_contiguous */  \
					 0,							  /* override_flags */ \
					 FI_REMOTE_CQ_DATA,					  /* flags */          \
					 CAPS | FI_TAGGED, RELIABILITY, OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL),     \
					 HFI1_TYPE, CTX_SHARING);                                                      \
	}                                                                                                              \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_tinjectdata_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(     \
			struct fid_ep *ep, const void *buf, size_t len, uint64_t data, fi_addr_t dest_addr,            \
			uint64_t tag)                                                                                  \
	{                                                                                                              \
		return fi_opx_ep_tx_inject(ep, buf, len, dest_addr, tag, data, LOCK, FI_REMOTE_CQ_DATA,                \
					   CAPS | FI_TAGGED, RELIABILITY, OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL),   \
					   HFI1_TYPE, CTX_SHARING);                                                    \
	}                                                                                                              \
	__OPX_FORCE_INLINE__ ssize_t                                                                                   \
		fi_opx_tsendmsg_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL(        \
			struct fid_ep *ep, const struct fi_msg_tagged *msg, uint64_t flags)                            \
	{                                                                                                              \
		const size_t niov = msg->iov_count;                                                                    \
		if (niov == 0) {                                                                                       \
			if (!msg->context) {                                                                           \
				return fi_opx_ep_tx_inject(ep, 0, 0, msg->addr, msg->tag, msg->data, LOCK, flags,      \
							   CAPS | FI_TAGGED, RELIABILITY,                              \
							   OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL), HFI1_TYPE,       \
							   CTX_SHARING);                                               \
			}                                                                                              \
			return fi_opx_ep_tx_send(ep, 0, 0, msg->desc, msg->addr, msg->tag, msg->context, msg->data,    \
						 LOCK, OPX_CONTIG_TRUE, OPX_FLAGS_OVERRIDE_TRUE, flags,                \
						 CAPS | FI_TAGGED, RELIABILITY,                                        \
						 OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL), HFI1_TYPE, CTX_SHARING);   \
		}                                                                                                      \
		if (niov == 1) {                                                                                       \
			return fi_opx_ep_tx_send(ep, msg->msg_iov->iov_base, msg->msg_iov->iov_len, msg->desc,         \
						 msg->addr, msg->tag, msg->context, msg->data, LOCK, OPX_CONTIG_TRUE,  \
						 OPX_FLAGS_OVERRIDE_TRUE, flags, CAPS | FI_TAGGED, RELIABILITY,        \
						 OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL), HFI1_TYPE, CTX_SHARING);   \
		}                                                                                                      \
		return fi_opx_ep_tx_send(ep, msg->msg_iov, msg->iov_count, msg->desc, msg->addr, msg->tag,             \
					 msg->context, msg->data, LOCK, OPX_CONTIG_FALSE, OPX_FLAGS_OVERRIDE_TRUE,     \
					 flags, CAPS | FI_TAGGED, RELIABILITY,                                         \
					 OPX_FLOW_CONTROL_MODE_VALUE(FLOW_CONTROL), HFI1_TYPE, CTX_SHARING);           \
	}

#define FI_OPX_TAGGED_SPECIALIZED_FUNC_NAME(TYPE, LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL) \
	FI_OPX_TAGGED_SPECIALIZED_FUNC_NAME_(TYPE, LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL)

#define FI_OPX_TAGGED_SPECIALIZED_FUNC_NAME_(TYPE, LOCK, CAPS, RELIABILITY, HFI1_TYPE, CTX_SHARING, FLOW_CONTROL) \
	fi_opx_##TYPE##_##LOCK##_##CAPS##_##RELIABILITY##_##HFI1_TYPE##_##CTX_SHARING##_##FLOW_CONTROL

#endif /* _FI_PROV_OPX_TAGGED_H_ */
