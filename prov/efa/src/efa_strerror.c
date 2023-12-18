/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa_errno.h"

#define UNKNOWN_ERR_STR     "Unknown error"
#define FI_EFA_ERRNO_OFFSET FI_EFA_ERR_OTHER

static const char *efa_io_comp_status_str(enum efa_errno status)
{
	static const char *status_str[] = {
		[FI_EFA_OK]                            = "Success",
		[FI_EFA_FLUSHED]                       = "Flushed during queue pair destroy",
		[FI_EFA_LOCAL_ERROR_QP_INTERNAL_ERROR] = "Internal queue pair error",
		[FI_EFA_LOCAL_ERROR_INVALID_OP_TYPE]   = "Invalid operation type",
		[FI_EFA_LOCAL_ERROR_INVALID_AH]        = "Invalid address handle",
		[FI_EFA_LOCAL_ERROR_INVALID_LKEY]      = "Invalid local key (LKEY)",
		[FI_EFA_LOCAL_ERROR_BAD_LENGTH]        = "Message too long",
		[FI_EFA_REMOTE_ERROR_BAD_ADDRESS]      = "Invalid address",
		[FI_EFA_REMOTE_ERROR_ABORT]            = "Receiver connection aborted",
		[FI_EFA_REMOTE_ERROR_BAD_DEST_QPN]     = "Invalid receiver queue pair number (QPN). "
		                                         "This error is typically caused by a crashed peer. "
		                                         "Please verify the peer application has not "
		                                         "terminated unexpectedly.",
		[FI_EFA_REMOTE_ERROR_RNR]              = "Receiver not ready",
		[FI_EFA_REMOTE_ERROR_BAD_LENGTH]       = "Receiver scatter-gather list (SGL) too short",
		[FI_EFA_REMOTE_ERROR_BAD_STATUS]       = "Unexpected status received from remote",
		[FI_EFA_LOCAL_ERROR_UNRESP_REMOTE]     = "Unresponsive receiver. "
		                                         "This error is typically caused by a peer hardware failure or "
		                                         "incorrect inbound/outbound rules in the security group - "
		                                         "EFA requires \"All traffic\" type allowlisting. "
		                                         "Please also verify the peer application has not "
		                                         "terminated unexpectedly.",
	};

	return (status < FI_EFA_OK || status > FI_EFA_LOCAL_ERROR_UNRESP_REMOTE)
		? UNKNOWN_ERR_STR
		: status_str[status];
}

static const char *efa_errno_str(enum efa_errno err)
{
	static const char *errno_str[] = {
		[FI_EFA_ERR_OTHER                 - FI_EFA_ERRNO_OFFSET] = UNKNOWN_ERR_STR,
		[FI_EFA_ERR_DEPRECATED_PKT_TYPE   - FI_EFA_ERRNO_OFFSET] = "Deprecated packet type encountered",
		[FI_EFA_ERR_INVALID_PKT_TYPE      - FI_EFA_ERRNO_OFFSET] = "Invalid packet type encountered",
		[FI_EFA_ERR_UNKNOWN_PKT_TYPE      - FI_EFA_ERRNO_OFFSET] = "Unknown packet type encountered",
		[FI_EFA_ERR_PKT_POST              - FI_EFA_ERRNO_OFFSET] = "Failure to post packet",
		[FI_EFA_ERR_PKT_SEND              - FI_EFA_ERRNO_OFFSET] = "Failure to send packet",
		[FI_EFA_ERR_PKT_PROC_MSGRTM       - FI_EFA_ERRNO_OFFSET] = "Error processing non-tagged RTM",
		[FI_EFA_ERR_PKT_PROC_TAGRTM       - FI_EFA_ERRNO_OFFSET] = "Error processing tagged RTM",
		[FI_EFA_ERR_PKT_ALREADY_PROCESSED - FI_EFA_ERRNO_OFFSET] = "Packet already processed",
		[FI_EFA_ERR_OOM                   - FI_EFA_ERRNO_OFFSET] = "Out of memory",
		[FI_EFA_ERR_MR_DEREG              - FI_EFA_ERRNO_OFFSET] = "MR deregistration error",
		[FI_EFA_ERR_RXE_COPY         - FI_EFA_ERRNO_OFFSET] = "rxe copy error",
		[FI_EFA_ERR_RXE_POOL_EXHAUSTED  - FI_EFA_ERRNO_OFFSET] = "RX entries exhausted",
		[FI_EFA_ERR_TXE_POOL_EXHAUSTED  - FI_EFA_ERRNO_OFFSET] = "TX entries exhausted",
		[FI_EFA_ERR_AV_INSERT             - FI_EFA_ERRNO_OFFSET] = "Failure inserting address into address vector",
		[FI_EFA_ERR_RMA_ADDR              - FI_EFA_ERRNO_OFFSET] = "RMA address verification failed",
		[FI_EFA_ERR_INTERNAL_RX_BUF_POST  - FI_EFA_ERRNO_OFFSET] = "Failure to post internal receive buffers",
		[FI_EFA_ERR_PEER_HANDSHAKE        - FI_EFA_ERRNO_OFFSET] = "Failure to post handshake to peer",
		[FI_EFA_ERR_WR_POST_SEND          - FI_EFA_ERRNO_OFFSET] = "Failure to post work request(s) to send queue",
		[FI_EFA_ERR_RTM_MISMATCH          - FI_EFA_ERRNO_OFFSET] = "RTM size mismatch",
		[FI_EFA_ERR_READ_POST             - FI_EFA_ERRNO_OFFSET] = "Error posting read request",
		[FI_EFA_ERR_RDMA_READ_POST        - FI_EFA_ERRNO_OFFSET] = "Error posting RDMA read request",
		[FI_EFA_ERR_INVALID_DATATYPE      - FI_EFA_ERRNO_OFFSET] = "Invalid datatype encountered",
		[FI_EFA_ERR_WRITE_SEND_COMP       - FI_EFA_ERRNO_OFFSET] = "Failure to write send completion",
		[FI_EFA_ERR_WRITE_RECV_COMP       - FI_EFA_ERRNO_OFFSET] = "Failure to write receive completion",
		[FI_EFA_ERR_DGRAM_CQ_READ         - FI_EFA_ERRNO_OFFSET] = "Error reading from DGRAM CQ",
		[FI_EFA_ERR_SHM_INTERNAL_ERROR    - FI_EFA_ERRNO_OFFSET] = "SHM internal error",
		[FI_EFA_ERR_WRITE_SHM_CQ_ENTRY    - FI_EFA_ERRNO_OFFSET] = "Failure to write CQ entry for SHM operation",
	};

	return (err < FI_EFA_ERRNO_OFFSET || err >= FI_EFA_ERRNO_MAX)
		? UNKNOWN_ERR_STR
		: errno_str[err - FI_EFA_ERRNO_OFFSET];
}

/**
 * @brief Convert an EFA error code into a short, printable string
 *
 * Given a non-negative EFA-specific error code, this function returns a pointer
 * to a null-terminated string that corresponds to it; suitable for
 * interpolation in logging messages.
 *
 * @param[in]	err    An EFA-specific error code
 * @return	Null-terminated string with static storage duration (caller does
 *		not free).
 */
const char *efa_strerror(enum efa_errno err)
{
	return err >= FI_EFA_ERRNO_OFFSET
		? efa_errno_str(err)
		: efa_io_comp_status_str(err);
}
