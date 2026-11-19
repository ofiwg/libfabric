/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_ERRNO_H
#define EFA_ERRNO_H

#include <ofi_osd.h>

#include <rdma/fi_efa_errno.h>

/**
 * @brief Convert an EFA error code into a common Libfabric error code
 *
 * @param[in]	err	An EFA-specific error code
 * @return	Analogous common Libfabric error code
 *
 * @sa fi_errno(3)
 */
static inline int to_fi_errno(enum efa_errno err) {
	switch (err) {
	case EFA_IO_COMP_STATUS_OK:
		return FI_SUCCESS;
	case EFA_IO_COMP_STATUS_FLUSHED:
		return FI_ECANCELED;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_QP_INTERNAL_ERROR:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_AH:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_LKEY:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNSUPPORTED_OP:
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_ADDRESS:
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_FEATURE_MISMATCH:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_INVALID_ACTION:
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_INVALID_ACTION:
		return FI_EINVAL;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNREACH_REMOTE:
		return FI_EHOSTUNREACH;
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_BAD_LENGTH:
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_LENGTH:
		return FI_EMSGSIZE;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_ABORT:
	case EFA_IO_COMP_STATUS_LOCAL_ERROR_UNRESP_REMOTE:
	case FI_EFA_ERR_ESTABLISHED_RECV_UNRESP:
	case FI_EFA_ERR_UNESTABLISHED_RECV_UNRESP:
		return FI_ECONNABORTED;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_DEST_QPN:
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_UNKNOWN_PEER:
		return FI_ENOTCONN;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_RNR:
		return FI_ENORX;
	case EFA_IO_COMP_STATUS_REMOTE_ERROR_BAD_STATUS:
		return FI_EREMOTEIO;
	case FI_EFA_ERR_OOM:
		return FI_ENOMEM;
	default:
		return FI_EOTHER;
	}
}

const char *efa_strerror(enum efa_errno);
void efa_show_help(enum efa_errno);

#endif
