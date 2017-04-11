/*
* Copyright (c) 2015-2016 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _FI_NETDIR_ERR_H_
#define _FI_NETDIR_ERR_H_

#include <windows.h>

#include "rdma/fabric.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#define H2F(x) ofi_nd_hresult_2_fierror(x)

static inline int ofi_nd_hresult_2_fierror(HRESULT hr)
{
	switch (hr) {
	case S_OK:
	case ND_PENDING:
		return FI_SUCCESS;
	case ND_BUFFER_OVERFLOW:
		return -EOVERFLOW; /* FI_EOVERFLOW */
	case ND_CONNECTION_REFUSED:
		return -FI_ECONNREFUSED;
	case ND_TIMEOUT:
		return -FI_ETIMEDOUT;
	default:
		return -FI_EOTHER;
	}
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _FI_NETDIR_ERR_H_ */

