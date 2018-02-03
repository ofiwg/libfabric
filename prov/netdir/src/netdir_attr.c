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

#ifdef _WIN32

#include "ofi.h"
#include "ofi_util.h"
#include "rdma/fabric.h"

#include "netdir.h"

const char ofi_nd_prov_name[] = "netdir";

struct fi_provider ofi_nd_prov = {
	.name = ofi_nd_prov_name,
	.version = FI_VERSION(OFI_ND_MAJOR_VERSION, OFI_ND_MINOR_VERSION),
	.fi_version = FI_VERSION(1, 6),
	.getinfo = ofi_nd_getinfo,
	.fabric = ofi_nd_fabric,
	.cleanup = ofi_nd_fini
};

struct util_prov ofi_nd_util_prov = {
	.prov = &ofi_nd_prov,
	.info = 0,
	.flags = UTIL_RX_SHARED_CTX,
};

#endif /* _WIN32 */
