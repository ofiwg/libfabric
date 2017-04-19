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

#ifndef _FI_NETDIR_LOG_H_
#define _FI_NETDIR_LOG_H_

#include <windows.h>

#include "rdma/providers/fi_log.h"

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

extern struct fi_provider ofi_nd_prov;

#define ND_LOG(level, subsystem, ...) FI_LOG(&ofi_nd_prov, level, subsystem, __VA_ARGS__)

#define ND_LOG_WARN(subsystem, ...) ND_LOG(FI_LOG_WARN, subsystem, __VA_ARGS__)
#define ND_LOG_INFO(subsystem, ...) ND_LOG(FI_LOG_INFO, subsystem, __VA_ARGS__)
#define ND_LOG_DEBUG(subsystem, ...) ND_LOG(FI_LOG_DEBUG, subsystem, __VA_ARGS__)

#define FI_ND_GUID_FORMAT "%08lX-%04hX-%04hX-%02hhX%02hhX-%02hhX%02hhX%02hhX%02hhX%02hhX%02hhX"
#define FI_ND_GUID_ARG(guid)					\
	(guid).Data1, (guid).Data2, (guid).Data3,		\
	(guid).Data4[0], (guid).Data4[1], (guid).Data4[2],	\
	(guid).Data4[3], (guid).Data4[4], (guid).Data4[5],	\
	(guid).Data4[6], (guid).Data4[7]

/* ofi_nd_strerror generates string message based on err value (GetLastError)
   returned string is valid till next call of ofi_nd_strerror
*/
static inline char *ofi_nd_strerror(DWORD err, HMODULE module)
{
	static char *message = 0;

	/* if message is allocated - free it */
	if (message)
		LocalFree(message);

	size_t size = FormatMessageA(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM |
		FORMAT_MESSAGE_IGNORE_INSERTS |
		(module ? FORMAT_MESSAGE_FROM_HMODULE : 0),
		module, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		(LPSTR)&message, 0, NULL);

	return size ? message : (char*)"";
}

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* _FI_NETDIR_LOG_H_ */

