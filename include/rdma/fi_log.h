/*
 * Copyright (c) 2015, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015, Intel Corp., Inc. All rights reserved.
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
 *
 */

#ifndef FI_LOG_H
#define FI_LOG_H

#include <rdma/fabric.h>
#include <rdma/fi_prov.h>

#ifdef __cplusplus
extern "C" {
#endif

enum fi_log_subsys {
	FI_LOG_CORE,
	FI_LOG_FABRIC,
	FI_LOG_DOMAIN,
	FI_LOG_EP_CTRL,
	FI_LOG_EP_DATA,
	FI_LOG_AV,
	FI_LOG_CQ,
	FI_LOG_EQ,
	FI_LOG_MR,
	FI_LOG_SUBSYS_MAX
};

enum fi_log_level {
	FI_LOG_WARN,
	FI_LOG_TRACE,
	FI_LOG_INFO,
	FI_LOG_DEBUG,
	FI_LOG_MAX
};

int fi_log_enabled(const struct fi_provider *prov, enum fi_log_level level,
		   enum fi_log_subsys subsys);
void fi_log(const struct fi_provider *prov, enum fi_log_level level,
	    enum fi_log_subsys subsys, const char *func, int line,
	    const char *fmt, ...);

#ifdef __cplusplus
}
#endif

#endif /* FI_LOG_H */
