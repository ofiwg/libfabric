/*
 * Copyright (c) 2015, Cisco Systems, Inc. All rights reserved.
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

#if !defined(FI_LOG_H)
#define FI_LOG_H

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <fi_list.h>
#include <stdbool.h>
#include <stdio.h>

/*
 * Define the basic subsystems that can be used to enable fine grained logging.
 */
enum {
	FI_FABRIC,
	FI_DOMAIN,
	FI_EP_CM,
	FI_EP_DM,
	FI_AV,
	FI_CQ,
	FI_EQ,
	FI_MR
};

enum {
	FI_LOG_WARN = 0,
	FI_LOG_TRACE = 3,
	FI_LOG_INFO = 7
};

/*
 * Log level occupies bit positions 0-2
 * Subsystem occupies bit positions 3-31
 * Provider occupes bit positions 32-63
 */
#define SUBSYS_OFFSET 3
#define PROV_OFFSET 32

#define MASK(width) ((UINT64_C(1) << (width)) - 1)

#define SUBSYS_MASK (MASK(29) << SUBSYS_OFFSET)
#define PROV_MASK (MASK(32) << PROV_OFFSET)

/*
 * Take bit position and offset and set bit in UINT64
 */
#define EXPAND(position, offset) (UINT64_C(1) << ((position) + (offset)))

#define FI_LOG_TAG(prov, subsys, level)                                        \
	(EXPAND((prov), PROV_OFFSET) | EXPAND((subsys), SUBSYS_OFFSET) | level)

extern uint64_t log_mask;

struct log_providers {
	struct slist names;
	bool negated;
};

struct provider_parameter {
	struct slist_entry entry;
	char *name;
};

void fi_log_init(void);
void set_provider(const char *prov_name, int position);

void fi_err_impl(const char *prov, int subsystem, const char *fmt, ...);
void fi_log_impl(const char *prov, int level, int subsystem, const char *func,
		 int line, const char *fmt, ...);
void fi_debug_impl(const char *prov, int subsystem, const char *func, int line,
		   const char *fmt, ...);

#define FI_ERR(prov_name, subsystem, ...)                                      \
	fi_err_impl(prov_name, subsystem, __VA_ARGS__)

#define FI_LOG(prov_name, prov, level, subsystem, ...)                         \
	do {                                                                   \
		if ((FI_LOG_TAG(prov, subsystem, level) & log_mask) ==         \
		    FI_LOG_TAG(prov, subsystem, level))                        \
			fi_log_impl(prov_name, level, subsystem, __func__,     \
				    __LINE__, __VA_ARGS__);                    \
	} while (0)

/*
 * FI_DEBUG always print if configured with --enable-debug.
 */
#if ENABLE_DEBUG
#define FI_DEBUG(prov_name, subsystem, ...)                                    \
	fi_debug_impl(prov_name, subsystem, __func__, __LINE__, __VA_ARGS__)
#else
#define FI_DEBUG(prov_name, subsystem, ...)
#endif

#endif /* !defined(FI_LOG_H) */
