/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_PROV_H
#define EFA_PROV_H

extern struct fi_provider efa_prov;
extern struct util_prov efa_util_prov;

#define EFA_WARN(flags, ...) FI_WARN(&efa_prov, flags, __VA_ARGS__)
#define EFA_WARN_ONCE(flags, ...) FI_WARN_ONCE(&efa_prov, flags, __VA_ARGS__)
#define EFA_TRACE(flags, ...) FI_TRACE(&efa_prov, flags, __VA_ARGS__)
#define EFA_INFO(flags, ...) FI_INFO(&efa_prov, flags, __VA_ARGS__)
#define EFA_INFO_ERRNO(flags, fn, errno) \
	EFA_INFO(flags, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_WARN_ERRNO(flags, fn, errno) \
	EFA_WARN(flags, fn ": %s(%d)\n", strerror(errno), errno)
#define EFA_DBG(flags, ...) FI_DBG(&efa_prov, flags, __VA_ARGS__)

#endif