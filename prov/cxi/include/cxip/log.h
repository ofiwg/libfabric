/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_LOG_H_
#define _CXIP_LOG_H_


/* Macros */
#define CXIP_LOG(fmt,  ...) \
	fi_log(&cxip_prov, FI_LOG_WARN, FI_LOG_CORE, \
	       __func__, __LINE__, "%s: " fmt "", cxip_env.hostname, \
	       ##__VA_ARGS__)

#define CXIP_FATAL(fmt, ...)					\
	do {							\
		CXIP_LOG(fmt, ##__VA_ARGS__);			\
		abort();					\
	} while (0)

#endif /* _CXIP_LOG_H_ */
