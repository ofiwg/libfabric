/*
 * Copyright (c) 2015-2016, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015, Intel Corp., Inc.  All rights reserved.
 * Copyright (c) 2022 DataDirect Networks, Inc. All rights reserved.
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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_ext.h>

#include "ofi.h"
#include "ofi_str.h"
#include "ofi_enosys.h"
#include "ofi_util.h"


static const char * const log_levels[] = {
	[FI_LOG_WARN] = "warn",
	[FI_LOG_TRACE] = "trace",
	[FI_LOG_INFO] = "info",
	[FI_LOG_DEBUG] = "debug",
};

static int log_interval = 2000;
static int log_level = -1;
struct ofi_filter prov_log_filter;
extern struct ofi_common_locks common_locks;

static pid_t pid;

static int fi_convert_log_str(const char *value)
{
	int i;

	if (!value)
		return -1;

	for (i = 0; log_levels[i]; i++) {
		if (!strcasecmp(value, log_levels[i]))
			return i;
	}
	return 0;
}

void fi_log_init(void)
{
	char *levelstr = NULL, *provstr = NULL;

	fi_param_define(NULL, "log_interval", FI_PARAM_INT,
			"Delay in ms between rate limited log messages "
			"(default 2000)");
	fi_param_get_int(NULL, "log_interval", &log_interval);

	fi_param_define(NULL, "log_level", FI_PARAM_STRING,
			"Specify logging level: warn, trace, info, debug (default: warn)");
	fi_param_get_str(NULL, "log_level", &levelstr);
	log_level = fi_convert_log_str(levelstr);

	fi_param_define(NULL, "log_prov", FI_PARAM_STRING,
			"Specify specific provider to log (default: all)");
	fi_param_get_str(NULL, "log_prov", &provstr);
	ofi_create_filter(&prov_log_filter, provstr);

	pid = getpid();
}

static int ofi_log_enabled(const struct fi_provider *prov,
			   enum fi_log_level level, int prov_flags,
			   uint64_t flags)
{
	return (level <= log_level) && !(flags & FI_LOG_PROV_FILTERED);
}

static void ofi_log(const struct fi_provider *prov, enum fi_log_level level,
		    int flags, const char *func, int line,
		    const char *msg)
{
	fprintf(stderr, "%s:%d:%ld:%s:%s:%s():%d<%s> %s",
		PACKAGE, pid, (unsigned long) time(NULL), log_prefix,
		prov->name, func, line, log_levels[level], msg);
}

static int ofi_log_ready(const struct fi_provider *prov,
			 enum fi_log_level level, int prov_flags,
			 uint64_t flags, uint64_t *showtime);

static struct fi_ops_log ofi_import_log_ops = {
	.size = sizeof(struct fi_ops),
	.enabled = ofi_log_enabled,
	.ready = ofi_log_ready,
	.log = ofi_log,
};

static int ofi_close_logging_fid(struct fid *fid)
{
	return 0;
}

static int ofi_bind_logging_fid(struct fid *fid, struct fid *bfid,
				uint64_t flags);

static struct fi_ops ofi_logging_ops = {
	.size = sizeof(struct fi_ops),
	.close = ofi_close_logging_fid,
	.bind = ofi_bind_logging_fid,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static struct fid_logging log_fid = {
	.fid = {
		.fclass = FI_CLASS_LOG,
		.ops = &ofi_logging_ops,
	},
	.ops = &ofi_import_log_ops,
};

static int ofi_close_import(struct fid *fid)
{
	/* Reset logging ops to default */
	pthread_mutex_lock(&common_locks.ini_lock);
	log_fid.ops->enabled = ofi_log_enabled;
	log_fid.ops->ready = ofi_log_ready;
	log_fid.ops->log = ofi_log;
	pthread_mutex_unlock(&common_locks.ini_lock);
	return 0;
}

static struct fi_ops impfid_ops = {
	.size = sizeof(struct fi_ops),
	.close = ofi_close_import,
	.bind = fi_no_bind,
	.control = fi_no_control,
	.ops_open = fi_no_ops_open,
	.tostr = fi_no_tostr,
	.ops_set = fi_no_ops_set,
};

static int ofi_logging_import(struct fid *fid)
{
	struct fid_logging *impfid;

	if (fid->fclass != FI_CLASS_LOG)
		return -FI_EINVAL;

	impfid = container_of(fid, struct fid_logging, fid);
	if (impfid->ops->size < sizeof(struct fi_ops_log))
		return -FI_EINVAL;

	pthread_mutex_lock(&common_locks.ini_lock);
	if (impfid->ops->enabled)
		log_fid.ops->enabled = impfid->ops->enabled;
	if (impfid->ops->ready)
		log_fid.ops->ready = impfid->ops->ready;
	if (impfid->ops->log)
		log_fid.ops->log = impfid->ops->log;

	impfid->fid.ops = &impfid_ops;
	pthread_mutex_unlock(&common_locks.ini_lock);
	return 0;
}

static int ofi_bind_logging_fid(struct fid *fid, struct fid *bfid,
				uint64_t flags)
{
	if (flags || bfid->fclass != FI_CLASS_LOG)
		return -FI_EINVAL;

	return ofi_logging_import(bfid);
}

static int ofi_log_ready(const struct fi_provider *prov,
			 enum fi_log_level level, int prov_flags,
			 uint64_t flags, uint64_t *showtime)
{
    uint64_t cur;

    if (log_fid.ops->enabled(prov, level, prov_flags, flags)) {
	    cur = ofi_gettime_ms();
	    if (cur >= *showtime) {
		    *showtime = cur + (uint64_t) log_interval;
		    return true;
	    }
    }

    return false;
}

int ofi_open_log(uint32_t version, void *attr, size_t attr_len,
		 uint64_t flags, struct fid **fid, void *context)
{
	int ret;

	if (FI_VERSION_LT(version, FI_VERSION(1, 13)) || attr_len)
		return -FI_EINVAL;

	if (flags)
		return -FI_EBADFLAGS;

	/* The logging subsystem can be opened once only! */
	pthread_mutex_lock(&common_locks.ini_lock);
	if (log_fid.ops->enabled != ofi_log_enabled ||
	    log_fid.ops->ready != ofi_log_ready ||
	    log_fid.ops->log != ofi_log) {
		ret = -FI_EALREADY;
		goto unlock;
	}

	log_fid.fid.context = context;
	*fid = &log_fid.fid;
	ret = 0;

unlock:
	pthread_mutex_unlock(&common_locks.ini_lock);
	return ret;
}

void ofi_tostr_log_level(char *buf, size_t len, enum fi_log_level level)
{
    if (level > FI_LOG_DEBUG)
	ofi_strncatf(buf, len, "Unknown");
    else
	ofi_strncatf(buf, len, log_levels[level]);
}

void fi_log_fini(void)
{
	ofi_free_filter(&prov_log_filter);
}

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
int DEFAULT_SYMVER_PRE(fi_log_enabled)(const struct fi_provider *prov,
		enum fi_log_level level, int flags)
{
	uint64_t enable_flags = 0;

	if (ofi_prov_ctx(prov)->disable_logging)
		enable_flags |= FI_LOG_PROV_FILTERED;

	return log_fid.ops->enabled(prov, level, flags, enable_flags);
}
DEFAULT_SYMVER(fi_log_enabled_, fi_log_enabled, FABRIC_1.0);

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
int DEFAULT_SYMVER_PRE(fi_log_ready)(const struct fi_provider *prov,
		enum fi_log_level level, int flags,
		uint64_t *showtime)
{
	uint64_t ready_flags = 0;

	if (ofi_prov_ctx(prov)->disable_logging)
		ready_flags |= FI_LOG_PROV_FILTERED;

	return log_fid.ops->ready(prov, level, flags, ready_flags, showtime);
}
DEFAULT_SYMVER(fi_log_ready_, fi_log_ready, FABRIC_1.6);

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
void DEFAULT_SYMVER_PRE(fi_log)(const struct fi_provider *prov,
		enum fi_log_level level,
		int flags, const char *func, int line,
		const char *fmt, ...)
{
	char msg[1024];
	int size = 0;
	va_list vargs;

	va_start(vargs, fmt);
	vsnprintf(msg + size, sizeof(msg) - size, fmt, vargs);
	va_end(vargs);

	log_fid.ops->log(prov, level, flags, func, line, msg);
}
DEFAULT_SYMVER(fi_log_, fi_log, FABRIC_1.0);
