/*
 * Copyright (c) 2021-2024 Hewlett Packard Enterprise Development LP
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 */

/**
 * @brief TRACE function for producing runtime debugging logs
 *
 * This creates log files on each running node of a multinode job, which
 * contain trace information produced on that node. SLURM nodes inherit
 * stdout/stderr from the launch node.
 *
 * If ENABLE_DEBUG is unset, or evaluates to false at compile time, CXIP_TRACE
 * is a syntactically robust NOOP which results in no code being emitted,
 * ensuring that these trace calls do not affect performance in production, and
 * none of the following comments apply. ENABLE_DEBUG is normally true for
 * development builds, and false for production builds.
 *
 * To use this feature, create a .bashrc file in the runtime directory of each
 * node in the job (or use a shared FS for all nodes in the job), containing one
 * or more of the CXIP_TRC_* environment variables.
 *
 * Note that the environment settings for each node in a multi-node job may
 * be different, and will result in different tracing on each node.
 *
 * For instance:
 * export CXIP_TRC_PATHNAME = "/mypath/myfile"
 * export CXIP_TRC_LINEBUF=1
 * export CXIP_TRC_COLL_JOIN=1
 * export CXIP_TRC_COLL_PKT=1
 *
 * All instances of the following in the code will result in output:
 * CXIP_TRACE(CXIP_TRC_COLL_JOIN, fmt, ...);
 * CXIP_TRACE(CXIP_TRC_COLL_PKT, fmt, ...);
 * All instances of other CXIP_TRC_* values will be silent.
 *
 * Environment variables used in setup:
 * CXIP_TRC_PATHNAME defines the output path name, and defaults to "trace"
 * CXIP_TRC_LINEBUF sets or clears line buffering out output, and defaults to 0.
 * CXIP_TRC_APPEND sets or clears open append mode, and defaults to 0.
 *
 * CXIP_TRC_APPEND is needed for NETSIM tests under Criterion, since each
 * test is run in a separate process and closes all files at completion of
 * each test. If CXIP_TRC_APPEND is not set, you will see only the tracie of
 * the last test run.
 *
 * Specifying PMI_RANK as a rank value will apply a prefix to the trace lines
 * that identifies the rank of the trace. Note that this is normally exported
 * by the SLURM environment, or the multinode test framework.
 *
 * Specifying PMI_SIZE will expand the prefix to show the number of ranks.
 * Note that this is normally exported by the SLURM environment, or the
 * multinode test framework.
 *
 * cxip_trace_fid is exposed, and can be manipulated using the normal stream
 * file functions. Default buffering is character buffered output, which can
 * result in delays in the appearance of logging information but minimizes
 * logging effects on runtime performance. Using setlinebuf() will impact
 * runtime performance, but will minimize output delays.
 *
 * Tracing is initialized by cxip_trace_enable(true). If this is not called,
 * or is called only with an argument of false, no trace files will be
 * opened, and no tracing will occur. Tracing can be dynamically paused and
 * renabled during run-time using cxip_trace_enable(). This does not close
 * the logging file.
 *
 * cxip_trace_flush() forces all output be flushed AND written to disk, but
 * leaves the file open for more writing.
 *
 * cxip_trace_close() flushes all output and closes the file.
 *
 * cxip_trace() is used to generate trace messages, and is normally called
 * through the CXIP_TRACE() macro.
 */
#include "config.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "cxip.h"

bool cxip_trace_initialized;
bool cxip_trace_enabled;
bool cxip_trace_append;
bool cxip_trace_linebuf;	// set line buffering for trace
int cxip_trace_rank;
int cxip_trace_numranks;
char *cxip_trace_pathname;
FILE *cxip_trace_fid;
uint64_t cxip_trace_mask;

/* Static initialization of default trace functions, can be overridden */
cxip_trace_t cxip_trace_attr cxip_trace_fn = cxip_trace;
cxip_trace_flush_t cxip_trace_flush_fn = cxip_trace_flush;
cxip_trace_close_t cxip_trace_close_fn = cxip_trace_close;
cxip_trace_enable_t cxip_trace_enable_fn = cxip_trace_enable;

/* Get environment variable as string representation of int */
static int getenv_int(const char *name)
{
	char *env;
	int value;

	value = -1;
	env = getenv(name);
	if (env)
		sscanf(env, "%d", &value);
	return value;
}

static void cxip_trace_init(void)
{
	const char *fpath;
	int ret;

	/* must disable before re-enabling */
	if (cxip_trace_initialized)
		return;

	cxip_trace_mask = 0L;
	cxip_trace_enabled = false;
	cxip_trace_fid = NULL;
	cxip_trace_pathname = NULL;

	cxip_trace_append = !!getenv("CXIP_TRC_APPEND");
	cxip_trace_linebuf = !!getenv("CXIP_TRC_LINEBUF");
	cxip_trace_rank = getenv_int("PMI_RANK");
	cxip_trace_numranks = getenv_int("PMI_SIZE");
	fpath = getenv("CXIP_TRC_PATHNAME");

	/* set bits in cxip_trace_mask */
	if (getenv("CXIP_TRC_CTRL"))
		cxip_trace_set(CXIP_TRC_CTRL);
	if (getenv("CXIP_TRC_ZBCOLL"))
		cxip_trace_set(CXIP_TRC_ZBCOLL);
	if (getenv("CXIP_TRC_CURL"))
		cxip_trace_set(CXIP_TRC_CURL);
	if (getenv("CXIP_TRC_COLL_PKT"))
		cxip_trace_set(CXIP_TRC_COLL_PKT);
	if (getenv("CXIP_TRC_COLL_JOIN"))
		cxip_trace_set(CXIP_TRC_COLL_JOIN);
	if (getenv("CXIP_TRC_COLL_DEBUG"))
		cxip_trace_set(CXIP_TRC_COLL_DEBUG);
	if (getenv("CXIP_TRC_TEST_CODE"))
		cxip_trace_set(CXIP_TRC_TEST_CODE);

	/* if no trace masks set, do nothing */
	if (!cxip_trace_mask)
		return;

	if (!fpath)
		fpath = "trace";
	ret = asprintf(&cxip_trace_pathname, "%s%d", fpath, cxip_trace_rank);
	if (ret <= 0) {
		fprintf(stderr, "asprintf() failed = %s\n", strerror(ret));
		return;
	}
	cxip_trace_fid =
		fopen(cxip_trace_pathname, cxip_trace_append ? "a" : "w");
	if (!cxip_trace_fid) {
		fprintf(stderr, "open('%s') failed: %s\n", cxip_trace_pathname,
			strerror(errno));
		free(cxip_trace_pathname);
		cxip_trace_pathname = NULL;
		return;
	}
	if (cxip_trace_linebuf && cxip_trace_fid)
		setlinebuf(cxip_trace_fid);

	cxip_trace_initialized = true;
}

void cxip_trace_flush(void)
{
	if (cxip_trace_fid) {
		fflush(cxip_trace_fid);
		fsync(fileno(cxip_trace_fid));
	}
}

void cxip_trace_close(void)
{
	cxip_trace_enabled = false;
	if (cxip_trace_fid) {
		cxip_trace_flush();
		fclose(cxip_trace_fid);
		cxip_trace_fid = NULL;
		if (cxip_trace_pathname) {
			free(cxip_trace_pathname);
			cxip_trace_pathname = NULL;
		}
	}
	cxip_trace_initialized = false;
}

int cxip_trace_attr cxip_trace(const char *fmt, ...)
{
	va_list args;
	char *str;
	int len;

	if (!cxip_trace_enabled)
		return 0;
	va_start(args, fmt);
	len = vasprintf(&str, fmt, args);
	va_end(args);
	if (len >= 0) {
		len = fprintf(cxip_trace_fid, "[%2d|%2d] %s", cxip_trace_rank,
			      cxip_trace_numranks, str);
		free(str);
	}
	return len;
}

bool cxip_trace_enable(bool enable)
{
	bool was_enabled = cxip_trace_enabled;

	if (enable)
		cxip_trace_init();
	cxip_trace_enabled = enable;
	return was_enabled;
}
