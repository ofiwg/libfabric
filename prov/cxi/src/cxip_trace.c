/*
 * Copyright (c) 2021-2023 Hewlett Packard Enterprise Development LP
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 */

/**
 * @brief TRACE function for producing runtime debugging logs
 *
 * The following should be inserted at the top of a code module to trace:
 *
 *   #define TRACE(fmt, ...) CXIP_TRACE(<module>, fmt, ##__VA_ARGS__)
 *
 * If ENABLE_DEBUG is false at compile time, CXIP_TRACE is a syntactically
 * robust NOOP which results in no code being emitted, ensuring that these
 * trace calls do not affect performance in production, and none of the
 * following comment apply.
 *
 * - cxip_trace_fn is the function that logs a trace message.
 * - cxip_trace_flush_fn can be used to flush buffered trace messages.
 * - cxip_trace_close_fn can be used to flush and close the output.
 * - cxip_trace_enable_fn is used to enable/disable all tracing.
 * - cxip_trace_set() is used to enable a tracing module.
 * - cxip_trace_clr() is used to disable a tracing module.
 *
 * Modules are defined by the list of enum cxip_trace_module values, which
 * can be extended as needed to provide finer control over tracing.
 *
 * The initial values are set in cxip_trace_init() below, using run-time
 * environment variables. cxip_trace_enable() can be used to dynamically
 * enable or disable tracing. cxip_trace_set() and cxip_trace_clr() can be
 * used to dynamically modify which traces will generate output.
 *
 * Some initialization is required by the use of environment variables:
 *
 * Specifying the environment variable CXIP_TRACE_FILENAME will deliver
 * output to a file with the specified name, followed by the PMI_RANK value
 * (if there is one).
 *
 * Specifying CXIP_TRACE_APPEND in conjunction with CXIP_TRACE_FILENAME will
 * open the file in append mode. This is important for NETSIM tests under
 * Criterion, since each test is run in a separate process and closes all
 * files at completion of each test.
 *
 * Specifying PMI_RANK as a rank value will apply a prefix to the trace lines
 * that identifies the rank of the trace.
 *
 * Specifying PMI_SIZE will expand the prefix to show the number of ranks.
 *
 * cxip_trace_fid is exposed, and can be manipulated using the normal stream
 * file functions. Default buffering is fully buffered output, which can
 * result in delays in the appearance of logging information. Using
 * setlinebuf() will run slower, but will display lines more quickly.
 *
 * cxip_trace_flush() forces all output be flushed AND written to disk, but
 * leaves the file open for more writing.
 *
 * cxip_trace_close() flushes all output and closes the file.
 */
#include "config.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>

#include "cxip.h"

bool cxip_trace_initialized;
bool cxip_trace_enabled;
bool cxip_trace_append;
bool cxip_trace_linebuf;	// set line buffering for trace
int cxip_trace_rank;
int cxip_trace_numranks;
char *cxip_trace_filename;
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

void cxip_trace_init(void)
{
	const char *fname;

	if (cxip_trace_initialized)
		return;

	cxip_trace_initialized = true;
	cxip_trace_enabled = !!getenv("CXIP_TRACE_ENABLE");
	cxip_trace_append = !!getenv("CXIP_TRACE_APPEND");
	cxip_trace_linebuf = !!getenv("CXIP_TRACE_LINEBUF");
	cxip_trace_rank = getenv_int("PMI_RANK");
	cxip_trace_numranks = getenv_int("PMI_SIZE");
	cxip_trace_append = getenv("CXIP_TRACE_APPEND");
	fname = getenv("CXIP_TRACE_FILENAME");

	cxip_trace_mask = 0L;
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

	cxip_trace_filename = NULL;
	cxip_trace_fid = NULL;
	if (!fname)
		fname = "trace";
	if (asprintf(&cxip_trace_filename, "./%s%d",
			fname, cxip_trace_rank) > 0) {
		cxip_trace_fid = fopen(cxip_trace_filename,
				       cxip_trace_append ? "a" : "w");
		if (!cxip_trace_fid) {
			fprintf(stderr, "open(%s) failed: %s\n",
				cxip_trace_filename, strerror(errno));
			free(cxip_trace_filename);
			cxip_trace_filename = NULL;
		}
		if (cxip_trace_linebuf && cxip_trace_fid)
			setlinebuf(cxip_trace_fid);
	}
}

void cxip_trace_flush(void)
{
	cxip_trace_init();
	if (cxip_trace_fid) {
		fflush(cxip_trace_fid);
		fsync(fileno(cxip_trace_fid));
	}
}

void cxip_trace_close(void)
{
	cxip_trace_init();
	if (cxip_trace_fid) {
		cxip_trace_flush();
		fclose(cxip_trace_fid);
		cxip_trace_fid = NULL;
		if (cxip_trace_filename) {
			free(cxip_trace_filename);
			cxip_trace_filename = NULL;
		}
		cxip_trace_initialized = false;
	}
}

int cxip_trace_attr cxip_trace(const char *fmt, ...)
{
	va_list args;
	char *str;
	int len;

	cxip_trace_init();
	if (!cxip_trace_enabled)
		return 0;
	va_start(args, fmt);
	len = vasprintf(&str, fmt, args);
	va_end(args);
	if (len >= 0) {
		len = fprintf(cxip_trace_fid, "[%2d|%2d] %s",
			      cxip_trace_rank, cxip_trace_numranks, str);
		free(str);
	}
	return len;
}

bool cxip_trace_enable(bool enable)
{
	bool was_enabled = cxip_trace_enabled;

	cxip_trace_init();
	cxip_trace_enabled = enable;
	return was_enabled;
}
