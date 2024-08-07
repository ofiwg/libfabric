/*
 * Copyright (c) 2018-2024 GigaIO, Inc. All Rights Reserved.
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

#include <sys/param.h>
#include <string.h>

#include "test.h"
#include "error.h"
#include "ipc.h"

bool debug_quiet = false;
bool verbose = false;

void debug_trace_push(struct rank_info *ri, int line, const char *func,
		      const char *file, const char *debugstr)
{
	assert(ri->tracei < MAXTRACE);
	ri->trace_lines[ri->tracei] = line;
	ri->trace_funcs[ri->tracei] = func;
	ri->trace_files[ri->tracei] = file;
	ri->tracei++;

	char newstr[128];
	strncpy(newstr, debugstr, sizeof(newstr));
	newstr[127] = '\0';
	// Truncate the str to keep lines reasonable lengths.
	size_t len = strlen(newstr);
	const int trunc = 70;
	newstr[MIN(len, trunc)] = '\0';
	if (len > trunc) {
		strcat(newstr, "...\n");
	} else {
		strcat(newstr, "\n");
	}
	debugln(file, line, "[rank:%ld node:%c iter:%ld test:%s] %s",
		(ri)->rank, my_node_name, (ri)->iteration, (ri)->cur_test_name,
		newstr);
}

void debug_trace_pop(struct rank_info *ri)
{
	ri->tracei--;
}

void debug_dump_trace(struct rank_info *ri)
{
	for (int i = 0; i < ri->tracei; i++) {
		fprintf(stderr,
			RED_CODE
			"trace: %s():%s:%d [rank:%ld node:%c iter:%ld]" RESET_CODE
			"\n",
			ri->trace_funcs[i], ri->trace_files[i],
			ri->trace_lines[i], ri->rank, my_node_name,
			ri->iteration);
	}
}
