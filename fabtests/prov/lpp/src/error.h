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

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <stdbool.h>

#define RED_CODE "\033[1;31m"
#define RESET_CODE "\033[0m"
#define MAXTRACE (64)

#define my_node_name	(my_node == NODE_A ? 'A' : 'B')

#define debugln(file, line, fmt, ...)                                          \
	do {                                                                   \
		if (!debug_quiet) {                                            \
			fprintf(stderr, "%14s:%-10d: " fmt, file, line,        \
				##__VA_ARGS__);                                \
		}                                                              \
	} while (0)

#define debug(fmt, ...) debugln(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

#define debugt(ri, msg)                                                        \
	debugln(__FILE__, __LINE__, "[rank:%ld node:%c iter:%ld test:%s] %s",  \
		(ri)->rank, my_node_name, (ri)->iteration,                     \
		(ri)->cur_test_name, msg)

// Like errx() from err.h.
#define errorx(fmt, ...)                                                       \
	fprintf(stderr, RED_CODE "%s():%s:%d: " fmt RESET_CODE "\n", __func__, \
		__FILE__, __LINE__, ##__VA_ARGS__);                            \
	exit(1);

// Like err() from err.h.
#define error(fmt, ...)                                                        \
	fprintf(stderr,                                                        \
		RED_CODE "%s():%s:%d (errno: %s): " fmt RESET_CODE "\n",       \
		__func__, __FILE__, __LINE__, strerror(errno), ##__VA_ARGS__); \
	exit(1);

#define warn(fmt, ...)                                                         \
	fprintf(stderr,                                                        \
		RED_CODE "%s():%s:%d (errno: %s): " fmt RESET_CODE "\n",       \
		__func__, __FILE__, __LINE__, strerror(errno), ##__VA_ARGS__);

// Like errorx(), but with context about the current rank.
#define ERRORX(ri, fmt, ...)                                                   \
	do {                                                                   \
		debug_dump_trace((ri));                                        \
		errorx("[rank:%ld node:%c iter:%ld] " fmt, ri->rank,           \
		       my_node_name, ri->iteration, ##__VA_ARGS__)             \
	} while (0)

// Like error(), but with context about the current rank.
#define ERROR(ri, fmt, ...)                                                    \
	do {                                                                   \
		debug_dump_trace((ri));                                        \
		error("[rank:%ld node:%c iter:%ld] " fmt, ri->rank,            \
		      my_node_name, ri->iteration, ##__VA_ARGS__)              \
	} while (0)

#define INSIST(ri, condition)                                                  \
	do {                                                                   \
		if (!(condition)) {                                            \
			ERRORX(ri, "condition failed: " #condition);           \
		}                                                              \
	} while (0)

#define INSIST_EQ(ri, value, testvalue, fmt)                                   \
	do {                                                                   \
		typeof(value) __cached_value = (value);                        \
		typeof(value) __cached_testvalue = (testvalue);                \
		if ((__cached_value) != (__cached_testvalue)) {                \
			ERRORX(ri,                                             \
			       "(" #value ") != (" #testvalue ")  (" fmt       \
			       ") != (" fmt ")",                               \
			       __cached_value, __cached_testvalue);            \
		}                                                              \
	} while (0)

#define INSIST_FI_EQ(ri, value, testvalue)                                     \
	do {                                                                   \
		int insist_ret;                                                \
		debug_trace_push((ri), __LINE__, __func__, __FILE__, #value);  \
		if ((insist_ret = (value)) != (testvalue)) {                   \
			ERRORX((ri),                                           \
			       #value " == %d (fi_errno: %s), != " #testvalue  \
				      " (%d)",                                 \
			       insist_ret, fi_strerror(labs(insist_ret)),      \
			       (testvalue));                                   \
		}                                                              \
		debug_trace_pop((ri));                                         \
	} while (0)

#define TRACE(ri, func)                                                        \
	do {                                                                   \
		debug_trace_push((ri), __LINE__, __func__, __FILE__, #func);   \
		func;                                                          \
		debug_trace_pop((ri));                                         \
	} while (0)

struct rank_info;
void debug_trace_push(struct rank_info *ri, int line, const char *func,
		      const char *file, const char *debugstr);
void debug_trace_pop(struct rank_info *ri);
void debug_dump_trace(struct rank_info *ri);

extern bool debug_quiet;
extern bool verbose;
