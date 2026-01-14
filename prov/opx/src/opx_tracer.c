/*
 * Copyright (C) 2024-2026 Cornelis Networks.
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

#ifdef OPX_TRACER_ENABLED

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <time.h>
#include <errno.h>
#include <pthread.h>

#include "rdma/opx/opx_tracer.h"

/* Minimum buffer size to prevent degenerate cases */
#define OPX_TRACE_MIN_BUFFER_SIZE 4096

/* Maximum buffer size (1GB) to prevent excessive memory usage */
#define OPX_TRACE_MAX_BUFFER_SIZE (1UL << 30)

/* Maximum event name length for bounds checking */
#define OPX_TRACE_MAX_EVENT_NAME_LEN 128

/* Buffer size for trace output filename */
#define OPX_TRACE_FILENAME_BUF_LEN 512

struct opx_trace_global			 opx_trace_global     = {0};
__thread struct opx_trace_thread_buffer *opx_trace_tls_buffer = NULL;

/* pthread_once control for thread-safe initialization */
static pthread_once_t opx_trace_init_once = PTHREAD_ONCE_INIT;

/* Thread-local storage key for automatic cleanup on thread exit */
static pthread_key_t opx_trace_tls_key;
static bool	     opx_trace_tls_key_created = false;

/*
 * Thread-local storage destructor - called automatically when a thread exits
 * This ensures trace buffers are flushed and cleaned up even if the thread
 * doesn't explicitly call opx_trace_fini_thread_buffer()
 *
 * Note: We check opx_trace_global.initialized to avoid double-free if
 * opx_trace_global_fini() was called before thread exit.
 */
static void opx_trace_tls_destructor(void *buf)
{
	if (buf && opx_trace_global.initialized) {
		opx_trace_fini_thread_buffer((struct opx_trace_thread_buffer *) buf);
	}
}

/*
 * Initialize timing subsystem based on timer mode
 *
 * TSC mode (OPX_TRACE_TIMER_TSC):
 *   - Calibrates TSC frequency by measuring cycles over a known time interval
 *   - tsc_frequency = cycles per second
 *   - start_tsc = RDTSC value at trace start
 *
 * Gettime mode (OPX_TRACE_TIMER_GETTIME):
 *   - No calibration needed - timestamps are already nanoseconds
 *   - tsc_frequency = 1,000,000,000 (1 tick = 1 nanosecond)
 *   - start_tsc = CLOCK_MONOTONIC nanoseconds at trace start
 */
static void opx_trace_calibrate_tsc(void)
{
	struct timespec ts_mono_start, ts_real;

	/* Get wall-clock time for cross-host alignment (always needed) */
	clock_gettime(CLOCK_REALTIME, &ts_real);
	clock_gettime(CLOCK_MONOTONIC, &ts_mono_start);

	opx_trace_global.start_time_ns	   = ts_mono_start.tv_sec * 1000000000ULL + ts_mono_start.tv_nsec;
	opx_trace_global.start_realtime_ns = ts_real.tv_sec * 1000000000ULL + ts_real.tv_nsec;

#if OPX_TRACE_TIMER_MODE == OPX_TRACE_TIMER_GETTIME
	/*
	 * Gettime mode: timestamps are already nanoseconds
	 * Set frequency to 1e9 so conversion is: ns = (ts - start_tsc) * 1e9 / 1e9 = ts - start_tsc
	 */
	opx_trace_global.tsc_frequency = 1000000000ULL;
	opx_trace_global.start_tsc     = opx_trace_global.start_time_ns;

	fprintf(stderr, "OPX Tracer [%d]: using clock_gettime timer mode\n", getpid());
#else
	/*
	 * TSC mode: calibrate by measuring cycles over a known time interval
	 */
	struct timespec ts_mono_end;
	uint64_t	tsc_start, tsc_end;

	tsc_start = opx_trace_rdtsc();
	usleep(10000); /* 10ms calibration interval */
	clock_gettime(CLOCK_MONOTONIC, &ts_mono_end);
	tsc_end = opx_trace_rdtsc();

	uint64_t ns_elapsed = (ts_mono_end.tv_sec - ts_mono_start.tv_sec) * 1000000000ULL +
			      (ts_mono_end.tv_nsec - ts_mono_start.tv_nsec);
	uint64_t tsc_elapsed = tsc_end - tsc_start;

	/* Guard against division by zero (e.g., if usleep was interrupted) */
	if (ns_elapsed == 0) {
		ns_elapsed = 1;
		fprintf(stderr, "OPX Tracer [%d]: WARNING - TSC calibration: zero elapsed time, using fallback\n",
			getpid());
	}

	opx_trace_global.tsc_frequency = (tsc_elapsed * 1000000000ULL) / ns_elapsed;
	opx_trace_global.start_tsc     = tsc_start;

	fprintf(stderr, "OPX Tracer [%d]: using TSC timer mode (frequency=%lu Hz)\n", getpid(),
		(unsigned long) opx_trace_global.tsc_frequency);
#endif
}

static void opx_trace_parse_filter(const char *filter_str)
{
	if (!filter_str || !*filter_str) {
		return;
	}

	char *filter_copy = strdup(filter_str);
	if (!filter_copy) {
		fprintf(stderr, "OPX Tracer [%d]: WARNING - failed to allocate filter string, using defaults\n",
			getpid());
		return;
	}

	char *saveptr1 = NULL;
	char *token    = strtok_r(filter_copy, ",", &saveptr1);

	while (token) {
		char *colon = strchr(token, ':');
		if (colon) {
			*colon		      = '\0';
			const char *cat_name  = token;
			const char *level_str = colon + 1;

			enum opx_trace_filter_level level = OPX_TRACE_FILTER_ALL;
			if (strcasecmp(level_str, "NONE") == 0) {
				level = OPX_TRACE_FILTER_NONE;
			} else if (strcasecmp(level_str, "COMPLETE") == 0) {
				level = OPX_TRACE_FILTER_COMPLETE;
			}

			for (int i = 0; i < OPX_TRACE_NUM_CATEGORIES; i++) {
				if (strcasecmp(cat_name, opx_trace_category_names[i]) == 0) {
					opx_trace_global.runtime_filters[i] = level;
					break;
				}
			}
		}
		token = strtok_r(NULL, ",", &saveptr1);
	}

	free(filter_copy);
}

/*
 * Internal initialization function called via pthread_once
 * This ensures thread-safe single initialization even with concurrent calls
 */
static void opx_trace_do_global_init(void)
{
	char *env_path = getenv("FI_OPX_TRACER_OUT_PATH");
	fprintf(stderr, "OPX Tracer [%d]: FI_OPX_TRACER_OUT_PATH=%s\n", getpid(), env_path ? env_path : "(null)");
	if (!env_path || !*env_path) {
		fprintf(stderr, "OPX Tracer [%d]: disabled (no output path)\n", getpid());
		return;
	}

	strncpy(opx_trace_global.output_path, env_path, OPX_TRACE_MAX_PATH_LEN - 1);
	opx_trace_global.output_path[OPX_TRACE_MAX_PATH_LEN - 1] = '\0';

	char *env_size = getenv("FI_OPX_TRACER_BUFFER_SIZE");
	if (env_size && *env_size) {
		char	     *endptr = NULL;
		unsigned long val    = strtoul(env_size, &endptr, 10);
		if (endptr && *endptr != '\0') {
			fprintf(stderr, "OPX Tracer [%d]: invalid buffer size '%s', using default\n", getpid(),
				env_size);
			opx_trace_global.buffer_size = OPX_TRACE_DEFAULT_BUFFER_SIZE;
		} else if (val == 0) {
			fprintf(stderr, "OPX Tracer [%d]: buffer size cannot be zero, using default\n", getpid());
			opx_trace_global.buffer_size = OPX_TRACE_DEFAULT_BUFFER_SIZE;
		} else if (val > OPX_TRACE_MAX_BUFFER_SIZE) {
			fprintf(stderr, "OPX Tracer [%d]: buffer size %lu exceeds maximum %lu, using maximum\n",
				getpid(), val, OPX_TRACE_MAX_BUFFER_SIZE);
			opx_trace_global.buffer_size = OPX_TRACE_MAX_BUFFER_SIZE;
		} else if (val < OPX_TRACE_MIN_BUFFER_SIZE) {
			fprintf(stderr, "OPX Tracer [%d]: buffer size %lu below minimum %d, using minimum\n", getpid(),
				val, OPX_TRACE_MIN_BUFFER_SIZE);
			opx_trace_global.buffer_size = OPX_TRACE_MIN_BUFFER_SIZE;
		} else {
			opx_trace_global.buffer_size = (size_t) val;
		}
	} else {
		opx_trace_global.buffer_size = OPX_TRACE_DEFAULT_BUFFER_SIZE;
	}

	for (int i = 0; i < OPX_TRACE_NUM_CATEGORIES; i++) {
		opx_trace_global.runtime_filters[i] = OPX_TRACE_FILTER_ALL;
	}

	char *env_filter = getenv("FI_OPX_TRACER_FILTER");
	if (env_filter && *env_filter) {
		opx_trace_parse_filter(env_filter);
	}

	opx_trace_global.pid = (uint32_t) getpid();
	if (gethostname(opx_trace_global.hostname, OPX_TRACE_HOSTNAME_LEN) != 0) {
		strncpy(opx_trace_global.hostname, "unknown", OPX_TRACE_HOSTNAME_LEN - 1);
	}
	opx_trace_global.hostname[OPX_TRACE_HOSTNAME_LEN - 1] = '\0';

	opx_trace_global.enabled_categories = 0;
#ifdef OPX_TRACER_TX
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_TX;
#endif
#ifdef OPX_TRACER_RX
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_RX;
#endif
#ifdef OPX_TRACER_RELI
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_RELI;
#endif
#ifdef OPX_TRACER_SDMA
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_SDMA;
#endif
#ifdef OPX_TRACER_PIO
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_PIO;
#endif
#ifdef OPX_TRACER_CQ
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_CQ;
#endif
#ifdef OPX_TRACER_MR
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_MR;
#endif
#ifdef OPX_TRACER_TID
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_TID;
#endif
#ifdef OPX_TRACER_PROGRESS
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_PROGRESS;
#endif
#ifdef OPX_TRACER_HMEM
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_HMEM;
#endif
#ifdef OPX_TRACER_ATOMIC
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_ATOMIC;
#endif
#ifdef OPX_TRACER_RMA
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_RMA;
#endif
#ifdef OPX_TRACER_LOCK
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_LOCK;
#endif
	opx_trace_global.enabled_categories |= OPX_TRACE_CAT_INTERNAL;

	fprintf(stderr, "OPX Tracer [%d]: enabled_categories=0x%x, output_path=%s\n", getpid(),
		opx_trace_global.enabled_categories, opx_trace_global.output_path);

	opx_trace_calibrate_tsc();

	/* Create TLS key for automatic thread cleanup */
	if (pthread_key_create(&opx_trace_tls_key, opx_trace_tls_destructor) == 0) {
		opx_trace_tls_key_created = true;
	} else {
		fprintf(stderr,
			"OPX Tracer [%d]: WARNING - failed to create TLS key, "
			"thread cleanup may leak memory\n",
			getpid());
	}

	opx_trace_global.initialized = true;
	fprintf(stderr, "OPX Tracer [%d]: initialized successfully\n", getpid());
}

/*
 * Public initialization function - thread-safe via pthread_once
 */
void opx_trace_global_init(void)
{
	pthread_once(&opx_trace_init_once, opx_trace_do_global_init);
}

void opx_trace_global_fini(void)
{
	if (!opx_trace_global.initialized) {
		return;
	}

	/*
	 * Must be called after all tracing threads have stopped.
	 * Mark uninitialized first to prevent TLS destructor double-free.
	 */
	opx_trace_global.initialized = false;

	if (opx_trace_tls_buffer) {
		opx_trace_fini_thread_buffer(opx_trace_tls_buffer);
		opx_trace_tls_buffer = NULL;
		/*
		 * Clear the TLS key value to prevent the destructor from
		 * being called with a stale pointer on thread exit.
		 */
		if (opx_trace_tls_key_created) {
			pthread_setspecific(opx_trace_tls_key, NULL);
		}
	}

	/* Clean up TLS key */
	if (opx_trace_tls_key_created) {
		pthread_key_delete(opx_trace_tls_key);
		opx_trace_tls_key_created = false;
	}
}

static void opx_trace_write_file_header(struct opx_trace_thread_buffer *buf)
{
	struct opx_trace_file_header hdr = {0};

	hdr.magic	      = OPX_TRACE_MAGIC;
	hdr.version	      = OPX_TRACE_VERSION;
	hdr.pid		      = opx_trace_global.pid;
	hdr.tid		      = buf->tid;
	hdr.tsc_frequency     = opx_trace_global.tsc_frequency;
	hdr.start_time_ns     = opx_trace_global.start_time_ns;
	hdr.start_realtime_ns = opx_trace_global.start_realtime_ns;
	hdr.start_tsc	      = opx_trace_global.start_tsc;
	strncpy(hdr.hostname, opx_trace_global.hostname, OPX_TRACE_HOSTNAME_LEN);
	hdr.enabled_categories = opx_trace_global.enabled_categories;
	hdr.num_event_strings  = OPX_TRACE_EVENT_COUNT;
	hdr.timer_mode	       = OPX_TRACE_TIMER_MODE;
	hdr.self_lid	       = opx_trace_global.self_lid;

	ssize_t written = write(buf->output_fd, &hdr, sizeof(hdr));
	if (written != sizeof(hdr)) {
		fprintf(stderr, "OPX Tracer: failed to write file header: %s\n", strerror(errno));
		return;
	}

	for (int i = 0; i < OPX_TRACE_EVENT_COUNT; i++) {
		const char *name = opx_trace_event_names[i];
		if (!name) {
			name = "UNKNOWN";
		}
		size_t name_len = strlen(name);

		/* Bounds check to prevent buffer overflow */
		if (name_len >= OPX_TRACE_MAX_EVENT_NAME_LEN) {
			name_len = OPX_TRACE_MAX_EVENT_NAME_LEN - 1;
		}

		struct opx_trace_string_def str_def = {0};
		str_def.hdr = opx_trace_make_header(OPX_TRACE_RECORD_STRING_DEF, OPX_TRACE_STATUS_INSTANT, 0, 0, 0);
		str_def.id  = (uint16_t) i;
		str_def.len = (uint16_t) name_len;

		written = write(buf->output_fd, &str_def, sizeof(str_def));
		if (written != sizeof(str_def)) {
			fprintf(stderr, "OPX Tracer: failed to write string def: %s\n", strerror(errno));
			return;
		}

		size_t padded_len = (name_len + 8) & ~((size_t) 7);
		/* Use bounded buffer size based on max name length */
		char padded_name[OPX_TRACE_MAX_EVENT_NAME_LEN + 8] = {0};
		memcpy(padded_name, name, name_len);

		written = write(buf->output_fd, padded_name, padded_len);
		if (written != (ssize_t) padded_len) {
			fprintf(stderr, "OPX Tracer: failed to write string: %s\n", strerror(errno));
			return;
		}
	}

	buf->header_written = true;
}

struct opx_trace_thread_buffer *opx_trace_init_thread_buffer(void)
{
	if (!opx_trace_global.initialized) {
		return NULL;
	}

	struct opx_trace_thread_buffer *buf = calloc(1, sizeof(struct opx_trace_thread_buffer));
	if (!buf) {
		fprintf(stderr, "OPX Tracer [%d]: FATAL - failed to allocate thread buffer struct, aborting\n",
			getpid());
		abort();
	}

	buf->buffer = malloc(opx_trace_global.buffer_size);
	if (!buf->buffer) {
		fprintf(stderr, "OPX Tracer [%d]: FATAL - failed to allocate %zu byte trace buffer, aborting\n",
			getpid(), opx_trace_global.buffer_size);
		free(buf);
		abort();
	}

	buf->buffer_size    = opx_trace_global.buffer_size;
	buf->write_offset   = 0;
	buf->tid	    = (uint32_t) syscall(SYS_gettid);
	buf->flush_count    = 0;
	buf->blocked_ns	    = 0;
	buf->header_written = false;

	char filename[OPX_TRACE_FILENAME_BUF_LEN];
	snprintf(filename, sizeof(filename), "%s/pid%u_tid%u.bin", opx_trace_global.output_path, opx_trace_global.pid,
		 buf->tid);

	buf->output_fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
	if (buf->output_fd < 0) {
		fprintf(stderr, "OPX Tracer [%d]: FATAL - failed to open %s: %s, aborting\n", getpid(), filename,
			strerror(errno));
		free(buf->buffer);
		free(buf);
		abort();
	}

	opx_trace_write_file_header(buf);

	/* Register with TLS key for automatic cleanup on thread exit */
	if (opx_trace_tls_key_created) {
		pthread_setspecific(opx_trace_tls_key, buf);
	}

	return buf;
}

void opx_trace_fini_thread_buffer(struct opx_trace_thread_buffer *buf)
{
	if (!buf) {
		return;
	}

	if (buf->write_offset > 0) {
		opx_trace_flush_buffer(buf);
	}

	if (buf->output_fd >= 0) {
		close(buf->output_fd);
	}

	free(buf->buffer);
	free(buf);
}

void opx_trace_flush_buffer(struct opx_trace_thread_buffer *buf)
{
	if (!buf || buf->write_offset == 0) {
		return;
	}

	buf->flush_count++;

	/*
	 * Write the TRACER_FLUSH BEGIN event to the buffer BEFORE flushing to disk.
	 * This ensures the BEGIN event is included in the buffered data.
	 *
	 * Note: We write directly to the buffer here rather than calling
	 * opx_trace_write_event() to avoid recursion through opx_trace_ensure_space().
	 */
	uint64_t flush_start = opx_trace_rdtsc();

	if (buf->write_offset + OPX_TRACE_EVENT_SIZE <= buf->buffer_size) {
		struct opx_trace_event *ev = (struct opx_trace_event *) (buf->buffer + buf->write_offset);

		ev->hdr = opx_trace_make_header(OPX_TRACE_RECORD_EVENT, OPX_TRACE_STATUS_BEGIN, OPX_TRACE_CAT_INTERNAL,
						OPX_TRACE_EVENT_FLUSH, 2);
		ev->timestamp = flush_start;
		ev->args[0]   = buf->write_offset; /* bytes_to_flush */
		ev->args[1]   = buf->flush_count;

		buf->write_offset += OPX_TRACE_EVENT_SIZE;
	}

	/* Flush the buffer (including the BEGIN event) to disk */
	ssize_t written = write(buf->output_fd, buf->buffer, buf->write_offset);

	uint64_t flush_end   = opx_trace_rdtsc();
	uint64_t flush_ticks = flush_end - flush_start;

	/*
	 * Handle partial writes properly:
	 * - On full success: reset offset to 0
	 * - On partial write: move unwritten data to start of buffer
	 * - On error: log and reset (data loss, but prevents infinite loop)
	 */
	if (written == (ssize_t) buf->write_offset) {
		/* Full success */
		buf->write_offset = 0;
	} else if (written > 0) {
		/* Partial write - preserve unwritten data */
		size_t remaining = buf->write_offset - (size_t) written;
		memmove(buf->buffer, buf->buffer + written, remaining);
		buf->write_offset = remaining;
		fprintf(stderr,
			"OPX Tracer: partial flush write: %zd of %zu bytes, "
			"%zu bytes remaining\n",
			written, buf->write_offset + (size_t) written, remaining);
	} else {
		/* Error - log and reset to prevent infinite loop */
		fprintf(stderr, "OPX Tracer: flush write failed: %s (losing %zu bytes)\n", strerror(errno),
			buf->write_offset);
		buf->write_offset = 0;
	}

	/* Avoid integer overflow in blocked_ns calculation */
	if (opx_trace_global.tsc_frequency > 0) {
		/* Split calculation to avoid overflow: (ticks / freq) * 1e9 + remainder */
		uint64_t secs	   = flush_ticks / opx_trace_global.tsc_frequency;
		uint64_t remainder = flush_ticks % opx_trace_global.tsc_frequency;
		buf->blocked_ns += secs * 1000000000ULL + (remainder * 1000000000ULL) / opx_trace_global.tsc_frequency;
	}

	/*
	 * Write the TRACER_FLUSH END event directly to disk with a second write().
	 * This captures the actual flush duration (flush_ticks) and ensures the
	 * END event is persisted even on the final flush before file close.
	 *
	 * We accept the minor inefficiency of a second write() call because:
	 * 1. Flushes are infrequent (only when buffer fills or at shutdown)
	 * 2. The file I/O path is inherently slow anyway
	 * 3. Capturing accurate flush duration is valuable for performance analysis
	 */
	struct opx_trace_event end_event;
	end_event.hdr	    = opx_trace_make_header(OPX_TRACE_RECORD_EVENT, OPX_TRACE_STATUS_END_SUCCESS,
						    OPX_TRACE_CAT_INTERNAL, OPX_TRACE_EVENT_FLUSH, 2);
	end_event.timestamp = flush_end;
	end_event.args[0]   = flush_ticks; /* duration in TSC ticks */
	end_event.args[1]   = buf->flush_count;

	written = write(buf->output_fd, &end_event, sizeof(end_event));
	if (written != sizeof(end_event)) {
		fprintf(stderr, "OPX Tracer: failed to write flush END event: %s\n", strerror(errno));
	}
}

void opx_trace_emit_event(uint16_t category, enum opx_trace_status status, enum opx_trace_event_id event_id,
			  uint64_t arg0, uint64_t arg1)
{
	struct opx_trace_thread_buffer *buf = opx_trace_get_buffer();
	opx_trace_write_event(buf, category, status, event_id, arg0, arg1);
}

void opx_trace_set_self_lid(uint32_t lid)
{
	/*
	 * Use atomic store to prevent data race if multiple endpoints
	 * are created concurrently from different threads. The self_lid
	 * should be the same for all endpoints on the same HFI port, so
	 * concurrent writes with the same value are benign.
	 */
	__atomic_store_n(&opx_trace_global.self_lid, lid, __ATOMIC_RELEASE);
}

#endif /* OPX_TRACER_ENABLED */
