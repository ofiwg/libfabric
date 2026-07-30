/*
 * Copyright (C) 2026 Cornelis Networks.
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

#ifndef _FI_PROV_OPX_SHM_CACHE_H_
#define _FI_PROV_OPX_SHM_CACHE_H_

/*
 * opx_shm_cache - run a node-global computation exactly once, cached in shm.
 *
 * Some OPX setup work is expensive (e.g. shelling out to query the HFI driver
 * version) yet produces a result that is identical for every rank on a node.
 * At high PPN having every rank repeat it is wasteful.  This utility lets one
 * process per node do the work while the rest read the cached answer.
 *
 * Ranks rendezvous through a POSIX shm segment whose name the caller supplies
 * (/dev/shm is node-local, so a job-wide name is per-node-per-job).  The
 * O_CREAT|O_EXCL "winner" runs the caller's producer once and publishes the
 * result; every other rank ("loser", EEXIST) attaches and copies it out.
 *
 * The caller owns naming and fallback policy: opx_shm_cache_get() returns
 * OPX_SHM_CACHE_SUCCESS with the result filled, or OPX_SHM_CACHE_ERROR on any
 * shm/mmap error or timeout, leaving the caller to decide what a missing result
 * means.  The utility never re-runs the producer on the error path and never
 * blocks indefinitely.
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OPX_SHM_CACHE_SUCCESS (0)
#define OPX_SHM_CACHE_ERROR   (-1)

/* Upper bound on a cached payload, in bytes.  Keeps a segment to one small page. */
#define OPX_SHM_CACHE_MAX_RESULT_SIZE (256)

/* Upper bound on the caller's validity token (a NUL-terminated string). */
#define OPX_SHM_CACHE_MAX_TOKEN_LEN (128)

/*
 * A producer fills up to result_len bytes at result and returns 0 on success.
 * It runs at most once per node, in the winner, after the segment is mapped.
 */
typedef int (*opx_shm_cache_produce_fn)(void *result, size_t result_len, void *context);

/*
 * Build the canonical segment name "/opx." <purpose> "." <jobkey> for a caller.
 *
 * The <purpose> namespaces one init-time use from another so two distinct uses
 * can never map the same segment (which would alias unrelated payloads).  Put
 * only the *use* in the name; put any validity/version discriminators in the
 * token passed to opx_shm_cache_get() instead.  Both components are sanitized so
 * arbitrary input stays legal in a POSIX shm name.  Returns 0, or -1 if buf is
 * too small for the full name (a truncated name is never emitted).
 */
int opx_shm_cache_name(char *buf, size_t len, const char *purpose, const char *jobkey);

/*
 * Best-effort normal-exit cleanup of a published segment.  The winner (the
 * process for which opx_shm_cache_get() set *is_winner) records the segment
 * name and calls this at teardown so a normally-exiting job does not leave the
 * segment behind in /dev/shm.  A NULL or empty name is a no-op, so losers can
 * call it unconditionally.  Crash/abort paths are handled separately by the
 * dead-creator reclaim in opx_shm_cache_get(); this only covers orderly exit.
 */
void opx_shm_cache_unlink(const char *name);

/*
 * Obtain a node-global cached result, running produce() at most once per node.
 *
 *   name      - POSIX shm name (must start with '/'), ideally from
 *               opx_shm_cache_name() so each use gets its own "/opx.<purpose>."
 *               namespace.  The name identifies the *use*; the token below, not
 *               the name, distinguishes stale from valid data.
 *   token     - NUL-terminated string identifying everything the cached result
 *               depends on (driver version, generation, etc.).  A loser whose
 *               token differs from the stored one treats the segment as stale,
 *               reclaims it, and recomputes.  Pass "" if the name alone is
 *               sufficient.  Bounded by OPX_SHM_CACHE_MAX_TOKEN_LEN.
 *   result    - buffer that receives result_len bytes on success.
 *   result_len- payload size, <= OPX_SHM_CACHE_MAX_RESULT_SIZE.
 *   produce   - winner-only callback that fills result_len bytes.
 *   context   - opaque, passed through to produce.
 *   is_winner - if non-NULL, set to 1 iff this process created the segment (so
 *               the caller can record name for a best-effort shm_unlink() at
 *               teardown; the crash path is handled here via the pid sweep).
 *
 * Returns OPX_SHM_CACHE_SUCCESS with result filled, or OPX_SHM_CACHE_ERROR on
 * any shm/mmap error, producer failure, or timeout.  Never re-runs produce on
 * the error path and never blocks past OPX_SHM_CACHE_TIMEOUT_SEC.
 */
int opx_shm_cache_get(const char *name, const char *token, void *result, size_t result_len,
		      opx_shm_cache_produce_fn produce, void *context, int *is_winner);

#ifdef __cplusplus
}
#endif

#endif /* _FI_PROV_OPX_SHM_CACHE_H_ */
