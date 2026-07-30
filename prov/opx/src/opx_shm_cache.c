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

#include "config.h"

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <assert.h>

#include "ofi_atom.h"
#include "opx_shm_cache.h"

/*
 * Naming convention for opx_shm_cache segments: "/opx." <purpose> "." <jobkey>
 * (see opx_shm_cache_name()).  The <purpose> component is what keeps distinct
 * init-time uses of this utility from ever sharing a segment.
 */
#define OPX_SHM_CACHE_NAME_PREFIX "/opx."

/* Winner stamps this ("OPXC") before publishing; a loser rejects a segment that lacks it. */
#define OPX_SHM_CACHE_MAGIC (0x4f505843u)

/* Bound loser waits so a winner that dies before publishing cannot hang startup. */
#define OPX_SHM_CACHE_TIMEOUT_SEC (10.0)

/* Loser backoff bounds: start at 1us, cap at 1ms. */
#define OPX_SHM_CACHE_BACKOFF_MIN_NS (1000UL)
#define OPX_SHM_CACHE_BACKOFF_MAX_NS (1000000UL)

/*
 * No explicit "initializing" state is needed: O_CREAT|O_EXCL is the claim, so a
 * loser waits for state == INITIALIZED (acquire) before trusting the payload
 * rather than trusting mere segment existence.
 */
enum opx_shm_cache_state {
	OPX_SHM_CACHE_UNINIT	  = 0, /* ftruncate() zero-fills to this */
	OPX_SHM_CACHE_INITIALIZED = 1
};

/*
 * On-shm layout version, owned by this utility (independent of any caller
 * token).  Bump on ANY change to struct opx_shm_cache_hdr so a segment written
 * by a differently-built libfabric is rejected instead of misinterpreted.
 */
#define OPX_SHM_CACHE_LAYOUT_VERSION (1u)

struct opx_shm_cache_hdr {
	ofi_atomic64_t state;				   /* enum opx_shm_cache_state, published with release */
	ofi_atomic64_t creator_pid;			   /* winner pid; 0 until stamped, probed for liveness */
	uint32_t       magic;				   /* OPX_SHM_CACHE_MAGIC, written before the release */
	uint32_t       layout_version;			   /* OPX_SHM_CACHE_LAYOUT_VERSION; guards the header ABI */
	uint32_t       hdr_size;			   /* sizeof(struct opx_shm_cache_hdr); guards the header ABI */
	uint32_t       result_len;			   /* payload bytes; a loser rejects a mismatch */
	uint32_t       token_len;			   /* validity-token length including NUL */
	uint32_t       pad_;				   /* keep the payload that follows 8-byte aligned */
	char	       token[OPX_SHM_CACHE_MAX_TOKEN_LEN]; /* caller validity token (see get()) */
	/* result_len payload bytes follow, 8-byte aligned by the header layout */
};

/*
 * Tripwire: adding or removing a header field changes this sum and fails the
 * build, forcing a matching OPX_SHM_CACHE_LAYOUT_VERSION bump so a
 * differently-built peer rejects the segment instead of reading fields at the
 * wrong offsets.  The expected size is expressed from the member types (not a
 * magic constant) so it stays correct across debug builds, where each
 * ofi_atomic64_t carries an extra debug field.
 */
static_assert(sizeof(struct opx_shm_cache_hdr) ==
		      2 * sizeof(ofi_atomic64_t) + 6 * sizeof(uint32_t) + OPX_SHM_CACHE_MAX_TOKEN_LEN,
	      "opx_shm_cache_hdr layout changed: bump OPX_SHM_CACHE_LAYOUT_VERSION");

/*
 * Cross-process sharing is only valid if the 64-bit atomic is lock-free; the
 * fallback ofi_atomic embeds a process-local spinlock that would break it.  The
 * #ifdef makes this a no-op unless the C11 stdatomic macros are visible (the
 * path the OPX build uses), so it can never spuriously fail an unrelated build.
 */
#ifdef ATOMIC_LLONG_LOCK_FREE
static_assert(ATOMIC_LLONG_LOCK_FREE == 2 && ATOMIC_LONG_LOCK_FREE == 2,
	      "opx_shm_cache requires a lock-free 64-bit atomic for cross-process use");
#endif

static double opx_shm_cache_elapsed(const struct timespec *start)
{
	struct timespec now;
	clock_gettime(CLOCK_MONOTONIC, &now);
	return (double) (now.tv_sec - start->tv_sec) + (double) (now.tv_nsec - start->tv_nsec) / 1e9;
}

/* Sleep the current backoff interval, then ramp toward the cap (avoids a busy spin). */
static void opx_shm_cache_backoff(uint64_t *sleep_ns)
{
	struct timespec ts;
	ts.tv_sec  = 0;
	ts.tv_nsec = (long) *sleep_ns;
	nanosleep(&ts, NULL);

	*sleep_ns <<= 1;
	if (*sleep_ns > OPX_SHM_CACHE_BACKOFF_MAX_NS) {
		*sleep_ns = OPX_SHM_CACHE_BACKOFF_MAX_NS;
	}
}

/* Map any char outside [0-9A-Za-z._] to '_' so src is safe in a POSIX shm name (empty -> "none"). */
static void opx_shm_cache_sanitize(char *dst, size_t len, const char *src)
{
	if (len == 0) {
		return;
	}
	if (src == NULL || src[0] == '\0') {
		snprintf(dst, len, "none");
		return;
	}
	size_t i = 0;
	for (; src[i] != '\0' && i < len - 1; i++) {
		char c = src[i];
		if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '.' ||
		    c == '_') {
			dst[i] = c;
		} else {
			dst[i] = '_';
		}
	}
	dst[i] = '\0';
}

/*
 * ABI/ownership gate: is this a segment written by a compatible build of this
 * utility for a same-sized payload?  Verified before any other header field is
 * trusted (a differing hdr_size means the field offsets themselves are suspect).
 * A failure here is a foreign or incompatible segment we must not touch.
 */
static int opx_shm_cache_hdr_abi_ok(const struct opx_shm_cache_hdr *hdr, size_t result_len)
{
	return hdr->magic == OPX_SHM_CACHE_MAGIC && hdr->layout_version == OPX_SHM_CACHE_LAYOUT_VERSION &&
	       hdr->hdr_size == (uint32_t) sizeof(struct opx_shm_cache_hdr) && hdr->result_len == (uint32_t) result_len;
}

/*
 * Does an ABI-compatible segment answer OUR question?  A token mismatch means
 * it caches a result for different inputs (e.g. an in-place driver upgrade) and
 * is stale, so it should be reclaimed and recomputed rather than trusted.
 */
static int opx_shm_cache_token_match(const struct opx_shm_cache_hdr *hdr, const char *token, size_t token_len)
{
	return hdr->token_len == (uint32_t) token_len && memcmp(hdr->token, token, token_len) == 0;
}

/*
 * Drop a stale/abandoned segment so a subsequent create can recompute.  Only
 * unlink if the name still resolves to the inode the caller mapped, so we avoid
 * destroying a fresh segment another rank already recreated under this name.
 *
 * The inode guard is best-effort: fstat() checks the fd's inode but shm_unlink()
 * acts on the name, so a peer can recreate the name between the two calls and we
 * could unlink the newer inode.  This only costs an extra producer run (the new
 * winner's peers get ENOENT and re-create), so exactly-once degrades to
 * at-least-once under stale-token contention -- bounded by the shared deadline
 * and safe because the producer is idempotent and the result is node-global.
 * Returns 1 (caller should retry creation) regardless of who won the race.
 */
static int opx_shm_cache_reclaim(const char *name, ino_t mapped_ino)
{
	int probe_fd = shm_open(name, O_RDWR, 0600);
	if (probe_fd >= 0) {
		struct stat cur;
		if (fstat(probe_fd, &cur) == 0 && cur.st_ino == mapped_ino) {
			shm_unlink(name);
		}
		close(probe_fd);
	}
	return 1;
}

int opx_shm_cache_name(char *buf, size_t len, const char *purpose, const char *jobkey)
{
	if (buf == NULL || len == 0) {
		return -1;
	}

	char purpose_s[OPX_SHM_CACHE_MAX_TOKEN_LEN];
	char jobkey_s[OPX_SHM_CACHE_MAX_TOKEN_LEN];
	opx_shm_cache_sanitize(purpose_s, sizeof(purpose_s), purpose);
	opx_shm_cache_sanitize(jobkey_s, sizeof(jobkey_s), jobkey);

	int n = snprintf(buf, len, OPX_SHM_CACHE_NAME_PREFIX "%s.%s", purpose_s, jobkey_s);
	if (n < 0 || (size_t) n >= len) {
		return -1;
	}
	return 0;
}

void opx_shm_cache_unlink(const char *name)
{
	if (name && name[0] != '\0') {
		shm_unlink(name);
	}
}

int opx_shm_cache_get(const char *name, const char *token, void *result, size_t result_len,
		      opx_shm_cache_produce_fn produce, void *context, int *is_winner)
{
	if (is_winner) {
		*is_winner = 0;
	}

	if (name == NULL || name[0] != '/' || token == NULL || result == NULL || produce == NULL || result_len == 0 ||
	    result_len > OPX_SHM_CACHE_MAX_RESULT_SIZE) {
		return OPX_SHM_CACHE_ERROR;
	}

	const size_t token_len = strlen(token) + 1;
	if (token_len > OPX_SHM_CACHE_MAX_TOKEN_LEN) {
		return OPX_SHM_CACHE_ERROR;
	}

	const size_t segment_size = sizeof(struct opx_shm_cache_hdr) + result_len;

	struct timespec start;
	clock_gettime(CLOCK_MONOTONIC, &start);
	uint64_t backoff_ns = OPX_SHM_CACHE_BACKOFF_MIN_NS;

	/*
	 * Create-or-attach loop, bounded by OPX_SHM_CACHE_TIMEOUT_SEC.  Each pass
	 * becomes the winner (O_CREAT|O_EXCL), attaches as a loser (EEXIST) and
	 * reads the result, or reclaims a segment whose creator died before
	 * publishing and retries creation.
	 */
	while (opx_shm_cache_elapsed(&start) < OPX_SHM_CACHE_TIMEOUT_SEC) {
		/*
		 * O_CREAT|O_EXCL gives exactly one creator per node.  Do NOT
		 * shm_unlink() first: a blind unlink on this shared name could
		 * destroy a live segment other ranks are attaching to.
		 */
		int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fd >= 0) {
			/* WINNER: size, map, run the single producer, publish. */
			if (ftruncate(fd, (off_t) segment_size) != 0) {
				close(fd);
				shm_unlink(name);
				return OPX_SHM_CACHE_ERROR;
			}

			void *addr = mmap(NULL, segment_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
			close(fd);
			if (addr == MAP_FAILED) {
				shm_unlink(name);
				return OPX_SHM_CACHE_ERROR;
			}
			struct opx_shm_cache_hdr *hdr	  = (struct opx_shm_cache_hdr *) addr;
			void			 *payload = (void *) (hdr + 1);

			/* Stamp pid BEFORE the slow producer so a loser can detect a mid-run death. */
			ofi_atomic_store_explicit64(&hdr->creator_pid, (int64_t) getpid(), memory_order_release);

			if (produce(payload, result_len, context) != 0) {
				/* Producer failed: drop the segment so a loser retries instead of waiting. */
				munmap(addr, segment_size);
				shm_unlink(name);
				return OPX_SHM_CACHE_ERROR;
			}

			memcpy(result, payload, result_len);
			hdr->result_len	    = (uint32_t) result_len;
			hdr->layout_version = OPX_SHM_CACHE_LAYOUT_VERSION;
			hdr->hdr_size	    = (uint32_t) sizeof(struct opx_shm_cache_hdr);
			hdr->token_len	    = (uint32_t) token_len;
			memcpy(hdr->token, token, token_len);
			hdr->magic = OPX_SHM_CACHE_MAGIC;

			/* Release-store state last so a loser observing INITIALIZED also sees the header+payload. */
			ofi_atomic_store_explicit64(&hdr->state, OPX_SHM_CACHE_INITIALIZED, memory_order_release);

			if (is_winner) {
				*is_winner = 1;
			}

			munmap(addr, segment_size);
			return OPX_SHM_CACHE_SUCCESS;
		}

		if (errno != EEXIST) {
			/* Unexpected creation failure. */
			return OPX_SHM_CACHE_ERROR;
		}

		/* LOSER: attach to the existing segment. */
		fd = shm_open(name, O_RDWR, 0600);
		if (fd < 0) {
			/* Name vanished (another rank reclaimed a stale segment) -> retry creation. */
			if (errno == ENOENT) {
				opx_shm_cache_backoff(&backoff_ns);
				continue;
			}
			return OPX_SHM_CACHE_ERROR;
		}

		/* Wait until the winner has ftruncate()d it, so we never map a short segment. */
		struct stat st;
		while (opx_shm_cache_elapsed(&start) < OPX_SHM_CACHE_TIMEOUT_SEC) {
			if (fstat(fd, &st) == 0 && (size_t) st.st_size >= segment_size) {
				break;
			}
			opx_shm_cache_backoff(&backoff_ns);
		}
		if (fstat(fd, &st) != 0 || (size_t) st.st_size < segment_size) {
			close(fd);
			return OPX_SHM_CACHE_ERROR;
		}
		const ino_t mapped_ino = st.st_ino;

		void *addr = mmap(NULL, segment_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		close(fd);
		if (addr == MAP_FAILED) {
			return OPX_SHM_CACHE_ERROR;
		}
		struct opx_shm_cache_hdr *hdr	  = (struct opx_shm_cache_hdr *) addr;
		void			 *payload = (void *) (hdr + 1);

		/*
		 * Wait for the winner to publish, probing its liveness meanwhile: if
		 * it died before publishing (kill(pid,0) == ESRCH), reclaim the stale
		 * segment and retry creation so the producer still runs once per node.
		 */
		int reclaimed = 0;
		while (opx_shm_cache_elapsed(&start) < OPX_SHM_CACHE_TIMEOUT_SEC) {
			if (ofi_atomic_load_explicit64(&hdr->state, memory_order_acquire) ==
			    OPX_SHM_CACHE_INITIALIZED) {
				/* A foreign/incompatible segment sharing the name is not ours to reclaim. */
				if (!opx_shm_cache_hdr_abi_ok(hdr, result_len)) {
					munmap(addr, segment_size);
					return OPX_SHM_CACHE_ERROR;
				}
				/* Stale (different token): drop it and recompute for our inputs. */
				if (!opx_shm_cache_token_match(hdr, token, token_len)) {
					if (opx_shm_cache_reclaim(name, mapped_ino)) {
						reclaimed = 1;
						break;
					}
					munmap(addr, segment_size);
					return OPX_SHM_CACHE_ERROR;
				}
				memcpy(result, payload, result_len);
				munmap(addr, segment_size);
				return OPX_SHM_CACHE_SUCCESS;
			}

			int64_t pid = ofi_atomic_load_explicit64(&hdr->creator_pid, memory_order_acquire);
			if (pid > 0 && kill((pid_t) pid, 0) != 0 && errno == ESRCH) {
				/*
				 * The creator may have published then exited between the state
				 * load above and this probe.  Re-check state before reclaiming
				 * so we consume a valid result instead of destroying a good
				 * cache, applying the same ownership rules as the main wait
				 * path above.
				 */
				if (ofi_atomic_load_explicit64(&hdr->state, memory_order_acquire) ==
				    OPX_SHM_CACHE_INITIALIZED) {
					/* Published: a foreign segment is not ours to reclaim. */
					if (!opx_shm_cache_hdr_abi_ok(hdr, result_len)) {
						munmap(addr, segment_size);
						return OPX_SHM_CACHE_ERROR;
					}
					/* Ours and current: consume it. */
					if (opx_shm_cache_token_match(hdr, token, token_len)) {
						memcpy(result, payload, result_len);
						munmap(addr, segment_size);
						return OPX_SHM_CACHE_SUCCESS;
					}
					/* Ours but stale: fall through to reclaim below. */
				}

				/* Creator died before publishing a usable result: reclaim and retry. */
				(void) opx_shm_cache_reclaim(name, mapped_ino);
				reclaimed = 1;
				break;
			}

			opx_shm_cache_backoff(&backoff_ns);
		}

		munmap(addr, segment_size);
		if (reclaimed) {
			opx_shm_cache_backoff(&backoff_ns);
			continue;
		}

		/* Creator alive but slow past the deadline. */
		return OPX_SHM_CACHE_ERROR;
	}

	return OPX_SHM_CACHE_ERROR;
}
