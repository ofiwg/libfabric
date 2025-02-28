/*
 * Copyright (C) 2021-2024 by Cornelis Networks.
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
#include <assert.h>
#include <stdlib.h>
#include <numa.h>
#include <inttypes.h>
#include <sys/sysinfo.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>

#include "rdma/fabric.h" // only for 'fi_addr_t' ... which is a typedef to uint64_t
#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx_hfi1_inlines.h"
#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_eq.h"
#include "rdma/opx/fi_opx_hfi1_sdma.h"
#include "ofi_mem.h"

#include "fi_opx_hfi_select.h"
#include "rdma/opx/opx_hfi1_cn5000.h"

#include "rdma/opx/opx_tracer.h"

#define OPX_SHM_ENABLE_ON      1
#define OPX_SHM_ENABLE_OFF     0
#define OPX_SHM_ENABLE_DEFAULT OPX_SHM_ENABLE_ON

#define BYTE2DWORD_SHIFT (2)

/* RZV messages under FI_OPX_TID_MSG_MISALIGNED_THRESHOLD
 * will fallback to Eager Ring (not TID) RZV if the
 * buffer is misaligned more than FI_OPX_TID_MISALIGNED_THRESHOLD
 */

/* Number of bytes allowed to be misaligned on small TID RZV
 * FI_OPX_TID_MISALIGNED_THRESHOLD is arbitrary, based on testing.
 *  - 64 bytes
 */
#ifndef FI_OPX_TID_MISALIGNED_THRESHOLD
#define FI_OPX_TID_MISALIGNED_THRESHOLD 64
#endif

/* Maximum message size that falls back on misaligned buffers
 * FI_OPX_TID_MSG_MISALIGNED_THRESHOLD is arbitrary, based on testing.
 *  - 15 pages (64K)
 */
#ifndef FI_OPX_TID_MSG_MISALIGNED_THRESHOLD
#define FI_OPX_TID_MSG_MISALIGNED_THRESHOLD (15 * OPX_HFI1_TID_PAGESIZE)
#endif

/*
 * Return the NUMA node id where the process is currently running.
 */
static int opx_get_current_proc_location()
{
	int core_id, node_id;

	core_id = sched_getcpu();
	if (core_id < 0) {
		return -EINVAL;
	}

	node_id = numa_node_of_cpu(core_id);
	if (node_id < 0) {
		return -EINVAL;
	}

	return node_id;
}

static int opx_get_current_proc_core()
{
	int core_id;
	core_id = sched_getcpu();
	if (core_id < 0) {
		return -EINVAL;
	}
	return core_id;
}

static inline uint64_t fi_opx_hfi1_header_count_to_poll_mask(uint64_t rcvhdrq_cnt)
{
	/* For optimization, the fi_opx_hfi1_poll_once() function uses a mask to wrap around the end of the
	** ring buffer.  To compute the mask, multiply the number of entries in the ring buffer by the sizeof
	** one entry.  Since the count is 0-based, subtract 1 from the value of
	** /sys/module/hfi1/parameters/rcvhdrcnt, which is set in the hfi1 module parms and
	** will not change at runtime
	*/
	return (rcvhdrq_cnt - 1) * 32;
}

// Used by fi_opx_hfi1_context_open as a convenience.
static int opx_open_hfi_and_context(struct _hfi_ctrl **ctrl, struct fi_opx_hfi1_context_internal *internal,
				    uuid_t unique_job_key, int hfi_unit_number)
{
	int fd;

	int	     port = opx_select_port_index(hfi_unit_number) + 1;
	unsigned int user_version;
	fd = opx_hfi1_wrapper_context_open(internal, hfi_unit_number, port, 0, &user_version);
	FI_DBG_TRACE(&fi_opx_provider, FI_LOG_FABRIC, "opx_hfi_context_open fd %d.\n", fd);
	if (fd < 0) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to open HFI unit %d.\n", hfi_unit_number);
		fd = -1;
	} else {
		memset(&internal->user_info, 0, sizeof(internal->user_info));

		internal->user_info.userversion = user_version;

		/* do not share hfi contexts */
		internal->user_info.subctxt_id	= 0;
		internal->user_info.subctxt_cnt = 0;

		memcpy(internal->user_info.uuid, unique_job_key, sizeof(internal->user_info.uuid));

		*ctrl = opx_hfi1_wrapper_userinit(fd, internal, hfi_unit_number, port);
		if (!*ctrl) {
			opx_hfi_context_close(fd);
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to open a context on HFI unit %d.\n",
				hfi_unit_number);
			fd = -1;
		} else {
			assert((*ctrl)->__hfi_pg_sz == OPX_HFI1_TID_PAGESIZE);
		}
	}
	return fd;
}

void opx_reset_context(struct fi_opx_ep *opx_ep)
{
	fi_opx_compiler_msync_writes();
	opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INIT_VAL(OPX_HFI1_TYPE);
	opx_ep->rx->state.hdrq.head    = 0;

	if (opx_hfi1_wrapper_reset_context(opx_ep->hfi)) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Send context reset failed: %d.\n", errno);
		abort();
	}
	FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Send context reset successfully.\n");

	opx_ep->tx->pio_state->fill_counter   = 0;
	opx_ep->tx->pio_state->scb_head_index = 0;
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
	FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	fi_opx_hfi1_poll_sdma_completion(opx_ep);
	opx_hfi1_sdma_process_pending(opx_ep);
}

static int fi_opx_get_daos_hfi_rank_inst(const uint8_t hfi_unit_number, const uint32_t rank)
{
	struct fi_opx_daos_hfi_rank_key key;
	struct fi_opx_daos_hfi_rank    *hfi_rank = NULL;

	memset(&key, 0, sizeof(key));
	key.hfi_unit_number = hfi_unit_number;
	key.rank	    = rank;

	HASH_FIND(hh, fi_opx_global.daos_hfi_rank_hashmap, &key, sizeof(key), hfi_rank);

	if (hfi_rank) {
		hfi_rank->instance++;

		FI_INFO(fi_opx_global.prov, FI_LOG_EP_DATA, "HFI %d assigned rank %d again: %d.\n", key.hfi_unit_number,
			key.rank, hfi_rank->instance);
	} else {
		int rc __attribute__((unused));
		rc = posix_memalign((void **) &hfi_rank, 32, sizeof(*hfi_rank));
		assert(rc == 0);

		hfi_rank->key	   = key;
		hfi_rank->instance = 0;
		HASH_ADD(hh, fi_opx_global.daos_hfi_rank_hashmap, key, sizeof(hfi_rank->key), hfi_rank);

		FI_INFO(fi_opx_global.prov, FI_LOG_EP_DATA, "HFI %d assigned rank %d entry created.\n",
			key.hfi_unit_number, key.rank);
	}

	return hfi_rank->instance;
}

void process_hfi_lookup(int hfi_unit, unsigned int lid)
{
	struct fi_opx_hfi_local_lookup_key key;
	key.lid					   = (opx_lid_t) lid;
	struct fi_opx_hfi_local_lookup *hfi_lookup = NULL;

	HASH_FIND(hh, fi_opx_global.hfi_local_info.hfi_local_lookup_hashmap, &key, sizeof(key), hfi_lookup);

	if (hfi_lookup) {
		hfi_lookup->instance++;

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "HFI %d LID 0x%x again: %d.\n", hfi_lookup->hfi_unit,
			     key.lid, hfi_lookup->instance);
	} else {
		int rc __attribute__((unused));
		rc = posix_memalign((void **) &hfi_lookup, 32, sizeof(*hfi_lookup));
		assert(rc == 0);

		if (!hfi_lookup) {
			FI_WARN(&fi_opx_provider, FI_LOG_EP_DATA, "Unable to allocate HFI lookup entry.\n");
			return;
		}
		hfi_lookup->key	     = key;
		hfi_lookup->hfi_unit = hfi_unit;
		hfi_lookup->instance = 0;
		HASH_ADD(hh, fi_opx_global.hfi_local_info.hfi_local_lookup_hashmap, key, sizeof(hfi_lookup->key),
			 hfi_lookup);

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "HFI %hhu LID 0x%hx entry created.\n",
			     hfi_lookup->hfi_unit, key.lid);
	}
}

void fi_opx_init_hfi_lookup()
{
	int hfi_unit  = 0;
	int hfi_units = MIN(opx_hfi_get_num_units(), FI_OPX_MAX_HFIS);

	if (hfi_units == 0) {
		FI_WARN(&fi_opx_provider, FI_LOG_EP_DATA, "No HFI units found.\n");
		return;
	}

	int shm_enable_env;
	if (fi_param_get_bool(fi_opx_global.prov, "shm_enable", &shm_enable_env) != FI_SUCCESS) {
		FI_INFO(fi_opx_global.prov, FI_LOG_EP_DATA, "shm_enable param not specified\n");
		shm_enable_env = OPX_SHM_ENABLE_DEFAULT;
	}

	if (shm_enable_env == OPX_SHM_ENABLE_ON) {
		for (hfi_unit = 0; hfi_unit < hfi_units; hfi_unit++) {
			int num_ports = opx_hfi_get_num_ports(hfi_unit);
			for (int port = OPX_MIN_PORT; port <= num_ports; port++) {
				opx_lid_t lid = opx_hfi_get_port_lid(hfi_unit, port);
				if (lid > 0) {
					if (lid == fi_opx_global.hfi_local_info.lid) {
						/* This is the HFI and port to be used by the EP.  No need to add to the
						 * HFI hashmap.
						 */
						FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
							     "EP HFI %d LID 0x%x found.\n", hfi_unit, lid);
						continue;
					} else {
						process_hfi_lookup(hfi_unit, lid);
					}
				} else {
					FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
						"No LID found for HFI unit %d of %d units and port %d of %d ports: ret = %d, %s.\n",
						hfi_unit, hfi_units, port, num_ports, lid, strerror(errno));
				}
			}
		}
	}
}

/*
 * Open a context on the first HFI that shares our process' NUMA node.
 * If no HFI shares our NUMA node, grab the first active HFI.
 */
struct fi_opx_hfi1_context *fi_opx_hfi1_context_open(struct fid_ep *ep, uuid_t unique_job_key)
{
	struct fi_opx_ep *opx_ep		= (ep == NULL) ? NULL : container_of(ep, struct fi_opx_ep, ep_fid);
	int		  fd			= -1;
	int		  hfi_unit_number	= -6;
	int		  hfi_context_rank	= -1;
	int		  hfi_context_rank_inst = -1;
	const int	  numa_node_id		= opx_get_current_proc_location();
	const int	  core_id		= opx_get_current_proc_core();
	const int	  hfi_count		= opx_hfi_get_num_units();
	int		  hfi_candidates[FI_OPX_MAX_HFIS];
	int		  hfi_distances[FI_OPX_MAX_HFIS];
	int		  hfi_freectxs[FI_OPX_MAX_HFIS];
	int		  hfi_candidates_count = 0;
	int		  hfi_candidate_index  = -1;
	struct _hfi_ctrl *ctrl		       = NULL;
	bool		  use_default_logic    = true;
	int		  dirfd		       = -1;

	memset(hfi_candidates, 0, sizeof(*hfi_candidates) * FI_OPX_MAX_HFIS);
	memset(hfi_distances, 0, sizeof(*hfi_distances) * FI_OPX_MAX_HFIS);
	memset(hfi_freectxs, 0, sizeof(*hfi_freectxs) * FI_OPX_MAX_HFIS);

	struct fi_opx_hfi1_context_internal *internal = calloc(1, sizeof(struct fi_opx_hfi1_context_internal));
	if (!internal) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
			"Error: Memory allocation failure for fi_opx_hfi_context_internal.\n");
		return NULL;
	}

	struct fi_opx_hfi1_context *context = &internal->context;

	/*
	 * Force cpu affinity if desired. Normally you would let the
	 * job scheduler (such as mpirun) handle this.
	 */
	int force_cpuaffinity = 0;
	fi_param_get_bool(fi_opx_global.prov, "force_cpuaffinity", &force_cpuaffinity);
	if (force_cpuaffinity) {
		const int cpu_id = sched_getcpu();
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(cpu_id, &cpuset);
		if (sched_setaffinity(0, sizeof(cpuset), &cpuset)) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to force cpu affinity. %s\n", strerror(errno));
		}
	}

	/*
	 * open the hfi1 context
	 */
	context->fd    = -1;
	internal->ctrl = NULL;

	// If FI_OPX_HFI_SELECT is specified, skip all this and
	// use its value as the selected hfi unit.
	char *env = NULL;
	if (FI_SUCCESS == fi_param_get_str(&fi_opx_provider, "hfi_select", &env)) {
		struct hfi_selector selector = {0};
		use_default_logic	     = false;

		int selectors, matched;
		selectors = matched = 0;
		const char *s;
		for (s = env; *s != '\0';) {
			s = hfi_selector_next(s, &selector);
			if (!s) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error occurred parsing HFI selector string \"%s\"\n", env);
				goto ctxt_open_err;
			}

			if (selector.type == HFI_SELECTOR_DEFAULT) {
				use_default_logic = true;
				break;
			}

			if (selector.unit >= hfi_count) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error: selector unit %d >= number of HFIs %d\n", selector.unit, hfi_count);
				goto ctxt_open_err;
			} else if (!opx_hfi_get_unit_active(selector.unit)) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Error: selected unit %d is not active\n",
					selector.unit);
				goto ctxt_open_err;
			}

			if (selector.type == HFI_SELECTOR_FIXED) {
				hfi_unit_number = selector.unit;
				matched++;
				break;
			} else if (selector.type == HFI_SELECTOR_MAPBY) {
				if (selector.mapby.type == HFI_SELECTOR_MAPBY_NUMA) {
					int max_numa = numa_max_node();
					if (selector.mapby.rangeS > max_numa) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"Error: mapby numa %d > numa_max_node %d\n",
							selector.mapby.rangeS, max_numa);
						goto ctxt_open_err;
					}

					if (selector.mapby.rangeE > max_numa) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"mapby numa end of range %d > numa_max_node %d\n",
							selector.mapby.rangeE, max_numa);
						goto ctxt_open_err;
					}

					if (selector.mapby.rangeS <= numa_node_id &&
					    selector.mapby.rangeE >= numa_node_id) {
						hfi_unit_number = selector.unit;
						matched++;
						break;
					}
				} else if (selector.mapby.type == HFI_SELECTOR_MAPBY_CORE) {
					int max_core = get_nprocs();
					if (selector.mapby.rangeS > max_core) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"Error: mapby core %d > nprocs %d\n", selector.mapby.rangeS,
							max_core);
						goto ctxt_open_err;
					}
					if (selector.mapby.rangeE > max_core) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"mapby core end of range %d > nprocs %d\n",
							selector.mapby.rangeE, max_core);
						goto ctxt_open_err;
					}
					if (selector.mapby.rangeS <= core_id && selector.mapby.rangeE >= core_id) {
						hfi_unit_number = selector.unit;
						matched++;
						break;
					}
				} else {
					FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Error: unsupported mapby type %d\n",
						selector.mapby.type);
					goto ctxt_open_err;
				}
			} else {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Error: unsupported selector type %d\n",
					selector.type);
				goto ctxt_open_err;
			}
			selectors++;
		}

		(void) selectors;

		if (!use_default_logic) {
			if (!matched) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "No HFI selectors matched.\n");
				goto ctxt_open_err;
			}

			hfi_candidates[0]    = hfi_unit_number;
			hfi_distances[0]     = 0;
			hfi_candidates_count = 1;
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
				"User-specified HFI selection set to %d. Skipping HFI selection algorithm \n",
				hfi_unit_number);

			fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key, hfi_unit_number);
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Opened fd %u\n", fd);
			if (fd < 0) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to open user-specified HFI.\n");
				goto ctxt_open_err;
			}
		}

	} else if (opx_ep && opx_ep->common_info->src_addr &&
		   ((union fi_opx_addr *) (opx_ep->common_info->src_addr))->hfi1_unit != opx_default_addr.hfi1_unit) {
		union fi_opx_addr addr;
		use_default_logic = false;
		/*
		 * DAOS Persistent Address Support:
		 * No Context Resource Management Framework is supported by OPX to enable
		 * acquiring a context with attributes that exactly match the specified
		 * source address.
		 *
		 * Therefore, treat the source address as an opaque ID and extract the
		 * essential data required to create a context that at least maps to the
		 * same HFI and HFI port (Note, the assigned LID is unchanged unless modified
		 * by the OPA FM).
		 */
		memset(&addr, 0, sizeof(addr));
		memcpy(&addr.fi, opx_ep->common_info->src_addr, opx_ep->common_info->src_addrlen);

		uint32_t uid = addr.lid << 8 | addr.endpoint_id;

		if (addr.lid != UINT32_MAX) {
			hfi_context_rank = uid;
		}
		hfi_unit_number	     = addr.hfi1_unit;
		hfi_candidates[0]    = hfi_unit_number;
		hfi_distances[0]     = 0;
		hfi_candidates_count = 1;

		if (hfi_context_rank != -1) {
			hfi_context_rank_inst = fi_opx_get_daos_hfi_rank_inst(hfi_unit_number, hfi_context_rank);

			FI_DBG_TRACE(
				&fi_opx_provider, FI_LOG_FABRIC,
				"Application-specified HFI selection set to %d rank %d.%d. Skipping HFI selection algorithm\n",
				hfi_unit_number, hfi_context_rank, hfi_context_rank_inst);
		} else {
			FI_DBG_TRACE(
				&fi_opx_provider, FI_LOG_FABRIC,
				"Application-specified HFI selection set to %d. Skipping HFI selection algorithm\n",
				hfi_unit_number);
		}

		fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key, hfi_unit_number);
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Opened fd %u\n", fd);
		if (fd < 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to open application-specified HFI.\n");
			goto ctxt_open_err;
		}
	}
	if (use_default_logic) {
		/* Select the best HFI to open a context on */
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Found HFIs = %d\n", hfi_count);

		if (hfi_count == 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "FATAL: detected no HFIs, cannot continue\n");
			goto ctxt_open_err;
		}

		else if (hfi_count == 1) {
			if (opx_hfi_get_unit_active(0) > 0) {
				// Only 1 HFI, populate the candidate list and continue.
				FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
					"Detected one HFI and it has active ports, selected it\n");
				hfi_candidates[0]    = 0;
				hfi_distances[0]     = 0;
				hfi_candidates_count = 1;
			} else {
				// No active ports, we're done here.
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"FATAL: HFI has no active ports, cannot continue\n");
				goto ctxt_open_err;
			}

		} else {
			// Lock on the opx class directory path so that HFI selection based on distance and
			// number of free credits available is atomic. This is to avoid the situation where several
			// processes go to read the number of free contexts available in each HFI at the same time
			// and choose the same HFi with the smallest load as well as closest to the corresponding
			// process. If the processes of selection and then context openning is atomic here, this
			// situation is avoided and hfi selection should be evenly balanced.
			if ((dirfd = open(OPX_CLASS_DIR_PATH, O_RDONLY)) == -1) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Failed to open %s: %s for flock use.\n",
					OPX_CLASS_DIR_PATH, strerror(errno));
				goto ctxt_open_err;
			}

			if (flock(dirfd, LOCK_EX) == -1) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Flock exclusive lock failure: %s\n",
					strerror(errno));
				close(dirfd);
				goto ctxt_open_err;
			}

			// The system has multiple HFIs. Sort them by distance from
			// this process. HFIs with same distance are sorted by number of
			// free contexts available.
			int hfi_n, hfi_d, hfi_f;
			for (int i = 0; i < hfi_count; i++) {
				if (opx_hfi_get_unit_active(i) > 0) {
					hfi_n = opx_hfi_sysfs_unit_read_node_s64(i);
					hfi_d = numa_distance(hfi_n, numa_node_id);
					hfi_f = opx_hfi_get_num_free_contexts(i);
					FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
						"HFI unit %d in numa node %d has a distance of %d from this pid with"
						" %d free contexts available.\n",
						i, hfi_n, hfi_d, hfi_f);
					hfi_candidates[hfi_candidates_count] = i;
					hfi_distances[hfi_candidates_count]  = hfi_d;
					hfi_freectxs[hfi_candidates_count]   = hfi_f;
					int j				     = hfi_candidates_count;
					// Bubble the new HFI up till the list is sorted by distance
					// and then by number of free contexts. Yes, this is lame but
					// the practical matter is that there will never be so many HFIs
					// on a single system that a real insertion sort is justified.
					while (j > 0 && ((hfi_distances[j - 1] > hfi_distances[j]) ||
							 ((hfi_distances[j - 1] == hfi_distances[j]) &&
							  (hfi_freectxs[j - 1] < hfi_freectxs[j])))) {
						int t1		      = hfi_distances[j - 1];
						int t2		      = hfi_candidates[j - 1];
						int t3		      = hfi_freectxs[j - 1];
						hfi_distances[j - 1]  = hfi_distances[j];
						hfi_candidates[j - 1] = hfi_candidates[j];
						hfi_freectxs[j - 1]   = hfi_freectxs[j];
						hfi_distances[j]      = t1;
						hfi_candidates[j]     = t2;
						hfi_freectxs[j]	      = t3;
						j--;
					}
					hfi_candidates_count++;
				}
			}
		}

		// At this point we have a list of HFIs, sorted by distance from this pid (and by unit # as an implied
		// key). HFIs that have the same distance are sorted by number of free contexts available. Pick the
		// closest HFI that has the smallest load (largest number of free contexts). If we fail to open that
		// HFI, try another one at the same distance but potentially under a heavier load. If that fails, we
		// will try HFIs that are further away.
		int lower  = 0;
		int higher = 0;
		do {
			// Find the set of HFIs at this distance. Again, no attempt is
			// made to make this fast.
			higher = lower + 1;
			while (higher < hfi_candidates_count && hfi_distances[higher] == hfi_distances[lower]) {
				higher++;
			}

			// Select the hfi that is under the smallest load. All
			// hfis from [lower, higher) are sorted by number of free contexts
			// available with lower having the most contexts free.
			int range	    = higher - lower;
			hfi_candidate_index = lower;
			hfi_unit_number	    = hfi_candidates[hfi_candidate_index];

			fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key, hfi_unit_number);
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Opened fd %u\n", fd);
			int t = range;
			while (fd < 0 && t-- > 1) {
				hfi_candidate_index++;
				if (hfi_candidate_index >= higher) {
					hfi_candidate_index = lower;
				}
				hfi_unit_number = hfi_candidates[hfi_candidate_index];
				fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key, hfi_unit_number);
				FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Opened fd %u\n", fd);
			}

			// If we still haven't successfully chosen an HFI,
			// try HFIs that are further away.
			lower = higher;
		} while (fd < 0 && lower < hfi_candidates_count);

		if (dirfd != -1) {
			if (flock(dirfd, LOCK_UN) == -1) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Flock unlock failure: %s\n", strerror(errno));
				close(dirfd);

				if (fd >= 0) {
					opx_hfi_context_close(fd);
				}
				goto ctxt_open_err;
			}
			close(dirfd);
		}

		if (fd < 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"FATAL: Found %d active HFI device%s, unable to open %s.\n", hfi_candidates_count,
				(hfi_candidates_count > 1) ? "s" : "",
				(hfi_candidates_count > 1) ? "any of them" : "it");
			goto ctxt_open_err;
		}
	}

	FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
		"Selected HFI is %d; caller NUMA domain is %d; HFI NUMA domain is %" PRId64 "\n", hfi_unit_number,
		numa_node_id, opx_hfi_sysfs_unit_read_node_s64(hfi_unit_number));

	// Alert user if the final choice wasn't optimal.
	if (opx_hfi_sysfs_unit_read_node_s64(hfi_unit_number) != numa_node_id) {
		if (hfi_count == 1) {
			/* No choice, not worth a warning */
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
				"Selected HFI is %d. It does not appear to be local to this pid's numa domain which is %d\n",
				hfi_unit_number, numa_node_id);
		} else {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Selected HFI is %d. It does not appear to be local to this pid's numa domain which is %d\n",
				hfi_unit_number, numa_node_id);
		}
	} else {
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Selected HFI unit %d in the same numa node as this pid.\n",
			hfi_unit_number);
	}

	context->fd    = fd;
	internal->ctrl = ctrl; /* memory was allocated during opx_open_hfi_and_context() -> opx_hfi_userinit() */
	context->ctrl  = ctrl; /* TODO? move required fields ctrl -> context? */

	opx_lid_t lid = 0;
	lid	      = opx_hfi_get_port_lid(ctrl->__hfi_unit, ctrl->__hfi_port);
	FI_DBG_TRACE(&fi_opx_provider, FI_LOG_FABRIC, "lid = %d ctrl->__hfi_unit %u, ctrl->__hfi_port %u\n", lid,
		     ctrl->__hfi_unit, ctrl->__hfi_port);

	assert(lid > 0);

	uint64_t gid_hi, gid_lo;
	int	 rc __attribute__((unused)) = -1;
	rc = opx_hfi_get_port_gid(ctrl->__hfi_unit, ctrl->__hfi_port, &gid_hi, &gid_lo);
	assert(rc != -1);

	context->hfi_unit	     = ctrl->__hfi_unit;
	context->hfi_port	     = ctrl->__hfi_port;
	context->lid		     = (opx_lid_t) lid;
	context->gid_hi		     = gid_hi;
	context->gid_lo		     = gid_lo;
	context->daos_info.rank	     = hfi_context_rank;
	context->daos_info.rank_inst = hfi_context_rank_inst;

	// If a user wants an HPC job ran on a non-default Service Level,
	// they set FI_OPX_SL to the deseried SL with will then determine the SC and VL
	int user_sl = -1;
	if (fi_param_get_int(fi_opx_global.prov, "sl", &user_sl) == FI_SUCCESS) {
		if ((user_sl >= 0) && (user_sl <= 31)) {
			context->sl = user_sl;
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
				"Detected user specfied ENV FI_OPX_SL, so set the service level to %d\n", user_sl);
		} else {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Error: User specfied an env FI_OPX_SL.  Valid data is an positive integer 0 - 31 (Default is 0).  User specified %d.  Using default value of %d instead\n",
				user_sl, FI_OPX_HFI1_SL_DEFAULT);
			context->sl = FI_OPX_HFI1_SL_DEFAULT;
		}
	} else {
		context->sl = FI_OPX_HFI1_SL_DEFAULT;
	}

	rc = opx_hfi_get_port_sl2sc(ctrl->__hfi_unit, ctrl->__hfi_port, context->sl);
	if (rc < 0) {
		context->sc = FI_OPX_HFI1_SC_DEFAULT;
	} else {
		context->sc = rc;
	}

	rc = opx_hfi_get_port_sc2vl(ctrl->__hfi_unit, ctrl->__hfi_port, context->sc);
	if (rc < 0) {
		context->vl = FI_OPX_HFI1_VL_DEFAULT;
	} else {
		context->vl = rc;
	}

	if (context->sc == FI_OPX_HFI1_SC_ADMIN || context->vl == FI_OPX_HFI1_VL_ADMIN) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
			"Detected user set ENV FI_OPX_SL of %ld, which has translated to admin-level Service class (SC=%ld) and/or admin-level Virtual Lane(VL=%ld), which is invalid for user traffic.  Using default values instead\n",
			context->sl, context->sc, context->vl);
		context->sl = FI_OPX_HFI1_SL_DEFAULT;
		context->sc = FI_OPX_HFI1_SC_DEFAULT;
		context->vl = FI_OPX_HFI1_VL_DEFAULT;
	}

	if (context->vl > 7) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA,
			"VL is > 7, this may not be supported.  SL=%ld SC=%ld VL=%ld\n", context->sl, context->sc,
			context->vl);
	}

	context->mtu = opx_hfi_get_port_vl2mtu(ctrl->__hfi_unit, ctrl->__hfi_port, context->vl);
	assert(context->mtu >= 0);

	// If a user wants an HPC job ran on a non-default Partition key,
	// they set FI_OPX_PKEY env to specify it (Same behavior as PSM2_PKEY)
	int user_pkey = -1;
	if (fi_param_get_int(fi_opx_global.prov, "pkey", &user_pkey) == FI_SUCCESS) {
		if (user_pkey < 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Detected user specified FI_OPX_PKEY of %d (0x%x), which is an invalid value.\n",
				user_pkey, user_pkey);
			if (fd >= 0) {
				opx_hfi_context_close(fd);
			}
			goto ctxt_open_err;
		}
		rc = opx_hfi1_wrapper_set_pkey(internal, user_pkey);
		if (rc) {
			fprintf(stderr,
				"Detected user specified FI_OPX_PKEY of 0x%x, but got internal driver error on set. This pkey is likely not registered/valid.\n",
				user_pkey);
			if (fd >= 0) {
				opx_hfi_context_close(fd);
			}
			goto ctxt_open_err;
		} else {
			context->pkey = user_pkey;
			FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
				"Detected user specfied ENV FI_OPX_PKEY, so set partition key to 0x%x\n", user_pkey);
		}
	} else {
		int default_pkey = opx_hfi_get_port_index2pkey(context->hfi_unit, context->hfi_port, 0);
		if (default_pkey < 0) {
			fprintf(stderr,
				"Unable to get default Pkey. Please specify a different Pkey using FI_OPX_PKEY\n");
			if (fd >= 0) {
				opx_hfi_context_close(fd);
			}
			goto ctxt_open_err;
		}
		rc = opx_hfi1_wrapper_set_pkey(internal, default_pkey);
		if (rc) {
			fprintf(stderr,
				"Error in setting default Pkey %#x. Please specify a different Pkey using FI_OPX_PKEY\n",
				default_pkey);
			if (fd >= 0) {
				opx_hfi_context_close(fd);
			}
			goto ctxt_open_err;
		} else {
			context->pkey = default_pkey;
		}
	}

	FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Service Level: SL=%ld SC=%ld VL=%ld PKEY=0x%lx MTU=%d\n", context->sl,
		context->sc, context->vl, context->pkey, context->mtu);

	/*
	 * Initialize the hfi tx context
	 */

	const struct hfi1_base_info *base_info = &ctrl->base_info;
	const struct hfi1_ctxt_info *ctxt_info = &ctrl->ctxt_info;

	context->bthqp	   = (uint8_t) base_info->bthqp;
	context->jkey	   = base_info->jkey;
	context->send_ctxt = ctxt_info->send_ctxt;

	OPX_OPEN_BAR(context->hfi_unit);
	context->info.pio.scb_sop_first =
		OPX_HFI1_INIT_PIO_SOP(context->send_ctxt, (volatile uint64_t *) (ptrdiff_t) base_info->pio_bufbase_sop);
	context->info.pio.scb_first =
		OPX_HFI1_INIT_PIO(context->send_ctxt, (volatile uint64_t *) (ptrdiff_t) base_info->pio_bufbase);
	context->info.pio.credits_addr = (volatile uint64_t *) (ptrdiff_t) base_info->sc_credits_addr;

	const uint64_t credit_return	       = *(context->info.pio.credits_addr);
	context->state.pio.free_counter_shadow = (uint16_t) (credit_return & 0x00000000000007FFul);
	context->state.pio.fill_counter	       = 0;
	context->state.pio.scb_head_index      = 0;
	context->state.pio.credits_total = ctxt_info->credits; /* yeah, yeah .. THIS field is static, but there was an
								  unused halfword at this spot, so .... */

	/* move to domain ? */
	uint8_t i;
	for (i = 0; i < 32; ++i) {
		rc = opx_hfi_get_port_sl2sc(ctrl->__hfi_unit, ctrl->__hfi_port, i);

		if (rc < 0) {
			context->sl2sc[i] = FI_OPX_HFI1_SC_DEFAULT;
		} else {
			context->sl2sc[i] = rc;
		}

		rc = opx_hfi_get_port_sc2vl(ctrl->__hfi_unit, ctrl->__hfi_port, i);
		if (rc < 0) {
			context->sc2vl[i] = FI_OPX_HFI1_VL_DEFAULT;
		}
		context->sc2vl[i] = rc;
	}

	// TODO: There is a bug in the driver that does not properly handle all
	//       queue entries in use at once. As a temporary workaround, pretend
	//       there is one less entry than there actually is.
	context->info.sdma.queue_size	     = ctxt_info->sdma_ring_size - 1;
	context->info.sdma.available_counter = context->info.sdma.queue_size;
	context->info.sdma.fill_index	     = 0;
	context->info.sdma.done_index	     = 0;
	context->info.sdma.completion_queue  = (struct hfi1_sdma_comp_entry *) base_info->sdma_comp_bufbase;
	assert(context->info.sdma.queue_size <= FI_OPX_HFI1_SDMA_MAX_COMP_INDEX);
	memset(context->info.sdma.queued_entries, 0, sizeof(context->info.sdma.queued_entries));

	/*
	 * initialize the hfi rx context
	 */

	context->info.rxe.id	       = ctrl->ctxt_info.ctxt;
	context->info.rxe.hdrq.rhf_off = (ctxt_info->rcvhdrq_entsize - 8) >> BYTE2DWORD_SHIFT;

	/* hardware registers */
	volatile uint64_t *uregbase =
		OPX_HFI1_INIT_UREGS(ctrl->ctxt_info.ctxt, (volatile uint64_t *) (uintptr_t) base_info->user_regbase);
	context->info.rxe.hdrq.head_register = (volatile uint64_t *) &uregbase[ur_rcvhdrhead];
	context->info.rxe.egrq.head_register = (volatile uint64_t *) &uregbase[ur_rcvegrindexhead];
	volatile uint64_t *tidflowtable	     = (volatile uint64_t *) &uregbase[ur_rcvtidflowtable];

#ifndef NDEBUG
	uint64_t debug_value = OPX_HFI1_BAR_LOAD(&uregbase[ur_rcvhdrtail]);
	FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "&uregbase[ur_rcvhdrtail]       %p = %#16.16lX \n",
	       &uregbase[ur_rcvhdrtail], debug_value);
	debug_value = OPX_HFI1_BAR_LOAD(&uregbase[ur_rcvhdrhead]);
	FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "&uregbase[ur_rcvhdrhead]       %p = %#16.16lX \n",
	       &uregbase[ur_rcvhdrhead], debug_value);
	debug_value = OPX_HFI1_BAR_LOAD(&uregbase[ur_rcvegrindextail]);
	FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "&uregbase[ur_rcvegrindextail]  %p = %#16.16lX \n",
	       &uregbase[ur_rcvegrindextail], debug_value);
	debug_value = OPX_HFI1_BAR_LOAD(&uregbase[ur_rcvegrindexhead]);
	FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "&uregbase[ur_rcvegrindexhead]  %p = %#16.16lX \n",
	       &uregbase[ur_rcvegrindexhead], debug_value);
	debug_value = OPX_HFI1_BAR_LOAD(&uregbase[ur_rcvegroffsettail]);
	FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "&uregbase[ur_rcvegroffsettail] %p = %#16.16lX \n",
	       &uregbase[ur_rcvegroffsettail], debug_value);
	for (int i = 0; i < 32; ++i) {
		debug_value = OPX_HFI1_BAR_LOAD(&tidflowtable[i]);
		FI_DBG(fi_opx_global.prov, FI_LOG_CORE, "uregbase[ur_rcvtidflowtable][%u] = %#16.16lX \n", i,
		       debug_value);
	}
#endif
	/* TID flows aren't cleared between jobs, do it now. */
	for (int i = 0; i < 32; ++i) {
		OPX_HFI1_BAR_STORE(&tidflowtable[i], 0UL);
	}
	assert(ctrl->__hfi_tidexpcnt <= OPX_MAX_TID_COUNT);
	context->runtime_flags = ctxt_info->runtime_flags;

	/* OPX relies on RHF.SeqNum, not the RcvHdrTail */
	assert(!(context->runtime_flags & HFI1_CAP_DMA_RTAIL));

	context->info.rxe.hdrq.elemsz = ctxt_info->rcvhdrq_entsize >> BYTE2DWORD_SHIFT;
	if (context->info.rxe.hdrq.elemsz != FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS) {
		FI_WARN(fi_opx_global.prov, FI_LOG_CORE, "Invalid hdrq_entsize %u (only %lu is supported)\n",
			context->info.rxe.hdrq.elemsz, FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS);
		abort();
	}
	context->info.rxe.hdrq.elemcnt	    = ctxt_info->rcvhdrq_cnt;
	context->info.rxe.hdrq.elemlast	    = ((context->info.rxe.hdrq.elemcnt - 1) * context->info.rxe.hdrq.elemsz);
	context->info.rxe.hdrq.rx_poll_mask = fi_opx_hfi1_header_count_to_poll_mask(ctxt_info->rcvhdrq_cnt);
	context->info.rxe.hdrq.base_addr    = (uint32_t *) (uintptr_t) base_info->rcvhdr_bufbase;
	context->info.rxe.hdrq.rhf_base	    = context->info.rxe.hdrq.base_addr + context->info.rxe.hdrq.rhf_off;

	context->info.rxe.egrq.base_addr = (uint32_t *) (uintptr_t) base_info->rcvegr_bufbase;
	context->info.rxe.egrq.elemsz	 = ctxt_info->rcvegr_size;
	context->info.rxe.egrq.size	 = ctxt_info->rcvegr_size * ctxt_info->egrtids;

	fi_opx_ref_init(&context->ref_cnt, "HFI context");
	FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Context configured with HFI=%d PORT=%d LID=0x%x JKEY=%d\n",
		context->hfi_unit, context->hfi_port, context->lid, context->jkey);

	context->status_lasterr = 0;
	context->status_check_next_usec =
		fi_opx_timer_now(&context->link_status_timestamp, &context->link_status_timer);

	opx_print_context(context);

	return context;

ctxt_open_err:
	free(internal);
	return NULL;
}

int init_hfi1_rxe_state(struct fi_opx_hfi1_context *context, struct fi_opx_hfi1_rxe_state *rxe_state)
{
	rxe_state->hdrq.head = 0;

	assert(!(context->runtime_flags & HFI1_CAP_DMA_RTAIL));
	rxe_state->hdrq.rhf_seq = OPX_RHF_SEQ_INIT_VAL(OPX_HFI1_TYPE);
	/*  OPX relies on RHF.SeqNum, not the RcvHdrTail
		if (context->runtime_flags & HFI1_CAP_DMA_RTAIL) {
			rxe_state->hdrq.rhf_seq = 0;
		} else {
			rxe_state->hdrq.rhf_seq = OPX_WFR_RHF_SEQ_INIT_VAL;
		}
	*/
	return 0;
}

#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_reliability.h"

ssize_t fi_opx_hfi1_tx_connect(struct fi_opx_ep *opx_ep, fi_addr_t peer)
{
	ssize_t rc = FI_SUCCESS;

	if ((opx_ep->tx->caps & FI_LOCAL_COMM) || ((opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == 0)) {
		const union fi_opx_addr addr = {.fi = peer};

		if (opx_lid_is_intranode(addr.lid)) {
			char		  buffer[128];
			union fi_opx_addr addr;
			addr.raw64b = (uint64_t) peer;

			uint8_t	 hfi_unit = addr.hfi1_unit;
			unsigned rx_index = addr.hfi1_rx;
			int	 inst	  = 0;

			assert(rx_index < 256);
			uint32_t segment_index = OPX_SHM_SEGMENT_INDEX(hfi_unit, rx_index);
			assert(segment_index < OPX_SHM_MAX_CONN_NUM);

#ifdef OPX_DAOS
			/* HFI Rank Support:  Rank and PID included in the SHM file name */
			if (opx_ep->daos_info.hfi_rank_enabled) {
				rx_index = opx_shm_daos_rank_index(opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst);
				inst	 = opx_ep->daos_info.rank_inst;
				segment_index = rx_index;
			}
#endif

			snprintf(buffer, sizeof(buffer), OPX_SHM_FILE_NAME_PREFIX_FORMAT,
				 opx_ep->domain->unique_job_key_str, hfi_unit, inst);

			rc = opx_shm_tx_connect(&opx_ep->tx->shm, (const char *const) buffer, segment_index, rx_index,
						FI_OPX_SHM_FIFO_SIZE, FI_OPX_SHM_PACKET_SIZE);
		}
	}

	return rc;
}

int opx_hfi1_rx_rzv_rts_send_cts_intranode(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	  = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	  = params->opx_ep;
	const uint64_t			      lrh_dlid_9B = params->lrh_dlid;
	const uint64_t			      bth_rx	  = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RENDEZVOUS RTS (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-RTS-SHM");
	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	hdr->qw_9B[0] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid_9B;
	hdr->qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
	hdr->qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2];
	hdr->qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
	hdr->qw_9B[4] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | (params->niov << 48) | params->opcode;
	hdr->qw_9B[5] = params->origin_byte_counter_vaddr;
	hdr->qw_9B[6] = (uint64_t) params->rzv_comp;

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-RTS-SHM");
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_cts_intranode_16B(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	   = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	   = params->opx_ep;
	const uint64_t			      lrh_dlid_16B = params->lrh_dlid;
	const uint64_t			      bth_rx	   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV 16B, SHM -- RENDEZVOUS RTS (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-RTS-SHM");
	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	hdr->qw_16B[0] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
		((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B));
	hdr->qw_16B[1] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
		((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >> OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
		(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
	hdr->qw_16B[2] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx;
	hdr->qw_16B[3] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[3];
	hdr->qw_16B[4] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[4];
	hdr->qw_16B[5] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] | (params->niov << 48) | params->opcode;
	hdr->qw_16B[6] = params->origin_byte_counter_vaddr;
	hdr->qw_16B[7] = (uint64_t) params->rzv_comp;

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-RTS-SHM");
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV 16B, SHM -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_cts(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	  = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	  = params->opx_ep;
	const uint64_t			      lrh_dlid_9B = params->lrh_dlid;
	const uint64_t			      bth_rx	  = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (begin) (params=%p rzv_comp=%p context=%p)\n",
	       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp, params->rzv_comp->context);
	assert(params->rzv_comp->context->byte_counter >= params->dput_iov[0].bytes);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-CTS-HFI:%p", params->rzv_comp);
	const uint64_t tid_payload =
		params->tid_info.npairs ? ((params->tid_info.npairs + 4) * sizeof(params->tidpairs[0])) : 0;
	const uint64_t payload_bytes = (params->niov * sizeof(union fi_opx_hfi1_dput_iov)) + tid_payload;
	const uint64_t pbc_dws	     = 2 + /* pbc */
				 2 +	   /* lrh */
				 3 +	   /* bth */
				 9 +	   /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 ((payload_bytes + 3) >> 2);
	const uint16_t lrh_dws = htons(
		pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */
	union fi_opx_hfi1_pio_state pio_state		 = *opx_ep->tx->pio_state;
	const uint16_t		    total_credits_needed = 1 +		   /* packet header */
					      ((payload_bytes + 63) >> 6); /* payload blocks needed */
	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);

	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available	   = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									   total_credits_needed);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;

		if (total_credits_available < total_credits_needed) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (EAGAIN credits) (params=%p rzv_comp=%p context=%p)\n",
			       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp,
			       params->rzv_comp->context);
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, OPX_HFI1_TYPE);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (EAGAIN psn/replay) (params=%p rzv_comp=%p context=%p)\n",
		       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp,
		       params->rzv_comp->context);
		return -FI_EAGAIN;
	}

	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

	// The "memcopy first" code is here as an alternative to the more complicated
	// direct write to pio followed by memory copy of the reliability buffer

	replay->scb.scb_9B.qw0 = opx_ep->rx->tx.cts_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid;
	replay->scb.scb_9B.hdr.qw_9B[0] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32);
	replay->scb.scb_9B.hdr.qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
	replay->scb.scb_9B.hdr.qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2] | psn;
	replay->scb.scb_9B.hdr.qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
	replay->scb.scb_9B.hdr.qw_9B[4] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] |
					  ((uint64_t) params->tid_info.npairs << 32) | (params->niov << 48) |
					  params->opcode;
	replay->scb.scb_9B.hdr.qw_9B[5] = params->origin_byte_counter_vaddr;
	replay->scb.scb_9B.hdr.qw_9B[6] = (uint64_t) params->rzv_comp;

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) replay->payload;
	assert(((uint8_t *) tx_payload) == ((uint8_t *) &replay->data));

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

	/* copy tidpairs to packet */
	if (params->tid_info.npairs) {
		assert(params->tid_info.npairs < FI_OPX_MAX_DPUT_TIDPAIRS);
		assert(params->tidpairs[0] != 0);
		assert(params->niov == 1);
		assert(params->rzv_comp->context->byte_counter >= params->dput_iov[0].bytes);

		/* coverity[missing_lock] */
		tx_payload->tid_cts.tid_offset		       = params->tid_info.offset;
		tx_payload->tid_cts.ntidpairs		       = params->tid_info.npairs;
		tx_payload->tid_cts.origin_byte_counter_adjust = params->tid_info.origin_byte_counter_adj;
		for (int i = 0; i < params->tid_info.npairs; ++i) {
			tx_payload->tid_cts.tidpairs[i] = params->tidpairs[i];
		}
	}

#ifdef HAVE_CUDA
	if (params->dput_iov[0].rbuf_iface == FI_HMEM_CUDA) {
		int err = cuda_set_sync_memops((void *) params->dput_iov[0].rbuf);
		if (OFI_UNLIKELY(err != 0)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_MR, "cuda_set_sync_memops(%p) FAILED (returned %d)\n",
				(void *) params->dput_iov[0].rbuf, err);
		}
	}
#endif

	fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->origin_rx, psn_ptr, replay, params->reliability,
							    OPX_HFI1_TYPE);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-CTS-HFI:%p", params->rzv_comp);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (end) (params=%p rzv_comp=%p context=%p)\n",
	       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp, params->rzv_comp->context);
	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_cts_16B(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	   = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	   = params->opx_ep;
	const uint64_t			      lrh_dlid_16B = params->lrh_dlid;
	const uint64_t			      bth_rx	   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV 16B, HFI -- RENDEZVOUS %s RTS (begin) (params=%p rzv_comp=%p context=%p)\n",
	       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp, params->rzv_comp->context);
	assert(params->rzv_comp->context->byte_counter >= params->dput_iov[0].bytes);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-CTS-HFI:%p", params->rzv_comp);
	const uint64_t tid_payload =
		params->tid_info.npairs ? ((params->tid_info.npairs + 4) * sizeof(params->tidpairs[0])) : 0;
	const uint64_t payload_bytes = (params->niov * sizeof(union fi_opx_hfi1_dput_iov)) + tid_payload;
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "payload_bytes = %ld\n", payload_bytes);
	const uint64_t pbc_dws = 2 +				     /* pbc */
				 4 +				     /* lrh uncompressed */
				 3 +				     /* bth */
				 9 +				     /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 (((payload_bytes + 7) & -8) >> 2) + /* 16B is QW length/padded */
				 2;				     /* ICRC/tail */
	const uint16_t		    lrh_qws   = (pbc_dws - 2) >> 1;  /* (LRH QW) does not include pbc (8 bytes) */
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
	const uint16_t		    total_credits_needed = 1 +		   /* packet header */
					      ((payload_bytes + 63) >> 6); /* payload blocks needed */
	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);

	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available	   = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									   total_credits_needed);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;

		if (total_credits_available < total_credits_needed) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (EAGAIN credits) (params=%p rzv_comp=%p context=%p)\n",
			       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp,
			       params->rzv_comp->context);
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, OPX_HFI1_TYPE);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (EAGAIN psn/replay) (params=%p rzv_comp=%p context=%p)\n",
		       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp,
		       params->rzv_comp->context);
		return -FI_EAGAIN;
	}

	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

	// The "memcopy first" code is here as an alternative to the more complicated
	// direct write to pio followed by memory copy of the reliability buffer
	replay->scb.scb_16B.qw0 = opx_ep->rx->tx.cts_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid;
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "replay->scb_16B.qw0 = %#lx pbc_dws = %ld\n",
	       replay->scb.scb_16B.qw0, pbc_dws);
	replay->scb.scb_16B.hdr.qw_16B[0] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
		((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
		((uint64_t) lrh_qws << 20);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "lrh_qws = %d replay->scb_16B.hdr.lrh_16B.pktlen = %d\n", lrh_qws,
	       replay->scb.scb_16B.hdr.lrh_16B.pktlen);
	replay->scb.scb_16B.hdr.qw_16B[1] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
		((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >> OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
		(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
	replay->scb.scb_16B.hdr.qw_16B[2] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx;
	replay->scb.scb_16B.hdr.qw_16B[3] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[3] | psn;
	replay->scb.scb_16B.hdr.qw_16B[4] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[4];
	replay->scb.scb_16B.hdr.qw_16B[5] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] |
					    ((uint64_t) params->tid_info.npairs << 32) | (params->niov << 48) |
					    params->opcode;
	replay->scb.scb_16B.hdr.qw_16B[6] = params->origin_byte_counter_vaddr;

	replay->scb.scb_16B.hdr.qw_16B[7] = (uint64_t) params->rzv_comp;

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) (replay->payload);

	assert(((uint8_t *) tx_payload) == ((uint8_t *) &(replay->data)));

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

	/* copy tidpairs to packet */
	if (params->tid_info.npairs) {
		assert(params->tid_info.npairs < FI_OPX_MAX_DPUT_TIDPAIRS);
		assert(params->tidpairs[0] != 0);
		assert(params->niov == 1);
		assert(params->rzv_comp->context->byte_counter >= params->dput_iov[0].bytes);

		/* coverity[missing_lock] */
		tx_payload->tid_cts.tid_offset		       = params->tid_info.offset;
		tx_payload->tid_cts.ntidpairs		       = params->tid_info.npairs;
		tx_payload->tid_cts.origin_byte_counter_adjust = params->tid_info.origin_byte_counter_adj;
		for (int i = 0; i < params->tid_info.npairs; ++i) {
			tx_payload->tid_cts.tidpairs[i] = params->tidpairs[i];
		}
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "fi_opx_reliability_service_do_replay &opx_ep->reliability->service %p, replay %p\n",
		     &opx_ep->reliability->service, replay);
	fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->origin_rx, psn_ptr, replay, params->reliability,
							    OPX_HFI1_TYPE);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-CTS-HFI:%p", params->rzv_comp);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS %s RTS (end) (params=%p rzv_comp=%p context=%p)\n",
	       params->tid_info.npairs ? "EXPECTED TID" : "EAGER", params, params->rzv_comp, params->rzv_comp->context);
	return FI_SUCCESS;
}

__OPX_FORCE_INLINE__
int opx_hfi1_rx_rzv_rts_tid_eligible(struct fi_opx_ep *opx_ep, struct fi_opx_hfi1_rx_rzv_rts_params *params,
				     const uint64_t niov, const uint64_t immediate_data, const uint64_t immediate_tail,
				     const uint64_t is_hmem, const uint64_t is_hmem_unified,
				     const uint64_t is_intranode, const enum fi_hmem_iface iface, uint8_t opcode)
{
	if (is_intranode || !opx_ep->use_expected_tid_rzv || (niov != 1) ||
	    (params->dput_iov[0].bytes < opx_ep->tx->tid_min_payload_bytes) ||
	    (opcode != FI_OPX_HFI_DPUT_OPCODE_RZV && opcode != FI_OPX_HFI_DPUT_OPCODE_RZV_NONCONTIG) ||
	    is_hmem_unified ||
	    !fi_opx_hfi1_sdma_use_sdma(opx_ep, params->dput_iov[0].bytes, opcode, is_hmem, OPX_INTRANODE_FALSE)) {
		FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.expected_receive.rts_tid_ineligible);
		return 0;
	}

#ifndef NDEBUG
	const uintptr_t rbuf_end = params->dst_vaddr + params->dput_iov[0].bytes;
#endif

	/* Caller adjusted pointers and lengths past the immediate data.
	 * Now align the destination buffer to be page aligned for expected TID writes
	 * This should point/overlap into the immediate data area.
	 * Then realign source buffer and lengths appropriately.
	 */
	/* TID writes must start on 64 byte boundaries */
	uintptr_t vaddr_aligned64 = params->dst_vaddr & -64;

	int64_t byte_counter_adjust;

	if (vaddr_aligned64 >= (params->dst_vaddr - immediate_data)) {
		size_t alignment_adjustment = params->dst_vaddr - vaddr_aligned64;
		params->dst_vaddr -= alignment_adjustment;
		params->dput_iov[0].rbuf -= alignment_adjustment;
		params->dput_iov[0].sbuf -= alignment_adjustment;
		params->dput_iov[0].bytes += alignment_adjustment;

		byte_counter_adjust = alignment_adjustment;

		params->elided_head.bytes = 0;
	} else {
		// Round up to next 64-byte boundary.
		vaddr_aligned64 = (params->dst_vaddr + 63) & -64;

		// If params->dst_vaddr is already on a 64-byte boundary, then
		// adding 63 to it and ANDing that with -64 would result in the
		// same address. *But* in that situation, we should not have
		// taken this else branch, so rounding up to the next boundary
		// should definitely result in vaddr being > params->dst_vaddr
		assert(vaddr_aligned64 > params->dst_vaddr);

		// Get the portion of bytes at the start of the buffer that
		// we'll need to send a separate CTS for, and then adjust the
		// original buffers
		params->elided_head.bytes	= vaddr_aligned64 - params->dst_vaddr;
		params->elided_head.rbuf	= params->dst_vaddr;
		params->elided_head.rbuf_iface	= params->dput_iov[0].rbuf_iface;
		params->elided_head.rbuf_device = params->dput_iov[0].rbuf_device;
		params->elided_head.sbuf	= params->dput_iov[0].sbuf;
		params->elided_head.sbuf_iface	= params->dput_iov[0].sbuf_iface;
		params->elided_head.sbuf_device = params->dput_iov[0].sbuf_device;

		params->dst_vaddr	 = vaddr_aligned64;
		params->dput_iov[0].rbuf = vaddr_aligned64;
		params->dput_iov[0].sbuf += params->elided_head.bytes;
		params->dput_iov[0].bytes -= params->elided_head.bytes;

		// No byte counter adjustment necessary because we didn't
		// overlap with immediate data so we aren't requesting bytes
		// to be sent that were already sent.
		byte_counter_adjust = 0;
	}

	// Make sure that our buffer still ends in the same place, even after
	// adjusting the start to be cacheline-aligned
	assert((params->dst_vaddr + params->dput_iov[0].bytes) == rbuf_end);

	/* We need to ensure the length is a qw multiple. If a shorter length
	   is needed, and no immediate tail data was sent, we'll need to get
	   the elided tail data via separate CTS packet */
	const size_t elided_tail_bytes = params->dput_iov[0].bytes & 7;
	const size_t qw_floor_length   = params->dput_iov[0].bytes & -8;
	if (elided_tail_bytes && !immediate_tail) {
		params->elided_tail.bytes	= elided_tail_bytes;
		params->elided_tail.rbuf	= params->dput_iov[0].rbuf + qw_floor_length;
		params->elided_tail.sbuf	= params->dput_iov[0].sbuf + qw_floor_length;
		params->elided_tail.rbuf_iface	= params->dput_iov[0].rbuf_iface;
		params->elided_tail.rbuf_device = params->dput_iov[0].rbuf_device;
		params->elided_tail.sbuf_iface	= params->dput_iov[0].sbuf_iface;
		params->elided_tail.sbuf_device = params->dput_iov[0].sbuf_device;
	} else {
		// If elided_tail_bytes was non-zero, then it must be the case
		// that we had immediate_tail data and don't need to request those
		// bytes to be sent via separate CTS packet. But we still do need
		// to subtract them from the byte counter.
		byte_counter_adjust -= elided_tail_bytes;
		params->elided_tail.bytes = 0;
	}

	params->dput_iov[0].bytes = qw_floor_length;
	params->rzv_comp->context->byte_counter += byte_counter_adjust;
	params->tid_info.origin_byte_counter_adj = (int32_t) byte_counter_adjust;

	FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.expected_receive.rts_tid_eligible);

	return 1;
}

__OPX_FORCE_INLINE__
union fi_opx_hfi1_deferred_work *opx_hfi1_rx_rzv_rts_tid_prep_cts(union fi_opx_hfi1_deferred_work      *work,
								  struct fi_opx_hfi1_rx_rzv_rts_params *params,
								  const struct opx_tid_addr_block      *tid_addr_block,
								  const size_t cur_addr_range_tid_len,
								  const bool   last_cts)
{
	union fi_opx_hfi1_deferred_work	     *cts_work;
	struct fi_opx_hfi1_rx_rzv_rts_params *cts_params;

	// If this will not be the last CTS we send, allocate a new deferred
	// work item and rzv completion to use for the CTS, and copy the first
	// portion of the current work item into it. If this will be the last
	// CTS, we'll just use the existing deferred work item and rzv completion
	if (!last_cts) {
		cts_work = ofi_buf_alloc(params->opx_ep->tx->work_pending_pool);
		if (OFI_UNLIKELY(cts_work == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Failed to allocate deferred work item!\n");
			return NULL;
		}
		struct fi_opx_rzv_completion *rzv_comp = ofi_buf_alloc(params->opx_ep->rzv_completion_pool);
		if (OFI_UNLIKELY(rzv_comp == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Failed to allocate rendezvous completion item!\n");
			OPX_BUF_FREE(cts_work);
			return NULL;
		}

		// Add 1 to the offset so we end up with a cacheline multiple length
		const size_t copy_length = offsetof(struct fi_opx_hfi1_rx_rzv_rts_params, multi_cts_copy_boundary) + 1;
		assert(copy_length < sizeof(*work));
		memcpy(cts_work, work, copy_length);

		cts_work->work_elem.slist_entry.next = NULL;
		cts_params			     = &cts_work->rx_rzv_rts;
		cts_params->rzv_comp		     = rzv_comp;
		cts_params->rzv_comp->context	     = params->rzv_comp->context;
	} else {
		cts_work   = work;
		cts_params = params;
	}

	// Calculate the offset of the target buffer relative to the
	// original target buffer address, and then use that to set
	// the address for the source buffer
	size_t	  target_offset	      = params->tid_info.cur_addr_range.buf - params->dput_iov[params->cur_iov].rbuf;
	uintptr_t adjusted_source_buf = params->dput_iov[params->cur_iov].sbuf + target_offset;

	cts_params->niov		    = 1;
	cts_params->dput_iov[0].rbuf_iface  = params->dput_iov[params->cur_iov].rbuf_iface;
	cts_params->dput_iov[0].rbuf_device = params->dput_iov[params->cur_iov].rbuf_device;
	cts_params->dput_iov[0].sbuf_iface  = params->dput_iov[params->cur_iov].sbuf_iface;
	cts_params->dput_iov[0].sbuf_device = params->dput_iov[params->cur_iov].sbuf_device;
	cts_params->dput_iov[0].rbuf	    = params->tid_info.cur_addr_range.buf;
	cts_params->dput_iov[0].sbuf	    = adjusted_source_buf;
	cts_params->dput_iov[0].bytes	    = cur_addr_range_tid_len;
	cts_params->dst_vaddr		    = params->tid_info.cur_addr_range.buf;

	cts_params->rzv_comp->tid_vaddr		= params->tid_info.cur_addr_range.buf;
	cts_params->rzv_comp->tid_length	= cur_addr_range_tid_len;
	cts_params->rzv_comp->byte_counter	= cur_addr_range_tid_len;
	cts_params->rzv_comp->bytes_accumulated = 0;

	cts_params->tid_info.npairs		     = tid_addr_block->npairs;
	cts_params->tid_info.offset		     = tid_addr_block->offset;
	cts_params->tid_info.origin_byte_counter_adj = params->tid_info.origin_byte_counter_adj;

	assert(cur_addr_range_tid_len <= cts_params->rzv_comp->context->byte_counter);
	assert(tid_addr_block->npairs < FI_OPX_MAX_DPUT_TIDPAIRS);
	for (int i = 0; i < tid_addr_block->npairs; i++) {
		cts_params->tidpairs[i] = tid_addr_block->pairs[i];
	}

	assert(cur_addr_range_tid_len <= cts_params->rzv_comp->context->byte_counter);

	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		cts_params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts;
	} else {
		cts_params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_16B;
	}
	cts_params->work_elem.work_type = OPX_WORK_TYPE_PIO;

	return cts_work;
}

__OPX_FORCE_INLINE__
int opx_hfi1_rx_rzv_rts_tid_fallback(union fi_opx_hfi1_deferred_work	  *work,
				     struct fi_opx_hfi1_rx_rzv_rts_params *params)
{
	/* Since we may have already sent one or more CTS packets covering
	   some portion of the receive range using TID, we now need to
	   adjust the buf pointers and length in the dput_iov we were
	   working on to reflect only the unsent portion */
	assert(params->tid_info.cur_addr_range.buf >= ((uintptr_t) params->dput_iov[params->cur_iov].rbuf));
	size_t bytes_already_sent =
		params->tid_info.cur_addr_range.buf - ((uintptr_t) params->dput_iov[params->cur_iov].rbuf);
	assert(bytes_already_sent < params->dput_iov[params->cur_iov].bytes);

	params->dput_iov[params->cur_iov].rbuf = params->tid_info.cur_addr_range.buf;
	params->dput_iov[params->cur_iov].sbuf += bytes_already_sent;
	params->dput_iov[params->cur_iov].bytes -= bytes_already_sent;
	params->dst_vaddr = params->dput_iov[params->cur_iov].rbuf;

	params->tid_info.npairs = 0;

	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts;
	} else {
		params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_16B;
	}
	params->work_elem.work_type = OPX_WORK_TYPE_PIO;
	params->opcode		    = FI_OPX_HFI_DPUT_OPCODE_RZV;

	FI_OPX_DEBUG_COUNTERS_INC(params->opx_ep->debug_counters.expected_receive.rts_fallback_eager_reg_rzv);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS RTS TID SETUP (end) EPERM, switching to non-TID send CTS (params=%p rzv_comp=%p context=%p)\n",
	       params, params->rzv_comp, params->rzv_comp->context);

	return params->work_elem.work_fn(work);
}

int opx_hfi1_rx_rzv_rts_tid_setup(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS RTS TID SETUP (begin) (params=%p rzv_comp=%p context=%p)\n",
	       params, params->rzv_comp, params->rzv_comp->context);

	struct opx_tid_addr_block tid_addr_block = {};

	int register_rc = opx_register_for_rzv(params->opx_ep, &params->tid_info.cur_addr_range, &tid_addr_block);

	/* TID has been disabled for this endpoint, fall back to rendezvous */
	if (OFI_UNLIKELY(register_rc == -FI_EPERM)) {
		return opx_hfi1_rx_rzv_rts_tid_fallback(work, params);
	} else if (register_rc != FI_SUCCESS) {
		assert(register_rc == -FI_EAGAIN);
		FI_OPX_DEBUG_COUNTERS_INC(params->opx_ep->debug_counters.expected_receive.rts_tid_setup_retries);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS RTS TID SETUP (end) EAGAIN (No Progress) (params=%p rzv_comp=%p context=%p)\n",
		       params, params->rzv_comp, params->rzv_comp->context);
		return -FI_EAGAIN;
	}

	void *cur_addr_range_end = (void *) (params->tid_info.cur_addr_range.buf + params->tid_info.cur_addr_range.len);
	void *tid_addr_block_end =
		(void *) ((uintptr_t) tid_addr_block.target_iov.iov_base + tid_addr_block.target_iov.iov_len);

	// The start of the Current Address Range should always fall within the
	// resulting tid_addr_block IOV
	assert(tid_addr_block.target_iov.iov_base <= (void *) params->tid_info.cur_addr_range.buf);
	assert(tid_addr_block_end > (void *) params->tid_info.cur_addr_range.buf);

	// Calculate the portion of cur_addr_range that we were able to get TIDs for
	size_t cur_addr_range_tid_len =
		((uintptr_t) MIN(tid_addr_block_end, cur_addr_range_end)) - params->tid_info.cur_addr_range.buf;
	assert(cur_addr_range_tid_len <= params->rzv_comp->context->byte_counter);

	// If this is the last IOV and the tid range covers the end of the current
	// range, then this will be the last CTS we need to send.
	const bool last_cts = (params->cur_iov == (params->niov - 1)) && (tid_addr_block_end >= cur_addr_range_end);

	union fi_opx_hfi1_deferred_work *cts_work =
		opx_hfi1_rx_rzv_rts_tid_prep_cts(work, params, &tid_addr_block, cur_addr_range_tid_len, last_cts);

	if (last_cts) {
		assert(cts_work == work);

		if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			assert(work->work_elem.work_fn == opx_hfi1_rx_rzv_rts_send_cts);
		} else {
			assert(work->work_elem.work_fn == opx_hfi1_rx_rzv_rts_send_cts_16B);
		}
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS RTS TID SETUP (end) SUCCESS (params=%p rzv_comp=%p context=%p)\n",
		       params, params->rzv_comp, params->rzv_comp->context);

		FI_OPX_DEBUG_COUNTERS_INC(params->opx_ep->debug_counters.expected_receive.rts_tid_setup_success);

		// This is the "FI_SUCCESS" exit point for this function
		return cts_work->work_elem.work_fn(cts_work);
	}

	assert(cts_work != work);

	int rc = cts_work->work_elem.work_fn(cts_work);
	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(cts_work);
	} else {
		assert(rc == -FI_EAGAIN);
		slist_insert_tail(&cts_work->work_elem.slist_entry,
				  &params->opx_ep->tx->work_pending[cts_work->work_elem.work_type]);
	}

	// We shouldn't need to adjust the origin byte counter after sending the
	// first CTS packet.
	params->tid_info.origin_byte_counter_adj = 0;

	/* Adjust Current Address Range for next iteration */
	if (tid_addr_block_end >= cur_addr_range_end) {
		// We finished processing the current IOV, so move on to the next one
		++params->cur_iov;
		assert(params->cur_iov < params->niov);
		params->tid_info.cur_addr_range.buf    = params->dput_iov[params->cur_iov].rbuf;
		params->tid_info.cur_addr_range.len    = params->dput_iov[params->cur_iov].bytes;
		params->tid_info.cur_addr_range.iface  = params->dput_iov[params->cur_iov].rbuf_iface;
		params->tid_info.cur_addr_range.device = params->dput_iov[params->cur_iov].rbuf_device;
	} else {
		params->tid_info.cur_addr_range.buf += cur_addr_range_tid_len;
		params->tid_info.cur_addr_range.len -= cur_addr_range_tid_len;
	}

	// Wait until the next poll cycle before trying to register more TIDs.
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS RTS TID SETUP (end) EAGAIN (Progress) (params=%p rzv_comp=%p context=%p)\n",
	       params, params->rzv_comp, params->rzv_comp->context);

	return -FI_EAGAIN;
}

int opx_hfi1_rx_rzv_rts_elided(struct fi_opx_ep *opx_ep, union fi_opx_hfi1_deferred_work *work,
			       struct fi_opx_hfi1_rx_rzv_rts_params *params)
{
	assert(params->elided_head.bytes || params->elided_tail.bytes);
	assert(!params->is_intranode); // We should never be doing this function for intranode

	union fi_opx_hfi1_deferred_work	     *cts_work;
	struct fi_opx_hfi1_rx_rzv_rts_params *cts_params;

	cts_work = ofi_buf_alloc(params->opx_ep->tx->work_pending_pool);
	if (OFI_UNLIKELY(cts_work == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Failed to allocate deferred work item!\n");
		return -FI_ENOMEM;
	}
	struct fi_opx_rzv_completion *rzv_comp = ofi_buf_alloc(params->opx_ep->rzv_completion_pool);
	if (OFI_UNLIKELY(rzv_comp == NULL)) {
		FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Failed to allocate rendezvous completion item!\n");
		OPX_BUF_FREE(cts_work);
		return -FI_ENOMEM;
	}

	// Add 1 to the offset so we end up with a cacheline multiple length
	const size_t copy_length = offsetof(struct fi_opx_hfi1_rx_rzv_rts_params, multi_cts_copy_boundary) + 1;
	assert(copy_length < sizeof(*cts_work));
	memcpy(cts_work, work, copy_length);

	cts_work->work_elem.slist_entry.next = NULL;
	cts_params			     = &cts_work->rx_rzv_rts;
	cts_params->rzv_comp		     = rzv_comp;
	cts_params->rzv_comp->context	     = params->rzv_comp->context;

	int niov = 0;

	if (params->elided_head.bytes) {
		cts_params->dput_iov[niov++] = params->elided_head;
	}

	if (params->elided_tail.bytes) {
		cts_params->dput_iov[niov++] = params->elided_tail;
	}

	cts_params->dst_vaddr	    = cts_params->dput_iov[0].rbuf;
	cts_params->cur_iov	    = 0;
	cts_params->niov	    = niov;
	cts_params->tid_info.npairs = 0;

	rzv_comp->byte_counter	    = params->elided_head.bytes + params->elided_tail.bytes;
	rzv_comp->bytes_accumulated = 0;

	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		cts_params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts;
	} else {
		cts_params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_16B;
	}
	cts_params->work_elem.work_type = OPX_WORK_TYPE_PIO;

	int rc = cts_params->work_elem.work_fn(cts_work);
	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(cts_work);
	} else {
		assert(rc == -FI_EAGAIN);
		/* Try again later*/
		assert(cts_work->work_elem.slist_entry.next == NULL);
		slist_insert_tail(&cts_work->work_elem.slist_entry, &opx_ep->tx->work_pending[OPX_WORK_TYPE_PIO]);
	}
	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_etrunc_intranode(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;

	struct fi_opx_ep *opx_ep      = params->opx_ep;
	const uint64_t	  lrh_dlid_9B = params->lrh_dlid;
	const uint64_t	  bth_rx      = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RENDEZVOUS RTS ETRUNC (begin)\n");
	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const tx_hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!tx_hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	tx_hdr->qw_9B[0] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid_9B;
	tx_hdr->qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
	tx_hdr->qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2];
	tx_hdr->qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
	tx_hdr->qw_9B[4] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | params->opcode;
	tx_hdr->qw_9B[5] = params->origin_byte_counter_vaddr;

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) tx_hdr, pos);

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RENDEZVOUS RTS ETRUNC (end)\n");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_etrunc_intranode_16B(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	   = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	   = params->opx_ep;
	const uint64_t			      lrh_dlid_16B = params->lrh_dlid;
	const uint64_t			      bth_rx	   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV 16B, SHM -- RENDEZVOUS RTS ETRUNC (begin)\n");
	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const tx_hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!tx_hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	tx_hdr->qw_16B[0] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
		((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B));
	tx_hdr->qw_16B[1] =
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
		((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >> OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
		(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
	tx_hdr->qw_16B[2] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx;
	tx_hdr->qw_16B[3] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[3];
	tx_hdr->qw_16B[4] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[4];
	tx_hdr->qw_16B[5] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] | params->opcode;
	tx_hdr->qw_16B[6] = params->origin_byte_counter_vaddr;

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) tx_hdr, pos);

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RENDEZVOUS RTS ETRUNC (end)\n");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_etrunc(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	  = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	  = params->opx_ep;
	const uint64_t			      lrh_dlid_9B = params->lrh_dlid;
	const uint64_t			      bth_rx	  = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (begin)\n");

	const uint64_t pbc_dws = 2 + /* pbc */
				 2 + /* lrh */
				 3 + /* bth */
				 9;  /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
	const uint16_t lrh_dws = htons(
		pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	if (OFI_UNLIKELY(FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, 1) < 1)) {
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;
		if (FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, 1) < 1) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (EAGAIN credits)\n");
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, OPX_HFI1_TYPE);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (EAGAIN psn/replay)\n");
		return -FI_EAGAIN;
	}

	volatile uint64_t *const scb = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);

	fi_opx_store_and_copy_scb_9B(scb, &replay->scb.scb_9B,
				     opx_ep->rx->tx.cts_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid,
				     opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32),
				     opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx,
				     opx_ep->rx->tx.cts_9B.hdr.qw_9B[2] | psn, opx_ep->rx->tx.cts_9B.hdr.qw_9B[3],
				     opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | params->opcode,
				     params->origin_byte_counter_vaddr, 0);

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);

	/* consume one credit */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	/* save the updated txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->origin_rx, psn_ptr, replay, params->reliability,
							    OPX_HFI1_TYPE);

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (end)");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rzv_rts_send_etrunc_16B(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rzv_rts_params *params	   = &work->rx_rzv_rts;
	struct fi_opx_ep		     *opx_ep	   = params->opx_ep;
	const uint64_t			      lrh_dlid_16B = params->lrh_dlid;
	const uint64_t			      bth_rx	   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (begin)\n");

	const uint64_t pbc_dws = 2 +				    /* pbc */
				 4 +				    /* lrh uncompressed */
				 3 +				    /* bth */
				 9 +				    /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 2;				    /* ICRC/tail */
	const uint16_t		    lrh_qws   = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	// Note: Only need 1 credit here for the message truncation error case. Just
	// the opcode and origin_byte_counter_vaddr is needed for replaying back to the
	// sender.
	if (OFI_UNLIKELY(FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, 2) < 2)) {
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;
		if (FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, 2) < 2) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (EAGAIN credits)\n");
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, OPX_HFI1_TYPE);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (EAGAIN psn/replay)\n");
		return -FI_EAGAIN;
	}

	volatile uint64_t *const scb = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);

	fi_opx_store_and_copy_scb_16B(
		scb, &replay->scb.scb_16B,
		opx_ep->rx->tx.cts_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid,
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
			((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
			((uint64_t) lrh_qws << 20),
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
			((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
				     OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
			(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B),
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx, opx_ep->rx->tx.cts_16B.hdr.qw_16B[3] | psn,
		opx_ep->rx->tx.cts_16B.hdr.qw_16B[4], opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] | params->opcode,
		params->origin_byte_counter_vaddr);

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);

	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);

	// 2nd cacheline
	volatile uint64_t *const scb2 = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

	fi_opx_store_and_copy_qw(scb2, &replay->scb.scb_16B.hdr.qw_16B[7], 0, 0, 0, 0, 0, 0, 0, 0);

	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	/* save the updated txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->origin_rx, psn_ptr, replay, params->reliability,
							    OPX_HFI1_TYPE);

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RENDEZVOUS EAGER RTS ETRUNC (end)");

	return FI_SUCCESS;
}

void fi_opx_hfi1_rx_rzv_rts_etrunc(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
				   const uint8_t u8_rx, uintptr_t origin_byte_counter_vaddr,
				   const unsigned is_intranode, const enum ofi_reliability_kind reliability,
				   const uint32_t u32_extended_rx, const enum opx_hfi1_type hfi1_type)
{
	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;
	params->opx_ep				     = opx_ep;
	params->work_elem.slist_entry.next	     = NULL;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "is_intranode %u, opcode=%u\n", is_intranode,
	       FI_OPX_HFI_DPUT_OPCODE_RZV_ETRUNC);

	opx_lid_t lid;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		lid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
	} else {
		lid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
	}

	if (is_intranode) {
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_etrunc_intranode;
		} else {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_etrunc_intranode_16B;
		}
		params->work_elem.work_type = OPX_WORK_TYPE_SHM;

		if (lid == opx_ep->rx->self.lid) {
			params->target_hfi_unit = opx_ep->rx->self.hfi1_unit;
		} else {
			struct fi_opx_hfi_local_lookup *hfi_lookup = fi_opx_hfi1_get_lid_local(lid);
			assert(hfi_lookup);
			params->target_hfi_unit = hfi_lookup->hfi_unit;
		}
	} else {
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_etrunc;
		} else {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_etrunc_16B;
		}
		params->work_elem.work_type = OPX_WORK_TYPE_PIO;
		params->target_hfi_unit	    = 0xFF;
	}

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		params->slid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
		if (hfi1_type & OPX_HFI1_WFR) {
			params->lrh_dlid = (hdr->lrh_9B.qw[0] & 0xFFFF000000000000ul) >> 32;
		} else {
			params->lrh_dlid = hdr->lrh_9B.slid << 16;
		}
	} else {
		params->slid	 = lid;
		params->lrh_dlid = lid; // Send CTS to the SLID that sent RTS
	}

	params->pbc_dlid		  = OPX_PBC_DLID_TO_PBC_DLID(lid, hfi1_type);
	params->origin_rx		  = hdr->rendezvous.origin_rx;
	params->origin_rs		  = hdr->rendezvous.origin_rs;
	params->u8_rx			  = u8_rx;
	params->u32_extended_rx		  = u32_extended_rx;
	params->origin_byte_counter_vaddr = origin_byte_counter_vaddr;
	params->is_intranode		  = is_intranode;
	params->reliability		  = reliability;
	params->opcode			  = FI_OPX_HFI_DPUT_OPCODE_RZV_ETRUNC;

	int rc = params->work_elem.work_fn(work);
	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(work);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_SUCCESS\n");
		return;
	}
	assert(rc == -FI_EAGAIN);
	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[params->work_elem.work_type]);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
}

void fi_opx_hfi1_rx_rzv_rts(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
			    const void *const payload, const uint8_t u8_rx, const uint64_t niov,
			    uintptr_t origin_byte_counter_vaddr, struct opx_context *const target_context,
			    const uintptr_t dst_vaddr, const enum fi_hmem_iface dst_iface, const uint64_t dst_device,
			    const uint64_t immediate_data, const uint64_t immediate_end_bytes,
			    const struct fi_opx_hmem_iov *src_iovs, uint8_t opcode, const unsigned is_intranode,
			    const enum ofi_reliability_kind reliability, const uint32_t u32_extended_rx,
			    const enum opx_hfi1_type hfi1_type)
{
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RZV-RTS-HFI:%ld", hdr->qw_9B[6]);
	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;
	params->opx_ep				     = opx_ep;
	params->work_elem.slist_entry.next	     = NULL;

	assert(niov <= MIN(FI_OPX_MAX_HMEM_IOV, FI_OPX_MAX_DPUT_IOV));

	const struct fi_opx_hmem_iov *src_iov	  = src_iovs;
	uint64_t		      is_hmem	  = dst_iface;
	uint64_t		      rbuf_offset = 0;
	for (int i = 0; i < niov; i++) {
#ifdef OPX_HMEM
		is_hmem |= src_iov->iface;
#endif
		params->dput_iov[i].sbuf	= src_iov->buf;
		params->dput_iov[i].sbuf_iface	= src_iov->iface;
		params->dput_iov[i].sbuf_device = src_iov->device;
		params->dput_iov[i].rbuf	= dst_vaddr + rbuf_offset;
		params->dput_iov[i].rbuf_iface	= dst_iface;
		params->dput_iov[i].rbuf_device = dst_device;
		params->dput_iov[i].bytes	= src_iov->len;
		rbuf_offset += src_iov->len;
		++src_iov;
	}

	opx_lid_t lid;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		lid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
	} else {
		lid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
	}

	if (is_intranode) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "is_intranode %u\n", is_intranode);
		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_intranode;
		} else {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_intranode_16B;
		}
		params->work_elem.work_type = OPX_WORK_TYPE_SHM;

		if (lid == opx_ep->rx->self.lid) {
			params->target_hfi_unit = opx_ep->rx->self.hfi1_unit;
		} else {
			struct fi_opx_hfi_local_lookup *hfi_lookup = fi_opx_hfi1_get_lid_local(lid);
			assert(hfi_lookup);
			params->target_hfi_unit = hfi_lookup->hfi_unit;
		}
	} else {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "opx_ep->use_expected_tid_rzv=%u niov=%lu opcode=%u\n",
		       opx_ep->use_expected_tid_rzv, niov, params->opcode);

		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts;
		} else {
			params->work_elem.work_fn = opx_hfi1_rx_rzv_rts_send_cts_16B;
		}
		params->work_elem.work_type = OPX_WORK_TYPE_PIO;
		params->target_hfi_unit	    = 0xFF;
	}
	params->work_elem.completion_action = NULL;
	params->work_elem.payload_copy	    = NULL;
	params->work_elem.complete	    = false;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		params->slid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
		if (hfi1_type & OPX_HFI1_WFR) {
			params->lrh_dlid = (hdr->lrh_9B.qw[0] & 0xFFFF000000000000ul) >> 32;
		} else {
			params->lrh_dlid = hdr->lrh_9B.slid << 16;
		}
	} else {
		params->slid	 = (opx_lid_t) lid;
		params->lrh_dlid = lid; // Send CTS to the SLID that sent RTS
	}
	params->pbc_dlid = OPX_PBC_DLID_TO_PBC_DLID(lid, hfi1_type);

	params->origin_rx			 = hdr->rendezvous.origin_rx;
	params->origin_rs			 = hdr->rendezvous.origin_rs;
	params->u8_rx				 = u8_rx;
	params->u32_extended_rx			 = u32_extended_rx;
	params->niov				 = niov;
	params->cur_iov				 = 0;
	params->origin_byte_counter_vaddr	 = origin_byte_counter_vaddr;
	params->rzv_comp			 = ofi_buf_alloc(opx_ep->rzv_completion_pool);
	params->rzv_comp->tid_vaddr		 = 0UL;
	params->rzv_comp->tid_length		 = 0UL;
	params->rzv_comp->byte_counter		 = 0UL;
	params->rzv_comp->bytes_accumulated	 = 0UL;
	params->rzv_comp->context		 = target_context;
	params->dst_vaddr			 = dst_vaddr;
	params->is_intranode			 = is_intranode;
	params->reliability			 = reliability;
	params->tid_info.npairs			 = 0;
	params->tid_info.offset			 = 0;
	params->tid_info.origin_byte_counter_adj = 0;
	params->opcode				 = opcode;
	params->elided_head.bytes		 = 0;
	params->elided_tail.bytes		 = 0;

	if (opx_hfi1_rx_rzv_rts_tid_eligible(opx_ep, params, niov, immediate_data, immediate_end_bytes, is_hmem,
					     ((struct fi_opx_hmem_info *) target_context->hmem_info_qws)->is_unified,
					     is_intranode, dst_iface, opcode)) {
		if (params->elided_head.bytes || params->elided_tail.bytes) {
			opx_hfi1_rx_rzv_rts_elided(opx_ep, work, params);
		}
		params->tid_info.cur_addr_range.buf    = params->dput_iov[0].rbuf;
		params->tid_info.cur_addr_range.len    = params->dput_iov[0].bytes;
		params->tid_info.cur_addr_range.iface  = params->dput_iov[0].rbuf_iface;
		params->tid_info.cur_addr_range.device = params->dput_iov[0].rbuf_device;

		params->work_elem.work_fn   = opx_hfi1_rx_rzv_rts_tid_setup;
		params->work_elem.work_type = OPX_WORK_TYPE_TID_SETUP;
		params->opcode		    = FI_OPX_HFI_DPUT_OPCODE_RZV_TID;
	}

	params->rzv_comp->byte_counter = target_context->byte_counter;

	int rc = params->work_elem.work_fn(work);
	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RZV-RTS-HFI:%ld", hdr->qw_9B[6]);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_SUCCESS\n");
		return;
	}
	assert(rc == -FI_EAGAIN);
	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[params->work_elem.work_type]);
	OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "RECV-RZV-RTS-HFI:%ld", hdr->qw_9B[6]);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
}

int opx_hfi1_do_dput_fence(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_dput_fence_params *params = &work->fence;
	struct fi_opx_ep			*opx_ep = params->opx_ep;

	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);
	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);
	if (hdr == NULL) {
		return rc;
	}

	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		const uint64_t pbc_dws = 2 + /* pbc */
					 2 + /* lrh */
					 3 + /* bth */
					 9;  /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
		const uint16_t lrh_dws =
			htons(pbc_dws - 2 +
			      1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */

		hdr->qw_9B[0] = opx_ep->rx->tx.dput_9B.hdr.qw_9B[0] | params->lrh_dlid | ((uint64_t) lrh_dws << 32);
		hdr->qw_9B[1] = opx_ep->rx->tx.dput_9B.hdr.qw_9B[1] | params->bth_rx;
		hdr->qw_9B[2] = opx_ep->rx->tx.dput_9B.hdr.qw_9B[2];
		hdr->qw_9B[3] = opx_ep->rx->tx.dput_9B.hdr.qw_9B[3];
		hdr->qw_9B[4] = opx_ep->rx->tx.dput_9B.hdr.qw_9B[4] | FI_OPX_HFI_DPUT_OPCODE_FENCE;
		hdr->qw_9B[5] = (uint64_t) params->cc;
		hdr->qw_9B[6] = params->bytes_to_fence;
	} else {
		const uint64_t bth_rx  = params->bth_rx;
		const uint64_t pbc_dws = 2 +		     /* pbc */
					 4 +		     /* lrh uncompressed */
					 3 +		     /* bth */
					 9 +		     /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 2;		     /* ICRC/tail */
		const uint16_t lrh_dws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
		hdr->qw_16B[0]	       = opx_ep->rx->tx.dput_16B.hdr.qw_16B[0] |
				 ((uint64_t) (params->lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B)
				  << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
				 ((uint64_t) lrh_dws << 20);
		hdr->qw_16B[1] = opx_ep->rx->tx.dput_16B.hdr.qw_16B[1] |
				 ((uint64_t) ((params->lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					      OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				 (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
		hdr->qw_16B[2] = opx_ep->rx->tx.dput_16B.hdr.qw_16B[2] | bth_rx;
		hdr->qw_16B[3] = opx_ep->rx->tx.dput_16B.hdr.qw_16B[3];
		hdr->qw_16B[4] = opx_ep->rx->tx.dput_16B.hdr.qw_16B[4];
		hdr->qw_16B[5] = opx_ep->rx->tx.dput_16B.hdr.qw_16B[5] | FI_OPX_HFI_DPUT_OPCODE_FENCE | (0ULL << 32);
		hdr->qw_16B[6] = (uintptr_t) params->cc;
		hdr->qw_16B[7] = params->bytes_to_fence;
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

	return FI_SUCCESS;
}

void opx_hfi1_dput_fence(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr, const uint8_t u8_rx,
			 const uint32_t u32_extended_rx, const enum opx_hfi1_type hfi1_type)
{
	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct fi_opx_hfi1_rx_dput_fence_params *params = &work->fence;
	params->opx_ep					= opx_ep;
	params->work_elem.slist_entry.next		= NULL;
	params->work_elem.work_fn			= opx_hfi1_do_dput_fence;
	params->work_elem.completion_action		= NULL;
	params->work_elem.payload_copy			= NULL;
	params->work_elem.complete			= false;
	params->work_elem.work_type			= OPX_WORK_TYPE_SHM;

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		params->lrh_dlid = (hdr->lrh_9B.qw[0] & 0xFFFF000000000000ul) >> 32;
	} else {
		params->lrh_dlid = hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid;
	}

	params->bth_rx		= (uint64_t) u8_rx << OPX_BTH_RX_SHIFT;
	params->u8_rx		= u8_rx;
	params->u32_extended_rx = u32_extended_rx;
	params->bytes_to_fence	= hdr->dput.target.fence.bytes_to_fence;
	params->cc		= (struct fi_opx_completion_counter *) hdr->dput.target.fence.completion_counter;
	opx_lid_t slid;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		slid = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
	} else {
		slid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
	}

	if (slid == opx_ep->rx->self.lid) {
		params->target_hfi_unit = opx_ep->rx->self.hfi1_unit;
	} else {
		struct fi_opx_hfi_local_lookup *hfi_lookup = fi_opx_hfi1_get_lid_local(slid);
		assert(hfi_lookup);
		params->target_hfi_unit = hfi_lookup->hfi_unit;
	}

	int rc = opx_hfi1_do_dput_fence(work);

	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(work);
		return;
	}
	assert(rc == -FI_EAGAIN);
	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[OPX_WORK_TYPE_SHM]);
}

int opx_hfi1_rx_rma_rts_send_cts_intranode(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rma_rts_params *params	= &work->rx_rma_rts;
	struct fi_opx_ep		     *opx_ep	= params->opx_ep;
	const uint64_t			      lrh_dlid	= params->lrh_dlid;
	const uint64_t			      bth_rx	= ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;
	const enum opx_hfi1_type	      hfi1_type = OPX_HFI1_TYPE;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RMA RTS (begin)\n");
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RMA-RTS-SHM");
	uint64_t pos;
	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a CTS packet, lrh.pktlen
	   isn't used/needed */
	uint64_t niov = params->niov << 48;
	uint64_t op64 = ((uint64_t) params->op) << 40;
	uint64_t dt64 = ((uint64_t) params->dt) << 32;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		hdr->qw_9B[0] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid;
		hdr->qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
		hdr->qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2];
		hdr->qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
		hdr->qw_9B[4] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | niov | op64 | dt64 | params->opcode;
		hdr->qw_9B[5] = (uint64_t) params->origin_rma_req;
		hdr->qw_9B[6] = (uint64_t) params->rma_req;
	} else {
		hdr->qw_16B[0] =
			opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
			((uint64_t) (lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B);
		hdr->qw_16B[1] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
				 ((uint64_t) ((lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					      OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				 (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
		hdr->qw_16B[2] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx;
		hdr->qw_16B[3] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[3];
		hdr->qw_16B[4] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[4];
		hdr->qw_16B[5] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] | niov | op64 | dt64 | params->opcode;
		hdr->qw_16B[6] = (uint64_t) params->origin_rma_req;
		hdr->qw_16B[7] = (uint64_t) params->rma_req;
	}

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RMA-RTS-SHM");
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, SHM -- RMA RTS (end)\n");

	return FI_SUCCESS;
}

int opx_hfi1_rx_rma_rts_send_cts(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rma_rts_params *params   = &work->rx_rma_rts;
	struct fi_opx_ep		     *opx_ep   = params->opx_ep;
	const uint64_t			      lrh_dlid = params->lrh_dlid;
	const uint64_t			      bth_rx   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	const enum opx_hfi1_type hfi1_type = OPX_HFI1_TYPE;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RMA RTS (begin) (params=%p rma_req=%p context=%p)\n",
	       params, params->rma_req, params->rma_req->context);
	assert(params->rma_req->context->byte_counter >= params->dput_iov[0].bytes);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RMA-CTS-HFI:%p", params->rma_req);
	const uint64_t		    payload_bytes	 = (params->niov * sizeof(union fi_opx_hfi1_dput_iov));
	union fi_opx_hfi1_pio_state pio_state		 = *opx_ep->tx->pio_state;
	const uint16_t		    total_credits_needed = 1 +		   /* packet header */
					      ((payload_bytes + 63) >> 6); /* payload blocks needed */
	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);

	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available	   = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									   total_credits_needed);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;

		if (total_credits_available < total_credits_needed) {
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== RECV, HFI -- RMA RTS (EAGAIN credits) (params=%p rzv_comp=%p context=%p)\n",
			       params, params->rma_req, params->rma_req->context);
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, hfi1_type);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== RECV, HFI -- RMA RTS (EAGAIN psn/replay) (params=%p rzv_comp=%p context=%p)\n",
		       params, params->rma_req, params->rma_req->context);
		return -FI_EAGAIN;
	}

	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

	// The "memcopy first" code is here as an alternative to the more complicated
	// direct write to pio followed by memory copy of the reliability buffer
	uint64_t niov = params->niov << 48;
	uint64_t op64 = ((uint64_t) params->op) << 40;
	uint64_t dt64 = ((uint64_t) params->dt) << 32;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		const uint64_t pbc_dws = 2 + /* pbc */
					 2 + /* lrh */
					 3 + /* bth */
					 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 ((payload_bytes + 3) >> 2);
		const uint16_t lrh_dws =
			htons(pbc_dws - 2 +
			      1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */
		replay->scb.scb_9B.qw0 = opx_ep->rx->tx.cts_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid;
		replay->scb.scb_9B.hdr.qw_9B[0] =
			opx_ep->rx->tx.cts_9B.hdr.qw_9B[0] | lrh_dlid | ((uint64_t) lrh_dws << 32);
		replay->scb.scb_9B.hdr.qw_9B[1] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[1] | bth_rx;
		replay->scb.scb_9B.hdr.qw_9B[2] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[2] | psn;
		replay->scb.scb_9B.hdr.qw_9B[3] = opx_ep->rx->tx.cts_9B.hdr.qw_9B[3];
		replay->scb.scb_9B.hdr.qw_9B[4] =
			opx_ep->rx->tx.cts_9B.hdr.qw_9B[4] | niov | op64 | dt64 | params->opcode;
		replay->scb.scb_9B.hdr.qw_9B[5] = (uint64_t) params->origin_rma_req;
		replay->scb.scb_9B.hdr.qw_9B[6] = (uint64_t) params->rma_req;
	} else {
		const uint64_t pbc_dws = 2 + /* pbc */
					 4 + /* lrh uncompressed */
					 3 + /* bth */
					 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 (((payload_bytes + 7) & -8) >> 2) + /* 16B is QW length/padded */
					 2;				     /* ICRC/tail */
		const uint16_t lrh_qws	= (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
		replay->scb.scb_16B.qw0 = opx_ep->rx->tx.cts_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) |
					  OPX_PBC_DLID_TO_PBC_DLID(lrh_dlid, OPX_HFI1_JKR);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "replay->scb_16B.qw0 = %#lx pbc_dws = %ld\n",
		       replay->scb.scb_16B.qw0, pbc_dws);
		replay->scb.scb_16B.hdr.qw_16B[0] =
			opx_ep->rx->tx.cts_16B.hdr.qw_16B[0] |
			((uint64_t) (lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
			((uint64_t) lrh_qws << 20);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "lrh_qws = %d replay->scb_16B.hdr.lrh_16B.pktlen = %d\n",
		       lrh_qws, replay->scb.scb_16B.hdr.lrh_16B.pktlen);
		replay->scb.scb_16B.hdr.qw_16B[1] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[1] |
						    ((uint64_t) ((lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
								 OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
						    (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);

		replay->scb.scb_16B.hdr.qw_16B[2] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[2] | bth_rx;
		replay->scb.scb_16B.hdr.qw_16B[3] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[3] | psn;
		replay->scb.scb_16B.hdr.qw_16B[4] = opx_ep->rx->tx.cts_16B.hdr.qw_16B[4];
		replay->scb.scb_16B.hdr.qw_16B[5] =
			opx_ep->rx->tx.cts_16B.hdr.qw_16B[5] | niov | op64 | dt64 | params->opcode;
		replay->scb.scb_16B.hdr.qw_16B[6] = (uint64_t) params->origin_rma_req;
		replay->scb.scb_16B.hdr.qw_16B[7] = (uint64_t) params->rma_req;
	}

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) replay->payload;
	assert(((uint8_t *) tx_payload) == ((uint8_t *) &replay->data));

	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
	}

#ifdef HAVE_CUDA
	if (params->dput_iov[0].rbuf_iface == FI_HMEM_CUDA) {
		int err = cuda_set_sync_memops((void *) params->dput_iov[0].rbuf);
		if (OFI_UNLIKELY(err != 0)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_MR, "cuda_set_sync_memops(%p) FAILED (returned %d)\n",
				(void *) params->dput_iov[0].rbuf, err);
		}
	}
#endif

	fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->origin_rx, psn_ptr, replay, params->reliability,
							    hfi1_type);
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RMA-CTS-HFI:%p", params->rma_comp);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== RECV, HFI -- RMA RTS (end) (params=%p rma_req=%p context=%p)\n",
	       params, params->rma_req, params->rma_req->context);
	return FI_SUCCESS;
}

void fi_opx_hfi1_rx_rma_rts(struct fi_opx_ep *opx_ep, const union opx_hfi1_packet_hdr *const hdr,
			    const void *const payload, const uint64_t niov, uintptr_t origin_rma_req,
			    struct opx_context *const target_context, const uintptr_t dst_vaddr,
			    const enum fi_hmem_iface dst_iface, const uint64_t dst_device,
			    const union fi_opx_hfi1_dput_iov *src_iovs, const unsigned is_intranode,
			    const enum ofi_reliability_kind reliability, const enum opx_hfi1_type hfi1_type)
{
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "RECV-RMA-RTS-HFI:%ld", hdr->qw_9B[6]);
	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct fi_opx_hfi1_rx_rma_rts_params *params = &work->rx_rma_rts;
	params->work_elem.slist_entry.next	     = NULL;
	params->opx_ep				     = opx_ep;

	opx_lid_t lid;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		lid	     = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
		params->slid = lid;
		if (hfi1_type & OPX_HFI1_WFR) {
			params->lrh_dlid = (hdr->lrh_9B.qw[0] & 0xFFFF000000000000ul) >> 32;
		} else {
			params->lrh_dlid = hdr->lrh_9B.slid << 16;
		}
	} else {
		lid		 = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
		params->slid	 = lid;
		params->lrh_dlid = lid;
	}
	params->pbc_dlid = OPX_PBC_DLID_TO_PBC_DLID(lid, hfi1_type);

	assert(niov <= MIN(FI_OPX_MAX_HMEM_IOV, FI_OPX_MAX_DPUT_IOV));
	params->niov				      = niov;
	const union fi_opx_hfi1_dput_iov *src_iov     = src_iovs;
	uint64_t			  rbuf_offset = 0;
	for (int i = 0; i < niov; i++) {
		params->dput_iov[i].sbuf	= src_iov->sbuf;
		params->dput_iov[i].sbuf_iface	= src_iov->sbuf_iface;
		params->dput_iov[i].sbuf_device = src_iov->sbuf_device;
		params->dput_iov[i].rbuf	= dst_vaddr + rbuf_offset;
		params->dput_iov[i].rbuf_iface	= dst_iface;
		params->dput_iov[i].rbuf_device = dst_device;
		params->dput_iov[i].bytes	= src_iov->bytes;
		rbuf_offset += src_iov->bytes;
		++src_iov;
	}
	target_context->len = target_context->byte_counter = rbuf_offset;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "is_intranode=%u niov=%lu opcode=%u\n", is_intranode, niov,
	       params->opcode);
	if (is_intranode) {
		params->work_elem.work_fn   = opx_hfi1_rx_rma_rts_send_cts_intranode;
		params->work_elem.work_type = OPX_WORK_TYPE_SHM;

		const uint64_t lid = params->slid;
		if (lid == opx_ep->rx->self.lid) {
			params->target_hfi_unit = opx_ep->rx->self.hfi1_unit;
		} else {
			struct fi_opx_hfi_local_lookup *hfi_lookup = fi_opx_hfi1_get_lid_local(lid);
			assert(hfi_lookup);
			params->target_hfi_unit = hfi_lookup->hfi_unit;
		}
	} else {
		params->work_elem.work_fn   = opx_hfi1_rx_rma_rts_send_cts;
		params->work_elem.work_type = OPX_WORK_TYPE_PIO;
		params->target_hfi_unit	    = 0xFF;
	}
	params->work_elem.completion_action = NULL;
	params->work_elem.payload_copy	    = NULL;
	params->work_elem.complete	    = false;

	params->u32_extended_rx = fi_opx_ep_get_u32_extended_rx(opx_ep, is_intranode, hdr->rma_rts.origin_rx);
	params->reliability	= reliability;
	params->origin_rx	= hdr->rma_rts.origin_rx;
	params->is_intranode	= is_intranode;
	params->opcode		= FI_OPX_HFI_DPUT_OPCODE_PUT_CQ;
	params->u8_rx		= hdr->rma_rts.origin_rx;
	params->dt		= hdr->rma_rts.dt;
	params->op		= hdr->rma_rts.op;

	params->origin_rma_req	     = (struct fi_opx_rma_request *) origin_rma_req;
	params->rma_req		     = ofi_buf_alloc(opx_ep->tx->rma_request_pool);
	params->rma_req->context     = target_context;
	params->rma_req->hmem_device = dst_device;
	params->rma_req->hmem_iface  = dst_iface;

	int rc = params->work_elem.work_fn(work);
	if (rc == FI_SUCCESS) {
		OPX_BUF_FREE(work);
		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "RECV-RMA-RTS-HFI:%ld", hdr->qw_9B[6]);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_SUCCESS\n");
		return;
	}
	assert(rc == -FI_EAGAIN);
	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[params->work_elem.work_type]);
	OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "RECV-RMA-RTS-HFI:%ld", hdr->qw_9B[6]);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
}

int opx_hfi1_tx_rma_rts(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rma_rts_params *params   = &work->rx_rma_rts;
	struct fi_opx_ep		     *opx_ep   = params->opx_ep;
	const uint64_t			      lrh_dlid = params->lrh_dlid;
	const uint64_t			      bth_rx   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "===================================== SEND, HFI -- RMA RTS (begin) (params=%p origin_rma_req=%p cc=%p)\n",
	       params, params->origin_rma_req, params->origin_rma_req->cc);
	assert(params->origin_rma_req->cc->byte_counter >= params->dput_iov[0].bytes);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RMA-RTS-HFI:%p", params->origin_rma_req);

	const uint64_t payload_bytes = (params->niov * sizeof(union fi_opx_hfi1_dput_iov));
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "payload_bytes = %ld\n", payload_bytes);
	union fi_opx_hfi1_pio_state pio_state		 = *opx_ep->tx->pio_state;
	const uint16_t		    total_credits_needed = 1 +		   /* packet header */
					      ((payload_bytes + 63) >> 6); /* payload blocks needed */
	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);

	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									total_credits_needed);
		if (total_credits_available < total_credits_needed) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
			       "===================================== SEND, HFI -- RMA RTS (EAGAIN credits) (params=%p origin_rma_req=%p cc=%p)\n",
			       params, params->origin_rma_req, params->origin_rma_req->cc);
			OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "RECV-RMA-RTS-HFI:%ld", hdr->qw_9B[6]);
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
					    params->origin_rs, &psn_ptr, &replay, params->reliability, OPX_HFI1_TYPE);
	if (OFI_UNLIKELY(psn == -1)) {
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "===================================== SEND, HFI -- RMA RTS (EAGAIN psn/replay) (params=%p origin_rma_req=%p cc=%p) opcode=%d\n",
		       params, params->origin_rma_req, params->origin_rma_req->cc, params->opcode);
		OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "RECV-RMA-RTS-HFI:%p", params->origin_rma_req);
		return -FI_EAGAIN;
	}

	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);

	// The "memcopy first" code is here as an alternative to the more complicated
	// direct write to pio followed by memory copy of the reliability buffer

	uint64_t cq_data = ((uint64_t) params->data) << 32;
	uint64_t niov	 = params->niov << 48;
	uint64_t op64	 = ((uint64_t) params->op) << 40;
	uint64_t dt64	 = ((uint64_t) params->dt) << 32;
	assert(params->dt == (FI_VOID - 1) || params->dt < FI_DATATYPE_LAST);
	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		const uint64_t pbc_dws = 2 +		       /* pbc */
					 2 +		       /* lhr */
					 3 +		       /* bth */
					 9 +		       /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 (payload_bytes << 4); /* payload blocks for rma data */

		const uint16_t lrh_dws =
			htons(pbc_dws - 2 +
			      1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */

		replay->scb.scb_9B.qw0 =
			opx_ep->rx->tx.rma_rts_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid;
		replay->scb.scb_9B.hdr.qw_9B[0] =
			opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[0] | lrh_dlid | ((uint64_t) lrh_dws << 32);
		replay->scb.scb_9B.hdr.qw_9B[1] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[1] | bth_rx;
		replay->scb.scb_9B.hdr.qw_9B[2] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[2] | psn;
		replay->scb.scb_9B.hdr.qw_9B[3] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[3] | cq_data;
		replay->scb.scb_9B.hdr.qw_9B[4] =
			opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[4] | niov | op64 | dt64 | params->opcode;
		replay->scb.scb_9B.hdr.qw_9B[5] = params->key;
		replay->scb.scb_9B.hdr.qw_9B[6] = (uint64_t) params->origin_rma_req;
	} else {
		const uint64_t pbc_dws = 2 + /* pbc */
					 4 + /* lrh uncompressed */
					 3 + /* bth */
					 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					 (((payload_bytes + 7) & -8) >> 2) + /* 16B is QW length/padded */
					 2;				     /* ICRC/tail */
		const uint16_t lrh_qws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
		replay->scb.scb_16B.qw0 =
			opx_ep->rx->tx.rma_rts_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | params->pbc_dlid;
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "replay->scb_16B.qw0 = %#lx pbc_dws = %ld\n",
		       replay->scb.scb_16B.qw0, pbc_dws);
		replay->scb.scb_16B.hdr.qw_16B[0] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[0] |
						    ((uint64_t) (params->lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B)
						     << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
						    ((uint64_t) lrh_qws << 20);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA, "lrh_qws = %d replay->scb_16B.hdr.lrh_16B.pktlen = %d\n",
		       lrh_qws, replay->scb.scb_16B.hdr.lrh_16B.pktlen);
		replay->scb.scb_16B.hdr.qw_16B[1] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[1] |
						    ((uint64_t) ((params->lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
								 OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
						    (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);

		replay->scb.scb_16B.hdr.qw_16B[2] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[2] | bth_rx;
		replay->scb.scb_16B.hdr.qw_16B[3] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[3] | psn;
		replay->scb.scb_16B.hdr.qw_16B[4] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[4] | cq_data;
		replay->scb.scb_16B.hdr.qw_16B[5] =
			opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[5] | niov | op64 | dt64 | params->opcode;
		replay->scb.scb_16B.hdr.qw_16B[6] = params->key;
		replay->scb.scb_16B.hdr.qw_16B[7] = (uint64_t) params->origin_rma_req;
	}

	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) replay->payload;
	assert(((uint8_t *) tx_payload) == ((uint8_t *) &replay->data));

	for (int i = 0; i < params->niov; i++) {
		tx_payload->rma_rts.iov[i] = params->dput_iov[i];
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "fi_opx_reliability_service_do_replay &opx_ep->reliability->service %p, replay %p\n",
		     &opx_ep->reliability->service, replay);
	fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, params->origin_rs,
							    params->u8_rx, psn_ptr, replay, params->reliability,
							    OPX_HFI1_TYPE);

	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RMA-RTS-HFI:%p", params->origin_rma_req);
	FI_DBG_TRACE(
		fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- RMA RTS (end) (params=%p origin_rma_req=%p cc=%p)\n",
		params, params->origin_rma_req, params->origin_rma_req->cc);

	params->work_elem.complete = true;
	return FI_SUCCESS;
}

int opx_hfi1_tx_rma_rts_intranode(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_rx_rma_rts_params *params   = &work->rx_rma_rts;
	struct fi_opx_ep		     *opx_ep   = params->opx_ep;
	const uint64_t			      lrh_dlid = params->lrh_dlid;
	const uint64_t			      bth_rx   = ((uint64_t) params->u8_rx) << OPX_BTH_RX_SHIFT;
	uint64_t			      pos;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, SHM -- RENDEZVOUS RMA (begin) context %p\n", NULL);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-RMA-SHM");

	/* Possible SHM connections required for certain applications (i.e., DAOS)
	 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
	 */
	ssize_t rc = fi_opx_shm_dynamic_tx_connect(OPX_INTRANODE_TRUE, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

	if (OFI_UNLIKELY(rc)) {
		return -FI_EAGAIN;
	}

	union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
		&opx_ep->tx->shm, params->target_hfi_unit, params->u8_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
		params->u32_extended_rx, opx_ep->daos_info.rank_inst, &rc);

	if (!hdr) {
		return rc;
	}

	/* Note that we do not set stl.hdr.lrh.pktlen here (usually lrh_dws << 32),
	   because this is intranode and since it's a RTS packet, lrh.pktlen
	   isn't used/needed */

	uint64_t cq_data = ((uint64_t) params->data) << 32;
	uint64_t niov	 = params->niov << 48;
	uint64_t op64	 = ((uint64_t) params->op) << 40;
	uint64_t dt64	 = ((uint64_t) params->dt) << 32;
	assert(params->dt == (FI_VOID - 1) || params->dt < FI_DATATYPE_LAST);

	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		hdr->qw_9B[0] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[0] | lrh_dlid;
		hdr->qw_9B[1] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[1] | bth_rx;
		hdr->qw_9B[2] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[2];
		hdr->qw_9B[3] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[3] | cq_data;
		hdr->qw_9B[4] = opx_ep->rx->tx.rma_rts_9B.hdr.qw_9B[4] | niov | op64 | dt64 | params->opcode;
		hdr->qw_9B[5] = params->key;
		hdr->qw_9B[6] = (uint64_t) params->origin_rma_req;
	} else {
		hdr->qw_16B[0] =
			opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[0] |
			((uint64_t) ((lrh_dlid & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B));
		hdr->qw_16B[1] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[1] |
				 ((uint64_t) ((lrh_dlid & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					      OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				 (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
		hdr->qw_16B[2] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[2] | bth_rx;
		hdr->qw_16B[3] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[3];
		hdr->qw_16B[4] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[4] | cq_data;
		hdr->qw_16B[5] = opx_ep->rx->tx.rma_rts_16B.hdr.qw_16B[5] | niov | op64 | dt64 | params->opcode;
		hdr->qw_16B[6] = params->key;
		hdr->qw_16B[7] = (uint64_t) params->origin_rma_req;
	}
	union fi_opx_hfi1_packet_payload *const tx_payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

	uintptr_t vaddr_with_offset = 0; /* receive buffer virtual address */
	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i] = params->dput_iov[i];
		tx_payload->cts.iov[i].rbuf += vaddr_with_offset;
		vaddr_with_offset += params->dput_iov[i].bytes;
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-RTS-SHM");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, SHM -- RENDEZVOUS RTS (end) context %p\n", NULL);

	params->work_elem.complete = true;
	return FI_SUCCESS;
}

int fi_opx_hfi1_do_dput(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_dput_params	       *params			  = &work->dput;
	struct fi_opx_ep		       *opx_ep			  = params->opx_ep;
	struct fi_opx_mr		       *opx_mr			  = params->opx_mr;
	const uint8_t				u8_rx			  = params->u8_rx;
	const uint32_t				niov			  = params->niov;
	const union fi_opx_hfi1_dput_iov *const dput_iov		  = params->dput_iov;
	const uintptr_t				target_byte_counter_vaddr = params->target_byte_counter_vaddr;
	uint64_t			       *origin_byte_counter	  = params->origin_byte_counter;
	uint64_t				key			  = params->key;
	struct fi_opx_completion_counter       *cc			  = params->cc;
	uint64_t				op64			  = params->op;
	uint64_t				dt64			  = params->dt;
	uint32_t				opcode			  = params->opcode;
	const unsigned				is_intranode		  = params->is_intranode;
	const enum ofi_reliability_kind		reliability		  = params->reliability;
	/* use the slid from the lrh header of the incoming packet
	 * as the dlid for the lrh header of the outgoing packet */
	const enum opx_hfi1_type hfi1_type = OPX_HFI1_TYPE;
	const uint64_t		 lrh_dlid  = params->lrh_dlid;
	const uint64_t		 bth_rx	   = ((uint64_t) u8_rx) << OPX_BTH_RX_SHIFT;

	enum fi_hmem_iface cbuf_iface  = params->compare_iov.iface;
	uint64_t	   cbuf_device = params->compare_iov.device;

	assert((opx_ep->tx->pio_max_eager_tx_bytes & 0x3fu) == 0);
	unsigned    i;
	const void *sbuf_start = (opx_mr == NULL) ? 0 : opx_mr->iov.iov_base;

	/* Note that lrh_dlid is just the version of params->slid shifted so
	   that it can be OR'd into the correct position in the packet header */
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		assert(__cpu24_to_be16(params->slid) == (lrh_dlid >> 16));
	} else {
		assert(params->slid == lrh_dlid);
	}

	uint64_t max_bytes_per_packet;

	ssize_t rc;
	if (is_intranode) {
		/* Possible SHM connections required for certain applications (i.e., DAOS)
		 * exceeds the max value of the legacy u8_rx field.  Use u32_extended field.
		 */
		rc = fi_opx_shm_dynamic_tx_connect(params->is_intranode, opx_ep, params->u32_extended_rx,
						   params->target_hfi_unit);

		if (OFI_UNLIKELY(rc)) {
			return -FI_EAGAIN;
		}

		max_bytes_per_packet = FI_OPX_HFI1_PACKET_MTU;
	} else {
		max_bytes_per_packet = opx_ep->tx->pio_flow_eager_tx_bytes;
	}

	assert(((opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH ||
		 opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH) &&
		params->payload_bytes_for_iovec == sizeof(struct fi_opx_hfi1_dput_fetch)) ||
	       (opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH &&
		opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH && params->payload_bytes_for_iovec == 0));

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND DPUT, %s opcode %d -- (begin)\n",
		     is_intranode ? "SHM" : "HFI", opcode);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-DPUT-%s", is_intranode ? "SHM" : "HFI");

	for (i = params->cur_iov; i < niov; ++i) {
		uint8_t *sbuf =
			(uint8_t *) ((uintptr_t) sbuf_start + (uintptr_t) dput_iov[i].sbuf + params->bytes_sent);
		uintptr_t rbuf = dput_iov[i].rbuf + params->bytes_sent;

		enum fi_hmem_iface sbuf_iface  = dput_iov[i].sbuf_iface;
		uint64_t	   sbuf_device = dput_iov[i].sbuf_device;

		uint64_t bytes_to_send = dput_iov[i].bytes - params->bytes_sent;
		while (bytes_to_send > 0) {
			uint64_t bytes_to_send_this_packet;
			uint64_t blocks_to_send_in_this_packet;
			uint64_t pbc_dws;
			uint16_t lrh_dws;
			if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
				bytes_to_send_this_packet =
					MIN(bytes_to_send + params->payload_bytes_for_iovec, max_bytes_per_packet);
				uint64_t tail_bytes	      = bytes_to_send_this_packet & 0x3Ful;
				blocks_to_send_in_this_packet = (bytes_to_send_this_packet >> 6) + (tail_bytes ? 1 : 0);
				pbc_dws			      = 2 + /* pbc */
					  2 +			    /* lrh */
					  3 +			    /* bth */
					  9 +			    /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					  (blocks_to_send_in_this_packet << 4);
				lrh_dws = htons(pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8 bytes), but
								     does include icrc (4 bytes) */
			} else {
				/* 1 QW for hdr that spills to 2nd cacheline + 1 QW for ICRC/tail */
				const uint64_t additional_hdr_tail_byte = 2 * 8;
				uint64_t       payload_n_additional_hdr_tail_bytes =
					(MIN(bytes_to_send + params->payload_bytes_for_iovec + additional_hdr_tail_byte,
					     max_bytes_per_packet));
				uint64_t tail_bytes = payload_n_additional_hdr_tail_bytes & 0x3Ful;
				blocks_to_send_in_this_packet =
					(payload_n_additional_hdr_tail_bytes >> 6) + (tail_bytes ? 1 : 0);
				bytes_to_send_this_packet =
					payload_n_additional_hdr_tail_bytes - additional_hdr_tail_byte;
				pbc_dws = 2 + /* pbc */
					  4 + /* lrh uncompressed */
					  3 + /* bth */
					  7 + /* kdeth */
					  (blocks_to_send_in_this_packet
					   << 4); // ICRC and the kdeth in the second cacheline are accounted for here
				lrh_dws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
			}

			uint64_t bytes_sent;
			if (is_intranode) {
				uint64_t		   pos;
				union opx_hfi1_packet_hdr *hdr =
					opx_shm_tx_next(&opx_ep->tx->shm, params->target_hfi_unit, u8_rx, &pos,
							opx_ep->daos_info.hfi_rank_enabled, params->u32_extended_rx,
							opx_ep->daos_info.rank_inst, &rc);

				if (!hdr) {
					return rc;
				}

				union fi_opx_hfi1_packet_payload *const tx_payload =
					(union fi_opx_hfi1_packet_payload *) (hdr + 1);

				bytes_sent = opx_hfi1_dput_write_header_and_payload(
					opx_ep, hdr, tx_payload, opcode, 0, lrh_dws, op64, dt64, lrh_dlid, bth_rx,
					bytes_to_send_this_packet, key, (const uint64_t) params->fetch_vaddr,
					target_byte_counter_vaddr, params->rma_request_vaddr, params->bytes_sent, &sbuf,
					sbuf_iface, sbuf_device, (uint8_t **) &params->compare_vaddr, cbuf_iface,
					cbuf_device, &rbuf, hfi1_type);

				opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);
			} else {
				union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

				const uint16_t credits_needed	       = blocks_to_send_in_this_packet + 1 /* header */;
				uint32_t       total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(
					      pio_state, &opx_ep->tx->force_credit_return, credits_needed);

				if (total_credits_available < (uint32_t) credits_needed) {
					fi_opx_compiler_msync_writes();
					FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
					total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(
						pio_state, &opx_ep->tx->force_credit_return, credits_needed);
					if (total_credits_available < (uint32_t) credits_needed) {
						opx_ep->tx->pio_state->qw0 = pio_state.qw0;
						return -FI_EAGAIN;
					}
				}

				struct fi_opx_reliability_tx_replay *replay;
				union fi_opx_reliability_tx_psn	    *psn_ptr;
				int64_t				     psn;

				psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state,
								    params->slid, u8_rx, params->origin_rs, &psn_ptr,
								    &replay, reliability, hfi1_type);
				if (OFI_UNLIKELY(psn == -1)) {
					return -FI_EAGAIN;
				}

				assert(replay != NULL);
				union fi_opx_hfi1_packet_payload *replay_payload =
					(union fi_opx_hfi1_packet_payload *) replay->payload;
				assert(!replay->use_iov);
				assert(((uint8_t *) replay_payload) == ((uint8_t *) &replay->data));

				if (hfi1_type & OPX_HFI1_JKR) {
					replay->scb.scb_16B.qw0 =
						opx_ep->rx->tx.dput_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) |
						OPX_PBC_CR(opx_ep->tx->force_credit_return, hfi1_type) |
						params->pbc_dlid;
				} else {
					replay->scb.scb_9B.qw0 =
						opx_ep->rx->tx.dput_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) |
						OPX_PBC_CR(opx_ep->tx->force_credit_return, hfi1_type) |
						params->pbc_dlid;
				}
				bytes_sent = opx_hfi1_dput_write_header_and_payload(
					opx_ep, OPX_REPLAY_HDR(replay), replay_payload, opcode, psn, lrh_dws, op64,
					dt64, lrh_dlid, bth_rx, bytes_to_send_this_packet, key,
					(const uint64_t) params->fetch_vaddr, target_byte_counter_vaddr,
					params->rma_request_vaddr, params->bytes_sent, &sbuf, sbuf_iface, sbuf_device,
					(uint8_t **) &params->compare_vaddr, cbuf_iface, cbuf_device, &rbuf, hfi1_type);

				FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

				if (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT || opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ) {
					if (bytes_to_send == bytes_sent) {
						/* This is the last packet to send for this PUT.
						   Turn on the immediate ACK request bit so the
						   user gets control of their buffer back ASAP */
						const uint64_t set_ack_bit = (uint64_t) htonl(0x80000000);
						if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
							replay->scb.scb_9B.hdr.qw_9B[2] |= set_ack_bit;
							replay->scb.scb_9B.hdr.dput.target.last_bytes =
								replay->scb.scb_9B.hdr.dput.target.bytes;
						} else {
							replay->scb.scb_16B.hdr.qw_16B[3] |= set_ack_bit;
							replay->scb.scb_16B.hdr.dput.target.last_bytes =
								replay->scb.scb_16B.hdr.dput.target.bytes;
						}
					}
					fi_opx_reliability_client_replay_register_with_update(
						&opx_ep->reliability->state, params->slid, params->origin_rs, u8_rx,
						psn_ptr, replay, cc, bytes_sent, reliability);

					fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
				} else {
					fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
					fi_opx_compiler_msync_writes();

					fi_opx_reliability_client_replay_register_no_update(
						&opx_ep->reliability->state, params->origin_rs, u8_rx, psn_ptr, replay,
						reliability, hfi1_type);
				}
			}

			bytes_to_send -= bytes_sent;
			params->bytes_sent += bytes_sent;

			if (origin_byte_counter) {
				*origin_byte_counter -= bytes_sent;
				assert(((int64_t) *origin_byte_counter) >= 0);
			}
		} /* while bytes_to_send */

		if ((opcode == FI_OPX_HFI_DPUT_OPCODE_PUT || opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ) &&
		    is_intranode) { // RMA-type put, so send a ping/fence to better latency
			fi_opx_shm_write_fence(opx_ep, params->target_hfi_unit, u8_rx, lrh_dlid, cc, params->bytes_sent,
					       params->u32_extended_rx, hfi1_type);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-DPUT-%s", is_intranode ? "SHM" : "HFI");
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND DPUT, %s finished IOV=%d bytes_sent=%ld -- (end)\n",
			is_intranode ? "SHM" : "HFI", params->cur_iov, params->bytes_sent);

		params->bytes_sent = 0;
		params->cur_iov++;
	} /* for niov */

	params->work_elem.complete = true;
	return FI_SUCCESS;
}

__OPX_FORCE_INLINE__
void fi_opx_hfi1_dput_copy_to_bounce_buf(uint32_t opcode, uint8_t *target_buf, uint8_t *source_buf,
					 uint8_t *compare_buf, void *fetch_vaddr, uintptr_t target_byte_counter_vaddr,
					 uint64_t buf_packet_bytes, uint64_t total_bytes, uint64_t bytes_sent,
					 enum fi_hmem_iface sbuf_iface, uint64_t sbuf_device,
					 enum fi_hmem_iface cbuf_iface, uint64_t cbuf_device)
{
	if (opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH) {
		while (total_bytes) {
			size_t dput_bytes = MIN(buf_packet_bytes, total_bytes);

			opx_hfi1_dput_write_payload_atomic_fetch((union fi_opx_hfi1_packet_payload *) target_buf,
								 dput_bytes, (const uint64_t) fetch_vaddr,
								 target_byte_counter_vaddr, bytes_sent, source_buf,
								 sbuf_iface, sbuf_device);

			target_buf += dput_bytes + sizeof(struct fi_opx_hfi1_dput_fetch);
			source_buf += dput_bytes;
			bytes_sent += dput_bytes;

			total_bytes -= dput_bytes;
		}
	} else if (opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH) {
		buf_packet_bytes >>= 1;
		while (total_bytes) {
			size_t dput_bytes      = MIN(buf_packet_bytes, total_bytes);
			size_t dput_bytes_half = dput_bytes >> 1;

			opx_hfi1_dput_write_payload_atomic_compare_fetch(
				(union fi_opx_hfi1_packet_payload *) target_buf, dput_bytes_half,
				(const uint64_t) fetch_vaddr, target_byte_counter_vaddr, bytes_sent, source_buf,
				sbuf_iface, sbuf_device, compare_buf, cbuf_iface, cbuf_device);

			target_buf += dput_bytes + sizeof(struct fi_opx_hfi1_dput_fetch);
			source_buf += dput_bytes_half;
			compare_buf += dput_bytes_half;
			bytes_sent += dput_bytes;

			total_bytes -= dput_bytes;
		}
	} else {
		assert(total_bytes <= FI_OPX_HFI1_SDMA_WE_BUF_LEN);
		OPX_HMEM_COPY_FROM(target_buf, source_buf, total_bytes, OPX_HMEM_NO_HANDLE,
				   OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET, sbuf_iface, sbuf_device);
	}
}

int fi_opx_hfi1_do_dput_sdma(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_dput_params	       *params			  = &work->dput;
	struct fi_opx_ep		       *opx_ep			  = params->opx_ep;
	struct fi_opx_mr		       *opx_mr			  = params->opx_mr;
	const uint8_t				u8_rx			  = params->u8_rx;
	const uint32_t				niov			  = params->niov;
	const union fi_opx_hfi1_dput_iov *const dput_iov		  = params->dput_iov;
	const uintptr_t				target_byte_counter_vaddr = params->target_byte_counter_vaddr;
	uint64_t				key			  = params->key;
	uint64_t				op64			  = params->op;
	uint64_t				dt64			  = params->dt;
	uint32_t				opcode			  = params->opcode;
	const enum ofi_reliability_kind		reliability		  = params->reliability;
	/* use the slid from the lrh header of the incoming packet
	 * as the dlid for the lrh header of the outgoing packet */
	const enum opx_hfi1_type hfi1_type = OPX_HFI1_TYPE;
	const uint64_t		 lrh_dlid  = params->lrh_dlid;
	const uint64_t		 bth_rx	   = ((uint64_t) u8_rx) << OPX_BTH_RX_SHIFT;
	assert((opx_ep->tx->pio_max_eager_tx_bytes & 0x3fu) == 0);
	unsigned    i;
	const void *sbuf_start	       = (opx_mr == NULL) ? 0 : opx_mr->iov.iov_base;
	const bool  sdma_no_bounce_buf = params->sdma_no_bounce_buf;

	/* Note that lrh_dlid is just the version of params->slid shifted so
	   that it can be OR'd into the correct position in the packet header */
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		assert(__cpu24_to_be16(params->slid) == (lrh_dlid >> 16));
		assert(opx_ep->rx->tx.dput_9B.hdr.lrh_9B.slid != params->slid);
	} else {
		assert(params->slid == lrh_dlid);
	}

	// We should never be in this function for intranode ops
	assert(!params->is_intranode);

	assert(((opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH ||
		 opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH) &&
		params->payload_bytes_for_iovec == sizeof(struct fi_opx_hfi1_dput_fetch)) ||
	       (opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH &&
		opcode != FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH && params->payload_bytes_for_iovec == 0));

	assert((opcode == FI_OPX_HFI_DPUT_OPCODE_PUT && params->sdma_no_bounce_buf) ||
	       (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ && params->sdma_no_bounce_buf) ||
	       (opcode == FI_OPX_HFI_DPUT_OPCODE_GET && params->sdma_no_bounce_buf) ||
	       (opcode != FI_OPX_HFI_DPUT_OPCODE_PUT && opcode != FI_OPX_HFI_DPUT_OPCODE_PUT_CQ &&
		opcode != FI_OPX_HFI_DPUT_OPCODE_GET));

	uint64_t max_eager_bytes = opx_ep->tx->pio_max_eager_tx_bytes;
	uint64_t max_dput_bytes	 = max_eager_bytes - params->payload_bytes_for_iovec;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "%p:===================================== SEND DPUT SDMA, opcode %X -- (begin)\n", params, opcode);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-DPUT-SDMA:%p:%ld", (void *) target_byte_counter_vaddr,
			 dput_iov[params->cur_iov].bytes);

	for (i = params->cur_iov; i < niov; ++i) {
		uint8_t *sbuf =
			(uint8_t *) ((uintptr_t) sbuf_start + (uintptr_t) dput_iov[i].sbuf + params->bytes_sent);
		uintptr_t rbuf = dput_iov[i].rbuf + params->bytes_sent;

		uint64_t bytes_to_send = dput_iov[i].bytes - params->bytes_sent;
		while (bytes_to_send > 0) {
			if (!fi_opx_hfi1_sdma_queue_has_room(opx_ep, OPX_SDMA_NONTID_IOV_COUNT)) {
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA QUEUE FULL FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND-DPUT-SDMA:%p",
						 (void *) target_byte_counter_vaddr);
				return -FI_EAGAIN;
			}
			if (!params->sdma_we) {
				/* Get an SDMA work entry since we don't already have one */
				params->sdma_we = opx_sdma_get_new_work_entry(opx_ep, &params->sdma_reqs_used,
									      &params->sdma_reqs, params->sdma_we);
				if (!params->sdma_we) {
					FI_OPX_DEBUG_COUNTERS_INC_COND(
						(params->sdma_reqs_used < FI_OPX_HFI1_SDMA_MAX_WE_PER_REQ),
						opx_ep->debug_counters.sdma.eagain_sdma_we_none_free);
					FI_OPX_DEBUG_COUNTERS_INC_COND(
						(params->sdma_reqs_used == FI_OPX_HFI1_SDMA_MAX_WE_PER_REQ),
						opx_ep->debug_counters.sdma.eagain_sdma_we_max_used);
					FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
					       "%p:===================================== SEND DPUT SDMA, !WE FI_EAGAIN\n",
					       params);
					OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND-DPUT-SDMA:%p",
							 (void *) target_byte_counter_vaddr);
					return -FI_EAGAIN;
				}
				assert(params->sdma_we->total_payload == 0);
				fi_opx_hfi1_sdma_init_we(params->sdma_we, params->cc, params->slid, params->origin_rs,
							 params->u8_rx, dput_iov[i].sbuf_iface,
							 (int) dput_iov[i].sbuf_device);
			}
			assert(!fi_opx_hfi1_sdma_has_unsent_packets(params->sdma_we));

			/* The driver treats the offset as a 4-byte value, so we
			 * need to avoid sending a payload size that would wrap
			 * that in a single SDMA send */
			uintptr_t rbuf_wrap	= (rbuf + 0x100000000ul) & 0xFFFFFFFF00000000ul;
			uint64_t  sdma_we_bytes = MIN(bytes_to_send, (rbuf_wrap - rbuf));
			uint64_t  packet_count =
				(sdma_we_bytes / max_dput_bytes) + ((sdma_we_bytes % max_dput_bytes) ? 1 : 0);

			assert(packet_count > 0);
			packet_count = MIN(packet_count, FI_OPX_HFI1_SDMA_MAX_PACKETS);

			int32_t psns_avail = fi_opx_reliability_tx_available_psns(
				&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
				params->origin_rs, &params->sdma_we->psn_ptr, packet_count, max_eager_bytes);

			if (psns_avail < (int64_t) packet_count) {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.sdma.eagain_psn);
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA, !PSN FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND-DPUT-SDMA:%p",
						 (void *) target_byte_counter_vaddr);
				return -FI_EAGAIN;
			}
			/* In the unlikely event that we'll be sending a single
			 * packet who's payload size is not a multiple of 4,
			 * we'll need to add padding, in which case we'll need
			 * to use a bounce buffer, regardless if we're
			 * doing delivery completion. This is because the
			 * SDMA engine requires the LRH DWs add up to exactly
			 * the number of bytes used to fill the packet. To do
			 * the padding, we'll copy the payload to the
			 * bounce buffer, and then add the necessary padding
			 * to the iovec length we pass to the SDMA engine.
			 * The extra pad bytes will be ignored by the receiver,
			 * since it uses the byte count in the DPUT header
			 * which will still be set correctly.
			 */
			bool need_padding = (packet_count == 1 && (sdma_we_bytes & 0x3ul));
			params->sdma_we->use_bounce_buf =
				(!sdma_no_bounce_buf || opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_FETCH ||
				 opcode == FI_OPX_HFI_DPUT_OPCODE_ATOMIC_COMPARE_FETCH || need_padding);

			uint8_t *sbuf_tmp;
			bool	 replay_use_sdma;
			if (params->sdma_we->use_bounce_buf) {
				fi_opx_hfi1_dput_copy_to_bounce_buf(
					opcode, params->sdma_we->bounce_buf.buf, sbuf,
					(uint8_t *) params->compare_iov.buf, params->fetch_vaddr,
					params->target_byte_counter_vaddr, max_dput_bytes,
					MIN((packet_count * max_dput_bytes), sdma_we_bytes), params->bytes_sent,
					dput_iov[i].sbuf_iface, dput_iov[i].sbuf_device, params->compare_iov.iface,
					params->compare_iov.device);
				sbuf_tmp	= params->sdma_we->bounce_buf.buf;
				replay_use_sdma = false;
			} else {
				sbuf_tmp	= sbuf;
				replay_use_sdma = (dput_iov[i].sbuf_iface != FI_HMEM_SYSTEM);
			}
			// At this point, we have enough SDMA queue entries and PSNs
			// to send packet_count packets. The only limit now is how
			// many replays can we get.
			for (int p = 0; (p < packet_count) && sdma_we_bytes; ++p) {
				uint64_t packet_bytes =
					MIN(sdma_we_bytes, max_dput_bytes) + params->payload_bytes_for_iovec;
				assert(packet_bytes <= FI_OPX_HFI1_PACKET_MTU);

				struct fi_opx_reliability_tx_replay *replay;
				replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state, true);
				if (OFI_UNLIKELY(replay == NULL)) {
					FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
					       "%p:!REPLAY on packet %u out of %lu, params->sdma_we->num_packets %u\n",
					       params, p, packet_count, params->sdma_we->num_packets);
					break;
				}
				replay->use_sdma = replay_use_sdma;

				// Round packet_bytes up to the next multiple of 4,
				// then divide by 4 to get the correct number of dws.
				uint64_t payload_dws = ((packet_bytes + 3) & -4) >> 2;
				uint64_t pbc_dws;
				uint16_t lrh_dws;
				if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
					pbc_dws = 2 + /* pbc */
						  2 + /* lrh */
						  3 + /* bth */
						  9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
						  payload_dws;
					lrh_dws = htons(pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8
									     bytes), but does include icrc (4 bytes) */
				} else {
					pbc_dws = 2 + /* pbc */
						  4 + /* lrh uncompressed */
						  3 + /* bth */
						  9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
						  2 + /* ICRC/tail */
						  payload_dws;
					lrh_dws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
				}

				assert(replay != NULL);

				if (OPX_HFI1_TYPE & OPX_HFI1_JKR) {
					replay->scb.scb_16B.qw0 = opx_ep->rx->tx.dput_16B.qw0 |
								  OPX_PBC_LEN(pbc_dws, OPX_HFI1_TYPE) |
								  params->pbc_dlid;
				} else {
					replay->scb.scb_9B.qw0 = opx_ep->rx->tx.dput_9B.qw0 |
								 OPX_PBC_LEN(pbc_dws, OPX_HFI1_TYPE) | params->pbc_dlid;
				}

				uint64_t bytes_sent = opx_hfi1_dput_write_header_and_iov(
					opx_ep, OPX_REPLAY_HDR(replay), replay->iov, opcode, lrh_dws, op64, dt64,
					lrh_dlid, bth_rx, packet_bytes, key, (const uint64_t) params->fetch_vaddr,
					target_byte_counter_vaddr, params->rma_request_vaddr, params->bytes_sent,
					&sbuf_tmp, (uint8_t **) &params->compare_vaddr, &rbuf, OPX_HFI1_TYPE);
				params->cc->byte_counter += params->payload_bytes_for_iovec;
				fi_opx_hfi1_sdma_add_packet(params->sdma_we, replay, packet_bytes);

				bytes_to_send -= bytes_sent;
				sdma_we_bytes -= bytes_sent;
				params->bytes_sent += bytes_sent;
				params->origin_bytes_sent += bytes_sent;
				sbuf += bytes_sent;
			}

			// Must be we had trouble getting a replay buffer
			if (OFI_UNLIKELY(params->sdma_we->num_packets == 0)) {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.sdma.eagain_replay);
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA, !REPLAY FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN, "SEND-DPUT-SDMA:%p",
						 (void *) target_byte_counter_vaddr);
				return -FI_EAGAIN;
			}

			opx_hfi1_sdma_flush(opx_ep, params->sdma_we, &params->sdma_reqs, 0, /* do not use tid */
					    NULL, 0, 0, 0, 0, reliability);
			params->sdma_we = NULL;

		} /* while bytes_to_send */

		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "%p:===================================== SEND DPUT SDMA, finished IOV=%d(%d) bytes_sent=%ld\n",
		       params, params->cur_iov, niov, params->bytes_sent);

		params->bytes_sent = 0;
		params->cur_iov++;
	} /* for niov */
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-DPUT-SDMA:%p", (void *) target_byte_counter_vaddr);
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "%p:===================================== SEND DPUT SDMA, exit (end)\n", params);

	// At this point, all SDMA WE should have succeeded sending, and only reside on the reqs list
	assert(params->sdma_we == NULL);
	assert(!slist_empty(&params->sdma_reqs));

	// If we're not doing delivery completion, the user's payload would have
	// been copied to bounce buffer(s), so at this point, it should be safe
	// for the user to alter the send buffer even though the send may still
	// be in progress.
	if (!params->sdma_no_bounce_buf) {
		assert(params->origin_byte_counter);
		assert((*params->origin_byte_counter) >= params->origin_bytes_sent);
		*params->origin_byte_counter -= params->origin_bytes_sent;
		params->origin_byte_counter = NULL;
	}
	params->work_elem.work_type = OPX_WORK_TYPE_LAST;
	params->work_elem.work_fn   = fi_opx_hfi1_dput_sdma_pending_completion;

	// The SDMA request has been queued for sending, but not actually sent
	// yet, so there's no point in checking for completion right away. Wait
	// until the next poll cycle.
	return -FI_EAGAIN;
}

int fi_opx_hfi1_do_dput_sdma_tid(union fi_opx_hfi1_deferred_work *work)
{
	struct fi_opx_hfi1_dput_params	       *params			  = &work->dput;
	struct fi_opx_ep		       *opx_ep			  = params->opx_ep;
	struct fi_opx_mr		       *opx_mr			  = params->opx_mr;
	const uint8_t				u8_rx			  = params->u8_rx;
	const uint32_t				niov			  = params->niov;
	const union fi_opx_hfi1_dput_iov *const dput_iov		  = params->dput_iov;
	const uintptr_t				target_byte_counter_vaddr = params->target_byte_counter_vaddr;
	uint64_t				key			  = params->key;
	uint64_t				op64			  = params->op;
	uint64_t				dt64			  = params->dt;
	uint32_t				opcode			  = params->opcode;
	const enum ofi_reliability_kind		reliability		  = params->reliability;
	/* use the slid from the lrh header of the incoming packet
	 * as the dlid for the lrh header of the outgoing packet */
	const uint64_t lrh_dlid = params->lrh_dlid;
	const uint64_t bth_rx	= ((uint64_t) u8_rx) << OPX_BTH_RX_SHIFT;
	unsigned       i;
	const void    *sbuf_start	  = (opx_mr == NULL) ? 0 : opx_mr->iov.iov_base;
	const bool     sdma_no_bounce_buf = params->sdma_no_bounce_buf;
	assert(params->ntidpairs != 0);
	assert(niov == 1);

	/* Note that lrh_dlid is just the version of params->slid shifted so
	   that it can be OR'd into the correct position in the packet header */
	if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		assert(__cpu24_to_be16(params->slid) == (lrh_dlid >> 16));
		assert(opx_ep->rx->tx.dput_9B.hdr.lrh_9B.slid != params->slid);
	} else {
		assert(params->slid == lrh_dlid);
	}

	// We should never be in this function for intranode ops
	assert(!params->is_intranode);

	assert((opcode == FI_OPX_HFI_DPUT_OPCODE_RZV_TID) && (params->payload_bytes_for_iovec == 0));

	// With SDMA replay we can support MTU packet sizes even
	// on credit-constrained systems with smaller PIO packet
	// sizes. Ignore pio_max_eager_tx_bytes
	uint64_t       max_eager_bytes = FI_OPX_HFI1_PACKET_MTU;
	const uint64_t max_dput_bytes  = max_eager_bytes;

	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "%p:===================================== SEND DPUT SDMA TID, opcode %X -- (begin)\n", params, opcode);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-DPUT-SDMA-TID");

	for (i = params->cur_iov; i < niov; ++i) {
		uint32_t *tidpairs = (uint32_t *) params->tid_iov.iov_base;
		uint32_t  tididx   = params->tididx;
		uint32_t  tidlen_consumed;
		uint32_t  tidlen_remaining;
		uint32_t  prev_tididx		= 0;
		uint32_t  prev_tidlen_consumed	= 0;
		uint32_t  prev_tidlen_remaining = 0;
		uint32_t  tidoffset		= 0;
		uint32_t  tidOMshift		= 0;
		if (tididx == -1U) { /* first time */
			FI_OPX_DEBUG_COUNTERS_INC_COND_N(
				(opx_ep->debug_counters.expected_receive.first_tidpair_minoffset == 0),
				params->tidoffset, opx_ep->debug_counters.expected_receive.first_tidpair_minoffset);
			FI_OPX_DEBUG_COUNTERS_MIN_OF(opx_ep->debug_counters.expected_receive.first_tidpair_minoffset,
						     params->tidoffset);
			FI_OPX_DEBUG_COUNTERS_MAX_OF(opx_ep->debug_counters.expected_receive.first_tidpair_maxoffset,
						     params->tidoffset);

			tididx		 = 0;
			tidlen_remaining = FI_OPX_EXP_TID_GET(tidpairs[0], LEN);
			/* When reusing TIDs we can offset <n> pages into the TID
			   so "consume" that */
			tidlen_consumed =
				(params->tidoffset & -(int32_t) OPX_HFI1_TID_PAGESIZE) / OPX_HFI1_TID_PAGESIZE;
			tidlen_remaining -= tidlen_consumed;
			if (tidlen_consumed) {
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "params->tidoffset %u, tidlen_consumed %u, tidlen_remaining %u, length  %llu\n",
				       params->tidoffset, tidlen_consumed, tidlen_remaining,
				       FI_OPX_EXP_TID_GET(tidpairs[0], LEN));
			}
		} else { /* eagain retry, restore previous TID state */
			tidlen_consumed	 = params->tidlen_consumed;
			tidlen_remaining = params->tidlen_remaining;
		}

		uint32_t first_tidoffset;
		uint32_t first_tidoffset_page_adj;
		if (tididx == 0) {
			first_tidoffset		 = params->tidoffset;
			first_tidoffset_page_adj = first_tidoffset & (OPX_HFI1_TID_PAGESIZE - 1);
		} else {
			first_tidoffset		 = 0;
			first_tidoffset_page_adj = 0;
		}

		uint32_t starting_tid_idx = tididx;

		uint8_t *sbuf =
			(uint8_t *) ((uintptr_t) sbuf_start + (uintptr_t) dput_iov[i].sbuf + params->bytes_sent);
		uintptr_t rbuf = dput_iov[i].rbuf + params->bytes_sent;

		uint64_t bytes_to_send = dput_iov[i].bytes - params->bytes_sent;
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       " sbuf %p, sbuf_start %p, dput_iov[%u].sbuf %p, dput_iov[i].bytes %lu/%#lX, bytes sent %lu/%#lX, bytes_to_send %lu/%#lX, origin_byte_counter %ld\n",
		       sbuf, sbuf_start, i, (void *) dput_iov[i].sbuf, dput_iov[i].bytes, dput_iov[i].bytes,
		       params->bytes_sent, params->bytes_sent, bytes_to_send, bytes_to_send,
		       params->origin_byte_counter ? *(params->origin_byte_counter) : -1UL);
		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       " rbuf %p, dput_iov[%u].rbuf %p, dput_iov[i].bytes %lu/%#lX, bytes sent %lu/%#lX, bytes_to_send %lu/%#lX, first_tidoffset %u/%#X first_tidoffset_page_adj %u/%#X \n",
		       (void *) rbuf, i, (void *) dput_iov[i].rbuf, dput_iov[i].bytes, dput_iov[i].bytes,
		       params->bytes_sent, params->bytes_sent, bytes_to_send, bytes_to_send, first_tidoffset,
		       first_tidoffset, first_tidoffset_page_adj, first_tidoffset_page_adj);
		while (bytes_to_send > 0) {
			if (!fi_opx_hfi1_sdma_queue_has_room(opx_ep, OPX_SDMA_TID_IOV_COUNT)) {
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA QUEUE FULL FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN_SDMA_QUEUE_FULL, "SEND-DPUT-SDMA-TID");
				return -FI_EAGAIN;
			}
			if (!params->sdma_we) {
				/* Get an SDMA work entry since we don't already have one */
				params->sdma_we = opx_sdma_get_new_work_entry(opx_ep, &params->sdma_reqs_used,
									      &params->sdma_reqs, params->sdma_we);
				if (!params->sdma_we) {
					FI_OPX_DEBUG_COUNTERS_INC_COND(
						(params->sdma_reqs_used < FI_OPX_HFI1_SDMA_MAX_WE_PER_REQ),
						opx_ep->debug_counters.sdma.eagain_sdma_we_none_free);
					FI_OPX_DEBUG_COUNTERS_INC_COND(
						(params->sdma_reqs_used == FI_OPX_HFI1_SDMA_MAX_WE_PER_REQ),
						opx_ep->debug_counters.sdma.eagain_sdma_we_max_used);
					FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
					       "%p:===================================== SEND DPUT SDMA TID, !WE FI_EAGAIN\n",
					       params);
					OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN_SDMA_NO_WE, "SEND-DPUT-SDMA-TID");
					return -FI_EAGAIN;
				}
				assert(params->sdma_we->total_payload == 0);
				fi_opx_hfi1_sdma_init_we(params->sdma_we, params->cc, params->slid, params->origin_rs,
							 params->u8_rx, dput_iov[i].sbuf_iface,
							 (int) dput_iov[i].sbuf_device);
			}
			assert(!fi_opx_hfi1_sdma_has_unsent_packets(params->sdma_we));

			uint64_t packet_count =
				(bytes_to_send / max_dput_bytes) + ((bytes_to_send % max_dput_bytes) ? 1 : 0);

			assert(packet_count > 0);
			packet_count = MIN(packet_count, FI_OPX_HFI1_SDMA_MAX_PACKETS_TID);

			if (packet_count < FI_OPX_HFI1_SDMA_MAX_PACKETS_TID) {
				packet_count = (bytes_to_send + (OPX_HFI1_TID_PAGESIZE - 1)) / OPX_HFI1_TID_PAGESIZE;
				packet_count = MIN(packet_count, FI_OPX_HFI1_SDMA_MAX_PACKETS_TID);
			}
			int32_t psns_avail = fi_opx_reliability_tx_available_psns(
				&opx_ep->ep_fid, &opx_ep->reliability->state, params->slid, params->u8_rx,
				params->origin_rs, &params->sdma_we->psn_ptr, packet_count, max_dput_bytes);

			if (psns_avail < (int64_t) packet_count) {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.sdma.eagain_psn);
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA TID, !PSN FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN_SDMA_PSNS, "SEND-DPUT-SDMA-TID:%d:%ld",
						 psns_avail, packet_count);
				return -FI_EAGAIN;
			}
#ifndef OPX_RELIABILITY_TEST /* defining this will force reliability replay of some packets */
			{
				const int psn = params->sdma_we->psn_ptr->psn.psn;
				/* SDMA header auto-generation splits psn into
				 * generation and sequence numbers.
				 * In a writev, the generation is not incremented,
				 * instead the sequence wraps resulting in a psn
				 * that is dropped by the remote, forcing reliability
				 * replay.  We must break the writev at the wrap point
				 * and start the next writev with the next generation
				 * incremented.
				 *
				 * Since this is useful debug, it's #ifndef'd
				 * instead of just being implemented (correctly) */
				uint64_t const prev_packet_count = packet_count;
				packet_count			 = MIN(packet_count, 0x800 - (psn & 0x7FF));
				if (packet_count < prev_packet_count) {
					FI_OPX_DEBUG_COUNTERS_INC(
						opx_ep->debug_counters.expected_receive.generation_wrap);
				}
			}
#endif
			/* TID cannot add padding and has aligned buffers
			 * appropriately.  Assert that. Bounce buffers
			 * are used when not DC or fetch, not for "padding".
			 */
			assert(!(packet_count == 1 && (bytes_to_send & 0x3ul)));
			params->sdma_we->use_bounce_buf = !sdma_no_bounce_buf;

			uint8_t *sbuf_tmp;
			if (params->sdma_we->use_bounce_buf) {
				OPX_HMEM_COPY_FROM(params->sdma_we->bounce_buf.buf, sbuf,
						   MIN((packet_count * max_dput_bytes), bytes_to_send),
						   OPX_HMEM_NO_HANDLE, OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET,
						   dput_iov[i].sbuf_iface, dput_iov[i].sbuf_device);
				sbuf_tmp = params->sdma_we->bounce_buf.buf;
			} else {
				sbuf_tmp = sbuf;
			}
			// At this point, we have enough SDMA queue entries and PSNs
			// to send packet_count packets. The only limit now is how
			// many replays can we get.
			for (int p = 0; (p < packet_count) && bytes_to_send; ++p) {
#ifndef NDEBUG
				bool first_tid_last_packet = false; /* for debug assert only */
#endif
				assert(tididx < params->ntidpairs);

				uint64_t packet_bytes = MIN(bytes_to_send, max_dput_bytes);
				assert(packet_bytes <= FI_OPX_HFI1_PACKET_MTU);
				if (p == 0) { /* First packet header is user's responsibility even with SDMA header
						 auto-generation*/
					/* set fields for first header */
					unsigned offset_shift;
					starting_tid_idx = tididx; /* first tid this write() */
					if ((FI_OPX_EXP_TID_GET(tidpairs[tididx], LEN)) >=
					    (KDETH_OM_MAX_SIZE / OPX_HFI1_TID_PAGESIZE)) {
						tidOMshift   = (1 << HFI_KHDR_OM_SHIFT);
						offset_shift = KDETH_OM_LARGE_SHIFT;
					} else {
						tidOMshift   = 0;
						offset_shift = KDETH_OM_SMALL_SHIFT;
					}
					tidoffset = ((tidlen_consumed * OPX_HFI1_TID_PAGESIZE) +
						     first_tidoffset_page_adj) >>
						    offset_shift;
					FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
					       "%p:tidoffset %#X/%#X, first_tid_offset %#X, first_tidoffset_page_adj %#X\n",
					       params, tidoffset, tidoffset << offset_shift, first_tidoffset,
					       first_tidoffset_page_adj);
				}

				/* Save current values in case we can't process this packet (!REPLAY)
					   and need to restore state */
				prev_tididx	      = tididx;
				prev_tidlen_consumed  = tidlen_consumed;
				prev_tidlen_remaining = tidlen_remaining;
				/* If we offset into this TID, SDMA header auto-generation will have sent
				 * 4k/8k packets but now we have to adjust our length on the last packet
				 * to not exceed the pinned pages (subtract the offset from the last
				 * packet) like SDMA header auto-generation will do.
				 */
				if (first_tidoffset && (tidlen_remaining < 3)) {
					if (tidlen_remaining == 1) {
						packet_bytes = MIN(packet_bytes,
								   OPX_HFI1_TID_PAGESIZE - first_tidoffset_page_adj);
					} else {
						packet_bytes = MIN(packet_bytes,
								   FI_OPX_HFI1_PACKET_MTU - first_tidoffset_page_adj);
					}
					assert(tididx == 0);
					first_tidoffset		 = 0; /* offset ONLY for first tid from cts*/
					first_tidoffset_page_adj = 0;
				}
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:tid[%u], tidlen_remaining %u, packet_bytes %#lX, first_tid_offset %#X, first_tidoffset_page_adj %#X, packet_count %lu\n",
				       params, tididx, tidlen_remaining, packet_bytes, first_tidoffset,
				       first_tidoffset_page_adj, packet_count);

				/* Check tid for each packet and determine if SDMA header auto-generation will
				   use 4k or 8k packet */
				/* Assume any CTRL 3 tidpair optimizations were already done, or are not wanted,
				   so only a single tidpair per packet is possible. */
				if (packet_bytes > OPX_HFI1_TID_PAGESIZE && tidlen_remaining >= 2) {
					/* at least 2 pages, 8k mapped by this tidpair,
					   calculated packet_bytes is ok. */
					tidlen_remaining -= 2;
					tidlen_consumed += 2;
				} else if (tidlen_remaining >= 1) {
					/* only 1 page left or only 4k packet possible */
					packet_bytes = MIN(packet_bytes, OPX_HFI1_TID_PAGESIZE);
					tidlen_remaining -= 1;
					tidlen_consumed += 1;
				} else {
					assert(tidlen_remaining);
				}
				if (tidlen_remaining == 0 && tididx < (params->ntidpairs - 1)) {
#ifndef NDEBUG
					if (tididx == 0) {
						first_tid_last_packet = true; /* First tid even though tididx ++*/
					}
#endif
					tididx++;
					tidlen_remaining = FI_OPX_EXP_TID_GET(tidpairs[tididx], LEN);
					tidlen_consumed	 = 0;
				}
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:tid[%u/%u], tidlen_remaining %u, packet_bytes %#lX, first_tid_offset %#X, first_tidoffset_page_adj %#X, packet_count %lu\n",
				       params, tididx, params->ntidpairs, tidlen_remaining, packet_bytes,
				       first_tidoffset, first_tidoffset_page_adj, packet_count);

				struct fi_opx_reliability_tx_replay *replay;
				replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state, true);
				if (OFI_UNLIKELY(replay == NULL)) {
					/* Restore previous values in case since we can't process this
					 * packet. We may or may not -FI_EAGAIN later (!REPLAY).*/
					tididx		 = prev_tididx;
					tidlen_consumed	 = prev_tidlen_consumed;
					tidlen_remaining = prev_tidlen_remaining;
					FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
					       "%p:!REPLAY on packet %u out of %lu, params->sdma_we->num_packets %u\n",
					       params, p, packet_count, params->sdma_we->num_packets);
					break;
				}
				replay->use_sdma = true; /* Always replay TID packets with SDMA */

				// Round packet_bytes up to the next multiple of 4,
				// then divide by 4 to get the correct number of dws.
				uint64_t pbc_dws;
				uint16_t lrh_dws;
				if (OPX_HFI1_TYPE & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
					uint64_t payload_dws = (packet_bytes + 3) >> 2;
					pbc_dws		     = 2 + /* pbc */
						  2 +		   /* lrh */
						  3 +		   /* bth */
						  9 +		   /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
						  payload_dws;
					lrh_dws = htons(pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8
									     bytes), but does include icrc (4 bytes) */
				} else {
					uint64_t payload_dws =
						((packet_bytes + 7) & -8) >> 2; /* 16B is QW length/padded */
					pbc_dws = 2 +				/* pbc */
						  4 +				/* lrh uncompressed */
						  3 +				/* bth */
						  9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
						  2 + /* ICRC/tail */
						  payload_dws;
					lrh_dws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */
				}

				assert(replay != NULL);

				if (OPX_HFI1_TYPE & OPX_HFI1_JKR) {
					replay->scb.scb_16B.qw0 = opx_ep->rx->tx.dput_16B.qw0 |
								  OPX_PBC_LEN(pbc_dws, OPX_HFI1_TYPE) |
								  params->pbc_dlid;
				} else {
					replay->scb.scb_9B.qw0 = opx_ep->rx->tx.dput_9B.qw0 |
								 OPX_PBC_LEN(pbc_dws, OPX_HFI1_TYPE) | params->pbc_dlid;
				}

				/* The fetch_vaddr and cbuf arguments are only used
				   for atomic fetch operations, which by their one-
				   sided nature will never use TID, so they are
				   hard-coded to 0/NULL respectively */
				uint64_t bytes_sent = opx_hfi1_dput_write_header_and_iov(
					opx_ep, OPX_REPLAY_HDR(replay), replay->iov, opcode, lrh_dws, op64, dt64,
					lrh_dlid, bth_rx, packet_bytes, key, 0ul, target_byte_counter_vaddr,
					params->rma_request_vaddr, params->bytes_sent, &sbuf_tmp, NULL, &rbuf,
					OPX_HFI1_TYPE);
				/* tid packets are page aligned and 4k/8k length except
				   first TID and last (remnant) packet */
				assert((tididx == 0) || (first_tid_last_packet) ||
				       (bytes_to_send < FI_OPX_HFI1_PACKET_MTU) || ((rbuf & 0xFFF) == 0) ||
				       ((bytes_sent & 0xFFF) == 0));
				fi_opx_hfi1_sdma_add_packet(params->sdma_we, replay, packet_bytes);

				bytes_to_send -= bytes_sent;
				params->bytes_sent += bytes_sent;
				params->origin_bytes_sent += bytes_sent;
				sbuf += bytes_sent;
			}

			// Must be we had trouble getting a replay buffer
			if (OFI_UNLIKELY(params->sdma_we->num_packets == 0)) {
				FI_OPX_DEBUG_COUNTERS_INC(opx_ep->debug_counters.sdma.eagain_replay);
				FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
				       "%p:===================================== SEND DPUT SDMA TID, !REPLAY FI_EAGAIN\n",
				       params);
				OPX_TRACER_TRACE(OPX_TRACER_END_EAGAIN_SDMA_REPLAY_BUFFER, "SEND-DPUT-SDMA-TID");
				return -FI_EAGAIN;
			}

			/* after first tid, should have made necessary adjustments and zeroed it */
			assert(((first_tidoffset == 0) && (first_tidoffset_page_adj == 0)) || (tididx == 0));

			opx_hfi1_sdma_flush(opx_ep, params->sdma_we, &params->sdma_reqs, 1, /* use tid */
					    &params->tid_iov, starting_tid_idx, tididx, tidOMshift, tidoffset,
					    reliability);
			params->sdma_we = NULL;
			/* save our 'done' tid state incase we return EAGAIN next loop */
			params->tididx		 = tididx;
			params->tidlen_consumed	 = tidlen_consumed;
			params->tidlen_remaining = tidlen_remaining;

		} /* while bytes_to_send */

		FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
		       "%p:===================================== SEND DPUT SDMA TID, finished IOV=%d(%d) bytes_sent=%ld\n",
		       params, params->cur_iov, niov, params->bytes_sent);

		params->bytes_sent = 0;
		params->cur_iov++;
	} /* for niov */
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-DPUT-SDMA-TID");
	FI_DBG(fi_opx_global.prov, FI_LOG_EP_DATA,
	       "%p:===================================== SEND DPUT SDMA TID, exit (end)\n", params);

	// At this point, all SDMA WE should have succeeded sending, and only reside on the reqs list
	assert(params->sdma_we == NULL);
	assert(!slist_empty(&params->sdma_reqs));

	// If we're not doing delivery completion, the user's payload would have
	// been copied to bounce buffer(s), so at this point, it should be safe
	// for the user to alter the send buffer even though the send may still
	// be in progress.
	if (!params->sdma_no_bounce_buf) {
		assert(params->origin_byte_counter);
		assert((*params->origin_byte_counter) >= params->origin_bytes_sent);
		*params->origin_byte_counter -= params->origin_bytes_sent;
		params->origin_byte_counter = NULL;
	}
	params->work_elem.work_type = OPX_WORK_TYPE_LAST;
	params->work_elem.work_fn   = fi_opx_hfi1_dput_sdma_pending_completion;

	// The SDMA request has been queued for sending, but not actually sent
	// yet, so there's no point in checking for completion right away. Wait
	// until the next poll cycle.
	return -FI_EAGAIN;
}

union fi_opx_hfi1_deferred_work *
fi_opx_hfi1_rx_rzv_cts(struct fi_opx_ep *opx_ep, struct fi_opx_mr *opx_mr, const union opx_hfi1_packet_hdr *const hdr,
		       const void *const payload, size_t payload_bytes_to_copy, const uint8_t u8_rx,
		       const uint8_t origin_rs, const uint32_t niov, const union fi_opx_hfi1_dput_iov *const dput_iov,
		       const uint8_t op, const uint8_t dt, const uintptr_t rma_request_vaddr,
		       const uintptr_t target_byte_counter_vaddr, uint64_t *origin_byte_counter, uint32_t opcode,
		       void (*completion_action)(union fi_opx_hfi1_deferred_work *work_state),
		       const unsigned is_intranode, const enum ofi_reliability_kind reliability,
		       const uint32_t u32_extended_rx, const enum opx_hfi1_type hfi1_type)
{
	union fi_opx_hfi1_deferred_work *work	= ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	struct fi_opx_hfi1_dput_params	*params = &work->dput;

	params->work_elem.slist_entry.next  = NULL;
	params->work_elem.completion_action = completion_action;
	params->work_elem.payload_copy	    = NULL;
	params->work_elem.complete	    = false;
	params->opx_ep			    = opx_ep;
	params->opx_mr			    = opx_mr;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		params->slid	 = (opx_lid_t) __be16_to_cpu24((__be16) hdr->lrh_9B.slid);
		params->lrh_dlid = (hdr->lrh_9B.qw[0] & 0xFFFF000000000000ul) >> 32;
	} else {
		params->slid = (opx_lid_t) __le24_to_cpu(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid);
		params->lrh_dlid =
			(hdr->lrh_16B.slid20 << 20 | hdr->lrh_16B.slid); // Send dput to the SLID that sent CTS
	}
	params->pbc_dlid	  = OPX_PBC_DLID_TO_PBC_DLID(params->slid, hfi1_type);
	params->origin_rs	  = origin_rs;
	params->u8_rx		  = u8_rx;
	params->u32_extended_rx	  = u32_extended_rx;
	params->niov		  = niov;
	params->dput_iov	  = &params->iov[0];
	params->cur_iov		  = 0;
	params->bytes_sent	  = 0;
	params->origin_bytes_sent = 0;

	if (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT_CQ) {
		params->cc = ((struct fi_opx_rma_request *) rma_request_vaddr)->cc;
		OPX_BUF_FREE((struct fi_opx_rma_request *) rma_request_vaddr);
	} else {
		params->cc = NULL;
	}
	params->user_cc			= NULL;
	params->payload_bytes_for_iovec = 0;
	params->sdma_no_bounce_buf	= false;

	params->target_byte_counter_vaddr = target_byte_counter_vaddr;
	params->rma_request_vaddr	  = rma_request_vaddr;
	params->origin_byte_counter	  = origin_byte_counter;
	params->opcode			  = opcode;
	params->op			  = op;
	params->dt			  = dt;
	params->is_intranode		  = is_intranode;
	params->reliability		  = reliability;
	if (is_intranode) {
		if (params->slid == opx_ep->rx->self.lid) {
			params->target_hfi_unit = opx_ep->rx->self.hfi1_unit;
		} else {
			struct fi_opx_hfi_local_lookup *hfi_lookup = fi_opx_hfi1_get_lid_local(params->slid);
			assert(hfi_lookup);
			params->target_hfi_unit = hfi_lookup->hfi_unit;
		}
	} else {
		params->target_hfi_unit = 0xFF;
	}

	uint64_t is_hmem	 = 0;
	uint64_t iov_total_bytes = 0;
	for (int idx = 0; idx < niov; idx++) {
#ifdef OPX_HMEM
		/* If either the send or receive buffer's iface is non-zero, i.e. not system memory, set hmem on */
		is_hmem |= (dput_iov[idx].rbuf_iface | dput_iov[idx].sbuf_iface);
#endif
		params->iov[idx] = dput_iov[idx];
		iov_total_bytes += dput_iov[idx].bytes;
	}
	/* Only RZV TID sets ntidpairs */
	uint32_t  ntidpairs = 0;
	uint32_t  tidoffset = 0;
	uint32_t *tidpairs  = NULL;

	if (opcode == FI_OPX_HFI_DPUT_OPCODE_RZV_TID) {
		ntidpairs = hdr->cts.target.vaddr.ntidpairs;
		if (ntidpairs) {
			union fi_opx_hfi1_packet_payload *tid_payload = (union fi_opx_hfi1_packet_payload *) payload;
			tidpairs				      = tid_payload->tid_cts.tidpairs;
			tidoffset				      = tid_payload->tid_cts.tid_offset;
			/* Receiver may have adjusted the length for expected TID alignment.*/
			if (origin_byte_counter) {
				(*origin_byte_counter) += tid_payload->tid_cts.origin_byte_counter_adjust;
			}
		}
	}
	assert((ntidpairs == 0) || (niov == 1));
	assert(origin_byte_counter == NULL || iov_total_bytes <= *origin_byte_counter);
	fi_opx_hfi1_dput_sdma_init(opx_ep, params, iov_total_bytes, tidoffset, ntidpairs, tidpairs, is_hmem);

	FI_OPX_DEBUG_COUNTERS_INC_COND(is_hmem && is_intranode, opx_ep->debug_counters.hmem.dput_rzv_intranode);
	FI_OPX_DEBUG_COUNTERS_INC_COND(is_hmem && !is_intranode && params->work_elem.work_fn == fi_opx_hfi1_do_dput,
				       opx_ep->debug_counters.hmem.dput_rzv_pio);
	FI_OPX_DEBUG_COUNTERS_INC_COND(is_hmem && params->work_elem.work_fn == fi_opx_hfi1_do_dput_sdma,
				       opx_ep->debug_counters.hmem.dput_rzv_sdma);
	FI_OPX_DEBUG_COUNTERS_INC_COND(is_hmem && params->work_elem.work_fn == fi_opx_hfi1_do_dput_sdma_tid,
				       opx_ep->debug_counters.hmem.dput_rzv_tid);

	// We can't/shouldn't start this work until any pending work is finished.
	if (params->work_elem.work_type != OPX_WORK_TYPE_SDMA &&
	    slist_empty(&opx_ep->tx->work_pending[params->work_elem.work_type])) {
		int rc = params->work_elem.work_fn(work);
		if (rc == FI_SUCCESS) {
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				     "===================================== CTS done %u\n", params->work_elem.complete);
			assert(params->work_elem.complete);
			OPX_BUF_FREE(work);
			return NULL;
		}
		assert(rc == -FI_EAGAIN);
		if (params->work_elem.work_type == OPX_WORK_TYPE_LAST) {
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
				     "===================================== CTS FI_EAGAIN queued low priority %u\n",
				     params->work_elem.complete);
			slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending_completion);
			return NULL;
		}
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== CTS FI_EAGAIN queued %u, payload_bytes_to_copy %zu\n",
			params->work_elem.complete, payload_bytes_to_copy);
	} else {
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== CTS queued with work pending %u, payload_bytes_to_copy %zu\n",
			params->work_elem.complete, payload_bytes_to_copy);
	}

	/* Try again later*/
	if (payload_bytes_to_copy) {
		params->work_elem.payload_copy = ofi_buf_alloc(opx_ep->tx->rma_payload_pool);
		memcpy(params->work_elem.payload_copy, payload, payload_bytes_to_copy);
	}
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending[params->work_elem.work_type]);
	return work;
}

uint64_t num_sends;
uint64_t total_sendv_bytes;
ssize_t	 fi_opx_hfi1_tx_sendv_rzv(struct fid_ep *ep, const struct iovec *iov, size_t niov, size_t total_len, void *desc,
				  fi_addr_t dest_addr, uint64_t tag, void *user_context, const uint32_t data,
				  int lock_required, const unsigned override_flags, const uint64_t tx_op_flags,
				  const uint64_t dest_rx, const uint64_t caps,
				  const enum ofi_reliability_kind reliability, const uint64_t do_cq_completion,
				  const enum fi_hmem_iface hmem_iface, const uint64_t hmem_device,
				  const enum opx_hfi1_type hfi1_type)
{
	// We should already have grabbed the lock prior to calling this function
	assert(!lock_required);

	struct fi_opx_ep       *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr   = {.fi = dest_addr};
	const uint64_t		bth_rx = ((uint64_t) dest_rx) << OPX_BTH_RX_SHIFT;
	assert(niov <= MIN(FI_OPX_MAX_DPUT_IOV, FI_OPX_MAX_HMEM_IOV));

	FI_OPX_DEBUG_COUNTERS_DECLARE_TMP(hmem_non_system);

	/* This is a hack to trick an MPICH test to make some progress    */
	/* As it erroneously overflows the send buffers by never checking */
	/* for multi-receive overflows properly in some onesided tests    */
	/* There are almost certainly better ways to do this */
	if ((tx_op_flags & FI_MSG) && (total_sendv_bytes += total_len > opx_ep->rx->min_multi_recv)) {
		total_sendv_bytes = 0;
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
		return -FI_EAGAIN;
	}

	// Calculate space for each IOV, then add in the origin_byte_counter_vaddr,
	// and round to the next 64-byte block.
	const uint64_t icrc_and_tail_block = ((hfi1_type == OPX_HFI1_JKR) ? 1 : 0);
	const uint64_t payload_blocks_total =
		((niov * sizeof(struct fi_opx_hmem_iov)) + sizeof(uintptr_t) + icrc_and_tail_block + 63) >> 6;
	assert(payload_blocks_total > 0 && payload_blocks_total < (FI_OPX_HFI1_PACKET_MTU >> 6));

	uint64_t pbc_dws;
	uint16_t lrh_dws;

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		pbc_dws = 2 + /* pbc */
			  2 + /* lrh */
			  3 + /* bth */
			  9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
			  (payload_blocks_total << 4);

		lrh_dws = htons(pbc_dws - 2 +
				1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */
	} else {
		pbc_dws = 2 +			       /* pbc */
			  4 +			       /* lrh uncompressed */
			  3 +			       /* bth */
			  9 +			       /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
			  (payload_blocks_total << 4); /* ICRC/tail is accounted for here */
		lrh_dws = (pbc_dws - 2) >> 1;	       /* (LRH QW) does not include pbc (8 bytes) */
	}

	if (fi_opx_hfi1_tx_is_intranode(opx_ep, addr, caps)) {
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SENDV, SHM -- RENDEZVOUS RTS Noncontig (begin) context %p\n",
			user_context);

		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SENDV-RZV-RTS-NONCONTIG-SHM");
		uint64_t			 pos;
		ssize_t				 rc;
		union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
			&opx_ep->tx->shm, addr.hfi1_unit, dest_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
			opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst, &rc);

		if (!hdr) {
			return rc;
		}

		struct opx_context *context;
		uintptr_t	    origin_byte_counter_vaddr;
		if (OFI_LIKELY(do_cq_completion)) {
			context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
			if (OFI_UNLIKELY(context == NULL)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
				return -FI_ENOMEM;
			}
			context->err_entry.err	      = 0;
			context->err_entry.op_context = user_context;
			context->next		      = NULL;
			context->byte_counter	      = total_len;
			origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
		} else {
			context			  = NULL;
			origin_byte_counter_vaddr = (uintptr_t) NULL;
		}

		if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
			const uint64_t lrh_dlid_9B = FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(addr.lid);
			hdr->qw_9B[0] = opx_ep->tx->rzv_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32);
			hdr->qw_9B[1] = opx_ep->tx->rzv_9B.hdr.qw_9B[1] | bth_rx |
					((caps & FI_MSG) ? ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
								    (uint64_t) FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
								    FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
							   ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
								    (uint64_t) FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
								    FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));
			hdr->qw_9B[2] = opx_ep->tx->rzv_9B.hdr.qw_9B[2];
			hdr->qw_9B[3] = opx_ep->tx->rzv_9B.hdr.qw_9B[3] | (((uint64_t) data) << 32);
			hdr->qw_9B[4] =
				opx_ep->tx->rzv_9B.hdr.qw_9B[4] | (niov << 48) | FI_OPX_PKT_RZV_FLAGS_NONCONTIG_MASK;
			hdr->qw_9B[5] = total_len;
			hdr->qw_9B[6] = tag;
		} else {
			const uint64_t lrh_dlid_16B = addr.lid;
			hdr->qw_16B[0]		    = opx_ep->tx->rzv_16B.hdr.qw_16B[0] |
					 ((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B)
					  << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
					 ((uint64_t) lrh_dws << 20);
			hdr->qw_16B[1] = opx_ep->tx->rzv_16B.hdr.qw_16B[1] |
					 ((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
						      OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
					 (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);
			hdr->qw_16B[2] = opx_ep->tx->rzv_16B.hdr.qw_16B[2] | bth_rx |
					 ((caps & FI_MSG) ? ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
								     (uint64_t) FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
								     FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
							    ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
								     (uint64_t) FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
								     FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));
			hdr->qw_16B[3] = opx_ep->tx->rzv_16B.hdr.qw_16B[3];
			hdr->qw_16B[4] = opx_ep->tx->rzv_16B.hdr.qw_16B[4] | (((uint64_t) data) << 32);
			hdr->qw_16B[5] =
				opx_ep->tx->rzv_16B.hdr.qw_16B[5] | (niov << 48) | FI_OPX_PKT_RZV_FLAGS_NONCONTIG_MASK;
			hdr->qw_16B[6] = total_len;
			hdr->qw_16B[7] = tag;
		}

		union fi_opx_hfi1_packet_payload *const payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

		payload->rendezvous.noncontiguous.origin_byte_counter_vaddr = origin_byte_counter_vaddr;
		struct fi_opx_hmem_iov *payload_iov			    = &payload->rendezvous.noncontiguous.iov[0];
		struct iovec	       *input_iov			    = (struct iovec *) iov;

		for (int i = 0; i < niov; i++) {
#ifdef OPX_HMEM
			// TODO: desc is plumbed into this function as a single pointer
			//       only representing the first IOV. It should be changed
			//       to void ** to get an array of desc, one for each IOV.
			//       For now, just use the first iov's desc, assuming all
			//       the IOVs will reside in the same HMEM space.
			FI_OPX_DEBUG_COUNTERS_INC_COND(hmem_iface != FI_HMEM_SYSTEM, hmem_non_system);
#endif
			payload_iov->buf    = (uintptr_t) input_iov->iov_base;
			payload_iov->len    = input_iov->iov_len;
			payload_iov->device = hmem_device;
			payload_iov->iface  = hmem_iface;
			payload_iov++;
			input_iov++;
		}

		FI_OPX_DEBUG_COUNTERS_INC_COND(
			hmem_non_system,
			opx_ep->debug_counters.hmem.intranode.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG]
				.send.rzv_noncontig);
		opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

		if (OFI_LIKELY(do_cq_completion)) {
			fi_opx_ep_tx_cq_completion_rzv(ep, context, total_len, lock_required, tag, caps);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SENDV-RZV-RTS-NONCONTIG-SHM");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== SENDV, SHM -- RENDEZVOUS RTS (end) context %p\n",
			     user_context);
		fi_opx_shm_poll_many(&opx_ep->ep_fid, 0, hfi1_type);
		return FI_SUCCESS;
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SENDV, HFI -- RENDEZVOUS RTS (begin) context %p\n",
		     user_context);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SENDV-RZV-RTS-HFI");

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	const uint16_t total_credits_needed = 1 +		    /* packet header */
					      payload_blocks_total; /* packet payload */

	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									total_credits_needed);
		if (total_credits_available < total_credits_needed) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			return -FI_EAGAIN;
		}
	}

	struct opx_context *context;
	uintptr_t	    origin_byte_counter_vaddr;
	if (OFI_LIKELY(do_cq_completion)) {
		context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
		if (OFI_UNLIKELY(context == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
			return -FI_ENOMEM;
		}
		context->err_entry.err	      = 0;
		context->err_entry.op_context = user_context;
		context->next		      = NULL;
		context->byte_counter	      = total_len;
		origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
	} else {
		context			  = NULL;
		origin_byte_counter_vaddr = (uintptr_t) NULL;
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, addr.lid, dest_rx,
					    addr.reliability_rx, &psn_ptr, &replay, reliability, hfi1_type);
	if (OFI_UNLIKELY(psn == -1)) {
		if (OFI_LIKELY(do_cq_completion)) {
			OPX_BUF_FREE(context);
		}
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
		return -FI_EAGAIN;
	}

	struct fi_opx_hmem_iov hmem_iov[FI_OPX_MAX_HMEM_IOV];
	unsigned	       hmem_niov = MIN(niov, FI_OPX_MAX_HMEM_IOV);
	for (int i = 0; i < hmem_niov; ++i) {
		hmem_iov[i].buf = (uintptr_t) iov[i].iov_base;
		hmem_iov[i].len = iov[i].iov_len;
#ifdef OPX_HMEM
		hmem_iov[i].iface  = hmem_iface;
		hmem_iov[i].device = hmem_device;
		FI_OPX_DEBUG_COUNTERS_INC_COND(hmem_iov[i].iface != FI_HMEM_SYSTEM, hmem_non_system);
#else
		hmem_iov[i].iface  = FI_HMEM_SYSTEM;
		hmem_iov[i].device = 0;
#endif
	}
	FI_OPX_DEBUG_COUNTERS_INC_COND(
		hmem_non_system,
		opx_ep->debug_counters.hmem.hfi.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG]
			.send.rzv_noncontig);

	assert(opx_ep->tx->rzv_9B.qw0 == 0);
	const uint64_t force_credit_return = OPX_PBC_CR(opx_ep->tx->force_credit_return, hfi1_type);

	volatile uint64_t *const scb		= FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);
	uint64_t		 local_temp[16] = {0};

	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		const uint64_t lrh_dlid_9B = FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(addr.lid);
		fi_opx_store_and_copy_qw(
			scb, local_temp,
			opx_ep->tx->rzv_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | force_credit_return |
				OPX_PBC_DLID_TO_PBC_DLID(addr.lid, hfi1_type),
			opx_ep->tx->rzv_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32),
			opx_ep->tx->rzv_9B.hdr.qw_9B[1] | bth_rx |
				((caps & FI_MSG) ? ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
							    (uint64_t) FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
							    FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
						   ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
							    (uint64_t) FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
							    FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS)),
			opx_ep->tx->rzv_9B.hdr.qw_9B[2] | psn,
			opx_ep->tx->rzv_9B.hdr.qw_9B[3] | (((uint64_t) data) << 32),
			opx_ep->tx->rzv_9B.hdr.qw_9B[4] | (niov << 48) | FI_OPX_PKT_RZV_FLAGS_NONCONTIG_MASK, total_len,
			tag);
		fi_opx_copy_hdr9B_cacheline(&replay->scb.scb_9B, local_temp);
	} else {
		const uint64_t lrh_dlid_16B = addr.lid;
		fi_opx_store_and_copy_qw(
			scb, local_temp,
			opx_ep->tx->rzv_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | force_credit_return |
				OPX_PBC_DLID_TO_PBC_DLID(addr.lid, hfi1_type),
			opx_ep->tx->rzv_16B.hdr.qw_16B[0] |
				((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B)
				 << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
				((uint64_t) lrh_dws << 20),
			opx_ep->tx->rzv_16B.hdr.qw_16B[1] |
				((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					     OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B),
			opx_ep->tx->rzv_16B.hdr.qw_16B[2] | bth_rx |
				((caps & FI_MSG) ? ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
							    (uint64_t) FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
							    FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
						   ((tx_op_flags & FI_REMOTE_CQ_DATA) ?
							    (uint64_t) FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
							    FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS)),
			opx_ep->tx->rzv_16B.hdr.qw_16B[3] | psn,
			opx_ep->tx->rzv_16B.hdr.qw_16B[4] | (((uint64_t) data) << 32),
			opx_ep->tx->rzv_16B.hdr.qw_16B[5] | (niov << 48) | FI_OPX_PKT_RZV_FLAGS_NONCONTIG_MASK,
			total_len);
	}

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	/* consume one credit for the packet header */
	--total_credits_available;
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	unsigned credits_consumed = 1;
#endif

	/* write the payload */
	uint64_t	  *iov_qws     = (uint64_t *) &hmem_iov[0];
	volatile uint64_t *scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

	uint64_t local_temp_payload[16] = {0};
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		fi_opx_store_and_copy_qw(scb_payload, local_temp_payload, origin_byte_counter_vaddr, iov_qws[0],
					 iov_qws[1], iov_qws[2], iov_qws[3], iov_qws[4], iov_qws[5], iov_qws[6]);
		iov_qws += 7;
	} else {
		fi_opx_store_and_copy_qw(scb_payload, local_temp_payload, tag, origin_byte_counter_vaddr, iov_qws[0],
					 iov_qws[1], iov_qws[2], iov_qws[3], iov_qws[4], iov_qws[5]);
		iov_qws += 6;
	}

	/* consume one credit for the rendezvous payload metadata */
	--total_credits_available;
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

	uint64_t *replay_payload = replay->payload;
	assert(!replay->use_iov);
	assert(((uint8_t *) replay_payload) == ((uint8_t *) &replay->data));
	uint64_t rem_payload_size;
	if (hfi1_type & (OPX_HFI1_WFR | OPX_HFI1_JKR_9B)) {
		fi_opx_copy_cacheline(replay_payload, local_temp_payload);
		replay_payload += FI_OPX_CACHE_LINE_QWS;
		rem_payload_size = sizeof(struct fi_opx_hmem_iov) * (niov - 2);
	} else {
		local_temp[7] = local_temp_payload[0];
		fi_opx_copy_hdr16B_cacheline(&replay->scb.scb_16B, local_temp);
		fi_opx_copy_cacheline(replay_payload, &local_temp_payload[1]);
		replay_payload += 7;
		rem_payload_size =
			(sizeof(struct fi_opx_hmem_iov) * (niov - 2) + 8); // overflow 8 bytes from 2nd cacheline
	}

	if (payload_blocks_total > 1) {
		assert(niov > 2);

#ifndef NDEBUG
		credits_consumed +=
#endif
			fi_opx_hfi1_tx_egr_store_full_payload_blocks(opx_ep, &pio_state, iov_qws,
								     payload_blocks_total - 1, total_credits_available);

		memcpy(replay_payload, iov_qws, rem_payload_size);
	}

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
#ifndef NDEBUG
	assert(credits_consumed == total_credits_needed);
#endif

	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, addr.reliability_rx, dest_rx,
							    psn_ptr, replay, reliability, hfi1_type);

	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	if (OFI_LIKELY(do_cq_completion)) {
		fi_opx_ep_tx_cq_completion_rzv(ep, context, total_len, lock_required, tag, caps);
	}
	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SENDV-RZV-RTS-HFI");
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SENDV, HFI -- RENDEZVOUS RTS (end) context %p\n", context);

	return FI_SUCCESS;
}

ssize_t fi_opx_hfi1_tx_send_rzv(struct fid_ep *ep, const void *buf, size_t len, void *desc, fi_addr_t dest_addr,
				uint64_t tag, void *user_context, const uint32_t data, int lock_required,
				const unsigned override_flags, const uint64_t tx_op_flags, const uint64_t dest_rx,
				const uint64_t caps, const enum ofi_reliability_kind reliability,
				const uint64_t do_cq_completion, const enum fi_hmem_iface src_iface,
				const uint64_t src_device_id, const enum opx_hfi1_type hfi1_type)
{
	// We should already have grabbed the lock prior to calling this function
	assert(!lock_required);

	// Need at least one full block of payload
	assert(len >= FI_OPX_HFI1_TX_MIN_RZV_PAYLOAD_BYTES);

	struct fi_opx_ep       *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr   = {.fi = dest_addr};

	const uint64_t is_intranode = fi_opx_hfi1_tx_is_intranode(opx_ep, addr, caps);

	const uint64_t bth_rx	   = ((uint64_t) dest_rx) << OPX_BTH_RX_SHIFT;
	const uint64_t lrh_dlid_9B = FI_OPX_ADDR_TO_HFI1_LRH_DLID_9B(addr.lid);

	const bool send_immed_data = (src_iface == FI_HMEM_SYSTEM);

#ifndef NDEBUG
	const uint64_t max_immediate_block_count = (FI_OPX_HFI1_PACKET_MTU >> 6) - 2;
#endif
	/* Expected tid needs to send a leading data block and trailing data
	 * for alignment. TID writes must start on a 64-byte boundary, so we
	 * need to send 64 bytes of leading immediate data that allow us
	 * to shift the receive buffer starting offset to a TID-friendly value.
	 * TID writes must also be a length that is a multiple of a DW (WFR & JKR 9B)
	 * or a QW (JKR), so send the last 7 bytes of the source data immediately
	 * so we can adjust the length after proper alignment has been achieved. */
	const uint8_t immediate_block =
		(send_immed_data && !is_intranode && opx_ep->use_expected_tid_rzv &&
		 len >= opx_ep->tx->sdma_min_payload_bytes && len >= opx_ep->tx->tid_min_payload_bytes) ?
			1 :
			0;
	const uint8_t immediate_tail = immediate_block;

	assert(immediate_block <= 1);
	assert(immediate_tail <= 1);
	assert(immediate_block <= max_immediate_block_count);

	const uint8_t immediate_byte_count = send_immed_data ? (uint8_t) (len & 0x0007ul) : 0;
	const uint8_t immediate_qw_count   = send_immed_data ? (uint8_t) ((len >> 3) & 0x0007ul) : 0;
	const uint8_t immediate_fragment   = send_immed_data ? (uint8_t) (((len & 0x003Ful) + 63) >> 6) : 0;
	assert(immediate_fragment == 1 || immediate_fragment == 0);

	/* Immediate total does not include trailing block */
	const uint64_t immediate_total = immediate_byte_count + immediate_qw_count * sizeof(uint64_t) +
					 immediate_block * sizeof(union cacheline);

	union fi_opx_hfi1_rzv_rts_immediate_info immediate_info = {
		.count = (immediate_byte_count << OPX_IMMEDIATE_BYTE_COUNT_SHIFT) |
			 (immediate_qw_count << OPX_IMMEDIATE_QW_COUNT_SHIFT) |
			 (immediate_block << OPX_IMMEDIATE_BLOCK_SHIFT) | (immediate_tail << OPX_IMMEDIATE_TAIL_SHIFT),
		.tail_bytes = {}};

	assert(((len - immediate_total) & 0x003Fu) == 0);

	const uint64_t payload_blocks_total = 1 + /* rzv metadata */
					      immediate_fragment + immediate_block;

	const uint64_t pbc_dws = 2 + /* pbc */
				 2 + /* lhr */
				 3 + /* bth */
				 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 (payload_blocks_total << 4);

	const uint16_t lrh_dws = htons(
		pbc_dws - 2 + 1); /* (BE: LRH DW) does not include pbc (8 bytes), but does include icrc (4 bytes) */

	if (is_intranode) {
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== SEND, SHM -- RENDEZVOUS RTS (begin) context %p\n",
			     user_context);
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-RTS-SHM");
		uint64_t			 pos;
		ssize_t				 rc;
		union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
			&opx_ep->tx->shm, addr.hfi1_unit, dest_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
			opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst, &rc);

		if (!hdr) {
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "return %zd\n", rc);
			return rc;
		}

		struct opx_context *context;
		uintptr_t	    origin_byte_counter_vaddr;
		if (OFI_LIKELY(do_cq_completion)) {
			context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
			if (OFI_UNLIKELY(context == NULL)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
				return -FI_ENOMEM;
			}
			context->err_entry.err	      = 0;
			context->err_entry.op_context = user_context;
			context->next		      = NULL;
			context->byte_counter	      = len - immediate_total;
			origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
		} else {
			context			  = NULL;
			origin_byte_counter_vaddr = (uintptr_t) NULL;
		}

		FI_OPX_DEBUG_COUNTERS_INC_COND(
			src_iface != FI_HMEM_SYSTEM,
			opx_ep->debug_counters.hmem.intranode.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG]
				.send.rzv);

		const uint64_t opcode =
			(uint64_t) ((caps & FI_MSG) ?
					    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
										 FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
					    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
										 FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));

		hdr->qw_9B[0] = opx_ep->tx->rzv_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32);
		hdr->qw_9B[1] = opx_ep->tx->rzv_9B.hdr.qw_9B[1] | bth_rx | opcode;
		hdr->qw_9B[2] = opx_ep->tx->rzv_9B.hdr.qw_9B[2];
		hdr->qw_9B[3] = opx_ep->tx->rzv_9B.hdr.qw_9B[3] | (((uint64_t) data) << 32);
		hdr->qw_9B[4] = opx_ep->tx->rzv_9B.hdr.qw_9B[4] | (1ull << 48); /* effectively 1 iov */
		hdr->qw_9B[5] = len;
		hdr->qw_9B[6] = tag;

		union fi_opx_hfi1_packet_payload *const payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"hdr %p, payload %p, sbuf %p, sbuf+immediate_total %p, immediate_total %#lX, adj len %#lX\n",
			hdr, payload, buf, ((char *) buf + immediate_total), immediate_total, (len - immediate_total));

		struct opx_payload_rzv_contig *contiguous = &payload->rendezvous.contiguous;
		payload->rendezvous.contig_9B_padding	  = 0;
		contiguous->src_vaddr			  = (uintptr_t) buf + immediate_total;
		contiguous->src_len			  = len - immediate_total;
		contiguous->src_device_id		  = src_device_id;
		contiguous->src_iface			  = (uint64_t) src_iface;
		contiguous->immediate_info		  = immediate_info.qw0;
		contiguous->origin_byte_counter_vaddr	  = origin_byte_counter_vaddr;
		contiguous->unused			  = 0;

		if (immediate_total) {
			uint8_t *sbuf;
			if (src_iface != FI_HMEM_SYSTEM) {
				struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
				opx_copy_from_hmem(src_iface, src_device_id,
						   desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE,
						   opx_ep->hmem_copy_buf, buf, immediate_total,
						   desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD :
							     OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
				sbuf = opx_ep->hmem_copy_buf;
			} else {
				sbuf = (uint8_t *) buf;
			}

			for (int i = 0; i < immediate_byte_count; ++i) {
				contiguous->immediate_byte[i] = sbuf[i];
			}
			sbuf += immediate_byte_count;

			uint64_t *sbuf_qw = (uint64_t *) sbuf;
			for (int i = 0; i < immediate_qw_count; ++i) {
				contiguous->immediate_qw[i] = sbuf_qw[i];
			}

			if (immediate_block) {
				sbuf_qw += immediate_qw_count;
				uint64_t *payload_cacheline =
					(uint64_t *) (&contiguous->cache_line_1 + immediate_fragment);
				fi_opx_copy_cacheline(payload_cacheline, sbuf_qw);
			}
		}

		opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

		if (OFI_LIKELY(do_cq_completion)) {
			fi_opx_ep_tx_cq_completion_rzv(ep, context, len, lock_required, tag, caps);
		}

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-RTS-SHM");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== SEND, SHM -- RENDEZVOUS RTS (end) context %p\n",
			     user_context);

		return FI_SUCCESS;
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, HFI -- RENDEZVOUS RTS (begin) context %p\n",
		     user_context);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-RTS-HFI:%ld", tag);

	/*
	 * While the bulk of the payload data will be sent via SDMA once we
	 * get the CTS from the receiver, the initial RTS packet is sent via PIO.
	 */

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	const uint16_t total_credits_needed = 1 +		    /* packet header */
					      payload_blocks_total; /* packet payload */

	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									total_credits_needed);
		if (total_credits_available < total_credits_needed) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			return -FI_EAGAIN;
		}
	}

	struct opx_context *context;
	uintptr_t	    origin_byte_counter_vaddr;
	if (OFI_LIKELY(do_cq_completion)) {
		context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
		if (OFI_UNLIKELY(context == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
			return -FI_ENOMEM;
		}
		context->err_entry.err	      = 0;
		context->err_entry.op_context = user_context;
		context->next		      = NULL;
		context->byte_counter	      = len - immediate_total;
		origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
	} else {
		context			  = NULL;
		origin_byte_counter_vaddr = (uintptr_t) NULL;
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, addr.lid, dest_rx,
					    addr.reliability_rx, &psn_ptr, &replay, reliability, hfi1_type);
	if (OFI_UNLIKELY(psn == -1)) {
		if (OFI_LIKELY(do_cq_completion)) {
			OPX_BUF_FREE(context);
		}
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "FI_EAGAIN\n");
		return -FI_EAGAIN;
	}

	FI_OPX_DEBUG_COUNTERS_INC_COND(
		src_iface != FI_HMEM_SYSTEM,
		opx_ep->debug_counters.hmem.hfi.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG].send.rzv);

	if (immediate_tail) {
		uint8_t *buf_tail_bytes = ((uint8_t *) buf + len) - OPX_IMMEDIATE_TAIL_BYTE_COUNT;
		if (src_iface != FI_HMEM_SYSTEM) {
			struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
			opx_copy_from_hmem(
				src_iface, src_device_id, desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE,
				opx_ep->hmem_copy_buf, buf_tail_bytes, OPX_IMMEDIATE_TAIL_BYTE_COUNT,
				desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD : OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
			buf_tail_bytes = opx_ep->hmem_copy_buf;
		}

		for (int i = 0; i < OPX_IMMEDIATE_TAIL_BYTE_COUNT; ++i) {
			immediate_info.tail_bytes[i] = buf_tail_bytes[i];
		}
	}

	/*
	 * Write the 'start of packet' (hw+sw header) 'send control block'
	 * which will consume a single pio credit.
	 */

	uint64_t       force_credit_return = OPX_PBC_CR(opx_ep->tx->force_credit_return, hfi1_type);
	const uint64_t opcode =
		(uint64_t) ((caps & FI_MSG) ?
				    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
									 FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
				    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
									 FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));

	volatile uint64_t *const scb = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);

	uint64_t temp[8];

	fi_opx_store_and_copy_qw(scb, temp,
				 opx_ep->tx->rzv_9B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | force_credit_return |
					 OPX_PBC_DLID_TO_PBC_DLID(addr.lid, hfi1_type),
				 opx_ep->tx->rzv_9B.hdr.qw_9B[0] | lrh_dlid_9B | ((uint64_t) lrh_dws << 32),
				 opx_ep->tx->rzv_9B.hdr.qw_9B[1] | bth_rx | opcode,
				 opx_ep->tx->rzv_9B.hdr.qw_9B[2] | psn,
				 opx_ep->tx->rzv_9B.hdr.qw_9B[3] | (((uint64_t) data) << 32),
				 opx_ep->tx->rzv_9B.hdr.qw_9B[4] | (1ull << 48), len, tag);

	/* consume one credit for the packet header */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	unsigned credits_consumed = 1;
#endif

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	fi_opx_copy_hdr9B_cacheline(&replay->scb.scb_9B, temp);

	/*
	 * write the rendezvous payload "send control blocks"
	 */

	volatile uint64_t *scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
	fi_opx_store_and_copy_qw(scb_payload, temp, 0,		    /* contig_9B_padding */
				 (uintptr_t) buf + immediate_total, /* src_vaddr */
				 (len - immediate_total),	    /* src_len */
				 src_device_id, (uint64_t) src_iface, immediate_info.qw0, origin_byte_counter_vaddr,
				 0 /* unused */);

	/* consume one credit for the rendezvous payload metadata */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

	uint64_t *replay_payload = replay->payload;

	assert(!replay->use_iov);
	assert(((uint8_t *) replay_payload) == ((uint8_t *) &replay->data));
	fi_opx_copy_cacheline(replay_payload, temp);
	replay_payload += FI_OPX_CACHE_LINE_QWS;

	uint8_t *sbuf;
	if (src_iface != FI_HMEM_SYSTEM && immediate_total) {
		struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
		opx_copy_from_hmem(src_iface, src_device_id,
				   desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE, opx_ep->hmem_copy_buf,
				   buf, immediate_total,
				   desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD : OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
		sbuf = opx_ep->hmem_copy_buf;
	} else {
		sbuf = (uint8_t *) buf;
	}

	scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

	/* immediate_byte and immediate_qw are "packed" in the current implementation             */
	/* meaning the immediate bytes are filled, then followed by the rest of the data directly */
	/* adjacent to the packed bytes.  It's probably more efficient to leave a pad and not go  */
	/* through the confusion of finding these boundaries on both sides of the rendezvous      */
	/* That is, just pack the immediate bytes, then pack the "rest" in the immediate qws      */
	/* This would lead to more efficient packing on both sides at the expense of              */
	/* wasting space of a common 0 byte immediate                                             */
	/* tmp_payload_t represents the second cache line of the rts packet                       */
	/* fi_opx_hfi1_packet_payload -> rendezvous -> contiguous                                 */
	struct tmp_payload_t {
		uint8_t	 immediate_byte[8];
		uint64_t immediate_qw[7];
	} __attribute__((packed));

	uint64_t *sbuf_qw = (uint64_t *) (sbuf + immediate_byte_count);
	if (immediate_fragment) {
		struct tmp_payload_t *tmp_payload = (void *) temp;

		for (int i = 0; i < immediate_byte_count; ++i) {
			tmp_payload->immediate_byte[i] = sbuf[i];
		}

		for (int i = 0; i < immediate_qw_count; ++i) {
			tmp_payload->immediate_qw[i] = sbuf_qw[i];
		}
		fi_opx_store_scb_qw(scb_payload, temp);
		sbuf_qw += immediate_qw_count;

		fi_opx_copy_cacheline(replay_payload, temp);
		replay_payload += FI_OPX_CACHE_LINE_QWS;

		/* consume one credit for the rendezvous payload immediate data */
		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
	}

	if (immediate_block) {
		scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		fi_opx_store_scb_qw(scb_payload, sbuf_qw);
		fi_opx_copy_cacheline(replay_payload, sbuf_qw);

		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
	}

	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, addr.reliability_rx, dest_rx,
							    psn_ptr, replay, reliability, hfi1_type);

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
#ifndef NDEBUG
	assert(credits_consumed == total_credits_needed);
#endif

	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	if (OFI_LIKELY(do_cq_completion)) {
		fi_opx_ep_tx_cq_completion_rzv(ep, context, len, lock_required, tag, caps);
	}

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-RTS-HFI:%ld", tag);
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, HFI -- RENDEZVOUS RTS (end) context %p\n",
		     user_context);

	return FI_SUCCESS;
}

ssize_t fi_opx_hfi1_tx_send_rzv_16B(struct fid_ep *ep, const void *buf, size_t len, void *desc, fi_addr_t dest_addr,
				    uint64_t tag, void *user_context, const uint32_t data, int lock_required,
				    const unsigned override_flags, const uint64_t tx_op_flags, const uint64_t dest_rx,
				    const uint64_t caps, const enum ofi_reliability_kind reliability,
				    const uint64_t do_cq_completion, const enum fi_hmem_iface src_iface,
				    const uint64_t src_device_id, const enum opx_hfi1_type hfi1_type)
{
	// We should already have grabbed the lock prior to calling this function
	assert(!lock_required);

	// Need at least one full block of payload
	assert(len >= FI_OPX_HFI1_TX_MIN_RZV_PAYLOAD_BYTES);

	struct fi_opx_ep       *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr   = {.fi = dest_addr};

	const uint64_t is_intranode = fi_opx_hfi1_tx_is_intranode(opx_ep, addr, caps);

	const uint64_t bth_rx	    = ((uint64_t) dest_rx) << OPX_BTH_RX_SHIFT;
	const uint64_t lrh_dlid_16B = addr.lid;

	const bool send_immed_data = (src_iface == FI_HMEM_SYSTEM);

#ifndef NDEBUG
	const uint64_t max_immediate_block_count = (FI_OPX_HFI1_PACKET_MTU >> 6) - 2;
#endif
	/* Expected tid needs to send a leading data block and trailing data
	 * for alignment. TID writes must start on a 64-byte boundary, so we
	 * need to send 64 bytes of leading immediate data that allow us
	 * to shift the receive buffer starting offset to a TID-friendly value.
	 * TID writes must also be a length that is a multiple of a DW (WFR & JKR 9B)
	 * or a QW (JKR), so send the last 7 bytes of the source data immediately
	 * so we can adjust the length after proper alignment has been achieved. */
	const uint8_t immediate_block =
		(send_immed_data && !is_intranode && opx_ep->use_expected_tid_rzv &&
		 len >= opx_ep->tx->sdma_min_payload_bytes && len >= opx_ep->tx->tid_min_payload_bytes) ?
			1 :
			0;
	const uint8_t immediate_tail = immediate_block;

	assert(immediate_block <= 1);
	assert(immediate_tail <= 1);
	assert(immediate_block <= max_immediate_block_count);

	const uint8_t immediate_byte_count = send_immed_data ? (uint8_t) (len & 0x0007ul) : 0;
	const uint8_t immediate_qw_count   = send_immed_data ? (uint8_t) ((len >> 3) & 0x0007ul) : 0;
	const uint8_t immediate_fragment   = send_immed_data ? (uint8_t) (((len & 0x003Ful) + 63) >> 6) : 0;
	assert(immediate_fragment == 1 || immediate_fragment == 0);

	/* Need a full block for ICRC after the end block... */
	const uint64_t icrc_end_block = immediate_block;

	/* ... otherwise need a qw (or block) in the immediate fragment */
	const uint64_t icrc_fragment = icrc_end_block ? 0 : immediate_fragment;

	/* if there are already 7 qw's need a full block */
	const uint64_t icrc_fragment_block = icrc_fragment && (immediate_qw_count == 7) ? 1 : 0;

	/* Summary: we can add the tail qw in...
	 * - rzv metadata if there is no other immediate data
	 * - an empty fragment qw if there are no other blocks (icrc_fragment & !icrc_fragment_block)
	 * - a full (additional) fragment block if there are no other blocks (icrc_fragment & icrc_fragment_block)
	 * - a full (additional) trailing block after the end (icrc_end_block)
	 */

	/* Immediate total does not include trailing block */
	const uint64_t immediate_total = immediate_byte_count + immediate_qw_count * sizeof(uint64_t) +
					 immediate_block * sizeof(union cacheline);

	union fi_opx_hfi1_rzv_rts_immediate_info immediate_info = {
		.count = (immediate_byte_count << OPX_IMMEDIATE_BYTE_COUNT_SHIFT) |
			 (immediate_qw_count << OPX_IMMEDIATE_QW_COUNT_SHIFT) |
			 (immediate_block << OPX_IMMEDIATE_BLOCK_SHIFT) | (immediate_tail << OPX_IMMEDIATE_TAIL_SHIFT),
		.tail_bytes = {}};

	assert(icrc_end_block + icrc_fragment_block < 2); /* not both */
	assert(((len - immediate_total) & 0x003Fu) == 0);

	/* full blocks only. icrc_end_block/icrc_fragment_block count 1 qw only */
	const uint64_t payload_blocks_total = 1 + /* last kdeth + rzv metadata */
					      immediate_fragment + immediate_block;

	const uint64_t pbc_dws = 2 + /* pbc */
				 4 + /* lhr */
				 3 + /* bth */
				 /* 9 +  kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 7 +			       /* kdeth */
				 (payload_blocks_total << 4) + /* includes last kdeth + metadata + immediate data */
				 ((icrc_end_block | icrc_fragment_block) << 1); /* 1 QW of any added tail block */

	const uint16_t lrh_qws = (pbc_dws - 2) >> 1; /* (LRH QW) does not include pbc (8 bytes) */

	if (is_intranode) {
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND 16B, SHM -- RENDEZVOUS RTS (begin) context %p\n",
			user_context);
		OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-RTS-SHM");
		uint64_t			 pos;
		ssize_t				 rc;
		union opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
			&opx_ep->tx->shm, addr.hfi1_unit, dest_rx, &pos, opx_ep->daos_info.hfi_rank_enabled,
			opx_ep->daos_info.rank, opx_ep->daos_info.rank_inst, &rc);

		if (!hdr) {
			FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA, "return %zd\n", rc);
			return rc;
		}

		struct opx_context *context;
		uintptr_t	    origin_byte_counter_vaddr;
		if (OFI_LIKELY(do_cq_completion)) {
			context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
			if (OFI_UNLIKELY(context == NULL)) {
				FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
				return -FI_ENOMEM;
			}
			context->err_entry.err	      = 0;
			context->err_entry.op_context = user_context;
			context->next		      = NULL;
			context->byte_counter	      = len - immediate_total;
			origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
		} else {
			context			  = NULL;
			origin_byte_counter_vaddr = (uintptr_t) NULL;
		}

		FI_OPX_DEBUG_COUNTERS_INC_COND(
			src_iface != FI_HMEM_SYSTEM,
			opx_ep->debug_counters.hmem.intranode.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG]
				.send.rzv);

		const uint64_t opcode =
			(uint64_t) ((caps & FI_MSG) ?
					    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
										 FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
					    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
										 FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));
		hdr->qw_16B[0] =
			opx_ep->tx->rzv_16B.hdr.qw_16B[0] |
			((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
			((uint64_t) lrh_qws << 20);

		hdr->qw_16B[1] = opx_ep->tx->rzv_16B.hdr.qw_16B[1] |
				 ((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
					      OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
				 (uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B);

		hdr->qw_16B[2] = opx_ep->tx->rzv_16B.hdr.qw_16B[2] | bth_rx | opcode;
		hdr->qw_16B[3] = opx_ep->tx->rzv_16B.hdr.qw_16B[3];
		hdr->qw_16B[4] = opx_ep->tx->rzv_16B.hdr.qw_16B[4] | (((uint64_t) data) << 32);
		hdr->qw_16B[5] = opx_ep->tx->rzv_16B.hdr.qw_16B[4] | (1ull << 48); /* effectively 1 iov */
		hdr->qw_16B[6] = len;
		hdr->qw_16B[7] = tag;

		union fi_opx_hfi1_packet_payload *const payload = (union fi_opx_hfi1_packet_payload *) (hdr + 1);

		struct opx_payload_rzv_contig *contiguous = &payload->rendezvous.contiguous_16B;
		contiguous->src_vaddr			  = (uintptr_t) buf + immediate_total;
		contiguous->src_len			  = len - immediate_total;
		contiguous->src_device_id		  = src_device_id;
		contiguous->src_iface			  = (uint64_t) src_iface;
		contiguous->immediate_info		  = immediate_info.qw0;
		contiguous->origin_byte_counter_vaddr	  = origin_byte_counter_vaddr;
		contiguous->unused			  = 0;

		if (immediate_total) {
			uint8_t *sbuf;
			if (src_iface != FI_HMEM_SYSTEM) {
				struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
				opx_copy_from_hmem(src_iface, src_device_id,
						   desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE,
						   opx_ep->hmem_copy_buf, buf, immediate_total,
						   desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD :
							     OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
				sbuf = opx_ep->hmem_copy_buf;
			} else {
				sbuf = (uint8_t *) buf;
			}

			for (int i = 0; i < immediate_byte_count; ++i) {
				contiguous->immediate_byte[i] = sbuf[i];
			}
			sbuf += immediate_byte_count;

			uint64_t *sbuf_qw = (uint64_t *) sbuf;
			for (int i = 0; i < immediate_qw_count; ++i) {
				contiguous->immediate_qw[i] = sbuf_qw[i];
			}

			if (immediate_block) {
				sbuf_qw += immediate_qw_count;
				uint64_t *payload_cacheline =
					(uint64_t *) (&contiguous->cache_line_1 + immediate_fragment);
				fi_opx_copy_cacheline(payload_cacheline, sbuf_qw);
			}
		}

		opx_shm_tx_advance(&opx_ep->tx->shm, (void *) hdr, pos);

		OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-RTS-SHM");
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			     "===================================== SEND 16B, SHM -- RENDEZVOUS RTS (end) context %p\n",
			     user_context);

		if (OFI_LIKELY(do_cq_completion)) {
			fi_opx_ep_tx_cq_completion_rzv(ep, context, len, lock_required, tag, caps);
		}
		return FI_SUCCESS;
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND 16B, HFI -- RENDEZVOUS RTS (begin) context %p\n",
		     user_context);
	OPX_TRACER_TRACE(OPX_TRACER_BEGIN, "SEND-RZV-RTS-HFI:%ld", tag);

	/*
	 * While the bulk of the payload data will be sent via SDMA once we
	 * get the CTS from the receiver, the initial RTS packet is sent via PIO.
	 */

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	const uint16_t total_credits_needed = (lrh_qws + 1 /* pbc */ + 7) >> 3;

	uint64_t total_credits_available =
		FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return,
									total_credits_needed);
		if (total_credits_available < total_credits_needed) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;

			return -FI_EAGAIN;
		}
	}

	struct opx_context *context;
	uintptr_t	    origin_byte_counter_vaddr;
	if (OFI_LIKELY(do_cq_completion)) {
		context = (struct opx_context *) ofi_buf_alloc(opx_ep->rx->ctx_pool);
		if (OFI_UNLIKELY(context == NULL)) {
			FI_WARN(fi_opx_global.prov, FI_LOG_EP_DATA, "Out of memory.\n");
			return -FI_ENOMEM;
		}
		context->err_entry.err	      = 0;
		context->err_entry.op_context = user_context;
		context->next		      = NULL;
		context->byte_counter	      = len - immediate_total;
		origin_byte_counter_vaddr     = (uintptr_t) &context->byte_counter;
	} else {
		context			  = NULL;
		origin_byte_counter_vaddr = (uintptr_t) NULL;
	}

	struct fi_opx_reliability_tx_replay *replay;
	union fi_opx_reliability_tx_psn	    *psn_ptr;
	int64_t				     psn;

	psn = fi_opx_reliability_get_replay(&opx_ep->ep_fid, &opx_ep->reliability->state, addr.lid, dest_rx,
					    addr.reliability_rx, &psn_ptr, &replay, reliability, hfi1_type);
	if (OFI_UNLIKELY(psn == -1)) {
		if (OFI_LIKELY(do_cq_completion)) {
			OPX_BUF_FREE(context);
		}
		return -FI_EAGAIN;
	}

	FI_OPX_DEBUG_COUNTERS_INC_COND(
		src_iface != FI_HMEM_SYSTEM,
		opx_ep->debug_counters.hmem.hfi.kind[(caps & FI_MSG) ? FI_OPX_KIND_MSG : FI_OPX_KIND_TAG].send.rzv);

	if (immediate_tail) {
		uint8_t *buf_tail_bytes = ((uint8_t *) buf + len) - OPX_IMMEDIATE_TAIL_BYTE_COUNT;
		if (src_iface != FI_HMEM_SYSTEM) {
			struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
			opx_copy_from_hmem(
				src_iface, src_device_id, desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE,
				opx_ep->hmem_copy_buf, buf_tail_bytes, OPX_IMMEDIATE_TAIL_BYTE_COUNT,
				desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD : OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
			buf_tail_bytes = opx_ep->hmem_copy_buf;
		}

		for (int i = 0; i < OPX_IMMEDIATE_TAIL_BYTE_COUNT; ++i) {
			immediate_info.tail_bytes[i] = buf_tail_bytes[i];
		}
	}

	/*
	 * Write the 'start of packet' (hw+sw header) 'send control block'
	 * which will consume a single pio credit.
	 */

	uint64_t       force_credit_return = OPX_PBC_CR(opx_ep->tx->force_credit_return, hfi1_type);
	const uint64_t opcode =
		(uint64_t) ((caps & FI_MSG) ?
				    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS_CQ :
									 FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS) :
				    ((tx_op_flags & FI_REMOTE_CQ_DATA) ? FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS_CQ :
									 FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS));

	volatile uint64_t *const scb = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);

	struct fi_opx_hfi1_txe_scb_16B tmp;

	fi_opx_store_and_copy_scb_16B(
		scb, &tmp,
		opx_ep->tx->rzv_16B.qw0 | OPX_PBC_LEN(pbc_dws, hfi1_type) | force_credit_return |
			OPX_PBC_DLID_TO_PBC_DLID(addr.lid, hfi1_type),
		opx_ep->tx->rzv_16B.hdr.qw_16B[0] |
			((uint64_t) (lrh_dlid_16B & OPX_LRH_JKR_16B_DLID_MASK_16B) << OPX_LRH_JKR_16B_DLID_SHIFT_16B) |
			((uint64_t) lrh_qws << 20),
		opx_ep->tx->rzv_16B.hdr.qw_16B[1] |
			((uint64_t) ((lrh_dlid_16B & OPX_LRH_JKR_16B_DLID20_MASK_16B) >>
				     OPX_LRH_JKR_16B_DLID20_SHIFT_16B)) |
			(uint64_t) (bth_rx >> OPX_LRH_JKR_BTH_RX_ENTROPY_SHIFT_16B),
		opx_ep->tx->rzv_16B.hdr.qw_16B[2] | bth_rx | opcode, opx_ep->tx->rzv_16B.hdr.qw_16B[3] | psn,
		opx_ep->tx->rzv_16B.hdr.qw_16B[4] | (((uint64_t) data) << 32),
		opx_ep->tx->rzv_16B.hdr.qw_16B[5] | (1ull << 48), len);

	/* consume one credit for the packet header */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	unsigned credits_consumed = 1;
#endif
	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);
	tmp.hdr.qw_16B[7] = tag;
	fi_opx_copy_hdr16B_cacheline(&replay->scb.scb_16B, (uint64_t *) &tmp);

	/*
	 * write the rendezvous payload "send control blocks"
	 */

	volatile uint64_t *scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
	uint64_t	   temp[8];

	fi_opx_store_and_copy_qw(scb_payload, temp, tag, /* end of header */
				 /* start of receiver payload/cacheline                                             */
				 (uintptr_t) buf + immediate_total, /* rzv.contiguous.src_vaddr                 */
				 (len - immediate_total),	    /* rzv.contiguous.src_len                   */
				 src_device_id,			    /* rzv.contiguous.src_device_id             */
				 (uint64_t) src_iface,		    /* rzv.contiguous.src_iface                 */
				 immediate_info.qw0,		    /* rzv.contiguous.immediate_info            */
				 origin_byte_counter_vaddr,	    /* rzv.contiguous.origin_byte_counter_vaddr */
				 -1UL /* unused */);		    /* rzv.contiguous.unused[0]                 */

	/* consume one credit for the rendezvous payload metadata */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
#ifndef NDEBUG
	++credits_consumed;
#endif

	uint64_t *replay_payload = replay->payload;

	assert(!replay->use_iov);
	assert(((uint8_t *) replay_payload) == ((uint8_t *) &replay->data));

	/* temp is hdr (1 QW) + payload (7 QW) */
	replay_payload[0] = temp[1];
	replay_payload[1] = temp[2];
	replay_payload[2] = temp[3];
	replay_payload[3] = temp[4];
	replay_payload[4] = temp[5];
	replay_payload[5] = temp[6];
	replay_payload[6] = temp[7];

	replay_payload += OPX_JKR_16B_PAYLOAD_AFTER_HDR_QWS;

	uint8_t *sbuf;
	if (src_iface != FI_HMEM_SYSTEM && immediate_total) {
		struct fi_opx_mr *desc_mr = (struct fi_opx_mr *) desc;
		opx_copy_from_hmem(src_iface, src_device_id,
				   desc_mr ? desc_mr->hmem_dev_reg_handle : OPX_HMEM_NO_HANDLE, opx_ep->hmem_copy_buf,
				   buf, immediate_total,
				   desc_mr ? OPX_HMEM_DEV_REG_SEND_THRESHOLD : OPX_HMEM_DEV_REG_THRESHOLD_NOT_SET);
		sbuf = opx_ep->hmem_copy_buf;
	} else {
		sbuf = (uint8_t *) buf;
	}

	/* immediate_byte and immediate_qw are "packed" in the current implementation             */
	/* meaning the immediate bytes are filled, then followed by the rest of the data directly */
	/* adjacent to the packed bytes.  It's probably more efficient to leave a pad and not go  */
	/* through the confusion of finding these boundaries on both sides of the rendezvous      */
	/* That is, just pack the immediate bytes, then pack the "rest" in the immediate qws      */
	/* This would lead to more efficient packing on both sides at the expense of              */
	/* wasting space of a common 0 byte immediate                                             */
	/* tmp_payload_t represents the second cache line of the rts packet                       */
	/* fi_opx_hfi1_packet_payload -> rendezvous -> contiguous                                 */
	struct tmp_payload_t {
		uint8_t	 immediate_byte[8]; /* rendezvous.contiguous.immediate_byte */
		uint64_t immediate_qw[7];   /* rendezvous.contiguous.immediate_qw */
	} __attribute__((packed));

	uint64_t *sbuf_qw = (uint64_t *) (sbuf + immediate_byte_count);
	if (immediate_fragment) {
		struct tmp_payload_t *tmp_payload = (void *) temp;

		for (int i = 0; i < immediate_byte_count; ++i) {
			tmp_payload->immediate_byte[i] = sbuf[i];
		}

		for (int i = 0; i < immediate_qw_count; ++i) {
			tmp_payload->immediate_qw[i] = sbuf_qw[i];
		}
		scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		fi_opx_store_scb_qw(scb_payload, temp);
		sbuf_qw += immediate_qw_count;

		fi_opx_copy_cacheline(replay_payload, temp);
		replay_payload += FI_OPX_CACHE_LINE_QWS;

		/* consume one credit for the rendezvous payload immediate data */
		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
		/* Need a full tail block */
		if (icrc_fragment_block) {
			/* No other tail or immediate block after this */
			assert(!icrc_end_block && !immediate_block);

			/* Write another block to accomodate the ICRC and tail */
			uint64_t temp_0[8] = {-2UL};
			scb_payload	   = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

			fi_opx_store_scb_qw(scb_payload, temp_0);
			fi_opx_copy_cacheline(replay_payload, temp_0);
			replay_payload += FI_OPX_CACHE_LINE_QWS;

			FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
			++credits_consumed;
#endif
		}
#ifndef NDEBUG
		else if (icrc_fragment) { /* used an immediate qw for tail */
			/* No other tail or immediate block after this */
			assert(!icrc_end_block && !immediate_block);
		} else {
			/* Must be tail and immediate blocks after this */
			assert(icrc_end_block && immediate_block);
		}
#endif
	}

	if (immediate_block) {
		/* Tail will be it's own block */
		assert(icrc_end_block && !icrc_fragment_block && !icrc_fragment);
		scb_payload = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		fi_opx_store_scb_qw(scb_payload, sbuf_qw);
		fi_opx_copy_cacheline(replay_payload, sbuf_qw);
		replay_payload += FI_OPX_CACHE_LINE_QWS;

		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
		/* Write another block to accomodate the ICRC and tail */
		uint64_t temp_0[8] = {-3UL};
		scb_payload	   = FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		fi_opx_store_scb_qw(scb_payload, temp_0);
		fi_opx_copy_cacheline(replay_payload, temp_0);

		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
	}

	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state, addr.reliability_rx, dest_rx,
							    psn_ptr, replay, reliability, hfi1_type);
#ifndef NDEBUG
	assert(credits_consumed == total_credits_needed);
#endif

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);

	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	if (OFI_LIKELY(do_cq_completion)) {
		fi_opx_ep_tx_cq_completion_rzv(ep, context, len, lock_required, tag, caps);
	}

	OPX_TRACER_TRACE(OPX_TRACER_END_SUCCESS, "SEND-RZV-RTS-HFI:%ld", tag);
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND 16B, HFI -- RENDEZVOUS RTS (end) context %p\n",
		     user_context);

	return FI_SUCCESS;
}

unsigned fi_opx_hfi1_handle_poll_error(struct fi_opx_ep *opx_ep, volatile uint64_t *rhe_ptr, volatile uint32_t *rhf_ptr,
				       const uint32_t rhf_msb, const uint32_t rhf_lsb, const uint64_t rhf_seq,
				       const uint64_t hdrq_offset, const uint64_t rhf_rcvd,
				       const union opx_hfi1_packet_hdr *const hdr, const enum opx_hfi1_type hfi1_type)
{
	/* We are assuming that we can process any error and consume this header,
	   let reliability detect and replay it as needed. */

	(void) rhf_ptr; /* unused unless debug is turned on */

	/* drop this packet and allow reliability protocol to retry */
#ifdef OPX_RELIABILITY_DEBUG
	fprintf(stderr,
		"%s:%s():%d drop this packet and allow reliability protocol to retry, psn = %u, RHF %#16.16lX, OPX_RHF_IS_USE_EGR_BUF %u, hdrq_offset %lu\n",
		__FILE__, __func__, __LINE__, FI_OPX_HFI1_PACKET_PSN(hdr), rhf_rcvd,
		OPX_RHF_IS_USE_EGR_BUF(rhf_rcvd, hfi1_type), hdrq_offset);

#endif

	OPX_RHE_DEBUG(opx_ep, rhe_ptr, rhf_ptr, rhf_msb, rhf_lsb, rhf_seq, hdrq_offset, rhf_rcvd, hdr, hfi1_type);

	if (OPX_RHF_IS_USE_EGR_BUF(rhf_rcvd, hfi1_type)) {
		/* "consume" this egrq element */
		const uint32_t egrbfr_index	 = OPX_RHF_EGR_INDEX(rhf_rcvd, hfi1_type);
		const uint32_t last_egrbfr_index = opx_ep->rx->egrq.last_egrbfr_index;
		if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
			OPX_HFI1_BAR_STORE(opx_ep->rx->egrq.head_register, ((const uint64_t) last_egrbfr_index));
			opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
		}
	}

	/* "consume" this hdrq element */
	opx_ep->rx->state.hdrq.rhf_seq = OPX_RHF_SEQ_INCREMENT(rhf_seq, hfi1_type);
	opx_ep->rx->state.hdrq.head    = hdrq_offset + FI_OPX_HFI1_HDRQ_ENTRY_SIZE_DWS;

	fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset);

	return 1;
}
