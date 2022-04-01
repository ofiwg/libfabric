/*
 * Copyright (C) 2022 by Cornelis Networks.
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

#include "rdma/fabric.h" // only for 'fi_addr_t' ... which is a typedef to uint64_t
#include "rdma/opx/fi_opx_hfi1.h"
#include "rdma/opx/fi_opx.h"
#include "rdma/opx/fi_opx_eq.h"
#include "ofi_mem.h"
#include "opa_user.h"
#include "fi_opx_hfi_select.h"

#define BYTE2DWORD_SHIFT	(2)

#define ESSP_SL_DEFAULT		(0)	/* PSMI_SL_DEFAULT */
#define ESSP_SC_DEFAULT		(0)	/* PSMI_SC_DEFAULT */
#define ESSP_VL_DEFAULT		(0)	/* PSMI_VL_DEFAULT */
#define ESSP_SC_ADMIN		(15)	/* PSMI_SC_ADMIN */
#define ESSP_VL_ADMIN		(15)	/* PSMI_VL_ADMIN */

struct fi_opx_hfi1_context_internal {
	struct fi_opx_hfi1_context	context;

	struct hfi1_user_info_dep	user_info;
	struct _hfi_ctrl *		ctrl;

};

/*
 * Return the NUMA node id where the process is currently running.
 */
static int opx_get_current_proc_location()
{
        int core_id, node_id;

    core_id = sched_getcpu();
    if (core_id < 0)
        return -EINVAL;

    node_id = numa_node_of_cpu(core_id);
    if (node_id < 0)
        return -EINVAL;

    return node_id;
}

static int opx_get_current_proc_core()
{
	int core_id;
	core_id = sched_getcpu();
	if (core_id < 0)
		return -EINVAL;
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
	return  (rcvhdrq_cnt - 1) * 32;
}

// Used by fi_opx_hfi1_context_open as a convenience.
static int opx_open_hfi_and_context(struct _hfi_ctrl **ctrl,
				    struct fi_opx_hfi1_context_internal *internal,
					uuid_t unique_job_key,
				    int hfi_unit_number)
{
	int fd;

	fd = opx_hfi_context_open(hfi_unit_number, 0, 0);
	if (fd < 0) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "Unable to open HFI unit %d.\n",
			hfi_unit_number);
		fd = -1;
	} else {
		memset(&internal->user_info, 0, sizeof(internal->user_info));
		internal->user_info.userversion =
			HFI1_USER_SWMINOR |
			(opx_hfi_get_user_major_version() << HFI1_SWMAJOR_SHIFT);

		/* do not share hfi contexts */
		internal->user_info.subctxt_id = 0;
		internal->user_info.subctxt_cnt = 0;

		memcpy(internal->user_info.uuid, unique_job_key,
			sizeof(internal->user_info.uuid));

		*ctrl = opx_hfi_userinit(fd, &internal->user_info);
		if (!*ctrl) {
			opx_hfi_context_close(fd);
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Unable to open a context on HFI unit %d.\n",
				hfi_unit_number);
			fd = -1;
		}
	}
	return fd;
}

/*
 * Open a context on the first HFI that shares our process' NUMA node.
 * If no HFI shares our NUMA node, grab the first active HFI.
 */
struct fi_opx_hfi1_context *fi_opx_hfi1_context_open(struct fid_ep *ep, uuid_t unique_job_key)
{
	struct fi_opx_ep *opx_ep = (ep == NULL) ? NULL : container_of(ep, struct fi_opx_ep, ep_fid);
	int fd = -1;
	int hfi_unit_number = -1;
	const int numa_node_id = opx_get_current_proc_location();
	const int core_id = opx_get_current_proc_core();
	const int hfi_count = opx_hfi_get_num_units();
	int hfi_candidates[FI_OPX_MAX_HFIS];
	int hfi_distances[FI_OPX_MAX_HFIS];
	int hfi_candidates_count = 0;
	int hfi_candidate_index = -1;
	struct _hfi_ctrl *ctrl = NULL;
	bool use_default_logic = true;

	struct fi_opx_hfi1_context_internal *internal =
		calloc(1, sizeof(struct fi_opx_hfi1_context_internal));

	struct fi_opx_hfi1_context *context = &internal->context;

	/*
	 * Force cpu affinity if desired. Normally you would let the
	 * job scheduler (such as mpirun) handle this.
	 */
	int force_cpuaffinity = 0;
	fi_param_get_bool(fi_opx_global.prov,"force_cpuaffinity",
		&force_cpuaffinity);
	if (force_cpuaffinity) {
		const int cpu_id = sched_getcpu();
		cpu_set_t cpuset;
		CPU_ZERO(&cpuset);
		CPU_SET(cpu_id, &cpuset);
		if (sched_setaffinity(0, sizeof(cpuset), &cpuset)) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Unable to force cpu affinity. %s\n", strerror(errno));
		}
	}

	/*
	 * open the hfi1 context
	 */
	context->fd = -1;
	internal->ctrl = NULL;

	// If FI_OPX_HFI_SELECT is specified, skip all this and
	// use its value as the selected hfi unit.
	char *env = NULL;
	if (FI_SUCCESS == fi_param_get_str(&fi_opx_provider, "hfi_select", &env)) {

		struct hfi_selector selector = {0};
		use_default_logic = false;

		int selectors, matched;
		selectors = matched = 0;
		const char *s;
		for (s = env; *s != '\0'; ) {
			s = hfi_selector_next(s, &selector);
			if (!s) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error occurred parsing HFI selector string \"%s\"\n", env);
				return NULL;
			}

			if (selector.type == HFI_SELECTOR_DEFAULT) {
				use_default_logic = true;
				break;
			}

			if (selector.unit >= hfi_count) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error: selector unit %d >= number of HFIs %d\n",
					selector.unit, hfi_count);
				return NULL;
			} else if (!opx_hfi_get_unit_active(selector.unit)) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error: selected unit %d is not active\n", selector.unit);
				return NULL;
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
						return NULL;
					}

					if (selector.mapby.rangeE > max_numa){
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"mapby numa end of range %d > numa_max_node %d\n",
							selector.mapby.rangeE, max_numa);
						return NULL;
					}

					if (selector.mapby.rangeS <= numa_node_id && selector.mapby.rangeE >= numa_node_id){
						hfi_unit_number = selector.unit;
						matched++;
						break;
					}
				} else if (selector.mapby.type == HFI_SELECTOR_MAPBY_CORE) {
					int max_core = get_nprocs();
					if (selector.mapby.rangeS > max_core) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"Error: mapby core %d > nprocs %d\n",
							selector.mapby.rangeS, max_core);
						return NULL;
					}
					if (selector.mapby.rangeE > max_core) {
						FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
							"mapby core end of range %d > nprocs %d\n",
							selector.mapby.rangeE, max_core);
						return NULL;
					}
					if (selector.mapby.rangeS <= core_id && selector.mapby.rangeE >= core_id){
						hfi_unit_number = selector.unit;
						matched++;
						break;
					} 
				} else {
					FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
						"Error: unsupported mapby type %d\n", selector.mapby.type);
					return NULL;
				}
			} else {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Error: unsupported selector type %d\n", selector.type);
				return NULL;
			}
			selectors++;
		}

		if (!use_default_logic) {
			if (!matched) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "No HFI selectors matched.\n");
				return NULL;
			}

			hfi_candidates[0] = hfi_unit_number;
			hfi_distances[0] = 0;
			hfi_candidates_count = 1;
			FI_TRACE(&fi_opx_provider, FI_LOG_FABRIC,
				"User-specified HFI selection set to %d. Skipping HFI selection algorithm \n",
				hfi_unit_number);

			fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key,
				hfi_unit_number);
			if (fd < 0) {
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"Unable to open user-specified HFI.\n");
				return NULL;
			}
		}

	} else if (opx_ep && opx_ep->common_info->src_addr &&
		((union fi_opx_addr *)(opx_ep->common_info->src_addr))->hfi1_unit != opx_default_addr.hfi1_unit) {
		union fi_opx_addr addr;
		use_default_logic = false;
		/*
		 * No Context Resource Management Framework supported by OPX to enable
		 * acquiring a context with attributes that exactly match the specified
		 * source address.
		 *
		 * Therefore, treat the source address as an ‘opaque’ ID and extract the
		 * essential data required to create a context that at least maps to the
		 * same HFI and HFI port (Note, assigned LID unchanged unless modified
		 * by the OPA FM).
		 */
		memset(&addr, 0, sizeof(addr));
		memcpy(&addr.fi, opx_ep->common_info->src_addr, opx_ep->common_info->src_addrlen);

		hfi_unit_number = addr.hfi1_unit;
		hfi_candidates[0] = hfi_unit_number;
		hfi_distances[0] = 0;
		hfi_candidates_count = 1;
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
			"Application-specified HFI selection set to %d. Skipping HFI selection algorithm \n",
			hfi_unit_number);

		fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key, hfi_unit_number);
		if (fd < 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"Unable to open application-specified HFI.\n");
			return NULL;
		}

	} 
	if (use_default_logic){
		/* Select the best HFI to open a context on */
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC, "Found HFIs = %d\n", hfi_count);

		if (hfi_count == 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"FATAL: detected no HFIs, cannot continue\n");
			return NULL;
		}

		else if (hfi_count == 1) {
			if (opx_hfi_get_unit_active(0) > 0) {
				// Only 1 HFI, populate the candidate list and continue.
				FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
					"Detected one HFI and it has active ports, selected it\n");
				hfi_candidates[0] = 0;
				hfi_distances[0] = 0;
				hfi_candidates_count = 1;
			} else {
				// No active ports, we're done here.
				FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
					"FATAL: HFI has no active ports, cannot continue\n");
				return NULL;
			}

		} else {
			// The system has multiple HFIs. Sort them by distance from
			// this process.
			int hfi_n, hfi_d;
			for (int i = 0; i < hfi_count; i++) {
				if (opx_hfi_get_unit_active(i) > 0) {
					hfi_n = opx_hfi_sysfs_unit_read_node_s64(i);
					hfi_d = numa_distance(hfi_n, numa_node_id);
					FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
						"HFI unit %d in numa node %d has a distance of %d from this pid.\n",
						i, hfi_n, hfi_d);
					hfi_candidates[hfi_candidates_count] = i;
					hfi_distances[hfi_candidates_count] = hfi_d;
					int j = hfi_candidates_count;
					// Bubble the new HFI up till the list is sorted.
					// Yes, this is lame but the practical matter is that
					// there will never be so many HFIs on a single system
					// that a real insertion sort is justified. Also, doing it
					// this way results in a deterministic result - HFIs will
					// be implicitly sorted by their unit number as well as
					// by distance ensuring that all processes in a NUMA node
					// will see the HFIs in the same order.
					while (j > 0 && hfi_distances[j - 1] > hfi_distances[j]) {
						int t1 = hfi_distances[j - 1];
						int t2 = hfi_candidates[j - 1];
						hfi_distances[j - 1] = hfi_distances[j];
						hfi_candidates[j - 1] = hfi_candidates[j];
						hfi_distances[j] = t1;
						hfi_candidates[j] = t2;
						j--;
					}
					hfi_candidates_count++;
				}
			}
		}

		// At this point we have a list of HFIs, sorted by distance from this
		// pid (and by unit # as an implied key).  Pick from the closest HFIs
		// based on the modulo of the pid. If we fail to open that HFI, try
		// another one at the same distance. If that fails, we will try HFIs
		// that are further away.
		int lower = 0;
		int higher = 0;
		do {
			// Find the set of HFIs at this distance. Again, no attempt is
			// made to make this fast.
			higher = lower + 1;
			while (higher < hfi_candidates_count &&
			       hfi_distances[higher] == hfi_distances[lower]) {
				higher++;
			}

			// Use the modulo of the pid to select an HFI. The intent
			// is to use HFIs evenly rather than have many pids open
			// the 1st HFi then have many select the next HFI, etc...
			int range = higher - lower;
			hfi_candidate_index = getpid() % range + lower;
			hfi_unit_number = hfi_candidates[hfi_candidate_index];

			// Try to open the HFI. If we fail, try the other HFIs
			// at that distance until we run out of HFIs at that
			// distance.
			fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key,
				hfi_unit_number);
			int t = range;
			while (fd < 0 && t-- > 1) {
				hfi_candidate_index++;
				if (hfi_candidate_index >= higher)
					hfi_candidate_index = lower;
				hfi_unit_number = hfi_candidates[hfi_candidate_index];
				fd = opx_open_hfi_and_context(&ctrl, internal, unique_job_key,
					hfi_unit_number);
			}

			// If we still haven't successfully chosen an HFI,
			// try HFIs that are further away.
			lower = higher;
		} while (fd < 0 && lower < hfi_candidates_count);

		if (fd < 0) {
			FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
				"FATAL: Found %d active HFI device%s, unable to open %s.\n",
				hfi_candidates_count, (hfi_candidates_count > 1) ? "s" : "",
				(hfi_candidates_count > 1) ? "any of them" : "it");
			return NULL;
		}
	}

	FI_TRACE(&fi_opx_provider, FI_LOG_FABRIC,
		"Selected HFI is %d; caller NUMA domain is %d; HFI NUMA domain is %"PRId64"\n",
		hfi_unit_number, numa_node_id, opx_hfi_sysfs_unit_read_node_s64(hfi_unit_number));

	// Alert user if the final choice wasn't optimal.
	if (opx_hfi_sysfs_unit_read_node_s64(hfi_unit_number) != numa_node_id) {
		FI_WARN(&fi_opx_provider, FI_LOG_FABRIC,
			"Selected HFI is %d. It does not appear to be local to this pid's numa domain which is %d\n",
			hfi_unit_number, numa_node_id);
	} else {
		FI_INFO(&fi_opx_provider, FI_LOG_FABRIC,
			"Selected HFI unit %d in the same numa node as this pid.\n",
			hfi_unit_number);
	}

	context->fd = fd;
	internal->ctrl = ctrl; /* memory was allocated during 'opx_hfi_userinit()' */

	int lid = 0;
	lid = opx_hfi_get_port_lid(ctrl->__hfi_unit, ctrl->__hfi_port);
	assert(lid > 0);

	uint64_t gid_hi, gid_lo;
	int rc __attribute__((unused)) = -1;
	rc = opx_hfi_get_port_gid(ctrl->__hfi_unit, ctrl->__hfi_port, &gid_hi, &gid_lo);
	assert(rc != -1);

	/* these don't change - move to domain ? */
	context->hfi_unit = ctrl->__hfi_unit;
	context->hfi_port = ctrl->__hfi_port;
	context->lid = (uint16_t)lid;
	context->gid_hi = gid_hi;
	context->gid_lo = gid_lo;

	context->sl = ESSP_SL_DEFAULT;

	rc = opx_hfi_get_port_sl2sc(ctrl->__hfi_unit, ctrl->__hfi_port, ESSP_SL_DEFAULT);
	if (rc < 0)
		context->sc = ESSP_SC_DEFAULT;
	else
		context->sc = rc;

	rc = opx_hfi_get_port_sc2vl(ctrl->__hfi_unit, ctrl->__hfi_port, context->sc);
	if (rc < 0)
		context->vl = ESSP_VL_DEFAULT;
	else
		context->vl = rc;

	assert(context->sc != ESSP_SC_ADMIN);
	assert(context->vl != ESSP_VL_ADMIN);
	assert((context->vl == 15) || (context->vl <= 7));

	context->mtu = opx_hfi_get_port_vl2mtu(ctrl->__hfi_unit, ctrl->__hfi_port, context->vl);
	assert(context->mtu >= 0);

	rc = opx_hfi_set_pkey(ctrl, HFI_DEFAULT_P_KEY);

	const struct hfi1_base_info *base_info = &ctrl->base_info;
	const struct hfi1_ctxt_info *ctxt_info = &ctrl->ctxt_info;

	/*
	 * initialize the hfi tx context
	 */

	context->bthqp = (uint8_t)base_info->bthqp;
	context->jkey = base_info->jkey;
	context->send_ctxt = ctxt_info->send_ctxt;

	context->info.pio.scb_sop_first =
		(volatile uint64_t *)(ptrdiff_t)base_info->pio_bufbase_sop; // tx->pio_bufbase_sop
	context->info.pio.scb_first =
		(volatile uint64_t *)(ptrdiff_t)base_info->pio_bufbase; // tx->pio_bufbase
	context->info.pio.credits_addr = (volatile uint64_t *)(ptrdiff_t)base_info->sc_credits_addr;

	const uint64_t credit_return = *(context->info.pio.credits_addr);
	context->state.pio.free_counter_shadow = (uint16_t)(credit_return & 0x00000000000007FFul);
	context->state.pio.fill_counter = 0;
	context->state.pio.scb_head_index = 0;
	context->state.pio.credits_total =
		ctxt_info->credits; /* yeah, yeah .. THIS field is static, but there was an unused halfword at this spot, so .... */

	/* move to domain ? */
	uint8_t i;
	for (i = 0; i < 32; ++i) {
		rc = opx_hfi_get_port_sl2sc(ctrl->__hfi_unit, ctrl->__hfi_port, i);

		if (rc < 0)
			context->sl2sc[i] = ESSP_SC_DEFAULT;
		else
			context->sl2sc[i] = rc;

		rc = opx_hfi_get_port_sc2vl(ctrl->__hfi_unit, ctrl->__hfi_port, i);
		if (rc < 0)
			context->sc2vl[i] = ESSP_VL_DEFAULT;
		context->sc2vl[i] = rc;
	}

	context->info.sdma.queue_size = ctxt_info->sdma_ring_size - 1;
	context->info.sdma.available_counter = context->info.sdma.queue_size;
	context->info.sdma.fill_index = 0;
	context->info.sdma.done_index = 0;
	context->info.sdma.completion_queue =
		(struct hfi1_sdma_comp_entry *)base_info->sdma_comp_bufbase;

	/*
	 * initialize the hfi rx context
	 */

	context->info.rxe.id = ctrl->ctxt_info.ctxt;
	context->info.rxe.hdrq.rhf_off = (ctxt_info->rcvhdrq_entsize - 8) >> BYTE2DWORD_SHIFT;

	/* hardware registers */
	volatile uint64_t *uregbase = (volatile uint64_t *)(uintptr_t)base_info->user_regbase;
	context->info.rxe.hdrq.head_register = (volatile uint64_t *)&uregbase[ur_rcvhdrhead];
	context->info.rxe.hdrq.tail_register = (volatile uint64_t *)&uregbase[ur_rcvhdrtail];
	context->info.rxe.egrq.head_register = (volatile uint64_t *)&uregbase[ur_rcvegrindexhead];
	context->info.rxe.egrq.tail_register = (volatile uint64_t *)&uregbase[ur_rcvegrindextail];
	context->info.rxe.uregbase = uregbase;

	context->runtime_flags = ctxt_info->runtime_flags;

	if (context->runtime_flags & HFI1_CAP_DMA_RTAIL) {
		context->info.rxe.hdrq.rhf_notail = 0;
	} else {
		context->info.rxe.hdrq.rhf_notail = 1;
	}

	context->info.rxe.hdrq.elemsz = ctxt_info->rcvhdrq_entsize >> BYTE2DWORD_SHIFT;
	context->info.rxe.hdrq.elemcnt = ctxt_info->rcvhdrq_cnt;
	context->info.rxe.hdrq.elemlast =
		((context->info.rxe.hdrq.elemcnt - 1) * context->info.rxe.hdrq.elemsz);
	context->info.rxe.hdrq.rx_poll_mask =
		fi_opx_hfi1_header_count_to_poll_mask(ctxt_info->rcvhdrq_cnt);
	context->info.rxe.hdrq.base_addr = (uint32_t *)(uintptr_t)base_info->rcvhdr_bufbase;
	context->info.rxe.hdrq.rhf_base =
		context->info.rxe.hdrq.base_addr + context->info.rxe.hdrq.rhf_off;

	context->info.rxe.egrq.base_addr = (uint32_t *)(uintptr_t)base_info->rcvegr_bufbase;
	context->info.rxe.egrq.elemsz = ctxt_info->rcvegr_size;
	context->info.rxe.egrq.size = ctxt_info->rcvegr_size * ctxt_info->egrtids;

	return context;
}

int init_hfi1_rxe_state (struct fi_opx_hfi1_context * context,
		struct fi_opx_hfi1_rxe_state * rxe_state)
{
	rxe_state->hdrq.head = 0;

	if (context->runtime_flags & HFI1_CAP_DMA_RTAIL) {
		rxe_state->hdrq.rhf_seq = 0;		/* will be ignored */
	} else {
		rxe_state->hdrq.rhf_seq = 0x10000000u;
	}

	rxe_state->egrq.countdown = 8;

	return 0;
}



#include "rdma/opx/fi_opx_endpoint.h"
#include "rdma/opx/fi_opx_reliability.h"

void fi_opx_hfi1_tx_connect (struct fi_opx_ep *opx_ep, fi_addr_t peer)
{

	if ((opx_ep->tx->caps & FI_LOCAL_COMM) || ((opx_ep->tx->caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == 0)) {

		const uint64_t lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID(peer);
		const uint16_t dlid_be16 = (uint16_t)(FI_OPX_HFI1_LRH_DLID_TO_LID(lrh_dlid));
		const uint16_t slid_be16 = htons(opx_ep->hfi->lid);

		if (slid_be16 == dlid_be16) {
			char buffer[128];
			union fi_opx_addr addr;
			addr.raw64b = (uint64_t)peer;

			snprintf(buffer,sizeof(buffer),"%s-%02x",
				opx_ep->domain->unique_job_key_str, addr.hfi1_unit);
			opx_shm_tx_connect(&opx_ep->tx->shm, (const char * const)buffer,
				addr.hfi1_rx, FI_OPX_SHM_FIFO_SIZE, FI_OPX_SHM_PACKET_SIZE);
		}
	}

	return;
}

__OPX_FORCE_INLINE__
int fi_opx_hfi1_do_rx_rzv_rts_intranode (struct fi_opx_hfi1_rx_rzv_rts_params *params) {

	struct fi_opx_ep * opx_ep = params->opx_ep;
	const uint64_t lrh_dlid = params->lrh_dlid;
	const uint64_t bth_rx = ((uint64_t)params->u8_rx) << 56;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV, SHM -- RENDEZVOUS RTS (begin)\n");
	uint64_t pos;
	union fi_opx_hfi1_packet_hdr * const tx_hdr =
		opx_shm_tx_next(&opx_ep->tx->shm, params->u8_rx, &pos);

	if(!tx_hdr) return -FI_EAGAIN;
	tx_hdr->qw[0] = opx_ep->rx->tx.cts.hdr.qw[0] | lrh_dlid;
	tx_hdr->qw[1] = opx_ep->rx->tx.cts.hdr.qw[1] | bth_rx;
	tx_hdr->qw[2] = opx_ep->rx->tx.cts.hdr.qw[2];
	tx_hdr->qw[3] = opx_ep->rx->tx.cts.hdr.qw[3];
	tx_hdr->qw[4] = opx_ep->rx->tx.cts.hdr.qw[4] | (params->niov << 32) | params->opcode;
	tx_hdr->qw[5] = params->origin_byte_counter_vaddr;
	tx_hdr->qw[6] = params->target_byte_counter_vaddr;


	union fi_opx_hfi1_packet_payload * const tx_payload =
		(union fi_opx_hfi1_packet_payload *)(tx_hdr+1);

	uintptr_t vaddr_with_offset = params->dst_vaddr;
	for(int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i].rbuf = vaddr_with_offset;		/* receive buffer virtual address */
		tx_payload->cts.iov[i].sbuf = (uintptr_t)params->src_iov[i].iov_base;		/* send buffer virtual address */
		tx_payload->cts.iov[i].bytes = params->src_iov[i].iov_len;	/* number of bytes to transfer */
		vaddr_with_offset += params->src_iov[i].iov_len;
	}

	opx_shm_tx_advance(&opx_ep->tx->shm, (void*)tx_hdr, pos);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV, SHM -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}

int fi_opx_hfi1_do_rx_rzv_rts (union fi_opx_hfi1_deferred_work *work) {

	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;
	if (params->is_intranode) {	/* compile-time constant expression */
		return fi_opx_hfi1_do_rx_rzv_rts_intranode(params);
	}

	struct fi_opx_ep * opx_ep = params->opx_ep;
	const uint64_t lrh_dlid = params->lrh_dlid;
	const uint64_t bth_rx = ((uint64_t)params->u8_rx) << 56;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV, HFI -- RENDEZVOUS RTS (begin)\n");
	const uint64_t payload_bytes = params->niov * sizeof(struct fi_opx_hfi1_dput_iov);
	const uint64_t pbc_dws =
		2 + /* pbc */
		2 + /* lrh */
		3 + /* bth */
		9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
		(payload_bytes >> 2) +
		((payload_bytes & 0x3) ? 1 : 0); /* "struct fi_opx_hfi1_dput_iov" * niov */
	const uint16_t lrh_dws = htons(pbc_dws - 1);
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
	const uint16_t total_credits_needed = 1 +   /* packet header */
		((payload_bytes >> 6) + ((payload_bytes & 0x3f) ? 1 : 0)); /* payload blocks needed */
	uint64_t total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);

	if (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;
		if (total_credits_available < total_credits_needed) {
			return -FI_EAGAIN;
		}
	}

	struct fi_opx_reliability_tx_replay *replay =
		fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state, false);
	if (replay == NULL) {
		return -FI_EAGAIN;
	}

	union fi_opx_reliability_tx_psn *psn_ptr = NULL;
	const int64_t psn = (params->reliability != OFI_RELIABILITY_KIND_NONE) ? /* compile-time constant expression */
			fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
							&opx_ep->reliability->state,
							params->slid,
							params->u8_rx,
							params->origin_rs,
							&psn_ptr) :
			0;
	if(OFI_UNLIKELY(psn == -1)) {
		fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
		return -FI_EAGAIN;
	}

	assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);
	// The "memcopy first" code is here as an alternative to the more complicated
	// direct write to pio followed by memory copy of the reliability buffer
	replay->scb.qw0 = opx_ep->rx->tx.cts.qw0 | pbc_dws |
			  ((opx_ep->tx->force_credit_return & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT);
	replay->scb.hdr.qw[0] =
		opx_ep->rx->tx.cts.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
	replay->scb.hdr.qw[1] = opx_ep->rx->tx.cts.hdr.qw[1] | bth_rx;
	replay->scb.hdr.qw[2] = opx_ep->rx->tx.cts.hdr.qw[2] | psn;
	replay->scb.hdr.qw[3] = opx_ep->rx->tx.cts.hdr.qw[3];
	replay->scb.hdr.qw[4] = opx_ep->rx->tx.cts.hdr.qw[4] | (params->niov << 32) | params->opcode;
	replay->scb.hdr.qw[5] = params->origin_byte_counter_vaddr;
	replay->scb.hdr.qw[6] = params->target_byte_counter_vaddr;

	uint8_t *replay_payload = (uint8_t *)replay->payload;
	union fi_opx_hfi1_packet_payload *const tx_payload =
		(union fi_opx_hfi1_packet_payload *)replay_payload;

	uintptr_t vaddr_with_offset = params->dst_vaddr;
	for (int i = 0; i < params->niov; i++) {
		tx_payload->cts.iov[i].rbuf =
			vaddr_with_offset; /* receive buffer virtual address */
		tx_payload->cts.iov[i].sbuf =
			(uintptr_t)params->src_iov[i].iov_base; /* send buffer virtual address */
		tx_payload->cts.iov[i].bytes =
			params->src_iov[i].iov_len; /* number of bytes to transfer */
		vaddr_with_offset += params->src_iov[i].iov_len;
	}

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	fi_opx_reliability_service_do_replay(&opx_ep->reliability->service, replay);
	fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state,
							    params->slid,
							    params->origin_rs,
							    params->origin_rx,
							    psn_ptr, replay, params->reliability);
	FI_DBG_TRACE(
		fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== RECV, HFI -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}

void fi_opx_hfi1_rx_rzv_rts (struct fi_opx_ep *opx_ep,
							 const void * const hdr, const void * const payload,
							 const uint8_t u8_rx, const uint64_t niov,
							 uintptr_t origin_byte_counter_vaddr,
							 uintptr_t target_byte_counter_vaddr,
							 const uintptr_t dst_vaddr,
							 const struct iovec * src_iov,
							 uint8_t opcode,
							 const unsigned is_intranode,
							 const enum ofi_reliability_kind reliability) {

	const union fi_opx_hfi1_packet_hdr * const hfi1_hdr =
		(const union fi_opx_hfi1_packet_hdr * const) hdr;

	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	assert(work != NULL);
	struct fi_opx_hfi1_rx_rzv_rts_params *params = &work->rx_rzv_rts;
	params->opx_ep = opx_ep;
	params->work_elem.slist_entry.next = NULL;
	params->work_elem.work_fn = fi_opx_hfi1_do_rx_rzv_rts;
	params->work_elem.completion_action = NULL;
	params->work_elem.payload_copy = NULL;
	params->lrh_dlid = (hfi1_hdr->stl.lrh.qw[0] & 0xFFFF000000000000ul) >> 32;
	params->slid = hfi1_hdr->stl.lrh.slid;

	params->origin_rx = hfi1_hdr->rendezvous.origin_rx;
	params->origin_rs = hfi1_hdr->rendezvous.origin_rs;
	params->u8_rx = u8_rx;
	params->niov = niov;
	params->origin_byte_counter_vaddr = origin_byte_counter_vaddr;
	params->target_byte_counter_vaddr = target_byte_counter_vaddr;
	params->dst_vaddr = dst_vaddr;
	params->opcode = opcode;
	params->is_intranode = is_intranode;
	params->reliability = reliability;

	assert(niov <= FI_OPX_MAX_DPUT_IOV);
	for(int idx=0; idx < niov; idx++) {
		params->src_iov[idx] = src_iov[idx];
	}

	int rc = fi_opx_hfi1_do_rx_rzv_rts(work);
	if(rc == FI_SUCCESS) {
		ofi_buf_free(work);
		return;
	}
	assert(rc == -FI_EAGAIN);
	/* Try again later*/
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending);
}

int fi_opx_hfi1_do_dput (union fi_opx_hfi1_deferred_work * work) {
	struct fi_opx_hfi1_dput_params *params = &work->dput;
	struct fi_opx_ep * opx_ep = params->opx_ep;
	struct fi_opx_mr * opx_mr = params->opx_mr;
	const uint8_t u8_rx = params->u8_rx;
	const uint32_t niov = params->niov;
	const struct fi_opx_hfi1_dput_iov * const dput_iov = params->dput_iov;
	const uintptr_t target_byte_counter_vaddr = params->target_byte_counter_vaddr;
	uint64_t * origin_byte_counter = params->origin_byte_counter;
	uint64_t key = params->key;
	struct fi_opx_completion_counter *cc = params->cc;
	uint64_t op64 = params->op;
	uint64_t dt64 = params->dt;
	uint32_t opcode = params->opcode;
	const unsigned is_intranode = params->is_intranode;
	const enum ofi_reliability_kind reliability = params->reliability;
	/* use the slid from the lrh header of the incoming packet
	 * as the dlid for the lrh header of the outgoing packet */
	const uint64_t lrh_dlid = params->lrh_dlid;
	const uint64_t bth_rx = ((uint64_t)u8_rx) << 56;
	assert ((opx_ep->tx->pio_max_eager_tx_bytes & 0x3fu) == 0);
	unsigned i;
	const void* sbuf_start = (opx_mr == NULL) ? 0 : opx_mr->buf;
	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
	const uint64_t max_credits = .66 * pio_state.credits_total; // 66% (33% threshold) look up driver threshold
	const uint64_t eager = MIN(max_credits << 6,opx_ep->tx->pio_max_eager_tx_bytes);

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND DPUT, %s opcode %d -- (begin)\n", is_intranode ? "SHM" : "HFI", opcode);

	/* Note that lrh_dlid is just the version of params->slid shifted so
	   that it can be OR'd into the correct position in the packet header */
	assert(params->slid == (lrh_dlid >> 16));

	for (i=params->cur_iov; i<niov; ++i) {
		uint8_t * sbuf = (uint8_t*)((uintptr_t)sbuf_start + (uintptr_t)dput_iov[i].sbuf + params->bytes_sent);
		uintptr_t rbuf = dput_iov[i].rbuf + params->bytes_sent;

		uint64_t bytes_to_send = dput_iov[i].bytes - params->bytes_sent;
		while (bytes_to_send > 0) {
			uint64_t payload_bytes = 0;
			/* compile-time constant expression */
			if (is_intranode) {
				payload_bytes = (bytes_to_send < FI_OPX_SHM_PACKET_SIZE) ? bytes_to_send : FI_OPX_SHM_PACKET_SIZE;
				payload_bytes = MIN(FI_OPX_HFI1_PACKET_MTU, payload_bytes); // avoid assert on receiver
				uint64_t pos;
				union fi_opx_hfi1_packet_hdr * tx_hdr =
					opx_shm_tx_next(&opx_ep->tx->shm, u8_rx, &pos);

				if(!tx_hdr) return -FI_EAGAIN;

				if (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT) {  // RMA-type put
					assert(payload_bytes <= opx_ep->tx->pio_max_eager_tx_bytes);

					const size_t   xfer_bytes_tail   = payload_bytes & 0x07ul;
					const uint64_t payload_qws_tail  = (payload_bytes >> 3) &0x7ul;
					//Note: full_block_credits_needed does not include 1 credit for the packet header
					uint16_t full_block_credits_needed =  (payload_bytes >> 6) + (payload_qws_tail || xfer_bytes_tail);

					const uint64_t pbc_dws = 2 + /* pbc */
								2 + /* lrh */
								3 + /* bth */
								9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
								(full_block_credits_needed << 4);

					const uint16_t lrh_dws = htons(pbc_dws - 1);

					tx_hdr->qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32); //Might not need to set lrh_dws
					tx_hdr->qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
					tx_hdr->qw[2] = opx_ep->rx->tx.dput.hdr.qw[2];
					tx_hdr->qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
					tx_hdr->qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (opcode) | (dt64 << 32) | (op64 << 40) | (payload_bytes << 48);
					tx_hdr->qw[5] = rbuf;
					tx_hdr->qw[6] = key;

				} else {
					tx_hdr->qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid;//  | ((uint64_t)lrh_dws << 32);
					tx_hdr->qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
					tx_hdr->qw[2] = opx_ep->rx->tx.dput.hdr.qw[2];
					tx_hdr->qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
					tx_hdr->qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (opcode) | (payload_bytes << 32);
					tx_hdr->qw[5] = rbuf;
					tx_hdr->qw[6] = target_byte_counter_vaddr;
				}

				union fi_opx_hfi1_packet_payload * const tx_payload =
					(union fi_opx_hfi1_packet_payload *)(tx_hdr+1);

				memcpy((void *)tx_payload->byte,
					(const void *)sbuf,
					payload_bytes);

				opx_shm_tx_advance(&opx_ep->tx->shm, (void*)tx_hdr, pos);

			} else {
				pio_state = *opx_ep->tx->pio_state;
				payload_bytes = (bytes_to_send < eager) ? bytes_to_send : eager;
				uint32_t payload_credits = (payload_bytes >> 6) + ((payload_bytes & 0x3f) ? 1:0);
				uint32_t total_credits_available = pio_state.credits_total - fi_opx_credits_in_use(&pio_state);

				if (total_credits_available <  payload_credits) {
					fi_opx_compiler_msync_writes();
					FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
					total_credits_available = pio_state.credits_total - fi_opx_credits_in_use(&pio_state);
					if (total_credits_available <  payload_credits) {
						opx_ep->tx->pio_state->qw0 = pio_state.qw0;
						return -FI_EAGAIN;
					}
				}

				const uint64_t pbc_dws = 2 + /* pbc */
					2 + /* lrh */
					3 + /* bth */
					9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
					((payload_bytes >> 2) + ((payload_bytes & 0x3) ? 1:0));
				assert(((int64_t)payload_bytes) >= 0);

				const uint16_t lrh_dws = htons(pbc_dws-1);
				struct fi_opx_reliability_tx_replay *replay = NULL;
				if (reliability != OFI_RELIABILITY_KIND_NONE) {
					replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state,
					    false);

					if(replay == NULL) {
						return -FI_EAGAIN;
					}
				}

				union fi_opx_reliability_tx_psn *psn_ptr = NULL;
				const int64_t psn = (reliability != OFI_RELIABILITY_KIND_NONE) ?	/* compile-time constant expression */
					fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
									&opx_ep->reliability->state,
									params->slid,
									u8_rx,
									params->origin_rs,
									&psn_ptr) :
					0;
				if(OFI_UNLIKELY(psn == -1)) {
					fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
					return -FI_EAGAIN;
				}

				assert(payload_bytes <= FI_OPX_HFI1_PACKET_MTU);
				// The "memcopy first" code is here as an alternative to the more complicated
				// direct write to pio followed by memory copy of the reliability buffer
				// Useful for debugging and performance comparisons.  Some platforms this
				// may actually perform better, using system optimized memory copy for reliability
				// copy and an optimized replay inject to kick the packet off
				// This also requires the reliability service so it's only suitable for onload only builds

				if (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT) {  // RMA-type put
					replay->scb.qw0       = opx_ep->rx->tx.dput.qw0 | pbc_dws;
					replay->scb.hdr.qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
					replay->scb.hdr.qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
					replay->scb.hdr.qw[2] = opx_ep->rx->tx.dput.hdr.qw[2] | psn;
					replay->scb.hdr.qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
					replay->scb.hdr.qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (opcode) | (dt64 << 32) | (op64 << 40) | (payload_bytes << 48);
					replay->scb.hdr.qw[5] = rbuf;
					replay->scb.hdr.qw[6] = key;

					//uint8_t *replay_payload = (uint8_t *)replay->payload;
					//memcpy((void *)replay_payload, (const void *)sbuf, payload_bytes);

					struct iovec iov = {sbuf, payload_bytes};
					ssize_t remain = payload_bytes, iov_idx = 0, iov_base_offset = 0;
					uint64_t *payload = replay->payload;
					while (false ==
						fi_opx_hfi1_fill_from_iov8(
							&iov, /* In:  iovec array */
							1, /* In:  total iovecs */
							payload, /* In:  target buffer to fill */
							&remain, /* In/Out:  buffer length to fill */
							&iov_idx, /* In/Out:  start index, returns end */
							&iov_base_offset)) { /* In/Out:  start offset, returns offset */
						// copy until done;
					}
					fi_opx_reliability_client_replay_register_with_update(
						&opx_ep->reliability->state, params->slid,
						params->origin_rs, u8_rx, psn_ptr, replay, cc, payload_bytes, reliability);

					fi_opx_reliability_service_do_replay(&opx_ep->reliability->service,	replay);
					//fi_opx_compiler_msync_writes();

				} else {
					replay->scb.qw0 = opx_ep->rx->tx.dput.qw0 | pbc_dws;
					replay->scb.hdr.qw[0] = opx_ep->rx->tx.dput.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
					replay->scb.hdr.qw[1] = opx_ep->rx->tx.dput.hdr.qw[1] | bth_rx;
					replay->scb.hdr.qw[2] = opx_ep->rx->tx.dput.hdr.qw[2] | psn;
					replay->scb.hdr.qw[3] = opx_ep->rx->tx.dput.hdr.qw[3];
					replay->scb.hdr.qw[4] = opx_ep->rx->tx.dput.hdr.qw[4] | (opcode) | (payload_bytes << 32);
					replay->scb.hdr.qw[5] = rbuf;
					replay->scb.hdr.qw[6] = target_byte_counter_vaddr;

					uint8_t *replay_payload = (uint8_t *)replay->payload;
					memcpy((void *)replay_payload, (const void *)sbuf, payload_bytes);
					fi_opx_reliability_service_do_replay(&opx_ep->reliability->service,
										replay);
					fi_opx_compiler_msync_writes();

					fi_opx_reliability_client_replay_register_no_update(
						&opx_ep->reliability->state, params->slid,
						params->origin_rs, u8_rx, psn_ptr, replay, reliability);
				}
			} /* if !is_intranode */

			rbuf += payload_bytes;
			sbuf += payload_bytes;
			bytes_to_send -= payload_bytes;
			params->bytes_sent += payload_bytes;

			if(origin_byte_counter) {
				*origin_byte_counter -= payload_bytes;
				assert(((int64_t)*origin_byte_counter) >= 0);
			}
		} /* while bytes_to_send */

		if (opcode == FI_OPX_HFI_DPUT_OPCODE_PUT && is_intranode) {  // RMA-type put, so send a ping/fence to better latency
			fi_opx_shm_write_fence(opx_ep, u8_rx, lrh_dlid, cc, params->bytes_sent);
		}
		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND DPUT, %s finished IOV=%d bytes_sent=%ld -- (end)\n", 
			is_intranode ? "SHM" : "HFI", params->cur_iov, params->bytes_sent);

		params->bytes_sent = 0;
		params->cur_iov++;
	} /* for niov */

	return FI_SUCCESS;
}

union fi_opx_hfi1_deferred_work* fi_opx_hfi1_rx_rzv_cts (struct fi_opx_ep * opx_ep,
							 struct fi_opx_mr * opx_mr,
							 const void * const hdr, const void * const payload,
							 size_t payload_bytes_to_copy,
							 const uint8_t u8_rx, const uint32_t niov,
							 const struct fi_opx_hfi1_dput_iov * const dput_iov,
							 const uintptr_t target_byte_counter_vaddr,
							 uint64_t * origin_byte_counter,
							 uint32_t opcode,
							 void (*completion_action)(union fi_opx_hfi1_deferred_work * work_state),
							 const unsigned is_intranode,
							 const enum ofi_reliability_kind reliability) {
	const union fi_opx_hfi1_packet_hdr * const hfi1_hdr =
		(const union fi_opx_hfi1_packet_hdr * const) hdr;
	fi_opx_shm_dynamic_tx_connect(is_intranode, opx_ep, u8_rx);

	union fi_opx_hfi1_deferred_work *work = ofi_buf_alloc(opx_ep->tx->work_pending_pool);
	struct fi_opx_hfi1_dput_params *params = &work->dput;

	params->work_elem.slist_entry.next = NULL;
	params->work_elem.work_fn = fi_opx_hfi1_do_dput;
	params->work_elem.completion_action = completion_action;
	params->work_elem.payload_copy = NULL;
	params->opx_ep = opx_ep;
	params->opx_mr = opx_mr;
	params->lrh_dlid = (hfi1_hdr->stl.lrh.qw[0] & 0xFFFF000000000000ul) >> 32;
	params->slid = hfi1_hdr->stl.lrh.slid;
	params->origin_rs = hfi1_hdr->cts.target.vaddr.origin_rs;
	params->u8_rx = u8_rx;
	params->niov = niov;
	params->dput_iov = &params->iov[0];
	params->cur_iov = 0;
	params->bytes_sent = 0;
	params->cc = NULL;

	params->target_byte_counter_vaddr = target_byte_counter_vaddr;
	params->origin_byte_counter = origin_byte_counter;
	params->opcode = opcode;
	params->is_intranode = is_intranode;
	params->reliability = reliability;

	for(int idx=0; idx < niov; idx++) {
		params->iov[idx] = dput_iov[idx];
	}

	int rc = fi_opx_hfi1_do_dput(work);
	if(rc == FI_SUCCESS) {
		ofi_buf_free(work);
		return NULL;
	}
	assert(rc == -FI_EAGAIN);

	/* Try again later*/
	if(payload_bytes_to_copy) {
		params->work_elem.payload_copy = ofi_buf_alloc(opx_ep->tx->rma_payload_pool);
		memcpy(params->work_elem.payload_copy, payload, payload_bytes_to_copy);
	}
	assert(work->work_elem.slist_entry.next == NULL);
	slist_insert_tail(&work->work_elem.slist_entry, &opx_ep->tx->work_pending);
	return work;
}



static inline size_t fi_opx_iov_to_payload_blocks(size_t niov) {
	size_t sz_bytes = niov * sizeof(struct iovec);
	sz_bytes += (sizeof(uintptr_t) + // origin_byte_counter_vaddr
				 sizeof(size_t));    // unused field
	/* (bytes * 64) + ((bytes % 64) ? 1 : 0) */
	return (sz_bytes >> 6) + ((sz_bytes & 0x3f) ? 1 : 0);
}

uint64_t num_sends;
uint64_t total_sendv_bytes;
ssize_t fi_opx_hfi1_tx_sendv_rzv(struct fid_ep *ep, const struct iovec *iov, size_t niov,
				 size_t total_len, void *desc, fi_addr_t dest_addr, uint64_t tag,
				 void *context, const uint32_t data, int lock_required,
				 const unsigned override_flags, uint64_t tx_op_flags,
				 const uint64_t dest_rx, const uintptr_t origin_byte_counter_vaddr,
				 uint64_t *origin_byte_counter_value, const uint64_t caps,
				 const enum ofi_reliability_kind reliability)
{
	// We should already have grabbed the lock prior to calling this function
	assert(!lock_required);

	struct fi_opx_ep *opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr = { .fi = dest_addr };
	const uint64_t payload_blocks_total = fi_opx_iov_to_payload_blocks(niov);
	const uint64_t bth_rx = ((uint64_t)dest_rx) << 56;
	const uint64_t lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID(addr.fi);
	assert(niov <= FI_OPX_MAX_DPUT_IOV);
	*origin_byte_counter_value = total_len;

	/* This is a hack to trick an MPICH test to make some progress    */
	/* As it erroneously overflows the send buffers by never checking */
	/* for multi-receive overflows properly in some onesided tests    */
	/* There are almost certainly better ways to do this */
	if((tx_op_flags & FI_MSG) && (total_sendv_bytes+=total_len > opx_ep->rx->min_multi_recv)) {
		total_sendv_bytes = 0;
		return -FI_EAGAIN;
	}

	const uint64_t pbc_dws = 2 + /* pbc */
				 2 + /* lhr */
				 3 + /* bth */
				 9 + /* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
				 (payload_blocks_total << 4);

	const uint16_t lrh_dws = htons(pbc_dws - 1);

	if (((caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == FI_LOCAL_COMM) || /* compile-time constant expression */
	    (((caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == (FI_LOCAL_COMM | FI_REMOTE_COMM)) &&
	     (opx_ep->tx->send.hdr.stl.lrh.slid == addr.uid.lid))) {
		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, SHM -- RENDEZVOUS RTS Noncontig (begin)\n");

		uint64_t pos;
		union fi_opx_hfi1_packet_hdr *const hdr = opx_shm_tx_next(
			&opx_ep->tx->shm, dest_rx, &pos);

		if (!hdr) return -FI_EAGAIN;

		hdr->qw[0] = opx_ep->tx->rzv.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
		hdr->qw[1] = opx_ep->tx->rzv.hdr.qw[1] | bth_rx |
			     ((caps & FI_MSG) ? (uint64_t)FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS :
						(uint64_t)FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS);

		hdr->qw[2] = opx_ep->tx->rzv.hdr.qw[2];
		hdr->qw[3] = opx_ep->tx->rzv.hdr.qw[3] | (((uint64_t)data) << 32);
		hdr->qw[4] = opx_ep->tx->rzv.hdr.qw[4] | (niov << 48);
		hdr->qw[5] = total_len;
		hdr->qw[6] = tag;
		union fi_opx_hfi1_packet_payload *const payload =
			(union fi_opx_hfi1_packet_payload *)(hdr + 1);

		payload->rendezvous.noncontiguous.origin_byte_counter_vaddr = origin_byte_counter_vaddr;
		payload->rendezvous.noncontiguous.unused = 0;
		ssize_t idx;
		for(idx = 0; idx < niov; idx++) {
			payload->rendezvous.noncontiguous.iov[idx] = iov[idx];
		}

		opx_shm_tx_advance(&opx_ep->tx->shm, (void *)hdr, pos);

		FI_DBG_TRACE(
			fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, SHM -- RENDEZVOUS RTS (end)\n");
		fi_opx_shm_poll_many(&opx_ep->ep_fid, 0);
		return FI_SUCCESS;
	}
	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, HFI -- RENDEZVOUS RTS (begin)\n");

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;
	const uint16_t total_credits_needed = 1 +   /* packet header */
					      payload_blocks_total; /* packet payload */

	uint64_t total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	unsigned loop = 0;
	while (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		if (loop++ > FI_OPX_HFI1_TX_SEND_RZV_CREDIT_MAX_WAIT) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			return -FI_EAGAIN;
		}
		fi_opx_compiler_msync_writes();
		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	}
	if (OFI_UNLIKELY(loop)) {
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;
	}

	struct fi_opx_reliability_tx_replay *replay = (reliability != OFI_RELIABILITY_KIND_NONE) ?
		fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state, false) : NULL;
	if (replay == NULL) {
		return -FI_EAGAIN;
	}


	union fi_opx_reliability_tx_psn *psn_ptr = NULL;
	const int64_t psn =
		(reliability != OFI_RELIABILITY_KIND_NONE) ? /* compile-time constant expression */
			fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid,
							&opx_ep->reliability->state,
							addr.uid.lid,
							dest_rx,
							addr.reliability_rx,
							&psn_ptr) :
			0;
	if(OFI_UNLIKELY(psn == -1)) {
		fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
		return -FI_EAGAIN;
	}

	volatile uint64_t * const scb =
		FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);
	uint64_t tmp[8];
	assert(opx_ep->tx->rzv.qw0 == 0);
	tmp[0] = opx_ep->tx->rzv.qw0 | pbc_dws | ((opx_ep->tx->force_credit_return & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT);
	tmp[1] = opx_ep->tx->rzv.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);
	tmp[2] = opx_ep->tx->rzv.hdr.qw[1] | bth_rx |
		((caps & FI_MSG) ? (uint64_t)FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS :
		 (uint64_t)FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS);
	tmp[3] = opx_ep->tx->rzv.hdr.qw[2] | psn;
	tmp[4] = opx_ep->tx->rzv.hdr.qw[3] | (((uint64_t)data) << 32);
	tmp[5] = opx_ep->tx->rzv.hdr.qw[4] | (niov << 48);
	tmp[6] = total_len;
	tmp[7] = tag;

	scb[0] = tmp[0];
	scb[1] = tmp[1];
	scb[2] = tmp[2];
	scb[3] = tmp[3];
	scb[4] = tmp[4];
	scb[5] = tmp[5];
	scb[6] = tmp[6];
	scb[7] = tmp[7];

	/* consume one credit for the packet header */
	--total_credits_available;
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	unsigned credits_consumed = 1;
#endif

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	/* write the payload */
	const ssize_t total_payload_bytes = niov*sizeof(*iov) /* iovec array */
		             + 16; /* byte counter and unused fields */
	const size_t payload_qws_total = total_payload_bytes >> 3;
	const size_t payload_qws_tail = payload_qws_total & 0x07ul;
	ssize_t iov_idx = 0, iov_base_offset = 0;
	uint64_t tmp_value = 0;
	struct iovec src_iov[3] = {{ (void*)&origin_byte_counter_vaddr, 8 },
							   { &tmp_value, 8 },
							   { (void*)&iov[0], niov*sizeof(*iov)}
				};
	const uint16_t contiguous_credits_until_wrap =
		(uint16_t)(pio_state.credits_total - pio_state.scb_head_index);

	const uint16_t contiguous_credits_available =
		MIN(total_credits_available, contiguous_credits_until_wrap);

	uint16_t full_block_credits_needed = (uint16_t)(payload_qws_total >> 3);
	if(full_block_credits_needed > 0) {
		volatile uint64_t * scb_payload = (uint64_t *)FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		const uint16_t contiguous_full_blocks_to_write =
			MIN(full_block_credits_needed, contiguous_credits_available);
		int64_t remain = MIN(total_payload_bytes, contiguous_full_blocks_to_write << 6);
		while (false ==
		       fi_opx_hfi1_fill_from_iov8(
			       src_iov,             /* In:  iovec array                      */
			       3,                   /* In:  total iovecs                     */
			       scb_payload,         /* In:  target buffer to fill            */
			       &remain,             /* In/Out:  buffer length to fill        */
			       &iov_idx,            /* In/Out:  start index, returns end     */
			       &iov_base_offset)) { /* In/Out:  start offset, returns offset */
			// copy until done;
		}
		assert(remain == 0);
		full_block_credits_needed -= contiguous_full_blocks_to_write;
		FI_OPX_HFI1_CONSUME_CREDITS(pio_state, contiguous_full_blocks_to_write);
#ifndef NDEBUG
		credits_consumed += contiguous_full_blocks_to_write;
#endif
	}
	if (OFI_UNLIKELY(full_block_credits_needed > 0)) {
		/*
		 * handle wrap condition
		 */
		volatile uint64_t *scb_payload =
			FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

		int64_t remain = (full_block_credits_needed << 6);
		while (false ==
		       fi_opx_hfi1_fill_from_iov8(
			       src_iov,     /* In:  iovec array */
			       3,                   /* In:  total iovecs */
			       scb_payload,         /* In:  target buffer to fill */
			       &remain,             /* In/Out:  buffer length to fill */
			       &iov_idx,            /* In/Out:  start index, returns end */
			       &iov_base_offset)) { /* In/Out:  start offset, returns offset */
			// copy until done;
		}
		assert(remain == 0);
		FI_OPX_HFI1_CONSUME_CREDITS(pio_state, full_block_credits_needed);
#ifndef NDEBUG
		credits_consumed += full_block_credits_needed;
#endif
	}

	if (payload_qws_tail > 0) {
		volatile uint64_t *scb_payload =
			FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		assert(payload_qws_tail < 8);
		int64_t remain = (payload_qws_tail << 3);
		assert(remain < 64);
		while (false ==
		       fi_opx_hfi1_fill_from_iov8(
			       src_iov, /* In:  iovec array */
			       3, /* In:  total iovecs */
			       scb_payload, /* In:  target buffer to fill */
			       &remain, /* In/Out:  buffer length to fill */
			       &iov_idx, /* In/Out:  start index, returns end */
			       &iov_base_offset)) { /* In/Out:  start offset, returns offset */
			// copy until done;
		}
		for (int i = payload_qws_tail; i < 8; ++i) {
			scb_payload[i] = 0;
		}
		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
		++credits_consumed;
#endif
	}

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
#ifndef NDEBUG
	assert(credits_consumed == total_credits_needed);
#endif

	if (reliability != OFI_RELIABILITY_KIND_NONE) { /* compile-time constant expression */
		replay->scb.qw0 = tmp[0];
		replay->scb.hdr.qw[0] = tmp[1];
		replay->scb.hdr.qw[1] = tmp[2];
		replay->scb.hdr.qw[2] = tmp[3];
		replay->scb.hdr.qw[3] = tmp[4];
		replay->scb.hdr.qw[4] = tmp[5];
		replay->scb.hdr.qw[5] = tmp[6];
		replay->scb.hdr.qw[6] = tmp[7];
		iov_idx = 0;
		iov_base_offset = 0;
		uint64_t *payload = replay->payload;
		int64_t remain = total_payload_bytes;
		while (false ==
		       fi_opx_hfi1_fill_from_iov8(
			       src_iov, /* In:  iovec array */
			       3, /* In:  total iovecs */
			       payload, /* In:  target buffer to fill */
			       &remain, /* In/Out:  buffer length to fill */
			       &iov_idx, /* In/Out:  start index, returns end */
			       &iov_base_offset)) { /* In/Out:  start offset, returns offset */
			// copy until done;
		}
		assert(remain == 0);
		fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state,
								    addr.uid.lid,
								    addr.reliability_rx, dest_rx,
								    psn_ptr, replay, reliability);
	}


	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		     "===================================== SEND, HFI -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}

ssize_t fi_opx_hfi1_tx_send_rzv (struct fid_ep *ep,
		const void *buf, size_t len, void *desc,
		fi_addr_t dest_addr, uint64_t tag, void* context,
		const uint32_t data, int lock_required,
		const unsigned override_flags, uint64_t tx_op_flags,
		const uint64_t dest_rx,
		const uintptr_t origin_byte_counter_vaddr,
		uint64_t *origin_byte_counter_value,
		const uint64_t caps,
		const enum ofi_reliability_kind reliability)
{
	// We should already have grabbed the lock prior to calling this function
	assert(!lock_required);

	//Need at least one full block of payload
	assert(len >= FI_OPX_HFI1_TX_MIN_RZV_PAYLOAD_BYTES);

	struct fi_opx_ep * opx_ep = container_of(ep, struct fi_opx_ep, ep_fid);
	const union fi_opx_addr addr = { .fi = dest_addr };

#ifdef RZV_IMMEDIATE_BLOCK_ENABLED
	const uint64_t max_immediate_block_count = 2; /* alternatively: (FI_OPX_HFI1_PACKET_MTU >> 6)-2) */
	const uint64_t immediate_block_count = MIN((len >> 6), max_immediate_block_count);
#else
	const uint64_t immediate_block_count = 0;
#endif
	const uint64_t payload_blocks_total =
		1 +				/* rzv metadata */
		1 +				/* immediate data tail */
		immediate_block_count;


	const uint64_t bth_rx = ((uint64_t)dest_rx) << 56;
	const uint64_t lrh_dlid = FI_OPX_ADDR_TO_HFI1_LRH_DLID(dest_addr);

	const uint64_t immediate_byte_count = len & 0x0007ul;
	const uint64_t immediate_qw_count = (len >> 3) & 0x0007ul;
	const uint64_t immediate_total = immediate_byte_count +
		immediate_qw_count * sizeof(uint64_t) +
		immediate_block_count * sizeof(union cacheline);

	assert(((len - immediate_total) & 0x003Fu) == 0);

	*origin_byte_counter_value = len - immediate_total;

	const uint64_t pbc_dws =
		2 +			/* pbc */
		2 +			/* lhr */
		3 +			/* bth */
		9 +			/* kdeth; from "RcvHdrSize[i].HdrSize" CSR */
		(payload_blocks_total << 4);

	const uint16_t lrh_dws = htons(pbc_dws-1);

	if (((caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == FI_LOCAL_COMM) ||	/* compile-time constant expression */
		(((caps & (FI_LOCAL_COMM | FI_REMOTE_COMM)) == (FI_LOCAL_COMM | FI_REMOTE_COMM)) &&
			(opx_ep->tx->send.hdr.stl.lrh.slid == addr.uid.lid))) {

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, SHM -- RENDEZVOUS RTS (begin)\n");
		uint64_t pos;
		union fi_opx_hfi1_packet_hdr * const hdr =
			opx_shm_tx_next(&opx_ep->tx->shm, dest_rx, &pos);

		if (!hdr) return -FI_EAGAIN;

		hdr->qw[0] = opx_ep->tx->rzv.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32);

		hdr->qw[1] = opx_ep->tx->rzv.hdr.qw[1] | bth_rx |
			((caps & FI_MSG) ?
				(uint64_t)FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS :
				(uint64_t)FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS);

		hdr->qw[2] = opx_ep->tx->rzv.hdr.qw[2];
		hdr->qw[3] = opx_ep->tx->rzv.hdr.qw[3] | (((uint64_t)data) << 32);
		hdr->qw[4] = opx_ep->tx->rzv.hdr.qw[4] | (1ull << 48); /* effectively 1 iov */
		hdr->qw[5] = len;
		hdr->qw[6] = tag;


		union fi_opx_hfi1_packet_payload * const payload =
			(union fi_opx_hfi1_packet_payload *)(hdr+1);

		payload->rendezvous.contiguous.src_vaddr = (uintptr_t)buf + immediate_total;
		payload->rendezvous.contiguous.src_blocks = (len - immediate_total) >> 6;
		payload->rendezvous.contiguous.immediate_byte_count = immediate_byte_count;
		payload->rendezvous.contiguous.immediate_qw_count = immediate_qw_count;
		payload->rendezvous.contiguous.immediate_block_count = immediate_block_count;
		payload->rendezvous.contiguous.origin_byte_counter_vaddr = origin_byte_counter_vaddr;
		payload->rendezvous.contiguous.unused[0] = 0;
		payload->rendezvous.contiguous.unused[1] = 0;


		uint8_t *sbuf = (uint8_t *)buf;

		if (immediate_byte_count > 0) {
			memcpy((void*)&payload->rendezvous.contiguous.immediate_byte, (const void*)sbuf, immediate_byte_count);
			sbuf += immediate_byte_count;
		}

		uint64_t * sbuf_qw = (uint64_t *)sbuf;
		unsigned i=0;
		for (i=0; i<immediate_qw_count; ++i) {
			payload->rendezvous.contiguous.immediate_qw[i] = sbuf_qw[i];
		}

#ifdef RZV_IMMEDIATE_BLOCK_ENABLED
		sbuf_qw += immediate_qw_count;

		memcpy((void*)payload->rendezvous.contiguous.immediate_block,
			(const void *)sbuf_qw, immediate_block_count * 64);
#endif

		opx_shm_tx_advance(&opx_ep->tx->shm, (void*)hdr, pos);

		FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
			"===================================== SEND, SHM -- RENDEZVOUS RTS (end)\n");

		return FI_SUCCESS;
	}

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- RENDEZVOUS RTS (begin)\n");

	/*
	 * For now this is implemented as PIO-only protocol, no SDMA
	 * engines are used and no TIDs are allocated for expected
	 * receives.
	 *
	 * This will have lower performance because software on the
	 * initiator must copy the data into the injection buffer,
	 * rather than the hardware via SDMA engines, and the
	 * target must copy the data into the receive buffer, rather
	 * than the hardware.
	 */

	union fi_opx_hfi1_pio_state pio_state = *opx_ep->tx->pio_state;

	const uint16_t total_credits_needed =
		1 +				/* packet header */
		payload_blocks_total;		/* packet payload */

	uint64_t total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	unsigned loop = 0;
	while (OFI_UNLIKELY(total_credits_available < total_credits_needed)) {
		/*
		 * TODO: Implement PAUSE time-out functionality using time-out configuration
		 * parameter(s).
		 */
		if (loop++ > FI_OPX_HFI1_TX_SEND_RZV_CREDIT_MAX_WAIT) {
			opx_ep->tx->pio_state->qw0 = pio_state.qw0;
			return -FI_EAGAIN;
		}
		fi_opx_compiler_msync_writes();

		FI_OPX_HFI1_UPDATE_CREDITS(pio_state, opx_ep->tx->pio_credits_addr);
		total_credits_available = FI_OPX_HFI1_AVAILABLE_CREDITS(pio_state, &opx_ep->tx->force_credit_return, total_credits_needed);
	}
	if (OFI_UNLIKELY(loop)) {
		opx_ep->tx->pio_state->qw0 = pio_state.qw0;
	}

	struct fi_opx_reliability_tx_replay * replay = NULL;
	if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
		replay = fi_opx_reliability_client_replay_allocate(&opx_ep->reliability->state, false);
		if(replay == NULL) {
			return -FI_EAGAIN;
		}
	}

	union fi_opx_reliability_tx_psn *psn_ptr = NULL;
	const int64_t psn = (reliability != OFI_RELIABILITY_KIND_NONE) ?	/* compile-time constant expression */
		fi_opx_reliability_tx_next_psn(&opx_ep->ep_fid, &opx_ep->reliability->state,
						addr.uid.lid, dest_rx, addr.reliability_rx, &psn_ptr) :
		0;
	if(OFI_UNLIKELY(psn == -1)) {
		fi_opx_reliability_client_replay_deallocate(&opx_ep->reliability->state, replay);
		return -FI_EAGAIN;
	}

	/*
	 * Write the 'start of packet' (hw+sw header) 'send control block'
	 * which will consume a single pio credit.
	 */

	volatile uint64_t * const scb =
		FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_sop_first, pio_state);

	uint64_t tmp[8];

	fi_opx_set_scb(scb, tmp,
				   opx_ep->tx->rzv.qw0 | pbc_dws | ((opx_ep->tx->force_credit_return & FI_OPX_HFI1_PBC_CR_MASK) << FI_OPX_HFI1_PBC_CR_SHIFT),
				   opx_ep->tx->rzv.hdr.qw[0] | lrh_dlid | ((uint64_t)lrh_dws << 32),

				   opx_ep->tx->rzv.hdr.qw[1] | bth_rx |
				   ((caps & FI_MSG) ?
					(uint64_t)FI_OPX_HFI_BTH_OPCODE_MSG_RZV_RTS :
					(uint64_t)FI_OPX_HFI_BTH_OPCODE_TAG_RZV_RTS),

				   opx_ep->tx->rzv.hdr.qw[2] | psn,
				   opx_ep->tx->rzv.hdr.qw[3] | (((uint64_t)data) << 32),
				   opx_ep->tx->rzv.hdr.qw[4] | (1ull << 48),
				   len, tag);

	/* consume one credit for the packet header */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	unsigned credits_consumed = 1;
#endif

	FI_OPX_HFI1_CLEAR_CREDIT_RETURN(opx_ep);

	if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
		replay->scb.qw0 = tmp[0];
		replay->scb.hdr.qw[0] = tmp[1];
		replay->scb.hdr.qw[1] = tmp[2];
		replay->scb.hdr.qw[2] = tmp[3];
		replay->scb.hdr.qw[3] = tmp[4];
		replay->scb.hdr.qw[4] = tmp[5];
		replay->scb.hdr.qw[5] = tmp[6];
		replay->scb.hdr.qw[6] = tmp[7];
	}

	/*
	 * write the rendezvous payload "send control blocks"
	 */

	volatile uint64_t * scb_payload = (uint64_t *)FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

	fi_opx_set_scb(scb_payload, tmp,
				   (uintptr_t)buf + immediate_total,	/* src_vaddr */
				   (len - immediate_total) >> 6,		/* src_blocks */
				   immediate_byte_count,
				   immediate_qw_count,
				   immediate_block_count,
				   origin_byte_counter_vaddr,
				   0, /* unused */
				   0 /* unused */);

	/* consume one credit for the rendezvous payload metadata */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

	uint64_t * replay_payload =
		(reliability != OFI_RELIABILITY_KIND_NONE) ?	/* compile-time constant expression */
		replay->payload : NULL;

	if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
		fi_opx_copy_scb(replay_payload, tmp);
		replay_payload += 8;
	}

	scb_payload = (uint64_t *)FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);

	/* immediate_byte and immediate_qw are "packed" in the current implementation             */
	/* meaning the immediate bytes are filled, then followed by the rest of the data directly */
	/* adjacent to the packed bytes.  It's probably more efficient to leave a pad and not go  */
	/* through the confusion of finding these boundaries on both sides of the rendezvous      */
	/* That is, just pack the immediate bytes, then pack the "rest" in the immediate qws      */
	/* This would lead to more efficient packing on both sides at the expense of              */
	/* wasting space of a common 0 byte immediate                                             */
	/* tmp_payload_t represents the second cache line of the rts packet                       */
	/* fi_opx_hfi1_packet_payload -> rendezvous -> contiguous                               */
	struct tmp_payload_t {
		uint8_t		immediate_byte[8];
		uint64_t	immediate_qw[7];
	} __attribute__((packed));

	struct tmp_payload_t *tmp_payload = (void*)tmp;
	uint8_t *sbuf = (uint8_t *)buf;
	if (immediate_byte_count > 0) {
		memcpy((void*)tmp_payload->immediate_byte, (const void*)sbuf, immediate_byte_count);
		sbuf += immediate_byte_count;
	}

	uint64_t * sbuf_qw = (uint64_t *)sbuf;
	int i=0;
	for (i=0; i<immediate_qw_count; ++i) {
		tmp_payload->immediate_qw[i] = sbuf_qw[i];
	}
	fi_opx_copy_scb(scb_payload, tmp);
	scb_payload += 1;
	sbuf_qw += immediate_qw_count;

	if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
		fi_opx_copy_scb(replay_payload, tmp);
		replay_payload += 8;
	}

	/* consume one credit for the rendezvous payload immediate data */
	FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

#ifdef RZV_IMMEDIATE_BLOCK_ENABLED
	switch (immediate_block_count) {

	case 2:
		scb_payload = (uint64_t *)FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->pio_scb_first, pio_state);
		fi_opx_copy_scb(scb_payload, sbuf_qw);
		scb_payload += 8;

		if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
			fi_opx_copy_scb(replay_payload, sbuf_qw);
			replay_payload += 8;
		}

		sbuf_qw += 8;

		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

		/* break; is purposefully omitted */

	case 1:
		scb_payload = (uint64_t *)FI_OPX_HFI1_PIO_SCB_HEAD(opx_ep->tx->>pio_scb_first, pio_state);
		fi_opx_copy_scb(scb_payload, sbuf_qw);
		scb_payload += 8;

		FI_OPX_HFI1_CONSUME_SINGLE_CREDIT(pio_state);
#ifndef NDEBUG
	++credits_consumed;
#endif

		if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
			fi_opx_copy_scb(replay_payload, sbuf_qw);
			replay_payload += 8;
		}

		sbuf_qw += 8;

		break;

	default:
		break;

	}
#endif /* RZV_IMMEDIATE_BLOCK_ENABLED */

	if (reliability != OFI_RELIABILITY_KIND_NONE) {	/* compile-time constant expression */
		fi_opx_reliability_client_replay_register_no_update(&opx_ep->reliability->state,
															addr.uid.lid, addr.reliability_rx, dest_rx, psn_ptr, replay,
															reliability);
	}

	FI_OPX_HFI1_CHECK_CREDITS_FOR_ERROR(opx_ep->tx->pio_credits_addr);
#ifndef NDEBUG
	assert(credits_consumed == total_credits_needed);
#endif

	/* update the hfi txe state */
	opx_ep->tx->pio_state->qw0 = pio_state.qw0;

	FI_DBG_TRACE(fi_opx_global.prov, FI_LOG_EP_DATA,
		"===================================== SEND, HFI -- RENDEZVOUS RTS (end)\n");

	return FI_SUCCESS;
}


unsigned fi_opx_hfi1_handle_poll_error(struct fi_opx_ep * opx_ep,
									   volatile uint32_t * rhf_ptr,
									   const uint32_t rhf_msb,
									   const uint32_t rhf_lsb,
									   const uint32_t rhf_seq,
									   const uint64_t hdrq_offset,
									   const uint32_t hdrq_offset_notifyhw)
{
#define HFI1_RHF_ICRCERR (0x80000000u)
#define HFI1_RHF_ECCERR (0x20000000u)
#define HFI1_RHF_TIDERR (0x08000000u)
#define HFI1_RHF_DCERR (0x00800000u)
#define HFI1_RHF_DCUNCERR (0x00400000u)
	(void)rhf_ptr;  /* unused unless debug is turned on */
	if ((rhf_msb & (HFI1_RHF_ICRCERR | HFI1_RHF_ECCERR | HFI1_RHF_TIDERR | HFI1_RHF_DCERR |
			HFI1_RHF_DCUNCERR)) != 0) {
		/* drop this packet and allow reliability protocol to retry */
		if (rhf_seq == (rhf_lsb & 0xF0000000u)) {
#ifdef OPX_RELIABILITY_DEBUG
			const uint64_t hdrq_offset_dws = (rhf_msb >> 12) & 0x01FFu;

			uint32_t *pkt = (uint32_t *)rhf_ptr -
					32 + /* header queue entry size in dw */
					2 + /* rhf field size in dw */
					hdrq_offset_dws;

			const union fi_opx_hfi1_packet_hdr *const hdr =
				(union fi_opx_hfi1_packet_hdr *)pkt;

			fprintf(stderr,
				"%s:%s():%d drop this packet and allow reliability protocol to retry, psn = %u\n",
				__FILE__, __func__, __LINE__, hdr->reliability.psn);
#endif
			if ((rhf_lsb & 0x00008000u) == 0x00008000u) {
				/* "consume" this egrq element */
				const uint32_t egrbfr_index =
					(rhf_lsb >> FI_OPX_HFI1_RHF_EGRBFR_INDEX_SHIFT) &
					FI_OPX_HFI1_RHF_EGRBFR_INDEX_MASK;
				const uint32_t last_egrbfr_index =
					opx_ep->rx->egrq.last_egrbfr_index;
				if (OFI_UNLIKELY(last_egrbfr_index != egrbfr_index)) {
					*opx_ep->rx->egrq.head_register = last_egrbfr_index;
					opx_ep->rx->egrq.last_egrbfr_index = egrbfr_index;
				}
			}

			/* "consume" this hdrq element */
			opx_ep->rx->state.hdrq.rhf_seq = (rhf_seq < 0xD0000000u) * rhf_seq + 0x10000000u;
			opx_ep->rx->state.hdrq.head = hdrq_offset +	32;

			fi_opx_hfi1_update_hdrq_head_register(opx_ep, hdrq_offset, hdrq_offset_notifyhw);

		}
		/*
		 * The "else" case, where rhf_seq != (rhf_lsb & 0xF0000000u) indicates
		 * the WFR is dropping headers. We just ignore this and let
		 * reliability re-send the packet.
		 *
		 * TODO: Can we send a NACK in this case?
		 */
		return 1;
	}

	FI_WARN(&fi_opx_provider, FI_LOG_FABRIC, "RECEIVE ERROR: rhf_msb = 0x%08x, rhf_lsb = 0x%08x, rhf_seq = 0x%1x\n", rhf_msb, rhf_lsb, rhf_seq);

	abort();
}
