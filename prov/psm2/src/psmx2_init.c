/*
 * Copyright (c) 2013-2018 Intel Corporation. All rights reserved.
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

#include "psmx2.h"
#include "ofi_prov.h"
#include <glob.h>
#include <dlfcn.h>

static int psmx2_init_count = 0;
static int psmx2_lib_initialized = 0;
static pthread_mutex_t psmx2_lib_mutex;

struct psmx2_env psmx2_env = {
	.name_server	= 1,
	.tagged_rma	= 1,
	.uuid		= PSMX2_DEFAULT_UUID,
	.delay		= 0,
	.timeout	= 5,
	.prog_interval	= -1,
	.prog_affinity	= NULL,
	.multi_ep	= 0,
	.max_trx_ctxt	= 1,
	.free_trx_ctxt	= 1,
	.num_devunits	= 1,
	.inject_size	= 64,
	.lock_level	= 2,
	.lazy_conn	= 0,
	.disconnect	= 0,
#if (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_RUNTIME)
	.tag_layout	= "auto",
#endif
};

#if (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_RUNTIME)
uint64_t psmx2_tag_mask;
uint32_t psmx2_tag_upper_mask;
uint32_t psmx2_data_mask;
int	 psmx2_flags_idx;
int	 psmx2_tag_layout_locked = 0;
#endif

static void psmx2_init_env(void)
{
	if (getenv("OMPI_COMM_WORLD_RANK") || getenv("PMI_RANK"))
		psmx2_env.name_server = 0;

	fi_param_get_bool(&psmx2_prov, "name_server", &psmx2_env.name_server);
	fi_param_get_bool(&psmx2_prov, "tagged_rma", &psmx2_env.tagged_rma);
	fi_param_get_str(&psmx2_prov, "uuid", &psmx2_env.uuid);
	fi_param_get_int(&psmx2_prov, "delay", &psmx2_env.delay);
	fi_param_get_int(&psmx2_prov, "timeout", &psmx2_env.timeout);
	fi_param_get_int(&psmx2_prov, "prog_interval", &psmx2_env.prog_interval);
	fi_param_get_str(&psmx2_prov, "prog_affinity", &psmx2_env.prog_affinity);
	fi_param_get_int(&psmx2_prov, "inject_size", &psmx2_env.inject_size);
	fi_param_get_bool(&psmx2_prov, "lock_level", &psmx2_env.lock_level);
	fi_param_get_bool(&psmx2_prov, "lazy_conn", &psmx2_env.lazy_conn);
	fi_param_get_bool(&psmx2_prov, "disconnect", &psmx2_env.disconnect);
#if (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_RUNTIME)
	fi_param_get_str(&psmx2_prov, "tag_layout", &psmx2_env.tag_layout);
#endif
}

void psmx2_init_tag_layout(struct fi_info *info)
{
	int use_tag64;

#if (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_RUNTIME)
	use_tag64 = (psmx2_tag_mask == PSMX2_TAG_MASK_64);

	if (psmx2_tag_layout_locked) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"tag layout already set opened domain.\n");
		goto out;
	}

	if (strcasecmp(psmx2_env.tag_layout, "tag60") == 0) {
		psmx2_tag_upper_mask = PSMX2_TAG_UPPER_MASK_60;
		psmx2_tag_mask = PSMX2_TAG_MASK_60;
		psmx2_data_mask = PSMX2_DATA_MASK_60;
		psmx2_flags_idx = PSMX2_FLAGS_IDX_60;
		use_tag64 = 0;
	} else if (strcasecmp(psmx2_env.tag_layout, "tag64") == 0) {
		psmx2_tag_upper_mask = PSMX2_TAG_UPPER_MASK_64;
		psmx2_tag_mask = PSMX2_TAG_MASK_64;
		psmx2_data_mask = PSMX2_DATA_MASK_64;
		psmx2_flags_idx = PSMX2_FLAGS_IDX_64;
		use_tag64 = 1;
	} else {
		if (strcasecmp(psmx2_env.tag_layout, "auto") != 0) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"Invalid tag layout '%s', using 'auto'.\n",
				psmx2_env.tag_layout);
			psmx2_env.tag_layout = "auto";
		}
		if ((info->caps & (FI_TAGGED | FI_MSG)) &&
		    info->domain_attr->cq_data_size) {
			psmx2_tag_upper_mask = PSMX2_TAG_UPPER_MASK_60;
			psmx2_tag_mask = PSMX2_TAG_MASK_60;
			psmx2_data_mask = PSMX2_DATA_MASK_60;
			psmx2_flags_idx = PSMX2_FLAGS_IDX_60;
			use_tag64 = 0;
		} else {
			psmx2_tag_upper_mask = PSMX2_TAG_UPPER_MASK_64;
			psmx2_tag_mask = PSMX2_TAG_MASK_64;
			psmx2_data_mask = PSMX2_DATA_MASK_64;
			psmx2_flags_idx = PSMX2_FLAGS_IDX_64;
			use_tag64 = 1;
		}
	}

	psmx2_tag_layout_locked = 1;
out:
#elif (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_TAG64)
	use_tag64 = 1;
#else
	use_tag64 = 0;
#endif
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"use %s: tag_mask: %016" PRIX64 ", data_mask: %08" PRIX32 "\n",
		use_tag64 ? "tag64" : "tag60", (uint64_t)PSMX2_TAG_MASK,
		PSMX2_DATA_MASK);
}

static int psmx2_get_yes_no(char *s, int default_value)
{
	unsigned long value;
	char *end_ptr;

	if (!s || s[0] == '\0')
		return default_value;

	if (s[0] == 'Y' || s[0] == 'y')
		return 1;

	if (s[0] == 'N' || s[0] == 'n')
		return 0;

	value = strtoul(s, &end_ptr, 0);
	if (end_ptr == s)
		return default_value;

	return value ? 1 : 0;
}

static int psmx2_check_multi_ep_cap(void)
{
	uint64_t caps = PSM2_MULTI_EP_CAP;
	char *s = getenv("PSM2_MULTI_EP");

	if (psm2_get_capability_mask(caps) == caps && psmx2_get_yes_no(s, 0))
		psmx2_env.multi_ep = 1;
	else
		psmx2_env.multi_ep = 0;

	return psmx2_env.multi_ep;
}

static int psmx2_init_lib(void)
{
	int major, minor;
	int ret = 0, err;

	if (psmx2_lib_initialized)
		return 0;

	pthread_mutex_lock(&psmx2_lib_mutex);

	if (psmx2_lib_initialized)
		goto out;

	/* turn on multi-ep feature, but don't overwrite existing setting */
	setenv("PSM2_MULTI_EP", "1", 0);

	psm2_error_register_handler(NULL, PSM2_ERRHANDLER_NO_HANDLER);

	major = PSM2_VERNO_MAJOR;
	minor = PSM2_VERNO_MINOR;

	err = psm2_init(&major, &minor);
	if (err != PSM2_OK) {
		FI_WARN(&psmx2_prov, FI_LOG_CORE,
			"psm2_init failed: %s\n", psm2_error_get_string(err));
		ret = err;
		goto out;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM2 header version = (%d, %d)\n", PSM2_VERNO_MAJOR, PSM2_VERNO_MINOR);
	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"PSM2 library version = (%d, %d)\n", major, minor);

	if (psmx2_check_multi_ep_cap())
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "PSM2 multi-ep feature enabled.\n");
	else
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "PSM2 multi-ep feature not available or disabled.\n");

	psmx2_lib_initialized = 1;

out:
	pthread_mutex_unlock(&psmx2_lib_mutex);
	return ret;
}

#define PSMX2_SYSFS_PATH "/sys/class/infiniband/hfi1"
static int psmx2_read_sysfs_int(int unit, char *entry)
{
	char path[64];
	char buffer[32];
	int n, ret = 0;

	sprintf(path, "%s_%d", PSMX2_SYSFS_PATH, unit);
	n = fi_read_file(path, entry, buffer, 32);
	if (n > 0) {
		buffer[n] = 0;
		ret = strtol(buffer, NULL, 10);
		FI_INFO(&psmx2_prov, FI_LOG_CORE, "%s/%s: %d\n", path, entry, ret);
	}

	return ret;
}

static int psmx2_unit_active(int unit)
{
	return (4 == psmx2_read_sysfs_int(unit, "ports/1/state"));
}

#define PSMX2_MAX_UNITS	4
static int psmx2_active_units[PSMX2_MAX_UNITS];
static int psmx2_num_active_units;

static void psmx2_update_hfi_info(void)
{
	int i;
	int nctxts = 0;
	int nfreectxts = 0;
	int hfi_unit = -1;
	int multirail = 0;
	char *s;

	assert(psmx2_env.num_devunits <= PSMX2_MAX_UNITS);

	s = getenv("HFI_UNIT");
	if (s)
		hfi_unit = atoi(s);

	s = getenv("PSM2_MULTIRAIL");
	if (s)
		multirail = atoi(s);

	psmx2_num_active_units = 0;
	for (i = 0; i < psmx2_env.num_devunits; i++) {
		if (!psmx2_unit_active(i)) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hfi %d skipped: inactive\n", i);
			continue;
		}

		if (hfi_unit >=0 && i != hfi_unit) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"hfi %d skipped: HFI_UNIT=%d\n",
				i, hfi_unit);
			continue;
		}

		nctxts += psmx2_read_sysfs_int(i, "nctxts");
		nfreectxts += psmx2_read_sysfs_int(i, "nfreectxts");
		psmx2_active_units[psmx2_num_active_units++] = i;

		if (multirail)
			break;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"hfi1 units: total %d, active %d; "
		"hfi1 contexts: total %d, free %d\n",
		psmx2_env.num_devunits, psmx2_num_active_units,
		nctxts, nfreectxts);

	if (psmx2_env.multi_ep) {
		psmx2_env.max_trx_ctxt = nctxts;
		psmx2_env.free_trx_ctxt = nfreectxts;
	} else if (nfreectxts == 0) {
		psmx2_env.free_trx_ctxt = nfreectxts;
	}

	FI_INFO(&psmx2_prov, FI_LOG_CORE,
		"Tx/Rx contexts: %d in total, %d available.\n",
		psmx2_env.max_trx_ctxt, psmx2_env.free_trx_ctxt);
}

int psmx2_get_round_robin_unit(int idx)
{
	return psmx2_num_active_units ?
			psmx2_active_units[idx % psmx2_num_active_units] :
			-1;
}

static int psmx2_getinfo(uint32_t api_version, const char *node,
			 const char *service, uint64_t flags,
			 const struct fi_info *hints, struct fi_info **info)
{
	struct fi_info *prov_info = NULL;
	struct psmx2_ep_name *dest_addr = NULL;
	struct psmx2_ep_name *src_addr = NULL;
	int svc0, svc = PSMX2_ANY_SERVICE;
	size_t len;
	void *addr;
	uint32_t fmt;
	glob_t glob_buf;
	uint32_t cnt = 0;

	FI_INFO(&psmx2_prov, FI_LOG_CORE,"\n");

	if (psmx2_init_prov_info(hints, &prov_info))
		goto err_out;

	if (psmx2_init_lib())
		goto err_out;

	/*
	 * psm2_ep_num_devunits() may wait for 15 seconds before return
	 * when /dev/hfi1_0 is not present. Check the existence of any hfi1
	 * device interface first to avoid this delay. Note that the devices
	 * don't necessarily appear consecutively so we need to check all
	 * possible device names before returning "no device found" error.
	 * This also means if "/dev/hfi1_0" doesn't exist but other devices
	 * exist, we are still going to see the delay; but that's a rare case.
	 */
	if ((glob("/dev/hfi1_[0-9]", 0, NULL, &glob_buf) != 0) &&
	    (glob("/dev/hfi1_[0-9][0-9]", GLOB_APPEND, NULL, &glob_buf) != 0)) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no hfi1 device is found.\n");
		goto err_out;
	}
	globfree(&glob_buf);

	if (psm2_ep_num_devunits(&cnt) || !cnt) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no PSM2 device is found.\n");
		goto err_out;
	}
	psmx2_env.num_devunits = cnt;
	psmx2_update_hfi_info();

	if (!psmx2_num_active_units) {
		FI_INFO(&psmx2_prov, FI_LOG_CORE,
			"no PSM2 device is active.\n");
		goto err_out;
	}

	/* Set src or dest to used supplied address in native format */
	if (node &&
	    !ofi_str_toaddr(node, &fmt, &addr, &len) &&
	    fmt == FI_ADDR_PSMX2) {
		if (flags & FI_SOURCE) {
			src_addr = addr;
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s' is taken as src_addr: <unit=%d, port=%d, service=%d>\n",
				node, src_addr->unit, src_addr->port, src_addr->service);
		} else {
			dest_addr = addr;
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s' is taken as dest_addr: <epid=%"PRIu64">\n",
				node, dest_addr->epid);
		}
		node = NULL;
	}

	/* Initialize src address based on the "host:unit:port" format */
	if (!src_addr) {
		src_addr = calloc(1, sizeof(*src_addr));
		if (!src_addr) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"failed to allocate src addr.\n");
			goto err_out;
		}
		src_addr->type = PSMX2_EP_SRC_ADDR;
		src_addr->epid = PSMX2_RESERVED_EPID;
		src_addr->unit = PSMX2_DEFAULT_UNIT;
		src_addr->port = PSMX2_DEFAULT_PORT;
		src_addr->service = PSMX2_ANY_SERVICE;

		if (flags & FI_SOURCE) {
			if (node)
				sscanf(node, "%*[^:]:%" SCNi8 ":%" SCNu8,
				       &src_addr->unit, &src_addr->port);
			if (service)
				sscanf(service, "%" SCNu32, &src_addr->service);
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"node '%s' service '%s' converted to <unit=%d, port=%d, service=%d>\n",
				node, service, src_addr->unit, src_addr->port, src_addr->service);
		}
	}

	/* Resovle dest address using "node", "service" pair */
	if (!dest_addr && node && !(flags & FI_SOURCE)) {
		psm2_uuid_t uuid;

		psmx2_get_uuid(uuid);
		struct util_ns ns = {
			.port = psmx2_uuid_to_port(uuid),
			.name_len = sizeof(*dest_addr),
			.service_len = sizeof(svc),
			.service_cmp = psmx2_ns_service_cmp,
			.is_service_wildcard = psmx2_ns_is_service_wildcard,
		};
		ofi_ns_init(&ns);

		if (service)
			svc = atoi(service);
		svc0 = svc;
		dest_addr = (struct psmx2_ep_name *)
			ofi_ns_resolve_name(&ns, node, &svc);
		if (dest_addr) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"'%s:%u' resolved to <epid=%"PRIu64">:%d\n",
				node, svc0, dest_addr->epid, svc);
		} else {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"failed to resolve '%s:%u'.\n", node, svc);
			goto err_out;
		}
	}

	/* Update prov info with resovled addresses and environment settings */
	psmx2_update_prov_info(prov_info, src_addr, dest_addr);

	/* Remove prov info that don't match the hints */
	if (psmx2_check_prov_info(api_version, hints, &prov_info))
		goto err_out;

	/* Apply hints to the prov info */
	psmx2_alter_prov_info(api_version, hints, prov_info);
	*info = prov_info;
	return 0;

err_out:
	free(src_addr);
	free(dest_addr);
	fi_freeinfo(prov_info);
	*info = NULL;
	return -FI_ENODATA;
}

static void psmx2_fini(void)
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	if (! --psmx2_init_count && psmx2_lib_initialized) {
		/* This function is called from a library destructor, which is called
		 * automatically when exit() is called. The call to psm2_finalize()
		 * might cause deadlock if the applicaiton is terminated with Ctrl-C
		 * -- the application could be inside a PSM2 call, holding a lock that
		 * psm2_finalize() tries to acquire. This can be avoided by only
		 * calling psm2_finalize() when PSM2 is guaranteed to be unused.
		 */
		if (psmx2_active_fabric) {
			FI_INFO(&psmx2_prov, FI_LOG_CORE,
				"psmx2_active_fabric != NULL, skip psm2_finalize\n");
		} else {
			psm2_finalize();
			psmx2_lib_initialized = 0;
		}
	}
}

struct fi_provider psmx2_prov = {
	.name = PSMX2_PROV_NAME,
	.version = PSMX2_VERSION,
	.fi_version = PSMX2_VERSION,
	.getinfo = psmx2_getinfo,
	.fabric = psmx2_fabric,
	.cleanup = psmx2_fini
};

PROVIDER_INI
{
	FI_INFO(&psmx2_prov, FI_LOG_CORE, "build options: HAVE_PSM2_SRC=%d, "
			"HAVE_PSM2_AM_REGISTER_HANDLERS_2=%d, "
			"PSMX2_USE_REQ_CONTEXT=%d\n", HAVE_PSM2_SRC,
			HAVE_PSM2_AM_REGISTER_HANDLERS_2, PSMX2_USE_REQ_CONTEXT);

	fi_param_define(&psmx2_prov, "name_server", FI_PARAM_BOOL,
			"Whether to turn on the name server or not "
			"(default: yes)");

	fi_param_define(&psmx2_prov, "tagged_rma", FI_PARAM_BOOL,
			"Whether to use tagged messages for large size "
			"RMA or not (default: yes)");

	fi_param_define(&psmx2_prov, "uuid", FI_PARAM_STRING,
			"Unique Job ID required by the fabric");

	fi_param_define(&psmx2_prov, "delay", FI_PARAM_INT,
			"Delay (seconds) before finalization (for debugging)");

	fi_param_define(&psmx2_prov, "timeout", FI_PARAM_INT,
			"Timeout (seconds) for gracefully closing the PSM2 endpoint");

	fi_param_define(&psmx2_prov, "prog_interval", FI_PARAM_INT,
			"Interval (microseconds) between progress calls made in the "
			"progress thread (default: 1 if affinity is set, 1000 if not)");

	fi_param_define(&psmx2_prov, "prog_affinity", FI_PARAM_STRING,
			"When set, specify the set of CPU cores to set the progress "
			"thread affinity to. The format is "
			"<start>[:<end>[:<stride>]][,<start>[:<end>[:<stride>]]]*, "
			"where each triplet <start>:<end>:<stride> defines a block "
			"of core_ids. Both <start> and <end> can be either the core_id "
			"(when >=0) or core_id - num_cores (when <0). "
			"(default: affinity not set)");

	fi_param_define(&psmx2_prov, "inject_size", FI_PARAM_INT,
			"Maximum message size for fi_inject and fi_tinject (default: 64).");

	fi_param_define(&psmx2_prov, "lock_level", FI_PARAM_INT,
			"How internal locking is used. 0 means no locking. (default: 2).");

	fi_param_define(&psmx2_prov, "lazy_conn", FI_PARAM_BOOL,
			"Whether to use lazy connection or not (default: no).");

	fi_param_define(&psmx2_prov, "disconnect", FI_PARAM_BOOL,
			"Whether to issue disconnect request when process ends (default: no).");

#if (PSMX2_TAG_LAYOUT == PSMX2_TAG_LAYOUT_RUNTIME)
	fi_param_define(&psmx2_prov, "tag_layout", FI_PARAM_STRING,
			"How the 96 bit PSM2 tag is organized: "
			"tag60 means 32/4/60 for data/flags/tag;"
			"tag64 means 4/28/64 for flags/data/tag (default: tag60).");
#endif

	psmx2_init_env();

	pthread_mutex_init(&psmx2_lib_mutex, NULL);
	psmx2_init_count++;
	return (&psmx2_prov);
}

