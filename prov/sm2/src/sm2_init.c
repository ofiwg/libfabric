/*
 * Copyright (c) 2015-2021 Intel Corporation. All rights reserved.
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include <rdma/fi_errno.h>

#include <ofi_prov.h>
#include "sm2.h"
#include "sm2_signal.h"
#include <ofi_hmem.h>

struct sigaction *sm2_old_action = NULL;

struct sm2_env sm2_env = {
	.sar_threshold = SIZE_MAX,
	.disable_cma = false,
	.use_dsa_sar = false,
};

static void sm2_init_env(void)
{
	fi_param_get_size_t(&sm2_prov, "sar_threshold", &sm2_env.sar_threshold);
	fi_param_get_size_t(&sm2_prov, "tx_size", &sm2_info.tx_attr->size);
	fi_param_get_size_t(&sm2_prov, "rx_size", &sm2_info.rx_attr->size);
	fi_param_get_bool(&sm2_prov, "disable_cma", &sm2_env.disable_cma);
	fi_param_get_bool(&sm2_prov, "use_dsa_sar", &sm2_env.use_dsa_sar);
}

/**
 * @brief convert strings node + service into a single string addr
 *
 * @param[in] node		NULL or a node name
 * @param[in] service		NULL or service name
 * @param[out] addr		the string of the address we should use
*/
static void sm2_resolve_addr(const char *node, const char *service,
			     char **addr, size_t *addrlen)
{
	char temp_name[SM2_NAME_MAX];

	FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "resolving node=%s, service=%s\n", node?node:"NULL", service?service:"NULL");
	if (service) {
		if (node)
			*addrlen = snprintf(temp_name, SM2_NAME_MAX - 1, "%s%s:%s",
				 SM2_PREFIX_NS, node, service);
		else
			*addrlen = snprintf(temp_name, SM2_NAME_MAX - 1, "%s%s",
				 SM2_PREFIX_NS, service);
	} else {
		if (node)
			*addrlen = snprintf(temp_name, SM2_NAME_MAX - 1, "%s%s",
				 SM2_PREFIX, node);
		else
			*addrlen = snprintf(temp_name, SM2_NAME_MAX - 1, "%s%d",
				 SM2_PREFIX, getpid());
	}
	*addr = strndup(temp_name, SM2_NAME_MAX - 1);
	FI_WARN(&sm2_prov, FI_LOG_EP_CTRL, "resolved to %s\n", temp_name);
}

/*
 * The sm2_shm_space_check is to check if there's enough shm space we
 * need under /dev/shm.
 * Here we use #core instead of sm2_MAX_PEERS, as it is the most likely
 * value and has less possibility of failing fi_getinfo calls that are
 * currently passing, and breaking currently working app
 */
static int sm2_shm_space_check(size_t tx_count, size_t rx_count)
{
// TODO: call this routine AFTER we have mmap, but BEFORE we allocate our space.
// TODO: base the return value on the contents of the header region size.
// TODO: And ignore existing file allocation size when stat() /dev/shm/
	struct statvfs stat;
	char shm_fs[] = "/dev/shm";
	uint64_t available_size, shm_size_needed;
	int num_of_core, err;
	size_t num_fqe = tx_count + rx_count;

	num_of_core = ofi_sysconf(_SC_NPROCESSORS_ONLN);
	if (num_of_core < 0) {
		FI_WARN(&sm2_prov, FI_LOG_CORE,
			"Get number of processor failed (%s)\n",
			strerror(errno));
		return -errno;
	}
	shm_size_needed = num_of_core *
			  sm2_calculate_size_offsets(num_fqe,
						     NULL, NULL);
	err = statvfs(shm_fs, &stat);
	if (err) {
		FI_WARN(&sm2_prov, FI_LOG_CORE,
			"Get filesystem %s statistics failed (%s)\n",
			shm_fs, strerror(errno));
	} else {
		available_size = stat.f_bsize * stat.f_bavail;
		if (available_size < shm_size_needed) {
			FI_WARN(&sm2_prov, FI_LOG_CORE,
				"Not enough available space in %s.\n", shm_fs);
			return -FI_ENOSPC;
		}
	}
	// TODO: we should ignore space already reserved in the global shm file.
	return 0;
}

static int sm2_getinfo(uint32_t version, const char *node, const char *service,
		       uint64_t flags, const struct fi_info *hints,
		       struct fi_info **info)
{
	struct fi_info *cur;
	int ret;

	ret = util_getinfo(&sm2_util_prov, version, node, service, flags,
			   hints, info);
	if (ret)
		return ret;

	ret = sm2_shm_space_check((*info)->tx_attr->size, (*info)->rx_attr->size);
	if (ret) {
		fi_freeinfo(*info);
		return ret;
	}

	for (cur = *info; cur; cur = cur->next) {
		if (!(flags & FI_SOURCE) && !cur->dest_addr)
			sm2_resolve_addr(node, service, (char **) &cur->dest_addr,
					 &cur->dest_addrlen);

		if (!cur->src_addr) {
			if (flags & FI_SOURCE)
				sm2_resolve_addr(node, service, (char **) &cur->src_addr,
						 &cur->src_addrlen);
			else
				sm2_resolve_addr(NULL, NULL, (char **) &cur->src_addr,
						 &cur->src_addrlen);
		}
	}
	return 0;
}

static void sm2_fini(void)
{
#if HAVE_SM2_DL
	ofi_hmem_cleanup();
#endif
	sm2_cleanup();
	free(sm2_old_action);
}

struct fi_provider sm2_prov = {
	.name = "sm2",
	.version = OFI_VERSION_DEF_PROV,
	.fi_version = OFI_VERSION_LATEST,
	.getinfo = sm2_getinfo,
	.fabric = sm2_fabric,
	.cleanup = sm2_fini
};

struct util_prov sm2_util_prov = {
	.prov = &sm2_prov,
	.info = &sm2_info,
	.flags = 0
};

SM2_INI
{
#if HAVE_SM2_DL
	ofi_hmem_init();
#endif
	fi_param_define(&sm2_prov, "sar_threshold", FI_PARAM_SIZE_T,
			"Max size to use for alternate SAR protocol if CMA \
			 is not available before switching to mmap protocol \
			 Default: SIZE_MAX (18446744073709551615)");
	fi_param_define(&sm2_prov, "tx_size", FI_PARAM_SIZE_T,
			"Max number of outstanding tx operations \
			 Default: 1024");
	fi_param_define(&sm2_prov, "rx_size", FI_PARAM_SIZE_T,
			"Max number of outstanding rx operations \
			 Default: 1024");
	fi_param_define(&sm2_prov, "disable_cma", FI_PARAM_BOOL,
			"Manually disables CMA. Default: false");
	fi_param_define(&sm2_prov, "use_dsa_sar", FI_PARAM_BOOL,
			"Enable use of DSA in SAR protocol. Default: false");
	fi_param_define(&sm2_prov, "enable_dsa_page_touch", FI_PARAM_BOOL,
			"Enable CPU touching of memory pages in DSA command \
			 descriptor when page fault is reported. \
			 Default: false");

	sm2_init_env();

	sm2_old_action = calloc(SIGRTMIN, sizeof(*sm2_old_action));
	if (!sm2_old_action)
		return NULL;

	return &sm2_prov;
}
