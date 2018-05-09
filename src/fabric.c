/*
 * Copyright (c) 2004, 2005 Topspin Communications.  All rights reserved.
 * Copyright (c) 2006-2016 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2013-2017 Intel Corp., Inc.  All rights reserved.
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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>

#include <rdma/fi_errno.h>
#include "ofi_util.h"
#include "ofi.h"
#include "ofi_prov.h"

#ifdef HAVE_LIBDL
#include <dlfcn.h>
#endif

struct ofi_prov {
	struct ofi_prov		*next;
	char			*prov_name;
	struct fi_provider	*provider;
	void			*dlhandle;
};

static struct ofi_prov *prov_head, *prov_tail;
int ofi_init = 0;
extern struct ofi_common_locks common_locks;

static struct fi_filter prov_filter;

static int ofi_find_name(char **names, const char *name)
{
	int i;

	for (i = 0; names[i]; i++) {
		if (!strcasecmp(name, names[i]))
			return i;
	}
	return -1;
}

static int ofi_is_util_prov(const struct fi_provider *provider)
{
	const struct fi_prov_context *ctx;
	ctx = (const struct fi_prov_context *) &provider->context;
	return ctx->is_util_prov;
}

int ofi_apply_filter(struct fi_filter *filter, const char *name)
{
	if (filter->names) {
		if (ofi_find_name(filter->names, name) >= 0)
			return filter->negated ? 1 : 0;

		return filter->negated ? 0 : 1;
	}
	return 0;
}

/*
 * Utility providers may be disabled, but do not need to be explicitly
 * enabled.  This allows them to always be available when only a core
 * provider is enabled.
 */
static int ofi_getinfo_filter(const struct fi_provider *provider)
{
	if (!prov_filter.negated && ofi_is_util_prov(provider))
		return 0;

	return ofi_apply_filter(&prov_filter, provider->name);
}

static struct ofi_prov *ofi_getprov(const char *prov_name, size_t len)
{
	struct ofi_prov *prov;

	for (prov = prov_head; prov; prov = prov->next) {
		if ((strlen(prov->prov_name) == len) &&
		    !strncmp(prov->prov_name, prov_name, len))
			return prov;
	}

	return NULL;
}

static void cleanup_provider(struct fi_provider *provider, void *dlhandle)
{
	OFI_UNUSED(dlhandle);

	if (provider) {
		fi_param_undefine(provider);

		if (provider->cleanup)
			provider->cleanup();
	}

#ifdef HAVE_LIBDL
	if (dlhandle)
		dlclose(dlhandle);
#endif
}

static struct ofi_prov *ofi_create_prov_entry(const char *prov_name)
{
	struct ofi_prov *prov = NULL;
	prov = calloc(sizeof *prov, 1);
	if (!prov) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Not enough memory to allocate provider registry\n");
		return NULL;
	}

	prov->prov_name = strdup(prov_name);
	if (!prov->prov_name) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Failed to init pre-registered provider name\n");
		free(prov);
		return NULL;
	}
	if (prov_tail)
		prov_tail->next = prov;
	else
		prov_head = prov;
	prov_tail = prov;

	return prov;
}

/* This is the default order that providers will be reported when a provider
 * is available.  Initialize the socket(s) provider last.  This will result in
 * it being the least preferred provider.
 */
static void ofi_ordered_provs_init(void)
{
	char *ordered_prov_names[] =
			{"psm2", "psm", "usnic", "mlx", "gni",
			 "bgq", "netdir", "ofi_rxm", "ofi_rxd", "verbs",
			/* Initialize the socket(s) provider last.  This will result in
			 * it being the least preferred provider. */

			/* Before you add ANYTHING here, read the comment above!!! */
			"UDP", "sockets", "tcp" /* NOTHING GOES HERE! */};
			/* Seriously, read it! */
	int num_provs = sizeof(ordered_prov_names)/sizeof(ordered_prov_names[0]), i;

	for (i = 0; i < num_provs; i++)
		ofi_create_prov_entry(ordered_prov_names[i]);
}

static int ofi_register_provider(struct fi_provider *provider, void *dlhandle)
{
	struct fi_prov_context *ctx;
	struct ofi_prov *prov = NULL;
	size_t len;
	int ret;

	if (!provider || !provider->name) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"no provider structure or name\n");
		ret = -FI_EINVAL;
		goto cleanup;
	}

	FI_INFO(&core_prov, FI_LOG_CORE,
	       "registering provider: %s (%d.%d)\n", provider->name,
	       FI_MAJOR(provider->version), FI_MINOR(provider->version));

	if (!provider->getinfo || !provider->fabric) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"provider missing mandatory entry points\n");
		ret = -FI_EINVAL;
		goto cleanup;
	}

	/* The current core implementation is not backward compatible
	 * with providers that support a release earlier than v1.3.
	 * See commit 0f4b6651.
	 */
	if (provider->fi_version < FI_VERSION(1, 3)) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"provider has unsupported FI version "
			"(provider %d.%d != libfabric %d.%d); ignoring\n",
			FI_MAJOR(provider->fi_version),
			FI_MINOR(provider->fi_version), FI_MAJOR_VERSION,
			FI_MINOR_VERSION);

		ret = -FI_ENOSYS;
		goto cleanup;
	}

	ctx = (struct fi_prov_context *) &provider->context;
	ctx->is_util_prov = (ofi_util_name(provider->name, &len) != NULL);

	if (ofi_getinfo_filter(provider)) {
		FI_INFO(&core_prov, FI_LOG_CORE,
			"\"%s\" filtered by provider include/exclude "
			"list, skipping\n", provider->name);
		ret = -FI_ENODEV;
		goto cleanup;
	}

	if (ofi_apply_filter(&prov_log_filter, provider->name))
		ctx->disable_logging = 1;

	prov = ofi_getprov(provider->name, strlen(provider->name));
	if (prov) {
		/* If this provider has not been init yet, then we add the
		 * provider and dlhandle to the struct and exit.
		 */
		if (prov->provider == NULL)
			goto update_prov_registry;

		/* If this provider is older than an already-loaded
		 * provider of the same name, then discard this one.
		 */
		if (FI_VERSION_GE(prov->provider->version, provider->version)) {
			FI_INFO(&core_prov, FI_LOG_CORE,
				"a newer %s provider was already loaded; "
				"ignoring this one\n", provider->name);
			ret = -FI_EALREADY;
			goto cleanup;
		}

		/* This provider is newer than an already-loaded
		 * provider of the same name, so discard the
		 * already-loaded one.
		 */
		FI_INFO(&core_prov, FI_LOG_CORE,
			"an older %s provider was already loaded; "
			"keeping this one and ignoring the older one\n",
			provider->name);
		cleanup_provider(prov->provider, prov->dlhandle);
	} else {
		prov = ofi_create_prov_entry(provider->name);
		if (!prov) {
			ret = -FI_EOTHER;
			goto cleanup;
		}
	}

update_prov_registry:
	prov->dlhandle = dlhandle;
	prov->provider = provider;
	return 0;

cleanup:
	cleanup_provider(provider, dlhandle);
	return ret;
}

#ifdef HAVE_LIBDL
static int lib_filter(const struct dirent *entry)
{
	size_t l = strlen(entry->d_name);
	size_t sfx = sizeof (FI_LIB_SUFFIX) - 1;

	if (l > sfx)
		return !strcmp(&(entry->d_name[l-sfx]), FI_LIB_SUFFIX);
	else
		return 0;
}
#endif

/* split the given string "s" using the specified delimiter(s) in the string
 * "delim" and return an array of strings.  The array is terminated with a NULL
 * pointer.  You can clean this array up with a call to free_string_array().
 *
 * Returns NULL on failure.
 */
static char **split_and_alloc(const char *s, const char *delim)
{
	int i, n;
	char *tmp;
	char *dup = NULL;
	char **arr = NULL;

	if (!s || !delim)
		return NULL;

	dup = strdup(s);
	if (!dup) {
		FI_WARN(&core_prov, FI_LOG_CORE, "failed to allocate memory\n");
		return NULL;
	}

	/* compute the array size */
	n = 1;
	for (tmp = dup; *tmp != '\0'; ++tmp) {
		for (i = 0; delim[i] != '\0'; ++i) {
			if (*tmp == delim[i]) {
				++n;
				break;
			}
		}
	}

	/* +1 to leave space for NULL terminating pointer */
	arr = calloc(n + 1, sizeof(*arr));
	if (!arr) {
		FI_WARN(&core_prov, FI_LOG_CORE, "failed to allocate memory\n");
		goto cleanup;
	}

	/* set array elts to point inside the dup'ed string */
	for (tmp = dup, i = 0; tmp != NULL; ++i) {
		arr[i] = strsep(&tmp, delim);
	}
	assert(i == n);

	return arr;

cleanup:
	free(dup);
	free(arr);
	return NULL;
}

/* see split_and_alloc() */
static void free_string_array(char **s)
{
	/* all strings are allocated from the same strdup'ed slab, so just free
	 * the first element */
	if (s != NULL)
		free(s[0]);

	/* and then the actual array of pointers */
	free(s);
}

void ofi_free_filter(struct fi_filter *filter)
{
	free_string_array(filter->names);
}

void ofi_create_filter(struct fi_filter *filter, const char *raw_filter)
{
	memset(filter, 0, sizeof *filter);
	if (raw_filter == NULL)
		return;

	if (*raw_filter == '^') {
		filter->negated = 1;
		++raw_filter;
	}

	filter->names = split_and_alloc(raw_filter, ",");
	if (!filter->names)
		FI_WARN(&core_prov, FI_LOG_CORE,
			"unable to parse filter from: %s\n", raw_filter);
}

#ifdef HAVE_LIBDL
static void ofi_ini_dir(const char *dir)
{
	int n = 0;
	char *lib;
	void *dlhandle;
	struct dirent **liblist = NULL;
	struct fi_provider* (*inif)(void);

	n = scandir(dir, &liblist, lib_filter, NULL);
	if (n < 0)
		goto libdl_done;

	while (n--) {
		if (asprintf(&lib, "%s/%s", dir, liblist[n]->d_name) < 0) {
			FI_WARN(&core_prov, FI_LOG_CORE,
			       "asprintf failed to allocate memory\n");
			goto libdl_done;
		}
		FI_DBG(&core_prov, FI_LOG_CORE, "opening provider lib %s\n", lib);

		dlhandle = dlopen(lib, RTLD_NOW);
		free(liblist[n]);
		if (dlhandle == NULL) {
			FI_WARN(&core_prov, FI_LOG_CORE,
			       "dlopen(%s): %s\n", lib, dlerror());
			free(lib);
			continue;
		}
		free(lib);

		inif = dlsym(dlhandle, "fi_prov_ini");
		if (inif == NULL) {
			FI_WARN(&core_prov, FI_LOG_CORE, "dlsym: %s\n", dlerror());
			dlclose(dlhandle);
		} else {
			ofi_register_provider((inif)(), dlhandle);
		}
	}

libdl_done:
	while (n-- > 0)
		free(liblist[n]);
	free(liblist);
}
#endif

void fi_ini(void)
{
	char *param_val = NULL;

	pthread_mutex_lock(&common_locks.ini_lock);

	if (ofi_init)
		goto unlock;

	ofi_ordered_provs_init();
	fi_param_init();
	fi_log_init();
	ofi_osd_init();
	ofi_pmem_init();

	fi_param_define(NULL, "provider", FI_PARAM_STRING,
			"Only use specified provider (default: all available)");
	fi_param_define(NULL, "fork_unsafe", FI_PARAM_BOOL,
			"Whether use of fork() may be unsafe for some providers"
			" (default: no). Setting this to yes could improve"
			" performance at the expense of making fork() potentially"
			" unsafe");
	fi_param_define(NULL, "universe_size", FI_PARAM_SIZE_T,
			"Defines the maximum number of processes that will be"
			" used by distribute OFI application. The provider uses"
			" this to optimize resource allocations"
			" (default: OFI service specific)");
	fi_param_get_str(NULL, "provider", &param_val);
	ofi_create_filter(&prov_filter, param_val);

#ifdef HAVE_LIBDL
	int n = 0;
	char **dirs;
	char *provdir = NULL;
	void *dlhandle;

	/* If dlopen fails, assume static linking and just return
	   without error */
	dlhandle = dlopen(NULL, RTLD_NOW);
	if (dlhandle == NULL) {
		goto libdl_done;
	}
	dlclose(dlhandle);

	fi_param_define(NULL, "provider_path", FI_PARAM_STRING,
			"Search for providers in specific path (default: "
			PROVDLDIR ")");
	fi_param_get_str(NULL, "provider_path", &provdir);
	if (!provdir)
		provdir = PROVDLDIR;

	dirs = split_and_alloc(provdir, ":");
	if (dirs) {
		for (n = 0; dirs[n]; ++n) {
			ofi_ini_dir(dirs[n]);
		}
		free_string_array(dirs);
	}
libdl_done:
#endif

	ofi_register_provider(PSM2_INIT, NULL);
	ofi_register_provider(PSM_INIT, NULL);
	ofi_register_provider(USNIC_INIT, NULL);
	ofi_register_provider(MLX_INIT, NULL);
	ofi_register_provider(GNI_INIT, NULL);
	ofi_register_provider(BGQ_INIT, NULL);
	ofi_register_provider(NETDIR_INIT, NULL);
	ofi_register_provider(SHM_INIT, NULL);
	ofi_register_provider(RXM_INIT, NULL);
	ofi_register_provider(VERBS_INIT, NULL);

	{
		/* TODO: RXD is not stable for now. Disable it by default */
		int enable_rxd = 0;
		fi_param_define(NULL, "rxd_enable", FI_PARAM_BOOL,
				"Enable RXD provider (default: no)");
		fi_param_get_bool(NULL, "rxd_enable", &enable_rxd);
		if (enable_rxd)
			ofi_register_provider(RXD_INIT, NULL);
	}

	ofi_register_provider(UDP_INIT, NULL);
	ofi_register_provider(SOCKETS_INIT, NULL);

	ofi_init = 1;

unlock:
	pthread_mutex_unlock(&common_locks.ini_lock);
}

FI_DESTRUCTOR(fi_fini(void))
{
	struct ofi_prov *prov;

	if (!ofi_init)
		return;

	while (prov_head) {
		prov = prov_head;
		prov_head = prov->next;
		cleanup_provider(prov->provider, prov->dlhandle);
		free(prov->prov_name);
		free(prov);
	}

	ofi_free_filter(&prov_filter);
	fi_log_fini();
	fi_param_fini();
	ofi_osd_fini();
}

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
void DEFAULT_SYMVER_PRE(fi_freeinfo)(struct fi_info *info)
{
	struct fi_info *next;

	for (; info; info = next) {
		next = info->next;

		free(info->src_addr);
		free(info->dest_addr);
		free(info->tx_attr);
		free(info->rx_attr);
		if (info->ep_attr) {
			free(info->ep_attr->auth_key);
			free(info->ep_attr);
		}
		if (info->domain_attr) {
			free(info->domain_attr->auth_key);
			free(info->domain_attr->name);
			free(info->domain_attr);
		}
		if (info->fabric_attr) {
			free(info->fabric_attr->name);
			free(info->fabric_attr->prov_name);
			free(info->fabric_attr);
		}
		free(info);
	}
}
CURRENT_SYMVER(fi_freeinfo_, fi_freeinfo);

/*
 * Make a dummy info object for each provider, and copy in the
 * provider name and version.  We report utility providers directly
 * to export their version.
 */
static int ofi_getprovinfo(struct fi_info **info)
{
	struct ofi_prov *prov;
	struct fi_info *tail, *cur;
	int ret = -FI_ENODATA;

	*info = tail = NULL;
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->provider)
			continue;

		cur = fi_allocinfo();
		if (!cur) {
			ret = -FI_ENOMEM;
			goto err;
		}

		cur->fabric_attr->prov_name = strdup(prov->provider->name);
		cur->fabric_attr->prov_version = prov->provider->version;

		if (!*info) {
			*info = tail = cur;
		} else {
			tail->next = cur;
		}
		tail = cur;

		ret = 0;
	}

	return ret;

err:
	while (tail) {
		cur = tail->next;
		fi_freeinfo(tail);
		tail = cur;
	}
	return ret;
}

static void ofi_set_prov_attr(struct fi_fabric_attr *attr,
			      struct fi_provider *prov)
{
	char *core_name;

	core_name = attr->prov_name;
	if (core_name) {
		assert(ofi_is_util_prov(prov));
		attr->prov_name = ofi_strdup_append(core_name, prov->name);
		free(core_name);
	} else {
		assert(!ofi_is_util_prov(prov));
		attr->prov_name = strdup(prov->name);
	}
	attr->prov_version = prov->version;
}

/*
 * The layering of utility providers over core providers follows these rules.
 * 1. If both are specified, then only return that layering
 * 2. If a utility provider is specified, return it over any* core provider.
 * 3. If a core provider is specified, return any utility provider that can
 *    layer over it, plus the core provider itself, if possible.
 * 4* A utility provider will not layer over the sockets provider unless the
 *    user explicitly requests that combination.
 *
 * Utility providers use an internal flag, OFI_CORE_PROV_ONLY, to indicate
 * that only core providers should respond to an fi_getinfo query.  This
 * prevents utility providers from layering over other utility providers.
 */
static int ofi_layering_ok(const struct fi_provider *provider,
			   const char *util_name, size_t util_len,
			   const char *core_name, size_t core_len,
			   uint64_t flags)
{
	if (flags & OFI_CORE_PROV_ONLY) {
		if (ofi_is_util_prov(provider)) {
			FI_INFO(&core_prov, FI_LOG_CORE,
				"Need core provider, skipping util %s\n",
				provider->name);
			return 0;
		}

		if ((!core_len || !core_name) &&
		    !strcasecmp(provider->name, "sockets")) {
			FI_INFO(&core_prov, FI_LOG_CORE,
				"Skipping util;sockets layering\n");
			return 0;
		}
	}

	if (util_len && util_name) {
		assert(!(flags & OFI_CORE_PROV_ONLY));
		if ((strlen(provider->name) != util_len) ||
		    strncasecmp(util_name, provider->name, util_len))
			return 0;

	} else if (core_len && core_name) {
		if (!strncasecmp(core_name, "sockets", core_len) &&
		    ofi_is_util_prov(provider)) {
			FI_INFO(&core_prov, FI_LOG_CORE,
				"Sockets requested, skipping util layering\n");
			return 0;
		}

		if (!ofi_is_util_prov(provider) &&
		    ((strlen(provider->name) != core_len) ||
		     strncasecmp(core_name, provider->name, core_len)))
			return 0;
	}

	return 1;
}

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
int DEFAULT_SYMVER_PRE(fi_getinfo)(uint32_t version, const char *node,
		const char *service, uint64_t flags,
		const struct fi_info *hints, struct fi_info **info)
{
	struct ofi_prov *prov;
	struct fi_info *tail, *cur;
	const char *util_name = NULL, *core_name = NULL;
	size_t util_len = 0, core_len = 0;
	int ret;

	if (!ofi_init)
		fi_ini();

	if (FI_VERSION_LT(fi_version(), version)) {
		FI_WARN(&core_prov, FI_LOG_CORE,
			"Requested version is newer than library\n");
		return -FI_ENOSYS;
	}

	if (flags == FI_PROV_ATTR_ONLY) {
		return ofi_getprovinfo(info);
	}

	if (hints && hints->fabric_attr && hints->fabric_attr->prov_name) {
		util_name = ofi_util_name(hints->fabric_attr->prov_name,
					  &util_len);
		core_name = ofi_core_name(hints->fabric_attr->prov_name,
					  &core_len);
	}

	*info = tail = NULL;
	for (prov = prov_head; prov; prov = prov->next) {
		if (!prov->provider)
			continue;

		if (!ofi_layering_ok(prov->provider, util_name, util_len,
				     core_name, core_len, flags))
			continue;

		if (FI_VERSION_LT(prov->provider->fi_version, version)) {
			FI_WARN(&core_prov, FI_LOG_CORE,
				"Provider %s fi_version %d.%d < requested %d.%d\n",
				prov->provider->name,
				FI_MAJOR(prov->provider->fi_version),
				FI_MINOR(prov->provider->fi_version),
				FI_MAJOR(version), FI_MINOR(version));
			continue;
		}

		ret = prov->provider->getinfo(version, node, service, flags,
					      hints, &cur);
		if (ret) {
			FI_WARN(&core_prov, FI_LOG_CORE,
			       "fi_getinfo: provider %s returned -%d (%s)\n",
			       prov->provider->name, -ret, fi_strerror(-ret));
			continue;
		}

		if (!cur) {
			FI_WARN(&core_prov, FI_LOG_CORE,
				"fi_getinfo: provider %s output empty list\n",
				prov->provider->name);
			continue;
		}

		if (!*info)
			*info = cur;
		else
			tail->next = cur;

		for (tail = cur; tail->next; tail = tail->next) {
			ofi_set_prov_attr(tail->fabric_attr, prov->provider);
			tail->fabric_attr->api_version = version;
		}
		ofi_set_prov_attr(tail->fabric_attr, prov->provider);
		tail->fabric_attr->api_version = version;
	}

	return *info ? 0 : -FI_ENODATA;
}
CURRENT_SYMVER(fi_getinfo_, fi_getinfo);

struct fi_info *ofi_allocinfo_internal(void)
{
	struct fi_info *info;

	info = calloc(1, sizeof(*info));
	if (!info)
		return NULL;

	info->tx_attr = calloc(1, sizeof(*info->tx_attr));
	info->rx_attr = calloc(1, sizeof(*info->rx_attr));
	info->ep_attr = calloc(1, sizeof(*info->ep_attr));
	info->domain_attr = calloc(1, sizeof(*info->domain_attr));
	info->fabric_attr = calloc(1, sizeof(*info->fabric_attr));
	if (!info->tx_attr|| !info->rx_attr || !info->ep_attr ||
	    !info->domain_attr || !info->fabric_attr)
		goto err;

	return info;
err:
	fi_freeinfo(info);
	return NULL;
}


__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
struct fi_info *DEFAULT_SYMVER_PRE(fi_dupinfo)(const struct fi_info *info)
{
	struct fi_info *dup;

	if (!info)
		return ofi_allocinfo_internal();

	dup = mem_dup(info, sizeof(*dup));
	if (dup == NULL) {
		return NULL;
	}
	dup->src_addr = NULL;
	dup->dest_addr = NULL;
	dup->tx_attr = NULL;
	dup->rx_attr = NULL;
	dup->ep_attr = NULL;
	dup->domain_attr = NULL;
	dup->fabric_attr = NULL;
	dup->next = NULL;

	if (info->src_addr != NULL) {
		dup->src_addr = mem_dup(info->src_addr, info->src_addrlen);
		if (dup->src_addr == NULL)
			goto fail;
	}
	if (info->dest_addr != NULL) {
		dup->dest_addr = mem_dup(info->dest_addr, info->dest_addrlen);
		if (dup->dest_addr == NULL)
			goto fail;
	}
	if (info->tx_attr != NULL) {
		dup->tx_attr = mem_dup(info->tx_attr, sizeof(*info->tx_attr));
		if (dup->tx_attr == NULL)
			goto fail;
	}
	if (info->rx_attr != NULL) {
		dup->rx_attr = mem_dup(info->rx_attr, sizeof(*info->rx_attr));
		if (dup->rx_attr == NULL)
			goto fail;
	}
	if (info->ep_attr != NULL) {
		dup->ep_attr = mem_dup(info->ep_attr, sizeof(*info->ep_attr));
		if (dup->ep_attr == NULL)
			goto fail;
		if (info->ep_attr->auth_key != NULL) {
			dup->ep_attr->auth_key =
				mem_dup(info->ep_attr->auth_key,
					info->ep_attr->auth_key_size);
			if (dup->ep_attr->auth_key == NULL)
				goto fail;
		}
	}
	if (info->domain_attr) {
		dup->domain_attr = mem_dup(info->domain_attr,
					   sizeof(*info->domain_attr));
		if (dup->domain_attr == NULL)
			goto fail;
		dup->domain_attr->name = NULL;
		dup->domain_attr->auth_key = NULL;
		if (info->domain_attr->name != NULL) {
			dup->domain_attr->name = strdup(info->domain_attr->name);
			if (dup->domain_attr->name == NULL)
				goto fail;
		}
		if (info->domain_attr->auth_key != NULL) {
			dup->domain_attr->auth_key =
				mem_dup(info->domain_attr->auth_key,
					info->domain_attr->auth_key_size);
			if (dup->domain_attr->auth_key == NULL)
				goto fail;
		}
	}
	if (info->fabric_attr) {
		dup->fabric_attr = mem_dup(info->fabric_attr,
					   sizeof(*info->fabric_attr));
		if (dup->fabric_attr == NULL)
			goto fail;
		dup->fabric_attr->name = NULL;
		dup->fabric_attr->prov_name = NULL;
		if (info->fabric_attr->name != NULL) {
			dup->fabric_attr->name = strdup(info->fabric_attr->name);
			if (dup->fabric_attr->name == NULL)
				goto fail;
		}
		if (info->fabric_attr->prov_name != NULL) {
			dup->fabric_attr->prov_name = strdup(info->fabric_attr->prov_name);
			if (dup->fabric_attr->prov_name == NULL)
				goto fail;
		}
	}
	return dup;

fail:
	fi_freeinfo(dup);
	return NULL;
}
CURRENT_SYMVER(fi_dupinfo_, fi_dupinfo);

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
int DEFAULT_SYMVER_PRE(fi_fabric)(struct fi_fabric_attr *attr,
		struct fid_fabric **fabric, void *context)
{
	struct ofi_prov *prov;
	const char *top_name;
	size_t len;
	int ret;

	if (!attr || !attr->prov_name || !attr->name)
		return -FI_EINVAL;

	if (!ofi_init)
		fi_ini();

	top_name = ofi_util_name(attr->prov_name, &len);
	if (!top_name)
		top_name = ofi_core_name(attr->prov_name, &len);

	if (!top_name)
		return -FI_EINVAL;

	prov = ofi_getprov(top_name, len);
	if (!prov || !prov->provider || !prov->provider->fabric)
		return -FI_ENODEV;

	ret = prov->provider->fabric(attr, fabric, context);
	if (!ret && FI_VERSION_GE(prov->provider->fi_version, FI_VERSION(1, 5)))
		(*fabric)->api_version = attr->api_version;
	return ret;
}
CURRENT_SYMVER(fi_fabric_, fi_fabric);

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
uint32_t DEFAULT_SYMVER_PRE(fi_version)(void)
{
	return FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION);
}
DEFAULT_SYMVER(fi_version_, fi_version, FABRIC_1.0);

static const char *const errstr[] = {
	[FI_EOTHER - FI_ERRNO_OFFSET] = "Unspecified error",
	[FI_ETOOSMALL - FI_ERRNO_OFFSET] = "Provided buffer is too small",
	[FI_EOPBADSTATE - FI_ERRNO_OFFSET] = "Operation not permitted in current state",
	[FI_EAVAIL - FI_ERRNO_OFFSET]  = "Error available",
	[FI_EBADFLAGS - FI_ERRNO_OFFSET] = "Flags not supported",
	[FI_ENOEQ - FI_ERRNO_OFFSET] = "Missing or unavailable event queue",
	[FI_EDOMAIN - FI_ERRNO_OFFSET] = "Invalid resource domain",
	[FI_ENOCQ - FI_ERRNO_OFFSET] = "Missing or unavailable completion queue",
	[FI_ECRC - FI_ERRNO_OFFSET] = "CRC error",
	[FI_ETRUNC - FI_ERRNO_OFFSET] = "Truncation error",
	[FI_ENOKEY - FI_ERRNO_OFFSET] = "Required key not available",
	[FI_ENOAV - FI_ERRNO_OFFSET] = "Missing or unavailable address vector",
	[FI_EOVERRUN - FI_ERRNO_OFFSET] = "Queue has been overrun",
};

__attribute__((visibility ("default"),EXTERNALLY_VISIBLE))
const char *DEFAULT_SYMVER_PRE(fi_strerror)(int errnum)
{
	if (errnum < FI_ERRNO_OFFSET)
		return strerror(errnum);
	else if (errnum < FI_ERRNO_MAX)
		return errstr[errnum - FI_ERRNO_OFFSET];
	else
		return errstr[FI_EOTHER - FI_ERRNO_OFFSET];
}
DEFAULT_SYMVER(fi_strerror_, fi_strerror, FABRIC_1.0);
