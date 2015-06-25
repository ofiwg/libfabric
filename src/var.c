/*
 * Copyright (c) 2015, Cisco Systems, Inc. All rights reserved.
 * Copyright (c) 2015, Intel Corp., Inc.  All rights reserved.
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
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_log.h>

#include "fi.h"


struct fi_param_entry {
	const struct fi_provider *provider;
	char *name;
	char *help_string;
	char *env_var_name;
	struct fi_param_entry *next;
};

static struct fi_param_entry *param_list;

static int fi_param_get(const struct fi_provider *provider, const char *param_name,
		char **value)
{
	struct fi_param_entry *v;

	// Check for bozo cases
	if (param_name == NULL || value == NULL) {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to read %s variable: provider coding error\n",
			param_name);
		return -FI_EINVAL;
	}

	for (v = param_list; v; v = v->next) {
		if (v->provider == provider &&
		    strcmp(v->name, param_name) == 0) {
			*value = getenv(v->env_var_name);
			return FI_SUCCESS;
		}
	}

	FI_DBG(provider, FI_LOG_CORE,
		"Failed to read %s variable: was not registered\n", param_name);
	return -FI_ENOENT;
}

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_getparams)(struct fi_param **params, int *count)
{
	struct fi_param *vhead = NULL;
	struct fi_param_entry *v;
	int ret, len = 0, i;
	char *tmp = NULL;

	// just get a count
	for (v = param_list; v; v = v->next, ++len)
		continue;

	if (len == 0)
		goto out;

	// last extra entry will be all NULL
	vhead = calloc(len + 1, sizeof (*vhead));
	if (!vhead)
		return -FI_ENOMEM;

	for (v = param_list, i = 0; v; v = v->next, ++i, tmp = NULL) {
		vhead[i].name = strdup(v->env_var_name);
		vhead[i].help_string = strdup(v->help_string);

		ret = fi_param_get(v->provider, v->name, &tmp);
		if (ret == FI_SUCCESS && tmp)
			vhead[i].value = strdup(tmp);

		if (!vhead[i].name || !vhead[i].help_string) {
			fi_freeparams(vhead);
			return -FI_ENOMEM;
		}
	}

out:
	*count = len;
	*params = vhead;
	return FI_SUCCESS;
}
DEFAULT_SYMVER(fi_getparams_, fi_getparams);

__attribute__((visibility ("default")))
void DEFAULT_SYMVER_PRE(fi_freeparams)(struct fi_param *params)
{
	for (int i = 0; params[i].name; ++i) {
		free((void*) params[i].name);
		free((void*) params[i].help_string);
		free((void*) params[i].value);
	}
	free(params);
}
DEFAULT_SYMVER(fi_freeparams_, fi_freeparams);

static void fi_free_param(struct fi_param_entry *param)
{
	free(param->name);
	free(param->help_string);
	free(param->env_var_name);
	free(param);
}

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_param_register)(const struct fi_provider *provider,
		const char *param_name, const char *help_string)
{
	int i, ret;
	struct fi_param_entry *v;

	// Check for bozo cases
	if (provider == NULL || param_name == NULL || help_string == NULL ||
	    *help_string == '\0') {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: provider coding error\n",
			param_name);
		return -FI_EINVAL;
	}

	v = calloc(1, sizeof(*v));
	if (!v) {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: ENOMEM\n", param_name);
		return -FI_ENOMEM;
	}

	v->provider = provider;
	v->name = strdup(param_name);
	ret = asprintf(&v->help_string, "%s: %s", provider->name, help_string);
	if (ret < 0)
		v->help_string = NULL;
	ret = asprintf(&v->env_var_name, "FI_%s_%s", provider->name, param_name);
	if (ret < 0)
		v->env_var_name = NULL;

	if (!v->name || !v->help_string || !v->env_var_name) {
		fi_free_param(v);
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: ENOMEM\n", param_name);
		return -FI_ENOMEM;
	}

	for (i = 0; v->env_var_name[i]; ++i)
		v->env_var_name[i] = toupper(v->env_var_name[i]);

	v->next = param_list;
	param_list = v;

	FI_INFO(provider, FI_LOG_CORE, "registered var %s\n", param_name);

	return FI_SUCCESS;
}
DEFAULT_SYMVER(fi_param_register_, fi_param_register);

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_param_get_str)(struct fi_provider *provider,
		const char *param_name, char **value)
{
	int ret;

	ret = fi_param_get(provider, param_name, value);
	if (ret == FI_SUCCESS) {
		if (*value) {
			FI_INFO(provider, FI_LOG_CORE,
				"read string var %s=%s\n", param_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read string var %s=<not set>\n", param_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}
DEFAULT_SYMVER(fi_param_get_str_, fi_param_get_str);

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_param_get_int)(struct fi_provider *provider,
		const char *param_name, int *value)
{
	int ret;
	char *str_value;

	ret = fi_param_get(provider, param_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			*value = atoi(str_value);
			FI_INFO(provider, FI_LOG_CORE,
				"read int var %s=%d\n", param_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read int var %s=<not set>\n", param_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}
DEFAULT_SYMVER(fi_param_get_int_, fi_param_get_int);

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_param_get_long)(struct fi_provider *provider,
		const char *param_name, long *value)
{
	int ret;
	char *str_value;

	ret = fi_param_get(provider, param_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			*value = strtol(str_value, NULL, 10);
			FI_INFO(provider, FI_LOG_CORE,
				"read long var %s=%ld\n", param_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read long var %s=<not set>\n", param_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}
DEFAULT_SYMVER(fi_param_get_long_, fi_param_get_long);

__attribute__((visibility ("default")))
int DEFAULT_SYMVER_PRE(fi_param_get_bool)(struct fi_provider *provider,
		const char *param_name, int *value)
{
	int ret;
	char *str_value;

	ret = fi_param_get(provider, param_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			if (strcmp(param_name, "0") == 0 ||
				strcasecmp(param_name, "false") == 0 ||
				strcasecmp(param_name, "no") == 0 ||
				strcasecmp(param_name, "off") == 0) {
				*value = 0;
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=false\n", param_name);
				ret = FI_SUCCESS;
			} else if (strcmp(param_name, "1") == 0 ||
				strcasecmp(param_name, "true") == 0 ||
				strcasecmp(param_name, "yes") == 0 ||
				strcasecmp(param_name, "on") == 0) {
				*value = 1;
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=true\n", param_name);
				ret = FI_SUCCESS;
			} else {
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=<unknown> (%s)\n",
					param_name, str_value);
				ret = -FI_EINVAL;
			}
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read boolean var %s=<not set>\n", param_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}
DEFAULT_SYMVER(fi_param_get_bool_, fi_param_get_bool);

void fi_param_fini(void)
{
	struct fi_param_entry *v, *v2;

	for (v = param_list; v; v = v2) {
		v2 = v->next;
		fi_free_param(v);
	}
}
