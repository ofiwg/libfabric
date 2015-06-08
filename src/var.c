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
#include <rdma/fi_var.h>
#include <rdma/fi_log.h>

struct fi_var_t {
	const struct fi_provider *provider;
	char *var_name;
	char *help_string;

	char *env_var_name;

	struct fi_var_t *next;
};

static struct fi_var_t *fi_vars = NULL;


int fi_var_register(const struct fi_provider *provider, const char *var_name,
		const char *help_string)
{
	int i;
	struct fi_var_t *v;

	// Check for bozo cases
	if (provider == NULL || var_name == NULL || help_string == NULL ||
		*help_string == '\0') {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: provider coding error\n",
			var_name);
		return -FI_EINVAL;
	}

	v = calloc(1, sizeof(*v));
	if (!v) {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: ENOMEM\n", var_name);
		return -FI_ENOMEM;
	}

	v->provider = provider;
	v->var_name = strdup(var_name);
	v->help_string = strdup(help_string);
	asprintf(&v->env_var_name, "FI_%s_%s",
		v->provider->name,
		v->var_name);
	if (!v->var_name || !v->help_string || !v->env_var_name) {
		free(v);
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to register %s variable: ENOMEM\n", var_name);
		return -FI_ENOMEM;
	}

	for (i = 0; v->env_var_name[i]; ++i)
		v->env_var_name[i] = toupper(v->env_var_name[i]);

	v->next = fi_vars;
	fi_vars = v;

	FI_INFO(provider, FI_LOG_CORE, "registered var %s\n", var_name);

	return FI_SUCCESS;
}

static int fi_var_get(struct fi_provider *provider, const char *var_name,
		char **value)
{
	struct fi_var_t *v;

	// Check for bozo cases
	if (var_name == NULL || value == NULL) {
		FI_DBG(provider, FI_LOG_CORE,
			"Failed to read %s variable: provider coding error\n",
			var_name);
		return -FI_EINVAL;
	}

	for (v = fi_vars; v; v = v->next) {
		if (strcmp(v->provider->name, provider->name) == 0 &&
			strcmp(v->var_name, var_name) == 0) {
			*value = getenv(v->env_var_name);

			return FI_SUCCESS;
		}
	}

	FI_DBG(provider, FI_LOG_CORE,
		"Failed to read %s variable: was not registered\n", var_name);
	return -FI_ENOENT;
}

int fi_var_get_str(struct fi_provider *provider, const char *var_name,
		char **value)
{
	int ret;

	ret = fi_var_get(provider, var_name, value);
	if (ret == FI_SUCCESS) {
		if (*value) {
			FI_INFO(provider, FI_LOG_CORE,
				"read string var %s=%s\n", var_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read string var %s=<not set>\n", var_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}

int fi_var_get_int(struct fi_provider *provider, const char *var_name,
		int *value)
{
	int ret;
	char *str_value;

	ret = fi_var_get(provider, var_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			*value = atoi(str_value);
			FI_INFO(provider, FI_LOG_CORE,
				"read int var %s=%d\n", var_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read int var %s=<not set>\n", var_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}

int fi_var_get_long(struct fi_provider *provider, const char *var_name,
		long *value)
{
	int ret;
	char *str_value;

	ret = fi_var_get(provider, var_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			*value = strtol(str_value, NULL, 10);
			FI_INFO(provider, FI_LOG_CORE,
				"read long var %s=%ld\n", var_name, *value);
			ret = FI_SUCCESS;
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read long var %s=<not set>\n", var_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}

int fi_var_get_bool(struct fi_provider *provider, const char *var_name,
		int *value)
{
	int ret;
	char *str_value;

	ret = fi_var_get(provider, var_name, &str_value);
	if (ret == FI_SUCCESS) {
		if (str_value) {
			if (strcmp(var_name, "0") == 0 ||
				strcasecmp(var_name, "false") == 0 ||
				strcasecmp(var_name, "no") == 0 ||
				strcasecmp(var_name, "off") == 0) {
				*value = 0;
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=false\n", var_name);
				ret = FI_SUCCESS;
			} else if (strcmp(var_name, "1") == 0 ||
				strcasecmp(var_name, "true") == 0 ||
				strcasecmp(var_name, "yes") == 0 ||
				strcasecmp(var_name, "on") == 0) {
				*value = 1;
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=true\n", var_name);
				ret = FI_SUCCESS;
			} else {
				FI_INFO(provider, FI_LOG_CORE,
					"read boolean var %s=<unknown> (%s)\n", var_name, str_value);
				ret = -FI_EINVAL;
			}
		} else {
			FI_INFO(provider, FI_LOG_CORE,
				"read boolean var %s=<not set>\n", var_name);
			ret = -FI_ENODATA;
		}
	}

	return ret;
}

void fi_var_fini(void)
{
	struct fi_var_t *v, *v2;

	for (v = fi_vars; v; v = v2) {
		free(v->var_name);
		free(v->help_string);
		free(v->env_var_name);

		v2 = v->next;
		free(v);
	}
}
