/*
 * Copyright (c) 2015, Cisco Systems, Inc. All rights reserved.
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

#include <errno.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include <rdma/fi_errno.h>
#include "fi.h"
#include "fi_log.h"

static struct log_providers provider_list;

static const char * const subsystems[] = {
	[FI_FABRIC] = "fabric",
	[FI_DOMAIN] = "domain",
	[FI_EP_CM] = "ep_cm",
	[FI_EP_DM] = "ep_dm",
	[FI_AV] = "av",
	[FI_CQ] = "cq",
	[FI_EQ] = "eq",
	[FI_MR] = "mr"
};

static const char * const log_levels[] = {
	[FI_LOG_WARN] = "warn",
	[FI_LOG_TRACE] = "trace",
	[FI_LOG_INFO] = "info"
};

/*
 * By default support logging from all providers and subsystems. This will
 * change if the relevant environment variables are set.
 */
uint64_t log_mask = FI_PROV_MASK | FI_SUBSYS_MASK;

typedef void found_func_t(char *ptr);

static char *alloc_env(char *name)
{
	char *env = getenv(name);
	char *ret = NULL;

	if (!env || strlen(env) == 0)
		return ret;

	ret = strdup(env);
	if (!ret)
		fprintf(stderr, "%s: %s", __func__, strerror(errno));

	return ret;
}

static int compare_names(struct slist_entry *item, const void *arg)
{
	struct provider_parameter *target =
	    container_of(item, struct provider_parameter, entry);
	return !strcasecmp(target->name, arg);
}

void set_provider(const char *prov_name, int position)
{
	struct slist_entry *found;
	struct provider_parameter *item;

	found = slist_remove_first_match(&provider_list.names, compare_names,
					 prov_name);

	if (found) {
		if (provider_list.negated)
			log_mask &= ~FI_EXPAND(position, FI_PROV_OFFSET);
		else
			log_mask |= FI_EXPAND(position, FI_PROV_OFFSET);

		item = container_of(found, struct provider_parameter, entry);
		free(item->name);
		free(item);
	}
}

/*
 * Negatable describes whether the function should check for the caret symbol
 * (^) at the beginning of the given environment variable. Tokenize string input
 * and run given action function on each token found.
 *
 * On completion negated will contain whether the list was marked negated.
 */
static void find_options(bool negatable, bool *negated, char *input,
			 found_func_t *action)
{
	char *save_ptr = NULL;
	char *token = NULL;

	if (!input)
		return;

	if (negatable) {
		if (*input == '^') {
			input++;
			*negated = true;
		} else {
			*negated = false;
		}
	}

	for (token = strtok_r(input, ",", &save_ptr); token;
	     token = strtok_r(NULL, ",", &save_ptr)) {
		action(token);
	}
}

static void handle_log_option(char *name)
{
	bool found = false;
	size_t i;

	for (i = 0; i < ARRAY_SIZE(log_levels); i++) {
		if (log_levels[i] && !strcasecmp(log_levels[i], name)) {
			found = true;
			break;
		}
	}

	if (found) {
		log_mask |= i;
	} else {
		fprintf(stderr,
			"%s: invalid option \"%s\" for env FI_LOG_LEVEL.\n",
			PACKAGE, name);
	}
}

static void handle_subsys_option(char *name)
{
	bool found = false;
	size_t i;

	for (i = 0; i < ARRAY_SIZE(subsystems); i++) {
		if (subsystems[i] && !strcasecmp(subsystems[i], name)) {
			found = true;
			break;
		}
	}

	if (found) {
		log_mask |= FI_EXPAND(i, FI_SUBSYS_OFFSET);
	} else {
		fprintf(stderr,
			"%s: invalid option \"%s\" for env FI_LOG_SUBSYSTEMS.\n",
			PACKAGE, name);
	}
}

static void handle_provider_option(char *name)
{
	struct provider_parameter *param = calloc(1, sizeof(*param));

	if (!param) {
		fprintf(stderr, "%s: %s", __func__, strerror(errno));
		return;
	}

	/*
	 * Insert into provider_list for lookup during registration.
	 */
	param->name = strdup(name);
	slist_insert_tail(&param->entry, &provider_list.names);
}

static void get_log_level(void)
{
	char *requested;

	requested = alloc_env("FI_LOG_LEVEL");
	if (!requested)
		return;

	find_options(false, NULL, requested, handle_log_option);
	free(requested);
}

static void get_providers(void)
{
	char *requested;

	requested = alloc_env("FI_LOG_PROV");
	if (!requested)
		return;

	find_options(true, &provider_list.negated, requested,
		     handle_provider_option);

	if (!provider_list.negated)
		log_mask &= ~FI_PROV_MASK;

	free(requested);
}

static void get_subsystems(void)
{
	char *requested;
	bool negated = false;

	requested = alloc_env("FI_LOG_SUBSYSTEMS");
	if (!requested)
		return;

	log_mask &= ~FI_SUBSYS_MASK;

	find_options(true, &negated, requested, handle_subsys_option);
	if (negated)
		log_mask ^= FI_SUBSYS_MASK;

	free(requested);
}

void fi_log_init(void)
{
	slist_init(&provider_list.names);

	get_log_level();
	get_providers();
	get_subsystems();
}

void __attribute__((destructor)) fi_log_fini(void)
{
	struct slist_entry *current, *next;
	struct provider_parameter *parameter;

	/*
	 * Free leftover provider names given in environment variable.
	 */
	for (current = provider_list.names.head; current; current = next) {
		next = current->next;
		parameter = container_of(current, struct provider_parameter,
					 entry);
		free(parameter->name);
		free(parameter);
	}
}

void fi_err_impl(const char *prov, int subsystem, const char *fmt,
		...)
{
	char buf[1024];
	int size;

	va_list vargs;

	size = snprintf(buf, sizeof(buf), "%s:%s:%s: ", PACKAGE, prov,
			subsystems[subsystem]);

	va_start(vargs, fmt);
	vsnprintf(buf + size, sizeof(buf) - size, fmt, vargs);
	va_end(vargs);

	fprintf(stderr, "%s", buf);
}

void fi_log_impl(const char *prov, int level, int subsystem, const
		char *func, int line, const char *fmt, ...)
{
	char buf[1024];
	int size;

	va_list vargs;

	size = snprintf(buf, sizeof(buf), "%s:%s:%s:%s():%d<%s> ", PACKAGE,
			prov, subsystems[subsystem], func, line,
			log_levels[level]);

	va_start(vargs, fmt);
	vsnprintf(buf + size, sizeof(buf) - size, fmt, vargs);
	va_end(vargs);

	fprintf(stderr, "%s", buf);
}

void fi_debug_impl(const char *prov, int subsystem, const char *func,
		int line, const char *fmt, ...)
{
	char buf[1024];
	int size;

	va_list vargs;

	size = snprintf(buf, sizeof(buf), "%s:%s:%s:%s():%d<DBG> ", PACKAGE,
			prov, subsystems[subsystem], func, line);

	va_start(vargs, fmt);
	vsnprintf(buf + size, sizeof(buf) - size, fmt, vargs);
	va_end(vargs);

	fprintf(stderr, "%s", buf);
}
