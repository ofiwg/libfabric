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

#ifndef _FI_VAR_H_
#define _FI_VAR_H_

#include <rdma/fi_prov.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Registers a configuration variable for use with libfabric.
 *
 * Example: fi_register_var(provider, "foo", "Very important help
 * string");
 *
 * This registers the configuration variable "foo" in the specified
 * provider.
 *
 * The help string cannot be NULL or empty.
 *
 * The var_name and help_string parameters will be copied internally;
 * they can be freed upon return from fi_var_register().
 */
int fi_var_register(const struct fi_provider *provider, const char *var_name,
		const char *help_string);

/* Get the string value of a configuration variable.
 *
 * Currently, configuration variables will only be read from the
 * environment.  The environment variable names will be of the form
 * upper_case(FI_<provider_name>_<var_name>).
 *
 * Someday this call could be expanded to also check config files.
 *
 * If the variable was previously registered and the user set a value,
 * FI_SUCCESS is returned and (*value) points to the user's
 * \0-terminated string value.  NOTE: The caller should not modify or
 * free (*value).
 *
 * If the variable name was previously registered, but the user did
 * not set a value, -FI_ENODATA is returned and the value of (*value)
 * is unchanged.
 *
 * If the variable name was not previously registered via
 * fi_var_register(), -FI_ENOENT will be returned and the value of
 * (*value) is unchanged.
 */
int fi_var_get_str(struct fi_provider *provider, const char *var_name,
	char **value);

/* Similar to fi_var_get_str(), but the value is converted to an int.
 * No checking is done to ensure that the value the user set is
 * actually an integer -- atoi() is simply called on whatever value
 * the user sets.
 */
int fi_var_get_int(struct fi_provider *provider, const char *var_name,
	int *value);

/* Similar to fi_var_get_str(), but the value is converted to a long.
 * No checking is done to ensure that the value the user set is
 * actually an integer -- strtol() is simply called on whatever value
 * the user sets.
 */
int fi_var_get_long(struct fi_provider *provider, const char *var_name,
	long *value);

/* Similar to fi_var_get_str(), but the value is converted to an
 * boolean (0 or 1) and returned in an int.  Accepted user values are:
 *
 * 0, off, false, no: these will all return 0 in (*value)
 * 1, on, true, yes: these will all return 1 in (*value)
 *
 * Any other user value will return -FI_EINVAL, and (*value) will be
 * unchanged.
 */
int fi_var_get_bool(struct fi_provider *provider, const char *var_name,
	int *value);

/* Clean up any resources used by the var system
 */
void fi_var_fini(void);

#ifdef __cplusplus
}
#endif

#endif /*_FI_VAR_H_ */
