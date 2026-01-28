/*
 * SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only
 *
 * Copyright (c) 2018-2024 Hewlett Packard Enterprise Development LP
 */

#ifndef _CXIP_CURL_H_
#define _CXIP_CURL_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Type definitions */
struct cxip_curl_handle {
	long status; // HTTP status, 0 for no server, -1 busy
	const char *endpoint; // HTTP server endpoint address
	const char *request; // HTTP request data
	const char *response; // HTTP response data, NULL until complete
	curlcomplete_t usrfunc; // user completion function
	void *usrptr; // user function argument
	void *recv; // opaque
	void *headers; // opaque
};

/* Function declarations */
int cxip_curl_init(void);

void cxip_curl_fini(void);

const char *cxip_curl_opname(enum curl_ops op);

int cxip_curl_perform(const char *endpoint, const char *request,
		      const char *sessionToken, size_t rsp_init_size,
		      enum curl_ops op, bool verbose, curlcomplete_t usrfunc,
		      void *usrptr);

int cxip_curl_progress(struct cxip_curl_handle **handleptr);

void cxip_curl_free(struct cxip_curl_handle *handle);

enum json_type cxip_json_obj(const char *desc, struct json_object *jobj,
			     struct json_object **jval);

int cxip_json_bool(const char *desc, struct json_object *jobj, bool *val);

int cxip_json_int(const char *desc, struct json_object *jobj, int *val);

int cxip_json_int64(const char *desc, struct json_object *jobj, int64_t *val);

int cxip_json_double(const char *desc, struct json_object *jobj, double *val);

int cxip_json_string(const char *desc, struct json_object *jobj,
		     const char **val);

struct json_object *cxip_json_tokener_parse(const char *str);

int cxip_json_object_put(struct json_object *obj);

#endif /* _CXIP_CURL_H_ */
