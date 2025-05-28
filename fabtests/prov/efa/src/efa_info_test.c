/*
 * Copyright (c) 2022, Amazon.com, Inc.  All rights reserved.
 *
 * This software is available to you under the BSD license
 * below:
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
 * This test ensures that the fabric id we get from the efa provider will
 * always be "efa".
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <shared.h>

int main(int argc, char **argv)
{
	int ret = 0, op;
	struct fi_info *info;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;


	while ((op = getopt(argc, argv, "f:p:")) != -1) {
		switch (op) {
		case 'f':
			hints->fabric_attr->name = strdup(optarg);
			break;
		case 'p':
			hints->fabric_attr->prov_name = strdup(optarg);
			break;
		default:
			printf("Unknown option");
			return EXIT_FAILURE;
		}
	}

	/* Set all mode bits to enable all providers and fabrics */
	hints->mode = ~0;
	hints->domain_attr->mode = ~0;
	hints->domain_attr->mr_mode = ~3; /* deprecated: (FI_MR_BASIC | FI_MR_SCALABLE) */

	ret = ft_init();
	if (ret) {
		FT_PRINTERR("ft_init", -ret);
		goto out;
	}
	ret = ft_getinfo(hints, &fi);
	if (ret) {
		FT_PRINTERR("ft_getinfo", -ret);
		goto out;
	}

	info = fi;
	while (NULL != info) {
		/* If a fabric name is explicitly provided, only info objects with that fabric must be returned */
		if (hints->fabric_attr->name) {
			if (0 != strcmp(info->fabric_attr->name, hints->fabric_attr->name)) {
				ret = EXIT_FAILURE;
				goto out;
			}
		} else {
			/* If no fabric name is provided, both efa and efa-direct info objects can be returned */
			if (0 != strcmp(info->fabric_attr->name, "efa") && 0 != strcmp(info->fabric_attr->name, "efa-direct")) {
				ret = EXIT_FAILURE;
				goto out;
			}
		}
		info = info->next;
	}

out:
	fi_freeinfo(hints);
	fi_freeinfo(fi);
	return ft_exit_code(ret);
}
