/*
 * Copyright (c) 2014 Intel Corporation, Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenIB.org BSD license below:
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

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <stdlib.h>
#include <string.h>

#include "sock.h"


//static struct fi_ops sock_av_fi_ops = {
//	.size = sizeof(struct fi_ops),
//	.close = sock_av_close,
//	.bind = sock_av_bind,
//};
//
//static struct fi_ops_av sock_am_ops = {
//	.size = sizeof(struct fi_ops_av),
//	.insert = sock_am_insert,
//	.remove = sock_am_remove,
//	.lookup = sock_am_lookup,
//	.straddr = sock_am_straddr
//};
//
//static struct fi_ops_av sock_av_ops = {
//	.size = sizeof(struct fi_ops_av),
//	.insert = sock_av_insert,
//	.remove = sock_av_remove,
//	.lookup = sock_av_lookup,
//	.straddr = sock_av_straddr
//};

int sock_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		 struct fid_av **av, void *context)
{
	return -FI_ENOSYS; /* TODO */
}
