/*
 * Copyright (c) 2018-2022 Amazon.com, Inc. or its affiliates. All rights reserved.
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

#include "efa.h"

int efa_base_ep_bind_av(struct efa_base_ep *base_ep, struct efa_av *av)
{
	/*
	 * Binding multiple endpoints to a single AV is currently not
	 * supported.
	 */
	if (av->base_ep) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector already has endpoint bound to it.\n");
		return -FI_ENOSYS;
	}
	if (base_ep->domain != av->domain) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector doesn't belong to same domain as EP.\n");
		return -FI_EINVAL;
	}
	if (base_ep->av) {
		EFA_WARN(FI_LOG_EP_CTRL,
			 "Address vector already bound to EP.\n");
		return -FI_EINVAL;
	}

	base_ep->av = av;
	base_ep->av->base_ep = base_ep;

	return 0;
}
