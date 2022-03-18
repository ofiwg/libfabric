/*
 * Copyright (c) 2022 ORNL. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL); Version 2, available from the file
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

#ifndef OFI_LINKX_H
#define OFI_LINKX_H

/* ofi_create_link()
 *   prov_list (IN): number of providers to link
 *   fabric (OUT): linkx fabric which abstracts the bond
 *   caps (IN): bond capabilities requested
 *   context (IN): user context to store.
 *
 * The LINKx provider is not inserted directly on the list
 * of core providers. In that sense, it's a special provider
 * that only gets returned on a call of fi_link(), if that
 * function determines that there are multiple providers to link.
 *
 * ofi_create_link() binds the core provider endpoints and returns
 * the LINKx fabric which abstracts away these provider endpoints.
 */
int ofi_create_link(struct fi_info *prov_list, struct fid_fabric **fabric,
					uint64_t caps, void *context);

/*
 * ofi_finish_link()
 *   Uninitialize and cleanup all the core providers
 */
void ofi_link_fini(void);

#endif /* OFI_LINKX_H */
