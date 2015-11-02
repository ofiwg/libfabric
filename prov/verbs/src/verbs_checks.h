/*
 * Copyright (c) 2013-2015 Intel Corporation, Inc.  All rights reserved.
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

#ifndef _VERBS_CHECKS_H
#define _VERBS_CHECKS_H

struct fi_info;
struct fi_ep_attr;
struct fi_fabric_attr;
struct fi_rx_attr;
struct fi_tx_attr;
struct fi_domain_attr;

int fi_ibv_check_fabric_attr(const struct fi_fabric_attr *attr,
		const struct fi_info *info);

int fi_ibv_check_ep_attr(const struct fi_ep_attr *attr,
		const struct fi_info *info);

int fi_ibv_check_rx_attr(const struct fi_rx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info);

int fi_ibv_check_tx_attr(const struct fi_tx_attr *attr,
		const struct fi_info *hints, const struct fi_info *info);

int fi_ibv_check_domain_attr(const struct fi_domain_attr *attr,
		const struct fi_info *info);

#endif /* _VERBS_CHECKS_H */