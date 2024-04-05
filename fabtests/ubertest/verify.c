/*
 * Copyright (c) 2017 Intel Corporation.  All rights reserved.
 * Copyright (c) 2016, Cisco Systems, Inc. All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

#include "ofi_atomic.h"
#include "fabtest.h"

int ft_sync_fill_bufs(size_t size)
{
	int ret;
	ft_sock_sync(sock, 0);

	if (test_info.caps & FI_ATOMIC) {
		(void)ft_fill_atomic(ft_tx_ctrl.buf, ft_atom_ctrl.count, ft_atom_ctrl.datatype);
		(void)ft_fill_atomic(ft_mr_ctrl.buf, ft_atom_ctrl.count, ft_atom_ctrl.datatype);
		memcpy(ft_atom_ctrl.orig_buf, ft_mr_ctrl.buf, size);
		memcpy(ft_tx_ctrl.cpy_buf, ft_tx_ctrl.buf, size);
	} else if (is_read_func(test_info.class_function)) {
		ret = ft_fill_buf(ft_mr_ctrl.buf, size);
		if (ret)
			return ret;
	} else {
		ret = ft_fill_buf(ft_tx_ctrl.buf, size);
		if (ret)
			return ret;

		ret = ft_hmem_copy_from(opts.iface, opts.device,
					ft_tx_ctrl.cpy_buf,
					ft_tx_ctrl.buf, size);
		if (ret)
			return ret;
	}

	ft_sock_sync(sock, 0);

	return 0;
}

int ft_verify_bufs()
{
	char *compare_buf;
	size_t compare_size;
	enum ft_atomic_opcodes opcode;

	if (test_info.caps & FI_ATOMIC) {
		if (is_compare_func(test_info.class_function))
			opcode = FT_ATOMIC_COMPARE;
		else if (is_fetch_func(test_info.class_function))
			opcode = FT_ATOMIC_FETCH;
		else
			opcode = FT_ATOMIC_BASE;

		return ft_check_atomic(opcode, ft_atom_ctrl.op,
				ft_atom_ctrl.datatype, ft_tx_ctrl.cpy_buf,
				ft_atom_ctrl.orig_buf, ft_mr_ctrl.buf,
				ft_atom_ctrl.comp_buf, ft_atom_ctrl.res_buf,
				ft_atom_ctrl.count);
	}

	if (test_info.caps & FI_RMA) {
		compare_size = ft_tx_ctrl.rma_msg_size;
		if (is_read_func(test_info.class_function))
			compare_buf = (char *) ft_tx_ctrl.buf;
		else
			compare_buf = (char *) ft_mr_ctrl.buf;
	} else {
		compare_size = ft_tx_ctrl.msg_size;
		compare_buf = (char *) ft_rx_ctrl.buf;
	}

	return ft_check_buf(compare_buf, compare_size);
}

void ft_verify_comp(void *buf)
{
	struct fi_cq_err_entry *comp = (struct fi_cq_err_entry *) buf;

	switch (ft_rx_ctrl.cq_format) {
	case FI_CQ_FORMAT_TAGGED:
		if ((test_info.test_class & FI_TAGGED) &&
		    (comp->tag != ft_tx_ctrl.check_tag++))
			return;
		/* fall through */
	case FI_CQ_FORMAT_DATA:
		if (test_info.msg_flags & FI_REMOTE_CQ_DATA ||
		    is_data_func(test_info.class_function)) {
			if (!(comp->flags & FI_REMOTE_CQ_DATA))
				return;
			comp->flags &= ~FI_REMOTE_CQ_DATA;
			if (comp->data != ft_tx_ctrl.remote_cq_data)
				return;
		}
		/* fall through */
	case FI_CQ_FORMAT_MSG:
		if (((test_info.test_class & FI_MSG) &&
		    (comp->flags != (FI_MSG | FI_RECV))) ||
		    ((test_info.test_class & FI_TAGGED) &&
		    (comp->flags != (FI_TAGGED | FI_RECV))))
			return;
		if ((test_info.test_class & (FI_MSG | FI_TAGGED)) &&
		    (comp->len != ft_tx_ctrl.msg_size))
			return;
		/* fall through */
	case FI_CQ_FORMAT_CONTEXT:
		if (test_info.test_class & (FI_MSG | FI_TAGGED)) {
			ft_rx_ctrl.check_ctx = (++ft_rx_ctrl.check_ctx >=
			    ft_rx_ctrl.max_credits) ? 0 : ft_rx_ctrl.check_ctx;
			if (comp->op_context != &(ft_rx_ctrl.ctx[ft_rx_ctrl.check_ctx]))
				return;
		}
		break;
	default:
		return;
	}
	ft_ctrl.verify_cnt++;
}
