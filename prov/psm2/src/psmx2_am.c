/*
 * Copyright (c) 2013-2017 Intel Corporation. All rights reserved.
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

#include "psmx2.h"

#define PSMX2_AM_MAX_TRX_CTXT PSMX2_MAX_TRX_CTXT
static struct {
	struct psmx2_trx_ctxt *trx_ctxts[PSMX2_AM_MAX_TRX_CTXT];
	psm2_am_handler_fn_t rma_handlers[PSMX2_AM_MAX_TRX_CTXT];
	psm2_am_handler_fn_t atomic_handlers[PSMX2_AM_MAX_TRX_CTXT];
	fastlock_t lock;
	int cnt;
} psmx2_am_global;

#define DEFINE_AM_HANDLERS(INDEX) \
	static int psmx2_am_rma_handler_##INDEX( \
			psm2_am_token_t token, psm2_amarg_t *args, \
			int nargs, void *src, uint32_t len) \
	{ \
		return psmx2_am_rma_handler_ext( \
				token, args, nargs, src, len, \
				psmx2_am_global.trx_ctxts[INDEX]); \
	} \
	\
	static int psmx2_am_atomic_handler_##INDEX( \
			psm2_am_token_t token, psm2_amarg_t *args, \
			int nargs, void *src, uint32_t len) \
	{ \
		return psmx2_am_atomic_handler_ext( \
				token, args, nargs, src, len, \
				psmx2_am_global.trx_ctxts[INDEX]); \
	}

DEFINE_AM_HANDLERS(0)
DEFINE_AM_HANDLERS(1)
DEFINE_AM_HANDLERS(2)
DEFINE_AM_HANDLERS(3)
DEFINE_AM_HANDLERS(4)
DEFINE_AM_HANDLERS(5)
DEFINE_AM_HANDLERS(6)
DEFINE_AM_HANDLERS(7)
DEFINE_AM_HANDLERS(8)
DEFINE_AM_HANDLERS(9)
DEFINE_AM_HANDLERS(10)
DEFINE_AM_HANDLERS(11)
DEFINE_AM_HANDLERS(12)
DEFINE_AM_HANDLERS(13)
DEFINE_AM_HANDLERS(14)
DEFINE_AM_HANDLERS(15)
DEFINE_AM_HANDLERS(16)
DEFINE_AM_HANDLERS(17)
DEFINE_AM_HANDLERS(18)
DEFINE_AM_HANDLERS(19)
DEFINE_AM_HANDLERS(20)
DEFINE_AM_HANDLERS(21)
DEFINE_AM_HANDLERS(22)
DEFINE_AM_HANDLERS(23)
DEFINE_AM_HANDLERS(24)
DEFINE_AM_HANDLERS(25)
DEFINE_AM_HANDLERS(26)
DEFINE_AM_HANDLERS(27)
DEFINE_AM_HANDLERS(28)
DEFINE_AM_HANDLERS(29)
DEFINE_AM_HANDLERS(30)
DEFINE_AM_HANDLERS(31)
DEFINE_AM_HANDLERS(32)
DEFINE_AM_HANDLERS(33)
DEFINE_AM_HANDLERS(34)
DEFINE_AM_HANDLERS(35)
DEFINE_AM_HANDLERS(36)
DEFINE_AM_HANDLERS(37)
DEFINE_AM_HANDLERS(38)
DEFINE_AM_HANDLERS(39)
DEFINE_AM_HANDLERS(40)
DEFINE_AM_HANDLERS(41)
DEFINE_AM_HANDLERS(42)
DEFINE_AM_HANDLERS(43)
DEFINE_AM_HANDLERS(44)
DEFINE_AM_HANDLERS(45)
DEFINE_AM_HANDLERS(46)
DEFINE_AM_HANDLERS(47)
DEFINE_AM_HANDLERS(48)
DEFINE_AM_HANDLERS(49)
DEFINE_AM_HANDLERS(50)
DEFINE_AM_HANDLERS(51)
DEFINE_AM_HANDLERS(52)
DEFINE_AM_HANDLERS(53)
DEFINE_AM_HANDLERS(54)
DEFINE_AM_HANDLERS(55)
DEFINE_AM_HANDLERS(56)
DEFINE_AM_HANDLERS(57)
DEFINE_AM_HANDLERS(58)
DEFINE_AM_HANDLERS(59)
DEFINE_AM_HANDLERS(60)
DEFINE_AM_HANDLERS(61)
DEFINE_AM_HANDLERS(62)
DEFINE_AM_HANDLERS(63)
DEFINE_AM_HANDLERS(64)
DEFINE_AM_HANDLERS(65)
DEFINE_AM_HANDLERS(66)
DEFINE_AM_HANDLERS(67)
DEFINE_AM_HANDLERS(68)
DEFINE_AM_HANDLERS(69)
DEFINE_AM_HANDLERS(70)
DEFINE_AM_HANDLERS(71)
DEFINE_AM_HANDLERS(72)
DEFINE_AM_HANDLERS(73)
DEFINE_AM_HANDLERS(74)
DEFINE_AM_HANDLERS(75)
DEFINE_AM_HANDLERS(76)
DEFINE_AM_HANDLERS(77)
DEFINE_AM_HANDLERS(78)
DEFINE_AM_HANDLERS(79)

#define ASSIGN_AM_HANDLERS(INDEX) \
	psmx2_am_global.rma_handlers[INDEX] =  psmx2_am_rma_handler_##INDEX; \
	psmx2_am_global.atomic_handlers[INDEX] =  psmx2_am_atomic_handler_##INDEX;

void psmx2_am_global_init(void)
{
	fastlock_init(&psmx2_am_global.lock);

	ASSIGN_AM_HANDLERS(0)
	ASSIGN_AM_HANDLERS(1)
	ASSIGN_AM_HANDLERS(2)
	ASSIGN_AM_HANDLERS(3)
	ASSIGN_AM_HANDLERS(4)
	ASSIGN_AM_HANDLERS(5)
	ASSIGN_AM_HANDLERS(6)
	ASSIGN_AM_HANDLERS(7)
	ASSIGN_AM_HANDLERS(8)
	ASSIGN_AM_HANDLERS(9)
	ASSIGN_AM_HANDLERS(10)
	ASSIGN_AM_HANDLERS(11)
	ASSIGN_AM_HANDLERS(12)
	ASSIGN_AM_HANDLERS(13)
	ASSIGN_AM_HANDLERS(14)
	ASSIGN_AM_HANDLERS(15)
	ASSIGN_AM_HANDLERS(16)
	ASSIGN_AM_HANDLERS(17)
	ASSIGN_AM_HANDLERS(18)
	ASSIGN_AM_HANDLERS(19)
	ASSIGN_AM_HANDLERS(20)
	ASSIGN_AM_HANDLERS(21)
	ASSIGN_AM_HANDLERS(22)
	ASSIGN_AM_HANDLERS(23)
	ASSIGN_AM_HANDLERS(24)
	ASSIGN_AM_HANDLERS(25)
	ASSIGN_AM_HANDLERS(26)
	ASSIGN_AM_HANDLERS(27)
	ASSIGN_AM_HANDLERS(28)
	ASSIGN_AM_HANDLERS(29)
	ASSIGN_AM_HANDLERS(30)
	ASSIGN_AM_HANDLERS(31)
	ASSIGN_AM_HANDLERS(32)
	ASSIGN_AM_HANDLERS(33)
	ASSIGN_AM_HANDLERS(34)
	ASSIGN_AM_HANDLERS(35)
	ASSIGN_AM_HANDLERS(36)
	ASSIGN_AM_HANDLERS(37)
	ASSIGN_AM_HANDLERS(38)
	ASSIGN_AM_HANDLERS(39)
	ASSIGN_AM_HANDLERS(40)
	ASSIGN_AM_HANDLERS(41)
	ASSIGN_AM_HANDLERS(42)
	ASSIGN_AM_HANDLERS(43)
	ASSIGN_AM_HANDLERS(44)
	ASSIGN_AM_HANDLERS(45)
	ASSIGN_AM_HANDLERS(46)
	ASSIGN_AM_HANDLERS(47)
	ASSIGN_AM_HANDLERS(48)
	ASSIGN_AM_HANDLERS(49)
	ASSIGN_AM_HANDLERS(50)
	ASSIGN_AM_HANDLERS(51)
	ASSIGN_AM_HANDLERS(52)
	ASSIGN_AM_HANDLERS(53)
	ASSIGN_AM_HANDLERS(54)
	ASSIGN_AM_HANDLERS(55)
	ASSIGN_AM_HANDLERS(56)
	ASSIGN_AM_HANDLERS(57)
	ASSIGN_AM_HANDLERS(58)
	ASSIGN_AM_HANDLERS(59)
	ASSIGN_AM_HANDLERS(60)
	ASSIGN_AM_HANDLERS(61)
	ASSIGN_AM_HANDLERS(62)
	ASSIGN_AM_HANDLERS(63)
	ASSIGN_AM_HANDLERS(64)
	ASSIGN_AM_HANDLERS(65)
	ASSIGN_AM_HANDLERS(66)
	ASSIGN_AM_HANDLERS(67)
	ASSIGN_AM_HANDLERS(68)
	ASSIGN_AM_HANDLERS(69)
	ASSIGN_AM_HANDLERS(70)
	ASSIGN_AM_HANDLERS(71)
	ASSIGN_AM_HANDLERS(72)
	ASSIGN_AM_HANDLERS(73)
	ASSIGN_AM_HANDLERS(74)
	ASSIGN_AM_HANDLERS(75)
	ASSIGN_AM_HANDLERS(76)
	ASSIGN_AM_HANDLERS(77)
	ASSIGN_AM_HANDLERS(78)
	ASSIGN_AM_HANDLERS(79)
}

void psmx2_am_global_fini(void)
{
	fastlock_destroy(&psmx2_am_global.lock);
}

int psmx2_am_progress(struct psmx2_trx_ctxt *trx_ctxt)
{
	struct slist_entry *item;
	struct psmx2_am_request *req;
	struct psmx2_trigger *trigger;

	if (psmx2_env.tagged_rma) {
		psmx2_lock(&trx_ctxt->rma_queue.lock, 2);
		while (!slist_empty(&trx_ctxt->rma_queue.list)) {
			item = slist_remove_head(&trx_ctxt->rma_queue.list);
			req = container_of(item, struct psmx2_am_request, list_entry);
			psmx2_unlock(&trx_ctxt->rma_queue.lock, 2);
			psmx2_am_process_rma(trx_ctxt, req);
			psmx2_lock(&trx_ctxt->rma_queue.lock, 2);
		}
		psmx2_unlock(&trx_ctxt->rma_queue.lock, 2);
	}

	psmx2_lock(&trx_ctxt->trigger_queue.lock, 2);
	while (!slist_empty(&trx_ctxt->trigger_queue.list)) {
		item = slist_remove_head(&trx_ctxt->trigger_queue.list);
		trigger = container_of(item, struct psmx2_trigger, list_entry);
		psmx2_unlock(&trx_ctxt->trigger_queue.lock, 2);
		psmx2_process_trigger(trx_ctxt, trigger);
		psmx2_lock(&trx_ctxt->trigger_queue.lock, 2);
	}
	psmx2_unlock(&trx_ctxt->trigger_queue.lock, 2);

	return 0;
}

int psmx2_am_init(struct psmx2_trx_ctxt *trx_ctxt)
{
	psm2_am_handler_fn_t psmx2_am_handlers[3];
	int psmx2_am_handlers_idx[3];
	int num_handlers = 3;
	psm2_ep_t psm2_ep = trx_ctxt->psm2_ep;
	size_t size;
	int err = 0;
	int idx;

	FI_INFO(&psmx2_prov, FI_LOG_CORE, "\n");

	if (!trx_ctxt->am_initialized) {
		err = psm2_am_get_parameters(psm2_ep, &trx_ctxt->psm2_am_param,
					     sizeof(struct psm2_am_parameters),
					     &size);
		if (err)
			return psmx2_errno(err);

		psmx2_lock(&psmx2_am_global.lock, 1);
		if (psmx2_am_global.cnt >= PSMX2_AM_MAX_TRX_CTXT) {
			psmx2_unlock(&psmx2_am_global.lock, 1);
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"number of PSM2 endpoints exceed limit %d.\n",
				PSMX2_AM_MAX_TRX_CTXT);
			return -FI_EBUSY;
		}

		idx = psmx2_am_global.cnt++;
		psmx2_am_handlers[0] = psmx2_am_global.rma_handlers[idx];
		psmx2_am_handlers[1] = psmx2_am_global.atomic_handlers[idx];
		psmx2_am_handlers[2] = psmx2_am_sep_handler;
		psmx2_am_global.trx_ctxts[idx] = trx_ctxt;
		psmx2_unlock(&psmx2_am_global.lock, 1);

		err = psm2_am_register_handlers(psm2_ep, psmx2_am_handlers,
						num_handlers, psmx2_am_handlers_idx);
		if (err)
			return psmx2_errno(err);

		if ((psmx2_am_handlers_idx[0] != PSMX2_AM_RMA_HANDLER) ||
		    (psmx2_am_handlers_idx[1] != PSMX2_AM_ATOMIC_HANDLER) ||
		    (psmx2_am_handlers_idx[2] != PSMX2_AM_SEP_HANDLER)) {
			FI_WARN(&psmx2_prov, FI_LOG_CORE,
				"failed to register one or more AM handlers "
				"at indecies %d, %d, %d\n", PSMX2_AM_RMA_HANDLER,
				PSMX2_AM_ATOMIC_HANDLER, PSMX2_AM_SEP_HANDLER);
			return -FI_EBUSY;
		}

		trx_ctxt->am_initialized = 1;
	}

	return err;
}

void psmx2_am_fini(struct psmx2_trx_ctxt *trx_ctxt)
{
	/* there is no way to unregister AM handlers */
}

