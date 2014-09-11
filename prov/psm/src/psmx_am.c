/*
 * Copyright (c) 2013 Intel Corporation. All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * OpenFabrics.org BSD license below:
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

#include "psmx.h"

#if PSMX_USE_AM

struct psm_am_parameters psmx_am_param;

static psm_am_handler_fn_t psmx_am_handlers[3] = {
	psmx_am_rma_handler,
	psmx_am_msg_handler,
	psmx_am_atomic_handler,
};

static int psmx_am_handlers_idx[3];
static int psmx_am_handlers_initialized = 0;

int psmx_am_progress(struct psmx_fid_domain *fid_domain)
{
	struct psmx_am_request *req;

#if PSMX_AM_USE_SEND_QUEUE
	pthread_mutex_lock(&fid_domain->send_queue.lock);
	while (fid_domain->send_queue.head) {
		req = fid_domain->send_queue.head;
		if (req->next)
			fid_domain->send_queue.head = req->next;
		else
			fid_domain->send_queue.head = fid_domain->send_queue.tail = NULL;

		if (req->state == PSMX_AM_STATE_DONE)
			free(req);
		else
			psmx_am_process_send(fid_domain, req);
	}
	pthread_mutex_unlock(&fid_domain->send_queue.lock);
#endif

	if (fid_domain->use_tagged_rma) {
		pthread_mutex_lock(&fid_domain->rma_queue.lock);
		while (fid_domain->rma_queue.head) {
			req = fid_domain->rma_queue.head;
			if (req->next)
				fid_domain->rma_queue.head = req->next;
			else
				fid_domain->rma_queue.head = fid_domain->rma_queue.tail = NULL;
			psmx_am_process_rma(fid_domain, req);
		}
		pthread_mutex_unlock(&fid_domain->rma_queue.lock);
	}

	return 0;
}

#if PSMX_AM_USE_SEND_QUEUE
static void *psmx_am_async_progress(void *args)
{
	struct psmx_fid_domain *fid_domain = args;
	struct timespec timeout;

	timeout.tv_sec = 1;
	timeout.tv_nsec = 1000;

	pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);

	while (1) {
		pthread_mutex_lock(&fid_domain->progress_mutex);
		pthread_cond_wait(&fid_domain->progress_cond, &fid_domain->progress_mutex);
		pthread_mutex_unlock(&fid_domain->progress_mutex);
		pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, NULL);

		psmx_am_progress(fid_domain);

		pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);
	}

	return NULL;
}
#endif

int psmx_am_init(struct psmx_fid_domain *fid_domain)
{
	psm_ep_t psm_ep = fid_domain->psm_ep;
	size_t size;
	int err = 0;

	if (!psmx_am_handlers_initialized) {
		err = psm_am_get_parameters(psm_ep, &psmx_am_param,
						sizeof(psmx_am_param), &size);
		if (err)
			return psmx_errno(err);

		err = psm_am_register_handlers(psm_ep, psmx_am_handlers, 3,
						psmx_am_handlers_idx);
		if (err)
			return psmx_errno(err);

		assert(psmx_am_handlers_idx[0] == PSMX_AM_RMA_HANDLER);
		assert(psmx_am_handlers_idx[1] == PSMX_AM_MSG_HANDLER);
		assert(psmx_am_handlers_idx[2] == PSMX_AM_ATOMIC_HANDLER);

		psmx_am_handlers_initialized = 1;
	}

	pthread_mutex_init(&fid_domain->rma_queue.lock, NULL);
	pthread_mutex_init(&fid_domain->recv_queue.lock, NULL);
	pthread_mutex_init(&fid_domain->unexp_queue.lock, NULL);
#if PSMX_AM_USE_SEND_QUEUE
	pthread_mutex_init(&fid_domain->send_queue.lock, NULL);
	pthread_mutex_init(&fid_domain->progress_mutex, NULL);
	pthread_cond_init(&fid_domain->progress_cond, NULL);
	err = pthread_create(&fid_domain->progress_thread, NULL, psmx_am_async_progress, (void *)fid_domain);
#endif

	return err;
}

int psmx_am_fini(struct psmx_fid_domain *fid_domain)
{
#if PSMX_AM_USE_SEND_QUEUE
        if (fid_domain->progress_thread) {
                pthread_cancel(fid_domain->progress_thread);
                pthread_join(fid_domain->progress_thread, NULL);
		pthread_mutex_destroy(&fid_domain->progress_mutex);
		pthread_cond_destroy(&fid_domain->progress_cond);
        }
#endif

	return 0;
}

int psmx_am_enqueue_recv(struct psmx_fid_domain *fid_domain,
			  struct psmx_am_request *req)
{
	if (fid_domain->recv_queue.tail)
		fid_domain->recv_queue.tail->next = req;
	else
		fid_domain->recv_queue.head = req;

	fid_domain->recv_queue.tail = req;

	return 0;
}

struct psmx_am_request *
	psmx_am_search_and_dequeue_recv(struct psmx_fid_domain *fid_domain,
					const void *src_addr)
{
	struct psmx_am_request *req, *prev = NULL;

	req = fid_domain->recv_queue.head;
	if (!req)
		return NULL;

	while (req) {
		if (!req->recv.src_addr || req->recv.src_addr == src_addr) {
			if (prev)
				prev->next = req->next;
			else
				fid_domain->recv_queue.head = req->next;

			if (!req->next)
				fid_domain->recv_queue.tail = prev;

			req->next = NULL;
			return req;
		}
		prev = req;
		req = req->next;
	}

	return NULL;
}

#if PSMX_AM_USE_SEND_QUEUE
int psmx_am_enqueue_send(struct psmx_fid_domain *fid_domain,
			  struct psmx_am_request *req)
{
	req->state = PSMX_AM_STATE_QUEUED;

	if (fid_domain->send_queue.tail)
		fid_domain->send_queue.tail->next = req;
	else
		fid_domain->send_queue.head = req;

	fid_domain->send_queue.tail = req;

	return 0;
}
#endif

int psmx_am_enqueue_rma(struct psmx_fid_domain *fid_domain,
			  struct psmx_am_request *req)
{
	req->state = PSMX_AM_STATE_QUEUED;

	if (fid_domain->rma_queue.tail)
		fid_domain->rma_queue.tail->next = req;
	else
		fid_domain->rma_queue.head = req;

	fid_domain->rma_queue.tail = req;

	return 0;
}

#endif /* PSMX_USE_AM */

