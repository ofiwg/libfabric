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

#ifndef _FI_PSM_VERSION_H_
#define _FI_PSM_VERSION_H_

#if HAVE_PSM2_SRC
#include "psm2/psm2.h"
#include "psm2/psm2_mq.h"
#include "psm2/psm2_am.h"
#else
#include <psm2.h>
#include <psm2_mq.h>
#include <psm2_am.h>
#endif

#define PSMX2_PROV_NAME		"psm2"
#define PSMX2_DOMAIN_NAME	"psm2"
#define PSMX2_FABRIC_NAME	"psm2"

#define PSMX2_DEFAULT_UUID	"00FF00FF-0000-0000-0000-00FF00FF00FF"
#define PROVIDER_INI		PSM2_INI

#define PSMX2_STATUS_TYPE	psm2_mq_status2_t
#define PSMX2_STATUS_ERROR(s)	((s)->error_code)
#define PSMX2_STATUS_TAG(s)	((s)->msg_tag)
#define PSMX2_STATUS_RCVLEN(s)	((s)->nbytes)
#define PSMX2_STATUS_SNDLEN(s)	((s)->msg_length)
#define PSMX2_STATUS_PEER(s)	((s)->msg_peer)
#define PSMX2_STATUS_CONTEXT(s)	((s)->context)

/*
 * psm2_mq_test2 is called immediately after psm2_mq_ipeek with a lock held to
 * prevent psm2_mq_ipeek from returning the same request multiple times under
 * different threads.
 */
#define PSMX2_POLL_COMPLETION(trx_ctxt, psm2_req, psm2_status, status, err) \
	do { \
		if (psmx2_trylock(&(trx_ctxt)->poll_lock, 2)) { \
			(err) = PSM2_MQ_NO_COMPLETIONS; \
		} else { \
			(err) = psm2_mq_ipeek((trx_ctxt)->psm2_mq, &(psm2_req), NULL); \
			if ((err) == PSM2_OK) { \
				psm2_mq_test2(&(psm2_req), &(psm2_status)); \
				(status) = &(psm2_status); \
			} \
			psmx2_unlock(&(trx_ctxt)->poll_lock, 2); \
		} \
	} while(0)

#define PSMX2_FREE_COMPLETION(trx_ctxt, status)

#endif

