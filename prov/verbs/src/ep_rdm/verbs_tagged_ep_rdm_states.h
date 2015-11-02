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

#ifndef _VERBS_TAGGED_EP_RDM_STATES_H
#define _VERBS_TAGGED_EP_RDM_STATES_H

#include <stddef.h>
#include <stdint.h>

struct fi_ibv_rdm_tagged_request;

typedef enum fi_ibv_rdm_tagged_request_eager_state {
    FI_IBV_STATE_EAGER_BEGIN = 0,      // must be 0

    FI_IBV_STATE_EAGER_SEND_POSTPONED,
    FI_IBV_STATE_EAGER_SEND_WAIT4LC,   // wait for local completion
    FI_IBV_STATE_EAGER_SEND_END,

    FI_IBV_STATE_EAGER_RECV_BEGIN,
    FI_IBV_STATE_EAGER_RECV_WAIT4PKT,
    FI_IBV_STATE_EAGER_RECV_WAIT4RECV,
    FI_IBV_STATE_EAGER_RECV_END,

    FI_IBV_STATE_EAGER_READY_TO_FREE,

    FI_IBV_STATE_EAGER_COUNT           // must be last
} fi_ibv_rdm_tagged_request_eager_state_t;

char *fi_ibv_rdm_tagged_req_eager_state_to_str
    (fi_ibv_rdm_tagged_request_eager_state_t state);

typedef enum fi_ibv_rdm_tagged_request_rndv_state {
    FI_IBV_STATE_RNDV_NOT_USED = 0,    // must be 0
    FI_IBV_STATE_RNDV_SEND_BEGIN,
    //    FI_IBV_STATE_RNDV_SEND_WAIT4CTS, // not implemented yet
    FI_IBV_STATE_RNDV_SEND_WAIT4SEND,
    FI_IBV_STATE_RNDV_SEND_WAIT4ACK,
    FI_IBV_STATE_RNDV_SEND_END,

    FI_IBV_STATE_RNDV_RECV_BEGIN,
    FI_IBV_STATE_RNDV_RECV_WAIT4RES,
    FI_IBV_STATE_RNDV_RECV_WAIT4RECV,
    FI_IBV_STATE_RNDV_RECV_WAIT4LC,
    FI_IBV_STATE_RNDV_RECV_END,

    FI_IBV_STATE_RNDV_COUNT            // must be last
} fi_ibv_rdm_tagged_request_rndv_state_t;

char *fi_ibv_rdm_tagged_req_rndv_state_to_str
        (fi_ibv_rdm_tagged_request_rndv_state_t);

typedef enum fi_ibv_rdm_tagged_request_event {
    FI_IBV_EVENT_SEND_START = 0,
    FI_IBV_EVENT_SEND_READY,
    FI_IBV_EVENT_SEND_COMPLETED,
    FI_IBV_EVENT_SEND_GOT_CTS,
    FI_IBV_EVENT_SEND_GOT_LC,

    FI_IBV_EVENT_RECV_START,
    FI_IBV_EVENT_RECV_GOT_PKT_PREPROCESS,
    FI_IBV_EVENT_RECV_GOT_PKT_PROCESS,
    FI_IBV_EVENT_RECV_GOT_ACK,

    FI_IBV_EVENT_COUNT                 // must be last
} fi_ibv_rdm_tagged_request_event_t;

char *
fi_ibv_rdm_tagged_req_event_to_str(fi_ibv_rdm_tagged_request_event_t event);

// Send service data types

typedef struct {
    const void *src_addr;
    void *context;
    struct fi_ibv_rdm_tagged_conn *conn;
    size_t tag;
    size_t data_len;
    size_t rndv_threshold;
    unsigned int imm;
} fi_ibv_rdm_tagged_send_start_data_t;

typedef struct {
    struct fi_ibv_rdm_ep *ep;
} fi_ibv_rdm_tagged_send_ready_data_t;

typedef struct {
    struct fi_ibv_rdm_ep *ep;
} fi_ibv_rdm_tagged_send_completed_data_t;

// Recv service data types

typedef struct {
    size_t tag;
    size_t tagmask;
    struct fi_context *context;
    void *dest_addr;
    struct fi_ibv_rdm_tagged_conn *conn;
    struct fi_ibv_rdm_ep *ep;
    size_t data_len;
} fi_ibv_rdm_tagged_recv_start_data_t;

typedef struct {
    struct fi_ibv_rdm_tagged_conn *conn;
    struct fi_ibv_rdm_ep *ep;
    void *rbuf;
    size_t arrived_len;
    uint64_t pkt_type;
    int imm_data;
} recv_got_pkt_preprocess_data_t;

typedef struct {
    struct fi_ibv_rdm_ep *ep;
} recv_got_pkt_process_data_t;

// Return codes

typedef enum {
    FI_EP_RDM_HNDL_SUCCESS = 0,
    FI_EP_RDM_HNDL_DELETED_REQUEST = 1,
    FI_EP_RDM_HNDL_ERROR = 2,
    FI_EP_RDM_HNDL_NOT_INIT = (int)-1
} fi_rdm_tagged_req_hndl_ret_t;

// Interfaces

fi_rdm_tagged_req_hndl_ret_t fi_ibv_rdm_tagged_req_hndls_init();
fi_rdm_tagged_req_hndl_ret_t fi_ibv_rdm_tagged_req_hndls_clean();
fi_rdm_tagged_req_hndl_ret_t fi_ibv_rdm_tagged_req_hndl
    (struct fi_ibv_rdm_tagged_request *request,
     fi_ibv_rdm_tagged_request_event_t event, void *data);

// Send/Recv counters control

#define FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS(_connection, _ep, _send_flags) \
do {                                                                \
    (_connection)->sends_outgoing++;                                \
    (_ep)->pend_send++;                                             \
    (_ep)->total_outgoing_send++;                                   \
    (_send_flags) |= IBV_SEND_SIGNALED;                             \
                                                                    \
    FI_VERBS_DBG("SEND_COUNTER++, conn %p, sends_outgoing %d\n",    \
           _connection,                                             \
           (_connection)->sends_outgoing);                          \
} while (0)

#define FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS_REQ(_request, _ep, _send_flags)  \
do {                                                                \
    (_request)->send_completions_wait++;                            \
    FI_IBV_RDM_TAGGED_INC_SEND_COUNTERS((_request)->conn,           \
                                        _ep, _send_flags);          \
                                                                    \
    FI_VERBS_DBG("SEND_COUNTER++, send_completions_wait %d\n",      \
           (_request)->send_completions_wait);                      \
} while (0)

#define FI_IBV_RDM_TAGGED_DEC_SEND_COUNTERS(_connection, _ep)       \
do {                                                                \
    (_connection)->sends_outgoing--;                                \
    (_ep)->total_outgoing_send--;                                   \
    (_ep)->pend_send--;                                             \
                                                                    \
    FI_VERBS_DBG("SEND_COUNTER--, conn %p, sends_outgoing %d\n",    \
           _connection, (_connection)->sends_outgoing);             \
    assert((_ep)->pend_send >= 0);                                  \
    assert((_connection)->sends_outgoing >= 0);                     \
} while (0)

#define FI_IBV_RDM_TAGGED_SENDS_OUTGOING_ARE_LIMITED(_connection, _ep)  \
    ((_connection)->sends_outgoing > 0.5*(_ep)->sq_wr_depth)

#define PEND_SEND_IS_LIMITED(_ep)                                       \
    ((_ep)->pend_send > 0.5 * (_ep)->scq_depth)

#define SEND_RESOURCES_IS_BUSY(_connection, _ep)                        \
    (FI_IBV_RDM_TAGGED_SENDS_OUTGOING_ARE_LIMITED(_connection, _ep) ||  \
     PEND_SEND_IS_LIMITED(_ep))

#endif                          /* _VERBS_TAGGED_EP_RDM_STATES_H */
