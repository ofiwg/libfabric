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

#ifndef _VERBS_RDM_H
#define _VERBS_RDM_H

#include <prov/verbs/src/verbs_utils.h>
#include <prov/verbs/src/ep_rdm/verbs_tagged_ep_rdm_states.h>

#define FI_IBV_RDM_CONN_SELF ((struct fi_ibv_rdm_tagged_conn *)0x1)

#define FI_IBV_RDM_ADDR_STR_FORMAT "[%02x:%02x:%02x:%02x:%02x:%02x]"

#define FI_IBV_RDM_ADDR_STR(addr)                               \
        *((unsigned char*)addr),*((unsigned char*)addr+1),      \
        *((unsigned char*)addr+2),*((unsigned char*)addr+3),    \
        *((unsigned char*)addr+4),*((unsigned char*)addr+5)

#define FI_IBV_RDM_ST_PKTTYPE_MASK  ((uint32_t)0xFF)
#define FI_IBV_RDM_EAGER_PKT            0
#define FI_IBV_RDM_RNDV_RTS_PKT         1
#define FI_IBV_RDM_RNDV_ACK_PKT         2
#define FI_IBV_RDM_SET_PKTTYPE(dest, type) (dest |= type)
#define FI_IBV_RDM_GET_PKTTYPE(value) (value & FI_IBV_RDM_ST_PKTTYPE_MASK)

/* ------- SERVICE_TAG layout ---------
 * [ 24 bits - Unused at the moment ] [ 8 bits - pkt type ]
 *                                    IBV_RDM_ST_PKTYPE_MASK
 */

// WR - work request
#define FI_IBV_RDM_SERVICE_WR_MASK              ((uint64_t)0x1)
#define FI_IBV_RDM_CHECK_SERVICE_WR_FLAG(value) \
        (value & FI_IBV_RDM_SERVICE_WR_MASK)

#define FI_IBV_RDM_PACK_WR(value)               ((uint64_t)value)
#define FI_IBV_RDM_UNPACK_WR(value)             ((void*)(uintptr_t)value)

#define FI_IBV_RDM_PACK_SERVICE_WR(value)       \
        (((uint64_t)(uintptr_t)(void*)value) | FI_IBV_RDM_SERVICE_WR_MASK)

#define FI_IBV_RDM_UNPACK_SERVICE_WR(value)     \
        ((void*)(uintptr_t)(value & (~(FI_IBV_RDM_SERVICE_WR_MASK))))

struct fi_ibv_rdm_tagged_header {
    uint64_t imm_data;          // TODO: not implemented
    uint64_t tag;
    uint32_t service_tag;
    uint32_t padding;
};

struct fi_ibv_rdm_tagged_rndv_header {
    struct fi_ibv_rdm_tagged_header base;
    uint64_t src_addr;
    void *id;
    int len;
    uint32_t mem_key;
};

typedef struct fi_ibv_rdm_tagged_unexpected_rbuff {
    struct fi_ibv_mem_pool_entry mpe;
    char payload[sizeof(void *)];
} fi_ibv_rdm_tagged_unexp_rbuff_t;

struct fi_ibv_rdm_tagged_request {
    struct fi_ibv_mem_pool_entry mpe;

    uint64_t tag;
    uint64_t tagmask;

    struct {
        void *dest_buf;

        union {
            void *expected_recv_buf;                        // user level
            fi_ibv_rdm_tagged_unexp_rbuff_t* unexp_rbuf;
            void *sbuf;                                     // in terms of verbs
        };
        int send_completions_wait;

        struct {
            fi_ibv_rdm_tagged_request_eager_state_t eager;
            fi_ibv_rdm_tagged_request_rndv_state_t rndv;
        } state;
    };

    struct {
        void *rndv_id;
        struct ibv_mr *rndv_mr;
        uint32_t rndv_remote_key;
    };
    const void *src_addr;
    int len;
    uint32_t imm;
    struct fi_context *context;
    struct fi_ibv_rdm_tagged_conn *conn;
    /* Request can be an element of only one queue at the moment */
    struct dlist_entry queue_entry;
};

static inline void
fi_ibv_rdm_tagged_zero_request(struct fi_ibv_rdm_tagged_request *request)
{
    char *p = (char *)request;
    fi_ibv_memset(p + sizeof(request->mpe), 0,
                  sizeof(*request) - sizeof(request->mpe));
}

void fi_ibv_rdm_tagged_print_request(char *buf,
     struct fi_ibv_rdm_tagged_request *request);

typedef struct fi_ibv_rdm_tagged_buf {
    struct fi_ibv_rdm_tagged_header header;
    char payload[sizeof(void *)];
} fi_ibv_rdm_tagged_buf_t;

struct fi_ibv_rdm_ep {
    struct fid_ep ep_fid;
    struct fi_ibv_domain *domain;
    struct fi_ibv_cq *fi_scq;
    struct fi_ibv_cq *fi_rcq;

    struct sockaddr_in my_ipoib_addr;
    char my_ipoib_addr_str[INET6_ADDRSTRLEN];
    struct rdma_cm_id *cm_listener;
    struct rdma_event_channel *cm_listener_ec;
    uint16_t cm_listener_port;
    struct fi_ibv_av *av;
    int fi_ibv_rdm_addrlen;
    char my_rdm_addr[FI_IBV_RDM_DFLT_ADDRLEN];

    int buff_len;
    int n_buffs;
    int rq_wr_depth;    // RQ depth
    int sq_wr_depth;    // SQ depth
    int total_outgoing_send;
    int pend_send;
    int pend_recv;
    int num_active_conns;
    int max_inline_rc;
    int rndv_threshold;
    struct ibv_cq *scq;
    struct ibv_cq *rcq;
    int scq_depth;
    int rcq_depth;
    pthread_t cm_progress_thread;
    pthread_mutex_t cm_lock;
    int is_closing;
    int recv_preposted_threshold;
};

#define FI_IBV_RDM_ALIGNMENT 64

enum {
    FI_VERBS_CONN_ALLOCATED,
    FI_VERBS_CONN_STARTED,
    FI_VERBS_CONN_REJECTED,
    FI_VERBS_CONN_ESTABLISHED,
    FI_VERBS_CONN_LOCAL_DISCONNECT,
    FI_VERBS_CONN_REMOTE_DISCONNECT,
    FI_VERBS_CONN_CLOSED
};

struct fi_ibv_rdm_tagged_conn {
    struct ibv_qp *qp;
    char addr[FI_IBV_RDM_DFLT_ADDRLEN];

    struct rdma_cm_id *id;
    int is_active;
    int state;

    char *sbuf_mem_reg;
    char *sbuf_head;
    char *sbuf_ack_head;

    char *rbuf_mem_reg;
    char *rbuf_head;

    struct dlist_entry postponed_requests_head;
    struct fi_ibv_rdm_tagged_postponed_entry *postponed_entry;

    struct ibv_mr *s_mr;
    struct ibv_mr *r_mr;

    uint32_t remote_sbuf_rkey;
    uint32_t remote_rbuf_rkey;

    char *remote_sbuf_mem_reg;
    char *remote_rbuf_mem_reg;
    char *remote_sbuf_head;

    int sends_outgoing;
    int recv_preposted;
    UT_hash_handle hh;
#if ENABLE_DEBUG
    size_t unexp_counter;
    size_t exp_counter;
#endif                          // ENABLE_DEBUG
};

struct fi_ibv_rdm_tagged_postponed_entry {
    struct fi_ibv_mem_pool_entry mpe;
    struct fi_ibv_rdm_tagged_conn *conn;

    struct dlist_entry queue_entry;
};

typedef enum fi_ibv_rdm_tagged_buffer_status {
    BUF_STATUS_FREE = 0,
    BUF_STATUS_BUSY
} fi_ibv_rdm_tagged_buffer_status_t;

typedef struct fi_ibv_rmd_tagged_buffer_service_data {
    fi_ibv_rdm_tagged_buffer_status_t status;
    int seq_number;
} fi_ibv_rdm_tagged_buffer_service_data_t;

#define FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE                               \
    (sizeof (fi_ibv_rdm_tagged_buffer_service_data_t) < FI_IBV_RDM_ALIGNMENT   \
    ? FI_IBV_RDM_ALIGNMENT                                                     \
    : (sizeof (fi_ibv_rdm_tagged_buffer_service_data_t) +                      \
    (FI_IBV_RDM_ALIGNMENT - sizeof (fi_ibv_rdm_tagged_buffer_service_data_t) % \
    FI_IBV_RDM_ALIGNMENT)))

static inline fi_ibv_rdm_tagged_buffer_service_data_t *
fi_ibv_rdm_tagged_get_buff_service_data(char *buff)
{
    return (fi_ibv_rdm_tagged_buffer_service_data_t *)
        (buff - FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE);
}

static inline void
fi_ibv_rdm_tagged_set_buffer_status(char *buff,
                                    fi_ibv_rdm_tagged_buffer_status_t status)
{
    fi_ibv_rdm_tagged_get_buff_service_data(buff)->status = status;
}

static inline fi_ibv_rdm_tagged_buffer_status_t
fi_ibv_rdm_tagged_get_buffer_status(char *buff)
{
    return fi_ibv_rdm_tagged_get_buff_service_data(buff)->status;
}

static inline uintptr_t
fi_ibv_rdm_tagged_get_remote_addr(struct fi_ibv_rdm_tagged_conn *conn,
                                  char *local_sbuff)
{
    return (uintptr_t) (conn->remote_rbuf_mem_reg +
                        (local_sbuff - conn->sbuf_mem_reg));
}

static inline void
fi_ibv_rdm_tagged_push_buff_pointer(char *area_start, size_t area_size,
                                    char **buff, size_t offset)
{
    FI_VERBS_DBG("old_pointer: %p, sn = %d\n", *buff,
                 fi_ibv_rdm_tagged_get_buff_service_data(*buff)->seq_number);

    char *buff_tmp = (*buff) + offset;
    (*buff) = buff_tmp < (area_start + area_size)
        ? buff_tmp : area_start + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE;

    FI_VERBS_DBG("new_pointer: %p, sn = %d\n", *buff,
                 fi_ibv_rdm_tagged_get_buff_service_data(*buff)->seq_number);
    assert(fi_ibv_rdm_tagged_get_buff_service_data(*buff)->seq_number ==
           (((*buff) - (area_start + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE))
            / offset));
}

static inline void
fi_ibv_rdm_tagged_push_sbuff_head(struct fi_ibv_rdm_tagged_conn *conn,
                                  struct fi_ibv_rdm_ep *ep)
{
    fi_ibv_rdm_tagged_push_buff_pointer(conn->sbuf_mem_reg,
                                        ep->buff_len * ep->n_buffs,
                                        &conn->sbuf_head, ep->buff_len);
}

static inline fi_ibv_rdm_tagged_buf_t *
fi_ibv_rdm_tagged_get_rbuf(struct fi_ibv_rdm_tagged_conn *conn,
                           struct fi_ibv_rdm_ep *ep,
                           int seq_num)
{
    FI_VERBS_DBG("recv buf %d\n", seq_num);

    char *rbuf = (conn->rbuf_mem_reg +
                  FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE +
                  (seq_num * ep->buff_len));
    assert(fi_ibv_rdm_tagged_get_buff_service_data(rbuf)->seq_number == seq_num);
    return (fi_ibv_rdm_tagged_buf_t *) rbuf;
}

static inline void
fi_ibv_rdm_tagged_buffer_lists_init(struct fi_ibv_rdm_tagged_conn *conn,
                                    struct fi_ibv_rdm_ep *ep)
{
    conn->sbuf_head =
        conn->sbuf_mem_reg + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE;
    conn->rbuf_head =
        conn->rbuf_mem_reg + FI_IBV_RDM_TAGGED_BUFF_SERVICE_DATA_SIZE;
    conn->sbuf_ack_head = conn->rbuf_head;      // used only service data

    int i;
    for (i = 0; i < ep->n_buffs; ++i) {
        fi_ibv_rdm_tagged_set_buffer_status(conn->sbuf_head +
                          i * ep->buff_len, BUF_STATUS_FREE);
        fi_ibv_rdm_tagged_set_buffer_status(conn->sbuf_ack_head +
                          i * ep->buff_len, BUF_STATUS_FREE);

        fi_ibv_rdm_tagged_get_buff_service_data(conn->sbuf_head +
                              i * ep->buff_len)->seq_number = i;
        fi_ibv_rdm_tagged_get_buff_service_data(conn->sbuf_ack_head +
                              i * ep->buff_len)->seq_number = i;
    }
}

int fi_ibv_rdm_tagged_poll(struct fi_ibv_rdm_ep *ep);
int fi_ibv_rdm_tagged_set_cq_ops(struct fi_ibv_cq *cq);
int fi_ibv_rdm_set_av_ops(struct fi_ibv_av *av);
int fi_ibv_rdm_tagged_cm_progress(struct fi_ibv_rdm_ep *ep);
int fi_ibv_rdm_start_disconnection(struct fi_ibv_rdm_ep *ep,
                                   struct fi_ibv_rdm_tagged_conn *conn);
int fi_ibv_rdm_tagged_conn_cleanup(struct fi_ibv_rdm_ep *ep,
                                   struct fi_ibv_rdm_tagged_conn *conn);
int fi_ibv_rdm_start_connection(struct fi_ibv_rdm_ep *ep,
                                struct fi_ibv_rdm_tagged_conn *conn);
int fi_ibv_rdm_tagged_repost_receives(struct fi_ibv_rdm_tagged_conn *conn,
                                      struct fi_ibv_rdm_ep *ep,
                                      int num_to_post);
int fi_ibv_rdm_tagged_open_ep(struct fid_domain *domain, struct fi_info *info,
                              struct fid_ep **ep, void *context);
int fi_ibv_rdm_tagged_prepare_send_resources
    (struct fi_ibv_rdm_tagged_request* request, struct fi_ibv_rdm_ep *ep);

#endif /* _VERBS_RDM_H */