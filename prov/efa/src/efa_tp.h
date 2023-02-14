#ifndef _EFA_TP_H
#define _EFA_TP_H

#include <config.h>

#if HAVE_LTTNG

#include "efa_tp_def.h"
#include "rdm/rxr_op_entry.h"

#include <lttng/tracef.h>
#include <lttng/tracelog.h>

#define efa_tracepoint(...)	lttng_ust_tracepoint(EFA_TP_PROV, __VA_ARGS__)

/*
 * Simple printf()-style tracepoints
 * Tracing events will be labeled `lttng_ust_tracef:*`
 */
#define efa_tracef	lttng_ust_tracef

/* tracelog() is similar to tracef(), but with a log level param */
#define efa_tracelog	lttng_ust_tracelog

static inline void efa_tracepoint_wr_id_post_send(const void *wr_id)
{
	struct rxr_pkt_entry *pkt_entry = (struct rxr_pkt_entry *) wr_id;
	struct rxr_op_entry *op_entry = (struct rxr_op_entry *) pkt_entry->x_entry;
	if (!op_entry)
		return;
	efa_tracepoint(post_send, (size_t) wr_id, (size_t) op_entry, (size_t) op_entry->cq_entry.op_context);
}

static inline void efa_tracepoint_wr_id_post_recv(const void *wr_id)
{
	struct rxr_pkt_entry *pkt_entry = (struct rxr_pkt_entry *) wr_id;
	struct rxr_op_entry *op_entry = (struct rxr_op_entry *) pkt_entry->x_entry;
	if (!op_entry)
		return;
	efa_tracepoint(post_recv, (size_t) wr_id, (size_t) op_entry, (size_t) op_entry->cq_entry.op_context);
}

#else

#define efa_tracepoint(...)	while (0) {}
#define efa_tracef(...)	while (0) {}
#define efa_tracelog(...)	while (0) {}

#endif /* HAVE_LTTNG */

#endif /* _EFA_TP_H */
