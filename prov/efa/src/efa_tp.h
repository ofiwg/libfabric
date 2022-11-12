#ifndef _EFA_TP_H_
#define _EFA_TP_H_

#include <config.h>

#if HAVE_LTTNG
#include "efa_tp_def.h"
#include "rxr/rxr_op_entry.h"

enum efa_tracing_ibv_post_type_t
{
	post_send,
	post_recv
} efa_tracing_ibv_post_type_t;

typedef enum efa_tracing_ibv_post_type_t efa_tracing_ibv_post_type;

static void efa_tracing_wr_id(const efa_tracing_ibv_post_type tp_name, const void *wr_id)
{
	struct rxr_pkt_entry *pkt_entry = (struct rxr_pkt_entry *)wr_id;
	struct rxr_op_entry *op_entry = (struct rxr_op_entry *)pkt_entry->x_entry;
	if (!op_entry)
		return;
	switch (tp_name)
	{
	case post_send:
		lttng_ust_tracepoint(EFA_TP_PROV, post_send,
				     (size_t)wr_id,
				     (size_t)op_entry,
				     (size_t)op_entry->cq_entry.op_context);
		break;
	case post_recv:
		lttng_ust_tracepoint(EFA_TP_PROV, post_recv,
				     (size_t)wr_id,
				     (size_t)op_entry,
				     (size_t)op_entry->cq_entry.op_context);
		break;
	default:
		assert(0);
	}
}
#define efa_tracing(tp_name, ...) efa_tracing_wr_id(tp_name, __VA_ARGS__);

#else
#define efa_tracing(tp_name, ...) \
	while (0)                 \
	{                         \
	}
#endif

#endif // _EFA_TP_H_