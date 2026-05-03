#ifndef RXM_QP_SELECTOR_H
#define RXM_QP_SELECTOR_H

#include <stddef.h>
#include <stdint.h>

struct rxm_conn;

enum rxm_op_type {
	RXM_OP_EAGER,
	RXM_OP_SAR_FIRST,
	RXM_OP_SAR_MIDDLE,
	RXM_OP_SAR_LAST,
	RXM_OP_RNDV_CTRL,
	RXM_OP_RNDV_RMA,
	RXM_OP_RMA,
	RXM_OP_ATOMIC,
};

struct rxm_selector_ctx {
	enum rxm_op_type op;
	uint64_t msg_id;
};

struct rxm_qp_selector {
	const char *name;
	uint8_t (*select)(struct rxm_conn *conn,
			  const struct rxm_selector_ctx *ctx);
};

extern const struct rxm_qp_selector rxm_selector_single_qp;

#endif /* RXM_QP_SELECTOR_H */
