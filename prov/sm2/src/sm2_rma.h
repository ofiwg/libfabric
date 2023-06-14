#include "sm2.h"

static inline int sm2_alloc_rma_ctx(struct sm2_ep *ep,
				    struct sm2_sar_ctx **rma_ctx)
{
	struct sm2_sar_ctx *ctx;

	ctx = ofi_buf_alloc(ep->sar_ctx_pool);
	if (!ctx) {
		FI_WARN(&sm2_prov, FI_LOG_EP_CTRL,
			"Error allocating rma ctx\n");
		return -FI_ENOMEM;
	}
	*rma_ctx = ctx;
	return FI_SUCCESS;
}

static inline void sm2_free_rma_ctx(struct sm2_sar_ctx *rma_ctx)
{
	ofi_buf_free(rma_ctx);
}

void sm2_fill_rma_ctx(struct sm2_ep *ep, const struct fi_msg_rma *msg,
		      uint32_t op, uint64_t op_flags, sm2_gid_t peer_gid,
		      struct sm2_sar_ctx *ctx);

ssize_t sm2_rma_cmd_fill_sar_xfer(struct sm2_xfer_entry *xfer_entry,
				  struct sm2_sar_ctx *ctx);
void sm2_rma_handle_local_error(struct sm2_ep *ep,
				struct sm2_xfer_entry *xfer_entry,
				struct sm2_sar_ctx *ctx, uint64_t err);
void sm2_rma_handle_remote_error(struct sm2_ep *ep,
				 struct sm2_xfer_entry *xfer_entry,
				 struct sm2_sar_ctx *ctx);