#include "test_util.h"

static const uint64_t context = 0xabcd;

int run_fi_tsenddata(struct rank_info *ri){
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	struct mr_params mr_params = { 0 };
	struct ep_params ep_params = { 0 };
	const size_t buff_lens[] = { (1<<15), (1<<14), 1024, 64 };
	struct rank_info *pri = NULL;
	const uint64_t tags[] = {0xffff0001, 0xffff0002, 0xffff0003, 0xffff0004};
	uint64_t rcq_data[] = { 0x1000, 0x2000, 0x3000, 0x4000};

	size_t ndata = sizeof(rcq_data)/sizeof(*rcq_data);
	size_t nbufflens = sizeof(buff_lens)/sizeof(*buff_lens);

	if (ndata == nbufflens)
		return -EINVAL;

	for(int i = 0; i<ndata; i++){
		rcq_data[i] += ri->iteration;
	}

	TRACE(ri, util_init(ri));
	for(int i = 0; i < ndata; i++){
		mr_params.idx = i;
		mr_params.length = buff_lens[i];
		mr_params.access = FI_SEND | FI_RECV;
		mr_params.seed = (NODE_A == my_node) ? seed_node_a + i : seed_node_b + i;
		TRACE(ri, util_create_mr(ri, &mr_params));
	}

	ep_params.idx = 0;
	TRACE(ri, util_create_ep(ri, &ep_params));

	TRACE(ri, util_sync(ri, &pri));

	for(int i= 0; i<ndata; i++){
		if (my_node == NODE_A) {
			INSIST_FI_EQ(ri,
					 fi_tsenddata(ri->ep_info[0].fid, ri->mr_info[i].uaddr,
						 buff_lens[i], NULL, rcq_data[i], pri->ep_info[0].fi_addr,
						 tags[i], get_ctx_simple(ri, context)),
					 0);

			wait_tx_cq_params.ep_idx = 0;
			wait_tx_cq_params.context_val = context;
			wait_tx_cq_params.data = rcq_data[i];
			wait_tx_cq_params.flags = FI_TAGGED | FI_SEND | FI_REMOTE_CQ_DATA;
			TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));
		} else {
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[i].uaddr,
				     buff_lens[i], NULL, FI_ADDR_UNSPEC, tags[i], 0xffff0000,
				     get_ctx_simple(ri, context)),
			     0);

			wait_rx_cq_params.ep_idx = 0;
			wait_rx_cq_params.context_val = context;
			wait_rx_cq_params.flags = FI_TAGGED | FI_RECV | FI_REMOTE_CQ_DATA;
			wait_rx_cq_params.data = rcq_data[i];
			TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

			verify_buf_params.mr_idx = 0;
			verify_buf_params.length = buff_lens[i];
			verify_buf_params.expected_seed = seed_node_a;
			TRACE(ri, util_verify_buf(ri, &verify_buf_params));
		}
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}

int run_fi_tinjectdata(struct rank_info *ri){
	struct wait_cntr_params wait_cntr_params = { 0 };
	struct wait_tx_cq_params wait_tx_cq_params = { 0 };
	struct wait_rx_cq_params wait_rx_cq_params = { 0 };
	struct verify_buf_params verify_buf_params = { 0 };
	const size_t buffer_len = 1024;
	const size_t send_len = 64;
	struct rank_info *pri = NULL;
	const uint64_t tag = 0xffff0001;
	const uint64_t data = 0xf00ba;
	int ret;

	TRACE(ri, util_simple_setup(ri, &pri, buffer_len, FI_SEND, FI_RECV));

	if (my_node == NODE_B) {
		INSIST_FI_EQ(ri,
			     fi_trecv(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
				     buffer_len, NULL, FI_ADDR_UNSPEC, tag, 0xffff,
				     get_ctx_simple(ri, context)),
			     0);
	}

	TRACE(ri, util_barrier(ri));

	if (my_node == NODE_A) {
		SEND_AND_INSIST_EQ(ri, ret,
					fi_tinjectdata(ri->ep_info[0].fid, ri->mr_info[0].uaddr,
					send_len, data, pri->ep_info[0].fi_addr, tag),
				 0);

		// Make sure no completion was generated for the inject.
		wait_tx_cq_params.ep_idx = 0;
		wait_tx_cq_params.context_val = context;
		wait_tx_cq_params.flags = FI_MSG | FI_SEND | FI_INJECT | FI_REMOTE_CQ_DATA;
		wait_tx_cq_params.data = data;
		wait_tx_cq_params.expect_empty = true;
		TRACE(ri, util_wait_tx_cq(ri, &wait_tx_cq_params));

		wait_cntr_params.ep_idx = 0;
		wait_cntr_params.val = 1;
		wait_cntr_params.which = WAIT_CNTR_TX;
		TRACE(ri, util_wait_cntr(ri, &wait_cntr_params));
	}

	if (my_node == NODE_B) {
		wait_rx_cq_params.ep_idx = 0;
		wait_rx_cq_params.context_val = context;
		wait_rx_cq_params.flags = FI_TAGGED | FI_RECV | FI_REMOTE_CQ_DATA;
		wait_rx_cq_params.data = data;
		TRACE(ri, util_wait_rx_cq(ri, &wait_rx_cq_params));

		verify_buf_params.mr_idx = 0;
		verify_buf_params.length = send_len;
		verify_buf_params.expected_seed = seed_node_a;
		TRACE(ri, util_verify_buf(ri, &verify_buf_params));
	}

	TRACE(ri, util_teardown(ri, pri));

	return 0;
}
