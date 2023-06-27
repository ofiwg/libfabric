#include "efa_unit_tests.h"

void test_efa_rdm_ope_prepare_to_post_send_impl(struct efa_resource *resource,
						enum fi_hmem_iface iface,
						size_t total_len,
						int expected_ret,
						int expected_pkt_entry_cnt,
						int *expected_pkt_entry_data_size_vec)
{
	struct efa_ep_addr raw_addr;
	struct efa_mr mock_mr;
	struct efa_rdm_ope mock_txe;
	size_t raw_addr_len;
	fi_addr_t addr;
	int pkt_entry_cnt, pkt_entry_data_size_vec[1024];
	int i, err, ret;

	ret = fi_getname(&resource->ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(ret, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;
	ret = fi_av_insert(resource->av, &raw_addr, 1, &addr, 0 /* flags */, NULL /* context */);
	assert_int_equal(ret, 1);

	mock_mr.peer.iface = iface;

	memset(&mock_txe, 0, sizeof(mock_txe));
	mock_txe.total_len = total_len;
	mock_txe.addr = addr;
	mock_txe.iov_count = 1;
	mock_txe.iov[0].iov_base = NULL;
	mock_txe.iov[0].iov_len = 9000;
	mock_txe.desc[0] = &mock_mr;

	mock_txe.ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	err = efa_rdm_ope_prepare_to_post_send(&mock_txe,
					       EFA_RDM_MEDIUM_MSGRTM_PKT,
					       &pkt_entry_cnt,
					       pkt_entry_data_size_vec);

	assert_int_equal(err, expected_ret);

	if (err)
		return;

	assert_int_equal(pkt_entry_cnt, expected_pkt_entry_cnt);

	for (i = 0; i < pkt_entry_cnt; ++i)
		assert_int_equal(pkt_entry_data_size_vec[i], expected_pkt_entry_data_size_vec[i]);
}

/**
 * @brief verify efa_rdm_ope_prepare_to_post_send()'s return code
 *
 * Verify that efa_rdm_ope_prepare_to_post_send() will return
 * -FI_EAGAIN, when there is not enough TX packet available,
 */
void test_efa_rdm_ope_prepare_to_post_send_with_no_enough_tx_pkts(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->efa_outstanding_tx_ops = efa_rdm_ep->efa_max_outstanding_tx_ops - 1;
	/* we need at least 2 packets to send this message, but only 1 is available,
	 * therefore efa_rdm_ope_prepare_to_post_send() should return
	 * -FI_EAGAIN.
	 */
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   9000, -FI_EAGAIN, -1, NULL);
	efa_rdm_ep->efa_outstanding_tx_ops = 0;
}

/**
 * @brief verify the pkt_entry_cnt and data size for host memory
 */
void test_efa_rdm_ope_prepare_to_post_send_host_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	/* data size should be aligned and evenly distributed.
	 * alignment for host memory is 8 byte by default.
	 */
	msg_length = 9000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 4496;
	expected_pkt_entry_data_size_vec[1] = 4504;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 6000;
	expected_pkt_entry_data_size_vec[1] = 6000;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 18004;
	expected_pkt_entry_cnt = 3;
	expected_pkt_entry_data_size_vec[0] = 6000;
	expected_pkt_entry_data_size_vec[1] = 6000;
	expected_pkt_entry_data_size_vec[2] = 6004;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

}

/**
 * @brief verify the pkt_entry_cnt and data size for host memory when align128 was requested
 */
void test_efa_rdm_ope_prepare_to_post_send_host_memory_align128(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = true;

	/* if user requested 128 byte alignment, then all but the last
	 * last packet's data size should be 128 aligned
	 */
	msg_length = 9000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 4480;
	expected_pkt_entry_data_size_vec[1] = 4520;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 6112;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);

	msg_length = 18004;
	expected_pkt_entry_cnt = 3;
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 5888;
	expected_pkt_entry_data_size_vec[2] = 6228;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_SYSTEM,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}

/**
 * @brief verify the pkt_entry_cnt and data size for cuda memory
 */
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM);

	/* default alignment of cuda memory is 64 bytes */
	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	expected_pkt_entry_data_size_vec[0] = 5952;
	expected_pkt_entry_data_size_vec[1] = 6048;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_CUDA,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}

/**
 * @brief verify the pkt_entry_cnt and data size for cuda memory when align128 was requested
 */
void test_efa_rdm_ope_prepare_to_post_send_cuda_memory_align128(struct efa_resource **state)
{
	struct efa_resource *resource = *state;
	struct efa_rdm_ep *efa_rdm_ep;
	size_t msg_length;
	int expected_pkt_entry_cnt;
	int expected_pkt_entry_data_size_vec[1024];

	efa_unit_test_resource_construct(resource, FI_EP_RDM);
	efa_rdm_ep = container_of(resource->ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	efa_rdm_ep->sendrecv_in_order_aligned_128_bytes = true;

	msg_length = 12000;
	expected_pkt_entry_cnt = 2;
	/* if user requested 128 byte alignment, then all but the last
	 * last packet's data size should be 128 aligned
	 */
	expected_pkt_entry_data_size_vec[0] = 5888;
	expected_pkt_entry_data_size_vec[1] = 6112;
	test_efa_rdm_ope_prepare_to_post_send_impl(resource, FI_HMEM_CUDA,
						   msg_length,
						   0,
						   expected_pkt_entry_cnt,
						   expected_pkt_entry_data_size_vec);
}
