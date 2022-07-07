#include "efa_unit_tests.h"

static int efa_unit_test_mocks_reset(void **state)
{
	efa_mock_ibv_send_wr_list_destruct(&g_ibv_send_wr_list);

	g_efa_unit_test_mocks = (struct efa_unit_test_mocks) {
		.ibv_create_ah = __real_ibv_create_ah,
		.efadv_query_device = __real_efadv_query_device,
	};

	return 0;
}

int main(void)
{
	int ret;
	/* Requires an EFA device to work */
	const struct CMUnitTest efa_unit_tests[] = {
		cmocka_unit_test_setup_teardown(test_av_insert_duplicate_raw_addr, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_av_insert_duplicate_gid, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_efa_device_construct_error_handling, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rxr_ep_pkt_pool_flags, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rxr_ep_pkt_pool_page_alignment, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rxr_ep_dc_atomic_error_handling, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_dgram_cq_read_empty_cq, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_dgram_cq_read_bad_wc_status, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_empty_cq, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_failed_poll, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_bad_send_status, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_bad_recv_status, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_recover_forgotten_peer_ah, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_rdm_cq_read_ignore_removed_peer, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_info_open_ep_with_wrong_info, efa_unit_test_mocks_reset, NULL),
		cmocka_unit_test_setup_teardown(test_info_open_ep_with_api_1_1_info, efa_unit_test_mocks_reset, NULL),
	};

	cmocka_set_message_output(CM_OUTPUT_XML);

	ret = cmocka_run_group_tests_name("efa unit tests", efa_unit_tests, NULL, NULL);

	return ret;
}
