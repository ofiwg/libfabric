#include "efa_unit_tests.h"

int main(void)
{
	int ret;
	/* Requires an EFA device to work */
	const struct CMUnitTest efa_unit_tests[] = {
		cmocka_unit_test(test_duplicate_efa_ah_creation),
		cmocka_unit_test(test_efa_device_construct_error_handling),
		cmocka_unit_test(test_rxr_ep_pkt_pool_flags),
		cmocka_unit_test(test_rxr_ep_pkt_pool_page_alignment),
		cmocka_unit_test(test_rxr_ep_dc_atomic_error_handling),
		cmocka_unit_test(test_dgram_ep_progress_happy),
		cmocka_unit_test(test_dgram_ep_progress_with_empty_cq),
		cmocka_unit_test(test_dgram_ep_progress_encounter_bad_wc_status),
		cmocka_unit_test(test_rdm_ep_progress_send_completion_happy),
		cmocka_unit_test(test_rdm_ep_progress_recv_completion_happy),
		cmocka_unit_test(test_rdm_ep_progress_send_empty_cq),
		cmocka_unit_test(test_rdm_ep_progress_failed_poll),
		cmocka_unit_test(test_rdm_ep_progress_bad_send_wc_status),
		cmocka_unit_test(test_rdm_ep_progress_bad_recv_wc_status),
	};

	cmocka_set_message_output(CM_OUTPUT_XML);

	ret = cmocka_run_group_tests_name("efa unit tests", efa_unit_tests, NULL, NULL);

	return ret;
}
