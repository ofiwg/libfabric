#include "efa_unit_tests.h"

int main(void)
{
	int ret;
	const struct CMUnitTest efa_unit_tests[] = {
		cmocka_unit_test(test_duplicate_efa_ah_creation),    /* Requires an EFA device to work */
		cmocka_unit_test(test_efa_device_construct_error_handling),    /* Requires an EFA device to work */
		cmocka_unit_test(test_rxr_ep_pkt_pool_flags), /* Requires an EFA device to work */
		cmocka_unit_test(test_rxr_ep_pkt_pool_page_alignment), /* Requires an EFA device to work */
	};
	cmocka_set_message_output(CM_OUTPUT_XML);

	ret = cmocka_run_group_tests_name("efa unit tests", efa_unit_tests, NULL, NULL);

	return ret;
}
