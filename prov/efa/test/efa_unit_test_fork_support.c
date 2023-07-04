#include "efa_unit_tests.h"

/**
 * @brief verify efa_fork_support_request_initialize() set value of g_efa_fork_status correctly
 * 
 * @brief
 * as long as user set FI_EFA_FORK_SAFE to 1, g_efa_fork_status should be
 * EFA_FORK_SUPPORT_ON, even what ibv_is_fork_initialize() return
 * IBV_FOKR_UNNEEDED.
 */
void test_efa_fork_support_request_initialize_when_ibv_fork_unneeded(void **state)
{
	setenv("FI_EFA_FORK_SAFE", "1", true);
	g_efa_unit_test_mocks.ibv_is_fork_initialized = &efa_mock_ibv_is_fork_initialize_return_unneeded;

	efa_fork_support_request_initialize();
	assert_int_equal(g_efa_fork_status, EFA_FORK_SUPPORT_ON);

	g_efa_unit_test_mocks.ibv_is_fork_initialized = __real_ibv_is_fork_initialized;
	unsetenv("FI_EFA_FORK_SAFE");
}