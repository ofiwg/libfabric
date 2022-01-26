#include "efa_unit_tests.h"
int main(void)
{
	int ret;
	const struct CMUnitTest getinfo_tests[] = {
		cmocka_unit_test(efa_getinfo_default),
	};
	cmocka_set_message_output(CM_OUTPUT_XML);

	ret = cmocka_run_group_tests_name("efa getinfo tests", getinfo_tests, NULL, NULL);
	if (ret != 0)
		return ret;
	return ret;
}
