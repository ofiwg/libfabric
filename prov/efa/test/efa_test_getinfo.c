#include "efa_unit_tests.h"

struct ibv_device **__real_ibv_get_device_list(int *num_devices);

/* This function is a bit tricky. Due to the fact
 * that it is called during provider init, we can't
 * use cmocka's mock_type and will_return. Thus this function
 * can only return fixed behavior. This is a placeholder
 * until we decide what values we would like to insert.
 */
struct ibv_device **__wrap_ibv_get_device_list(int *num_devices)
{
	return __real_ibv_get_device_list(num_devices);
}

void efa_getinfo_default()
{
	int ret;
	struct fi_info *provider = NULL;
	struct fi_info *hints = fi_allocinfo();
	hints->fabric_attr->prov_name = strdup("efa");

	ret = fi_getinfo(FI_VERSION(1, 14), NULL, NULL, 0ULL, hints, &provider);
	will_return_maybe(__wrap_ibv_get_device_list, 123);
	assert_int_equal(ret, 0);
	assert_non_null(provider);
	assert_string_equal(provider->fabric_attr->prov_name, "efa");
	assert_null(provider->next);
	assert_int_equal(provider->domain_attr->av_type, FI_AV_TABLE);
	assert_int_equal(provider->domain_attr->control_progress, FI_PROGRESS_AUTO);
	assert_int_equal(provider->domain_attr->data_progress, FI_PROGRESS_AUTO);
}
