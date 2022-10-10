#include "efa_unit_tests.h"

/*
 * Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with the same raw address,
 * and verifies that returned fi_addr is the same and
 * ibv_create_ah only gets called once.
 */
void test_av_insert_duplicate_raw_addr()
{
	assert_int_equal(1, 0);
	struct efa_resource resource = {0};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	err = efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	assert_int_equal(err, 0);
	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;

	err = fi_getname(&resource.ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	num_addr = fi_av_insert(resource.av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	num_addr = fi_av_insert(resource.av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_equal(addr1, addr2);

	efa_unit_test_resource_destruct(&resource);
}

/*
 * Only works on nodes with EFA devices
 * This test calls fi_av_insert() twice with two difference raw address with same GID,
 * and verifies that returned fi_addr is different and ibv_create_ah only gets called once.
 * this is because libfabric EFA provider has a cache for address handle (AH).
 */
void test_av_insert_duplicate_gid()
{
	struct efa_resource resource = {0};
	struct efa_ep_addr raw_addr = {0};
	size_t raw_addr_len = sizeof(struct efa_ep_addr);
	fi_addr_t addr1, addr2;
	int err, num_addr;

	err = efa_unit_test_resource_construct(&resource, FI_EP_RDM);
	assert_int_equal(err, 0);
	g_efa_unit_test_mocks.ibv_create_ah = &efa_mock_ibv_create_ah_check_mock;

	err = fi_getname(&resource.ep->fid, &raw_addr, &raw_addr_len);
	assert_int_equal(err, 0);
	raw_addr.qpn = 1;
	raw_addr.qkey = 0x1234;

	/* the following will_return ensures ibv_create_ah is called exactly once */
	will_return(efa_mock_ibv_create_ah_check_mock, 0);

	num_addr = fi_av_insert(resource.av, &raw_addr, 1, &addr1, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);

	raw_addr.qpn = 2;
	raw_addr.qkey = 0x5678;
	num_addr = fi_av_insert(resource.av, &raw_addr, 1, &addr2, 0 /* flags */, NULL /* context */);
	assert_int_equal(num_addr, 1);
	assert_int_not_equal(addr1, addr2);

	efa_unit_test_resource_destruct(&resource);
}

