/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_common_helpers.h"
#include "efa_gtest_domain_utils.h"
#include "fi_ext_efa.h"
#include "rdma/fi_ext.h"
#include <rdma/fi_rma.h>
#include <gtest/gtest.h>

using testing::_;
using testing::Invoke;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::Truly;

/*
 * Behavior tests for the EFA completion-action control path and endpoint
 * options. Rather than skip on hardware/builds without support, each test
 * asserts the documented behavior for BOTH cases:
 *   - efa_test_build_has_comp_action(): whether the provider was compiled with
 *     efadv support. When false the domain action ops are stubs returning
 *     -FI_ENOSYS.
 *   - efa_test_device_supports_comp_action(): whether the selected device (and
 *     build) advertise the capability. Governs whether enabling the endpoint
 *     option succeeds (0) or is rejected (-FI_EOPNOTSUPP).
 */
class EfaCompActionTest : public Test
{
      protected:
	struct efa_resource resource = {};
	struct fi_efa_ops_mem_comp_action *action_ops = nullptr;

	/*
	 * An attribute set the provider accepts, so that a test can change a
	 * single member and be sure the rejection it asserts comes from that
	 * member. A 4-byte aligned target is taken from the fixture rather than
	 * the stack so that the misalignment test can offset it by one.
	 */
	uint32_t target[2] = {};

	void init_attr(struct fi_efa_mem_comp_action_attr *attr)
	{
		memset(attr, 0, sizeof(*attr));
		attr->op = FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL;
		attr->location.type = FI_EFA_MEMORY_LOCATION_VA;
		attr->location.ptr = (uint8_t *) target;
		attr->op_mem_size = sizeof(uint32_t);
		attr->mem_size = sizeof(uint32_t);
	}

	/* Construct an efa-direct domain; enable the ep unless enable=false. */
	void construct(bool enable)
	{
		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);

		if (enable)
			efa_test_resource_construct(&resource, hints);
		else
			efa_test_resource_construct_no_enable(&resource, hints);
		ASSERT_NE(resource.domain, nullptr);
		ASSERT_NE(resource.ep, nullptr);
	}

	void open_ops()
	{
		ASSERT_EQ(fi_open_ops(&resource.domain->fid,
				      FI_EFA_MEM_COMP_ACTION_OPS, 0,
				      (void **) &action_ops, nullptr),
			  0);
		ASSERT_NE(action_ops, nullptr);
	}

	/*
	 * Construct an efa-direct endpoint and attempt to enable
	 * completion-action (setopt before fi_enable). Asserts the setopt
	 * result matches device support, enables the endpoint, and returns
	 * whether actions ended up enabled so the caller can branch.
	 */
	bool construct_action_enabled()
	{
		bool on = true;

		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		EXPECT_NE(hints, nullptr);
		efa_test_resource_construct_no_enable(&resource, hints);
		EXPECT_NE(resource.domain, nullptr);
		EXPECT_NE(resource.ep, nullptr);

		/* Queried only after fi_getinfo has selected a device: the
		 * device query asserts on an empty device list. */
		bool supported = efa_test_device_supports_comp_action();

		int ret = fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
				    FI_OPT_EFA_COMP_ACTION, &on, sizeof(on));
		if (supported)
			EXPECT_EQ(ret, 0);
		else
			EXPECT_EQ(ret, -FI_EOPNOTSUPP);

		EXPECT_EQ(fi_enable(resource.ep), 0);
		EXPECT_EQ(efa_test_get_comp_action_enabled(resource.ep),
			  supported);
		return supported;
	}

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/* The action ops table is exposed on any efa-direct domain (the individual
 * ops degrade to -FI_ENOSYS when the build lacks support). */
TEST_F(EfaCompActionTest, ops_present)
{
	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_NE(action_ops->create_mem_comp_action, nullptr);
	EXPECT_NE(action_ops->query_max_mem_comp_actions, nullptr);
	EXPECT_NE(action_ops->query_comp_action_block_offset, nullptr);
}

/*
 * query_max_mem_comp_actions: with build support it rejects NULL (-FI_EINVAL) and
 * returns a value for a valid pointer; without build support both calls return
 * -FI_ENOSYS from the stub.
 */
TEST_F(EfaCompActionTest, query_max_mem_comp_actions)
{
	uint32_t max_mem_comp_actions = 0;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	if (efa_test_build_has_comp_action()) {
		EXPECT_EQ(action_ops->query_max_mem_comp_actions(resource.domain,
								nullptr),
			  -FI_EINVAL);
		EXPECT_EQ(action_ops->query_max_mem_comp_actions(
				  resource.domain, &max_mem_comp_actions),
			  0);
	} else {
		EXPECT_EQ(action_ops->query_max_mem_comp_actions(resource.domain,
								nullptr),
			  -FI_ENOSYS);
		EXPECT_EQ(action_ops->query_max_mem_comp_actions(
				  resource.domain, &max_mem_comp_actions),
			  -FI_ENOSYS);
	}
}

/* The offset query is the one op in the table that reads device state through
 * efadv, so it gets the mock. It names an endpoint rather than the domain: the
 * action block belongs to the endpoint's send queue. */
class EfaCompActionBlockOffsetTest : public EfaCompActionTest
{
      protected:
	StrictMock<MockEfa> mock_efa;

	void TearDown() override
	{
		MockEfa::set(nullptr);
		EfaCompActionTest::TearDown();
	}
};

/*
 * The mocked offset matches no plausible derivation from the queue entry
 * layout, so a provider computing one instead of passing the device's value
 * through fails. The mock goes live only once the endpoint is built, since
 * building it goes through the same verb.
 */
TEST_F(EfaCompActionBlockOffsetTest, block_offset_is_the_devices)
{
	uint16_t block_offset = 0;

	ASSERT_NO_FATAL_FAILURE(construct(true));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	if (!efa_test_build_has_comp_action()) {
		EXPECT_EQ(action_ops->query_comp_action_block_offset(
				  resource.ep, &block_offset),
			  -FI_ENOSYS);
		return;
	}

	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efadv_query_qp_wqs, _, _, _, _)
		.WillOnce(Invoke(efa_test_mock_efadv_query_qp_wqs));

	EXPECT_EQ(action_ops->query_comp_action_block_offset(resource.ep,
							     &block_offset),
		  0);
	EXPECT_EQ(block_offset, efa_test_mock_efadv_action_block_offset());
}

TEST_F(EfaCompActionTest, block_offset_requires_enabled_ep)
{
	uint16_t block_offset = 0;
	int expected = efa_test_build_has_comp_action() ? -FI_EINVAL
							: -FI_ENOSYS;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_EQ(action_ops->query_comp_action_block_offset(resource.ep,
							     &block_offset),
		  expected);
}

/*
 * create_mem_comp_action: with build support NULL args are rejected
 * (-FI_EINVAL); without build support the stub returns -FI_ENOSYS.
 */
TEST_F(EfaCompActionTest, create_mem_comp_action_invalid_args)
{
	struct fi_efa_mem_comp_action_attr attr;
	struct fid_efa_comp_action *action = nullptr;
	int expected = efa_test_build_has_comp_action() ? -FI_EINVAL
							: -FI_ENOSYS;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());
	init_attr(&attr);

	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, nullptr,
						     &action),
		  expected);
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     nullptr),
		  expected);
}

/*
 * create_mem_comp_action versioning: comp_mask, flags and reserved are all
 * reserved to be zero today, so a set bit in any of them is rejected
 * (-FI_EINVAL) rather than silently ignored. Without build support the stub returns -FI_ENOSYS.
 */
TEST_F(EfaCompActionTest, create_mem_comp_action_reserved_fields)
{
	struct fi_efa_mem_comp_action_attr attr;
	struct fid_efa_comp_action *action = nullptr;
	int expected = efa_test_build_has_comp_action() ? -FI_EINVAL
							: -FI_ENOSYS;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	init_attr(&attr);
	attr.comp_mask = 1;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.flags = 1;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.reserved[3] = 1;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);
}

/*
 * The device writes op_mem_size bytes, so op_mem_size must be 1, 2 or 4,
 * mem_size must be a non-zero multiple of it, and a VA target must be aligned
 * to it; each is rejected with -FI_EINVAL. A region larger than one write is
 * not supported yet (-FI_EOPNOTSUPP). Without build support the stub returns
 * -FI_ENOSYS.
 */
TEST_F(EfaCompActionTest, create_mem_comp_action_bad_region)
{
	struct fi_efa_mem_comp_action_attr attr;
	struct fid_efa_comp_action *action = nullptr;
	int expected = efa_test_build_has_comp_action() ? -FI_EINVAL
							: -FI_ENOSYS;
	int expected_larger = efa_test_build_has_comp_action() ? -FI_EOPNOTSUPP
							       : -FI_ENOSYS;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	init_attr(&attr);
	attr.op_mem_size = 3;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.op_mem_size = 0;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.mem_size = 0;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.mem_size = 6;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);

	init_attr(&attr);
	attr.mem_size = 2 * attr.op_mem_size;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected_larger);

	/* Aligned target, offset by one byte. */
	init_attr(&attr);
	attr.location.ptr = ((uint8_t *) target) + 1;
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);
}

/*
 * An op outside enum fi_efa_mem_comp_action_op -- what an application built
 * against a newer header would pass -- is rejected (-FI_EINVAL) rather than
 * mapped onto some other device op. Without build support the stub returns
 * -FI_ENOSYS.
 */
TEST_F(EfaCompActionTest, create_mem_comp_action_unknown_op)
{
	struct fi_efa_mem_comp_action_attr attr;
	struct fid_efa_comp_action *action = nullptr;
	int expected = efa_test_build_has_comp_action() ? -FI_EINVAL
							: -FI_ENOSYS;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	ASSERT_NO_FATAL_FAILURE(open_ops());

	init_attr(&attr);
	attr.op = (enum fi_efa_mem_comp_action_op) (
		FI_EFA_MEM_COMP_ACTION_SET_INITIATOR_VAL + 1);
	EXPECT_EQ(action_ops->create_mem_comp_action(resource.domain, &attr,
						     &action),
		  expected);
}

/* FI_OPT_EFA_COMP_ACTION rejects a wrong optlen. */
TEST_F(EfaCompActionTest, setopt_bad_optlen)
{
	int intval = 1;

	ASSERT_NO_FATAL_FAILURE(construct(false));

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_EFA_COMP_ACTION, &intval, sizeof(intval)),
		  -FI_EINVAL);
}

/*
 * Disabling FI_OPT_EFA_COMP_ACTION (optval=false) always succeeds and leaves
 * the endpoint's action support off.
 */
TEST_F(EfaCompActionTest, setopt_disable)
{
	bool optval = false;

	ASSERT_NO_FATAL_FAILURE(construct(false));

	EXPECT_FALSE(efa_test_get_comp_action_enabled(resource.ep));

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_EFA_COMP_ACTION, &optval, sizeof(optval)),
		  0);
	EXPECT_FALSE(efa_test_get_comp_action_enabled(resource.ep));
}

/*
 * Enabling FI_OPT_EFA_COMP_ACTION succeeds (and flips the flag) when the device
 * supports it, and is rejected with -FI_EOPNOTSUPP otherwise.
 */
TEST_F(EfaCompActionTest, setopt_enable)
{
	bool optval = true;

	ASSERT_NO_FATAL_FAILURE(construct(false));

	/* Queried only after fi_getinfo has selected a device: the device
	 * query asserts on an empty device list. */
	bool supported = efa_test_device_supports_comp_action();

	if (supported) {
		EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
				    FI_OPT_EFA_COMP_ACTION, &optval,
				    sizeof(optval)),
			  0);
		EXPECT_TRUE(efa_test_get_comp_action_enabled(resource.ep));
	} else {
		EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
				    FI_OPT_EFA_COMP_ACTION, &optval,
				    sizeof(optval)),
			  -FI_EOPNOTSUPP);
		EXPECT_FALSE(efa_test_get_comp_action_enabled(resource.ep));
	}
}

/*
 * FI_EFA_MSG_ACTION on fi_writemsg is rejected when action support is not
 * enabled on the endpoint, whether or not the build has comp-action support (a
 * build without it rejects the flag outright). Which error comes back depends
 * on the device: with RDMA write the flag itself is rejected (-FI_EINVAL),
 * while a device without it leaves FI_RMA out of the endpoint's caps, so the
 * write is turned away earlier by efa_rma_check_cap (-FI_EOPNOTSUPP).
 */
TEST_F(EfaCompActionTest, writemsg_requires_enable)
{
	struct fi_efa_msg_rma_action emsg = {};
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t buf[8] = {};
	void *desc = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(true));

	/* Checked after construct: the device list is only populated by
	 * fi_getinfo. A device without RDMA write is turned away by
	 * efa_rma_check_cap one step earlier, with a different code. */
	int expected_err = efa_test_device_supports_rma() ? -FI_EINVAL
							  : -FI_EOPNOTSUPP;

	ASSERT_FALSE(efa_test_get_comp_action_enabled(resource.ep));

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	rma_iov.addr = 0x1000;
	rma_iov.len = sizeof(buf);
	rma_iov.key = 0x1;

	emsg.msg.msg_iov = &iov;
	emsg.msg.iov_count = 1;
	emsg.msg.desc = &desc;
	emsg.msg.addr = 0;
	emsg.msg.rma_iov = &rma_iov;
	emsg.msg.rma_iov_count = 1;
	emsg.feature_bits = FI_EFA_REMOTE_ACTION_ID;
	emsg.remote_id = 7;

	EXPECT_EQ(fi_writemsg(resource.ep, (struct fi_msg_rma *) &emsg,
			      FI_EFA_MSG_ACTION),
		  expected_err);
}

/*
 * A feature bit beyond the four this header defines -- what an application
 * built against a newer fi_ext_efa.h would set -- is rejected with
 * -FI_EOPNOTSUPP rather than being ignored. The check runs ahead of the
 * endpoint state check and is compiled into every build, so the result depends
 * on neither this device nor the build configuration. A device without RDMA
 * write returns the same code from efa_rma_check_cap instead, one step earlier.
 */
TEST_F(EfaCompActionTest, writemsg_unknown_feature_bits)
{
	struct fi_efa_msg_rma_action emsg = {};
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t buf[8] = {};
	void *desc = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct_action_enabled());

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	rma_iov.addr = 0x1000;
	rma_iov.len = sizeof(buf);
	rma_iov.key = 0x1;

	emsg.msg.msg_iov = &iov;
	emsg.msg.iov_count = 1;
	emsg.msg.desc = &desc;
	emsg.msg.addr = 0;
	emsg.msg.rma_iov = &rma_iov;
	emsg.msg.rma_iov_count = 1;
	/* First bit past the ones this header defines. */
	emsg.feature_bits = FI_EFA_REMOTE_ACTION_ID | (1ULL << 6);
	emsg.remote_id = 7;

	EXPECT_EQ(fi_writemsg(resource.ep, (struct fi_msg_rma *) &emsg,
			      FI_EFA_MSG_ACTION),
		  -FI_EOPNOTSUPP);
}

/*
 * An action's value is carried in the same device operand as its id, so
 * FI_EFA_*_ACTION_VALUE without the matching FI_EFA_*_ACTION_ID names no action
 * and is rejected with -FI_EINVAL. A device without RDMA write returns
 * -FI_EOPNOTSUPP from efa_rma_check_cap one step earlier instead.
 */
TEST_F(EfaCompActionTest, writemsg_value_without_id)
{
	struct fi_efa_msg_rma_action emsg = {};
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t buf[8] = {};
	void *desc = nullptr;

	construct_action_enabled();

	int expected_err = efa_test_device_supports_rma() ? -FI_EINVAL
							 : -FI_EOPNOTSUPP;

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	rma_iov.addr = 0x1000;
	rma_iov.len = sizeof(buf);
	rma_iov.key = 0x1;

	emsg.msg.msg_iov = &iov;
	emsg.msg.iov_count = 1;
	emsg.msg.desc = &desc;
	emsg.msg.addr = 0;
	emsg.msg.rma_iov = &rma_iov;
	emsg.msg.rma_iov_count = 1;
	emsg.feature_bits = FI_EFA_LOCAL_ACTION_VALUE;
	emsg.local_value = 42;

	EXPECT_EQ(fi_writemsg(resource.ep, (struct fi_msg_rma *) &emsg,
			      FI_EFA_MSG_ACTION),
		  expected_err);
}

class EfaCompActionWriteTest : public EfaCompActionTest
{
      protected:
	StrictMock<MockEfa> mock_efa;

	void TearDown() override
	{
		MockEfa::set(nullptr);
		EfaCompActionTest::TearDown();
	}
};

TEST_F(EfaCompActionWriteTest, writemsg_attaches_actions)
{
	const uint64_t fb = FI_EFA_LOCAL_ACTION_ID | FI_EFA_LOCAL_ACTION_VALUE |
			    FI_EFA_REMOTE_ACTION_ID | FI_EFA_REMOTE_ACTION_VALUE;
	struct fi_efa_msg_rma_action emsg = {};
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t buf[8] = {};
	struct fid_mr *mr = nullptr;
	void *desc = nullptr;
	fi_addr_t peer_addr = FI_ADDR_NOTAVAIL;

	if (!construct_action_enabled())
		GTEST_SKIP() << "device without completion action support";
	if (!efa_test_device_supports_rma())
		GTEST_SKIP() << "device without RDMA write";

	ASSERT_EQ(efa_test_av_insert_self(resource.ep, resource.av,
					  &peer_addr),
		  1);
	ASSERT_EQ(fi_mr_reg(resource.domain, buf, sizeof(buf), FI_WRITE, 0, 0,
			    0, &mr, nullptr),
		  0);
	desc = fi_mr_desc(mr);

	iov.iov_base = buf;
	iov.iov_len = sizeof(buf);
	rma_iov.addr = 0x1000;
	rma_iov.len = sizeof(buf);
	rma_iov.key = 0x1;

	emsg.msg.msg_iov = &iov;
	emsg.msg.iov_count = 1;
	emsg.msg.desc = &desc;
	emsg.msg.addr = peer_addr;
	emsg.msg.rma_iov = &rma_iov;
	emsg.msg.rma_iov_count = 1;
	emsg.feature_bits = fb;
	emsg.local_id = 7;
	emsg.local_value = 0x1122334455667788ULL;
	emsg.remote_id = 9;
	emsg.remote_value = 0xa5;

	auto sig_matches = Truly([fb](const struct efa_comp_action_wr *sig) {
		return efa_test_comp_action_wr_matches(
			sig, fb, 7, 0x1122334455667788ULL, 9, 0xa5);
	});
	MockEfa::set(&mock_efa);
	EFA_EXPECT_CALL(mock_efa, efa_qp_post_write, _, _, _, _, _, 0x1u,
			0x1000u, _, _, _, _, _, _, sig_matches)
		.WillOnce(Return(0));

	EXPECT_EQ(fi_writemsg(resource.ep, (struct fi_msg_rma *) &emsg,
			      FI_EFA_MSG_ACTION),
		  0);

	MockEfa::set(nullptr);
	fi_close(&mr->fid);
}

/*
 * fi_getopt(FI_OPT_TX_SIZE) reports the effective send-queue depth, which must
 * not exceed the endpoint's configured tx size (the action-mode wide WQEs can
 * only reduce it). Holds whether or not actions are enabled.
 */
TEST_F(EfaCompActionTest, getopt_tx_size)
{
	size_t tx_size = 0;
	size_t optlen = sizeof(tx_size);

	ASSERT_NO_FATAL_FAILURE(construct(true));

	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT, FI_OPT_TX_SIZE,
			    &tx_size, &optlen),
		  0);
	EXPECT_EQ(optlen, sizeof(tx_size));
	EXPECT_GT(tx_size, 0u);
	EXPECT_LE(tx_size, resource.info->tx_attr->size);
}

/*
 * fi_getopt on the inject-size options reports the effective inline size, which
 * must not exceed the endpoint's configured inject size.
 */
TEST_F(EfaCompActionTest, getopt_inject_size)
{
	size_t val = 0;
	size_t optlen = sizeof(val);

	ASSERT_NO_FATAL_FAILURE(construct(true));

	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &val, &optlen),
		  0);
	EXPECT_EQ(optlen, sizeof(val));
	EXPECT_LE(val, resource.info->tx_attr->inject_size);

	val = 0;
	optlen = sizeof(val);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_RMA_SIZE, &val, &optlen),
		  0);
	EXPECT_EQ(optlen, sizeof(val));
	EXPECT_LE(val, resource.info->tx_attr->inject_size);
}

/*
 * After attempting to enable actions, fi_getopt reports inject/tx sizes bounded
 * by the endpoint's configured sizes. When the device supports actions the
 * values reflect the action-adjusted (wide-WQE) reduction; otherwise they are
 * the regular values. Either way the bounds hold.
 */
TEST_F(EfaCompActionTest, getopt_size_action_enabled)
{
	size_t inject = 0, tx = 0;
	size_t optlen;

	construct_action_enabled();

	optlen = sizeof(inject);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &inject, &optlen),
		  0);
	EXPECT_LE(inject, resource.info->tx_attr->inject_size);

	optlen = sizeof(tx);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT, FI_OPT_TX_SIZE,
			    &tx, &optlen),
		  0);
	EXPECT_GT(tx, 0u);
	EXPECT_LE(tx, resource.info->tx_attr->size);
}

/*
 * The action block shares the send queue entry with the inline payload, so an
 * endpoint with actions enabled has less inline room than it was created with,
 * and fi_setopt cannot take it back up: the share is not negotiable. Both inject
 * options refuse the created size once actions are enabled.
 */
TEST_F(EfaCompActionTest, setopt_inject_size_bounded_by_action)
{
	struct fi_info *hints;
	ssize_t plain_max = efa_test_device_max_inline_data(0);
	ssize_t action_max = efa_test_device_max_inline_data(1);
	size_t configured, msg_enabled = 0, rma_enabled = 0, reported = 0;
	size_t optlen;
	bool on = true;

	if (!efa_test_device_supports_comp_action() || plain_max <= 0 ||
	    action_max <= 0)
		GTEST_SKIP() << "device without completion action support";
	if (plain_max <= action_max)
		GTEST_SKIP() << "device's action block costs no inline room";

	hints = efa_test_alloc_default_hints(FI_EP_RDM,
					     EFA_DIRECT_FABRIC_NAME);
	ASSERT_NE(hints, nullptr);
	/* The widest inline the device offers without an action block, which is
	 * more than it offers with one, and above the regular inline buffer
	 * size so the endpoint also gets inline rma room. */
	hints->tx_attr->inject_size = (size_t) plain_max;
	efa_test_resource_construct_no_enable(&resource, hints);
	ASSERT_NE(resource.ep, nullptr);

	configured = resource.info->tx_attr->inject_size;
	ASSERT_GT(configured, (size_t) action_max);

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_EFA_COMP_ACTION, &on, sizeof(on)),
		  0);

	optlen = sizeof(msg_enabled);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &msg_enabled, &optlen),
		  0);
	EXPECT_EQ(msg_enabled, (size_t) action_max);

	optlen = sizeof(rma_enabled);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_RMA_SIZE, &rma_enabled, &optlen),
		  0);
	EXPECT_EQ(rma_enabled, (size_t) action_max);

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &configured,
			    sizeof(configured)),
		  -FI_EINVAL);
	optlen = sizeof(reported);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &reported, &optlen),
		  0);
	EXPECT_EQ(reported, msg_enabled);

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_RMA_SIZE, &configured,
			    sizeof(configured)),
		  -FI_EINVAL);
	optlen = sizeof(reported);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_RMA_SIZE, &reported, &optlen),
		  0);
	EXPECT_EQ(reported, rma_enabled);
}

/* The size getopt options reject an undersized optlen. */
TEST_F(EfaCompActionTest, getopt_size_etoosmall)
{
	uint8_t tiny = 0;
	size_t optlen = sizeof(tiny);

	ASSERT_NO_FATAL_FAILURE(construct(true));

	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_MSG_SIZE, &tiny, &optlen),
		  -FI_ETOOSMALL);
	optlen = sizeof(tiny);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_INJECT_RMA_SIZE, &tiny, &optlen),
		  -FI_ETOOSMALL);
	optlen = sizeof(tiny);
	EXPECT_EQ(fi_getopt(&resource.ep->fid, FI_OPT_ENDPOINT, FI_OPT_TX_SIZE,
			    &tiny, &optlen),
		  -FI_ETOOSMALL);
}
