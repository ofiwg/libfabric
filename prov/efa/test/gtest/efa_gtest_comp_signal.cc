/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */
/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */

#include "efa_gtest_common_resource.h"
#include "efa_gtest_domain_utils.h"
#include "fi_ext_efa.h"
#include "rdma/fi_ext.h"
#include <rdma/fi_rma.h>
#include <gtest/gtest.h>

using testing::Test;

/*
 * Skip the current test (from the TEST_F body) when the device/build does not
 * support completion-with-signal. Used directly in the body so GTEST_SKIP()
 * returns from the test — GTEST_SKIP() in a helper method does NOT stop the
 * calling body, which would then dereference the un-opened ops table.
 */
#define SKIP_IF_NO_COMP_SIGNAL()                                             \
	do {                                                                 \
		if (!efa_test_device_supports_comp_signal())                 \
			GTEST_SKIP()                                         \
				<< "device/build lacks completion-with-signal"; \
	} while (0)

/*
 * Control-path and endpoint-option tests for EFA completion with signal. These
 * exercise the argument/option validation that fails before any device admin
 * command, so they do not require completion-with-signal-capable hardware.
 */
class EfaCompSignalTest : public Test
{
      protected:
	struct efa_resource resource = {};
	struct fi_efa_ops_signal *sig_ops = nullptr;

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
		ASSERT_EQ(fi_open_ops(&resource.domain->fid, FI_EFA_SIGNAL_OPS,
				      0, (void **) &sig_ops, nullptr),
			  0);
		ASSERT_NE(sig_ops, nullptr);
	}

	/*
	 * Construct an efa-direct endpoint with completion-with-signal enabled
	 * (setopt before fi_enable). Skips the test if the device lacks signal
	 * support so it stays portable across hardware.
	 */
	void construct_signal_enabled()
	{
		bool on = true;

		struct fi_info *hints = efa_test_alloc_default_hints(
			FI_EP_RDM, EFA_DIRECT_FABRIC_NAME);
		ASSERT_NE(hints, nullptr);
		efa_test_resource_construct_no_enable(&resource, hints);
		ASSERT_NE(resource.domain, nullptr);
		ASSERT_NE(resource.ep, nullptr);

		int ret = fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
				    FI_OPT_EFA_COMP_SIGNAL, &on, sizeof(on));
		if (ret == -FI_EOPNOTSUPP)
			GTEST_SKIP() << "device lacks completion-with-signal";
		ASSERT_EQ(ret, 0);
		ASSERT_EQ(fi_enable(resource.ep), 0);
		ASSERT_TRUE(efa_test_get_comp_signal_enabled(resource.ep));
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

/* The signal ops table exposes the completion-with-signal control path. */
TEST_F(EfaCompSignalTest, ops_present)
{
	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_NE(sig_ops->create_comp_mem_op, nullptr);
	EXPECT_NE(sig_ops->register_signal, nullptr);
	EXPECT_NE(sig_ops->query_max_comp_mem_ops, nullptr);
}

/* query_max_comp_mem_ops rejects NULL and, when supported, returns a value. */
TEST_F(EfaCompSignalTest, query_max_comp_mem_ops)
{
	uint32_t max_comp_mem_ops = 0;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_EQ(sig_ops->query_max_comp_mem_ops(resource.domain, nullptr),
		  -FI_EINVAL);
	EXPECT_EQ(sig_ops->query_max_comp_mem_ops(resource.domain,
						  &max_comp_mem_ops),
		  0);
}

/* create_comp_mem_op rejects NULL arguments regardless of hardware. */
TEST_F(EfaCompSignalTest, create_comp_mem_op_invalid_args)
{
	struct fi_efa_comp_mem_op_attr attr = {};
	struct fid_efa_comp_mem_op *mem_op = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_EQ(sig_ops->create_comp_mem_op(resource.domain, nullptr,
					      &mem_op),
		  -FI_EINVAL);
	EXPECT_EQ(sig_ops->create_comp_mem_op(resource.domain, &attr, nullptr),
		  -FI_EINVAL);
}

/*
 * create_comp_mem_op rejects unknown flag bits at the provider layer. The comp
 * and err flags are translated and passed to the device, which enforces which
 * combinations it supports.
 */
TEST_F(EfaCompSignalTest, create_comp_mem_op_flags)
{
	struct fi_efa_comp_mem_op_attr attr = {};
	struct fid_efa_comp_mem_op *mem_op = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	attr.op = FI_EFA_COMP_MEM_OP_SET_SIGNAL_VAL_32;
	attr.location.type = FI_EFA_MEMORY_LOCATION_VA;

	/* Unknown flag bit is rejected by the provider. */
	attr.flags = FI_EFA_COMP_MEM_OP_WITH_COMP_EXTERNAL_MEM | (1 << 5);
	EXPECT_EQ(sig_ops->create_comp_mem_op(resource.domain, &attr, &mem_op),
		  -FI_EINVAL);
}

/* register_signal rejects NULL arguments regardless of hardware. */
TEST_F(EfaCompSignalTest, register_signal_invalid_args)
{
	struct fi_efa_comp_signal_attr attr = {};
	struct fid_efa_comp_signal *signal = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	EXPECT_EQ(sig_ops->register_signal(resource.domain, nullptr, &signal),
		  -FI_EINVAL);
	EXPECT_EQ(sig_ops->register_signal(resource.domain, &attr, nullptr),
		  -FI_EINVAL);
}

/*
 * register_signal rejects a MEM_OP signal with no mem_op handle and a CNTR_INC
 * signal with no counter.
 */
TEST_F(EfaCompSignalTest, register_signal_bad_backing)
{
	struct fi_efa_comp_signal_attr attr = {};
	struct fid_efa_comp_signal *signal = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(false));
	SKIP_IF_NO_COMP_SIGNAL();
	ASSERT_NO_FATAL_FAILURE(open_ops());

	attr.type = FI_EFA_COMP_SIGNAL_MEM_OP;
	attr.mem_op = nullptr;
	EXPECT_EQ(sig_ops->register_signal(resource.domain, &attr, &signal),
		  -FI_EINVAL);

	memset(&attr, 0, sizeof(attr));
	attr.type = FI_EFA_COMP_SIGNAL_CNTR_INC;
	attr.cntr = nullptr;
	EXPECT_EQ(sig_ops->register_signal(resource.domain, &attr, &signal),
		  -FI_EINVAL);
}

/* FI_OPT_EFA_COMP_SIGNAL rejects a wrong optlen. */
TEST_F(EfaCompSignalTest, setopt_bad_optlen)
{
	int intval = 1;

	ASSERT_NO_FATAL_FAILURE(construct(false));

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_EFA_COMP_SIGNAL, &intval, sizeof(intval)),
		  -FI_EINVAL);
}

/*
 * Disabling FI_OPT_EFA_COMP_SIGNAL (optval=false) always succeeds and leaves
 * the endpoint's signal support off.
 */
TEST_F(EfaCompSignalTest, setopt_disable)
{
	bool optval = false;

	ASSERT_NO_FATAL_FAILURE(construct(false));

	EXPECT_FALSE(efa_test_get_comp_signal_enabled(resource.ep));

	EXPECT_EQ(fi_setopt(&resource.ep->fid, FI_OPT_ENDPOINT,
			    FI_OPT_EFA_COMP_SIGNAL, &optval, sizeof(optval)),
		  0);
	EXPECT_FALSE(efa_test_get_comp_signal_enabled(resource.ep));
}

/*
 * FI_EFA_EXTENDED_MSG on fi_writemsg is rejected when the endpoint has not
 * enabled signal support.
 */
TEST_F(EfaCompSignalTest, writemsg_requires_enable)
{
	struct fi_efa_msg_rma emsg = {};
	struct iovec iov = {};
	struct fi_rma_iov rma_iov = {};
	uint8_t buf[8] = {};
	void *desc = nullptr;

	ASSERT_NO_FATAL_FAILURE(construct(true));
	SKIP_IF_NO_COMP_SIGNAL();

	ASSERT_FALSE(efa_test_get_comp_signal_enabled(resource.ep));

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
	emsg.feature_bits = FI_EFA_REMOTE_SIGNAL_ID;
	emsg.remote_signal_id = 7;

	EXPECT_EQ(fi_writemsg(resource.ep, (struct fi_msg_rma *) &emsg,
			      FI_EFA_EXTENDED_MSG),
		  -FI_EINVAL);
}

/*
 * fi_getopt(FI_OPT_TX_SIZE) reports the effective send-queue depth, which must
 * not exceed the endpoint's configured tx size (the signal-mode wide WQEs can
 * only reduce it). Holds whether or not signals are enabled.
 */
TEST_F(EfaCompSignalTest, getopt_tx_size)
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
TEST_F(EfaCompSignalTest, getopt_inject_size)
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
 * With signals enabled the wide WQE reduces both the inline size and the SQ
 * depth. fi_getopt reports the signal-adjusted values, still bounded by the
 * endpoint's configured sizes.
 */
TEST_F(EfaCompSignalTest, getopt_size_signal_enabled)
{
	size_t inject = 0, tx = 0;
	size_t optlen = sizeof(size_t);

	ASSERT_NO_FATAL_FAILURE(construct_signal_enabled());

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

/* The size getopt options reject an undersized optlen. */
TEST_F(EfaCompSignalTest, getopt_size_etoosmall)
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
