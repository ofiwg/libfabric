/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All
 * rights reserved. */

#include "efa_gtest_common_helpers.h"
#include "efa_gtest_common_mocks.h"
#include "efa_gtest_common_resource.h"
#include "efa_gtest_rdm_proto_utils.h"
#include <gtest/gtest.h>
#include <rdma/fi_errno.h>
#include <string>

using testing::DoAll;
using testing::InvokeWithoutArgs;
using testing::Return;
using testing::StrictMock;
using testing::Test;
using testing::TestWithParam;
using testing::Values;

TEST(EfaRdmProtoRxTest, callbacks_preserve_return_values)
{
	struct efa_test_proto_rx_callback_result res = {};

	efa_test_proto_callbacks_preserve_return_values(&res);
	EXPECT_EQ(res.after_robuf_ret, -FI_EAGAIN);
	EXPECT_EQ(res.unexpected_match_ret, 0);
}

class EfaRdmProtoTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		if (!efa_test_proto_medium_len_in_band(
			    resource.ep, EFA_TEST_PROTO_MEDIUM_LEN))
			GTEST_SKIP() << "device's medium band excludes "
				     << EFA_TEST_PROTO_MEDIUM_LEN;
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief The medium protocol fans a message out over packets whose segments
 * tile it exactly once.
 *
 * The receiver reassembles from msg_length and seg_offset alone, so a gap or an
 * overlap silently corrupts the peer's copy.
 */
TEST_F(EfaRdmProtoTest, medium_construct_tiles_message_over_packets)
{
	struct efa_test_proto_construct_result res = {};

	ASSERT_EQ(efa_test_proto_medium_construct(resource.ep, resource.av,
						  resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected_medium);
	ASSERT_EQ(res.ret, 0);

	/* A single packet would mean eager should have won the selection. */
	ASSERT_GT(res.pke_cnt, 1u);
	EXPECT_EQ(res.total_len, EFA_TEST_PROTO_MEDIUM_LEN);

	/*
	 * The peer-abort protocol reads txe->req_pkt_type to tell a two-sided
	 * RTM from an operation it does not handle.
	 */
	EXPECT_EQ(res.req_pkt_type, efa_test_proto_medium_msgrtm_pkt_type());

	EXPECT_EQ(res.callbacks_set, res.pke_cnt);
	EXPECT_EQ(res.ope_backrefs_set, res.pke_cnt);

	uint64_t expected_offset = 0;
	for (size_t i = 0; i < res.pke_cnt; ++i) {
		EXPECT_GT(res.payload_sizes[i], 0u) << "packet " << i;
		EXPECT_EQ(res.msg_lengths[i], res.total_len) << "packet " << i;
		EXPECT_EQ(res.seg_offsets[i], expected_offset) << "packet " << i;
		expected_offset += res.payload_sizes[i];
	}
	EXPECT_EQ(expected_offset, res.total_len);
}

/**
 * @brief construct_tx_pkes() is idempotent, so a txe queued before the
 * handshake can be reposted.
 *
 * efa_rdm_ope_repost_ope_queued_before_handshake() re-enters it on a txe the
 * first attempt already set up, and efa_rdm_ope_process_queued_ope() retries a
 * txe that returned -FI_EAGAIN without clearing its queued state.
 */
TEST_F(EfaRdmProtoTest, medium_construct_is_idempotent_on_repost)
{
	struct efa_test_proto_construct_result first = {}, second = {};

	ASSERT_EQ(efa_test_proto_medium_construct_repost(
			  resource.ep, resource.av, resource.domain, &first,
			  &second),
		  0);
	ASSERT_TRUE(first.selected_medium);
	ASSERT_EQ(first.ret, 0);
	ASSERT_EQ(second.ret, 0);

	ASSERT_GT(first.pke_cnt, 1u);
	EXPECT_EQ(second.pke_cnt, first.pke_cnt);
	EXPECT_EQ(second.req_pkt_type, first.req_pkt_type);
	for (size_t i = 0; i < first.pke_cnt; ++i)
		EXPECT_EQ(second.seg_offsets[i], first.seg_offsets[i])
			<< "packet " << i;

	EXPECT_EQ(second.bytes_sent, second.total_len);
}

struct EfaRdmProtoPlanCase {
	const char *name;
	enum fi_hmem_iface iface;
	int align128;
	size_t total_len;
	int leave_one_tx_pkt;
	ssize_t expected_ret;
	size_t expected_cnt;
	size_t expected_sizes[EFA_TEST_PROTO_MAX_PKES];
};

/**
 * @brief The segmenting decision's packet count and per-packet sizes.
 *
 * Host memory aligns to 8 bytes and CUDA to 64 by default; requesting
 * FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES overrides both with 128, and
 * every packet but the last carries an aligned size.
 */
class EfaRdmProtoMediumPlanTest : public TestWithParam<EfaRdmProtoPlanCase>
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

TEST_P(EfaRdmProtoMediumPlanTest, packet_count_and_sizes)
{
	const EfaRdmProtoPlanCase &c = GetParam();
	struct efa_test_proto_plan_result res = {};

	ASSERT_EQ(efa_test_proto_medium_plan(resource.ep, resource.av, c.iface,
					     c.align128, c.total_len,
					     c.leave_one_tx_pkt, &res),
		  0);
	ASSERT_EQ(res.ret, c.expected_ret);
	if (c.expected_ret)
		return;

	ASSERT_EQ(res.pkt_entry_cnt, c.expected_cnt);

	size_t covered = 0;
	for (size_t i = 0; i < res.pkt_entry_cnt; ++i) {
		EXPECT_EQ(res.data_sizes[i], c.expected_sizes[i])
			<< "packet " << i;
		covered += res.data_sizes[i];
	}
	EXPECT_EQ(covered, c.total_len);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmProtoMediumPlanTest,
	Values(EfaRdmProtoPlanCase{"host_9000", FI_HMEM_SYSTEM, 0, 9000, 0, 0,
				   2, {4496, 4504}},
	       EfaRdmProtoPlanCase{"host_12000", FI_HMEM_SYSTEM, 0, 12000, 0, 0,
				   2, {6000, 6000}},
	       EfaRdmProtoPlanCase{"host_18004", FI_HMEM_SYSTEM, 0, 18004, 0, 0,
				   3, {6000, 6000, 6004}},
	       EfaRdmProtoPlanCase{"host_align128_9000", FI_HMEM_SYSTEM, 1, 9000,
				   0, 0, 2, {4480, 4520}},
	       EfaRdmProtoPlanCase{"host_align128_12000", FI_HMEM_SYSTEM, 1,
				   12000, 0, 0, 2, {5888, 6112}},
	       EfaRdmProtoPlanCase{"host_align128_18004", FI_HMEM_SYSTEM, 1,
				   18004, 0, 0, 3, {5888, 5888, 6228}},
	       EfaRdmProtoPlanCase{"cuda_12000", FI_HMEM_CUDA, 0, 12000, 0, 0, 2,
				   {5952, 6048}},
	       EfaRdmProtoPlanCase{"cuda_align128_12000", FI_HMEM_CUDA, 1, 12000,
				   0, 0, 2, {5888, 6112}},
	       EfaRdmProtoPlanCase{"declines_without_enough_tx_pkts",
				   FI_HMEM_SYSTEM, 0, 9000, 1, -FI_EAGAIN, 0,
				   {}}),
	[](const testing::TestParamInfo<EfaRdmProtoPlanCase> &info) {
		return std::string(info.param.name);
	});

class EfaRdmProtoMediumCompletionTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		if (!efa_test_proto_medium_len_in_band(
			    resource.ep, EFA_TEST_PROTO_MEDIUM_LEN))
			GTEST_SKIP() << "device's medium band excludes "
				     << EFA_TEST_PROTO_MEDIUM_LEN;

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief A medium message completes only once every one of its packets has
 * reported its send completion.
 *
 * An early completion has to leave the txe on the endpoint's list; releasing it
 * there would free a txe the packets still in flight hold a reference to.
 */
TEST_F(EfaRdmProtoMediumCompletionTest, completes_only_after_last_packet)
{
	struct efa_test_proto_completion_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillRepeatedly(Return(0));

	ASSERT_EQ(efa_test_proto_medium_completion(resource.ep, resource.av,
						   resource.domain, &res),
		  0);
	ASSERT_GT(res.pke_cnt, 1u);
	EXPECT_EQ(res.total_len, EFA_TEST_PROTO_MEDIUM_LEN);
	EXPECT_EQ(res.ope_list_after_send, 1u);

	uint64_t acked = 0;
	for (size_t i = 0; i < res.pke_cnt; ++i) {
		ASSERT_GT(res.payload_sizes[i], 0u) << "packet " << i;
		acked += res.payload_sizes[i];

		if (i + 1 < res.pke_cnt) {
			EXPECT_EQ(res.bytes_acked_after[i], acked)
				<< "packet " << i;
			EXPECT_EQ(res.ope_list_after[i], 1u) << "packet " << i;
		}
	}

	EXPECT_EQ(acked, res.total_len);
	/* The last packet acknowledged the message and reaped the txe. */
	EXPECT_EQ(res.ope_list_after[res.pke_cnt - 1], 0u);
}

/**
 * @brief A medium txe whose source MR was closed mid-transfer is completed
 * exactly once by the peer-abort drain helper, even though its remaining
 * packets complete successfully.
 *
 * This is why the EFA_RDM_OPE_PEER_ABORT_PENDING branch is not dead code the
 * way it is for the single-packet eager protocol: one medium message is several
 * packets sharing one txe, so an early failure can mark the txe peer-aborting
 * while the later packets are still in flight and go on to report success.
 * Without the branch the first such success walks into
 * efa_rdm_ope_handle_send_completed(), which asserts the flag is clear and
 * double-completes the txe.
 */
TEST_F(EfaRdmProtoMediumCompletionTest, peer_abort_completes_txe_once)
{
	struct efa_test_proto_peer_abort_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send).WillRepeatedly(Return(0));

	ASSERT_EQ(efa_test_proto_medium_peer_abort(resource.ep, resource.av,
						   resource.cq, resource.domain,
						   &res),
		  0);
	ASSERT_GT(res.pke_cnt, 1u);
	EXPECT_TRUE(res.req_pkt_type_is_rtm);

	/* Marked aborting, but every data WR is still outstanding. */
	EXPECT_TRUE(res.abort_pending_after_error);
	EXPECT_FALSE(res.emitted_after_error);
	EXPECT_EQ(res.readerr_after_error, -FI_EAGAIN);

	/* Only the WR that drains the txe may emit; the earlier ones stay silent. */
	for (size_t i = 0; i + 1 < res.pke_cnt; ++i) {
		EXPECT_FALSE(res.emitted_after[i]) << "packet " << i;
		EXPECT_EQ(res.readerr_after[i], -FI_EAGAIN) << "packet " << i;
	}
	EXPECT_TRUE(res.emitted_after[res.pke_cnt - 1]);

	/* The completion still waits for the PEER_ERROR_PKT to drain. */
	EXPECT_EQ(res.ope_list_after_data, 1u);
	EXPECT_EQ(res.readerr_after_data, -FI_EAGAIN);

	/* Its send completion releases the txe and writes the one error entry. */
	EXPECT_EQ(res.ope_list_final, 0u);
	ASSERT_EQ(res.readerr_final, 1);
	EXPECT_EQ(res.final_err, FI_ECANCELED);
	EXPECT_EQ(res.final_prov_errno, efa_test_proto_peer_abort_prov_errno());
}

struct EfaRdmProtoForceCase {
	const char *name;
	/* FI_EFA_RDM_FORCE_SEND_PROTO, nullptr when unset. */
	const char *forced;
	size_t len;
	/* Protocol expected, nullptr when the send must fail instead. */
	const char *expected_proto;
};

class EfaRdmProtoForceTest : public TestWithParam<EfaRdmProtoForceCase>
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

/**
 * @brief FI_EFA_RDM_FORCE_SEND_PROTO replaces the size decision with a name
 * match, so the named protocol is selected whether or not the size suits it,
 * and a name no registered protocol answers to fails the send.
 *
 * Overriding the size decision is the point: no threshold can steer a message
 * that fits in one packet away from eager, which claims it first.
 */
TEST_P(EfaRdmProtoForceTest, forced_name_replaces_the_size_decision)
{
	const EfaRdmProtoForceCase &c = GetParam();
	struct efa_test_proto_force_result res = {};

	ASSERT_EQ(efa_test_proto_force_select_send(resource.ep, resource.av,
						   resource.domain, c.forced,
						   c.len, &res),
		  0);

	// TODO: after all protocols are migrated, protocol selection should
	// fail with -FI_EINVAL
	if (!c.expected_proto) {
		EXPECT_EQ(res.ret, FI_SUCCESS);
		EXPECT_STREQ(res.proto_name, "");
		return;
	}

	ASSERT_EQ(res.ret, 0);
	EXPECT_STREQ(res.proto_name, c.expected_proto);
}

INSTANTIATE_TEST_SUITE_P(
	, EfaRdmProtoForceTest,
	Values(EfaRdmProtoForceCase{"unset_picks_eager", nullptr, 64, "eager"},
	       EfaRdmProtoForceCase{"eager_picks_eager", "eager", 64, "eager"},
	       EfaRdmProtoForceCase{"medium_beats_eager", "medium", 64, "medium"},
	       /* Both of these decline under can_use_protocol(). */
	       EfaRdmProtoForceCase{"eager_ignores_multi_packet_size", "eager",
				    EFA_TEST_PROTO_MEDIUM_LEN, "eager"},
	       EfaRdmProtoForceCase{"medium_ignores_zero_length", "medium", 0,
				    "medium"},
	       /* A read protocol's predicate would decline a 64 byte send. */
	       EfaRdmProtoForceCase{"runtread_picks_runtread", "runtread", 64,
				    "runtread"},
	       EfaRdmProtoForceCase{"longread_picks_longread", "longread", 64,
				    "longread"},
	       EfaRdmProtoForceCase{"longcts_picks_longcts", "longcts", 64,
				    "longcts"},
	       /* A write protocol's name is not a send protocol's name. */
	       EfaRdmProtoForceCase{"eager_write_fails", "eager_write", 64,
				    nullptr},
	       EfaRdmProtoForceCase{"unknown_name_fails", "no_such_proto", 64,
				    nullptr}),
	[](const testing::TestParamInfo<EfaRdmProtoForceCase> &info) {
		return std::string(info.param.name);
	});

class EfaRdmProtoReadTest : public Test
{
	protected:
	struct efa_resource resource = {};
	bool globals_saved = false;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		if (!efa_test_proto_device_supports_rdma_read())
			GTEST_SKIP() << "device does not support RDMA read";

		if (!efa_test_proto_large_len_beyond_medium(resource.ep))
			GTEST_SKIP() << "device's medium band includes "
				     << EFA_TEST_PROTO_LARGE_LEN;

		efa_test_proto_save_globals();
		globals_saved = true;
	}

	void TearDown() override
	{
		if (globals_saved)
			efa_test_proto_restore_globals();
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaRdmProtoReadTest, longread_selected_for_large_msg)
{
	struct efa_test_proto_select_result res = {};

	ASSERT_EQ(efa_test_proto_select(resource.ep, resource.av,
					resource.domain, 1, 0, 1, 0, &res),
		  0);
	ASSERT_EQ(res.ret, 0);
	EXPECT_STREQ(res.proto_name, "longread");
}

TEST_F(EfaRdmProtoReadTest, longread_declined_without_mr)
{
	struct efa_test_proto_select_result res = {};

	ASSERT_EQ(efa_test_proto_select(resource.ep, resource.av,
					resource.domain, 1, 0, 0, 0, &res),
		  0);
	ASSERT_EQ(res.ret, 0);
	EXPECT_STRNE(res.proto_name, "longread");
	EXPECT_STRNE(res.proto_name, "runtread");
}

TEST_F(EfaRdmProtoReadTest, runtread_declined_for_delivery_complete)
{
	struct efa_test_proto_select_result res = {};

	ASSERT_EQ(efa_test_proto_select(resource.ep, resource.av,
					resource.domain, 1,
					EFA_TEST_PROTO_RUNT_SIZE, 1,
					FI_DELIVERY_COMPLETE, &res),
		  0);
	ASSERT_EQ(res.ret, 0);
	EXPECT_STREQ(res.proto_name, "longread");
}

TEST_F(EfaRdmProtoReadTest, longread_construct_builds_one_pke)
{
	struct efa_test_proto_longread_construct_result res = {};

	ASSERT_EQ(efa_test_longread_construct_builds_one_pke(
			  resource.ep, resource.av, resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected);
	ASSERT_EQ(res.ret, 0);

	EXPECT_EQ(res.pke_cnt, 1u);
	EXPECT_EQ(res.req_pkt_type, res.expected_req_pkt_type);
	EXPECT_TRUE(res.handler_set);
	EXPECT_TRUE(res.ope_set);
	EXPECT_EQ(res.payload_size, 0u);
	EXPECT_EQ(res.msg_length, res.total_len);
	EXPECT_EQ(res.send_id, res.tx_id);
	EXPECT_EQ(res.read_iov_count, res.iov_count);
	EXPECT_EQ(res.read_iov_addr, res.buf_addr);
	EXPECT_EQ(res.read_iov_len, res.buf_len);
	EXPECT_EQ(res.read_iov_key, res.mr_key);
	EXPECT_EQ(res.pkt_size, res.expected_pkt_size);
}

TEST_F(EfaRdmProtoReadTest, longread_construct_is_repeatable)
{
	struct efa_test_proto_repeat_result res = {};

	ASSERT_EQ(efa_test_longread_construct_is_repeatable(
			  resource.ep, resource.av, resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected);
	ASSERT_EQ(res.first_ret, 0);
	ASSERT_EQ(res.second_ret, 0);

	EXPECT_EQ(res.first_pke_cnt, 1u);
	EXPECT_TRUE(res.read_msg_counted_after_first_hook);
	EXPECT_EQ(res.read_msg_in_flight_after_first_hook, 1);

	EXPECT_EQ(res.second_pke_cnt, 1u);
	EXPECT_EQ(res.second_req_pkt_type, res.first_req_pkt_type);
	EXPECT_EQ(res.second_msg_length, res.total_len);
	EXPECT_EQ(res.read_msg_in_flight_after_second_hook, 1);
}

TEST_F(EfaRdmProtoReadTest, runtread_construct_is_repeatable)
{
	struct efa_test_proto_repeat_result res = {};

	ASSERT_EQ(efa_test_runtread_construct_is_repeatable(
			  resource.ep, resource.av, resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected);
	ASSERT_EQ(res.first_ret, 0);
	ASSERT_EQ(res.second_ret, 0);

	ASSERT_GT(res.first_pke_cnt, 1u);
	EXPECT_EQ(res.bytes_sent_after_first_hook, res.first_bytes_runt);
	EXPECT_TRUE(res.read_msg_counted_after_first_hook);
	EXPECT_EQ(res.read_msg_in_flight_after_first_hook, 1);

	ASSERT_EQ(res.second_pke_cnt, res.first_pke_cnt);
	EXPECT_EQ(res.second_req_pkt_type, res.first_req_pkt_type);
	EXPECT_EQ(res.second_bytes_runt, res.first_bytes_runt);
	for (size_t i = 0; i < res.first_pke_cnt; ++i)
		EXPECT_EQ(res.second_seg_offsets[i], res.first_seg_offsets[i])
			<< "packet " << i;

	EXPECT_EQ(res.bytes_sent_after_second_hook,
		  res.bytes_sent_after_first_hook);
	EXPECT_EQ(res.read_msg_in_flight_after_second_hook, 1);
	EXPECT_STREQ(res.next_proto_name, "longread");
}

class EfaRdmProtoLongctsTest : public Test
{
	protected:
	struct efa_resource resource = {};
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);

		if (!efa_test_proto_large_len_beyond_medium(resource.ep))
			GTEST_SKIP() << "device's medium band includes "
				     << EFA_TEST_PROTO_LARGE_LEN;

		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaRdmProtoLongctsTest, longcts_selected_for_large_msg)
{
	struct efa_test_proto_select_result res = {};

	ASSERT_EQ(efa_test_proto_select(resource.ep, resource.av,
					resource.domain, 0, 0, 1, 0, &res),
		  0);
	ASSERT_EQ(res.ret, 0);
	EXPECT_STREQ(res.proto_name, "longcts");
}

TEST_F(EfaRdmProtoLongctsTest, longcts_construct_builds_one_pke)
{
	struct efa_test_proto_longcts_construct_result res = {};

	ASSERT_EQ(efa_test_longcts_construct_builds_one_pke(
			  resource.ep, resource.av, resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected);
	ASSERT_EQ(res.ret, 0);

	EXPECT_EQ(res.pke_cnt, 1u);
	EXPECT_EQ(res.req_pkt_type, res.expected_req_pkt_type);
	EXPECT_TRUE(res.handler_set);
	EXPECT_TRUE(res.ope_set);
	EXPECT_GT(res.payload_size, 0u);
	EXPECT_LT(res.payload_size, res.total_len);
	EXPECT_EQ(res.msg_length, res.total_len);
	EXPECT_EQ(res.send_id, res.tx_id);
	EXPECT_EQ(res.credit_request, res.expected_credit_request);
}

TEST_F(EfaRdmProtoLongctsTest, longcts_construct_is_repeatable)
{
	struct efa_test_proto_repeat_result res = {};

	ASSERT_EQ(efa_test_longcts_construct_is_repeatable(
			  resource.ep, resource.av, resource.domain, &res),
		  0);
	ASSERT_TRUE(res.selected);
	ASSERT_EQ(res.first_ret, 0);
	ASSERT_EQ(res.second_ret, 0);

	EXPECT_EQ(res.first_pke_cnt, 1u);
	EXPECT_EQ(res.bytes_sent_after_first_hook, res.first_payload_size);

	EXPECT_EQ(res.second_pke_cnt, 1u);
	EXPECT_EQ(res.second_req_pkt_type, res.first_req_pkt_type);
	EXPECT_EQ(res.second_payload_size, res.first_payload_size);
	EXPECT_EQ(res.second_msg_length, res.total_len);
	EXPECT_EQ(res.bytes_sent_after_second_hook, res.first_payload_size);
}

TEST_F(EfaRdmProtoLongctsTest, longcts_peer_abort_completes_txe_once)
{
	struct efa_test_proto_longcts_peer_abort_result res = {};

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.Times(2)
		.WillRepeatedly(Return(0));

	ASSERT_EQ(efa_test_longcts_peer_abort_completes_txe_once(
			  resource.ep, resource.av, resource.cq,
			  resource.domain, &res),
		  0);
	ASSERT_EQ(res.pke_cnt, 1u);
	EXPECT_TRUE(res.handler_set);
	EXPECT_EQ(res.req_pkt_type, res.expected_req_pkt_type);
	EXPECT_LT(res.payload_size, res.total_len);

	EXPECT_TRUE(res.abort_pending_after_error);
	EXPECT_FALSE(res.emitted_after_error);
	EXPECT_EQ(res.readerr_after_error, -FI_EAGAIN);

	EXPECT_TRUE(res.emitted_after_req);
	EXPECT_EQ(res.ope_list_after_req, 1u);
	EXPECT_EQ(res.readerr_after_req, -FI_EAGAIN);

	ASSERT_EQ(res.readerr_final, 1);
	EXPECT_EQ(res.final_err, FI_ECANCELED);
	EXPECT_EQ(res.final_prov_errno, efa_test_proto_peer_abort_prov_errno());
	EXPECT_EQ(res.readerr_after_final, -FI_EAGAIN);
	EXPECT_EQ(res.ope_list_final, 0u);
}

class EfaRdmProtoReadNackReqTest : public Test
{
	protected:
	struct efa_resource resource = {};

	void SetUp() override
	{
		memset(&resource, 0, sizeof(resource));
		efa_test_resource_construct(
			&resource, efa_test_alloc_default_hints(
					   FI_EP_RDM, EFA_FABRIC_NAME));
		ASSERT_NE(resource.ep, nullptr);
	}

	void TearDown() override
	{
		efa_test_resource_destruct(&resource);
	}
};

TEST_F(EfaRdmProtoReadNackReqTest, longcts_read_nack_req_completes_txe_last)
{
	struct efa_test_proto_read_nack_req_result res = {};

	ASSERT_EQ(efa_test_longcts_read_nack_req_completes_txe_last(
			  resource.ep, resource.av, resource.cq,
			  resource.domain, &res),
		  0);
	ASSERT_EQ(res.ret, 0);
	ASSERT_EQ(res.pke_cnt, 1u);
	EXPECT_TRUE(res.handler_set);
	EXPECT_EQ(res.payload_size, 0u);
	EXPECT_EQ(res.hdr_type, res.expected_hdr_type);
	EXPECT_TRUE(res.hdr_read_nack_flag);

	EXPECT_EQ(res.ope_list_after_ctsdata, 1u);
	EXPECT_EQ(res.cq_read_after_ctsdata, -FI_EAGAIN);

	EXPECT_EQ(res.ope_list_after_req, 0u);
	ASSERT_EQ(res.cq_read_after_req, 1);
	EXPECT_TRUE(res.cq_entry_is_send);
}

class EfaRdmProtoReadNackTest : public EfaRdmProtoReadTest
{
	protected:
	StrictMock<MockEfa> mock_efa;

	void SetUp() override
	{
		EfaRdmProtoReadTest::SetUp();
		if (IsSkipped() || HasFatalFailure())
			return;
		MockEfa::set(&mock_efa);
	}

	void TearDown() override
	{
		MockEfa::set(nullptr);
		EfaRdmProtoReadTest::TearDown();
	}
};

TEST_F(EfaRdmProtoReadNackTest, longcts_read_nack_continues_in_api_path)
{
	struct efa_test_proto_read_nack_result res = {};
	size_t posts = 0;

	EFA_EXPECT_CALL(mock_efa, efa_qp_post_send)
		.WillRepeatedly(DoAll(InvokeWithoutArgs([&posts]() { ++posts; }),
				      Return(0)));

	ASSERT_EQ(efa_test_longcts_read_nack_continues_in_api_path(
			  resource.ep, resource.av, resource.cq,
			  resource.domain, &res),
		  0);
	ASSERT_GT(res.runt_pke_cnt, 1u);
	EXPECT_EQ(res.runt_req_pkt_type, res.expected_runt_req_pkt_type);
	EXPECT_TRUE(res.read_msg_counted_before);
	EXPECT_EQ(res.bytes_sent_before, res.bytes_runt);
	EXPECT_LT(res.bytes_sent_before, res.total_len);

	EXPECT_TRUE(res.read_nack_flag);
	EXPECT_TRUE(res.proto_is_longcts);
	EXPECT_EQ(res.req_pkt_type, res.expected_req_pkt_type);
	EXPECT_FALSE(res.read_msg_counted_after);
	EXPECT_EQ(res.read_msg_in_flight_after, 0);

	ASSERT_EQ(res.req_pke_cnt, 1u);
	EXPECT_TRUE(res.req_handler_set);
	EXPECT_TRUE(res.req_ope_set);
	EXPECT_EQ(res.req_payload_size, 0u);
	EXPECT_EQ(res.bytes_sent_after, res.bytes_sent_before);

	EXPECT_EQ(res.hdr_type, res.expected_req_pkt_type);
	EXPECT_TRUE(res.hdr_read_nack_flag);
	EXPECT_EQ(res.hdr_msg_id, res.msg_id);
	EXPECT_EQ(res.hdr_msg_length, res.total_len);
	EXPECT_EQ(res.hdr_send_id, res.tx_id);

	EXPECT_EQ(res.ope_list_after_completions, 1u);
	EXPECT_EQ(res.cq_read_after_completions, -FI_EAGAIN);
	EXPECT_EQ(posts, res.runt_pke_cnt + 1);
}
