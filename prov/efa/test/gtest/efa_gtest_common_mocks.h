/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#ifndef EFA_GTEST_COMMON_MOCKS_H
#define EFA_GTEST_COMMON_MOCKS_H

#include <gmock/gmock.h>
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>

struct efa_av;
struct efa_cur_reverse_av;
struct efa_prv_reverse_av;
struct efa_conn;
struct ofi_mr_map;
struct fi_mr_attr;

/*
 * X macro infrastructure for mock generation.
 *
 * To add a new mocked function:
 *   1. Define EFA_MOCK_PARAMS_<name> and EFA_MOCK_ARGS_<name> below
 *   2. Add X(ret, name) to EFA_MOCK_FUNCTIONS
 *   3. Add any needed forward struct declarations above
 *   4. Add -Wl,--wrap=<name> to prov_efa_test_gtest_efa_gtest_LDFLAGS
 *      in prov/efa/Makefile.include
 */

/* --- Per-function parameter definitions --- */

#define EFA_MOCK_PARAMS_ibv_create_ah \
	struct ibv_pd *pd, struct ibv_ah_attr *attr
#define EFA_MOCK_ARGS_ibv_create_ah pd, attr

#define EFA_MOCK_PARAMS_ibv_destroy_ah struct ibv_ah *ibv_ah
#define EFA_MOCK_ARGS_ibv_destroy_ah   ibv_ah

#define EFA_MOCK_PARAMS_efadv_query_ah \
	struct ibv_ah *ibv_ah, struct efadv_ah_attr *attr, uint32_t inlen
#define EFA_MOCK_ARGS_efadv_query_ah ibv_ah, attr, inlen

#define EFA_MOCK_PARAMS_efa_av_reverse_av_add                          \
	struct efa_av *av, struct efa_cur_reverse_av **cur_reverse_av, \
		struct efa_prv_reverse_av **prv_reverse_av,            \
		struct efa_conn *conn
#define EFA_MOCK_ARGS_efa_av_reverse_av_add \
	av, cur_reverse_av, prv_reverse_av, conn

#define EFA_MOCK_PARAMS_ofi_mr_map_insert                                  \
	struct ofi_mr_map *map, const struct fi_mr_attr *attr,             \
		uint64_t *key, void *context, uint64_t flags
#define EFA_MOCK_ARGS_ofi_mr_map_insert map, attr, key, context, flags

/* --- Function list --- */

#define EFA_MOCK_FUNCTIONS(X)             \
	X(struct ibv_ah *, ibv_create_ah) \
	X(int, ibv_destroy_ah)            \
	X(int, efadv_query_ah)            \
	X(int, efa_av_reverse_av_add)     \
	X(int, ofi_mr_map_insert)

/* --- Generator macros --- */

#define EFA_MOCK_GEN_METHOD(ret, name) \
	MOCK_METHOD(ret, name, (EFA_MOCK_PARAMS_##name));

#define EFA_MOCK_GEN_REAL_DECL(ret, name) \
	ret __real_##name(EFA_MOCK_PARAMS_##name);

// Generate the definitions of the MOCK_METHOD's in the MockEfa class
class MockEfa
{
	public:
	EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_METHOD)

	static MockEfa *get();
	static void set(MockEfa *instance);
};

// Generate the declarations of the real function prototypes
extern "C" {
EFA_MOCK_FUNCTIONS(EFA_MOCK_GEN_REAL_DECL)
}

#endif /* EFA_GTEST_COMMON_MOCKS_H */
