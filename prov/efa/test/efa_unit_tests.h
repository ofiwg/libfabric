#ifndef EFA_UNIT_TESTS_H
#define EFA_UNIT_TESTS_H

#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "stdio.h"
#include "efa.h"
#include "rxr.h"

struct efa_resource {
	struct fi_info* info;
	struct fid_fabric* fabric;
	struct fid_domain* domain;
	struct fid_av* av;
};

void test_duplicate_efa_ah_creation();

void test_efa_device_construct_error_handling();

void test_rxr_ep_pkt_pool_flags();

void test_rxr_ep_pkt_pool_page_alignment();

void test_rxr_ep_dc_atomic_error_handling();

int efa_unit_test_resource_construct(struct efa_resource* resource);

void efa_unit_test_resource_destroy(struct efa_resource* resource);

#endif
