/*
 * SPDX-License-Identifier: GPL-2.0
 *
 * Copyright (c) 2021 Hewlett Packard Enterprise Development LP
 */
#include <stdio.h>
#include <stdlib.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>

#include <ofi.h>

#include "cxip.h"
#include "cxip_test_common.h"

TestSuite(ctrl, .init = cxit_setup_rma, .fini = cxit_teardown_rma,
	  .timeout = CXIT_DEFAULT_TIMEOUT);

Test(ctrl, zb_send)
{
	struct cxip_ep *ep;
	struct cxip_zb_coll_state *zbcoll;
	uint32_t dstnid;
	union cxip_match_bits mb;
	int ret;

	ep = container_of(cxit_ep, struct cxip_ep, ep);
	zbcoll = &ep->ep_obj->zb_coll;
	dstnid = ep->ep_obj->src_addr.nic;
	mb.zb_data = 0xffff;

	ret = cxip_zb_coll_send(zbcoll, dstnid, mb);
	printf("cxip_zb_coll_send() = %d\n", ret);
	printf("About to progress\n");
	usleep(2000);
	cxip_ep_ctrl_progress(ep->ep_obj);
	usleep(2000);
	cxip_ep_ctrl_progress(ep->ep_obj);
	usleep(2000);
	cxip_ep_ctrl_progress(ep->ep_obj);
	usleep(2000);
	printf("Completed\n");
}
