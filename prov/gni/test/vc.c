/*
 * Copyright (c) 2015 Los Alamos National Security, LLC. All rights reserved.
 * Copyright (c) 2015 Cray Inc.  All rights reserved.
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <getopt.h>
#include <poll.h>
#include <time.h>
#include <string.h>
#include <pthread.h>

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include "gnix_vc.h"
#include "gnix_nic.h"
#include "gnix_cm_nic.h"
#include "gnix_hashtable.h"
#include "gnix_av.h"

#include <criterion/criterion.h>

static struct fid_fabric *fab;
static struct fid_domain *dom;
static struct fid_ep *ep[2];
static struct fid_av *av;
static struct fi_info *hints;
static struct fi_info *fi;
void *ep_name[2];
fi_addr_t gni_addr[2];
struct gnix_av_addr_entry * gnix_addr[2];
size_t gnix_addrlen[2];

static void vc_setup_common(void);

static void vc_setup_auto(void)
{

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = 4;
	hints->domain_attr->control_progress = FI_PROGRESS_AUTO;
	hints->mode = ~0;

	vc_setup_common();
}

static void vc_setup_manual(void)
{

	hints = fi_allocinfo();
	cr_assert(hints, "fi_allocinfo");

	hints->domain_attr->cq_data_size = 4;
	hints->domain_attr->control_progress = FI_PROGRESS_MANUAL;
	hints->mode = ~0;
	vc_setup_common();
}

static void vc_setup_common(void)
{
	int ret = 0;
	struct fi_av_attr attr;
	size_t addrlen = 0;
	struct gnix_fid_av *gnix_av;

	hints->fabric_attr->name = strdup("gni");

	ret = fi_getinfo(FI_VERSION(1, 0), NULL, 0, 0, hints, &fi);
	cr_assert(!ret, "fi_getinfo");

	ret = fi_fabric(fi->fabric_attr, &fab, NULL);
	cr_assert(!ret, "fi_fabric");

	ret = fi_domain(fab, fi, &dom, NULL);
	cr_assert(!ret, "fi_domain");

	attr.type = FI_AV_MAP;
	attr.count = 16;

	ret = fi_av_open(dom, &attr, &av, NULL);
	cr_assert(!ret, "fi_av_open");

	gnix_av = container_of(av, struct gnix_fid_av, av_fid);

	ret = fi_endpoint(dom, fi, &ep[0], NULL);
	cr_assert(!ret, "fi_endpoint");

	ret = fi_getname(&ep[0]->fid, NULL, &addrlen);
	cr_assert(addrlen > 0);

	ep_name[0] = malloc(addrlen);
	cr_assert(ep_name[0] != NULL);

	ep_name[1] = malloc(addrlen);
	cr_assert(ep_name[1] != NULL);

	ret = fi_getname(&ep[0]->fid, ep_name[0], &addrlen);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_endpoint(dom, fi, &ep[1], NULL);
	cr_assert(!ret, "fi_endpoint");

	ret = fi_getname(&ep[1]->fid, ep_name[1], &addrlen);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_av_insert(av, ep_name[0], 1, &gni_addr[0], 0,
				NULL);
	cr_assert(ret == 1);

	ret = _gnix_av_lookup(gnix_av, gni_addr[0], &gnix_addr[0]);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_av_insert(av, ep_name[1], 1, &gni_addr[1], 0,
				NULL);
	cr_assert(ret == 1);

	ret = _gnix_av_lookup(gnix_av, gni_addr[1], &gnix_addr[1]);
	cr_assert(ret == FI_SUCCESS);

	ret = fi_ep_bind(ep[0], &av->fid, 0);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_enable(ep[0]);
	cr_assert(!ret, "fi_ep_enable");

	ret = fi_ep_bind(ep[1], &av->fid, 0);
	cr_assert(!ret, "fi_ep_bind");

	ret = fi_enable(ep[1]);
	cr_assert(!ret, "fi_ep_enable");
}

void vc_teardown(void)
{
	int ret = 0;

	ret = fi_close(&ep[0]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&ep[1]->fid);
	cr_assert(!ret, "failure in closing ep.");

	ret = fi_close(&av->fid);
	cr_assert(!ret, "failure in closing av.");

	ret = fi_close(&dom->fid);
	cr_assert(!ret, "failure in closing domain.");

	ret = fi_close(&fab->fid);
	cr_assert(!ret, "failure in closing fabric.");

	fi_freeinfo(fi);
	fi_freeinfo(hints);
	free(ep_name[0]);
	free(ep_name[1]);
}

/*******************************************************************************
 * Test vc functions.
 ******************************************************************************/

TestSuite(vc_management_auto, .init = vc_setup_auto, .fini = vc_teardown,
	  .disabled = false);

TestSuite(vc_management_manual, .init = vc_setup_manual, .fini = vc_teardown,
	  .disabled = false);

Test(vc_management_auto, vc_alloc_simple)
{
	int ret;
	struct gnix_vc *vc[2];
	struct gnix_fid_ep *ep_priv;

	ep_priv = container_of(ep[0], struct gnix_fid_ep, ep_fid);

	ret = _gnix_vc_alloc(ep_priv, gnix_addr[0], &vc[0]);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_alloc(ep_priv, gnix_addr[1], &vc[1]);
	cr_assert_eq(ret, FI_SUCCESS);

	/*
	 * vc_id's have to be different since the
	 * vc's were allocated using the same ep.
	 */
	cr_assert_neq(vc[0]->vc_id, vc[1]->vc_id);

	ret = _gnix_vc_destroy(vc[0]);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_destroy(vc[1]);
	cr_assert_eq(ret, FI_SUCCESS);
}

Test(vc_management_auto, vc_lookup_by_id)
{
	int ret;
	struct gnix_vc *vc[2], *vc_chk;
	struct gnix_fid_ep *ep_priv;

	ep_priv = container_of(ep[0], struct gnix_fid_ep, ep_fid);

	ret = _gnix_vc_alloc(ep_priv, gnix_addr[0], &vc[0]);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_alloc(ep_priv, gnix_addr[1], &vc[1]);
	cr_assert_eq(ret, FI_SUCCESS);

	vc_chk = __gnix_nic_elem_by_rem_id(ep_priv->nic, vc[0]->vc_id);
	cr_assert_eq(vc_chk, vc[0]);

	vc_chk = __gnix_nic_elem_by_rem_id(ep_priv->nic, vc[1]->vc_id);
	cr_assert_eq(vc_chk, vc[1]);

	ret = _gnix_vc_destroy(vc[0]);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_destroy(vc[1]);
	cr_assert_eq(ret, FI_SUCCESS);

}

Test(vc_management_auto, vc_connect)
{
	int ret;
	struct gnix_vc *vc_conn;
	struct gnix_fid_ep *ep_priv[2];
	gnix_ht_key_t key;
	enum gnix_vc_conn_state state;

	ep_priv[0] = container_of(ep[0], struct gnix_fid_ep, ep_fid);

	ep_priv[1] = container_of(ep[1], struct gnix_fid_ep, ep_fid);

	ret = _gnix_vc_alloc(ep_priv[0], gnix_addr[1], &vc_conn);
	cr_assert_eq(ret, FI_SUCCESS);

	memcpy(&key, &gni_addr[1],
		sizeof(gnix_ht_key_t));

	ret = _gnix_ht_insert(ep_priv[0]->vc_ht, key, vc_conn);
	cr_assert_eq(ret, FI_SUCCESS);
	vc_conn->modes |= GNIX_VC_MODE_IN_HT;

	ret = _gnix_vc_connect(vc_conn);
	cr_assert_eq(ret, FI_SUCCESS);

	/*
	 * since we asked for FI_PROGRESS_AUTO for control
	 * we can just spin here.  add a yield in case the
	 * test is only being run on one cpu.
	 */

	state = GNIX_VC_CONN_NONE;
	while (state != GNIX_VC_CONNECTED) {
		pthread_yield();
		state = _gnix_vc_state(vc_conn);
	}

	ret = _gnix_vc_disconnect(vc_conn);
	cr_assert_eq(ret, FI_SUCCESS);

	/* VC is destroyed by the EP */
}

Test(vc_management_auto, vc_connect2)
{
	int ret;
	struct gnix_vc *vc_conn0, *vc_conn1;
	struct gnix_fid_ep *ep_priv[2];
	gnix_ht_key_t key;
	enum gnix_vc_conn_state state;

	ep_priv[0] = container_of(ep[0], struct gnix_fid_ep, ep_fid);
	ep_priv[1] = container_of(ep[1], struct gnix_fid_ep, ep_fid);

	ret = _gnix_vc_alloc(ep_priv[0], gnix_addr[1], &vc_conn0);
	cr_assert_eq(ret, FI_SUCCESS);

	memcpy(&key, &gni_addr[1],
		sizeof(gnix_ht_key_t));

	ret = _gnix_ht_insert(ep_priv[0]->vc_ht, key, vc_conn0);
	cr_assert_eq(ret, FI_SUCCESS);

	vc_conn0->modes |= GNIX_VC_MODE_IN_HT;

	ret = _gnix_vc_alloc(ep_priv[1], gnix_addr[0], &vc_conn1);
	cr_assert_eq(ret, FI_SUCCESS);

	memcpy(&key, &gni_addr[0],
		sizeof(gnix_ht_key_t));

	ret = _gnix_ht_insert(ep_priv[1]->vc_ht, key, vc_conn1);
	cr_assert_eq(ret, FI_SUCCESS);

	vc_conn1->modes |= GNIX_VC_MODE_IN_HT;

	ret = _gnix_vc_connect(vc_conn0);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_connect(vc_conn1);
	cr_assert_eq(ret, FI_SUCCESS);

	/*
	 * since we asked for FI_PROGRESS_AUTO for control
	 * we can just spin here.  add a yield in case the
	 * test is only being run on one cpu.
	 */

	state = GNIX_VC_CONN_NONE;
	while (state != GNIX_VC_CONNECTED) {
		pthread_yield();
		state = _gnix_vc_state(vc_conn0);
	}

	state = GNIX_VC_CONN_NONE;
	while (state != GNIX_VC_CONNECTED) {
		pthread_yield();
		state = _gnix_vc_state(vc_conn1);
	}

	ret = _gnix_vc_disconnect(vc_conn0);
	cr_assert_eq(ret, FI_SUCCESS);

	ret = _gnix_vc_disconnect(vc_conn1);
	cr_assert_eq(ret, FI_SUCCESS);

	/* VC is destroyed by the EP */
}
