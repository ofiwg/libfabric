/* SPDX-License-Identifier: BSD-2-Clause OR GPL-2.0-only */
/* SPDX-FileCopyrightText: Copyright Amazon.com, Inc. or its affiliates. All rights reserved. */

#include "efa.h"
#include "efa_av.h"
#include "rdm/efa_rdm_av.h"
#include "rdm/efa_rdm_ep.h"
#include "rdm/efa_rdm_peer.h"
#include "efa_gtest_common_helpers.h"
#include "efa_gtest_rdm_av_utils.h"

/*
 * State shared between the setup, the injected reader and the read back. The
 * gtest suite runs one test at a time, so a single set is enough.
 */
static struct efa_rdm_ep *test_ep;
static struct efa_av *test_av;
static struct efa_ep_addr test_raw_addr;
static fi_addr_t test_implicit_fi_addr;
static struct efa_rdm_peer *test_implicit_peer;
static struct efa_test_av_publish_observation test_observation;

int efa_test_av_publish_ordering_setup(struct fid_ep *ep, struct fid_av *av)
{
	struct efa_av *efa_av = container_of(av, struct efa_av, util_av.av_fid);
	struct efa_rdm_av *rdm_av = (struct efa_rdm_av *) efa_av;
	struct efa_rdm_ep *efa_rdm_ep =
		container_of(ep, struct efa_rdm_ep, base_ep.util_ep.ep_fid);
	size_t addr_len = sizeof(test_raw_addr);
	static uint16_t next_qpn = 1;
	fi_addr_t implicit_fi_addr;
	int err;

	memset(&test_observation, 0, sizeof(test_observation));
	test_ep = efa_rdm_ep;
	test_av = efa_av;

	/* Keep the endpoint's own GID so the real ibv_create_ah succeeds, and
	 * vary only qpn/qkey to get a distinct peer address. A fabricated GID
	 * would need ibv_create_ah mocked, and the resulting dummy AH outlives
	 * the test body and would be destroyed for real during teardown. */
	memset(&test_raw_addr, 0, addr_len);
	err = fi_getname(&ep->fid, &test_raw_addr, &addr_len);
	if (err)
		return err;
	test_raw_addr.qpn = next_qpn++;
	test_raw_addr.qkey = 0x5678;

	ofi_genlock_lock(&efa_av->domain->util_domain.lock);
	ofi_genlock_lock(&rdm_av->util_av_implicit.lock);
	err = efa_rdm_av_insert_one_implicit(efa_av, &test_raw_addr,
					     &implicit_fi_addr, 0, NULL);

	/* Give the implicit fi_addr a peer, so the promotion has one to re-key.
	 * Both implicit AV accessors require these locks to be held. */
	if (!err) {
		test_implicit_peer = efa_rdm_ep_get_peer_implicit_unsafe(
			efa_rdm_ep, implicit_fi_addr);
		if (!test_implicit_peer)
			err = -FI_ENOMEM;
	}

	ofi_genlock_unlock(&rdm_av->util_av_implicit.lock);
	ofi_genlock_unlock(&efa_av->domain->util_domain.lock);
	if (err)
		return err;

	test_implicit_fi_addr = implicit_fi_addr;
	return 0;
}

int efa_test_av_publish_ordering_promote(struct fid_av *av,
					fi_addr_t *explicit_fi_addr)
{
	return fi_av_insert(av, &test_raw_addr, 1, explicit_fi_addr, 0, NULL);
}

/* Runs with util_av.lock held by the publish path under test, which the
 * analysis cannot see through the mock trampoline. */
void efa_test_av_publish_ordering_probe(struct efa_av_entry *entry)
	OFI_TSA_NO_ANALYSIS
{
	struct efa_rdm_peer *peer;
	fi_addr_t fi_addr = entry->fi_addr;
	uint16_t ahn = entry->ah->ahn;
	uint16_t qpn = efa_av_entry_ep_addr(entry)->qpn;

	test_observation.ran = true;
	test_observation.explicit_fi_addr = fi_addr;

	/* Lookup 1: must still miss, otherwise this probe is not running before
	 * the entry becomes visible and the test would prove nothing */
	test_observation.reverse_av_resolves =
		efa_rdm_av_reverse_lookup_unsafe(test_av, ahn, qpn, NULL) !=
		FI_ADDR_NOTAVAIL;

	/* Lookup 2: must already hit, which is the ordering under test */
	peer = efa_rdm_ep_peer_map_lookup(test_ep->fi_addr_to_peer_map, fi_addr);
	test_observation.peer_map_resolves = !!peer;
	test_observation.peer_is_migrated_peer = peer == test_implicit_peer;
}

const struct efa_test_av_publish_observation *
efa_test_av_publish_ordering_observation(void)
{
	return &test_observation;
}

bool efa_test_av_publish_ordering_single_peer(fi_addr_t explicit_fi_addr)
{
	return !efa_rdm_ep_peer_map_lookup(test_ep->fi_addr_to_peer_map_implicit,
					   test_implicit_fi_addr) &&
	       efa_rdm_ep_peer_map_lookup(test_ep->fi_addr_to_peer_map,
					  explicit_fi_addr) ==
		       test_implicit_peer;
}

int efa_test_av_publish_ordering_recycle_slot(struct fid_ep *ep,
					      struct fid_av *av,
					      fi_addr_t *recycled_fi_addr)
{
	struct efa_ep_addr throwaway = {0};
	size_t addr_len = sizeof(throwaway);
	fi_addr_t fi_addr;
	int err;

	err = fi_getname(&ep->fid, &throwaway, &addr_len);
	if (err)
		return err;
	throwaway.qpn = 0xfffe;
	throwaway.qkey = 0x9abc;

	if (fi_av_insert(av, &throwaway, 1, &fi_addr, 0, NULL) != 1)
		return -FI_EADDRNOTAVAIL;

	err = fi_av_remove(av, &fi_addr, 1, 0);
	if (err)
		return err;

	*recycled_fi_addr = fi_addr;
	return 0;
}
