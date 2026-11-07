/*
 * Copyright (c) 2026 DataDirect Networks, Inc. All rights reserved.
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

/*
 * Endpoint close/EQ event regression test: closing a MSG endpoint must
 * remove the events queued on its EQ that name that endpoint, and only
 * those. A stale event left behind holds a dangling fid, so the next close
 * on the same EQ reads freed memory and a later fi_eq_read returns it.
 *
 * The test runs in one process with no peer. It queues the events with
 * fi_eq_write, which uses the same queue as provider generated events such
 * as FI_SHUTDOWN after an error completion. Build and run it with
 * AddressSanitizer to catch the use after free and the leaked fi_info of
 * a FI_CONNREQ event.
 *
 * The test is provider agnostic. The verbs provider used to leave these
 * events queued and read freed memory on the next close; run it there with
 * -p verbs -s <local RDMA address>.
 */

#include <errno.h>
#include <getopt.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_errno.h>
#include <rdma/fabric.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_PORT "29385"

struct test_resources {
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_eq *eq;
};

static int report(const char *call, int ret)
{
	fprintf(stderr, "%s failed: %s (%d)\n", call,
		ret < 0 ? fi_strerror(-ret) : "unknown error", ret);
	return ret ? ret : -FI_EOTHER;
}

static int setup(struct test_resources *r, const char *provider,
		 const char *node, const char *service)
{
	struct fi_eq_attr eq_attr = {
		.wait_obj = FI_WAIT_UNSPEC,
		.flags = FI_WRITE,
	};
	struct fi_info *hints;
	int ret;

	hints = fi_allocinfo();
	if (!hints)
		return -FI_ENOMEM;

	hints->fabric_attr->prov_name = strdup(provider);
	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	/* Accept the MR modes verbs requires; other providers ignore them. */
	hints->domain_attr->mr_mode = FI_MR_LOCAL | FI_MR_VIRT_ADDR |
				      FI_MR_ALLOCATED | FI_MR_PROV_KEY;

	ret = fi_getinfo(FI_VERSION(1, 9), node, service, FI_SOURCE, hints,
			 &r->info);
	fi_freeinfo(hints);
	if (ret)
		return report("fi_getinfo", ret);

	ret = fi_fabric(r->info->fabric_attr, &r->fabric, NULL);
	if (ret)
		return report("fi_fabric", ret);

	ret = fi_eq_open(r->fabric, &eq_attr, &r->eq, NULL);
	if (ret)
		return report("fi_eq_open", ret);

	ret = fi_domain(r->fabric, r->info, &r->domain, NULL);
	if (ret)
		return report("fi_domain", ret);

	return 0;
}

static void teardown(struct test_resources *r)
{
	if (r->domain)
		fi_close(&r->domain->fid);
	if (r->eq)
		fi_close(&r->eq->fid);
	if (r->fabric)
		fi_close(&r->fabric->fid);
	if (r->info)
		fi_freeinfo(r->info);
}

static int open_ep(struct test_resources *r, struct fid_ep **ep)
{
	int ret;

	ret = fi_endpoint(r->domain, r->info, ep, NULL);
	if (ret)
		return report("fi_endpoint", ret);

	ret = fi_ep_bind(*ep, &r->eq->fid, 0);
	if (ret)
		return report("fi_ep_bind(eq)", ret);

	return 0;
}

static int queue_event(struct test_resources *r, uint32_t event,
		       struct fid_ep *ep, struct fi_info *info)
{
	struct fi_eq_cm_entry entry;
	ssize_t ret;

	memset(&entry, 0, sizeof(entry));
	entry.fid = &ep->fid;
	entry.info = info;

	ret = fi_eq_write(r->eq, event, &entry, sizeof(entry), 0);
	if (ret != sizeof(entry))
		return report("fi_eq_write", (int) ret);

	return 0;
}

/*
 * Read one event. Return 1 and fill event/entry when an event was read,
 * 0 when the EQ is empty, a negative error otherwise.
 */
static int read_event(struct test_resources *r, uint32_t *event,
		      struct fi_eq_cm_entry *entry)
{
	ssize_t ret;

	ret = fi_eq_read(r->eq, event, entry, sizeof(*entry), 0);
	if (ret == -FI_EAGAIN)
		return 0;
	if (ret < 0)
		return report("fi_eq_read", (int) ret);

	return 1;
}

/*
 * Events for two endpoints are queued, including a FI_CONNREQ that owns an
 * fi_info. Closing both endpoints must leave the EQ empty. With the bug,
 * closing B dereferences the freed endpoint A.
 */
static int test_close_purges_own_events(struct test_resources *r)
{
	struct fid_ep *ep_a = NULL, *ep_b = NULL;
	struct fi_eq_cm_entry entry;
	struct fi_info *connreq_info;
	uint32_t event;
	int ret;

	ret = open_ep(r, &ep_a);
	if (ret)
		goto out;
	ret = open_ep(r, &ep_b);
	if (ret)
		goto out;

	connreq_info = fi_dupinfo(r->info);
	if (!connreq_info) {
		ret = report("fi_dupinfo", -FI_ENOMEM);
		goto out;
	}

	ret = queue_event(r, FI_SHUTDOWN, ep_a, NULL);
	if (!ret)
		ret = queue_event(r, FI_SHUTDOWN, ep_b, NULL);
	if (!ret)
		ret = queue_event(r, FI_CONNREQ, ep_a, connreq_info);
	if (ret) {
		fi_freeinfo(connreq_info);
		goto out;
	}

	ret = fi_close(&ep_a->fid);
	ep_a = NULL;
	if (ret) {
		report("fi_close(A)", ret);
		goto out;
	}

	ret = fi_close(&ep_b->fid);
	ep_b = NULL;
	if (ret) {
		report("fi_close(B)", ret);
		goto out;
	}

	ret = read_event(r, &event, &entry);
	if (ret > 0) {
		fprintf(stderr, "stale event %u (fid %p) left on the EQ after "
			"its endpoint was closed\n", event,
			(void *) entry.fid);
		ret = -FI_EOTHER;
	}

out:
	if (ep_a)
		fi_close(&ep_a->fid);
	if (ep_b)
		fi_close(&ep_b->fid);
	return ret;
}

/*
 * Closing C must not remove the event queued for D.
 */
static int test_close_keeps_other_events(struct test_resources *r)
{
	struct fid_ep *ep_c = NULL, *ep_d = NULL;
	struct fi_eq_cm_entry entry;
	uint32_t event;
	int ret;

	ret = open_ep(r, &ep_c);
	if (ret)
		goto out;
	ret = open_ep(r, &ep_d);
	if (ret)
		goto out;

	ret = queue_event(r, FI_SHUTDOWN, ep_d, NULL);
	if (ret)
		goto out;

	ret = fi_close(&ep_c->fid);
	ep_c = NULL;
	if (ret) {
		report("fi_close(C)", ret);
		goto out;
	}

	ret = read_event(r, &event, &entry);
	if (ret < 0)
		goto out;
	if (ret == 0 || event != FI_SHUTDOWN || entry.fid != &ep_d->fid) {
		fprintf(stderr, "event for the open endpoint D was lost or "
			"altered (read %d, event %u, fid %p, want %p)\n", ret,
			ret ? event : 0, ret ? (void *) entry.fid : NULL,
			(void *) &ep_d->fid);
		ret = -FI_EOTHER;
		goto out;
	}
	ret = 0;

out:
	if (ep_c)
		fi_close(&ep_c->fid);
	if (ep_d)
		fi_close(&ep_d->fid);
	return ret;
}

static void usage(const char *prog_name)
{
	fprintf(stderr, "Usage: %s [OPTIONS]\n", prog_name);
	fprintf(stderr, "\n");
	fprintf(stderr, "Options:\n");
	fprintf(stderr, "  -p <provider>  Provider name (default: tcp)\n");
	fprintf(stderr, "  -s <address>   Local address (default: 127.0.0.1)\n");
	fprintf(stderr, "  -P <port>      Port/service (default: %s)\n",
		DEFAULT_PORT);
	fprintf(stderr, "  -h             Display this help\n");
}

int main(int argc, char **argv)
{
	const char *provider = "tcp";
	const char *node = "127.0.0.1";
	const char *service = DEFAULT_PORT;
	struct test_resources r;
	int op, ret;

	while ((op = getopt(argc, argv, "p:s:P:h")) != -1) {
		switch (op) {
		case 'p':
			provider = optarg;
			break;
		case 's':
			node = optarg;
			break;
		case 'P':
			service = optarg;
			break;
		case 'h':
		default:
			usage(argv[0]);
			return EXIT_FAILURE;
		}
	}

	memset(&r, 0, sizeof(r));
	ret = setup(&r, provider, node, service);
	if (ret)
		goto out;

	ret = test_close_purges_own_events(&r);
	printf("close purges own events: %s\n", ret ? "FAIL" : "PASS");
	if (ret)
		goto out;

	ret = test_close_keeps_other_events(&r);
	printf("close keeps other events: %s\n", ret ? "FAIL" : "PASS");

out:
	teardown(&r);
	return ret ? EXIT_FAILURE : EXIT_SUCCESS;
}
