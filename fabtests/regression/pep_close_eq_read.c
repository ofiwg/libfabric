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
 * PEP close/EQ read regression test: xnet must not use an unowned connection
 * handle after its PEP has been closed. Build and run with AddressSanitizer
 * when validating the provider lifetime fix.
 */

#include <errno.h>
#include <netinet/in.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fabric.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define DEFAULT_PORT "29384"
#define LOOPBACK "127.0.0.1"

struct test_resources {
	struct fi_info *info;
	struct fid_fabric *fabric;
	struct fid_domain *domain;
	struct fid_ep *ep;
	struct fid_pep *pep;
	struct fid_eq *eq;
	struct fid_cq *cq;
};

static struct fi_eq_attr eq_attr = {
	.wait_obj = FI_WAIT_NONE,
};

static int report(const char *call, int ret)
{
	fprintf(stderr, "%s failed: %s (%d)\n", call,
		ret < 0 ? fi_strerror(-ret) : "unknown error", ret);
	return ret;
}

static void close_resources(struct test_resources *r)
{
	if (r->ep)
		fi_close(&r->ep->fid);
	if (r->cq)
		fi_close(&r->cq->fid);
	if (r->pep)
		fi_close(&r->pep->fid);
	if (r->domain)
		fi_close(&r->domain->fid);
	if (r->eq)
		fi_close(&r->eq->fid);
	if (r->fabric)
		fi_close(&r->fabric->fid);
	if (r->info)
		fi_freeinfo(r->info);
	memset(r, 0, sizeof(*r));
}

static struct fi_info *make_hints(void)
{
	struct fi_info *hints = fi_allocinfo();

	if (!hints)
		return NULL;
	hints->fabric_attr->prov_name = strdup("tcp");
	hints->ep_attr->type = FI_EP_MSG;
	hints->caps = FI_MSG;
	hints->addr_format = FI_SOCKADDR_IN;
	return hints;
}

static int setup_server(struct test_resources *r, const char *port)
{
	struct fi_info *hints;
	int ret;

	hints = make_hints();
	if (!hints)
		return -FI_ENOMEM;
	ret = fi_getinfo(FI_VERSION(1, 9), LOOPBACK, port, FI_SOURCE,
			 hints, &r->info);
	fi_freeinfo(hints);
	if (ret)
		return report("fi_getinfo(server)", ret);
	ret = fi_fabric(r->info->fabric_attr, &r->fabric, NULL);
	if (ret)
		return report("fi_fabric(server)", ret);
	ret = fi_eq_open(r->fabric, &eq_attr, &r->eq, NULL);
	if (ret)
		return report("fi_eq_open(server)", ret);
	ret = fi_passive_ep(r->fabric, r->info, &r->pep, NULL);
	if (ret)
		return report("fi_passive_ep", ret);
	ret = fi_pep_bind(r->pep, &r->eq->fid, 0);
	if (ret)
		return report("fi_pep_bind", ret);
	ret = fi_listen(r->pep);
	if (ret)
		return report("fi_listen", ret);
	return 0;
}

static int setup_client(struct test_resources *r, const char *port)
{
	struct fi_info *hints;
	int ret;

	hints = make_hints();
	if (!hints)
		return -FI_ENOMEM;
	ret = fi_getinfo(FI_VERSION(1, 9), LOOPBACK, port, 0,
			 hints, &r->info);
	fi_freeinfo(hints);
	if (ret)
		return report("fi_getinfo(client)", ret);
	ret = fi_fabric(r->info->fabric_attr, &r->fabric, NULL);
	if (ret)
		return report("fi_fabric(client)", ret);
	ret = fi_eq_open(r->fabric, &eq_attr, &r->eq, NULL);
	if (ret)
		return report("fi_eq_open(client)", ret);
	ret = fi_domain(r->fabric, r->info, &r->domain, NULL);
	if (ret)
		return report("fi_domain(client)", ret);
	{
		struct fi_cq_attr cq_attr = {
			.size = 1,
			.format = FI_CQ_FORMAT_MSG,
		};
		ret = fi_cq_open(r->domain, &cq_attr, &r->cq, NULL);
	}
	if (ret)
		return report("fi_cq_open(client)", ret);
	ret = fi_endpoint(r->domain, r->info, &r->ep, NULL);
	if (ret)
		return report("fi_endpoint(client)", ret);
	ret = fi_ep_bind(r->ep, &r->cq->fid, FI_SEND | FI_RECV);
	if (ret)
		return report("fi_ep_bind(cq)", ret);
	ret = fi_ep_bind(r->ep, &r->eq->fid, 0);
	if (ret)
		return report("fi_ep_bind(client)", ret);
	ret = fi_enable(r->ep);
	if (ret)
		return report("fi_enable(client)", ret);
	ret = fi_connect(r->ep, r->info->dest_addr, NULL, 0);
	if (ret)
		return report("fi_connect(client)", ret);
	return 0;
}

static int write_byte(int fd, char value)
{
	return write(fd, &value, 1) == 1 ? 0 : -errno;
}

static int read_byte(int fd)
{
	char value;

	return read(fd, &value, 1) == 1 ? 0 : -errno;
}

static int drive_client_progress(struct fid_eq *eq)
{
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t ret;
	int i;

	/* Complete the nonblocking connect and send the CM request. Do not wait
	 * for FI_CONNECTED: the server closes the PEP before accepting it. */
	for (i = 0; i < 20; i++) {
		ret = fi_eq_read(eq, &event, &entry, sizeof(entry), 0);
		if (ret == (ssize_t) sizeof(entry)) {
			if (entry.info)
				fi_freeinfo(entry.info);
			continue;
		}
		if (ret != -FI_EAGAIN)
			return report("fi_eq_read(client)", (int) ret);
		usleep(1000);
	}
	return 0;
}

static int run_client(const char *port, int ready_fd, int connected_fd,
			      int accepted_fd, int cm_ready_fd, int done_fd)
{
	struct test_resources r = {0};
	int ret;

	ret = read_byte(ready_fd);
	if (!ret)
		ret = setup_client(&r, port);
	if (!ret)
		ret = write_byte(connected_fd, 'C');
	if (!ret)
		ret = read_byte(accepted_fd);
	if (!ret)
		ret = drive_client_progress(r.eq);
	if (!ret)
		ret = write_byte(cm_ready_fd, 'M');
	if (!ret)
		ret = read_byte(done_fd);
	close_resources(&r);
	return ret;
}

static int run_server(const char *port, int ready_fd, int connected_fd,
			      int accepted_fd, int cm_ready_fd, int done_fd)
{
	struct test_resources r = {0};
	struct fi_eq_cm_entry entry;
	uint32_t event;
	ssize_t ret;

	ret = setup_server(&r, port);
	if (ret)
		goto out;
	ret = write_byte(ready_fd, 'R');
	if (ret)
		goto out;
	ret = read_byte(connected_fd);
	if (ret)
		goto out;

	/* This poll accepts the TCP socket and creates xnet_conn_handle. */
	ret = fi_eq_read(r.eq, &event, &entry, sizeof(entry), 0);
	if (ret != -FI_EAGAIN) {
		fprintf(stderr, "initial fi_eq_read returned %zd; expected -FI_EAGAIN\n",
			ret);
		if (ret == (ssize_t) sizeof(entry) && entry.info)
			fi_freeinfo(entry.info);
		goto out;
	}
	ret = write_byte(accepted_fd, 'A');
	if (ret)
		goto out;
	ret = read_byte(cm_ready_fd);
	if (ret)
		goto out;

	/* Close the PEP while the unpublished handle still references it. */
	ret = fi_close(&r.pep->fid);
	r.pep = NULL;
	if (ret)
		goto out;

	/* The next progress call must not touch the discarded connection handle. */
	printf("PEP closed; polling EQ\n");
	ret = fi_eq_read(r.eq, &event, &entry, sizeof(entry), 0);
	printf("second fi_eq_read returned %zd\n", ret);
	if (ret == (ssize_t) sizeof(entry) && entry.info)
		fi_freeinfo(entry.info);
	if (ret == -FI_EAGAIN || ret == (ssize_t) sizeof(entry))
		ret = 0;
	(void) write_byte(done_fd, 'D');

out:
	close_resources(&r);
	return ret < 0 ? (int) ret : 0;
}

int main(int argc, char **argv)
{
	const char *port = argc > 1 ? argv[1] : DEFAULT_PORT;
	int ready[2], connected[2], accepted[2], cm_ready[2], done[2];
	pid_t child;
	int status;
	int ret;

	if (pipe(ready) || pipe(connected) || pipe(accepted) ||
	    pipe(cm_ready) || pipe(done)) {
		perror("pipe");
		return EXIT_FAILURE;
	}
	child = fork();
	if (child < 0) {
		perror("fork");
		return EXIT_FAILURE;
	}
	if (child == 0) {
		close(ready[1]);
		close(connected[0]);
		close(accepted[1]);
		close(cm_ready[0]);
		close(done[1]);
		ret = run_client(port, ready[0], connected[1], accepted[0],
				 cm_ready[1], done[0]);
		_exit(ret ? EXIT_FAILURE : EXIT_SUCCESS);
	}
	close(ready[0]);
	close(connected[1]);
	close(accepted[0]);
	close(cm_ready[1]);
	close(done[0]);
	ret = run_server(port, ready[1], connected[0], accepted[1],
			 cm_ready[0], done[1]);
	close(ready[1]);
	close(connected[0]);
	close(accepted[1]);
	close(cm_ready[0]);
	close(done[1]);
	waitpid(child, &status, 0);
	if (ret)
		return EXIT_FAILURE;
	if (!WIFEXITED(status) || WEXITSTATUS(status))
		return EXIT_FAILURE;
	return EXIT_SUCCESS;
}
