/*
 * Copyright (c) 2017-2019 Intel Corporation. All rights reserved.
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
 * THE SOFTWARE IS PROVIDED "AS IS", WITHWARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. const NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER const AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS const THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <limits.h>
#include <stdarg.h>

#include <rdma/fi_errno.h>
#include <rdma/fi_domain.h>
#include <rdma/fabric.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_trigger.h>
#include <rdma/fi_collective.h>

#include <core.h>
#include <coll_test.h>
#include <shared.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <assert.h>

struct fid_av_set *av_set;

int no_setup()
{
	return FI_SUCCESS;
}

int no_run()
{
	return FI_SUCCESS;
}

void no_teardown()
{
}

int coll_setup()
{
	int err;
	struct fi_av_set_attr av_set_attr;

	av_set_attr.count = pm_job.num_ranks;
	av_set_attr.start_addr = 0;
	av_set_attr.end_addr = pm_job.num_ranks-1;
	av_set_attr.stride = 1;

	err = fi_av_set(av, &av_set_attr, &av_set, NULL);
	if (err) {
		FT_DEBUG("av_set creation failed ret = %d\n", err);
	}

	return err;
}

void coll_teardown()
{
	free(av_set);
}

int join_test_run()
{
	int ret;
	uint32_t event;
	struct fi_cq_err_entry comp = {0};
	fi_addr_t world_addr;
	struct fid_mc *coll_mc;

	ret = fi_av_set_addr(av_set, &world_addr);
	if (ret) {
		FT_DEBUG("failed to get collective addr = %d\n", ret);
		return ret;
	}

	ret = fi_join_collective(ep, world_addr, av_set, 0, &coll_mc, NULL);
	if (ret) {
		FT_DEBUG("collective join failed ret = %d\n", ret);
		return ret;
	}

	while (1) {
		ret = fi_eq_read(eq, &event, NULL, 0, 0);
		if (ret >= 0) {
			FT_DEBUG("found eq entry ret %d\n", event);
			if (event == FI_JOIN_COMPLETE) {
				return FI_SUCCESS;
			}
		} else if(ret != -EAGAIN) {
			return ret;
		}

		ret = fi_cq_read(rxcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}

		ret = fi_cq_read(txcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}
	}

	fi_close(&coll_mc->fid);
}

int barrier_test_run()
{
	int ret;
	uint32_t event;
	struct fi_cq_err_entry comp = {0};
	uint64_t done_flag;
	fi_addr_t world_addr;
	fi_addr_t barrier_addr;
	struct fid_mc *coll_mc;

	ret = fi_av_set_addr(av_set, &world_addr);
	if (ret) {
		FT_DEBUG("failed to get collective addr = %d\n", ret);
		return ret;
	}

	ret = fi_join_collective(ep, world_addr, av_set, 0, &coll_mc, NULL);
	if (ret) {
		FT_DEBUG("collective join failed ret = %d\n", ret);
		return ret;
	}

	while (1) {
		ret = fi_eq_read(eq, &event, NULL, 0, 0);
		if (ret >= 0) {
			FT_DEBUG("found eq entry ret %d\n", event);
			if (event == FI_JOIN_COMPLETE) {
				barrier_addr = fi_mc_addr(coll_mc);
				ret = fi_barrier(ep, barrier_addr, &done_flag);
				if (ret) {
					FT_DEBUG("collective barrier failed ret = %d\n", ret);
					return ret;
				}
			}
		} else if(ret != -EAGAIN) {
			return ret;
		}

		ret = fi_cq_read(rxcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}

		if(comp.op_context && comp.op_context == &done_flag) {
			return FI_SUCCESS;
		}

		ret = fi_cq_read(txcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}

		if(comp.op_context && comp.op_context == &done_flag) {
			return FI_SUCCESS;
		}
	}

	fi_close(&coll_mc->fid);
}

int sum_all_reduce_test_run()
{
	int ret;
	uint32_t event;
	struct fi_cq_err_entry comp = {0};
	uint64_t done_flag;
	fi_addr_t world_addr;
	fi_addr_t allreduce_addr;
	struct fid_mc *coll_mc;
	uint64_t result = 0;
	uint64_t expect_result = 0;
	uint64_t data = pm_job.my_rank;
	size_t count = 1;
	uint64_t i;

	for(i = 0; i < pm_job.num_ranks; i++) {
		expect_result += i;
	}

	ret = fi_av_set_addr(av_set, &world_addr);
	if (ret) {
		FT_DEBUG("failed to get collective addr = %d\n", ret);
		return ret;
	}

	ret = fi_join_collective(ep, world_addr, av_set, 0, &coll_mc, NULL);
	if (ret) {
		FT_DEBUG("collective join failed ret = %d\n", ret);
		return ret;
	}

	while (1) {
		ret = fi_eq_read(eq, &event, NULL, 0, 0);
		if (ret >= 0) {
			FT_DEBUG("found eq entry ret %d\n", event);
			if (event == FI_JOIN_COMPLETE) {
				allreduce_addr = fi_mc_addr(coll_mc);
				ret = fi_allreduce(ep, &data, count, NULL, &result, NULL,
						   allreduce_addr, FI_UINT64, FI_SUM, 0,
						   &done_flag);
				if (ret) {
					FT_DEBUG("collective allreduce failed ret = %d\n", ret);
					return ret;
				}
			}
		} else if(ret != -EAGAIN) {
			return ret;
		}

		ret = fi_cq_read(rxcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}

		if(comp.op_context && comp.op_context == &done_flag) {
			if(result == expect_result)
				return FI_SUCCESS;
			FT_DEBUG("allreduce failed; expect: %ld, actual: %ld\n", expect_result, result);

			return FI_ENOEQ;
		}

		ret = fi_cq_read(txcq, &comp, 1);
		if(ret < 0 && ret != -EAGAIN) {
			return ret;
		}

		if(comp.op_context && comp.op_context == &done_flag) {
			if(result == expect_result)
				return FI_SUCCESS;
			FT_DEBUG("allreduce failed; expect: %ld, actual: %ld\n", expect_result, result);

			return FI_ENOEQ;
		}
	}

	fi_close(&coll_mc->fid);
}

struct coll_test tests[] = {
	{
		.name = "join_test",
		.setup = coll_setup,
		.run = join_test_run,
		.teardown = coll_teardown
	},
	{
		.name = "barrier_test",
		.setup = coll_setup,
		.run = barrier_test_run,
		.teardown = coll_teardown
	},
	{
		.name = "sum_all_reduce_test",
		.setup = coll_setup,
		.run = sum_all_reduce_test_run,
		.teardown = coll_teardown
	},
};

const int NUM_TESTS = ARRAY_SIZE(tests);

static inline
int setup_hints()
{
	hints->ep_attr->type			= FI_EP_RDM;
	hints->caps				= FI_MSG | FI_COLLECTIVE;
	hints->mode				= FI_CONTEXT;
	hints->domain_attr->control_progress	= FI_PROGRESS_MANUAL;
	hints->domain_attr->data_progress	= FI_PROGRESS_MANUAL;
	hints->fabric_attr->prov_name		= strdup("tcp");
	return FI_SUCCESS;
}

static int multinode_setup_fabric(int argc, char **argv)
{
	char my_name[FT_MAX_CTRL_MSG];
	size_t len;
	int ret;

	setup_hints();

	ret = ft_getinfo(hints, &fi);
	if (ret)
		return ret;

	ret = ft_open_fabric_res();
	if (ret)
		return ret;

	opts.av_size = pm_job.num_ranks;

	av_attr.type = FI_AV_TABLE;
	ret = ft_alloc_active_res(fi);
	if (ret)
		return ret;

	ret = ft_enable_ep(ep, eq, av, txcq, rxcq, txcntr, rxcntr);
	if (ret)
		return ret;

	len = FT_MAX_CTRL_MSG;
	ret = fi_getname(&ep->fid, (void *) my_name, &len);
	if (ret) {
		FT_PRINTERR("error determining local endpoint name\n", ret);
		goto err;
	}

	pm_job.name_len = len;
	pm_job.names = malloc(len * pm_job.num_ranks);
	if (!pm_job.names) {
		FT_ERR("error allocating memory for address exchange\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = pm_allgather(my_name, pm_job.names, pm_job.name_len);
	if (ret) {
		FT_PRINTERR("error exchanging addresses\n", ret);
		goto err;
	}

	pm_job.fi_addrs = calloc(pm_job.num_ranks, sizeof(*pm_job.fi_addrs));
	if (!pm_job.fi_addrs) {
		FT_ERR("error allocating memory for av fi addrs\n");
		ret = -FI_ENOMEM;
		goto err;
	}

	ret = fi_av_insert(av, pm_job.names, pm_job.num_ranks,
			   pm_job.fi_addrs, 0, NULL);
	if (ret != pm_job.num_ranks) {
		FT_ERR("unable to insert all addresses into AV table\n");
		ret = -1;
		goto err;
	}
	return 0;
err:
	ft_free_res();
	return ft_exit_code(ret);
}

static void pm_job_free_res()
{

	free(pm_job.names);

	free(pm_job.fi_addrs);
}

int multinode_run_tests(int argc, char **argv)
{
	int ret = FI_SUCCESS;
	int i;

	ret = multinode_setup_fabric(argc, argv);
	if (ret)
		return ret;

	for (i = 0; i < NUM_TESTS && !ret; i++) {
		FT_DEBUG("Running Test: %s \n", tests[i].name);

		ret = tests[i].setup();
		FT_DEBUG("Setup Complete...\n");
		if (ret)
			goto out;

		ret = tests[i].run();
		tests[i].teardown();
		FT_DEBUG("Run Complete...\n");
		if (ret)
			goto out;


		pm_barrier();
		FT_DEBUG("Test Complete: %s \n", tests[i].name);
	}

out:
	if (ret)
		printf("failed\n");
	else
		printf("passed\n");

	pm_job_free_res();
	ft_free_res();
	return ft_exit_code(ret);
}
