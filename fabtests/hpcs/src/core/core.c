/*
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
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

#include <core/user.h>
#include <pattern/user.h>
#include <test/user.h>

/* size of buffer to store fi_getname() result */
#define NAMELEN 256

/* buffer size for storing hostnames */
#define HOSTLEN 256

static int verbose = 0;

/*
 * Struct which tracks a single domain and the resources created within its
 * scope, including endpoints, completion objects, and data objects.  The
 * lengths of the object arrays depend on the window size or the sharing
 * configuration requested for that kind of object.
 */
struct domain_state {
	struct fid_domain *domain;
	struct fid_ep *endpoint;

	struct fid_av *av;

	/* Will contain only one completion object if
	 * COMPLETION_OBJECT_PER_DOMAIN const was specified. */
	struct fid_cntr *tx_cntr;
	struct fid_cntr *rx_cntr;
	struct fid_cq *tx_cq;
	struct fid_cq *rx_cq;

	/* array indexed by rank */
	fi_addr_t *addresses;
};

struct ofi_state {
	struct fid_fabric *fabric;
	struct domain_state *domain_state;
};

enum callback_order {
	CALLBACK_ORDER_NONE,
	CALLBACK_ORDER_EXPECTED,
	CALLBACK_ORDER_UNEXPECTED
};

struct arguments {
	char prov_name[128];
	size_t window_size;
	size_t buffer_size;
	enum callback_order callback_order;
	size_t iterations;

	struct pattern_api pattern_api;
	struct test_api test_api;

	struct pattern_arguments *pattern_arguments;
	struct test_arguments *test_arguments;
};

#define DEFAULT_WINDOW_SIZE 10
#define DEFAULT_CALLBACK_ORDER CALLBACK_ORDER_NONE

int init_ofi_cntrs(
		const size_t num,
		struct fid_domain *domain,
		struct fid_cntr *cntrs[])
{
	int our_ret = FI_SUCCESS;
	size_t i;
	int ret;

	struct fi_cntr_attr cntr_attr = {0};
	cntr_attr.events = FI_CNTR_EVENTS_COMP;
	cntr_attr.wait_obj = FI_WAIT_UNSPEC;

	for (i = 0; i < num; i += 1) {
		ret = fi_cntr_open(domain, &cntr_attr, cntrs + i, NULL);
		if (ret) {
			our_ret = ret;
			goto err_cntr_open;
		}
	}

err_cntr_open:

	return our_ret;
}

int init_ofi_cqs(
		const size_t num,
		const size_t cq_size,
		struct fid_domain *domain,
		struct fid_cq **cqs)
{
	int our_ret = FI_SUCCESS;
	size_t i;
	int ret;

	struct fi_cq_attr cq_attr = {0};
	cq_attr.size = cq_size;
	cq_attr.format = FI_CQ_FORMAT_TAGGED;
	cq_attr.wait_obj = FI_WAIT_UNSPEC;

	for (i = 0; i < num; i += 1) {
		ret = fi_cq_open(domain, &cq_attr, cqs + i, NULL);
		if (ret) {
			our_ret = ret;
			goto err_cq_open;
		}
	}

err_cq_open:

	return our_ret;
}

int init_ofi_completion(
		const struct arguments *arguments,
		const struct test_config *test_config,
		struct domain_state *domain_state,
		size_t cq_size)
{
	int ret;

	if (test_config->tx_use_cntr) {
		ret = init_ofi_cntrs(
				1,
				domain_state->domain,
				&domain_state->tx_cntr);
		if (ret) {
			hpcs_error("error initializing tx counters, ret=%d\n", ret);
			goto err_init_tx_cntrs;
		}
	}

	if (test_config->rx_use_cntr) {
		ret = init_ofi_cntrs(
				1,
				domain_state->domain,
				&domain_state->rx_cntr);
		if (ret) {
			hpcs_error("error initializing rx counters, ret=%d\n", ret);
			goto err_init_rx_cntrs;
		}
	}

	ret = init_ofi_cqs(
			1,
			cq_size,
			domain_state->domain,
			&domain_state->tx_cq);
	if (ret) {
		hpcs_error("initializing ofi tx cq failed\n");
		goto err_init_tx_cqs;
	}

	ret = init_ofi_cqs(
			1,
			arguments->window_size,
			domain_state->domain,
			&domain_state->rx_cq);
	if (ret) {
		hpcs_error("initializing ofi rx cq failed\n");
		goto err_init_rx_cqs;
	}

err_init_rx_cqs:
err_init_tx_cqs:
err_init_rx_cntrs:
err_init_tx_cntrs:

	return ret;
}


/*
 * Binds a domain's completion objects to an endpoint.  Does not have a
 * corresponding unbind function, since completion objects are implicitly
 * unbound when they are freed.
 */
int bind_ofi_endpoint_completion(struct domain_state *domain_state)
{
	int ret;

	if (domain_state->tx_cq) {
		/* If we have a tx counter, suppress cq completions by default. */
		uint64_t flags =
				domain_state->tx_cntr == NULL
					? FI_TRANSMIT
					: FI_TRANSMIT | FI_SELECTIVE_COMPLETION;

		ret = fi_ep_bind(domain_state->endpoint, &domain_state->tx_cq->fid, flags);
		if (ret) {
			hpcs_error("binding tx cq to ep failed\n");
			goto err;
		}
	}

	if (domain_state->rx_cq) {
		ret = fi_ep_bind(domain_state->endpoint, &domain_state->rx_cq->fid, FI_RECV);
		if (ret) {
			hpcs_error("binding rx cq to ep failed\n");
			goto err;
		}
	}

	if (domain_state->tx_cntr) {
		ret = fi_ep_bind(domain_state->endpoint, &domain_state->tx_cntr->fid, FI_TRANSMIT);
		if (ret) {
			hpcs_error("binding tx counter to ep failed\n");
			goto err;
		}
	}
err:
	return ret;
}


int init_ofi_endpoint(
		struct fi_info *info,
		struct domain_state *domain_state,
		struct fid_ep **endpoint)
{
	int ret;

	ret = fi_endpoint(domain_state->domain, info, endpoint, NULL);
	if (ret) {
		hpcs_error("fi_endpoint failed\n");
		return ret;
	}

	return 0;
}

int init_ofi_domain(
		const struct arguments *arguments,
		const struct test_config *test_config,
		struct fid_fabric *fabric,
		struct fi_info *info,
		struct domain_state *domain_state,
		address_exchange_t address_exchange,
		int num_mpi_ranks,
		int our_mpi_rank)
{
	int ret;
	int i;
	uint8_t *names = NULL;
	uint8_t our_name[NAMELEN];
	void *context = NULL;
	size_t len = NAMELEN;

	size_t cq_size = arguments->window_size * num_mpi_ranks *
			(test_config->tx_context_count + test_config->rx_context_count);

	struct fi_av_attr av_attr = (struct fi_av_attr) {
		.type = FI_AV_MAP,
		.count = num_mpi_ranks,
		.name = NULL
	};

	ret = fi_domain(fabric, info, &domain_state->domain, context);
	if (ret) {
		hpcs_error("fi_domain failed\n");
		goto err;
	}

	ret = init_ofi_endpoint(
		info,
		domain_state,
		&domain_state->endpoint
	);
	if (ret) {
		hpcs_error( "init_ofi_endpoint failed\n");
		goto err;
	}

	ret = init_ofi_completion(arguments, test_config, domain_state, cq_size);
	if (ret) {
		hpcs_error("init_ofi_completion failed\n");
		goto err;
	}

	ret = bind_ofi_endpoint_completion(domain_state);
	if (ret) {
		hpcs_error("bind_ofi_endpoint_completion failed\n");
		goto err;
	}

	ret = fi_av_open(domain_state->domain, &av_attr, &domain_state->av, NULL);
	if (ret) {
		hpcs_error("unable to open address vector\n");
		goto err;
	}

	ret = fi_ep_bind(domain_state->endpoint, &domain_state->av->fid, 0);

	ret = fi_enable(domain_state->endpoint);
	if (ret) {
		hpcs_error("error enabling endpoint\n");
		goto err;
	}

	ret = fi_getname(&domain_state->endpoint->fid, &our_name, &len);

	if (ret) {
		hpcs_error("error determining local endpoint name\n");
		goto err;
	}

	names = malloc(len * num_mpi_ranks);
	if (names == NULL) {
		hpcs_error("error allocating memory for address exchange\n");
		ret = -1;
		goto err;
	}

	ret = address_exchange(&our_name, names, len, num_mpi_ranks);
	if (ret) {
		hpcs_error("error exchanging addresses\n");
		goto err;
	}

	ret = fi_av_insert(domain_state->av,
			   names,
			   num_mpi_ranks,
			   domain_state->addresses,
			   0,
			   NULL);
	if (ret != num_mpi_ranks) {
		hpcs_error("unable to insert all addresses into AV table\n");
		ret = -1;
		goto err;
	}
	else {
		ret = 0;
	}

	if(verbose){
		hpcs_verbose("Rank %d peer addresses: ", our_mpi_rank);
		for (i=0; i<num_mpi_ranks; i++) {
			printf("%d:%lx ", i, (uint64_t)(domain_state->addresses[i]));
		}
		printf("\n");
	}


err:
	free (names);
	return ret;
}

/* Initializes OFI.  Calls fi_allocinfo, fi_getinfo, fi_fabric.  Local resources
 * will be cleaned up on finish / error, returned resources must be cleaned up
 * by calling fini_ofi. */

int init_ofi(
		const struct arguments *arguments,
		const void *test_arguments,
		const struct test_config *test_config,
		struct ofi_state *ofi_state,
		address_exchange_t address_exchange,
		int num_mpi_ranks,
		int our_mpi_rank)
{
	int our_ret = FI_SUCCESS;
	int ret;

	struct fi_info *hints;
	struct fi_info *info;

	hints = fi_allocinfo();
	if (!hints) {
		our_ret = -FI_ENOMEM;
		goto err_allocinfo;
	}

	hints->fabric_attr->prov_name = strdup(arguments->prov_name);
	hints->domain_attr->resource_mgmt = FI_RM_ENABLED;
	hints->domain_attr->mr_mode = FI_MR_RMA_EVENT;
	hints->domain_attr->mr_key_size = 4;
	hints->caps = test_config->minimum_caps;
	hints->mode = FI_CONTEXT;
	hints->ep_attr->type = FI_EP_RDM;
	hints->tx_attr->op_flags = FI_DELIVERY_COMPLETE;

	const char *node = NULL;
	const char *service = "2042";
	const uint64_t flags = 0;

	ret = fi_getinfo(
			fi_version (),
			node,
			service,
			flags,
			hints,
			&info);
	if (ret) {
		our_ret = ret;
		goto err_getinfo;
	}

	void *context = NULL;
	ret = fi_fabric(
			info->fabric_attr,
			&ofi_state->fabric,
			context);
	if (ret) {
		hpcs_error("fi_fabric failed\n");
		our_ret = ret;
		goto err_fabric;
	}

	fi_addr_t* domain_state_addr = ofi_state->domain_state->addresses;

	*ofi_state->domain_state = (struct domain_state) {0};

	ofi_state->domain_state->addresses = domain_state_addr;

	ret = init_ofi_domain(
			arguments,
			test_config,
			ofi_state->fabric,
			info,
			ofi_state->domain_state,
			address_exchange,
			num_mpi_ranks,
			our_mpi_rank);
	if (ret) {
		hpcs_error("init_ofi_domain failed\n");
		our_ret = ret;
		goto err_init_domains;
	}
	else {
		hpcs_verbose("init_ofi_domain complete, Using %s provider\n",
			     info->fabric_attr->name);
		goto err_allocinfo;
	}

err_init_domains:
err_fabric:

	fi_freeinfo(info);
err_getinfo:

	fi_freeinfo(hints);
err_allocinfo:

	return our_ret;
}

int fini_ofi_completion(struct domain_state *domain_state)
{
	int ret;

	if (domain_state->tx_cntr != NULL) {
		ret = fi_close(&domain_state->tx_cntr->fid);
		if (ret) {
			hpcs_error("unable to close tx counter\n");
			return -1;
		}
		domain_state->tx_cntr = NULL;
	}

	if (domain_state->rx_cntr != NULL) {
		ret = fi_close(&domain_state->rx_cntr->fid);
		if (ret) {
			hpcs_error("unable to close rx counter\n");
			return -1;
		}
		domain_state->rx_cntr = NULL;
	}

	if (domain_state->tx_cq != NULL) {
		ret = fi_close(&domain_state->tx_cq->fid);
		if (ret) {
			hpcs_error("unable to close tx cq\n");
			return -1;
		}
		domain_state->tx_cq = NULL;
	}

	if (domain_state->rx_cq != NULL) {
		ret = fi_close(&domain_state->rx_cq->fid);
		if (ret) {
			hpcs_error("unable to close rx cq\n");
			return -1;
		}
		domain_state->rx_cq = NULL;
	}

	return ret;
}

int fini_ofi_domain(struct domain_state *domain_state)
{
	int ret = 0;

	ret = fi_close(&domain_state->endpoint->fid);
	if (ret) {
		hpcs_error("unable to close endpoint\n");
		return ret;
	}
	domain_state->endpoint = NULL;

	ret = fi_close(&domain_state->av->fid);
	if (ret) {
		hpcs_error("unable to close address vector\n");
		return ret;
	}

	ret = fini_ofi_completion(domain_state);
	if (ret)
		return ret;

	ret = fi_close(&domain_state->domain->fid);
	if (ret) {
		hpcs_error("unable to close domain\n");
		return ret;
	}
	domain_state->domain = NULL;

	return 0;
}

/*
 * Finalizes OFI.  Safely cleans up resources created by init_ofi.  Takes a
 * "best effort" approach - goes as far as it can and returns the first error
 * encountered.
 */
int fini_ofi(struct ofi_state *ofi_state)
{
	int ret;

	ret = fini_ofi_domain(ofi_state->domain_state);
	if (ret)
		return ret;

	ofi_state->domain_state = NULL;

	ret = fi_close(&ofi_state->fabric->fid);
	if (ret) {
		hpcs_error("unable to close fabric\n");
		return ret;
	}

	return 0;
}

/*
 * HPCS uses groups of arguments separated by a "-" or "--" (which are treated
 * as special by getopt, and cause it to stop parsing arguments).
 *
 * This function updates argc and argv to start at the next group of args,
 * with argv[0] pointing to the separator.
 */
void next_args(int *argc, char * const**argv)
{
	int i;

	/* 0th element is either binary name or previous separator. */
	for (i=1; i<*argc; i++) {
		if (strcmp((*argv)[i], "-") == 0 || strcmp((*argv)[i], "--") == 0)
			break;
	}

	if (i < *argc) {
		*argc = *argc - i;
		*argv = &(*argv)[i];
	} else {
		*argc = 0;
		*argv = NULL;
	}

	/* Current option index global variable must be reset. */
	optind = 1;
}

static int parse_arguments(int argc, char * const* argv,
		struct arguments **arguments)
{
	int longopt_idx, op, onesided = 0, ret = 0;
	struct option longopt[] = {
		{"prov", required_argument, 0, 'p'},
		{"window", required_argument, 0, 'w'},
		{"order", required_argument, 0, 'o'},
		{"pattern", required_argument, 0, 'a'},
		{"iterations", required_argument, 0, 'n'},
		{"verbose", no_argument, &verbose, 1},
		{"help", no_argument, 0, 'h'},
		{0}
	};

	struct arguments *args = calloc(sizeof (struct arguments), 1);
	int have_pattern = 0;

	*args = (struct arguments) {
		.prov_name = "",
		.window_size = DEFAULT_WINDOW_SIZE,
		.callback_order = DEFAULT_CALLBACK_ORDER,
		.pattern_api = {0},
		.iterations = 1
	};

	if (args == NULL)
		return -ENOMEM;

	if(strstr(argv[0], "onesided") != NULL)
		onesided = 1;

	while ((op = getopt_long(argc, argv, "vp:w:o:a:n:h", longopt, &longopt_idx)) != -1) {
		switch (op) {
		case 0:
			if (longopt[longopt_idx].flag != 0)
				printf("verbose mode enabled\n");
			break;
		case 'p':
			if (sscanf(optarg, "%127s", args->prov_name) != 1)
				return -EINVAL;
			break;
		case 'w':
			if (sscanf(optarg, "%zu", &args->window_size) != 1)
				return -EINVAL;
			break;
		case 'o':
			if (strcmp(optarg, "none") == 0)
				args->callback_order = CALLBACK_ORDER_NONE;
			else if (strcmp(optarg, "expected") == 0)
				args->callback_order = CALLBACK_ORDER_EXPECTED;
			else if (strcmp(optarg, "unexpected") == 0)
			{
				if(onesided) {
					hpcs_error("Unexpected is not supported"
						   " in onesided test\n");
					return -EINVAL;
				}


				args->callback_order = CALLBACK_ORDER_UNEXPECTED;
			}
			else
				return -EINVAL;
			break;
		case 'a':
			if ((!strcmp(optarg, "alltoall")) || (!strcmp(optarg, "a2a")))
				args->pattern_api = a2a_pattern_api ();
			else if (!strcmp(optarg, "self"))
				args->pattern_api = self_pattern_api ();
			else if (!strcmp(optarg, "alltoone"))
				args->pattern_api = alltoone_pattern_api ();
			else
				return -EINVAL;
			have_pattern = 1;
			break;
		case 'n':
			if (sscanf(optarg, "%zu", &args->iterations) != 1)
				return -EINVAL;
			break;
		case 'v':
			verbose = 1;
			break;
		case 'h':
		default:
			hpcs_error("usage: %s [core-args] -- [pattern args] -- [test-args]\n", argv[0]);
			hpcs_error("[core-args] := "
				   "\t[-p | --prov=<provider name>]\n"
				   "\t[-w | --window=<size>]\n"
				   "\t[-o | --order=<expected|unexpected|none>]\n"
				   "\t[-n | --iterations=<n>]\n"
				   "\t -a | --pattern=<self|alltoall|alltoone>\n"
				   "\t[-h | --help]\n");
			return -1;
		}
	}

	if (!have_pattern) {
		hpcs_error("you must specify a pattern\n");
		return -EINVAL;
	}

	args->test_api = test_api();

	next_args(&argc, &argv);

	ret = args->pattern_api.parse_arguments(argc, argv,
			&args->pattern_arguments);

	if (ret) {
		hpcs_error("failed to parse pattern arguments\n");
		return ret;
	}

	next_args(&argc, &argv);

	ret = args->test_api.parse_arguments(argc, argv,
			&args->test_arguments,
                	&args->buffer_size);

	*arguments = args;

	if (ret) {
		hpcs_error("failed to parse test arguments\n");
		return ret;
	}

	return 0;
}

static void free_arguments (struct arguments *arguments)
{
	struct pattern_api _pattern_api = arguments->pattern_api;

	struct test_api _test_api = test_api ();

	_pattern_api.free_arguments (arguments->pattern_arguments);
	_test_api.free_arguments (arguments->test_arguments);

	free (arguments);
}

#define DATA_BUF(base, counter) ((base) + ((counter % window) * size))
#define CONTEXT(base, counter) (&base[counter % window])

static int core_inner (
		struct domain_state *domain_state,
		struct arguments *arguments,
		struct pattern_api *pattern,
		struct test_api *test,
		struct test_config *test_config,
		size_t rank,
		size_t ranks,
		address_exchange_t address_exchange,
		barrier_t barrier)
{
	size_t i, j;
	int ret;
	size_t size = arguments->buffer_size;
	size_t window = arguments->window_size;
	size_t iterations = arguments->iterations;

	struct op_context tx_context [window];
	struct op_context rx_context [window];
	uint8_t *tx_buf = calloc(window, size);
	uint8_t *rx_buf = calloc(window, size);
	uint64_t *keys = calloc(ranks, sizeof(uint64_t));

	struct test_arguments *test_args = arguments->test_arguments;
	struct pattern_arguments *pattern_args = arguments->pattern_arguments;

	int tx_window = window, rx_window = window;
	enum callback_order order = arguments->callback_order;

	struct fid_mr *rx_mr = NULL;
	int do_tx_reg = test_config->tx_use_cntr;

	uint64_t recvs_posted = 0, sends_posted = 0;
	uint64_t recvs_done = 0, sends_done = 0;

	memset((char*)&tx_context[0], 0, sizeof(tx_context[0])*window);
	memset((char*)&rx_context[0], 0, sizeof(rx_context[0])*window);

	for (i = 0; i < window; i++) {
		tx_context[i].ctxinfo =
				calloc(test_config->tx_context_count*sizeof(struct context_info), 1);
		rx_context[i].ctxinfo =
				calloc(test_config->rx_context_count*sizeof(struct context_info), 1);

		if (tx_context[i].ctxinfo == NULL || rx_context[i].ctxinfo == NULL)
			return -FI_ENOMEM;

		/* Populate backlinks. */
		for (j = 0; j < test_config->tx_context_count; j++) {
			tx_context[i].ctxinfo[j].op_context = &tx_context[i];
		}

		for (j = 0; j < test_config->rx_context_count; j++) {
			rx_context[i].ctxinfo[j].op_context = &rx_context[i];
		}

	}

	if (tx_buf == NULL || rx_buf == NULL || keys == NULL) {
		hpcs_error("unable to allocate memory\n");
		ret = -ENOMEM;
		goto err_mem;
	}

	hpcs_verbose("Beginning test: buffer_size=%ld window=%ld iterations=%ld %s%s%s\n",
			size, window, iterations,
			order == CALLBACK_ORDER_UNEXPECTED ? "unexpected" : "",
			order == CALLBACK_ORDER_EXPECTED ? "expected" : "",
			order == CALLBACK_ORDER_NONE ? "undefined order" : "");

	/*
	 * One-sided tests create a single memory region, and then share
	 * that key with peers (who may each write to some offset).
	 */
	if (test->rx_create_mr != NULL && test_config->rx_use_mr) {
		uint64_t my_key;

		if (window < ranks) {
			hpcs_error("for one-sided communication, window must be >= number of ranks\n");
			return (-EINVAL);
		}

		/* Key can be any arbitrary number. */
		rx_mr = test->rx_create_mr(test_args, domain_state->domain, 42+rank,
				rx_buf, window*size, test_config->mr_rx_flags);
		if (rx_mr == NULL) {
			hpcs_error("failed to create target memory region\n");
			return -1;
		}

		my_key = fi_mr_key(rx_mr);
		address_exchange(&my_key, keys, sizeof(uint64_t), ranks);

		if (verbose) {
			hpcs_verbose("mr key exchange complete: rank %ld my_key %ld len %ld keys: ",
					rank, my_key, window*size);
			for (i=0; i<ranks; i++) {
				printf("%ld ", keys[i]);
			}
			printf("\n");
		}

		if (test_config->rx_use_cntr) {
			if (domain_state->rx_cntr) {
				ret = fi_mr_bind(rx_mr, &domain_state->rx_cntr->fid, test_config->mr_rx_flags);
				if (ret) {
					hpcs_error("fi_mr_bind (rx_cntr) failed: %d\n", ret);

					/*
					 * Binding an MR with FI_REMOTE_READ isn't defined by the OFI spec,
	 				 * so we don't consider this a failure.
 					 */
					if (test_config->mr_rx_flags & FI_REMOTE_READ) {
						hpcs_error("FI_REMOTE_READ memory region bind flag unsupported by this provider, skipping test.\n");
						return 0;
					}

					return -1;
				}
			} else {
				hpcs_error("no rx counter to bind mr to\n");
				return -EINVAL;
			}
		}

		ret = fi_mr_enable(rx_mr);
		if (ret)
			hpcs_error("fi_mr_enable failed: %d\n", ret);

		barrier();
	}


	for (i = 0; i < iterations; i++) {
		int cur_sender = PATTERN_NO_CURRENT;
		int cur_receiver = PATTERN_NO_CURRENT;
		int prev;
		int completions_done = 0, rx_done = 0, tx_done = 0;
		struct op_context* op_context;

		uint64_t recvs_done_prev = recvs_done;
		uint64_t sends_done_prev = sends_done;

		while (!completions_done || !rx_done || !tx_done) {
			/* post receives */
			while (!rx_done) {
				if (order == CALLBACK_ORDER_UNEXPECTED && !tx_done)
					break;

				prev = cur_sender;
				if (pattern->next_sender(pattern_args, rank, ranks, &cur_sender) != 0) {
					rx_done = 1;
					if (order == CALLBACK_ORDER_EXPECTED)
						barrier();
					break;
				}

				/*
				 * Doing window check after calling next_sender allows us to
				 * mark receives as done if our window is zero but there are
				 * no more senders.
				 */
				if (rx_window == 0) {
					cur_sender = prev;
					break;
				}

				op_context = CONTEXT(rx_context, recvs_posted);
				if (op_context->state != DONE) {
					cur_sender = prev;
					break;
				}

				test->rx_init_buffer(test_args, DATA_BUF(rx_buf, recvs_posted));


				/* fprintf(stdout, "Posting rx:  rank %d, sender %d\n", rank, cur_sender); */

				op_context->buf = DATA_BUF(rx_buf, recvs_posted);

				ret = test->rx_transfer(test_args, cur_sender,
						1, domain_state->addresses[cur_sender],
						domain_state->endpoint,
						op_context,
						op_context->buf,
						NULL, NULL, NULL);


				if (ret == -FI_EAGAIN) {
					cur_sender = prev;
					break;
				}

				hpcs_verbose("rx_transfer initiated: ctx %p "
					     "from rank %ld\n",
					     op_context, cur_sender);

				if (ret) {
					hpcs_error("test receive failed, ret=%d\n", ret);
					return ret;
				}

				op_context->state = PENDING;
				op_context->core_context = recvs_posted;

				recvs_posted++;
				rx_window--;
			};

			/* post send(s) */
			while (!tx_done) {
				if (order == CALLBACK_ORDER_EXPECTED && !rx_done)
                                        break;

				prev = cur_receiver;
				if (pattern->next_receiver(pattern_args, rank, ranks, &cur_receiver) != 0) {
					if (order == CALLBACK_ORDER_UNEXPECTED)
						barrier();
					tx_done = 1;
					break;
				}

				if (tx_window == 0) {
					cur_receiver = prev;
					break;
				}

				op_context = CONTEXT(tx_context, sends_posted);
				if (op_context->state != DONE) {
					cur_receiver = prev;
					break;
				}

				test->tx_init_buffer(test_args, DATA_BUF(tx_buf, sends_posted));

				struct fid_mr* mr = NULL;
				void *mr_desc = NULL;
				if (do_tx_reg) {
					mr = test->tx_create_mr(test_args,
							domain_state->domain,
							0, /* key */
							DATA_BUF(tx_buf, sends_posted),
							size,
							FI_SEND);
					if (mr == NULL) {
						ret = -1;
						goto err_mem;
					}

					if (test_config->tx_use_cntr) {
						if (domain_state->tx_cntr) {
							ret = fi_mr_bind(mr, &domain_state->tx_cntr->fid, FI_SEND);
							if (ret) {
								hpcs_error("fi_mr_bind (tx_cntr) failed: ret %d\n", ret);
								return -1;
							}
						} else {
							hpcs_error("no counter to bind tx memory region to\n");
							return -EINVAL;
						}
					}

					mr_desc = fi_mr_desc(mr);
				}

				/* fprintf(stdout, "Posting tx:  rank %d, receiver %d\n", rank, cur_receiver);*/

				op_context->buf = DATA_BUF(tx_buf, sends_posted);
				op_context->tx_mr = mr;

				ret = test->tx_transfer(test_args, rank,
						1, domain_state->addresses[cur_receiver],
						domain_state->endpoint,
						op_context,
						op_context->buf,
						mr_desc, keys[cur_receiver], rank, NULL, NULL);

				if (ret == -FI_EAGAIN) {
					cur_receiver = prev;
					break;
				}

				hpcs_verbose("tx_transfer initiated from rank %ld "
					     "to rank %d: ctx %p key %ld ret %d\n",
					     rank, cur_receiver, op_context,
					     keys[cur_receiver], ret);

				if (ret) {
					hpcs_error("tx_transfer failed, ret=%d\n", ret);
					return ret;
				}

				op_context->state = PENDING;
				op_context->core_context = sends_posted;

				sends_posted++;
				tx_window--;
			};

			/* poll completions */
			if (test_config->rx_use_cq) {
				while ((ret = test->rx_cq_completion(test_args,
						&op_context,
						domain_state->rx_cq)) != -FI_EAGAIN) {
					if (ret) {
						hpcs_error("cq_completion (rx) failed, ret=%d\n", ret);
						return -1;
					}

					if (test->rx_datacheck(test_args, op_context->buf, 0)) {
						hpcs_error("rx data check error at iteration %ld\n", i);
						return -1;
					}

					op_context->state = DONE;
					recvs_done++;
					rx_window++;

					hpcs_verbose("ctx %p receive %ld complete\n",
						     op_context, op_context->core_context);
				}
			}

			if (test_config->tx_use_cq) {
				while ((ret = test->tx_cq_completion(test_args,
						&op_context,
						domain_state->tx_cq)) != -FI_EAGAIN) {
					if (ret) {
						hpcs_error("cq_completion (tx) failed, ret=%d\n", ret);
						return -1;
					}
					hpcs_verbose("Received tx completion for ctx %lx\n",
						     op_context);

					if (test->tx_datacheck(test_args, op_context->buf)) {
						hpcs_error("tx data check error at iteration %ld\n", i);
						return -1;
					}

					if (do_tx_reg && test->tx_destroy_mr != NULL) {
						ret = test->tx_destroy_mr(test_args, op_context->tx_mr);
						if (ret) {
							hpcs_error("unable to release tx memory region\n");
							return -1;
						}
					}

					op_context->state = DONE;
					op_context->test_state = 0;
					sends_done++;
					tx_window++;

					hpcs_verbose("ctx %p send %ld complete\n",
						     op_context, op_context->core_context);
				}
			}

			/*
			 * Counters are generally used for RMA/atomics and completion is handled
			 * as all-or-nothing rather than individual completions.
			 */
			if (rx_done && tx_done) {
				if (test_config->tx_use_cntr && sends_done < sends_posted) {
					ret = test->tx_cntr_completion(test_args, sends_posted, domain_state->tx_cntr);
					if (ret) {
						hpcs_error("cntr_completion (tx) failed, ret=%d\n", ret);
						return -1;
					}

					for (j = sends_done_prev; j < sends_posted; j++) {
						op_context = CONTEXT(tx_context, j);

						if (do_tx_reg && test->tx_destroy_mr != NULL) {
							ret = test->tx_destroy_mr(test_args, op_context->tx_mr);
							if (ret) {
								hpcs_error("unable to release tx memory region\n");
								return -1;
							}
						}
						sends_done++;
						op_context->state = DONE;
						op_context->test_state = 0;
						tx_window++;
					}

					if (sends_done != sends_posted) {
						hpcs_error("tx accounting internal error\n");
						return -EFAULT;
					}

					hpcs_verbose("TX counter completion done\n");
				}

				if (test_config->rx_use_cntr && recvs_done < recvs_posted) {
					ret = test->rx_cntr_completion(test_args, recvs_posted, domain_state->rx_cntr);
					if (ret) {
						hpcs_error("cntr_completion (rx) failed, ret=%d\n", ret);
						return -1;
					}

					for (j = recvs_done_prev; j < recvs_posted; j++) {
						op_context = CONTEXT(rx_context, j);
						recvs_done++;
						op_context->state = DONE;
						op_context->test_state = 0;
						rx_window++;
					}

					/* 
					 * note: counter tests use rx_buf directly,
					 * rather than DATA_BUF(rx_buf, j)
					 */

					if (test->rx_datacheck(test_args, rx_buf, recvs_posted - recvs_done_prev)) {
						hpcs_error("rx data check error at iteration %ld\n", i);
						return -1;
					}

					test->rx_init_buffer(test_args, rx_buf);

					if (recvs_done != recvs_posted) {
						hpcs_error("rx accounting internal error\n");
						return -EFAULT;
					}

					hpcs_verbose("rx counter completion done\n");
				}
			}

			if (recvs_posted == recvs_done && sends_posted == sends_done) {
				completions_done = 1;
			} else {
				hpcs_verbose("recvs_posted=%ld, recvs_done=%ld, sends_posted=%ld, sends_done=%ld\n",
					      recvs_posted, recvs_done, sends_posted, sends_done);
				usleep(50000);
			}
		}
	}

	/*
	 * OFI docs are unclear about proper order of closing memory region
	 * and counter that are bound to each other.
	 */
	if (rx_mr != NULL && test->rx_destroy_mr != NULL) {
		ret = test->rx_destroy_mr(test_args, rx_mr);
		if (ret) {
			hpcs_error("unable to release rx memory region\n");
			return -1;
		}
	}

	/* Make sure all our peers are done before exiting. */
	barrier();
	ret = 0;

err_mem:
	if (tx_buf)
		free(tx_buf);

	if (rx_buf)
		free(rx_buf);

	if (keys)
		free(keys);

	return ret;
}

int core (
		const int argc,
		char * const *argv,
		const int num_mpi_ranks,
		const int our_mpi_rank,
		address_exchange_t address_exchange,
		barrier_t barrier)
{
	int ret;

	struct arguments *arguments = NULL;
	struct test_api test = test_api ();
	struct test_config test_config = {0};
	struct ofi_state ofi_state = {0};
	struct domain_state domain_state = {0};
	struct pattern_api pattern = {0};

	fi_addr_t addresses[num_mpi_ranks];

	/*
	struct fid_cntr *tx_cntrs [num_tx_cntr (arguments, test_config)];
	struct fid_cntr *rx_cntrs [num_rx_cntr (arguments, test_config)];
	*/

	ret = parse_arguments(argc, argv, &arguments);

	if (ret < 0)
		return -EINVAL;

	pattern = arguments->pattern_api;

	test_config = test.config(arguments->test_arguments);

	hpcs_verbose("Initializing ofi resources\n");

	domain_state.addresses = &addresses[0];

	ofi_state.domain_state = &domain_state;

	ret = init_ofi(
			arguments,
			arguments->test_arguments,
			&test_config,
			&ofi_state,
			address_exchange,
			num_mpi_ranks,
			our_mpi_rank);
	if (ret) {
		hpcs_error("Init_ofi failed, ret=%d\n", ret);
		return -1;
	} else {
		hpcs_verbose("OFI resource initialization successful\n");
	}

	ret = core_inner(&domain_state, arguments, &pattern, &test, &test_config,
			our_mpi_rank, num_mpi_ranks, address_exchange, barrier);

	if (ret) {
		hpcs_error("Test failed, ret=%d\n", ret);
		return -1;
	}

	ret = fini_ofi(&ofi_state);
	if (ret) {
		hpcs_error("Resource cleanup failed, ret=%d\n", ret);
		return -1;
	}

	free_arguments(arguments);

	return 0;
}



void hpcs_error(const char* format, ...)
{
	char hostname[HOSTLEN];

        gethostname(hostname, HOSTLEN);
	hostname[HOSTLEN-1]='\0';

	va_list args;
	fprintf(stderr, "%s: ", hostname);
        va_start(args, format);
        vfprintf(stderr, format, args);
        va_end (args);

}

void hpcs_verbose(const char* format, ...)
{
	char hostname[HOSTLEN];

	if(!verbose)
		return;

        gethostname(hostname, HOSTLEN);
	hostname[HOSTLEN-1]='\0';

	va_list args;
	fprintf(stdout, "%s: ", hostname);
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end (args);

}
