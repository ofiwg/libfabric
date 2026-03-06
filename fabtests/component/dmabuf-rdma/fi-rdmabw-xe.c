/*
 * Copyright (c) Intel Corporation.  All rights reserved.
 *
 * This software is available to you under the BSD license below:
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

/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * \
 *			FI_MULTINODE_RDMABW				       *
 * This test is designed to run with one server and multiple clients. Each     *
 * peer will initialize their -d <type> buffers, HOST proxy buffers, and sync  *
 * buffer, and register them with the specified NIC(s). The server will then   *
 * open a listening socket and wait for -r ranks clients to connect. Once a    *
 * client connects, it will recieve its assigned rank from the server,         *
 * initialize its business card: the address and rkey for each of its buffers  *
 * on each NIC, and send the business card to the server. The server will      *
 * collect the business cards from every client, and then send the complete    *
 * list of business cards to every client. Once all clients have the complete  *
 * peer list, they will insert the address/rkey for each buffer on each NIC    *
 * for every peer into their AV. At this point, all peers can directly         *
 * communicate with each other without further involvement from the server.    *
 *									       *
 * After the initial setup, the test will perform -n iterations of -t <type>   *
 * transfers according to -A <algorithm> using messages of size -S <size> up   *
 * to -M <max_size>. If -P is set, then for DEVICE to DEVICE transfers, the    *
 * sender will first copy the data to a host proxy buffer, before transmitting *
 * the data to the receiver's data buffer.                                     *
 *									       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 			ALL TO ALL ALGORITHM 				       *
 * Using the setup detailed above, for each iteration, every peer will attempt *
 * to send a message to every other peer simultaneously. For the purpose of    *
 * this test All-to-All is per rank, not per nic. For SEND tests, each peer    *
 * will post a receive buffer for each expected incoming message before        *
 * posting the sends. After posting TX DEPTH messages, (WRITE, READ, SEND),    *
 * each peer will wait for all posted message completions before moving to the *
 * next iteration. For SEND tests, each peer will also wait for completion of  *
 * the posted receives. When using multiple NICs and/or multiple GPUs, the     *
 * test will distribute the transmissions across the NICs in a round-robin     *
 * selection manner.							       *
 *									       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * 			POINT TO POINT ALGORITHM 			       *
 * Using the setup detailed above, for each iteration, The server will post    *
 * RX_DEPTH receives for the client in a round-robin manner across the client's*
 * NICs before processing completions. The client will then transmit data to   *
 * the server TX_DEPTH times before processing completions. Once all iterations*
 * are complete, both sides will perform data validation (if requested) and the*
 * client will report bandwidth results.                                       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * All-to-All Example 0: 3 peers, 0:2:2, 1:2:2, 2:2:2 (peer:num_nics:num_gpus).*
 * Peer 0 (Server)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [BUF 2]				       *
 * NIC MLX5_1: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * Peer 1 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [BUF 2]				       *
 * NIC MLX5_1: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * Peer 2 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [BUF 2]				       *
 * NIC MLX5_1: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * All to All								       *
 * Iteration 1 (peer, gpu buf)						       *
 * (0, 0) -> (1, 0), (2, 0)						       *
 * (1, 1) -> (0, 1), (2, 1)						       *
 * (2, 2) -> (0, 2), (1, 2)						       *
 * Iteration 2 (peer, gpu buf)						       *
 * (0, 3) -> (1, 3), (2, 3)						       *
 * (1, 4) -> (0, 4), (2, 4)						       *
 * (2, 5) -> (0, 5), (1, 5)						       *
 * Iteration 3->n repeat iteration 1 & 2				       *
 * 									       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * All-to-All Example 1: 3 peers, 0:2:2, 1:2:1, 2:2:1 (peer:num_nics:num_gpus).*
 * Peer 0 (Server)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [BUF 2]				       *
 * NIC MLX5_1: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * Peer 1 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [Buf 2]				       *
 * NIC MLX5_1: GPU0: [BUF 3] [BUF 4] [BUF 5]				       *
 * Peer 2 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [Buf 2]				       *
 * NIC MLX5_1: GPU0: [BUF 3] [BUF 4] [BUF 5]				       *
 * 									       *
 * All to All								       *
 * Iteration 1 (peer, gpu buf)						       *
 * (0, 0) -> (1, 0), (2, 0)						       *
 * (1, 1) -> (0, 1), (2, 1)						       *
 * (2, 2) -> (0, 2), (1, 2)						       *
 * Iteration 2 (peer, gpu buf)						       *
 * (0, 3) -> (1, 3), (2, 3)						       *
 * (1, 4) -> (0, 4), (2, 4)						       *
 * (2, 5) -> (0, 5), (1, 5)						       *
 * Iteration 3->n repeat iteration 1 & 2				       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * All-to-All Example 2: 3 peers, 0:1:2, 1:1:1, 2:2:1 (peer:num_nics:num_gpus).*
 * Peer 0 (Server)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [BUF 2]				       *
 * NIC MLX5_0: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * Peer 1 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [Buf 2]				       *
 * Peer 2 (Client)							       *
 * NIC MLX5_0: GPU0: [BUF 0] [BUF 1] [Buf 2]				       *
 * NIC MLX5_1: GPU1: [BUF 3] [BUF 4] [BUF 5]				       *
 * 									       *
 * All to All								       *
 * Iteration 1 (peer, gpu buf)						       *
 * (0, 0) -> (1, 0), (2, 0)						       *
 * (1, 1) -> (0, 1), (2, 1)						       *
 * (2, 2) -> (0, 2), (1, 2)						       *
 * Iteration 2 (peer, gpu buf)						       *
 * (0, 3) -> (1, 3), (2, 3)						       *
 * (1, 1) -> (0, 1), (2, 1)						       *
 * (2, 5) -> (0, 5), (1, 5)						       *
 * Iteration 3->n repeat iteration 1 & 2				       *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <assert.h>
#include <getopt.h>
#include <arpa/inet.h>
#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_errno.h>
#include <level_zero/ze_api.h>
#include "shared.h"
#include "util.h"
#include "xe.h"
#include "ofi_ctx_pool.h"

static void set_default_options(void)
{
	options.max_ranks = 2;
}

static void print_opts()
{
	size_t len = 16;
	char *str = malloc(len);

	printf("\nOPTIONS:\n");
	if (options.client) {
		printf("\tMode:\t\t\tclient\n");
		printf("\tServer name:\t\t%s\n",
			   options.server_name ? options.server_name : "none");
	} else {
		printf("\tMode:\t\t\tserver\n");
	}
	printf("\tRank:\t\t\t%d\n", options.rank);
	printf("\tPort:\t\t\t%d\n", options.port);
	printf("\t%s", options.bidirectional ? "Bidirectional" :
	       "Unidirectional");
	printf("\tAlgorithm:\t\t%s\n",
		   options.algo == ALL_TO_ALL ? "all_to_all" :
		   options.algo == POINT_TO_POINT ? "point_to_point" :
		   "unknown");
	printf("\tProxy block size:\t%d\n", options.proxy_block);
	printf("\tDomain:GPU mappings:\t%s\n", options.mapping_str);
	printf("\tNumber of mappings:\t%d\n", options.num_mappings);
	printf("\tEndpoint type:\t\t%s\n",
		   options.ep_type == FI_EP_RDM ? "RDM" : "MSG");
	printf("\tMax message size:\t%zd\n", options.max_size);
	buf_location_str(options.loc1, str, len);
	printf("\tBuffer location 1:\t%s\n", str);
	str = memset(str, 0, len);
	buf_location_str(options.loc2, str, len);
	printf("\tBuffer location 2:\t%s\n", str);
	buf_location_str(options.buf_location, str, len);
	printf("\tBuffer location:\t%s\n", str);
	printf("\tIterations:\t\t%d\n", options.iters);
	printf("\tProvider name:\t\t%s\n",
		   options.prov_name ? options.prov_name : "default");
	printf("\tUse proxy:\t\t%s\n", options.use_proxy ? "yes" : "no");
	printf("\tUse raw key:\t\t%s\n", options.use_raw_key ? "yes" : "no");
	printf("\tMax ranks to connect: \t%d\n", options.max_ranks);
	printf("\tMessage size:\t\t%zd\n", options.msg_size);
	printf("\tTest type:\t\t%s\n",
		   options.test_type == READ ? "READ" :
		   options.test_type == WRITE ? "WRITE" : "SEND");
	printf("\tVerbose output:\t\t%s\n", options.verbose ? "yes" : "no");
	printf("\tVerify data:\t\t%s\n", options.verify ? "yes" : "no");
	printf("\n");
	free(str);
}

static void usage(char *prog)
{
	printf("\nUsage:\n");
	printf("server:  %s [options]\n", prog);
	printf("clients: %s [options] <server_name>\n", prog);
	printf("OPTIONS:\n");
	printf("\t-A <algorith>  Set the algorithm to use, can be 'all_to_all',"
				   "'point_to_point','ALL' default: "
				   "point_to_point\n");
	printf("\t-B <block_size>  Set the block size for proxying, default: "
				   "maximum message size\n");
	printf("\t-b		   Use bidirectional data movement "
				   "(for point-to-point algorithm only), "
				   "default: off\n");
	printf("\t-d <domain_name>:<gpu_dev>     Use the NIC:GPU device "
				   "specified as comma separated list of "
				   "<domain_name:<dev>[.<subdev>], default: "
				   "automatic:0.\n");
	printf("\t-e <ep_type>     Set the endpoint type, can be 'rdm' or "
				   "'msg', default: rdm\n");
	printf("\t-M <size>        Set the maximum message size to test, can "
				   "use suffix K/M/G, default: 4194304 (4M)\n");
	printf("\t-m <location>    Where to allocate the buffer, can be "
				   "'malloc', 'host', 'device' or 'shared', "
				   "default: malloc\n");
	printf("\t-n <iters>       Set the number of iterations for each "
				   "message size, default: 1000\n");
	printf("\t-p <prov_name>   Use the OFI provider named as <prov_name>, "
				   "default: the first one\n");
	printf("\t-P               Proxy device buffer through host buffer "
				   "(for write and send only), default: off\n");
	printf("\t-R		   Reverse the direction of data movement "
				   "(server initiates RDMA ops)\n");
	printf("\t-r <ranks>       Total number of ranks to connect before "
				   "starting (includes self). Max %d\n",
				   MAX_CLIENTS);
	printf("\t-S <size>        Set the message size to test (0: all, -1: "
				   "none), can use suffix K/M/G, default: 0\n");
	printf("\t-t <test_type>   Type of test to perform, can be 'read', "
				   "'write', or 'send', default: read\n");
	printf("\t-V               Enable verbose output\n");
	printf("\t-v               Verify the data\n");
	printf("\t-h | -?          Print this message\n");
	printf("\n");
}

static enum algorithm str_to_algo(char *str)
{
	char *remove = "_- ";
	remove_characters(str, remove);

	if (strcasecmp(str, "alltoall") == 0 ||
	    strcasecmp(str, "a2a") == 0)
		return ALL_TO_ALL;
	else if (strcasecmp(str, "pointtopoint") == 0 ||
		 strcasecmp(str, "p2p") == 0)
		return POINT_TO_POINT;
	else if (strcasecmp(str, "all") == 0)
		return MAX_ALGORITHM;

	fprintf(stderr, "Unknown algorithm %s\n", str);
	usage("fi-multinode-rdmabw");
	exit(-1);
	return MAX_ALGORITHM;
}

static void parse_mapping(char *mapping_str)
{
	char *token;
	char *domain;
	char *gpu_str;
	char *subdev;
	char *saveptr1;
	char *saveptr2;
	int nic, dev_num, subdev_num;

	options.num_mappings = nic = 0;
	token = strtok_r(mapping_str, ",", &saveptr1);
	while (token) {
		domain = strtok_r(token, ":", &saveptr2);
		gpu_str = strtok_r(NULL, ",", &saveptr2);
		dev_num = 0;
		subdev_num = -1;
		dev_num = atoi(gpu_str);
		subdev = strchr(gpu_str, '.');
		if (subdev)
			subdev_num = atoi(subdev + 1);

		if (!domain || !gpu_str || nic >= MAX_NICS ||
		    dev_num >= MAX_GPUS) {
			fprintf(stderr, "Invalid NIC:GPU mapping %d:%d<.%d>\n",
				nic, dev_num, subdev_num);
			exit(-1);
		}

		nics[nic].mapping.domain_name = strdup(domain);
		nics[nic].mapping.gpu.dev_num = dev_num;
		nics[nic].mapping.gpu.subdev_num = subdev_num;
		nic++;
		options.num_mappings++;
		token = strtok_r(NULL, ",", &saveptr1);
	}
}

static int check_opts(void)
{
	if (options.msg_size > 0 && options.msg_size > options.max_size) {
		options.max_size = options.msg_size;
		options.proxy_block = options.msg_size;
		fprintf(stderr,
			"Max_size smaller than message size, adjusted to %zd\n",
			options.max_size);
	}

	if (options.algo == POINT_TO_POINT) {
		if (options.max_ranks > 2) {
			fprintf(stderr, "Point to point algorithm only supports"
				" 2 ranks\n");
			usage("fi-multinode-rdmabw");
			return -1;
		}
		if (options.bidirectional) {
			options.algo = ALL_TO_ALL;
			/*
			 * We can use the all-to-all algorithm to perform
			 * bidirectional because we would be doing "all-to-all"
			 * between 2 ranks, which is effectively the same as
			 * bidirectional point-to-point.
			*/
			if (options.verbose) {
				printf("Using All-to-All algorithm instead to "
				       "perform bidirectional "
				       "point-to-point\n");
			}
		}
		if (!options.client && !options.bidirectional &&
		    options.test_type == SEND) {
			options.test_type = RECV;
			if (options.verbose) {
				printf("Server will be reciever for "
				       "unidirectional point-to-point test\n");
			}
		}
	} else {
		if (options.bidirectional) {
			fprintf(stderr, "Bidirectional option is only valid for"
				" point-to-point algorithm\n");
			usage("fi-multinode-rdmabw");
			return -1;
		}
	}

	return 0;
}

static void parse_opts(int argc, char **argv)
{
	int op;
	char *s;

	while ((op = getopt(argc, argv, OPTS)) != -1) {
		switch (op) {
			case 'A':
				options.algo = str_to_algo(optarg);
				break;
			case 'B':
				options.proxy_block = parse_size(optarg);
				if (options.proxy_block < MIN_PROXY_BLOCK) {
					fprintf(stderr,
						"Block size too small, "
						"adjusted to %d\n",
						MIN_PROXY_BLOCK);
					options.proxy_block = MIN_PROXY_BLOCK;
				}
				break;
			case 'b':
				options.bidirectional = true;
				break;
			case 'd':
				options.mapping_str = strdup(optarg);
				parse_mapping(strdup(optarg));
				break;
			case 'e':
				if (strcasecmp(optarg, "rdm") == 0) {
					options.ep_type = FI_EP_RDM;
				} else if (strcasecmp(optarg, "msg") == 0) {
					options.ep_type = FI_EP_MSG;
				} else {
					fprintf(stderr,
						"Unknown endpoint type %s\n",
						optarg);
					usage(argv[0]);
					exit(-1);
				}
				break;
			case 'M':
				options.max_size = parse_size(optarg);
				options.proxy_block = options.max_size;
				break;
			case 'm':
				parse_buf_location(optarg, &options.loc1,
						   &options.loc2, MALLOC);
				break;
			case 'n':
				options.iters = atoi(optarg);
				break;
			case 'p':
				options.prov_name = strdup(optarg);
				break;
			case 'P':
				options.use_proxy = true;
				break;
			case 'r':
				options.max_ranks = atoi(optarg);
				if (options.max_ranks < 2) {
					fprintf(stderr,
						"Ranks must be at least 2, "
						"adjusted to 2\n");
					options.max_ranks = 2;
				}
				if (options.max_ranks > MAX_CLIENTS) {
					fprintf(stderr,
						"Ranks exceed max supported "
						"%d\n", MAX_CLIENTS);
					usage(argv[0]);
					exit(-1);
				}
				break;
			case 'S':
				options.msg_size = parse_size(optarg);
				break;
			case 't':
				if (strcasecmp(optarg, "read") == 0) {
					options.test_type = READ;
				} else if (strcasecmp(optarg, "write") == 0) {
					options.test_type = WRITE;
				} else if (strcasecmp(optarg, "send") == 0) {
					options.test_type = SEND;
				} else {
					fprintf(stderr,
						"Unknown test type %s\n",
						optarg);
					usage(argv[0]);
					exit(-1);
				}
				break;
			case 'V':
				options.verbose = true;
				break;
			case 'v':
				options.verify = 1;
				break;
			case 'h':
			case '?':
			default:
				usage(argv[0]);
				exit(-1);
				break;
		}
	}

	if (argc > optind) {
		options.client = true;
		options.server_name = strdup(argv[optind]);
	}

	peers = calloc(options.max_ranks, sizeof(struct business_card));

	EXIT_ON_ERROR(check_opts());

	/*
	 * If started by a job launcher, perform pair-wise test.
	 */
	s = getenv("PMI_RANK");
	if (s) {
		options.rank = atoi(s);
		options.client = options.rank > 0;
		options.port += options.rank >> 1;
		if (!options.client && options.server_name) {
			free(options.server_name);
			options.server_name = NULL;
		}
	}
}

int main(int argc, char *argv[])
{
	int i;

	set_default_options();
	parse_opts(argc, argv);

	if (options.verbose)
		print_opts();

	options.buf_location = options.client ? options.loc2 : options.loc1;

	if (options.client)
		EXIT_ON_ERROR(start_client(options.server_name, options.port,
					   options.verbose, &options.sockfd));
	else
		EXIT_ON_ERROR(start_server(options.max_ranks, options.port,
					   options.verbose, &options.sockfd));

	for (i = 0; i < options.num_mappings; i++) {
		if (options.verbose)
			printf("Init NIC %s with GPU %d<.%d>.\n",
			       nics[i].mapping.domain_name,
			       nics[i].mapping.gpu.dev_num,
			       nics[i].mapping.gpu.subdev_num);

		if(xe_init(&nics[i].mapping.gpu))
			goto err_out;

		if (options.verbose)
			show_xe_resources(&nics[i].mapping.gpu);
	}


	CHECK_ERROR(init_ofi());

	if (options.verbose)
		print_nic_info();

	sync_barrier(options.max_ranks, options.rank, options.sockfd,
		     options.client, options.verbose);

	if (run_test())
		goto err_out;

	sync_barrier(options.max_ranks, options.rank, options.sockfd,
		     options.client, options.verbose);
err_out:
	if (!options.client)
		cleanup_sockets();

	finalize_ofi();
	free_buf();
	for (i = 0; i < options.num_mappings; i++)
		xe_cleanup_gpu(&nics[i].mapping.gpu);

	xe_cleanup();

	close(options.sockfd);
	if (peers)
		free(peers);
        return 0;
}
