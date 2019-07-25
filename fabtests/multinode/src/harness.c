/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
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
#include <errno.h>
#include <shared.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include <core.h>
struct pm_job_info pm_job;

static inline ssize_t socket_send(int sock, void *buf, size_t len, int flags)
{
	ssize_t ret;
	size_t m = 0;
	uint8_t *ptr = (uint8_t *) buf;

	do {
		ret = send(sock, (void *) &ptr[m], len-m, flags);
		if (ret < 0)
			return ret;

		m += ret;
	} while (m != len);

	return len;
}

static inline int socket_recv(int sock, void *buf, size_t len, int flags)
{
	ssize_t ret;
	size_t m = 0;
	uint8_t *ptr = (uint8_t *) buf;

	do {
		ret = recv(sock, (void *) &ptr[m], len-m, flags);
		if (ret <= 0)
			return -1;

		m += ret;
	} while (m < len);

	return len;
}

<<<<<<< HEAD
int pm_allgather(void *my_item, void *items, int item_size)
=======
static int pm_allgather(void *my_item, void *items, int item_size)
<<<<<<< HEAD
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test

=======
>>>>>>> Add common hpcs testing patterns
{
	int i, ret;
	uint8_t *offset;

	/* client */
	if (!pm_job.clients) {
		ret = socket_send(pm_job.sock, my_item, item_size, 0);
		if (ret < 0)
			return errno == EPIPE ? -FI_ENOTCONN : -errno;

		ret = socket_recv(pm_job.sock, items,
<<<<<<< HEAD
				  pm_job.num_ranks*item_size, 0);
=======
				  pm_job.ranks*item_size, 0);
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		if (ret <= 0)
			return (ret)? -errno : -FI_ENOTCONN;

		return 0;
	}

	/* server */
	memcpy(items, my_item, item_size);

<<<<<<< HEAD
	for (i = 0; i < pm_job.num_ranks-1; i++) {
=======
	for (i = 0; i < pm_job.ranks-1; i++) {
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		offset = (uint8_t *)items + item_size * (i+1);

		ret = socket_recv(pm_job.clients[i], (void *)offset,
				  item_size, 0);
		if (ret <= 0)
			return ret;
	}

<<<<<<< HEAD
	for (i = 0; i < pm_job.num_ranks-1; i++) {
		ret = socket_send(pm_job.clients[i], items,
				  pm_job.num_ranks*item_size, 0);
=======
	for (i = 0; i < pm_job.ranks-1; i++) {
		ret = socket_send(pm_job.clients[i], items,
				  pm_job.ranks*item_size, 0);
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		if (ret < 0)
			return ret;
	}
	return 0;
}

<<<<<<< HEAD
<<<<<<< HEAD
void pm_barrier()
{
	char ch;
	char chs[pm_job.num_ranks];

	pm_allgather(&ch, chs, 1);
}

static int server_connect()
{
	int new_sock;
	int ret, i;

	ret = listen(pm_job.sock, pm_job.num_ranks);
	if (ret)
		return ret;

	pm_job.clients = calloc(pm_job.num_ranks, sizeof(int));
	if (!pm_job.clients)
		return -FI_ENOMEM;

	for (i = 0; i < pm_job.num_ranks-1; i++){
		new_sock = accept(pm_job.sock, NULL, NULL);
=======
=======
static int pm_init_ranks()
{
	int ret;
	int send_rank, i;

	if(pm_job.clients){
		for(i = 0; i < pm_job.ranks-1; i++){
			send_rank = i + 1;
			ret = socket_send(pm_job.clients[i], &send_rank, sizeof(int), 0);
			if(ret < 0){
				return ret;
			}
		}
	}
	else{
		int recv; 
		ret = socket_recv(pm_job.sock, &(recv), sizeof(int), 0);

		pm_job.rank = recv;
		if(ret < 0){
			return ret;
		}

	}

	return ret;
}

>>>>>>> Add common hpcs testing patterns
static void pm_barrier()
{
	char ch;
	char chs[pm_job.ranks];

	pm_job.allgather(&ch, chs, 1);
}

static int server_init()
{
	int new_sock;
	int ret, i = 0;

	ret = listen(pm_job.sock, pm_job.ranks);
	if (ret)
		return ret;

	pm_job.clients = calloc(pm_job.ranks, sizeof(int));
	if (!pm_job.clients)
		return -FI_ENOMEM;

	while (i < pm_job.ranks-1 &&
	       (new_sock = accept(pm_job.sock, NULL, NULL))) {
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
		if (new_sock < 0) {
			FT_ERR("error during server init\n");
			goto err;
		}
		pm_job.clients[i] = new_sock;
<<<<<<< HEAD
<<<<<<< HEAD
		FT_DEBUG("connection established\n");
	}
	close(pm_job.sock);
	return 0;
err:
	while (i--) {
		close(pm_job.clients[i]);
	}
=======
=======

>>>>>>> Add common hpcs testing patterns
		i++;
		FT_DEBUG("connection established\n");
	}

	close(pm_job.sock);
	return 0;
err:
>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
	free(pm_job.clients);
	return new_sock;
}

<<<<<<< HEAD
=======
static inline int client_init()
{
	return connect(pm_job.sock, pm_job.oob_server_addr,
			sizeof(*pm_job.oob_server_addr));
}

>>>>>>> fabtests/multinode: Initial version of multinode sendrecv test
static int pm_conn_setup()
{
	int sock,  ret;
	int optval = 1;

	sock = socket(AF_INET, SOCK_STREAM, 0);
	if (sock < 0)
		return -1;

	pm_job.sock = sock;

	ret = setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, (char *) &optval,
			 sizeof(optval));
	if (ret) {
		FT_ERR("error setting socket options\n");
		return ret;
	}

	ret = bind(sock, (struct sockaddr *)&pm_job.oob_server_addr,
		   sizeof(pm_job.oob_server_addr));
	if (ret == 0) {
		ret = server_connect();
	} else {
		opts.dst_addr = opts.src_addr;
		opts.dst_port = opts.src_port;
		opts.src_addr = NULL;
		opts.src_port = 0;
		ret = connect(pm_job.sock, (struct sockaddr *)&pm_job.oob_server_addr,
			      sizeof(pm_job.oob_server_addr));
	}
	if (ret) {
		FT_ERR("OOB conn failed - %s\n", strerror(errno));
		return ret;
	}

	return 0;
}

static void pm_finalize()
{
	int i;

	if (!pm_job.clients) {
		close(pm_job.sock);
		return;
	}

	for (i = 0; i < pm_job.num_ranks-1; i++) {
		close(pm_job.clients[i]);
	}
	free(pm_job.clients);
}

int main(int argc, char **argv)
{
	struct sockaddr_in *sock_addr = (struct sockaddr_in *)&pm_job.oob_server_addr;
	extern char *optarg;
	int c, ret;

	opts = INIT_OPTS;
	opts.options |= (FT_OPT_SIZE | FT_OPT_ALLOC_MULT_MR);

	pm_job.clients = NULL;
	sock_addr->sin_port = 8228;

	while ((c = getopt(argc, argv, "n:h" ADDR_OPTS INFO_OPTS)) != -1) {
		switch (c) {
		default:
			ft_parse_addr_opts(c, optarg, &opts);
			ft_parseinfo(c, optarg, hints, &opts);
			break;
		case '?':
		case 'n':
			pm_job.num_ranks = atoi(optarg);
			break;
		case 'h':
			ft_usage(argv[0], "A simple multinode test");
			return EXIT_FAILURE;
		}
	}

	sock_addr->sin_family = AF_INET;
	if (inet_pton(AF_INET, opts.src_addr,
		      (void *) &sock_addr->sin_addr) != 1)
		return -1;

	ret = pm_conn_setup();
	if (ret
		goto err1;

	FT_DEBUG("OOB job setup done\n");

	ret = pm_init_ranks();
	if(ret < 0){
		return ret;
	}

	ret = multinode_run_tests(argc, argv);
	if (ret) {
		FT_ERR( "Tests failed\n");
		goto err2;
	}
	FT_DEBUG("Tests Passed\n");
err2:
	pm_finalize();
err1:
	return ret;
}
