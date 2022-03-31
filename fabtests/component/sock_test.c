/*
 * Copyright (c) 2021-2022 Intel Corporation. All rights reserved.
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
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/time.h>
#include <netdb.h>
#include <time.h>
#include <poll.h>
#include <sys/epoll.h>
#include <errno.h>

#include <shared.h>

int *clients;

fd_set select_set;
int max_sock, connections;
struct pollfd *poll_set;
struct epoll_event *events;
int efpd;

static int start_server()
{
	struct addrinfo *ai = NULL;
	int ret;
	int optval = 1;

	ret = getaddrinfo(opts.src_addr, opts.src_port, NULL, &ai);
	if (ret) {
		printf("getaddrinfo failed\n");
		goto out;
	}

	listen_sock = socket(ai->ai_family, SOCK_STREAM, 0);
	if (listen_sock < 0) {
		printf("socket error: %i: %s\n", errno, strerror(errno));
		ret = -1;
		goto out;
	}

	ret = setsockopt(listen_sock, SOL_SOCKET, SO_REUSEADDR, (char *) &optval, 
		sizeof(optval));
	if (ret) {
		printf("error setting socket options\n");
		goto out;
	}

	ret = bind(listen_sock, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		printf("bind failed\n");
		goto out;
	}

	ret = listen(listen_sock, connections);
	if (ret) {
		printf ("listen failed\n");
		goto out;
	}

	freeaddrinfo(ai);
	return ret;

out:
	close(listen_sock);
	freeaddrinfo(ai);
	return ret;
}

static int server_connect()
{
	int new_sock, i;

	for (i = 0; i < connections; i++) {
		new_sock = accept(listen_sock, NULL, NULL);
		if (new_sock < 0) {
			printf("error during server init\n");
			return -1;
		}

		clients[i] = new_sock;
		
	}
	return 0;
}

static int client_connect()
{
	struct addrinfo *res;
	int i, ret, new_sock;

	ret = getaddrinfo(opts.dst_addr, opts.dst_port, NULL, &res);
	if (ret) {
		printf("getaddrinfo failed\n");
		goto out;
	}

	for (i = 0; i < connections; i++) {

		new_sock = socket(res->ai_family, SOCK_STREAM, 0);
		if (new_sock < 0) {
			ret = -1;
			goto out;
		}

		ret = connect(new_sock, res->ai_addr, res->ai_addrlen);
		if (ret)
			goto out;

		clients[i] = new_sock;
	}

out:
	freeaddrinfo(res);
	return ret;
}

static void print_average()
{
	printf("%i: %f\n", connections, (double)(end.tv_nsec + end.tv_sec * 
			1000000000) - (start.tv_nsec + start.tv_sec * 1000000000)
			/ connections);
}

static int init_select() 
{
	FD_ZERO(&select_set);
	FD_SET(listen_sock, &select_set);
	max_sock = listen_sock;

	return 0;
}

static int time_select() 
{
	int i, j;
	int ret = 0;
	struct timeval timeout;
	timeout.tv_sec  = 0;
	timeout.tv_usec = 0;

	ft_start();
	for (i = 0; i< opts.iterations; i++) {
		for (j = 0; j < connections; j++) {
			FD_SET(clients[j], &select_set);

			if (clients[j] > max_sock)
				max_sock = clients[j];
		}
		ret = select(max_sock + 1, &select_set, NULL, NULL, &timeout);
		if (ret)
			return ret;
	}
	ft_stop();
	
	print_average();
	
	return ret;
}

static int init_poll()
{
	int i;
	poll_set = malloc(sizeof(struct pollfd) * connections);
	if (!poll_set)
		return -1;

	for (i = 0; i < connections; i++) {
		poll_set[i].fd = clients[i];
	}

	return 0;
}

static int time_poll()
{
	int i;
	
	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		poll(poll_set, POLLIN, 0);
	}
	ft_stop();

	print_average();

	return 0;
}

static int init_epoll()
{
	int i, ret;

	efpd = epoll_create1(0);
	events = malloc(sizeof(struct epoll_event) * connections);
	if (!events)
		return -1;

	for (i = 0; i < connections; i++) {
		events[i].events = EPOLLIN;
		events[i].data.fd = clients[i];
		ret = epoll_ctl(efpd, EPOLL_CTL_ADD, clients[i], &events[i]);
		if (ret)
			return ret;
	}

	return 0;
}

static int time_epoll()
{
	int i;
	
	ft_start();
	for (i = 0; i < opts.iterations; i++) {
		epoll_wait(EPOLLIN, events, connections, 0);
	}
	ft_stop();

	print_average();

	return 0;
}

static int init_resources(int (*poll_method) (int))
{
	int ret = 0;

	if (poll_method == time_select)
		ret = init_select();
	else if (poll_method == time_poll)
		ret = init_poll();
	else
		ret = init_epoll();

	return ret;
}

static void free_resources()
{
	free(clients);
	free(poll_set);
	free(events);
}

int main(int argc, char **argv)
{
	extern char *optarg;
	int c, ret, i;
	char *mode;
	int (*poll_method) ();

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;

	while ((c = getopt(argc, argv, "n:m:" ADDR_OPTS)) != -1) {
		switch (c) {
		default:
			ft_parse_addr_opts(c, optarg, &opts);
			break;
		case 'n':
			connections = atoi(optarg);
			break;
		case 'm':
			mode = optarg;
			if (strcasecmp(mode, "select") == 0) {
				printf("select mode\n");
				poll_method = time_select;
			} else if (strcasecmp(mode, "poll") == 0) {
				printf("poll mode\n");
				poll_method = time_poll;
			} else if (strcasecmp(mode, "epoll") == 0) {
				printf("epoll mode\n");
				poll_method = time_epoll;
			} else {
				printf("running all modes");
			}
			break;
		}
	}

	if (optind < argc)
		opts.dst_addr = argv[optind];

	clients = calloc(connections, sizeof(*clients));
	if (!clients)
		return -1;

	if (!opts.dst_addr) {

		if (!opts.src_port)
			opts.src_port = default_port;

		ret = start_server();
		if (ret)
			return ret;

		ret = server_connect();
		if (ret)
			return ret;
		if (poll_method) {
			ret = init_resources(poll_method);
			if (ret)
				return ret;

			ret = poll_method();
			if (ret)
				return ret;
		} else {
			ret = init_resources(time_select);
			if (ret)
				return ret;

			ret = time_select();
			if (ret)
				return ret;

			ret = init_resources(time_poll);
			if (ret)
				return ret;

			ret = time_poll();
			if (ret)
				return ret;

			ret = init_resources(time_epoll);
			if (ret)
				return ret;

			ret = time_epoll();
			if (ret)
				return ret;
		}
	} else {
		if (!opts.dst_port)
			opts.dst_port = default_port;

		ret = client_connect();
	}
	
	for (i = 0; i < connections; i++) {
		close(clients[i]);
	}
	close(listen_sock);

	free_resources();

	return ret;
}
