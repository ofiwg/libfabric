/*
 * Copyright (c) 2021-2022 Intel Corporation.  All rights reserved.
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

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netdb.h>
#include <arpa/inet.h>
#include "util.h"

int client_sockets[MAX_CLIENTS];
int num_client_connections = 0;

double when(void)
{
	struct timeval tv;
	static struct timeval tv0;
	static int first = 1;
	int err;

	err = gettimeofday(&tv, NULL);
	if (err) {
		perror("gettimeofday");
		return 0;
	}

	if (first) {
		tv0 = tv;
		first = 0;
	}
	return (double)(tv.tv_sec - tv0.tv_sec) * 1.0e6 +
	       (double)(tv.tv_usec - tv0.tv_usec);
}

int connect_tcp(char *host, int port)
{
	struct sockaddr_in sin;
	struct hostent *addr;
	int sockfd, newsockfd;
	socklen_t clen;

	memset(&sin, 0, sizeof(sin));

	if ((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
		perror("socket");
		exit(-1);
	}

	if (host) {
		if (atoi(host) > 0) {
			sin.sin_family = AF_INET;
			sin.sin_addr.s_addr = inet_addr(host);
		} else {
			if ((addr = gethostbyname(host)) == NULL){
				printf("invalid hostname '%s'\n", host);
				exit(-1);
			}
			sin.sin_family = addr->h_addrtype;
			memcpy(&sin.sin_addr.s_addr, addr->h_addr, addr->h_length);
		}
		sin.sin_port = htons(port);
		if(connect(sockfd, (struct sockaddr *) &sin, sizeof(sin)) < 0){
			perror("connect");
			exit(-1);
		}
		return sockfd;
	} else {
		int one = 1;
		if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(int))) {
			perror("setsockopt");
			exit(-1);
		}
		memset(&sin, 0, sizeof(sin));
		sin.sin_family = AF_INET;
		sin.sin_addr.s_addr = htonl(INADDR_ANY);
		sin.sin_port = htons(port);
		if (bind(sockfd, (struct sockaddr *) &sin, sizeof(sin)) < 0){
			perror("bind");
			exit(-1);
		}

		listen(sockfd, 5);
		clen = sizeof(sin);
		newsockfd = accept(sockfd, (struct sockaddr *) &sin, &clen);
		if(newsockfd < 0) {
			perror("accept");
			exit(-1);
		}

		close(sockfd);
		return newsockfd;
	}
}

void sync_tcp(int sockfd)
{
	int dummy1, dummy2;

	(void)! write(sockfd, &dummy1, sizeof dummy1);
	(void)! read(sockfd, &dummy2, sizeof dummy2);
}

int send_to_socket(int sockfd, size_t size, void *data)
{
	if (write(sockfd, data, size) != size) {
		fprintf(stderr, "Failed to send data to socket\n");
		return -1;
	}
	return 0;
}

int recv_from_socket(int sockfd, size_t size, void *data)
{
	if (recv(sockfd, data, size, MSG_WAITALL) != size) {
		fprintf(stderr, "Failed to receive data from socket\n");
		return -1;
	}
	return 0;
}

int exchange_info(int sockfd, size_t size, void *me, void *peer)
{
	if (write(sockfd, me, size) != size) {
		fprintf(stderr, "Failed to send local info\n");
		return -1;
	}

	if (recv(sockfd, peer, size, MSG_WAITALL) != size) {
		fprintf(stderr, "Failed to read peer info\n");
		return -1;
	}

	return 0;
}


void sync_barrier(int max_ranks, int my_rank, int my_fd, bool client,
		  bool verbose)
{
	int k;
	if (!client) {
		for (k = 0; k < max_ranks; k++) {
			if (k == my_rank)
				continue;

			if (verbose)
				printf("Server: Waiting for client %d "
				       "completion signal...\n", k + 1);
			sync_tcp(client_sockets[k]);
		}
	} else {
		sync_tcp(my_fd);
	}
}

int start_client(char *server_name, unsigned int port, bool verbose,
		 int *sockfd)
{
	*sockfd = connect_tcp(server_name, port);
	if (*sockfd < 0)
		return -1;

	if (verbose)
		printf("Client: Connected to server %s:%d\n",
			server_name, port);
	return 0;
}

int start_server(int max_ranks, unsigned int port, bool verbose, int *sockfd)
{
	int i;
	int reuse = 1;
	struct sockaddr_in server_addr, client_addr;
	socklen_t client_len = sizeof(client_addr);

	if ((client_sockets[0] = socket(AF_INET, SOCK_STREAM, 0)) < 0)
		return -1;

	if (setsockopt(client_sockets[0], SOL_SOCKET, SO_REUSEADDR, &reuse,
	    sizeof(int)))
		return -1;

	memset(&server_addr, 0, sizeof(server_addr));
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	server_addr.sin_port = htons(port);

	if (bind(client_sockets[0], (struct sockaddr*) &server_addr,
	    sizeof(server_addr)) < 0)
		return -1;

	if (listen(client_sockets[0], max_ranks) < 0)
		return -1;

	if (verbose)
		printf("Server: Waiting for %d clients to connect on "
			"port %d...\n", max_ranks - 1, port);

	for (i = 1; i < max_ranks; i++) {
		if (verbose)
			printf("Server: Accepting client %d...\n",
				i + 1);
		client_sockets[i] = accept(client_sockets[0],
					   (struct sockaddr*) &client_addr,
					   &client_len);
		if (client_sockets[i] < 0)
			return -1;

		num_client_connections++;
		if (verbose)
			printf("Server: Client %d connected from %s\n",
			       i, inet_ntoa(client_addr.sin_addr));
	}

	if (verbose)
		printf("Server: All %d clients connected\n",
			num_client_connections);

	*sockfd = client_sockets[0];
	return 0;
}

void cleanup_sockets(void)
{
	int i;
	for (i = 0; i < MAX_CLIENTS; i++) {
		if (client_sockets[i] > 0)
			close(client_sockets[i]);
	}
}
