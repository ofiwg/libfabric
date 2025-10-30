#include <arpa/inet.h>
#include <getopt.h>
#include <netdb.h>
#include <netinet/in.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include "hmem.h"
#include "shared.h"
#include <pthread.h>

#define MAX_WORKERS	64
#define MAX_PEERS	MAX_WORKERS
#define MAX_EP_ADDR_LEN 256
#define MAX_MESSAGES	1000
#define NOTIFY_PORT	8000

// Message types
#define MSG_TYPE_EP_UP	     1
#define MSG_TYPE_EP_DOWN     2
#define MSG_TYPE_EP_UPDATE   3
#define MSG_TYPE_EP_TEARDOWN 4

#define TEST_CQDATA 0xAAAA // Hardcoded CQ data for fi_writedata

// Configuration structure
struct test_opts {
	int num_sender_workers;
	int num_receiver_workers;
	int msgs_per_endpoint;
	/* Number of times for sender to recycle endpoints */
	int sender_ep_recycling;
	/* Number of times for receiver to recycle endpoints */
	int receiver_ep_recycling;
	bool shared_av;
	bool shared_cq;
	enum { OP_MSG_UNTAGGED = 0, OP_MSG_TAGGED, OP_RMA_WRITEDATA } op_type;
	bool verbose;
	char *sender_addr;
	time_t random_seed;
};

// Global variables
static struct test_opts topts = {
	.num_sender_workers = 1,
	.num_receiver_workers = 1,
	.msgs_per_endpoint = 1000,
	.sender_ep_recycling = 1, // Default to 1 recycling for sender
	.receiver_ep_recycling = 1, // Default to 1 recycling for receiver
	.shared_av = false, // Default: 1 AV per EP
	.shared_cq = false, // Default: 1 CQ per EP
	.op_type = OP_MSG_UNTAGGED,
	.verbose = true,
	.sender_addr = NULL,
};

enum {
	OPT_SENDER_WORKERS = 256,
	OPT_RECEIVER_WORKERS,
	OPT_MSGS_PER_EP,
	OPT_SENDER_EP_CYCLES,
	OPT_RECEIVER_EP_CYCLES,
	OPT_SHARED_AV,
	OPT_SHARED_CQ,
	OPT_OP_TYPE,
	OPT_SENDER_ADDR,
	OPT_RANDOM_SEED,
};

static struct option test_long_opts[] = {
	{"sender-workers", required_argument, NULL, OPT_SENDER_WORKERS},
	{"receiver-workers", required_argument, NULL, OPT_RECEIVER_WORKERS},
	{"msgs-per-ep", required_argument, NULL, OPT_MSGS_PER_EP},
	{"sender-ep-cycles", required_argument, NULL, OPT_SENDER_EP_CYCLES},
	{"receiver-ep-cycles", required_argument, NULL, OPT_RECEIVER_EP_CYCLES},
	{"shared-av", no_argument, NULL, OPT_SHARED_AV},
	{"shared-cq", no_argument, NULL, OPT_SHARED_CQ},
	{"op-type", required_argument, NULL, OPT_OP_TYPE},
	{"sender-addr", required_argument, NULL, OPT_SENDER_ADDR},
	{"random-seed", required_argument, NULL, OPT_RANDOM_SEED},
	{0, 0, 0, 0}};

// Endpoint status
enum ep_status { EP_INIT, EP_READY, EP_SENDING, EP_RECEIVING, EP_TEARDOWN };

// RMA information
struct rma_info {
	uint64_t remote_addr;
	uint64_t rkey;
	size_t length;
};

// Endpoint metadata
struct ep_info {
	uint32_t worker_id;
	char ep_addr[MAX_EP_ADDR_LEN];
	size_t addr_len;
	struct rma_info rma;
};

// Message structure for endpoint updates
struct ep_message {
	int msg_type;
	struct ep_info info;
};

// Worker status tracking
struct worker_status {
	pthread_mutex_t mutex;
	bool active;
	enum ep_status ep_status;
	uint64_t ep_generation;
	bool poll_cq;
};

// Common context structure
struct common_context {
	struct fid_ep *ep;
	struct fid_cq *cq;
	struct fid_av *av;
};

// Sender context
struct sender_context {
	struct common_context common;
	int worker_id;
	void *tx_buf;
	struct fid_mr *mr;
	struct fi_context2 *tx_ctx;
	int *peer_ids; // Store the indices of the remote workers it will talk to
	fi_addr_t *peer_addrs; // Store the fi_addr_t of the remote workers it will talk to
	char **peer_ep_addrs; // Store the latest ep raw address for ep recycling
	struct rma_info *peer_rma_info; // Array of RMA info for each peer
	int num_peers;
	int *control_socks; // Array of control sockets for multiple receivers
	struct worker_status status;
	pthread_t notification_thread;
	uint64_t total_sent;
};

// Receiver context
struct receiver_context {
	struct common_context common;
	int worker_id;
	void *rx_buf;
	struct fid_mr *mr;
	struct fi_context2 *rx_ctx;
	struct worker_status status;
	uint64_t total_received;
	int *control_socks; // Array of control sockets for multiple senders
	int num_senders; // Number of connected senders
};

struct fid_cq *shared_txcq, *shared_rxcq;
struct fid_av *shared_txav, *shared_rxav;
static pthread_mutex_t shared_cq_lock = PTHREAD_MUTEX_INITIALIZER;

// Setup endpoint for a worker
static int setup_endpoint(struct common_context *ctx)
{
	int ret;

	ret = fi_endpoint(domain, fi, &ctx->ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	if (!topts.shared_cq) {
		cq_attr.format = FI_CQ_FORMAT_CONTEXT;
		cq_attr.size = topts.msgs_per_endpoint;

		ret = fi_cq_open(domain, &cq_attr, &ctx->cq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			goto cleanup_ep;
		}
	}

	if (!topts.shared_av) {
		int ret = fi_av_open(domain, &av_attr, &ctx->av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			goto cleanup_cq;
		}
	}

	ret = fi_ep_bind(ctx->ep, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto cleanup_av;
	}

	ret = fi_ep_bind(ctx->ep, &ctx->av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto cleanup_av;
	}

	ret = fi_enable(ctx->ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		goto cleanup_av;
	}

	return 0;

cleanup_av:
	if (!topts.shared_av && ctx->av) {
		fi_close(&ctx->av->fid);
		ctx->av = NULL;
	}
cleanup_cq:
	if (!topts.shared_cq && ctx->cq) {
		fi_close(&ctx->cq->fid);
		ctx->cq = NULL;
	}
cleanup_ep:
	fi_close(&ctx->ep->fid);
	ctx->ep = NULL;
	return ret;
}

static void cleanup_endpoint(struct common_context *ctx)
{
	if (ctx->ep) {
		fi_close(&ctx->ep->fid);
		ctx->ep = NULL;
	}
	if (!topts.shared_av && ctx->av) {
		fi_close(&ctx->av->fid);
		ctx->av = NULL;
	}
	if (!topts.shared_cq && ctx->cq) {
		fi_close(&ctx->cq->fid);
		ctx->cq = NULL;
	}
}

// Notification handler thread for sender
void *notification_handler(void *arg)
{
	struct sender_context *ctx = (struct sender_context *) arg;

	while (ctx->status.active) {
		for (int sock_idx = 0; sock_idx < ctx->num_peers; sock_idx++) {
			if (ctx->control_socks[sock_idx] < 0)
				continue;
			struct ep_message msg;
			size_t bytes_received = 0;

			while (bytes_received < sizeof(msg)) {
				int ret = recv(ctx->control_socks[sock_idx],
					       (char *) &msg + bytes_received,
					       sizeof(msg) - bytes_received,
					       MSG_DONTWAIT);
				if (ret <= 0) {
					if (ret == 0) {
						fprintf(stderr,
							"Sender %d: control "
							"connection %d closed\n",
							ctx->worker_id,
							sock_idx);
						close(ctx->control_socks[sock_idx]);
						ctx->control_socks[sock_idx] = -1;
						break;
					} else if (errno == EAGAIN || errno == EWOULDBLOCK) {
						break;
					} else if (errno == EINTR) {
						continue;
					} else {
						fprintf(stderr,
							"Sender %d: recv failed on socket %d: "
							"%s\n",
							ctx->worker_id,
							sock_idx,
							strerror(errno));
						close(ctx->control_socks[sock_idx]);
						ctx->control_socks[sock_idx] = -1;
						break;
					}
				}
				bytes_received += ret;
			}

			// Process complete message if we got one
			if (bytes_received == sizeof(msg)) {
				pthread_mutex_lock(&ctx->status.mutex);

				// Find the correct peer index for this receiver
				// worker ID
				int worker_idx = -1;
				for (int i = 0; i < ctx->num_peers; i++) {
					if (ctx->peer_ids[i] == msg.info.worker_id) {
						worker_idx = i;
						break;
					}
				}

				if (worker_idx >= 0 &&
				    worker_idx < ctx->num_peers) {
					if (!ctx->peer_ep_addrs[worker_idx]) {
						ctx->peer_ep_addrs [worker_idx] = malloc(
							msg.info.addr_len);
					}
					if (ctx->peer_ep_addrs[worker_idx]) {
						memcpy(ctx->peer_ep_addrs[worker_idx],
						       msg.info.ep_addr,
						       msg.info.addr_len);
					}

					fi_addr_t fi_addr;
					if (ctx->common.av) {
						int ret = fi_av_insert(
							ctx->common.av,
							msg.info.ep_addr, 1,
							&fi_addr, 0, NULL);
						if (ret == 1) {
							ctx->peer_addrs[worker_idx] = fi_addr;
							memcpy(&ctx->peer_rma_info[worker_idx],
							       &msg.info.rma,
							       sizeof(struct rma_info));
							if (topts.verbose) {
								printf("Sender %d: Updated EP for receiver %d (fi_addr=%lu)\n",
								       ctx->worker_id,
								       msg.info.worker_id,
								       fi_addr);
							}
						} else {
							if (topts.verbose) {
								printf("Sender %d: Failed to insert address for receiver %d ret=%d\n",
								       ctx->worker_id,
								       msg.info.worker_id,
								       ret);
							}
						}
					} else {
						printf("Sender %d: No valid AV to insert address for receiver %d\n",
						       ctx->worker_id,
						       msg.info.worker_id);
					}
				} else if (topts.verbose) {
					printf("Sender %d: Ignoring notification from receiver %d (not assigned)\n",
					       ctx->worker_id,
					       msg.info.worker_id);
				}

				pthread_mutex_unlock(&ctx->status.mutex);
			}
		}

		// Re-insert addresses when sender has transient eps
		if (!topts.shared_av) {
			pthread_mutex_lock(&ctx->status.mutex);
			for (int i = 0; i < ctx->num_peers; i++) {
				if (ctx->peer_addrs[i] == FI_ADDR_UNSPEC &&
				    ctx->peer_ep_addrs[i] && ctx->common.av) {
					fi_addr_t fi_addr;
					int ret = fi_av_insert(
						ctx->common.av,
						ctx->peer_ep_addrs[i], 1,
						&fi_addr, 0, NULL);
					if (ret == 1) {
						ctx->peer_addrs[i] = fi_addr;
						if (topts.verbose) {
							printf("Sender %d: Re-inserted address for peer %d (fi_addr=%lu)\n",
							       ctx->worker_id, i, fi_addr);
						}
					}
				}
			}
			pthread_mutex_unlock(&ctx->status.mutex);
		}

		// Sleep briefly before next polling cycle
		usleep(10000); // 10ms
	}
	return NULL;
}

/*
 * Setup control socket server for endpoint notifications.
 * @param port_str: Port number as string
 * @param listen_sock: Pointer to store the listening socket
 * @return 0 on success, negative error code on failure
 */
static int control_setup_server(const char *port_str, int *listen_sock)
{
	int ret, optval = 1;
	struct addrinfo *ai, hints = {.ai_flags = AI_PASSIVE,
				      .ai_family = AF_UNSPEC,
				      .ai_socktype = SOCK_STREAM};

	ret = getaddrinfo(NULL, port_str, &hints, &ai);
	if (ret) {
		fprintf(stderr, "getaddrinfo() failed: %s\n",
			gai_strerror(ret));
		return -FI_EINVAL;
	}

	*listen_sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
	if (*listen_sock < 0) {
		ret = -errno;
		fprintf(stderr, "socket() failed: %s\n", strerror(errno));
		goto free_ai;
	}

	ret = setsockopt(*listen_sock, SOL_SOCKET, SO_REUSEADDR, &optval,
			 sizeof(optval));
	if (ret) {
		ret = -errno;
		fprintf(stderr, "setsockopt(SO_REUSEADDR) failed: %s\n",
			strerror(errno));
		goto close_listen_sock;
	}

	ret = bind(*listen_sock, ai->ai_addr, ai->ai_addrlen);
	if (ret) {
		ret = -errno;
		fprintf(stderr, "bind() failed: %s\n", strerror(errno));
		goto close_listen_sock;
	}

	ret = listen(*listen_sock, topts.num_receiver_workers);
	if (ret) {
		ret = -errno;
		fprintf(stderr, "listen() failed: %s\n", strerror(errno));
		goto close_listen_sock;
	}

	freeaddrinfo(ai);
	return 0;

close_listen_sock:
	close(*listen_sock);
	*listen_sock = -1;
free_ai:
	freeaddrinfo(ai);
	return ret;
}

/*
 * Setup control socket client for endpoint notifications.
 * @param server_addr: Server address
 * @param port_str: Port number as string
 * @param sock: Pointer to store the created socket
 * @return 0 on success, negative error code on failure
 */
static int control_setup_client(const char *server_addr, const char *port_str,
				int *sock)
{
	int ret;
	struct addrinfo *ai,
		hints = {.ai_family = AF_UNSPEC, .ai_socktype = SOCK_STREAM};

	ret = getaddrinfo(server_addr, port_str, &hints, &ai);
	if (ret) {
		fprintf(stderr, "getaddrinfo() failed: %s\n",
			gai_strerror(ret));
		return -FI_EINVAL;
	}

	*sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
	if (*sock < 0) {
		ret = -errno;
		fprintf(stderr, "socket() failed: %s\n", strerror(errno));
		goto free_ai;
	}

	// retry connect to wait for sender to start
	int connect_attempts = 0;
	do {
		ret = connect(*sock, ai->ai_addr, ai->ai_addrlen);
		if (!ret)
			break;
		if (errno == ECONNREFUSED && connect_attempts < 50) {
			usleep(100000); // 100ms
			connect_attempts++;
			continue;
		}
		ret = -errno;
		fprintf(stderr, "connect() failed: %s\n", strerror(errno));
		goto close_sock;
	} while (1);

	freeaddrinfo(ai);
	return 0;

close_sock:
	close(*sock);
	*sock = -1;
free_ai:
	freeaddrinfo(ai);
	return ret;
}

static int calculate_worker_distribution(int sender_id, int *num_peers,
					 int *peer_ids)
{
	if (topts.num_sender_workers <= topts.num_receiver_workers) {
		// Original round-robin distribution
		*num_peers =
			topts.num_receiver_workers / topts.num_sender_workers;
		for (int i = 0; i < *num_peers; i++) {
			peer_ids[i] = sender_id + i * topts.num_sender_workers;
		}
	} else {
		// Multiple senders share the same receiver
		*num_peers = 1;
		peer_ids[0] = sender_id % topts.num_receiver_workers;
	}
	return 0;
}

static int wait_for_comp(struct fid_cq *cq, int num_completions)
{
	struct fi_cq_data_entry comp;
	int ret;
	int completed = 0;
	struct timespec a, b;

	if (timeout >= 0)
		clock_gettime(CLOCK_MONOTONIC, &a);

	while (completed < num_completions) {
		if (topts.shared_cq) {
			// Acquire lock for shared CQ access
			pthread_mutex_lock(&shared_cq_lock);
			ret = fi_cq_read(cq, &comp, 1);
			pthread_mutex_unlock(&shared_cq_lock);
		} else {
			ret = fi_cq_read(cq, &comp, 1);
		}
		if (ret > 0) {
			completed++;
			continue;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			struct fi_cq_err_entry err_entry;
			if (topts.shared_cq)
				pthread_mutex_lock(&shared_cq_lock);
			fi_cq_readerr(cq, &err_entry, 0);
			if (topts.shared_cq)
				pthread_mutex_unlock(&shared_cq_lock);
			fprintf(stderr, "CQ read error: %s\n",
				fi_cq_strerror(cq, err_entry.prov_errno,
					       err_entry.err_data, NULL, 0));
		} else if (timeout >= 0) {
			clock_gettime(CLOCK_MONOTONIC, &b);
			if ((b.tv_sec - a.tv_sec) > timeout) {
				fprintf(stderr,
					"%ds timeout expired. Got %d "
					"completions\n",
					timeout, completed);
				return -FI_ENODATA;
			}
		}
	}

	return 0;
}

// Modified sender worker function
static void *run_sender_worker(void *arg)
{
	struct sender_context *ctx = (struct sender_context *) arg;
	int ret;
	uint64_t total_ops = 0;
	int msg_per_ep_lifecyle =
		topts.msgs_per_endpoint / topts.sender_ep_recycling;

	ctx->status.active = true;

	if (topts.shared_cq)
		ctx->common.cq = shared_txcq;

	if (topts.shared_av) {
		ctx->common.av = shared_txav;
	} else {
		av_attr.count = ctx->num_peers;
	}

	pthread_create(&ctx->notification_thread, NULL, notification_handler,
		       ctx);

	for (int cycle = 0; cycle < topts.sender_ep_recycling; cycle++) {
		ctx->status.poll_cq = true; // rand() % 2;
		printf("Sender %d: Starting EP cycle %d/%d\n", ctx->worker_id,
		       cycle + 1, topts.sender_ep_recycling);

		// Reset peer addresses for each cycle when AV will be recreated
		if (!topts.shared_av) {
			pthread_mutex_lock(&ctx->status.mutex);
			for (int i = 0; i < ctx->num_peers; i++) {
				ctx->peer_addrs[i] = FI_ADDR_UNSPEC;
			}
			pthread_mutex_unlock(&ctx->status.mutex);
		}

		// Setup new endpoint for this cycle
		ret = setup_endpoint(&ctx->common);
		if (ret) {
			fprintf(stderr, "Sender %d: endpoint setup failed\n",
				ctx->worker_id);
			goto out;
		}

		if (topts.verbose) {
			printf("Sender %d: Waiting for peer addresses for "
			       "cycle %d\n",
			       ctx->worker_id, cycle + 1);
		}
		bool peers_ready = false;
		int wait_attempts = 0;
		while (!peers_ready && ctx->status.active &&
		       wait_attempts < 100) {
			pthread_mutex_lock(&ctx->status.mutex);
			peers_ready = true;
			for (int i = 0; i < ctx->num_peers; i++) {
				if (ctx->peer_addrs[i] == FI_ADDR_UNSPEC) {
					peers_ready = false;
					break;
				}
			}
			pthread_mutex_unlock(&ctx->status.mutex);
			if (!peers_ready) {
				usleep(100000); // 100ms sleep before checking again
				wait_attempts++;
			}
		}

		if (!ctx->status.active) {
			ret = -FI_ECANCELED;
			goto out;
		}

		if (!peers_ready) {
			fprintf(stderr,
				"Sender %d: Timeout waiting for peer addresses "
				"in cycle %d\n",
				ctx->worker_id, cycle + 1);
			ret = -FI_ETIMEDOUT;
			goto cleanup;
		}

		if (topts.verbose) {
			printf("Sender %d: All peer addresses received for "
			       "cycle %d\n",
			       ctx->worker_id, cycle + 1);
		}

		// sleep random time up to 100ms to emulate the real workload
		int sleep_time = rand() % 100000;
		usleep(sleep_time);
		printf("Sender %d: Sleeping for %d microseconds\n",
		       ctx->worker_id, sleep_time);

		int pending_ops = 0;
		for (int i = 0; i < ctx->num_peers; i++) {
			// TODO: post messages one by one instead of in a loop,
			// allowing it to detect when the current_addr changes
			// during receiver ep recycling
			pthread_mutex_lock(&ctx->status.mutex);
			fi_addr_t current_addr = ctx->peer_addrs[i];
			pthread_mutex_unlock(&ctx->status.mutex);

			for (int j = 0; j < msg_per_ep_lifecyle; j++) {
				switch (topts.op_type) {
				case OP_MSG_UNTAGGED:
					ret = fi_send(
						ctx->common.ep, ctx->tx_buf,
						opts.transfer_size,
						fi_mr_desc(ctx->mr),
						current_addr,
						&ctx->tx_ctx[pending_ops]);
					break;
				case OP_MSG_TAGGED:
					ret = fi_tsend(
						ctx->common.ep, ctx->tx_buf,
						opts.transfer_size,
						fi_mr_desc(ctx->mr),
						current_addr, ft_tag,
						&ctx->tx_ctx[pending_ops]);
					break;
				case OP_RMA_WRITEDATA:
					pthread_mutex_lock(&ctx->status.mutex);
					struct rma_info *peer_rma =
						&ctx->peer_rma_info[i];
					pthread_mutex_unlock(
						&ctx->status.mutex);

					ret = fi_writedata(
						ctx->common.ep, ctx->tx_buf,
						opts.transfer_size,
						fi_mr_desc(ctx->mr),
						0xCAFE, // immediate data
						current_addr,
						peer_rma->remote_addr + (j * opts.transfer_size),
						peer_rma->rkey,
						&ctx->tx_ctx[pending_ops]);
					break;
				}

				if (ret) {
					fprintf(stderr,
						"Sender %d: operation failed. "
						"ret = %d, %s\n",
						ctx->worker_id, ret,
						fi_strerror(-ret));
					goto cleanup;
				}
				pending_ops++;
			}
		}

		// Wait for all operations to complete
		if (ctx->status.poll_cq) {
			printf("Sender %d EP cycle %d: Waiting for "
			       "completions\n",
			       ctx->worker_id, cycle + 1);
			ret = wait_for_comp(ctx->common.cq, pending_ops);
			if (ret) {
				fprintf(stderr,
					"Sender %d: completion failed: %s\n",
					ctx->worker_id, fi_strerror(-ret));
				goto cleanup;
			}
		} else {
			printf("Sender %d EP cycle %d: Not waiting for "
			       "completions\n",
			       ctx->worker_id, cycle + 1);
		}

		total_ops += pending_ops;

		if (topts.verbose) {
			printf("Sender %d: Completed cycle %d, ops=%d\n\n",
			       ctx->worker_id, cycle + 1, pending_ops);
		}

	cleanup:
		// Cleanup endpoint before next cycle
		cleanup_endpoint(&ctx->common);

		if (ret) {
			goto out;
		}

		// Small delay between cycles
		if (cycle < topts.sender_ep_recycling - 1) {
			usleep(1000);
		}
	}

	printf("Sender %d: All cycles completed, total ops=%lu\n",
	       ctx->worker_id, total_ops);

out:
	ctx->status.active = false;

	// Wait for notification thread to finish
	if (ctx->notification_thread) {
		pthread_join(ctx->notification_thread, NULL);
	}

	return (void *) (intptr_t) ret;
}

static int notify_endpoint_update(struct receiver_context *ctx)
{
	struct ep_message msg;
	msg.msg_type = MSG_TYPE_EP_UPDATE;
	msg.info.worker_id = ctx->worker_id;

	// Get endpoint address
	msg.info.addr_len = MAX_EP_ADDR_LEN;
	int ret = fi_getname(&ctx->common.ep->fid, msg.info.ep_addr,
			     &msg.info.addr_len);
	if (ret)
		return ret;

	// Fill RMA info
	msg.info.rma.remote_addr = (uint64_t) ctx->rx_buf;
	msg.info.rma.rkey = fi_mr_key(ctx->mr);
	msg.info.rma.length = opts.transfer_size * topts.msgs_per_endpoint * ctx->num_senders;

	// Send to all connected senders
	for (int i = 0; i < ctx->num_senders; i++) {
		if (ctx->control_socks[i] < 0)
			continue;

		size_t sent = 0;
		while (sent < sizeof(msg)) {
			ret = send(ctx->control_socks[i], (char *) &msg + sent,
				   sizeof(msg) - sent, 0);

			if (ret < 0) {
				if (errno == EAGAIN || errno == EWOULDBLOCK) {
					// Would block, try again after small delay
					usleep(1000); // 1ms delay
					continue;
				} else if (errno == EINTR) {
					// Interrupted by signal, retry
					// immediately
					continue;
				} else {
					// Real error
					fprintf(stderr,
						"Receiver %d: Failed to notify "
						"sender %d: %s\n",
						ctx->worker_id, i,
						strerror(errno));
					return -errno;
				}
			}
			sent += ret;
		}

		if (topts.verbose)
			printf("Receiver %d: Notified sender new EP\n",
			       ctx->worker_id);
	}

	return 0;
}

static void *run_receiver_worker(void *arg)
{
	struct receiver_context *ctx = (struct receiver_context *) arg;
	int ret = 0;
	int cycle = 0;
	int msg_per_ep_lifecyle =
		topts.msgs_per_endpoint / topts.receiver_ep_recycling;
	ctx->status.active = true;

	if (topts.shared_cq)
		ctx->common.cq = shared_rxcq;

	if (topts.shared_av) {
		ctx->common.av = shared_rxav;
	} else {
		av_attr.count = ctx->num_senders;
	}

	for (cycle = 0; cycle < topts.receiver_ep_recycling; cycle++) {
		ret = setup_endpoint(&ctx->common);
		if (ret)
			break;

		ctx->status.ep_generation++;
		ctx->status.ep_status = EP_RECEIVING;
		ctx->status.poll_cq = true; // rand() % 2;

		// Notify sender of new endpoint
		ret = notify_endpoint_update(ctx);
		if (ret < 0) {
			fprintf(stderr,
				"Receiver %d: Failed to notify endpoint update "
				"for cycle %d: %d\n",
				ctx->worker_id, cycle + 1, ret);
			cleanup_endpoint(&ctx->common);
			continue;
		}

		// sleep random time up to 100ms to emulate the real workload
		int sleep_time = rand() % 100000;
		usleep(sleep_time);
		printf("Receiver %d: Sleeping for %d microseconds\n",
		       ctx->worker_id, sleep_time);

		int completed = 0;
		int total_posts = 0;

		for (int sender = 0; sender < ctx->num_senders; sender++) {
			for (int msg = 0; msg < msg_per_ep_lifecyle; msg++) {
				void *rx_buf_offset =
					(char *) ctx->rx_buf +
					(sender * msg_per_ep_lifecyle + msg) *
						opts.transfer_size;

				switch (topts.op_type) {
				case OP_MSG_UNTAGGED:
					ret = fi_recv(ctx->common.ep,
						      rx_buf_offset,
						      opts.transfer_size,
						      fi_mr_desc(ctx->mr), 0,
						      &ctx->rx_ctx[msg]);
					break;
				case OP_MSG_TAGGED:
					ret = fi_trecv(
						ctx->common.ep, rx_buf_offset,
						opts.transfer_size,
						fi_mr_desc(ctx->mr), 0, ft_tag,
						0, &ctx->rx_ctx[msg]);
					break;
				case OP_RMA_WRITEDATA:
					continue;
				}

				if (ret) {
					fprintf(stderr,
						"Receiver %d: fi_recv failed "
						"for sender %d msg %d: %s\n",
						ctx->worker_id, sender, msg,
						fi_strerror(-ret));
					goto cleanup_cycle;
				}

				total_posts++;
			}

			if (topts.verbose &&
			    topts.op_type != OP_RMA_WRITEDATA) {
				printf("Receiver %d EP cycle %d: Posted %d "
				       "receives\n",
				       ctx->worker_id, cycle + 1,
				       msg_per_ep_lifecyle);
			}
		}

		if (ctx->status.poll_cq) {
			printf("Receiver %d EP cycle %d: Waiting for "
			       "completions\n",
			       ctx->worker_id, cycle + 1);
			int expected_completions =
				(topts.op_type == OP_RMA_WRITEDATA) ?
					ctx->num_senders * msg_per_ep_lifecyle :
					total_posts;

			if (expected_completions > 0) {
				ret = wait_for_comp(ctx->common.cq,
						    expected_completions);
				if (ret) {
					fprintf(stderr,
						"Receiver %d: Receive "
						"completion timeout, "
						"recycling endpoint\n",
						ctx->worker_id);
				} else {
					completed = expected_completions;
					ctx->total_received += completed;
				}
			}
		} else {
			printf("Receiver %d EP cycle %d: Not waiting for "
			       "completions\n",
			       ctx->worker_id, cycle + 1);
		}

		printf("Receiver %d EP cycle %d: Completed %d receives from %d "
		       "senders\n\n",
		       ctx->worker_id, cycle + 1, completed, ctx->num_senders);

	cleanup_cycle:
		cleanup_endpoint(&ctx->common);
		// Small delay between cycles
		if (cycle < topts.receiver_ep_recycling - 1) {
			usleep(1000);
		}
	}

	printf("Receiver %d: Completed %d EP cycles\n", ctx->worker_id, cycle);
	ctx->status.active = false;
	return (void *) (intptr_t) ret;
}

// Common function for buffer and MR setup
static int setup_worker_resources(void **buf, struct fi_context2 **ctx,
				  struct fid_mr **mr, uint64_t access,
				  int buffer_multiplier)
{
	int ret;

	// Allocate buffer - receivers need space for all senders' messages
	size_t buffer_size = opts.transfer_size * topts.msgs_per_endpoint *
			     buffer_multiplier;
	*buf = calloc(1, buffer_size);
	if (!*buf) {
		return -FI_ENOMEM;
	}

	// Allocate context array
	*ctx = calloc(topts.msgs_per_endpoint, sizeof(struct fi_context));
	if (!*ctx) {
		free(*buf);
		*buf = NULL;
		return -FI_ENOMEM;
	}

	// Register memory region
	ret = fi_mr_reg(domain, *buf, buffer_size, access, 0, 0, 0, mr, NULL);
	if (ret) {
		FT_PRINTERR("fi_mr_reg", ret);
		free(*ctx);
		free(*buf);
		*buf = NULL;
		*ctx = NULL;
		return ret;
	}

	return 0;
}

static int run_sender(void)
{
	int ret;
	struct sender_context *workers;
	pthread_t *threads;
	int *listen_socks = NULL;

	workers = calloc(topts.num_sender_workers, sizeof(*workers));
	threads = calloc(topts.num_sender_workers, sizeof(*threads));
	if (!workers || !threads) {
		ret = -FI_ENOMEM;
		goto out;
	}

	printf("\nSender Worker Distribution:\n");
	printf("-------------------------\n");
	printf("Total: %d senders, %d receivers\n", topts.num_sender_workers,
	       topts.num_receiver_workers);

	if (topts.shared_cq) {
		cq_attr.format = FI_CQ_FORMAT_CONTEXT;
		cq_attr.size = fi->tx_attr->size;
		ret = fi_cq_open(domain, &cq_attr, &shared_txcq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			goto out;
		}
	}

	if (topts.shared_av) {
		av_attr.type = FI_AV_TABLE;
		av_attr.count = MAX_PEERS * topts.num_sender_workers;
		ret = fi_av_open(domain, &av_attr, &shared_txav, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			goto out;
		}
	}

	listen_socks = calloc(topts.num_sender_workers, sizeof(int));
	if (!listen_socks) {
		ret = -FI_ENOMEM;
		goto out;
	}
	for (int i = 0; i < topts.num_sender_workers; i++) {
		listen_socks[i] = -1;
	}

	for (int i = 0; i < topts.num_sender_workers; i++) {
		workers[i].worker_id = i;
		pthread_mutex_init(&workers[i].status.mutex, NULL);

		int num_peers;
		int *peer_ids = calloc(MAX_PEERS, sizeof(int));
		if (!peer_ids) {
			ret = -FI_ENOMEM;
			fprintf(stderr,
				"Failed to allocate peer_ids for sender %d\n",
				i);
			goto out;
		}

		calculate_worker_distribution(i, &num_peers, peer_ids);
		workers[i].num_peers = num_peers;

		workers[i].peer_ids = peer_ids;
		// Setup common resources
		ret = setup_worker_resources(&workers[i].tx_buf,
					     &workers[i].tx_ctx, &workers[i].mr,
					     FI_SEND | FI_WRITE, 1);
		if (ret) {
			fprintf(stderr,
				"setup_worker_resources failed for sender %d: "
				"%d\n",
				i, ret);
			goto out;
		}

		workers[i].peer_addrs = calloc(num_peers, sizeof(fi_addr_t));
		if (!workers[i].peer_addrs) {
			ret = -FI_ENOMEM;
			goto out;
		}

		// Initialize all addresses to FI_ADDR_UNSPEC
		for (int j = 0; j < num_peers; j++) {
			workers[i].peer_addrs[j] = FI_ADDR_UNSPEC;
		}

		workers[i].peer_rma_info =
			calloc(num_peers, sizeof(struct rma_info));
		if (!workers[i].peer_rma_info) {
			ret = -FI_ENOMEM;
			goto out;
		}

		// Allocate arrays for caching endpoint addresses
		workers[i].peer_ep_addrs = calloc(num_peers, sizeof(char *));
		if (!workers[i].peer_ep_addrs) {
			ret = -FI_ENOMEM;
			goto out;
		}

		char port_str[16];
		snprintf(port_str, sizeof(port_str), "%d", NOTIFY_PORT + i);

		printf("\nSender Worker %d:\n", i);
		printf("  - Port: %s\n", port_str);
		printf("  - Number of receivers: %d\n", num_peers);
		printf("  - Assigned receivers: ");
		for (int j = 0; j < num_peers; j++) {
			printf("%d ", peer_ids[j]);
		}
		printf("\n");

		// Allocate control sockets array
		workers[i].control_socks = calloc(num_peers, sizeof(int));
		if (!workers[i].control_socks) {
			ret = -FI_ENOMEM;
			goto out;
		}

		// Initialize all control sockets to -1
		for (int j = 0; j < num_peers; j++) {
			workers[i].control_socks[j] = -1;
		}

		// Start listening socket for all workers
		ret = control_setup_server(port_str, &listen_socks[i]);
		if (ret) {
			fprintf(stderr,
				"control_setup_server failed for worker %d: "
				"%d\n",
				i, ret);
			goto out;
		}

		printf("Sender %d: listening on port %s\n", i, port_str);
	}

	for (int i = 0; i < topts.num_sender_workers; i++) {
		for (int conn = 0; conn < workers[i].num_peers; conn++) {
			workers[i].control_socks[conn] =
				accept(listen_socks[i], NULL, NULL);
			if (workers[i].control_socks[conn] < 0) {
				ret = -errno;
				fprintf(stderr,
					"accept() failed for worker %d "
					"connection %d: %s\n",
					i, conn, strerror(errno));
				goto out;
			}
			printf("Sender %d: Accepted receiver %d's "
			       "connection\n\n",
			       i, workers[i].peer_ids[conn]);
		}

		close(listen_socks[i]);
		listen_socks[i] = -1;
	}

	// Create worker threads
	for (int i = 0; i < topts.num_sender_workers; i++) {
		ret = pthread_create(&threads[i], NULL, run_sender_worker,
				     &workers[i]);
		if (ret) {
			printf("Failed to create sender thread: %d\n", ret);
			goto out;
		}
	}

	// Wait for completion
	for (int i = 0; i < topts.num_sender_workers; i++) {
		pthread_join(threads[i], NULL);
	}

out:
	if (workers) {
		if (topts.shared_av && shared_txav) {
			fi_close(&shared_txav->fid);
		}
		if (topts.shared_cq && shared_txcq) {
			fi_close(&shared_txcq->fid);
		}
		for (int i = 0; i < topts.num_sender_workers; i++) {
			if (workers[i].control_socks) {
				for (int j = 0; j < workers[i].num_peers; j++) {
					if (workers[i].control_socks[j] >= 0)
						close(workers[i].control_socks[j]);
				}
				free(workers[i].control_socks);
			}
			if (listen_socks && listen_socks[i] >= 0) {
				close(listen_socks[i]);
				listen_socks[i] = -1;
			}
			if (workers[i].mr)
				fi_close(&workers[i].mr->fid);
			if (workers[i].peer_ids)
				free(workers[i].peer_ids);
			// Free cached endpoint addresses
			if (workers[i].peer_ep_addrs) {
				for (int j = 0; j < workers[i].num_peers; j++) {
					if (workers[i].peer_ep_addrs[j]) {
						free(workers[i].peer_ep_addrs[j]);
					}
				}
				free(workers[i].peer_ep_addrs);
			}
			free(workers[i].tx_buf);
			free(workers[i].tx_ctx);
			free(workers[i].peer_addrs);
			free(workers[i].peer_rma_info);
			pthread_mutex_destroy(&workers[i].status.mutex);
		}
		if (listen_socks)
			free(listen_socks);
	}
	free(workers);
	free(threads);
	return ret;
}

static int run_receiver(void)
{
	int ret;
	struct receiver_context *workers;
	pthread_t *threads;

	workers = calloc(topts.num_receiver_workers, sizeof(*workers));
	threads = calloc(topts.num_receiver_workers, sizeof(*threads));
	if (!workers || !threads) {
		ret = -FI_ENOMEM;
		goto out;
	}

	printf("\nReceiver Worker Distribution:\n");
	printf("-------------------------\n");
	printf("Total: %d receivers, %d senders\n", topts.num_receiver_workers,
	       topts.num_sender_workers);

	if (topts.shared_cq) {
		cq_attr.format = FI_CQ_FORMAT_CONTEXT;
		cq_attr.size = fi->rx_attr->size;
		ret = fi_cq_open(domain, &cq_attr, &shared_rxcq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (topts.shared_av) {
		av_attr.type = FI_AV_TABLE;
		av_attr.count = MAX_PEERS * topts.num_receiver_workers;
		ret = fi_av_open(domain, &av_attr, &shared_rxav, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			goto out;
		}
	}

	// Initialize workers
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		workers[i].worker_id = i;
		pthread_mutex_init(&workers[i].status.mutex, NULL);

		// Calculate number of senders this receiver talks to
		if (topts.num_sender_workers <= topts.num_receiver_workers) {
			// Each receiver talks to one sender
			workers[i].num_senders = 1;
		} else {
			// Multiple senders may send to this receiver
			int count = 0;
			for (int j = i; j < topts.num_sender_workers;
			     j += topts.num_receiver_workers) {
				count++;
			}
			workers[i].num_senders = count;
		}

		// Setup common resources
		ret = setup_worker_resources(
			&workers[i].rx_buf, &workers[i].rx_ctx, &workers[i].mr,
			FI_RECV | FI_REMOTE_WRITE, workers[i].num_senders);
		if (ret) {
			goto out;
		}

		// Allocate control socket array
		workers[i].control_socks =
			calloc(workers[i].num_senders, sizeof(int));
		if (!workers[i].control_socks) {
			ret = -FI_ENOMEM;
			goto out;
		}

		if (topts.verbose) {
			printf("\nReceiver Worker %d:\n", i);
			printf("  - Connected by senders: ");
			if (topts.num_sender_workers <= topts.num_receiver_workers) {
				printf("%d ", i % topts.num_sender_workers);
			} else {
				for (int j = i; j < topts.num_sender_workers; j += topts.num_receiver_workers)
					printf("%d ", j);
			}
			printf("\n");
		}

		int sock_idx = 0;
		int sender_count = (topts.num_sender_workers <=
				    topts.num_receiver_workers) ?
					   1 :
					   workers[i].num_senders;
		for (int j = 0; j < sender_count; j++) {
			int sender_idx;
			if (topts.num_sender_workers <= topts.num_receiver_workers) {
				// Each receiver connects to one sender
				sender_idx = i % topts.num_sender_workers;
			} else {
				// Multiple senders connect to this receiver
				sender_idx = i + j * topts.num_receiver_workers;
				if (sender_idx >= topts.num_sender_workers)
					break;
			}
			char port_str[16];
			snprintf(port_str, sizeof(port_str), "%d",
				 NOTIFY_PORT + sender_idx);

			if (topts.verbose) {
				printf("Receiver %d connecting to sender %d on "
				       "port %s\n\n",
				       i, sender_idx, port_str);
			}

			if (sock_idx >= workers[i].num_senders) {
				fprintf(stderr,
					"Receiver %d: Too many sender "
					"connections (sock_idx=%d)\n",
					i, sock_idx);
				break;
			}

			int *control_sock = &workers[i].control_socks[sock_idx];
			ret = control_setup_client(topts.sender_addr, port_str,
						   control_sock);
			if (ret) {
				fprintf(stderr,
					"control_setup_client failed for "
					"worker %d->%d: %d\n",
					i, sender_idx, ret);
				goto out;
			}
			sock_idx++;
		}
	}

	// Create worker threads
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		ret = pthread_create(&threads[i], NULL, run_receiver_worker,
				     &workers[i]);
		if (ret) {
			printf("Failed to create receiver thread: %d\n", ret);
			goto out;
		}
	}

	// Wait for completion
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		pthread_join(threads[i], NULL);
	}

out:
	if (workers) {
		if (topts.shared_av && shared_rxav) {
			fi_close(&shared_rxav->fid);
		}
		if (topts.shared_cq && shared_rxcq) {
			fi_close(&shared_rxcq->fid);
		}
		for (int i = 0; i < topts.num_receiver_workers; i++) {
			if (workers[i].control_socks) {
				// Close all control sockets
				for (int j = 0; j < workers[i].num_senders; j++) {
					if (workers[i].control_socks[j] >= 0)
						close(workers[i].control_socks[j]);
				}
				free(workers[i].control_socks);
			}
			if (workers[i].mr)
				fi_close(&workers[i].mr->fid);
			if (!topts.shared_av && workers[i].common.av) {
				fi_close(&workers[i].common.av->fid);
				workers[i].common.av = NULL;
			}
			free(workers[i].rx_buf);
			free(workers[i].rx_ctx);
			pthread_mutex_destroy(&workers[i].status.mutex);
		}
	}
	free(workers);
	free(threads);
	return ret;
}

static int run_test(void)
{
	int ret;

	if (topts.random_seed != -1) {
		printf("-------------------------\n");
		printf("Using provided random seed: %ld\n", topts.random_seed);
		printf("-------------------------\n\n");
	} else {
		topts.random_seed = time(NULL);
		printf("-------------------------\n");
		printf("Generated random seed: %ld\n", topts.random_seed);
		printf("-------------------------\n\n");
	}

	/* Seed PRNG */
	srand(topts.random_seed);

	ret = ft_init_fabric();
	if (ret)
		return ret;

	// Run as sender or receiver based on dst_addr
	if (opts.dst_addr) {
		ret = run_sender();
	} else {
		ret = run_receiver();
	}

	return ret;
}

static void print_test_usage(void)
{
	FT_PRINT_OPTS_USAGE("--sender-workers <N>",
			    "number of sender workers (default: 1)");
	FT_PRINT_OPTS_USAGE("--receiver-workers <N>",
			    "number of receiver workers (default: 1)");
	FT_PRINT_OPTS_USAGE("--msgs-per-ep <N>",
			    "messages per endpoint (default: 1000)");
	FT_PRINT_OPTS_USAGE(
		"--sender-ep-cycles <N>",
		"number of sender endpoint recycling iterations (default: 1)");
	FT_PRINT_OPTS_USAGE("--receiver-ep-cycles <N>",
			    "number of receiver endpoint recycling iterations "
			    "(default: 1)");
	FT_PRINT_OPTS_USAGE("--shared-av",
			    "use shared AV among workers (default: off)");
	FT_PRINT_OPTS_USAGE("--shared-cq",
			    "use shared CQ among workers (default: off)");
	FT_PRINT_OPTS_USAGE("--op-type <type>",
			    "operation type: untagged|tagged|writedata "
			    "(default: untagged)");
	FT_PRINT_OPTS_USAGE("--sender-addr <addr>",
			    "address of the sender (required for receiver)");
	FT_PRINT_OPTS_USAGE("--random-seed <seed>",
			    "random seed to use for the test. Default value is time(NULL).");
}

static int parse_test_opts(int argc, char **argv)
{
	int op;

	topts.random_seed = -1;

	while ((op = getopt_long(argc, argv, "hAQ" ADDR_OPTS INFO_OPTS CS_OPTS,
				 test_long_opts, NULL)) != -1) {
		switch (op) {
		case OPT_SENDER_WORKERS:
			topts.num_sender_workers = atoi(optarg);
			if (topts.num_sender_workers < 1) {
				fprintf(stderr, "number of sender workers must "
						"be at least 1\n");
				return -1;
			}
			break;
		case OPT_RECEIVER_WORKERS:
			topts.num_receiver_workers = atoi(optarg);
			if (topts.num_receiver_workers < 1) {
				fprintf(stderr, "number of receiver workers "
						"must be at least 1\n");
				return -1;
			}
			break;
		case OPT_MSGS_PER_EP:
			topts.msgs_per_endpoint = atoi(optarg);
			if (topts.msgs_per_endpoint < 1) {
				fprintf(stderr, "messages per endpoint must be "
						"at least 1\n");
				return -1;
			}
			break;
		case OPT_SENDER_EP_CYCLES:
			topts.sender_ep_recycling = atoi(optarg);
			if (topts.sender_ep_recycling < 1) {
				fprintf(stderr, "sender EP recycling must be "
						"at least 1\n");
				return -1;
			}
			break;
		case OPT_RECEIVER_EP_CYCLES:
			topts.receiver_ep_recycling = atoi(optarg);
			if (topts.receiver_ep_recycling < 1) {
				fprintf(stderr, "receiver EP recycling must be "
						"at least 1\n");
				return -1;
			}
			break;
		case OPT_SHARED_AV:
			topts.shared_av = true;
			break;
		case OPT_SHARED_CQ:
			topts.shared_cq = true;
			break;
		case OPT_OP_TYPE:
			if (strcmp(optarg, "untagged") == 0) {
				topts.op_type = OP_MSG_UNTAGGED;
			} else if (strcmp(optarg, "tagged") == 0) {
				topts.op_type = OP_MSG_TAGGED;
				ft_tag = 0x123;
			} else if (strcmp(optarg, "writedata") == 0) {
				topts.op_type = OP_RMA_WRITEDATA;
			} else {
				fprintf(stderr, "invalid operation type: %s\n",
					optarg);
				return -1;
			}
			break;
		case OPT_SENDER_ADDR:
			topts.sender_addr = optarg;
			break;
		case OPT_RANDOM_SEED:
			topts.random_seed = atol(optarg);
			break;
		case '?':
		case 'h':
			ft_usage(argv[0], "Endpoint recycling test");
			print_test_usage();
			return -2;
		default:
			ft_parse_addr_opts(op, optarg, &opts);
			ft_parseinfo(op, optarg, hints, &opts);
			ft_parsecsopts(op, optarg, &opts);
			break;
		}
	}

	return 0;
}

// Main function
int main(int argc, char **argv)
{
	int ret;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SIZE;
	timeout = 10;

	hints = fi_allocinfo();
	if (!hints)
		return EXIT_FAILURE;

	ret = parse_test_opts(argc, argv);
	if (ret)
		goto out;

	// Set up hints
	opts.threading = FI_THREAD_SAFE;
	hints->caps = FI_MSG | FI_RMA;
	hints->mode = FI_CONTEXT | FI_CONTEXT2;
	hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_LOCAL |
				      FI_MR_VIRT_ADDR | FI_MR_PROV_KEY |
				      FI_MR_HMEM;
	hints->ep_attr->type = FI_EP_RDM;
	hints->fabric_attr->prov_name = strdup("efa");

	if (optind < argc)
		opts.dst_addr = argv[optind];

	if (!topts.sender_addr && !opts.dst_addr) {
		fprintf(stderr, "Error: --sender-addr must be specified in "
				"receiver command for socket connection\n");
		print_test_usage();
		ret = -1;
		goto out;
	}

	ret = run_test();

out:
	ft_free_res();
	return ft_exit_code(ret);
}
