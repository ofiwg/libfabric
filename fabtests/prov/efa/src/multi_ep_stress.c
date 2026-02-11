#include <getopt.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <rdma/fabric.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_errno.h>

#include <rdma/fi_rma.h>
#include <rdma/fi_tagged.h>

#include "hmem.h"
#include "shared.h"
#include "ft_random.h"

#define MAX_WORKERS	64
#define MAX_PEERS	MAX_WORKERS
#define MAX_EP_ADDR_LEN 256

// Configuration structure
struct test_opts {
	int num_sender_workers;
	int num_receiver_workers;
	int msgs_per_sender;
	/* Number of times for sender to recycle endpoints */
	int sender_ep_recycling;
	/* Number of times for receiver to recycle endpoints */
	int receiver_ep_recycling;
	bool remove_av;
	bool shared_av;
	bool shared_cq;
	enum { OP_MSG_UNTAGGED = 0, OP_MSG_TAGGED, OP_RMA_WRITEDATA } op_type;
	bool verbose;
	time_t random_seed;
};

// Global variables
static struct test_opts topts = {
	.num_sender_workers = 1,
	.num_receiver_workers = 1,
	.msgs_per_sender = 1000,
	.sender_ep_recycling = 1, // Default to 1 recycling for sender
	.receiver_ep_recycling = 1, // Default to 1 recycling for receiver
	.remove_av = false, // Default: do not remove old AV
	.shared_av = false, // Default: 1 AV per EP
	.shared_cq = false, // Default: 1 CQ per EP
	.op_type = OP_MSG_UNTAGGED,
	.verbose = true,
};

enum {
	OPT_SENDER_WORKERS = 256,
	OPT_RECEIVER_WORKERS,
	OPT_MSGS_PER_EP,
	OPT_SENDER_EP_CYCLES,
	OPT_RECEIVER_EP_CYCLES,
	OPT_REMOVE_AV,
	OPT_SHARED_AV,
	OPT_SHARED_CQ,
	OPT_OP_TYPE,
	OPT_RANDOM_SEED,
};

static struct option test_long_opts[] = {
	{"sender-workers", required_argument, NULL, OPT_SENDER_WORKERS},
	{"receiver-workers", required_argument, NULL, OPT_RECEIVER_WORKERS},
	{"msgs-per-ep", required_argument, NULL, OPT_MSGS_PER_EP},
	{"sender-ep-cycles", required_argument, NULL, OPT_SENDER_EP_CYCLES},
	{"receiver-ep-cycles", required_argument, NULL, OPT_RECEIVER_EP_CYCLES},
	{"remove-av", no_argument, NULL, OPT_REMOVE_AV},
	{"shared-av", no_argument, NULL, OPT_SHARED_AV},
	{"shared-cq", no_argument, NULL, OPT_SHARED_CQ},
	{"op-type", required_argument, NULL, OPT_OP_TYPE},
	{"random-seed", required_argument, NULL, OPT_RANDOM_SEED},
	{0, 0, 0, 0}};

// RMA information
struct rma_info {
	uint64_t remote_addr;
	uint64_t rkey;
};

// Endpoint metadata
struct ep_info {
	uint16_t worker_id; // destination - worker ID on sender side
	uint16_t peer_idx;  // source - index of receiver worker in sender's peer_ids table
	char ep_addr[MAX_EP_ADDR_LEN];
	struct rma_info rma;
};

// Message structure for endpoint updates
struct ep_message {
	enum {
		EP_MESSAGE_TYPE_TERMINATOR = 0,
		EP_MESSAGE_TYPE_UPDATE,
	} type;
	struct ep_info info;
};

// Multi-producer single consumer thread-safe queue
struct ep_message_queue {
	struct ep_message *messages;
	pthread_mutex_t mutex;
	pthread_cond_t cond;
	size_t size;
	size_t writer;
	size_t reader;
};

#define EP_MESSAGE_QUEUE_CAPACITY (8) // should be power of 2 for best performance

static int ep_message_queue_init(struct ep_message_queue *q) {
	int ret;
	ret = pthread_mutex_init(&q->mutex, NULL);
	if (ret)
		return ret;
	ret = pthread_cond_init(&q->cond, NULL);
	if (ret)
		return ret;
	q->messages = calloc(EP_MESSAGE_QUEUE_CAPACITY, sizeof(struct ep_message));
	if (q->messages == NULL)
		return -FI_ENOMEM;
	q->size = 0;
	q->writer = 0;
	q->reader = 0;
	return 0;
}

static void ep_message_queue_destroy(struct ep_message_queue *q) {
	pthread_mutex_destroy(&q->mutex);
	pthread_cond_destroy(&q->cond);
	free(q->messages);
}

static int ep_message_queue_push(struct ep_message_queue *q, const struct ep_message *src) {
	int ret = 0;
	ret = pthread_mutex_lock(&q->mutex);
	if (ret)
		return ret;
	while (q->size == EP_MESSAGE_QUEUE_CAPACITY) {
		ret = pthread_cond_wait(&q->cond, &q->mutex);
		if (ret)
			goto out;
	}

	q->messages[q->writer] = *src;
	q->writer = (q->writer + 1) % EP_MESSAGE_QUEUE_CAPACITY;
	q->size++;
	ret = pthread_cond_signal(&q->cond);
out:
	pthread_mutex_unlock(&q->mutex);
	return ret;
}

static int ep_message_queue_pop(struct ep_message_queue *q, struct ep_message *dst) {
	int ret = 0;
	ret = pthread_mutex_lock(&q->mutex);
	if (ret)
		return ret;
	while(q->size == 0) {
		ret = pthread_cond_wait(&q->cond, &q->mutex);
		if (ret)
			goto out;
	}

	*dst = q->messages[q->reader];
	q->size--;
	q->reader = (q->reader + 1) % EP_MESSAGE_QUEUE_CAPACITY;
	ret = pthread_cond_signal(&q->cond);
out:
	pthread_mutex_unlock(&q->mutex);
	return ret;
}

static int ep_message_queue_try_pop(struct ep_message_queue *q, struct ep_message *dst) {
	int ret = 0;
	ret = pthread_mutex_lock(&q->mutex);
	if (ret)
		return ret;
	if (q->size == 0) {
		pthread_mutex_unlock(&q->mutex);
		return -FI_EAGAIN;
	}
	*dst = q->messages[q->reader];
	q->size--;
	q->reader = (q->reader + 1) % EP_MESSAGE_QUEUE_CAPACITY;
	ret = pthread_cond_signal(&q->cond);
	pthread_mutex_unlock(&q->mutex);
	return ret;
}

// A memory pool for pre-registered buffers and fi_ctx
struct context_pool {
	void *buffers;
	struct fi_context2 *fi_ctx;
	size_t capacity;
	size_t allocated;
	size_t buffer_size;
	struct fid_mr *mr;
};

// Worker context structure
struct worker_context {
	size_t num_peers;
	uint16_t *peer_ids;
	struct ep_message_queue *control_queue;
	struct fid_ep *ep;
	struct fid_cq *cq;
	struct fid_av *av;
	struct context_pool pool;
	uint16_t worker_id;
};
struct fid_cq *shared_cq = NULL;
struct fid_av *shared_av = NULL;

int context_pool_init(struct context_pool *pool, size_t capacity, size_t buffer_size, uint64_t access) {
	pool->mr = NULL;
	pool->buffer_size = buffer_size;
	pool->capacity = capacity;
	pool->allocated = 0;
	pool->buffers = calloc(capacity, buffer_size);
	pool->fi_ctx = (struct fi_context2*) calloc(capacity, sizeof(struct fi_context2));
	if (pool->buffers == NULL || pool->fi_ctx == NULL) {
		pool->capacity = 0;
		return -FI_ENOMEM;
	}
	// Register buffer memory region
	return fi_mr_reg(domain, pool->buffers, capacity*buffer_size, access, 0, 0, 0, &pool->mr, NULL);
}

void context_pool_destroy(struct context_pool *pool) {
	if (pool->mr) {
		fi_close(&pool->mr->fid);
		pool->mr = NULL;
	}
	free(pool->buffers);
	free(pool->fi_ctx);
	pool->buffers = NULL;
	pool->fi_ctx = NULL;
	pool->capacity = 0;
	pool->allocated = 0;
}

int context_pool_alloc_ctx(struct context_pool *pool, void **buffer, struct fi_context2 **fi_ctx) {
	assert(pool->buffers != NULL && pool->fi_ctx != NULL);
	if (pool->capacity <= pool->allocated)
		return -FI_ENOMEM;
	*buffer = (void*)((char*) pool->buffers + pool->buffer_size * pool->allocated);
	*fi_ctx = pool->fi_ctx + pool->allocated;
	pool->allocated++;
	return 0;
}

int context_pool_alloc_buffers(struct context_pool *pool, size_t count, void **buffers) {
	assert(pool->buffers != NULL);
	if (pool->allocated + count > pool->capacity)
		return -FI_ENOMEM;
	*buffers = (void*)((char* )pool->buffers + pool->allocated);
	pool->allocated += count;
	return 0;
}

// Cleanup fabric resorsces for a worker
static void cleanup_endpoint(struct worker_context *ctx)
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

// Setup endpoint for a worker
static int setup_endpoint(struct worker_context *ctx, uint64_t total_ops)
{
	int ret;

	ret = fi_endpoint(domain, fi, &ctx->ep, NULL);
	if (ret) {
		FT_PRINTERR("fi_endpoint", ret);
		return ret;
	}

	if (topts.shared_cq) {
		ctx->cq = shared_cq;
	} else {
		struct fi_cq_attr attr = {
			.wait_obj = FI_WAIT_NONE,
			.format = FI_CQ_FORMAT_CONTEXT,
			.size = total_ops,
		};

		ret = fi_cq_open(domain, &attr, &ctx->cq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			goto error;
		}
	}

	if (topts.shared_av) {
		ctx->av = shared_av;
	} else {
		struct fi_av_attr attr = {
			.type = FI_AV_TABLE,
			.count = ctx->num_peers,
		};
		ret = fi_av_open(domain, &attr, &ctx->av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			goto error;
		}
	}

	ret = fi_ep_bind(ctx->ep, &ctx->cq->fid, FI_SEND | FI_RECV);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto error;
	}

	ret = fi_ep_bind(ctx->ep, &ctx->av->fid, 0);
	if (ret) {
		FT_PRINTERR("fi_ep_bind", ret);
		goto error;
	}

	ret = fi_enable(ctx->ep);
	if (ret) {
		FT_PRINTERR("fi_enable", ret);
		goto error;
	}

	return 0;
error:
	cleanup_endpoint(ctx);
	return ret;
}

/**
 * calculate_worker_distribution - Distribute peers across workers
 * @current_worker_id: ID of the worker to calculate distribution for
 * @total_workers: Total number of workers in the system
 * @total_peers: Total number of peers to distribute
 * @num_peers: Output - number of peers assigned to this worker
 * @peer_ids: Output - dynamically allocated array of peer IDs for this worker
 *
 * Distributes peers evenly across workers. When peers <= workers, each worker
 * gets one peer (worker_id % total_peers). When peers > workers, distributes
 * peers round-robin with remainder distributed to lower-numbered workers.
 *
 * Returns: FI_SUCCESS on success, -FI_ENOMEM on allocation failure
 *
 * Note: Caller must free the allocated peer_ids array
 */
static int calculate_worker_distribution(uint16_t current_worker_id,
					uint16_t total_workers,
					uint16_t total_peers,
					size_t *num_peers,
					uint16_t **peer_ids)
{
	if (total_peers <= total_workers) {
		// Each worker has one peer
		*num_peers = 1;
		*peer_ids = malloc(sizeof(int));
		if (*peer_ids == NULL)
			goto error;
		**peer_ids = current_worker_id % total_peers;
	} else {
		// Multiple peers per worker
		*num_peers = total_peers / total_workers;
		if (current_worker_id < total_peers % total_workers)
			(*num_peers)++;
		*peer_ids = malloc(*num_peers * sizeof(int));
		if (*peer_ids == NULL)
			goto error;
		for (size_t i = 0; i < *num_peers; i++) {
			(*peer_ids)[i] = current_worker_id + i * total_workers;
		}
	}
	return FI_SUCCESS;
error:
	fprintf(stderr,
		"Failed to allocate peer_ids for worker %d\n",
		current_worker_id);
	return -FI_ENOMEM;
}

/**
 * wait_for_comp - Poll completion queue until expected completions arrive
 * @cq: Completion queue to poll
 * @num_completions: Number of completions to wait for
 *
 * Polls the specified completion queue until the requested number of
 * completions are received or a timeout occurs. Uses the global 'timeout'
 * variable (default: 10 seconds) to limit wait duration. When shared CQ
 * mode is enabled, acquires a mutex lock with timeout to serialize access.
 *
 * Returns: Number of completions successfully received (may be less than
 *          requested if timeout expires or error occurs)
 */
static int wait_for_comp(struct fid_cq *cq, int num_completions)
{
	static pthread_mutex_t shared_cq_lock = PTHREAD_MUTEX_INITIALIZER;
	struct fi_cq_data_entry comp;
	int ret;
	int completed = 0;
	struct timespec a, b;

	clock_gettime(CLOCK_MONOTONIC, &a);
	if (topts.shared_cq) {
		memcpy(&b, &a, sizeof(b));
		b.tv_sec += timeout;
		ret = pthread_mutex_timedlock(&shared_cq_lock, &b);
		if (ret == ETIMEDOUT) {
			fprintf(stderr,
				"%ds timeout expired while waiting for shared CQ lock\n",
				timeout);
			return 0;
		}
		if (ret) {
			FT_PRINTERR("pthread_mutex_timedlock", ret);
			abort();
		}
	}

	while (completed < num_completions) {
		ret = fi_cq_read(cq, &comp, 1);
		if (ret > 0) {
			assert(ret == 1);
			completed++;
			continue;
		} else if (ret < 0 && ret != -FI_EAGAIN) {
			struct fi_cq_err_entry err_entry = {0};
			fi_cq_readerr(cq, &err_entry, 0);
			fprintf(stderr, "CQ read error: %s\n",
				fi_cq_strerror(cq, err_entry.prov_errno,
					       err_entry.err_data, NULL, 0));
		} else if (timeout > 0) {
			clock_gettime(CLOCK_MONOTONIC, &b);
			if (b.tv_sec - a.tv_sec >= timeout) {
				fprintf(stderr,
					"%ds timeout expired. Got %d "
					"completions\n",
					timeout, completed);
				break;
			}
		}
		sched_yield();
	}

	if (topts.shared_cq) {
		ret = pthread_mutex_unlock(&shared_cq_lock);
		if (ret) {
			FT_PRINTERR("pthread_mutex_timedlock", ret);
			abort();
		}
	}
	return completed;
}

static void *run_sender_worker(void *arg)
{
	struct worker_context *ctx = (struct worker_context*) arg;
	struct ep_message msg;
	int ret;
	struct random_data random_data;
	const uint64_t total_ops = ctx->num_peers * topts.msgs_per_sender;
	const uint64_t msg_per_ep_lifecyle = total_ops / topts.sender_ep_recycling;
	uint64_t ops_posted = 0, ops_completed = 0;
	int cycle = 0;
	uint16_t peer_idx = 0;
	uint64_t ops_total_in_this_cycle = 0, ops_posted_in_this_cycle = 0;
	uint64_t ops_completed_in_this_cycle = 0;
	char ep_addr[ctx->num_peers][MAX_EP_ADDR_LEN];
	size_t ops_posted_for_peer[ctx->num_peers];
	size_t transferred_bytes[ctx->num_peers];
	struct rma_info peer_rma[ctx->num_peers];
	fi_addr_t fi_addr[ctx->num_peers];

	ft_random_init_data(&random_data, topts.random_seed, ctx->worker_id);
	memset(ep_addr, 0, sizeof(ep_addr));
	memset(ops_posted_for_peer, 0, sizeof(ops_posted_for_peer));

	bool should_reset_cycle = true;
	while(ops_posted < total_ops) {
		if (should_reset_cycle) {
			// Start a new endpoint cycle
			should_reset_cycle = false;
			ops_posted_in_this_cycle = 0;
			ops_completed_in_this_cycle = 0;
			if (ops_posted + msg_per_ep_lifecyle > total_ops - msg_per_ep_lifecyle)
				ops_total_in_this_cycle = total_ops - ops_posted;
			else
				ops_total_in_this_cycle = msg_per_ep_lifecyle;

			cycle++;
			printf("Sender %u: Starting EP cycle %d/%d\n", ctx->worker_id,
				cycle, topts.sender_ep_recycling);

			// Setup new endpoint for this cycle
			cleanup_endpoint(ctx);
			ret = setup_endpoint(ctx, ops_total_in_this_cycle);
			if (ret) {
				fprintf(stderr, "Sender %u: endpoint setup failed\n",
					ctx->worker_id);
				goto out;
			}
			// Restore AV from cache
			for (int i = 0; i < ctx->num_peers; i++) {
				fi_addr[i] = FI_ADDR_UNSPEC;
				if (ep_addr[i][0] == 0)
					continue;
				ret = fi_av_insert(ctx->av, &ep_addr[i], 1, &fi_addr[i], 0, NULL);
				if (ret != 1) {
					FT_PRINTERR("fi_av_insert", ret);
					if (ret == 0)
						ret = -FI_EOTHER;
					goto out;
				}
			}
			// sleep random time up to 100ms to emulate the real workload
			int sleep_time = ft_random_sleep_ms(&random_data, 100);
			printf("Sender %u: Sleeping for %d microseconds\n",
				ctx->worker_id, sleep_time);
		}

		// Apply all pending AV updates
		while (true) {
			ret = ep_message_queue_try_pop(ctx->control_queue, &msg);
			if (ret == 0) {
				if (msg.type == EP_MESSAGE_TYPE_TERMINATOR) {
					printf("Sender %u: recevied terminator message before completing all ops. "
							"Current cycle: %d, total ops: %lu\n",
							ctx->worker_id, cycle, total_ops);
					goto out;
				}
				assert(msg.type == EP_MESSAGE_TYPE_UPDATE);
				assert(msg.info.worker_id == ctx->worker_id);
				printf("Sender %u: received an update for peer %u\n",
						ctx->worker_id, msg.info.peer_idx);
				if (topts.remove_av
					&& fi_addr[msg.info.peer_idx] != FI_ADDR_UNSPEC) {
					ret = fi_av_remove(ctx->av, &fi_addr[msg.info.peer_idx], 1, 0);
					if (ret) {
						FT_PRINTERR("fi_av_remove", ret);
						goto out;
					}
				}
				memcpy(ep_addr[msg.info.peer_idx], msg.info.ep_addr, MAX_EP_ADDR_LEN);
				memcpy(&peer_rma[msg.info.peer_idx], &msg.info.rma, sizeof(struct rma_info));
				fi_addr[msg.info.peer_idx] = FI_ADDR_UNSPEC;
				ret = fi_av_insert(ctx->av,
							&ep_addr[msg.info.peer_idx],
							1,
							&fi_addr[msg.info.peer_idx],
							0,
							NULL);
				if (ret != 1) {
					FT_PRINTERR("fi_av_insert", ret);
					if (ret >= 0)
						ret = -FI_EOTHER;
					goto out;
				}
				transferred_bytes[msg.info.peer_idx] = 0;
			} else if (ret != -FI_EAGAIN) {
				FT_PRINTERR("ep_message_queue_try_pop", ret);
				goto out;
			} else {
				ret = 0;
				break;
			}
		}

		// Skip not ready yet and completed already peers
		if (fi_addr[peer_idx] == FI_ADDR_UNSPEC
			|| ops_posted_for_peer[peer_idx] == topts.msgs_per_sender) {
			if (++peer_idx == ctx->num_peers)
				peer_idx = 0;
			// Relinquish current CPU core to break busy-wait loop
			// on the current thread. Main thread will be scheduled
			// before current thread, therefore it would likely push
			// an update to worker's control queue if the one was pending.
			sched_yield();
			continue;
		}

		// Post one operation
		struct fi_context2 *fi_ctx;
		void *buffer;
		ret = context_pool_alloc_ctx(&ctx->pool, &buffer, &fi_ctx);
		if (ret) {
			FT_PRINTERR("context_pool_alloc_ctx", ret);
			goto out;
		}
		do {
			switch (topts.op_type) {
			case OP_MSG_UNTAGGED:
				ret = fi_send(ctx->ep,
						buffer,
						opts.transfer_size,
						fi_mr_desc(ctx->pool.mr),
						fi_addr[peer_idx],
						fi_ctx);
				break;
			case OP_MSG_TAGGED:
				ret = fi_tsend(ctx->ep,
						buffer,
						opts.transfer_size,
						fi_mr_desc(ctx->pool.mr),
						fi_addr[peer_idx],
						ft_tag,
						fi_ctx);
				break;
			case OP_RMA_WRITEDATA:
				ret = fi_writedata(ctx->ep,
							buffer,
							opts.transfer_size,
							fi_mr_desc(ctx->pool.mr),
							0xCAFE, // immediate data
							fi_addr[peer_idx],
							peer_rma[peer_idx].remote_addr + transferred_bytes[peer_idx],
							peer_rma[peer_idx].rkey,
							fi_ctx);
				break;
			default:
				fprintf(stderr,
					"Sender %u: unsupported operation: %d\n",
					ctx->worker_id, topts.op_type);
				abort();
			}
			if (ret == 0) {
				break;
			} else if (ret == -FI_EAGAIN) {
				if (wait_for_comp(ctx->cq, 1) == 0) {
					fprintf(stderr,
						"Sender %d: operation has stuck\n",
						ctx->worker_id);
					goto out;
				}
				ops_completed++;
				ops_completed_in_this_cycle++;
			} else {
				fprintf(stderr,
					"Sender %d: operation failed. ret = %d, %s\n",
					ctx->worker_id, ret, fi_strerror(-ret));
					goto out;
			}
		 } while(ret == -FI_EAGAIN);

		// Increment loop counters
		ops_posted++;
		ops_posted_in_this_cycle++;
		ops_posted_for_peer[peer_idx]++;
		transferred_bytes[peer_idx] += opts.transfer_size;
		if (++peer_idx == ctx->num_peers)
			peer_idx = 0;

		 if (ops_posted_in_this_cycle == ops_total_in_this_cycle) {
			should_reset_cycle = true;
			// Maybe wait for all operations to complete
			if (ft_random_get_bool(&random_data)) {
				printf("Sender %u EP cycle %d: Waiting for "
					"completions\n",
					ctx->worker_id, cycle);
				uint64_t ops_pending =  ops_total_in_this_cycle - ops_completed_in_this_cycle;
				ops_completed += wait_for_comp(ctx->cq, ops_pending);
			} else {
				printf("Sender %u EP cycle %d: Not waiting for "
					"completions\n",
					ctx->worker_id, cycle);
			}

			if (topts.verbose) {
				printf("Sender %u: Completed cycle %d"
					", ops posted:%" PRIu64
					", ops completed:%" PRIu64
					"\n\n",
					ctx->worker_id, cycle,
					ops_posted, ops_completed);
			}
		}
	}

	printf("Sender %d: All cycles completed, total ops: %lu\n",
	       ctx->worker_id, total_ops);

	// Some updates might have arrived too late to be applied.
	// Draining such messages from control queue until terminator.
	do {
		ret = ep_message_queue_pop(ctx->control_queue, &msg); // blocks
		if (ret) {
			FT_PRINTERR("ep_message_queue_pop", ret);
			goto out;
		}
	} while (msg.type != EP_MESSAGE_TYPE_TERMINATOR);

out:
	if (ret) {
		int *retval = malloc(sizeof(int));
		*retval = ret;
		return (void *) retval;
	}
	return NULL;
}

static int notify_endpoint_update(struct worker_context *ctx, void *rma_buffer)
{
	struct ep_message msg = { .type = EP_MESSAGE_TYPE_UPDATE };

	// Get endpoint address
	size_t addr_len = MAX_EP_ADDR_LEN;
	int ret = fi_getname(&ctx->ep->fid, msg.info.ep_addr, &addr_len);
	if (ret)
		return ret;

	// Fill RMA info
	msg.info.rma.remote_addr = (uint64_t) rma_buffer;
	msg.info.rma.rkey = fi_mr_key(ctx->pool.mr);

	// Send to all connected senders
	for (size_t i = 0; i < ctx->num_peers; i++) {
		msg.info.worker_id = ctx->peer_ids[i];
		msg.info.peer_idx = ctx->worker_id / topts.num_sender_workers;
		ret = ep_message_queue_push(ctx->control_queue, &msg);
		if (ret)
			return ret;
		if (topts.verbose)
			printf("Receiver %d: Notified sender %d about new EP\n",
			       ctx->worker_id, msg.info.worker_id);
	}
	return 0;
}

static void *run_receiver_worker(void *arg)
{
	struct worker_context *ctx = (struct worker_context *) arg;
	int ret = 0;
	struct random_data random_data;
	const uint64_t total_ops = ctx->num_peers * topts.msgs_per_sender;
	const uint64_t msg_per_ep_lifecyle = total_ops / topts.receiver_ep_recycling;

	ft_random_init_data(&random_data, topts.random_seed, ctx->worker_id);

	uint64_t ops_posted = 0, ops_completed = 0;
	int cycle = 0;
	uint64_t ops_total_in_this_cycle = 0, ops_posted_in_this_cycle = 0;
	while(ops_posted < total_ops) {
		if (ops_posted_in_this_cycle == ops_total_in_this_cycle) {
			// Start a new endpoint cycle
			ops_posted_in_this_cycle = 0;
			if (ops_posted + msg_per_ep_lifecyle > total_ops - msg_per_ep_lifecyle)
				ops_total_in_this_cycle = total_ops - ops_posted;
			else
				ops_total_in_this_cycle = msg_per_ep_lifecyle;
			cycle++;
			printf("Receiver %u: Starting EP cycle %d/%d\n",
				ctx->worker_id, cycle, topts.receiver_ep_recycling);

			// Setup new endpoint for this cycle
			cleanup_endpoint(ctx);
			ret = setup_endpoint(ctx, ops_total_in_this_cycle);
			if (ret) {
				fprintf(stderr, "Receiver %u: endpoint setup failed\n",
						ctx->worker_id);
				goto out;
			}

			// Notify sender of new endpoint
			void *rma_buffer = NULL;
			if (topts.op_type == OP_RMA_WRITEDATA) {
				ret = context_pool_alloc_buffers(&ctx->pool,
						ops_total_in_this_cycle,
						&rma_buffer);
				if (ret) {
					FT_PRINTERR("context_pool_alloc_ctx", ret);
					goto out;
				}
			}
			ret = notify_endpoint_update(ctx, rma_buffer);
			if (ret < 0) {
				fprintf(stderr,
					"Receiver %u: Failed to notify endpoint update "
					"for cycle %d: %d\n",
					ctx->worker_id, cycle, ret);
				goto out;
			}

			// sleep random time up to 100ms to emulate the real workload
			int sleep_time = ft_random_sleep_ms(&random_data, 100);
			printf("Receiver %u: Sleeping for %d microseconds\n",
					ctx->worker_id, sleep_time);
		}

		if (topts.op_type != OP_RMA_WRITEDATA) {
			// Post one operation
			struct fi_context2 *fi_ctx;
			void *buffer;
			ret = context_pool_alloc_ctx(&ctx->pool, &buffer, &fi_ctx);
			if (ret) {
				FT_PRINTERR("context_pool_alloc_ctx", ret);
				goto out;
			}

			switch (topts.op_type) {
			case OP_MSG_UNTAGGED:
				ret = fi_recv(ctx->ep,
						buffer,
						opts.transfer_size,
						fi_mr_desc(ctx->pool.mr),
						0,
						fi_ctx);
				break;
			case OP_MSG_TAGGED:
				ret = fi_trecv(ctx->ep,
						buffer,
						opts.transfer_size,
						fi_mr_desc(ctx->pool.mr),
						0,
						ft_tag,
						0,
						fi_ctx);
				break;
			default:
				fprintf(stderr,
					"Receiver %u: unsupported operation: %d\n",
					ctx->worker_id, topts.op_type);
				abort();
			}

			if (ret) {
				fprintf(stderr,
					"Receiver %u fi_recv failed "
					"op # %" PRIu64 ": %s\n",
					ctx->worker_id,  ops_posted,
					fi_strerror(-ret));
				goto out;
			}
		}

		// Increment loop counters
		ops_posted++;
		ops_posted_in_this_cycle++;

		if (ops_posted_in_this_cycle == ops_total_in_this_cycle) {
			if (ft_random_get_bool(&random_data)) {
				printf("Receiver %u EP cycle %d: Waiting for "
					"completions\n",
					ctx->worker_id, cycle);
				ops_completed += wait_for_comp(ctx->cq, ops_total_in_this_cycle);
			} else {
				printf("Receiver %u EP cycle %d: Not waiting for "
					"completions\n",
					ctx->worker_id, cycle);
			}
			printf("Receiver %u EP cycle %d: Posted %" PRIu64
				" and completed %" PRIu64 " receives from %zu senders\n\n",
				ctx->worker_id, cycle, ops_posted, ops_completed, ctx->num_peers);
		}
	}
out:
	printf("Receiver %d: Completed %d EP cycles\n", ctx->worker_id, cycle);
	const struct ep_message terminator = { .type = EP_MESSAGE_TYPE_TERMINATOR };
	ep_message_queue_push(ctx->control_queue, &terminator);

	if (ret) {
		int *retval = malloc(sizeof(int));
		*retval = ret;
		return (void *) retval;
	}
	return NULL;
}

static void cleanup_worker_resourses(struct worker_context *worker) {
	context_pool_destroy(&worker->pool);
	if (worker->peer_ids) {
		free(worker->peer_ids);
		worker->peer_ids = NULL;
		worker->num_peers = 0;
	}
	cleanup_endpoint(worker);
}

// Common function for buffer and MR setup
static int setup_worker_resources(struct worker_context* worker, uint64_t access, size_t pool_capacity, size_t buffer_size)
{
	int ret;
	// Allocate memory pool
	ret = context_pool_init(&worker->pool, pool_capacity, buffer_size, access);
	if (ret) {
		FT_PRINTERR("context_pool_init", ret);
		goto error;
	}

	return 0;
error:
	cleanup_worker_resourses(worker);
	return ret;
}

static int setup_shared_resources(size_t num_workers, size_t max_peers)
{
	if (topts.shared_cq) {
		cq_attr.format = FI_CQ_FORMAT_CONTEXT;
		cq_attr.size = fi->tx_attr->size;
		int ret = fi_cq_open(domain, &cq_attr, &shared_cq, NULL);
		if (ret) {
			FT_PRINTERR("fi_cq_open", ret);
			return ret;
		}
	}

	if (topts.shared_av) {
		av_attr.type = FI_AV_TABLE;
		av_attr.count = max_peers * num_workers;
		int ret = fi_av_open(domain, &av_attr, &shared_av, NULL);
		if (ret) {
			FT_PRINTERR("fi_av_open", ret);
			return ret;
		}
	}

	return 0;
}

static void cleanup_shared_resources(void)
{
	if (shared_av) {
		fi_close(&shared_av->fid);
	}
	if (shared_cq) {
		fi_close(&shared_cq->fid);
	}
}

static int run_sender(void)
{
	int ret;
	struct worker_context *workers;
	pthread_t *threads;
	struct ep_message_queue *channels;

	workers = calloc(topts.num_sender_workers, sizeof(*workers));
	threads = calloc(topts.num_sender_workers, sizeof(*threads));
	channels = calloc(topts.num_sender_workers, sizeof(*channels));
	if (!workers || !threads || !channels) {
		ret = -FI_ENOMEM;
		goto out;
	}

	printf("\nSender Worker Distribution:\n");
	printf("-------------------------\n");
	printf("Total: %d senders, %d receivers\n", topts.num_sender_workers,
	       topts.num_receiver_workers);

	ret = setup_shared_resources(topts.num_sender_workers,
					topts.num_receiver_workers / topts.num_sender_workers + 1);
	if (ret)
		goto out;

	for (int i = 0; i < topts.num_sender_workers; i++) {
		struct worker_context *worker = &workers[i];

		ret = ep_message_queue_init(&channels[i]);
		if (ret)
			return ret;

		worker->worker_id = i;
		worker->control_queue = &channels[i];

		ret = calculate_worker_distribution(i,
						topts.num_sender_workers,
						topts.num_receiver_workers,
						&worker->num_peers,
						&worker->peer_ids);
		if (ret)
			goto out;
		// Setup common resources
		ret = setup_worker_resources(worker, FI_SEND | FI_WRITE,
						topts.msgs_per_sender * worker->num_peers,
						opts.transfer_size);
		if (ret) {
			fprintf(stderr,
				"setup_worker_resources failed for sender %d: "
				"%d\n",
				i, ret);
			goto out;
		}

		if (topts.verbose) {
			printf("\nSender Worker %d:\n", i);
			printf("  - Assigned receivers: ");
			for (size_t j = 0; j < worker->num_peers; j++) {
				printf("%d ", worker->peer_ids[j]);
			}
			printf("\n");
		}
	}

	// Create worker threads
	for (int i = 0; i < topts.num_sender_workers; i++) {
		ret = pthread_create(&threads[i], NULL, run_sender_worker, &workers[i]);
		if (ret) {
			printf("Failed to create sender thread: %d\n", ret);
			goto out;
		}
	}

	// Dispatch control messages from OOB channel to worker's control channels
	struct ep_message msg;
	while (true) {
		// ft_sock_recv blocks until the complete message recived.
		// It returns 0 on suceess and -FI_ENOTCONN if socket
		// had been closed by peer.
		ret = ft_sock_recv(oob_sock, (void*)&msg, sizeof(msg));
		if (ret) {
			FT_PRINTERR("ft_sock_recv", ret);
			goto out;
		}
		printf("OOB Message: type: %d, worker_id: %u, peer_idx: %u\n",
				msg.type, msg.info.worker_id, msg.info.peer_idx);
		if (msg.type == EP_MESSAGE_TYPE_TERMINATOR)
			break;
		ret = ep_message_queue_push(workers[msg.info.worker_id].control_queue, &msg);
		if (ret) {
			FT_PRINTERR("ep_message_queue_push", ret);
			goto out;
		}
	}
	// On the happy path we exit this loop because of terminator message
	// recived on OOB channnel. That means all reciver's workers has
	// completed all cycles and no one is listening anymore.

	// Terminate workers
	for (int i = 0; i < topts.num_sender_workers; i++) {
		assert(msg.type == EP_MESSAGE_TYPE_TERMINATOR);
		ret = ep_message_queue_push(workers[i].control_queue, &msg);
		if (ret) {
			FT_PRINTERR("ep_message_queue_push", ret);
			goto out;
		}
	}

	// Wait for completion
	for (int i = 0; i < topts.num_sender_workers; i++) {
		void *retval;
		ret = pthread_join(threads[i], &retval);
		if (ret) {
			FT_PRINTERR("pthread_join", ret);
			goto out;
		}
		if (retval != NULL) {
			ret = *(int *)retval;
			free(retval);
			fprintf(stderr,
				"Sender %d failed. Exit code: %d, %s\n",
				i, ret, fi_strerror(-ret));
			goto out;
		}
	}

out:
	if (channels) {
		for (int i = 0; i < topts.num_sender_workers; i++) {
		       ep_message_queue_destroy(&channels[i]);
		}
		free(channels);
	}
	if (threads)
		free(threads);
	if (workers) {
		for (int i = 0; i < topts.num_sender_workers; i++) {
			cleanup_worker_resourses(&workers[i]);
		}
		free(workers);
	}
	cleanup_shared_resources();
	return ret;
}

static int run_receiver(void)
{
	int ret;
	struct ep_message_queue control_queue;
	struct worker_context *workers;
	pthread_t *threads;

	ret = ep_message_queue_init(&control_queue);
	if (ret)
		return ret;

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

	ret = setup_shared_resources(topts.num_receiver_workers,
					topts.num_sender_workers / topts.num_receiver_workers + 1);
	if (ret)
		goto out;

	// Initialize workers
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		struct worker_context *worker = &workers[i];

		worker->worker_id = i;
		worker->control_queue = &control_queue;

		ret = calculate_worker_distribution(i,
						topts.num_receiver_workers,
						topts.num_sender_workers,
						&worker->num_peers,
						&worker->peer_ids);
		if (ret)
			goto out;

		// Setup common resources
		ret = setup_worker_resources(worker, FI_RECV | FI_REMOTE_WRITE,
				topts.msgs_per_sender * worker->num_peers,
				opts.transfer_size);
		if (ret) {
			fprintf(stderr,
				"setup_worker_resources failed for receiver %d: "
				"%d\n",
				i, ret);
			goto out;
		}

		if (topts.verbose) {
			printf("\nReceiver Worker %d:\n", i);
			printf("  - Connected by senders: ");
			for (size_t j = 0; j < worker->num_peers; j++)
				printf("%d ", worker->peer_ids[j]);
			printf("\n");
		}
	}

	// Create worker threads
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		ret = pthread_create(&threads[i], NULL, run_receiver_worker, &workers[i]);
		if (ret) {
			printf("Failed to create receiver thread: %d\n", ret);
			goto out;
		}
	}

	// Forward control messages from workers to OOB channel
	size_t completed_workers = 0;
	while(completed_workers < topts.num_receiver_workers) {
		struct ep_message msg;
		ret = ep_message_queue_pop(&control_queue, &msg); // blocks
		if (ret) {
			FT_PRINTERR("ep_message_queue_pop", ret);
			goto out;
		}
		if (msg.type == EP_MESSAGE_TYPE_TERMINATOR) {
			completed_workers++;
		} else {
			ret = ft_sock_send(oob_sock, (void*)&msg, sizeof(msg));
			if (ret) {
				FT_PRINTERR("ft_sock_send", ret);
				goto out;
			}
		}
	}
	struct ep_message terminator = { .type = EP_MESSAGE_TYPE_TERMINATOR };
	ret = ft_sock_send(oob_sock, &terminator, sizeof(terminator));
	if (ret) {
		FT_PRINTERR("ft_sock_send", ret);
		goto out;
	}

	// Wait for thread completion
	for (int i = 0; i < topts.num_receiver_workers; i++) {
		void *retval;
		ret = pthread_join(threads[i], &retval);
		if (ret) {
			FT_PRINTERR("pthread_join", ret);
			goto out;
		}
		if (retval != NULL) {
			ret = *(int *)retval;
			free(retval);
			fprintf(stderr,
				"Receiver %d failed. Exit code: %d, %s\n",
				i, ret, fi_strerror(-ret));
			goto out;
		}
	}

out:
	if (threads)
		free(threads);
	if (workers) {
		for (int i = 0; i < topts.num_receiver_workers; i++) {
			cleanup_worker_resourses(&workers[i]);
		}
		free(workers);
	}
	cleanup_shared_resources();
	ep_message_queue_destroy(&control_queue);
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
			topts.msgs_per_sender = atoi(optarg);
			if (topts.msgs_per_sender < 1) {
				fprintf(stderr, "number of messages per sender must be "
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
		case OPT_REMOVE_AV:
			topts.remove_av = true;
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

	if (optind < argc)
		opts.dst_addr = argv[optind];

	ret = ft_init_oob();
	if(ret)	{
		FT_PRINTERR("ft_init_oob", ret);
		goto out;
	}
	ret = ft_sync_oob();
	if(ret)	{
		FT_PRINTERR("ft_sync_oob", ret);
		goto out;
	}
	ret = run_test();
out:
	ft_sync_oob();
	ft_close_oob();
	ft_free_res();
	return ft_exit_code(ret);
}
