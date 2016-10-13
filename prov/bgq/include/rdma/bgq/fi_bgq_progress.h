#ifndef _FI_PROV_BGQ_PROGRESS_H_
#define _FI_PROV_BGQ_PROGRESS_H_


#include "rdma/bgq/fi_bgq_l2atomic.h"

#define MAX_ENDPOINTS	(128)	/* TODO - get this value from somewhere else */

struct fi_bgq_ep;
struct fi_bgq_domain;
union fi_bgq_context;

struct fi_bgq_progress {

	uint64_t			tag_ep_count;
	uint64_t			msg_ep_count;
	uint64_t			all_ep_count;
	volatile uint64_t		enabled;
	struct l2atomic_fifo_consumer	consumer;
	uint64_t			pad_0[8];

	/* == L2 CACHE LINE == */

	struct fi_bgq_ep		*tag_ep[MAX_ENDPOINTS];
	struct fi_bgq_ep		*msg_ep[MAX_ENDPOINTS];
	struct fi_bgq_ep		*all_ep[MAX_ENDPOINTS];

	/* == L2 CACHE LINE == */

	volatile uint64_t		active;
	struct l2atomic_fifo_producer	producer;
	struct fi_bgq_domain		*bgq_domain;
	pthread_t			pthread;
	uint64_t			pad_1[10];

} __attribute__((__aligned__(L2_CACHE_LINE_SIZE)));

int fi_bgq_progress_init (struct fi_bgq_domain *bgq_domain, const uint64_t max_threads);
int fi_bgq_progress_enable (struct fi_bgq_domain *bgq_domain, const unsigned id);
int fi_bgq_progress_disable (struct fi_bgq_domain *bgq_domain, const unsigned id);
int fi_bgq_progress_fini (struct fi_bgq_domain *bgq_domain);

int fi_bgq_progress_ep_enable (struct fi_bgq_progress *thread, struct fi_bgq_ep *bgq_ep);
int fi_bgq_progress_ep_disable (struct fi_bgq_ep *bgq_ep);

#endif /* _FI_PROV_BGQ_PROGRESS_H_ */
