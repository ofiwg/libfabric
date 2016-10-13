#ifndef _FI_PROV_BGQ_NODE_H_
#define _FI_PROV_BGQ_NODE_H_

#include "rdma/bgq/fi_bgq_l2atomic.h"

#include "rdma/bgq/fi_bgq_spi.h"

#define FI_BGQ_NODE_NUM_USER_SUBGROUPS (BGQ_MU_NUM_FIFO_SUBGROUPS_PER_NODE-2)					/* subgroups 66 and 67 are privileged */
#define FI_BGQ_NODE_BAT_SIZE (FI_BGQ_NODE_NUM_USER_SUBGROUPS * BGQ_MU_NUM_DATA_COUNTERS_PER_SUBGROUP)
#define FI_BGQ_NODE_APPLICATION_BAT_SIZE ((BGQ_MU_NUM_FIFO_GROUPS-1) * BGQ_MU_NUM_DATA_COUNTERS_PER_GROUP)	/* cnk and agents use group 16 */

struct fi_bgq_node {
	void *shm_ptr;
	void *abs_ptr;
	struct {
		struct l2atomic_counter	allocator;
	} counter;
	struct {
		struct l2atomic_counter	allocator;
	} lock;
	struct l2atomic_barrier		barrier;
	uint32_t			leader_tcoord;
	uint32_t			is_leader;
	struct {
		volatile uint64_t			*shadow;	/* in shared memory */
		volatile uint64_t			l2_cntr_paddr[FI_BGQ_NODE_APPLICATION_BAT_SIZE];
		MUSPI_BaseAddressTableSubGroup_t	subgroup[FI_BGQ_NODE_BAT_SIZE];
	} bat;
};

int fi_bgq_node_init (struct fi_bgq_node * node);

int fi_bgq_node_mu_lock_init (struct fi_bgq_node * node, struct l2atomic_lock * lock);

int fi_bgq_node_counter_allocate (struct fi_bgq_node * node, struct l2atomic_counter * counter);

int fi_bgq_node_lock_allocate (struct fi_bgq_node * node, struct l2atomic_lock * lock);

uint64_t fi_bgq_node_bat_allocate (struct fi_bgq_node * node, struct l2atomic_lock * lock);

void fi_bgq_node_bat_free (struct fi_bgq_node * node, struct l2atomic_lock * lock, uint64_t index);

void fi_bgq_node_bat_write (struct fi_bgq_node * node, struct l2atomic_lock * lock, uint64_t index, uint64_t offset);

void fi_bgq_node_bat_clear (struct fi_bgq_node * node, struct l2atomic_lock * lock, uint64_t index);

static inline
uint64_t fi_bgq_node_bat_read (struct fi_bgq_node * node, uint64_t index) {

	assert(index < FI_BGQ_NODE_BAT_SIZE);
	return node->bat.shadow[index];
}



#endif /* _FI_PROV_BGQ_NODE_H_ */

