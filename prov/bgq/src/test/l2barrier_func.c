
#include <unistd.h>

#include "rdma/bgq/fi_bgq_hwi.h"
#include "rdma/bgq/fi_bgq_spi.h"

#include "l2atomic.h"

#define ITERATIONS 1000000


void test_init_fn (void * buffer, uintptr_t cookie) {
	uint64_t * ptr = (uint64_t *) buffer;
	*ptr = cookie;
}

int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;
	memset((void*)&l2atomic, 0, sizeof(l2atomic));

	uint32_t tcoord = Kernel_MyTcoord();
	//uint32_t ppn = Kernel_ProcessCount();

	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

	/* race condition! how to determine the number of *active* ranks on the node
	 * without using Kernel_RanksToCoords() ? */
	usleep(5000);
	ppc_msync();
	int participants = l2atomic.shared->counter;
	fprintf(stderr, "%s:%d participants=%d\n", __FILE__, __LINE__, participants);
	/* end: race */

	struct l2atomic_barrier barrier;
	rc = l2atomic_barrier_alloc_generic(&l2atomic, &barrier, participants, "barrier_test"); lineno = __LINE__;
	if (rc) goto err; 
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	uint64_t start_time = GetTimeBase();
	l2atomic_barrier(&barrier);
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	if (tcoord == 0) usleep(1);
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	l2atomic_barrier(&barrier);
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	uint64_t end_time = GetTimeBase();

	fprintf(stdout, "barrier cycles: %lu\n", end_time - start_time);

	if (tcoord==0) fprintf(stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf(stderr, "Error at line %d\n", lineno);
	return 1;
}








