
#include "fi_bgq_spi.h"

#include "cq_agent.h"
#include "fi_bgq_memfifo.h"

#define ITERATIONS 10

void test_init_fn (void * buffer, uintptr_t cookie) {

	uint64_t * ptr = (uint64_t *) buffer;
	*ptr = cookie;
}

int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;
	memset((void*)&l2atomic, 0, sizeof(l2atomic));

	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

	struct memfifo mfifo;
	rc = memfifo_initialize(&l2atomic, "some name", &mfifo, 0); lineno = __LINE__;
	if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	unsigned i=0;
	for (i=0; i<ITERATIONS; ++i) {
		rc = memfifo_produce16(&mfifo.producer, 1000+i); lineno = __LINE__;
		if (rc) goto err;
	}
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	uint16_t entry_id;
	for (i=0; i<ITERATIONS; i++) {
//fprintf(stderr, "%s:%d (%d)\n", __FILE__, __LINE__, i);
		rc = memfifo_consume16(&mfifo.consumer, &entry_id); lineno = __LINE__;
		if (rc) goto err;
//fprintf(stderr, "%s:%d (%d) entry_id=%d\n", __FILE__, __LINE__, i, entry_id);
		if (entry_id != 1000+i) {lineno = __LINE__; goto err;}
	}

	/* should be no packets left */
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	rc = memfifo_consume16(&mfifo.consumer, &entry_id); lineno = __LINE__;
	if (!rc) goto err;

	fprintf (stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf (stderr, "Error at line %d\n", lineno);
	return 1;
}








