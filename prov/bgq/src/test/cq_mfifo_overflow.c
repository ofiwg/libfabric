
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

	unsigned production_count = 0;
	while (0 == memfifo_produce16(&mfifo.producer, production_count)) production_count++;
	if (production_count != CQ_MFIFO_SIZE) { lineno = __LINE__; goto err; }

	uint16_t entry_id;
	unsigned consumption_count = 0;
	while (0 == memfifo_consume16(&mfifo.consumer, &entry_id)) {
		if (entry_id != consumption_count++) { lineno = __LINE__; goto err; }
	}
	if (consumption_count != CQ_MFIFO_SIZE) { lineno = __LINE__; goto err; }

	while (0 == memfifo_produce16(&mfifo.producer, production_count)) production_count++;
	if (production_count != (CQ_MFIFO_SIZE*2)) { lineno = __LINE__; goto err; }

	rc = memfifo_consume16(&mfifo.consumer, &entry_id); lineno = __LINE__;
	if (rc) goto err;
	if (entry_id != (0x7FFF & consumption_count)) { lineno = __LINE__; goto err; }
	consumption_count++;

	rc = memfifo_produce16(&mfifo.producer, production_count++); lineno = __LINE__;
	if (rc) goto err;

	rc = memfifo_consume16(&mfifo.consumer, &entry_id); lineno = __LINE__;
	if (rc) goto err;
	if (entry_id != (0x7FFF & consumption_count)) { lineno = __LINE__; goto err; }
	consumption_count++;


	fprintf (stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf (stderr, "%s: Error at line %d\n", __FILE__, lineno);
	return 1;
}








