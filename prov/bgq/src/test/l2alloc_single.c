
#include "fi_bgq_spi.h"

#include "l2atomic.h"


void test_init_fn (void * buffer, uintptr_t cookie) {

	uintptr_t * ptr = (uintptr_t *) buffer;
	*ptr = cookie;
}

int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;
	memset((void*)&l2atomic, 0, sizeof(l2atomic));

	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

	uintptr_t tcoord = Kernel_MyTcoord();

	uintptr_t * buffer = NULL;
	rc = l2atomic_alloc(&l2atomic, "simple", 128, (void**)&buffer, test_init_fn, tcoord); lineno = __LINE__;
	if (rc) goto err;
	if (!buffer) { lineno = __LINE__; goto err; }

	fprintf (stdout, "l2atomic value: %lu\n", *buffer);	

	fprintf (stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf (stderr, "Error at line %d\n", lineno);
	return 1;
}








