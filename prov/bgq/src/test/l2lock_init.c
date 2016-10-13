


#include "l2atomic.h"



int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;
	memset((void*)&l2atomic, 0, sizeof(l2atomic));

	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

	L2_Lock_t * lock;
	rc = l2atomic_lock_alloc_generic (&l2atomic, &lock, "lock"); lineno = __LINE__;
	if (rc) goto err;


	fprintf (stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf (stderr, "Error at line %d\n", lineno);
	return 1;
}








