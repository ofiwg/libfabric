


#include "l2atomic.h"




int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;
	memset((void*)&l2atomic, 0, sizeof(l2atomic));

	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

	void * buffer = NULL;
	rc = l2atomic_alloc(&l2atomic, "simple", 128, &buffer, NULL, 0); lineno = __LINE__;
	if (rc) goto err;


	fprintf (stdout, "TEST SUCCESSFUL\n");
	return 0;
err:
	fprintf (stderr, "Error at line %d\n", lineno);
	return 1;
}








