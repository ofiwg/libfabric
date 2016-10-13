
#include <pthread.h>

#include "fi_bgq_spi.h"


#include "cq_agent.h"
#include "l2atomic.h"
#include "fi_bgq_memfifo.h"


int cq_agent_main_test (struct l2atomic_barrier * barrier);

void test_init_fn (void *buffer, uintptr_t cookie) {

	uint64_t *ptr = (uint64_t *) buffer;
	*ptr = cookie;
}

int main (int argc, char *argv[]) {

	struct l2atomic l2atomic;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	int rc, lineno;
	rc = l2atomic_init(&l2atomic); lineno = __LINE__;
	if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	uint32_t ppn = Kernel_ProcessCount(); lineno = __LINE__;
	if (ppn==1) {
		/* check for ofi agent environment variable */
		char * envvar = NULL;
		envvar = getenv("BG_APPAGENT"); lineno = __LINE__;
		if (!envvar) { fprintf(stderr, "Required environment variable 'BG_APPAGENT' is not set\n"); goto err; }
	}

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	struct l2atomic_barrier barrier;
	rc = l2atomic_barrier_alloc_generic(&l2atomic, &barrier, 2, "agent_barrier"); lineno = __LINE__;
	if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
	uint32_t tcoord = Kernel_MyTcoord();
	if (tcoord==1) {

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		//struct cq_agent_internal internal;
		//rc = cq_agent_init(&internal); lineno = __LINE__;
		rc = cq_agent_main_test(&barrier); lineno = __LINE__;
		if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		//l2atomic_barrier(&barrier);

		//while (0 == cq_agent_poll(&internal, 0));
		//l2atomic_barrier(&barrier);

	} else if (tcoord==0) {
//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		struct cq_agent_client client;
		rc = cq_agent_client_init(&client, &l2atomic); lineno = __LINE__;
		if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		union fi_bgq_addr self;
		fi_bgq_create_addr_self_cx(&self.fi, 0);

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		struct memfifo mfifo;
		MUHWI_Descriptor_t model;
		rc = cq_agent_client_register(&client, &l2atomic, &self, &mfifo, 8192, &model, 1); lineno = __LINE__;
		if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		struct cq_agent_client_test_mu test;
		rc = cq_agent_client_test_mu_setup(&test); lineno = __LINE__;
		if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		uint16_t entry_id = 1234;
		rc = cq_agent_client_test_mu_inject(&test, &model, entry_id, 1); lineno = __LINE__;
		if (rc) goto err;

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		if (ppn>1) l2atomic_barrier(&barrier);

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		/* spin until something is received from the mfifo */
		uint16_t id = (uint16_t)-1;
		while (0 != memfifo_consume16(&mfifo.consumer, &id));

//fprintf(stderr, "%s:%d\n", __FILE__, __LINE__);
		if (ppn>1) l2atomic_barrier(&barrier);
		fprintf (stdout, "TEST SUCCESSFUL\n");
	}

	return 0;
err:
	fprintf(stderr, "%s : Error at line %d (rc=%d)\n", __FILE__, lineno, rc);
	abort();
}








