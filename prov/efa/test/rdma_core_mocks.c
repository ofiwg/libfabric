#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
    check_expected(pd);
    check_expected(attr);
    return (struct ibv_ah*) mock();
}

int __real_ibv_destroy_ah(struct ibv_ah *ibv_ah);

int __wrap_ibv_destroy_ah(struct ibv_ah *ibv_ah)
{
	int val = mock();
	if (val == 4242) {
		return __real_ibv_destroy_ah(ibv_ah);
	}
	return val;
}

int __wrap_efadv_query_ah(struct ibv_ah *ibvah, struct efadv_ah_attr *attr, uint32_t inlen) {
    check_expected(ibvah);
    check_expected(attr);
    check_expected(inlen);
    return (int) mock();
}

int __real_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			     uint32_t inlen);

int __wrap_efadv_query_device(struct ibv_context *ibvctx, struct efadv_device_attr *attr,
			     uint32_t inlen)
{
	int retval;

	retval = mock();
	/* Expected return value being 0 means we want this function to work as expected
	 * hence call the real function in this case
	 */
	return (retval == 0) ? __real_efadv_query_device(ibvctx, attr, inlen) : retval;
}
