
#include <infiniband/efadv.h>
#include <infiniband/verbs.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>

/* Directions For Creating New Mock Function:
 * 1. Recreate the funciton signature with __wrap_<function>(the_params), and the function
 *    to the Makefile.include prov_efa_test_efa_unit_test_LDFLAGS list
 * 2. Check all parameters with check_expected()
 *   a. This allows test code to optionally check the parameters of the mocked function
 *      with the family of expect_value() functions.
 * 3. Make sure to return a casted mock()
 *   a. This gives the test code control of the return value of the mocked function,
 *      by calling the will_return() function.  The will_return() function creates a
 *      stack for each mocked function and returns the top of the stack first.
*/

struct ibv_ah *__wrap_ibv_create_ah(struct ibv_pd *pd, struct ibv_ah_attr *attr) {
    check_expected(pd);
    check_expected(attr);
    return (struct ibv_ah*) mock();
}

int __wrap_efadv_query_ah(struct ibv_ah *ibvah, struct efadv_ah_attr *attr, uint32_t inlen) {
    check_expected(ibvah);
    check_expected(attr);
    check_expected(inlen);
    return (int) mock();
}
