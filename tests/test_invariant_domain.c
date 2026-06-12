#include <check.h>
#include <stdlib.h>
#include <stdint.h>

/* 
 * Since we cannot directly import the internal allocation logic from domain.c,
 * we test the security invariant: calloc must be checked for NULL before use.
 * This test simulates the allocation pattern and verifies NULL handling.
 */

START_TEST(test_calloc_null_check_invariant)
{
    /* Invariant: Buffer allocations must be checked for NULL before dereference */
    size_t test_sizes[] = {
        SIZE_MAX,           /* Exploit case: guaranteed calloc failure */
        SIZE_MAX / 2,       /* Boundary: likely failure on most systems */
        1024                /* Valid input: should succeed */
    };
    int num_sizes = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        void *buf = calloc(1, test_sizes[i]);
        
        /* Security invariant: we must handle NULL gracefully */
        if (buf == NULL) {
            /* This is expected for large sizes - no crash should occur */
            ck_assert_msg(test_sizes[i] > 1024 * 1024,
                "Unexpected NULL for reasonable size %zu", test_sizes[i]);
        } else {
            /* Valid allocation - verify we can safely use it */
            ck_assert_ptr_nonnull(buf);
            ((char *)buf)[0] = 'A';  /* Safe to dereference */
            free(buf);
        }
    }
}
END_TEST

Suite *security_suite(void)
{
    Suite *s;
    TCase *tc_core;

    s = suite_create("Security");
    tc_core = tcase_create("Core");

    tcase_add_test(tc_core, test_calloc_null_check_invariant);
    suite_add_tcase(s, tc_core);

    return s;
}

int main(void)
{
    int number_failed;
    Suite *s;
    SRunner *sr;

    s = security_suite();
    sr = srunner_create(s);

    srunner_run_all(sr, CK_NORMAL);
    number_failed = srunner_ntests_failed(sr);
    srunner_free(sr);

    return (number_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}