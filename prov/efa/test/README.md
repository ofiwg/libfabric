# EFA unit tests

## How to run

To run efa unit tests, you will need to have cmocka installed.
Cmocka Mirror: https://cmocka.org/files/
Install Instructions: https://gitlab.com/cmocka/cmocka/-/blob/master/INSTALL.md

You will need to configure libfabric with ```--enable-efa-unit-test=<path_to_cmocka_install>```.

An example build and run command would look like:

```
./autogen.sh; ./configure --enable-efa-unit-test=/home/ec2-user/cmocka/install; make check;
```

## How to write

1. Add unit tests to the bottom of the c source file you wish to test.
2. Add headers to efa_unit_tests.h for each new unit test.
3. Create or find an existing cmocka test group in efa_unit_tests.c
4. Add your new unit test to the cmocka test group.

## Mocking

To mock a function, you will need to add ```-Wl,--wrap=<function to mock>```
to the Makefile.include as part of efatest_LIBS. Then, after declaring the function,
you can replace it with __wrap_<function to mock>. If you need to use the original
function, you can use __real_<function to mock> after declaring it.

### Manipulating mock behavior

You can use the cmocka API to change the behavior of your mock function.

The will_return class of functions will place values onto a stack that can
be popped within the function with the mock function. You can use this to
manipulate the function behavior. For example, if you pop the value 'true'
using
```
will_return(mock_function, true)
```

and add a check in your mock function like so:

```
int use_real_function = mock_type(bool);
if (use_real_function)
	return __real_mock_function(params);
```

you can use the real function whenever you wish to instead of your custom
behavior.

See: https://api.cmocka.org/group__cmocka__mock.html for more details
