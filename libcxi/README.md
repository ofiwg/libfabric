libcxi - The CXI library
=============================

The CXI library provides interfaces which interact directly with CXI drivers.
CXI drivers support communication over the Cray Cassini NIC.

How to build
------------

The libcxi project uses autotools.  To build, perform the following procedure:

```
./autogen.sh
./configure
make
```

How to test
-----------

The libcxi project contains unit tests written for the Criterion testing
framework (https://github.com/Snaipe/Criterion).  In order to build tests along
with the libcxi library, enable Criterion during configure:

```
./autogen.sh
./configure --with-criterion=<criterion_path>
make
```

To run the unit tests:

```
./tests/run_tests_vm.sh
```

To run the unit tests and remain in the VM:

```
./tests/run_tests_vm.sh -n
```

The Address Sanitizer (supported by gcc and clang) can also be enabled
by adding "--enable-asan" to the configure command line.

API Versioning
----------
The "API" / Overall Project revision will utilize Semantic versioning.

Given a version number MAJOR.MINOR.PATCH, increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards compatible manner, and
- PATCH version when you make backwards compatible bug fixes.

These changes should be made in configure.ac.
See: API Versioning - Semantic

Further details can be found here:
https://semver.org/

Ensure NEWS is updated when the libcxi API version is updated.

ABI Versioning
--------------
For the "ABI" libtool versioning is used.

Libtool versions follow the format of Current:Revision:Age and they *do not* map to the project revision.
'Current' and 'Age' together define a range of interface numbers that are compatible (i.e. does not require recompilation of any consumer libraries or executables.).

Official Rules from GNU
- If the library source code has changed at all since the last update, then increment revision (‘c:r:a’ becomes ‘c:r+1:a’).
- If any interfaces have been added, removed, or changed since the last update, increment current, and set revision to 0.
- If any interfaces have been added since the last public release, then increment age.
- If any interfaces have been removed or changed since the last public release, then set age to 0.

As a rule of thumb, Age should only be incremented if the binary interface is backwards compatible.

These changes should be made in configure.ac.
See: ABI Versioning - Libtool

For more details on libtool versioning see:
https://www.gnu.org/software/libtool/manual/html_node/Libtool-versioning.html#Libtool-versioning
https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html

Still confusing? This has further helpful clarifications:
https://autotools.io/libtool/version.html

Submitting a patch
------------------

This project follows the linux coding style. A git pre-commit hook can
be installed by running

    ./contrib/install-git-hook.sh

It will run checkpatch for every commit. If the commit has warnings or
errors, the commit can't happen. That check can be disabled by adding
the --no-verify option to git commit, or by setting environment
variable GIT\_HOOKS\_PRE\_COMMIT\_HPE\_CRAY\_CHECKPATCH\_SKIP to a
non-zero numeric value.
