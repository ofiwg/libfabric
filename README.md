[![Build Status](https://travis-ci.org/ofiwg/libfabric.svg?branch=master)](https://travis-ci.org/ofiwg/libfabric)

# fabtests

Fabtests provides a set of examples that uses [libfabric](http://libfabric.org)
-- a high-performance fabric software library.

## Building fabtests

If you are building Fabtests from a developer Git clone, you must first run the
`autogen.sh` script. This will invoke the GNU Autotools to bootstrap Fabtests'
configuration and build mechanisms. If you are building Fabtests from an
official distribution tarball from libfabric.org, there is no need to run
`autogen.sh`; Fabtests distribution tarballs are already bootstrapped for you.

Fabtests relies on being able to find an installed version of Libfabric. In
some cases, Libfabric may be in default compiler / linker search paths, and you
don't need to tell Fabtests where to find it. In other cases, you may need to
tell Fabtests where to find the installed Libfabric's header and library files
using the `--with-libfabric=PATH` option, described below.

### Configure options

Configure ships with many built in options (see `./configure --help`). Some
useful options are:

```
--prefix=<directory>
```

By default `make install` will place the files into a system wide location.
The prefix option specifies that the fabtests files should be installed into
the directory named `<directory>`. The executables will be located at
`<directory>/bin`.

```
--with-libfabric=<directory>
```

Specify the directory where the libfabric library and header files are
located. This is necessary if libfabric was installed in a non-standard
location and the files are not present in the compiler/linker search paths.
The libraries will be searched for in `<directory>/lib`, and headers will be
searched for in `<directory>/include`.

```
--with-valgrind=<directory>
```

Directory where valgrind is installed.If valgrind is found, then valgrind
annotations are enabled. This may incur a performance penalty.

### Examples

Consider the following example:

```
$ ./configure --with-libfabric=/opt/libfabric --prefix=/opt/fabtests && make -j 32 && sudo make install
```

This will tell the Fabtests to look for Libfabric libraries in the
`/opt/libfabric` tree, and to install the Fabtests in the `/opt/fabtests` tree.

Alternatively:

```
$ ./configure --prefix=/opt/fabtests && make -j 32 && sudo make install
```

Tells the Fabtests that it should be able to find the Libfabric header files
and libraries in default compiler / linker search paths (configure will abort
if it is not able to find them), and to install Fabtests in `/opt/fabtests`.
