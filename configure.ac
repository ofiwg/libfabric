dnl
dnl Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
dnl
dnl Process this file with autoconf to produce a configure script.

AC_PREREQ([2.60])
AC_INIT([libfabric], [1.6.2], [ofiwg@lists.openfabrics.org])
AC_CONFIG_SRCDIR([src/fabric.c])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_HEADERS(config.h)
AM_INIT_AUTOMAKE([1.11 dist-bzip2 foreign -Wall -Werror subdir-objects parallel-tests tar-ustar])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_CANONICAL_HOST

macos=0
linux=0
freebsd=0

case $host_os in
*darwin*)
	macos=1
	;;
*linux*)
	linux=1
	;;
*freebsd*)
	freebsd=1
	;;
*)
	AC_MSG_ERROR([libfabric only builds on Linux, OS X, and FreeBSD])
	;;
esac

AM_CONDITIONAL([MACOS], [test "x$macos" = "x1"])
AM_CONDITIONAL([LINUX], [test "x$linux" = "x1"])
AM_CONDITIONAL([FREEBSD], [test "x$freebsd" = "x1"])

base_c_warn_flags="-Wall -Wundef -Wpointer-arith"
debug_c_warn_flags="-Wextra -Wno-unused-parameter -Wno-sign-compare -Wno-missing-field-initializers"
debug_c_other_flags="-fstack-protector-strong"
picky_c_warn_flags="-Wno-long-long -Wmissing-prototypes -Wstrict-prototypes -Wcomment -pedantic"

AC_ARG_WITH([build_id],
	    [AC_HELP_STRING([--with-build_id],
			    [Enable build_id annotation @<:@default=no@:>@])],
	    [], [with_build_id=no])
AS_IF([test x"$with_build_id" = x"no"], [with_build_id=""])
AC_DEFINE_UNQUOTED([BUILD_ID],["$with_build_id"],
                   [adds build_id to version if it was defined])

# Fix autoconf's habit of setting CFLAGS="-g -O2" by default while still
# allowing the user to explicitly set CFLAGS=""
: ${CFLAGS="-fvisibility=hidden -O2 -DNDEBUG ${base_c_warn_flags}"}

# AM_PROG_AS would set CFLAGS="-g -O2" by default if not set already so it
# should not be called earlier
AM_PROG_AS()

# AM PROG_AR did not exist pre AM 1.11.x (where x is somewhere >0 and
# <3), but it is necessary in AM 1.12.x.
m4_ifdef([AM_PROG_AR], [AM_PROG_AR])

AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
		   [Enable valgrind annotations @<:@default=no@:>@]))

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
	AC_DEFINE([INCLUDE_VALGRIND], 1,
		  [Define to 1 to enable valgrind annotations])
	if test -d $with_valgrind; then
		CPPFLAGS="$CPPFLAGS -I$with_valgrind/include"
	fi
fi

AC_ARG_ENABLE([direct],
	[AS_HELP_STRING([--enable-direct=@<:@provider@:>@],
		[Enable direct calls to a fabric provider @<:@default=no@:>@])
	],
	[],
	[enable_direct=no])

dnl Checks for programs
AC_PROG_CC_C99
AM_PROG_CC_C_O
AC_PROG_CPP

AC_ARG_ENABLE([debug],
	      [AS_HELP_STRING([--enable-debug],
			      [Enable debugging @<:@default=no@:>@])
	      ],
	      [],
	      [enable_debug=no])

AS_IF([test x"$enable_debug" != x"no"],
      [dbg=1
       CFLAGS_save=
       # Remove -DNDEBUG flag if set
       for flag in $CFLAGS; do
           if test "$flag" != "-DNDEBUG"; then
               CFLAGS_save="$CFLAGS_save $flag"
           fi
       done
       # See if all the flags in $debug_c_other_flags work
       good_flags=
       for flag in $debug_c_other_flags; do
           AC_MSG_CHECKING([to see if compiler supports $flag])
           CFLAGS="$flag $CFLAGS_save"
           AC_COMPILE_IFELSE([AC_LANG_PROGRAM([[]], [[int i = 3;]])],
	                     [AC_MSG_RESULT([yes])
			      good_flags="$flag $good_flags"],
			     [AC_MSG_RESULT([no])])
       done
       debug_c_other_flags=$good_flags
       CFLAGS=$CFLAGS_save
       unset good_flags
       unset CFLAGS_save

       CFLAGS="-g -O0 ${base_c_warn_flags} ${debug_c_warn_flags} ${debug_c_other_flags} $CFLAGS"],
      [dbg=0])

AC_DEFINE_UNQUOTED([ENABLE_DEBUG],[$dbg],
                   [defined to 1 if libfabric was configured with --enable-debug, 0 otherwise])

dnl Checks for header files.
AC_HEADER_STDC

dnl Check for compiler features
AC_C_TYPEOF

LT_INIT
LT_OUTPUT

dnl dlopen support is optional
AC_ARG_WITH([dlopen],
	AC_HELP_STRING([--with-dlopen],
		       [dl-loadable provider support @<:@default=yes@:>@]),
	)

if test "$freebsd" == "0"; then
AS_IF([test x"$with_dlopen" != x"no"], [
AC_CHECK_LIB(dl, dlopen, [],
    AC_MSG_ERROR([dlopen not found.  libfabric requires libdl.]))
])
fi

dnl handle picky option
AC_ARG_ENABLE([picky],
    [AC_HELP_STRING([--enable-picky],
                    [Enable developer-level compiler pickyness when building @<:@default=no@:>@])])
AS_IF([test x"$enable_picky" = x"yes" && test x"$GCC" = x"yes"],
      [AS_IF([test x"$enable_debug" = x"no"],
             [CFLAGS="${base_c_warn_flags} ${debug_c_warn_flags}
		      ${debug_c_other_flags} ${picky_c_warn_flags} $CFLAGS"],
             [CFLAGS="${picky_c_warn_flags} $CFLAGS"])
      ])

dnl Checks for libraries
AC_CHECK_LIB(pthread, pthread_mutex_init, [],
    AC_MSG_ERROR([pthread_mutex_init() not found.  libfabric requires libpthread.]))

AC_CHECK_FUNC([pthread_spin_init],
	[have_spinlock=1],
	[have_spinlock=0])

dnl shm_open not used in the common code on os-x

if test "$macos" -eq 0; then
AC_CHECK_FUNC([shm_open],
	[],
	[AC_SEARCH_LIBS([shm_open],[rt],[],
	 [AC_MSG_ERROR([shm_open() not found.  libfabric requires shm_open.])])])
fi

AC_DEFINE_UNQUOTED([PT_LOCK_SPIN], [$have_spinlock],
	[Define to 1 if pthread_spin_init is available.])

AC_CHECK_FUNCS([epoll_create])
if test "$ac_cv_func_epoll_create" = yes; then
  AC_DEFINE([HAVE_EPOLL], [1], [Define if you have epoll support.])
fi

AC_CHECK_HEADER([linux/perf_event.h],
    [AC_CHECK_DECL([__builtin_ia32_rdpmc],
        [
            AC_TRY_LINK([#include <linux/perf_event.h>],
                [__builtin_ia32_rdpmc(0);],
                [linux_perf_rdpmc=1],
                [linux_perf_rdpmc=0])
	],
        [linux_perf_rdpmc=0],
        [#include <linux/perf_event.h>])],
    [linux_perf_rdpmc=0])
AC_DEFINE_UNQUOTED(HAVE_LINUX_PERF_RDPMC, [$linux_perf_rdpmc],
    [Whether we have __builtin_ia32_rdpmc() and linux/perf_event.h file or not])
AM_CONDITIONAL([HAVE_LINUX_PERF_RDPMC], [test "x$linux_perf_rdpmc" = "x1"])

dnl Check for gcc atomic intrinsics
AC_MSG_CHECKING(compiler support for c11 atomics)
AC_TRY_LINK([#include <stdatomic.h>],
    [atomic_int a;
     atomic_init(&a, 0);
     #ifdef __STDC_NO_ATOMICS__
       #error c11 atomics are not supported
     #else
       return 0;
     #endif
    ],
    [
	AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_ATOMICS, 1, [Set to 1 to use c11 atomic functions])
    ],
    [AC_MSG_RESULT(no)])


AC_MSG_CHECKING(compiler support for c11 atomic `least` types)
AC_TRY_LINK([#include <stdatomic.h>],
    [atomic_int_least32_t a;
     atomic_int_least64_t b;
    ],
    [
        AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_ATOMICS_LEAST_TYPES, 1,
                  [Set to 1 to use c11 atomic `least` types])
    ],
    [
        AC_MSG_RESULT(no)
    ])

dnl Check for gcc built-in atomics
AC_MSG_CHECKING(compiler support for built-in atomics)
AC_TRY_LINK([#include <stdint.h>],
    [int32_t a;
     __sync_add_and_fetch(&a, 0);
     __sync_sub_and_fetch(&a, 0);
     #if defined(__PPC__) && !defined(__PPC64__)
       #error compiler built-in atomics are not supported on PowerPC 32-bit
     #else
       return 0;
     #endif
    ],
    [
	AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_BUILTIN_ATOMICS, 1, [Set to 1 to use built-in intrincics atomics])
    ],
    [AC_MSG_RESULT(no)])

dnl Check for gcc cpuid intrinsics
AC_MSG_CHECKING(compiler support for cpuid)
AC_TRY_LINK([
     #include <stddef.h>
     #include <cpuid.h>],
    [
     int a, b, c, d;
     __cpuid_count(0, 0, a, b, c, d);
    ],
    [
	AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_CPUID, 1, [Set to 1 to to use cpuid])
    ],
    [AC_MSG_RESULT(no)])

dnl Check for glibc malloc hooks
AC_MSG_CHECKING(compiler support for glibc malloc hooks)
AC_TRY_LINK([#include <malloc.h>],
    [
     __malloc_hook = NULL;
     __realloc_hook = NULL;
     __memalign_hook = NULL;
     __free_hook = NULL;
    ],
    [
	AC_MSG_RESULT(yes)
        AC_DEFINE(HAVE_GLIBC_MALLOC_HOOKS, 1, [Set to 1 to use glibc malloc hooks])
    ],
    [AC_MSG_RESULT(no)])

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
AC_CHECK_HEADER(valgrind/memcheck.h, [],
    AC_MSG_ERROR([valgrind requested but <valgrind/memcheck.h> not found.]))
fi

AC_CACHE_CHECK(whether ld accepts --version-script, ac_cv_version_script,
    [if test -n "`$LD --help < /dev/null 2>/dev/null | grep version-script`"; then
        ac_cv_version_script=yes
    else
        ac_cv_version_script=no
    fi])

AC_ARG_ENABLE([embedded],
	      [AS_HELP_STRING([--enable-embedded],
			      [Enable embedded support (turns off symbol versioning) @<:@default=no@:>@])
	      ],
	      [ac_asm_symver_support=0
               icc_symver_hack=1],
	      [enable_embedded=no])
AM_CONDITIONAL([EMBEDDED], [test x"$enable_embedded" = x"yes"])

AM_CONDITIONAL(HAVE_LD_VERSION_SCRIPT, test "$ac_cv_version_script" = "yes")

dnl Disable symbol versioning when -ipo is in CFLAGS or ipo is disabled by icc.
dnl The gcc equivalent ipo (-fwhole-program) seems to work fine.
AS_CASE([$CFLAGS],
	[*-ipo*],[
		AC_MSG_NOTICE([disabling symbol versioning support with -ipo CFLAG])
		icc_symver_hack=1
		ac_asm_symver_support=0
	],
	[]
)

dnl Check for symbol versioning compiler + linker support.
dnl If icc + ipo, then print disabled and skip check
AC_MSG_CHECKING(for .symver assembler support)
AS_IF([test "$icc_symver_hack"],
	[AC_MSG_RESULT(disabled)],
[

AC_TRY_LINK([],
	[__asm__(".symver main_, main@ABIVER_1.0");],
	[
		AC_MSG_RESULT(yes)
		ac_asm_symver_support=1
	],
	[
		AC_MSG_RESULT(no)
		ac_asm_symver_support=0
	])

]) dnl AS_IF icc_symver_hack

AC_DEFINE_UNQUOTED([HAVE_SYMVER_SUPPORT], [$ac_asm_symver_support],
	  	   [Define to 1 if compiler/linker support symbol versioning.])

AC_MSG_CHECKING(for __alias__ attribute support)
AC_TRY_LINK(
	[
		int foo(int arg);
		int foo(int arg) { return arg + 3; };
		int foo2(int arg) __attribute__ (( __alias__("foo")));
	],
	[ /* empty main */ ],
	[
		AC_MSG_RESULT(yes)
		ac_prog_cc_alias_symbols=1
	],
	[
		AC_MSG_RESULT(no)
		ac_prog_cc_alias_symbols=0
	])

AC_DEFINE_UNQUOTED([HAVE_ALIAS_ATTRIBUTE], [$ac_prog_cc_alias_symbols],
	  	   [Define to 1 if the linker supports alias attribute.])
AC_CHECK_FUNCS([getifaddrs])

dnl Provider-specific checks
FI_PROVIDER_INIT
FI_PROVIDER_SETUP([psm])
FI_PROVIDER_SETUP([psm2])
FI_PROVIDER_SETUP([sockets])
FI_PROVIDER_SETUP([verbs])
dnl The usnic provider must be setup after the verbs provider.  See
dnl prov/usnic/configure.m4 for details.
FI_PROVIDER_SETUP([usnic])
FI_PROVIDER_SETUP([mlx])
FI_PROVIDER_SETUP([gni])
FI_PROVIDER_SETUP([udp])
FI_PROVIDER_SETUP([tcp])
FI_PROVIDER_SETUP([rxm])
FI_PROVIDER_SETUP([rxd])
FI_PROVIDER_SETUP([bgq])
FI_PROVIDER_SETUP([shm])
FI_PROVIDER_FINI
dnl Configure the .pc file
FI_PROVIDER_SETUP_PC

# If the user requested to build in direct mode, but
# we have more than one provider, error.
PROVIDER_DIRECT=no
FI_DIRECT_PROVIDER_API_10=/dev/null
AS_IF([test x"$enable_direct" != x"no"],
      [AS_IF([test "$PROVIDERS_COUNT" -gt 1],
	     [AC_MSG_WARN([Only one provider can be chosen when using --enable-direct])
	      AC_MSG_ERROR(Cannot continue)])

       PROVIDER_DIRECT=$enable_direct
       AC_DEFINE_UNQUOTED([FABRIC_DIRECT_ENABLED], [1], [define when building with FABRIC_DIRECT support])
       FI_DIRECT_PROVIDER_API_10="$srcdir/prov/$enable_direct/provider_FABRIC_1.0.map"
       AS_IF([test ! -r "$FI_DIRECT_PROVIDER_API_10"],
	     [AC_MSG_WARN([--enable-direct=$enable_direct specified, but $FI_DIRECT_PROVIDER_API_10 does not exist])
	      AC_MSG_ERROR([Cannot continue])])])

AC_SUBST(PROVIDER_DIRECT)
AC_SUBST_FILE(FI_DIRECT_PROVIDER_API_10)
AM_CONDITIONAL([HAVE_DIRECT], [test x"$enable_direct" != x"no"])

AC_CONFIG_FILES([Makefile libfabric.spec libfabric.map])
AC_OUTPUT

dnl helpful output
if test "$PROVIDERS_TO_BUILD" = ""; then
	echo "***"
	echo "*** No providers were configured. This may not be what you wanted."
	echo "***"
	exit 1
fi

for i in $PROVIDERS_TO_BUILD; do
	v=${i}_dl
	if test `eval echo \\$${v}` == "1"; then
		dso="$i ${dso}"
	else
		builtin="$i ${builtin}"
	fi
done
cat <<EOF
***
*** Built-in providers:	${builtin}
*** DSO providers:	${dso}
***
EOF
