dnl Process this file with autoconf to produce a configure script.

AC_PREREQ(2.57)
AC_INIT(libfabric, 0.0.1, linux-rdma@vger.kernel.org)
AC_CONFIG_SRCDIR([src/fabric.c])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIR(config)
AC_CONFIG_HEADERS(config.h)
AM_INIT_AUTOMAKE(libfabric, 0.0.1)
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

AC_ARG_ENABLE([debug],
	      [AS_HELP_STRING([--enable-debug],
			      [Enable debugging @<:@default=no@:>@])
	      ],
	      [CFLAGS="$CFLAGS -g -O0 -Wall"],
	      [enable_debug=no])

dnl Fix autoconf's habit of adding -g -O2 by default
AS_IF([test -z "$CFLAGS"],
      [CFLAGS='-O2 -DNDEBUG -Wall'])

AM_PROG_LIBTOOL

AC_ARG_WITH([valgrind],
    AC_HELP_STRING([--with-valgrind],
		   [Enable valgrind annotations - default NO]))

if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
	AC_DEFINE([INCLUDE_VALGRIND], 1,
		  [Define to 1 to enable valgrind annotations])
	if test -d $with_valgrind; then
		CPPFLAGS="$CPPLFAGS -I$with_valgrind/include"
	fi
fi

AC_ARG_ENABLE(libcheck, [  --disable-libcheck      do not test for presence of libraries],
[       if test "$enableval" = "no"; then
                disable_libcheck=yes
        fi
])

dnl Checks for programs
AC_PROG_CC

dnl Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_CHECK_SIZEOF(long)

dnl Checks for libraries
AC_CHECK_LIB(dl, dlsym, [],
    AC_MSG_ERROR([dlsym() not found.  libfabric requires libdl.]))
AC_CHECK_LIB(pthread, pthread_mutex_init, [],
    AC_MSG_ERROR([pthread_mutex_init() not found.  libfabric requires libpthread.]))

dnl Check for gcc atomic intrinsics
AC_MSG_CHECKING(compiler support for atomics)
AC_TRY_LINK([int i = 0;],
    [ return __sync_add_and_fetch(&i, 1) != __sync_sub_and_fetch(&i, 1); ],
    [ AC_MSG_RESULT(yes) ],
    [
        AC_MSG_RESULT(no)
        AC_DEFINE(DEFINE_ATOMICS, 1, [Set to 1 to implement atomics])
    ])

dnl Checks for header files.
AC_HEADER_STDC

if test "$disable_libcheck" != "yes"; then
if test "$with_valgrind" != "" && test "$with_valgrind" != "no"; then
AC_CHECK_HEADER(valgrind/memcheck.h, [],
    AC_MSG_ERROR([valgrind requested but <valgrind/memcheck.h> not found.]))
fi
fi

AC_CACHE_CHECK(whether ld accepts --version-script, ac_cv_version_script,
    if test -n "`$LD --help < /dev/null 2>/dev/null | grep version-script`"; then
        ac_cv_version_script=yes
    else
        ac_cv_version_script=no
    fi)

AM_CONDITIONAL(HAVE_LD_VERSION_SCRIPT, test "$ac_cv_version_script" = "yes")

AC_ARG_ENABLE([psm],
	      [AS_HELP_STRING([--enable-psm],
			      [Enable PSM provider @<:@default=auto@:>@])
	      ],
	      [],
	      [enable_psm=auto])

AC_ARG_WITH([psm],
	    [AS_HELP_STRING([--with-psm=@<:@PSM installation path@:>@],
			    [Provide path to PSM installation])
	    ],
	    [AS_CASE([$with_psm],
		     [yes|no], [],
		     [CPPFLAGS="-I$with_psm/include $CPPFLAGS"
		      LDFLAGS="-L$with_psm/lib64 -Wl,-rpath=$with_psm/lib64 $LDFLAGS"])
	     enable_psm=yes
	    ])

AC_ARG_WITH([psm-include],
            [AS_HELP_STRING([--with-psm-include=@<:@PSM include path@:>@],
                            [Provide path to PSM include files])
            ],
            [AS_CASE([$with_psm_include],
                     [yes|no], [],
                     [CPPFLAGS="-I$with_psm_include $CPPFLAGS"])
	     enable_psm=yes
            ])

AC_ARG_WITH([psm-lib],
            [AS_HELP_STRING([--with-psm-lib=@<:@PSM library path@:>@],
                            [Provide path to PSM library files])
            ],
            [AS_CASE([$with_psm_lib],
                     [yes|no], [],
                     [LDFLAGS="-L$with_psm_lib -Wl,-rpath=$with_psm_lib $LDFLAGS"])
	     enable_psm=yes
            ])

AS_CASE([$enable_psm],
	[auto], [AC_CHECK_LIB(psm_infinipath, psm_init,
			[AC_CHECK_HEADER([psm.h], [enable_psm=yes], [enable_psm=no])],
			[enable_psm=no])
		],
	[yes],	[AC_CHECK_LIB(psm_infinipath, psm_init,
			[AC_CHECK_HEADER([psm.h], [],
				[AC_MSG_ERROR([psm.h not found. Provide the correct path to PSM with --with-psm-include (or --with-psm)])])
			],
			[AC_MSG_ERROR([psm_init() not found. Provide the correct path to PSM --with-psm-lib])])
		],
	[no],	[],
	[])

AS_IF([test x"$enable_psm" = x"yes"],
	[AC_DEFINE([HAVE_PSM], [1], [Define if PSM is enabled])
	 LIBS="-lpsm_infinipath $LIBS"
	],
	[AC_MSG_NOTICE(PSM not enabled)])

AM_CONDITIONAL([HAVE_PSM], [test x"$enable_psm" = x"yes"])

AC_ARG_ENABLE([direct],
	[AS_HELP_STRING([--enable-direct=@<:@provider@:>@],
		[Enable direct calls to a fabric provider @<:@defaul=no@:>@])
	],
	[AS_CASE([$enable_direct],
		[mlx4], [AC_DEFINE([HAVE_MLX4_DIRECT], [1],
			[Define if mlx4 direct provider is enabled])],
		[])
	],
        [enable_direct=no])

AM_CONDITIONAL([HAVE_MLX4_DIRECT], [test x"$enable_direct" = x"mlx4"])

AC_CONFIG_FILES([Makefile libfabric.spec])
AC_OUTPUT
