bin_PROGRAMS = \
	hpcs_sendrecv \
	hpcs_onesided

noinst_LTLIBRARIES = 	libcore.la \
			libpattern.la

libcore_la_SOURCES = \
	src/core/core.c \
	include/core/user.h

libcore_la_CFLAGS = \
	$(AM_CFLAGS) \
	-I$(srcdir)/include

libpattern_la_SOURCES = \
	src/pattern/alltoall.c \
	src/pattern/self.c \
	src/pattern/alltoone.c \
	include/pattern/user.h \
	src/pattern/util.h

libpattern_la_CFLAGS = $(libcore_la_CFLAGS)


hpcs_sendrecv_SOURCES = src/test/sendrecv.c \
			include/test/user.h

hpcs_sendrecv_CFLAGS = $(libcore_la_CFLAGS)

hpcs_sendrecv_LDADD = \
	src/harness/libharness.la \
	libcore.la \
	libpattern.la

hpcs_onesided_SOURCES = src/test/onesided.c \
			include/test/user.h

hpcs_onesided_CFLAGS = $(libcore_la_CFLAGS)

hpcs_onesided_LDADD = \
	src/harness/libharness.la \
	libcore.la \
	libpattern.la

SUBDIRS = src/harness .

