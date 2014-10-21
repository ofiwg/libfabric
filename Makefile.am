AM_CFLAGS = -g -Wall -D_GNU_SOURCE -I$(srcdir)/include

bin_PROGRAMS = \
	simple/fi_info \
	simple/fi_pingpong \
	simple/fi_write_lat \
	simple/fi_write_bw \
	simple/fi_read_lat \
	simple/fi_read_bw \
	simple/fi_ud_pingpong \
	unit/fi_eq_test \
	ported/libibverbs/fi_rc_pingpong

simple_fi_info_SOURCES = \
	simple/info.c \
	common/shared.c

simple_fi_pingpong_SOURCES = \
	simple/pingpong.c \
	common/shared.c

simple_fi_write_lat_SOURCES = \
	simple/write_lat.c \
	common/shared.c

simple_fi_write_bw_SOURCES = \
	simple/write_bw.c \
	common/shared.c

simple_fi_read_lat_SOURCES = \
	simple/read_lat.c \
	common/shared.c

simple_fi_read_bw_SOURCES = \
	simple/read_bw.c \
	common/shared.c

simple_fi_ud_pingpong_SOURCES = \
	simple/ud_pingpong.c \
	common/shared.c

unit_fi_eq_test_SOURCES = \
    unit/eq_test.c \
	common/shared.c

ported_libibverbs_fi_rc_pingpong_SOURCES = \
	ported/libibverbs/rc_pingpong.c

man_MANS = man/fabtests.7

EXTRA_DIST = common/shared.h $(man_MANS)
