AM_CFLAGS = -g -Wall -D_GNU_SOURCE

bin_PROGRAMS = \
	simple/fi_pingpong \
	ported/libibverbs/fi_rc_pingpong

simple_fi_pingpong_SOURCES = \
	simple/pingpong.c \
	common/shared.c

ported_libibverbs_fi_rc_pingpong_SOURCES = \
	ported/libibverbs/rc_pingpong.c

man_MANS = man/fabtests.7

EXTRA_DIST = common/shared.h $(man_MANS)
