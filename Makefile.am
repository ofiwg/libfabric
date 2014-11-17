AM_CFLAGS = -g -Wall -D_GNU_SOURCE -I$(srcdir)/include

bin_PROGRAMS = \
	simple/fi_info \
	simple/fi_msg_pingpong \
	simple/fi_msg_rma \
	simple/fi_rdm_pingpong \
	simple/fi_rdm_tagged_pingpong \
	simple/fi_rdm_cntr_pingpong \
	simple/fi_ud_pingpong \
	unit/fi_eq_test \
	unit/fi_av_test \
	ported/libibverbs/fi_rc_pingpong

simple_fi_info_SOURCES = \
	simple/info.c \
	common/shared.c

simple_fi_msg_pingpong_SOURCES = \
	simple/msg_pingpong.c \
	common/shared.c

simple_fi_msg_rma_SOURCES = \
	simple/msg_rma.c \
	common/shared.c

simple_fi_rdm_pingpong_SOURCES = \
	simple/rdm_pingpong.c \
	common/shared.c

simple_fi_rdm_tagged_pingpong_SOURCES = \
	simple/rdm_tagged_pingpong.c \
	common/shared.c

simple_fi_rdm_cntr_pingpong_SOURCES = \
	simple/rdm_cntr_pingpong.c \
	common/shared.c

simple_fi_ud_pingpong_SOURCES = \
	simple/ud_pingpong.c \
	common/shared.c

unit_fi_eq_test_SOURCES = \
    unit/eq_test.c \
	unit/common.c \
	common/shared.c

unit_fi_av_test_SOURCES = \
    unit/av_test.c \
	unit/common.c

ported_libibverbs_fi_rc_pingpong_SOURCES = \
	ported/libibverbs/rc_pingpong.c

man_MANS = man/fabtests.7

EXTRA_DIST = common/shared.h $(man_MANS)
