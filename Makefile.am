configdir = $(datarootdir)/${PACKAGE_NAME}
AM_CFLAGS = -g -Wall -D_GNU_SOURCE -D 'CONFIG_PATH="${configdir}"' -I$(srcdir)/include
ACLOCAL_AMFLAGS = -I config

bin_PROGRAMS = \
	simple/fi_msg \
	simple/fi_msg_sockets \
	simple/fi_rdm \
	simple/fi_rdm_rma_simple \
	simple/fi_rdm_rma_trigger \
	simple/fi_rdm_deferred_wq \
	simple/fi_dgram \
	simple/fi_mcast \
	simple/fi_dgram_waitset \
	simple/fi_rdm_tagged_peek \
	simple/fi_cq_data \
	simple/fi_poll \
	simple/fi_scalable_ep \
	simple/fi_shared_ctx \
	simple/fi_msg_epoll \
	simple/fi_rdm_shared_av \
	simple/fi_cm_data \
	simple/fi_multi_mr \
	simple/fi_rdm_multi_domain \
	simple/fi_multi_ep \
	simple/fi_recv_cancel \
	simple/fi_unexpected_msg \
	streaming/fi_msg_rma \
	streaming/fi_rdm_atomic \
	streaming/fi_rdm_multi_recv \
	streaming/fi_rdm_rma \
	streaming/fi_msg_stream \
	benchmarks/fi_msg_pingpong \
	benchmarks/fi_msg_bw \
	benchmarks/fi_rma_bw \
	benchmarks/fi_rdm_cntr_pingpong \
	benchmarks/fi_dgram_pingpong \
	benchmarks/fi_rdm_pingpong \
	benchmarks/fi_rdm_tagged_pingpong \
	benchmarks/fi_rdm_tagged_bw \
	unit/fi_eq_test \
	unit/fi_cq_test \
	unit/fi_mr_test \
	unit/fi_cntr_test \
	unit/fi_av_test \
	unit/fi_dom_test \
	unit/fi_getinfo_test \
	ported/libibverbs/fi_rc_pingpong \
	ported/librdmacm/fi_cmatose \
	complex/fi_ubertest

dist_bin_SCRIPTS = \
	scripts/runfabtests.sh \
	scripts/rft_yaml_to_junit_xml

dist_noinst_SCRIPTS = \
	scripts/parseyaml.py

nobase_dist_config_DATA = \
        test_configs/eq_cq.test \
        test_configs/lat_bw.test \
        test_configs/sockets/all.test \
        test_configs/sockets/quick.test \
	test_configs/sockets/complete.test \
        test_configs/udp/all.test \
        test_configs/udp/lat_bw.test \
        test_configs/udp/quick.test \
        test_configs/udp/simple.test \
        test_configs/verbs/all.test \
        test_configs/verbs/quick.test \
	test_configs/usnic/all.test \
	test_configs/usnic/quick.test \
	test_configs/psm/all.test \
	test_configs/psm2/all.test

noinst_LTLIBRARIES = libfabtests.la
libfabtests_la_SOURCES = \
	common/shared.c \
	common/jsmn.c \
	include/shared.h \
	include/jsmn.h \
	include/unix/osd.h \
	include/ft_osd.h

benchmarks_srcs = \
	benchmarks/benchmark_shared.h \
	benchmarks/benchmark_shared.c

unit_srcs = \
	include/unit_common.h \
	unit/common.c

if MACOS
if !HAVE_CLOCK_GETTIME
libfabtests_la_SOURCES += common/osx/osd.c
endif
endif

simple_fi_msg_sockets_SOURCES = \
	simple/msg_sockets.c
simple_fi_msg_sockets_LDADD = libfabtests.la

simple_fi_msg_epoll_SOURCES = \
	simple/msg_epoll.c
simple_fi_msg_epoll_LDADD = libfabtests.la

simple_fi_msg_SOURCES = \
	simple/msg.c
simple_fi_msg_LDADD = libfabtests.la

simple_fi_rdm_SOURCES = \
	simple/rdm.c
simple_fi_rdm_LDADD = libfabtests.la

simple_fi_rdm_shared_av_SOURCES = \
	simple/rdm_shared_av.c
simple_fi_rdm_shared_av_LDADD = libfabtests.la

simple_fi_rdm_rma_simple_SOURCES = \
	simple/rdm_rma_simple.c
simple_fi_rdm_rma_simple_LDADD = libfabtests.la

simple_fi_rdm_rma_trigger_SOURCES = \
	simple/rdm_rma_trigger.c
simple_fi_rdm_rma_trigger_LDADD = libfabtests.la

simple_fi_rdm_deferred_wq_SOURCES = \
	simple/rdm_deferred_wq.c
simple_fi_rdm_deferred_wq_LDADD = libfabtests.la

simple_fi_dgram_SOURCES = \
	simple/dgram.c
simple_fi_dgram_LDADD = libfabtests.la

simple_fi_mcast_SOURCES = \
	simple/mcast.c
simple_fi_mcast_LDADD = libfabtests.la

simple_fi_dgram_waitset_SOURCES = \
	simple/dgram_waitset.c
simple_fi_dgram_waitset_LDADD = libfabtests.la

simple_fi_rdm_tagged_peek_SOURCES = \
	simple/rdm_tagged_peek.c
simple_fi_rdm_tagged_peek_LDADD = libfabtests.la

simple_fi_cq_data_SOURCES = \
	simple/cq_data.c
simple_fi_cq_data_LDADD = libfabtests.la

simple_fi_cm_data_SOURCES = \
	simple/cm_data.c
simple_fi_cm_data_LDADD = libfabtests.la

simple_fi_scalable_ep_SOURCES = \
	simple/scalable_ep.c
simple_fi_scalable_ep_LDADD = libfabtests.la

simple_fi_shared_ctx_SOURCES = \
	simple/shared_ctx.c
simple_fi_shared_ctx_LDADD = libfabtests.la

simple_fi_poll_SOURCES = \
	simple/poll.c
simple_fi_poll_LDADD = libfabtests.la

simple_fi_multi_ep_SOURCES = \
	simple/multi_ep.c
simple_fi_multi_ep_LDADD = libfabtests.la

simple_fi_multi_mr_SOURCES = \
	simple/multi_mr.c
simple_fi_multi_mr_LDADD = libfabtests.la

simple_fi_unexpected_msg_SOURCES = \
	simple/unexpected_msg.c
simple_fi_unexpected_msg_LDADD = libfabtests.la

simple_fi_rdm_multi_domain_SOURCES = \
	simple/rdm_multi_domain.c
simple_fi_rdm_multi_domain_LDADD = libfabtests.la

simple_fi_recv_cancel_SOURCES = \
	simple/recv_cancel.c
simple_fi_recv_cancel_LDADD = libfabtests.la

streaming_fi_msg_rma_SOURCES = \
	streaming/msg_rma.c
streaming_fi_msg_rma_LDADD = libfabtests.la

streaming_fi_msg_stream_SOURCES = \
	streaming/msg.c
streaming_fi_msg_stream_LDADD = libfabtests.la

streaming_fi_rdm_atomic_SOURCES = \
	streaming/rdm_atomic.c
streaming_fi_rdm_atomic_LDADD = libfabtests.la

streaming_fi_rdm_multi_recv_SOURCES = \
	streaming/rdm_multi_recv.c
streaming_fi_rdm_multi_recv_LDADD = libfabtests.la

streaming_fi_rdm_rma_SOURCES = \
	streaming/rdm_rma.c
streaming_fi_rdm_rma_LDADD = libfabtests.la

benchmarks_fi_msg_pingpong_SOURCES = \
	benchmarks/msg_pingpong.c \
	$(benchmarks_srcs)
benchmarks_fi_msg_pingpong_LDADD = libfabtests.la

benchmarks_fi_msg_bw_SOURCES = \
	benchmarks/msg_bw.c \
	$(benchmarks_srcs)
benchmarks_fi_msg_bw_LDADD = libfabtests.la

benchmarks_fi_rma_bw_SOURCES = \
	benchmarks/rma_bw.c \
	$(benchmarks_srcs)
benchmarks_fi_rma_bw_LDADD = libfabtests.la

benchmarks_fi_dgram_pingpong_SOURCES = \
	benchmarks/dgram_pingpong.c \
	$(benchmarks_srcs)
benchmarks_fi_dgram_pingpong_LDADD = libfabtests.la

benchmarks_fi_rdm_cntr_pingpong_SOURCES = \
	benchmarks/rdm_cntr_pingpong.c \
	$(benchmarks_srcs)
benchmarks_fi_rdm_cntr_pingpong_LDADD = libfabtests.la

benchmarks_fi_rdm_pingpong_SOURCES = \
	benchmarks/rdm_pingpong.c \
	$(benchmarks_srcs)
benchmarks_fi_rdm_pingpong_LDADD = libfabtests.la

benchmarks_fi_rdm_tagged_pingpong_SOURCES = \
	benchmarks/rdm_tagged_pingpong.c \
	$(benchmarks_srcs)
benchmarks_fi_rdm_tagged_pingpong_LDADD = libfabtests.la

benchmarks_fi_rdm_tagged_bw_SOURCES = \
	benchmarks/rdm_tagged_bw.c \
	$(benchmarks_srcs)
benchmarks_fi_rdm_tagged_bw_LDADD = libfabtests.la


unit_fi_eq_test_SOURCES = \
	unit/eq_test.c \
	$(unit_srcs)
unit_fi_eq_test_LDADD = libfabtests.la

unit_fi_cq_test_SOURCES = \
	unit/cq_test.c \
	$(unit_srcs)
unit_fi_cq_test_LDADD = libfabtests.la

unit_fi_mr_test_SOURCES = \
	unit/mr_test.c \
	$(unit_srcs)
unit_fi_mr_test_LDADD = libfabtests.la

unit_fi_cntr_test_SOURCES = \
	unit/cntr_test.c \
	$(unit_srcs)
unit_fi_cntr_test_LDADD = libfabtests.la

unit_fi_av_test_SOURCES = \
	unit/av_test.c \
	$(unit_srcs)
unit_fi_av_test_LDADD = libfabtests.la

unit_fi_dom_test_SOURCES = \
	unit/dom_test.c \
	$(unit_srcs)
unit_fi_dom_test_LDADD = libfabtests.la

unit_fi_getinfo_test_SOURCES = \
	unit/getinfo_test.c \
	$(unit_srcs)
unit_fi_getinfo_test_LDADD = libfabtests.la

ported_libibverbs_fi_rc_pingpong_SOURCES = \
	ported/libibverbs/rc_pingpong.c
ported_libibverbs_fi_rc_pingpong_LDADD = libfabtests.la

ported_librdmacm_fi_cmatose_SOURCES = \
	ported/librdmacm/cmatose.c
ported_librdmacm_fi_cmatose_LDADD = libfabtests.la

complex_fi_ubertest_SOURCES = \
	complex/fabtest.h \
	complex/ofi_atomic.h \
	complex/ofi_atomic.c \
	complex/ft_main.c \
	complex/ft_comm.c \
	complex/ft_comp.c \
	complex/ft_config.c \
	complex/ft_domain.c \
	complex/ft_endpoint.c \
	complex/ft_msg.c \
	complex/ft_verify.c \
	complex/ft_test.c
complex_fi_ubertest_LDADD = libfabtests.la

man_MANS = man/fabtests.7

EXTRA_DIST = \
	fabtests.spec.in $(man_MANS)

dist-hook: fabtests.spec
	cp fabtests.spec $(distdir)

test:
	./scripts/runfabtests.sh -vvv
	./scripts/runfabtests.sh -vvv udp
