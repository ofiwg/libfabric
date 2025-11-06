# SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
# Copyright 2018, 2020-2021 Hewlett Packard Enterprise Development LP

# Current version of the tree
GIT_VERSION := $(shell git describe --abbrev=8 --dirty --always --tags)

AM_CPPFLAGS = \
	-I$(srcdir)/include \
	-I$(srcdir)/../cxi-driver/include \
	-I$(srcdir)/../cassini-headers/install/include \
	-fvisibility=hidden -fsigned-char
AM_CFLAGS = \
	${pkg_CFLAGS}

lib_LTLIBRARIES = utils/libcxiutils.la src/libcxi.la
src_libcxi_la_SOURCES = \
	src/libcxi.c \
	src/devinfo.c \
	src/memory.c \
	include/libcxi_priv.h
src_libcxi_la_CPPFLAGS= $(AM_CPPFLAGS) ${LIBNL_CFLAGS}
src_libcxi_la_LDFLAGS = ${LIBNL_LIBS} \
			-version-info @lt_current@:@lt_revision@:@lt_age@ \
			-Wl,--version-script=$(top_srcdir)/src/libcxi.sym
src_libcxi_la_DEPENDENCIES = ${top_srcdir}/src/libcxi.sym

utils_libcxiutils_la_SOURCES = \
	utils/ctrl_connection.c \
	utils/cxi_context.c \
	utils/utils_common.c \
	utils/utils_common.h \
	utils/utils_gpu_cuda.c \
	utils/utils_gpu_hip.c \
	utils/utils_gpu_ze.c \
	utils/get_clock.c
utils_libcxiutils_la_LDFLAGS = -lm
utils_libcxiutils_la_CPPFLAGS=$(AM_CPPFLAGS) -fvisibility=default

pkginclude_HEADERS = include/libcxi.h
EXTRA_DIST = src/libcxi.sym README.md retry_handler/cxi_rh.conf

AM_DISTCHECK_CONFIGURE_FLAGS = \
  --with-systemdsystemunitdir=$$dc_install_base/$(systemdsystemunitdir) \
  --with-udevrulesdir=$$dc_install_base/$(udevrulesdir)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libcxi.pc
EXTRA_DIST += libcxi.pc.in

CLEANFILES = \
	libcxi.pc \
	utils/cxi_dump_csrs.h

bin_PROGRAMS = \
	tests/cxi_device_list tests/test_write_csr tests/test_map_csr retry_handler/cxi_rh \
	utils/cxi_send_lat utils/cxi_heatsink_check utils/cxi_write_bw \
	utils/cxi_stat utils/cxi_write_lat utils/cxi_read_bw utils/cxi_read_lat \
	utils/cxi_atomic_bw utils/cxi_atomic_lat utils/cxi_send_bw \
	utils/cxi_gpu_loopback_bw utils/cxi_udp_gen utils/cxi_service \
	utils/cxi_dump_csrs

utils_cxi_send_lat_SOURCES = utils/send_lat.c
utils_cxi_send_lat_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_send_bw_SOURCES = utils/send_bw.c
utils_cxi_send_bw_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_heatsink_check_SOURCES = utils/heatsink_check.c
utils_cxi_heatsink_check_LDFLAGS = -lpthread -lrt -lsensors -lnuma
utils_cxi_heatsink_check_LDADD = $(lib_LTLIBRARIES)

utils_cxi_write_bw_SOURCES = utils/write_bw.c
utils_cxi_write_bw_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_stat_SOURCES = utils/cxi_stat.c
utils_cxi_stat_LDADD = src/libcxi.la

cxi_yamldir = $(datadir)/cxi
dist_cxi_yaml_DATA = utils/cxi_service_template.yaml
utils_cxi_service_SOURCES = utils/cxi_service.c
utils_cxi_service_LDADD = ${LIBYAML_LIBS} src/libcxi.la

utils_cxi_write_lat_SOURCES = utils/write_lat.c
utils_cxi_write_lat_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_read_bw_SOURCES = utils/read_bw.c
utils_cxi_read_bw_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_read_lat_SOURCES = utils/read_lat.c
utils_cxi_read_lat_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_atomic_bw_SOURCES = utils/atomic_bw.c
utils_cxi_atomic_bw_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_atomic_lat_SOURCES = utils/atomic_lat.c
utils_cxi_atomic_lat_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_gpu_loopback_bw_SOURCES = utils/loopback_bw.c
utils_cxi_gpu_loopback_bw_LDADD = $(lib_LTLIBRARIES) -lm

utils_cxi_udp_gen_CFLAGS = -Wno-address-of-packed-member
utils_cxi_udp_gen_LDFLAGS = -lpthread
utils_cxi_udp_gen_LDADD = src/libcxi.la

utils_cxi_dump_csrs_SOURCES = utils/cxi_dump_csrs.c
utils_cxi_dump_csrs_LDADD = src/libcxi.la
utils/cxi_dump_csrs.c: utils/cxi_dump_csrs.h
utils/cxi_dump_csrs.h: utils/cxi_dump_csrs.py
	utils/cxi_dump_csrs.py > utils/cxi_dump_csrs.h

utils_MANS = \
	man/man1/cxi_atomic_bw.1 \
	man/man1/cxi_atomic_lat.1 \
	man/man1/cxi_dump_csrs.1 \
	man/man1/cxi_gpu_loopback_bw.1 \
	man/man1/cxi_heatsink_check.1 \
	man/man1/cxi_read_bw.1 \
	man/man1/cxi_read_lat.1 \
	man/man1/cxi_send_bw.1 \
	man/man1/cxi_send_lat.1 \
	man/man1/cxi_service.1 \
	man/man1/cxi_stat.1 \
	man/man1/cxi_udp_gen.1 \
	man/man1/cxi_write_bw.1 \
	man/man1/cxi_write_lat.1 \
	man/man7/cxi_diags.7 \
	man/man7/cxi_service.7

MANPAGE_SRCS = $(patsubst man/man%/%.%, man/\2.\3.md, $(utils_MANS))

manpages: $(utils_MANS)

if HAVE_PANDOC
%.1: ../%.1.md
	pandoc --standalone --to man $< -o $@

%.7: ../%.7.md
	pandoc --standalone --to man $< -o $@
endif

tests_cxi_device_list_CPPFLAGS = $(AM_CPPFLAGS)
tests_cxi_device_list_LDADD = src/libcxi.la

tests_test_map_csr_CPPFLAGS = $(AM_CPPFLAGS)
tests_test_map_csr_LDADD = src/libcxi.la

tests_test_write_csr_CPPFLAGS = $(AM_CPPFLAGS)
tests_test_write_csr_LDADD = src/libcxi.la

retry_handler_cxi_rh_CPPFLAGS = $(AM_CPPFLAGS) ${LIBCONFIG_CFLAGS} \
	${LIBUV_CFLAGS} ${LIBFUSE_CFLAGS} ${LIBSYSTEMD_CFLAGS} \
	-DGIT_VERSION=\"$(GIT_VERSION)\" \
	-DSYSCONFDIR=\"$(sysconfdir)\"
retry_handler_cxi_rh_LDADD = ${LIBCONFIG_LIBS} \
	${LIBUV_LIBS} ${LIBFUSE_LIBS} ${LIBSYSTEMD_LIBS} src/libcxi.la
retry_handler_cxi_rh_SOURCES = \
	retry_handler/config.c \
	retry_handler/list.h \
	retry_handler/logging.c \
	retry_handler/main.c \
	retry_handler/ordered.c \
	retry_handler/recovery.c \
	retry_handler/rh.h \
	retry_handler/sct_timeout.c \
	retry_handler/stats.c \
	retry_handler/tct_timeout.c \
	retry_handler/timers.c \
	retry_handler/unordered.c

sysconf_DATA = retry_handler/cxi_rh.conf

if HAVE_SYSTEMD
systemdsystemunit_DATA = retry_handler/cxi_rh@.service
dist_systemdsystemunit_DATA = retry_handler/cxi_rh.target
else
EXTRA_DIST += retry_handler/cxi_rh.target
endif

if HAVE_UDEV
dist_udevrules_DATA = retry_handler/60-cxi.rules
else
# ensure rules file is included in distribution even if udev is not present
# when configure is run
EXTRA_DIST += retry_handler/60-cxi.rules
endif

if HAVE_CRITERION
bin_PROGRAMS += \
	tests/libcxi_test
check_PROGRAMS = \
	tests/libcxi_test
TESTS = \
	tests/run_tests_vm.sh

tests_libcxi_test_SOURCES = \
	tests/libcxi_test_common.c \
	tests/ucxi_test.c \
	tests/libcxi_test.c \
	tests/libcxi_test_pte.c \
	tests/libcxi_test_svc.c \
	tests/libcxi_gpu_hip.c \
	tests/libcxi_gpu_cuda.c \
	tests/libcxi_gpu_ze.c \
	tests/map.c \
	tests/rgid.c \
	tests/rma.c \
	tests/atomic.c \
	tests/msg.c \
	tests/rendezvous.c \
	tests/ct.c \
	tests/fork.c
tests_libcxi_test_CPPFLAGS = $(AM_CPPFLAGS) $(libcxi_test_CPPFLAGS)
tests_libcxi_test_LDFLAGS = $(libcxi_test_LDFLAGS) -static
tests_libcxi_test_LDADD =  $(libcxi_test_LIBS) $(lib_LTLIBRARIES)

LOG_DRIVER = env AM_TAP_AWK='$(AWK)' $(SHELL) \
              $(top_srcdir)/tap-driver.sh

endif HAVE_CRITERION

# UML documentation for the retry handler
UML_FILES = \
	retry_handler/docs/request-timeout-nack.uml \
	retry_handler/docs/retry-sct-close.uml \
	retry_handler/docs/spt-status.uml \
	retry_handler/docs/tct-timeout.uml

rhdocs: $(patsubst %.uml, %.svg, $(UML_FILES))

.uml.svg:
if HAVE_PLANTUML
	$(PLANTUML) -tsvg $<
	dos2unix $@
else
	@echo "Note: plantuml needed for $(@)"
endif

dist_man_MANS = $(utils_MANS)

EXTRA_DIST += $(UML_FILES) $(patsubst %.uml, %.svg, $(UML_FILES)) \
	retry_handler/README retry_handler/dracut.conf.d/50-cxi-driver.conf \
	autogen.sh cray-libcxi.spec $(MANPAGE_SRCS)

$(PACKAGE)-$(VERSION).tar.gz: dist-gzip

rpm: $(PACKAGE)-$(VERSION).tar.gz
	BUILD_METADATA='0' rpmbuild -ta $<
