#
# Copyright (c) 2016 Cisco Systems, Inc.  All rights reserved.
# Copyright (c) 2017-2018 Intel Corporation, Inc. All right reserved.
# Copyright (c) 2018 Amazon.com, Inc. or its affiliates. All rights reserved.
# (C) Copyright 2020 Hewlett Packard Enterprise Development LP
#
# Makefile.am for libudp-fi

AM_CPPFLAGS = \
	-I$(srcdir)/include \
	-D_GNU_SOURCE -D__USE_XOPEN2K8 \
	-DSYSCONFDIR=\"$(sysconfdir)\" \
	-DPROVDLDIR=\"$(pkglibdir)\"

noinst_LTLIBRARIES =
lib_LTLIBRARIES =
# libfabric_pkglibdir = $(libdir)/libfabric
# ibfabric_pkglib_LTLIBRARIES =

if EMBEDDED
noinst_LTLIBRARIES += src/libudp-fi.la
else
lib_LTLIBRARIES += src/libudp-fi.la
# libfabric_pkglib_LTLIBRARIES += src/libudp-fi.la
endif

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -Wall

if HAVE_LD_VERSION_SCRIPT
    libudp_fi_version_script = -Wl,--version-script=$(builddir)/libudp-fi.map
else !HAVE_LD_VERSION_SCRIPT
    libudp_fi_version_script =
endif !HAVE_LD_VERSION_SCRIPT

# internal utility functions shared by in-tree providers:
common_srcs = \
	shared/hmem.c \
	shared/hmem_rocr.c \
	shared/hmem_cuda.c \
	shared/hmem_ze.c \
	shared/common.c \
	shared/enosys.c \
	shared/rbtree.c \
	shared/tree.c \
	shared/fasthash.c \
	shared/indexer.c \
	shared/mem.c \
	shared/iov.c \
	shared/shared/ofi_str.c \
	util/src/util_atomic.c \
	util/src/util_attr.c \
	util/src/util_av.c \
	util/src/util_cq.c \
	util/src/util_cntr.c
	util/src/util_domain.c
	util/src/util_ep.c \
	util/src/util_pep.c
	util/src/util_eq.c \
	util/src/util_fabric.c \
	util/src/util_main.c \
	util/src/util_poll.c \
	util/src/util_wait.c \
	util/src/util_buf.c \
	util/src/util_mr_map.c \
	util/src/util_ns.c \
	util/src/util_shm.c \
	util/src/util_mem_monitor.c \
	util/src/util_mem_hooks.c \
	util/src/util_mr_cache.c \
	util/src/cuda_mem_monitor.c \
	util/src/rocr_mem_monitor.c \
	util/src/util_coll.c


if MACOS
common_srcs += shared/unix/osd.c
common_srcs += include/osx/osd.h
common_srcs += include/unix/osd.h
endif

if FREEBSD
common_srcs += shared/unix/osd.c
common_srcs += include/freebsd/osd.h
common_srcs += include/unix/osd.h
endif

if LINUX
common_srcs += shared/unix/osd.c
common_srcs += shared/linux/osd.c
if HAVE_LINUX_PERF_RDPMC
common_srcs += shared/linux/rdpmc.c
endif
common_srcs += include/linux/rdpmc.h
common_srcs += include/linux/osd.h
common_srcs += include/unix/osd.h
endif

bin_SCRIPTS =

nodist_src_libudp_fi_la_SOURCES =
src_libudp_fi_la_SOURCES = \
	include/ofi_hmem.h \
	include/ofi.h \
	include/ofi_abi.h \
	include/ofi_atom.h \
	include/ofi_enosys.h \
	include/ofi_file.h \
	include/ofi_hook.h \
	include/ofi_indexer.h \
	include/ofi_iov.h \
	include/ofi_list.h \
	include/ofi_bitmask.h \
	include/shared/ofi_str.h \
	include/ofi_lock.h \
	include/ofi_mem.h \
	include/ofi_osd.h \
	include/ofi_proto.h \
	include/ofi_recvwin.h \
	include/ofi_rbuf.h \
	include/ofi_shm.h \
	include/ofi_signal.h \
	include/ofi_epoll.h \
	include/ofi_tree.h \
	include/ofi_util.h \
	include/ofi_atomic.h \
	include/ofi_mr.h \
	include/ofi_net.h \
	include/ofi_perf.h \
	include/ofi_coll.h \
	include/fasthash.h \
	include/rbtree.h \
	include/uthash.h \
	include/ofi_prov.h \
	include/rdma/providers/fi_log.h \
	include/rdma/providers/fi_prov.h
	include/rdma/fabric.h \
	include/rdma/fi_atomic.h \
	include/rdma/fi_cm.h \
	include/rdma/fi_collective.h \
	include/rdma/fi_domain.h \
	include/rdma/fi_eq.h \
	include/rdma/fi_rma.h \
	include/rdma/fi_endpoint.h \
	include/rdma/fi_errno.h \
	include/rdma/fi_tagged.h \
	include/rdma/fi_trigger.h \
	src/udpx_attr.c \
	src/udpx_cq.c \
	src/udpx_domain.c \
	src/udpx_ep.c \
	src/udpx_fabric.c \
	src/udpx_init.c \
	src/udpx.h \
	$(common_srcs)

src_libudp_fi_la_CPPFLAGS = $(AM_CPPFLAGS)
src_libudp_fi_la_LDFLAGS =
src_libudp_fi_la_LIBADD =
src_libudp_fi_la_DEPENDENCIES = libudp-fi.map

if !EMBEDDED
src_libudp_fi_la_LDFLAGS += -version-info 15:0:14
endif
src_libudp_fi_la_LDFLAGS += -export-dynamic \
			   $(libudp_fi_version_script)

pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = libudp-fi.pc

nroff:
	@for file in $(prov_install_man_pages); do \
	    source=`echo $$file | sed -e 's@/man[0-9]@@'`; \
	    perl $(top_srcdir)/config/md2nroff.pl --source=$(top_srcdir)/$$source.md; \
	done

dist-hook: libudp-fi.spec
	cp libudp-fi.spec $(distdir)
	perl $(top_srcdir)/config/distscript.pl "$(distdir)" "$(PACKAGE_VERSION)"

rpm: dist
	LDFLAGS=-Wl,--build-id rpmbuild -ta libudp-fi-$(PACKAGE_VERSION).tar.bz2

prov_install_man_pages = man/man7/fi_udp.7

prov_dist_man_pages = man/man7/fi_udp.7

man_MANS = $(prov_install_man_pages)

EXTRA_DIST = \
        libudp-fi.spec.in \
        config/distscript.pl \
        $(prov_dist_man_pages)
