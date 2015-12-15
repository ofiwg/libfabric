# Makefile.am for libfabric

AM_CPPFLAGS = \
	-I$(srcdir)/include \
	-D_GNU_SOURCE \
	-DSYSCONFDIR=\"$(sysconfdir)\" \
	-DRDMADIR=\"@rdmadir@\" \
	-DPROVDLDIR=\"$(pkglibdir)\"

lib_LTLIBRARIES = src/libfabric.la

pkglib_LTLIBRARIES = $(DL_PROVIDERS)

noinst_LTLIBRARIES =

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -g -Wall

if HAVE_LD_VERSION_SCRIPT
    libfabric_version_script = -Wl,--version-script=$(srcdir)/libfabric.map
else !HAVE_LD_VERSION_SCRIPT
    libfabric_version_script =
endif !HAVE_LD_VERSION_SCRIPT

rdmaincludedir = $(includedir)/rdma

rdmainclude_HEADERS =

# internal utility functions shared by in-tree providers:
common_srcs = \
	src/common.c \
	src/enosys.c \
	src/rbtree.c \
	src/fasthash.c \
	src/indexer.c

# ensure dl-built providers link back to libfabric
linkback = $(top_builddir)/src/libfabric.la

bin_PROGRAMS = \
	util/fi_info

bin_SCRIPTS =

util_fi_info_SOURCES = \
	util/info.c
util_fi_info_LDADD = $(linkback)

src_libfabric_la_SOURCES = \
	include/fi.h \
	include/fi_enosys.h \
	include/fi_indexer.h \
	include/fi_list.h \
	include/fi_signal.h \
	include/fi_rbuf.h \
	include/rbtree.h \
	include/fasthash.h \
	include/prov.h \
	src/fabric.c \
	src/fi_tostr.c \
	src/log.c \
	src/var.c \
	$(common_srcs)

if MACOS
src_libfabric_la_SOURCES += src/osx/osd.c
src_libfabric_la_SOURCES += include/osx/osd.h
endif

if LINUX
src_libfabric_la_SOURCES += include/linux/osd.h
endif

src_libfabric_la_CPPFLAGS = $(AM_CPPFLAGS)
src_libfabric_la_LDFLAGS =
src_libfabric_la_LIBADD =
src_libfabric_la_DEPENDENCIES = $(srcdir)/libfabric.map

src_libfabric_la_LDFLAGS += -version-info 2:0:1 -export-dynamic \
			   $(libfabric_version_script)
rdmainclude_HEADERS += \
	$(top_srcdir)/include/rdma/fabric.h \
	$(top_srcdir)/include/rdma/fi_atomic.h \
	$(top_srcdir)/include/rdma/fi_cm.h \
	$(top_srcdir)/include/rdma/fi_domain.h \
	$(top_srcdir)/include/rdma/fi_eq.h \
	$(top_srcdir)/include/rdma/fi_log.h \
	$(top_srcdir)/include/rdma/fi_prov.h \
	$(top_srcdir)/include/rdma/fi_rma.h \
	$(top_srcdir)/include/rdma/fi_endpoint.h \
	$(top_srcdir)/include/rdma/fi_errno.h \
	$(top_srcdir)/include/rdma/fi_tagged.h \
	$(top_srcdir)/include/rdma/fi_trigger.h

if HAVE_DIRECT
nodist_rdmainclude_HEADERS = \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_domain.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_endpoint.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_tagged.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_rma.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_atomic_def.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_atomic.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_cm.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_eq.h \
	$(top_srcdir)/prov/$(PROVIDER_DIRECT)/include/rdma/fi_direct_trigger.h
endif HAVE_DIRECT

real_man_pages = \
        man/man3/fi_av.3 \
        man/man3/fi_cm.3 \
        man/man3/fi_cntr.3 \
        man/man3/fi_control.3 \
        man/man3/fi_cq.3 \
        man/man3/fi_domain.3 \
        man/man3/fi_endpoint.3 \
        man/man3/fi_errno.3 \
        man/man3/fi_eq.3 \
        man/man3/fi_fabric.3 \
        man/man3/fi_getinfo.3 \
        man/man3/fi_mr.3 \
        man/man3/fi_msg.3 \
        man/man3/fi_poll.3 \
        man/man3/fi_rma.3 \
        man/man3/fi_tagged.3 \
        man/man3/fi_trigger.3 \
        man/man3/fi_version.3 \
        man/man7/fabric.7 \
        man/man7/fi_provider.7 \
        man/man7/fi_direct.7

dummy_man_pages = \
        man/man3/fi_accept.3 \
        man/man3/fi_alias.3 \
        man/man3/fi_atomic.3 \
        man/man3/fi_atomic_valid.3 \
        man/man3/fi_atomicmsg.3 \
        man/man3/fi_atomicv.3 \
        man/man3/fi_av_bind.3 \
        man/man3/fi_av_insert.3 \
        man/man3/fi_av_insertsvc.3 \
        man/man3/fi_av_lookup.3 \
        man/man3/fi_av_open.3 \
        man/man3/fi_av_remove.3 \
        man/man3/fi_av_straddr.3 \
        man/man3/fi_cancel.3 \
        man/man3/fi_close.3 \
        man/man3/fi_cntr_add.3 \
        man/man3/fi_cntr_open.3 \
        man/man3/fi_cntr_read.3 \
        man/man3/fi_cntr_set.3 \
        man/man3/fi_cntr_wait.3 \
        man/man3/fi_compare_atomic.3 \
        man/man3/fi_compare_atomic_valid.3 \
        man/man3/fi_compare_atomicmsg.3 \
        man/man3/fi_compare_atomicv.3 \
        man/man3/fi_connect.3 \
        man/man3/fi_cq_open.3 \
        man/man3/fi_cq_read.3 \
        man/man3/fi_cq_readerr.3 \
        man/man3/fi_cq_readfrom.3 \
        man/man3/fi_cq_sread.3 \
        man/man3/fi_cq_sreadfrom.3 \
        man/man3/fi_cq_strerror.3 \
        man/man3/fi_cq_signal.3 \
        man/man3/fi_domain_bind.3 \
        man/man3/fi_domain_query.3 \
        man/man3/fi_dupinfo.3 \
        man/man3/fi_enable.3 \
        man/man3/fi_ep_bind.3 \
        man/man3/fi_eq_open.3 \
        man/man3/fi_eq_read.3 \
        man/man3/fi_eq_readerr.3 \
        man/man3/fi_eq_sread.3 \
        man/man3/fi_eq_strerror.3 \
        man/man3/fi_eq_write.3 \
        man/man3/fi_fetch_atomic.3 \
        man/man3/fi_fetch_atomic_valid.3 \
        man/man3/fi_fetch_atomicmsg.3 \
        man/man3/fi_fetch_atomicv.3 \
        man/man3/fi_freeinfo.3 \
        man/man3/fi_getname.3 \
        man/man3/fi_getopt.3 \
        man/man3/fi_getpeer.3 \
        man/man3/fi_inject.3 \
        man/man3/fi_injectdata.3 \
        man/man3/fi_inject_atomic.3 \
        man/man3/fi_inject_write.3 \
        man/man3/fi_inject_writedata.3 \
        man/man3/fi_join.3 \
        man/man3/fi_leave.3 \
        man/man3/fi_listen.3 \
        man/man3/fi_mr_bind.3 \
        man/man3/fi_mr_desc.3 \
        man/man3/fi_mr_key.3 \
        man/man3/fi_mr_reg.3 \
        man/man3/fi_mr_regattr.3 \
        man/man3/fi_mr_regv.3 \
        man/man3/fi_open.3 \
        man/man3/fi_open_ops.3 \
        man/man3/fi_passive_ep.3 \
        man/man3/fi_pep_bind.3 \
        man/man3/fi_poll_add.3 \
        man/man3/fi_poll_del.3 \
        man/man3/fi_poll_open.3 \
        man/man3/fi_read.3 \
        man/man3/fi_readmsg.3 \
        man/man3/fi_readv.3 \
        man/man3/fi_recv.3 \
        man/man3/fi_recvmsg.3 \
        man/man3/fi_recvv.3 \
        man/man3/fi_reject.3 \
        man/man3/fi_rx_addr.3 \
        man/man3/fi_rx_size_left.3 \
        man/man3/fi_scalable_ep_bind.3 \
        man/man3/fi_send.3 \
        man/man3/fi_senddata.3 \
        man/man3/fi_sendmsg.3 \
        man/man3/fi_sendv.3 \
        man/man3/fi_setname.3 \
        man/man3/fi_setopt.3 \
        man/man3/fi_shutdown.3 \
        man/man3/fi_strerror.3 \
        man/man3/fi_tinject.3 \
        man/man3/fi_tinjectdata.3 \
        man/man3/fi_tostr.3 \
        man/man3/fi_trecv.3 \
        man/man3/fi_trecvmsg.3 \
        man/man3/fi_trecvv.3 \
        man/man3/fi_tsend.3 \
        man/man3/fi_tsenddata.3 \
        man/man3/fi_tsendmsg.3 \
        man/man3/fi_tsendv.3 \
        man/man3/fi_tx_size_left.3 \
        man/man3/fi_wait.3 \
        man/man3/fi_wait_open.3 \
        man/man3/fi_write.3 \
        man/man3/fi_writedata.3 \
        man/man3/fi_writemsg.3 \
        man/man3/fi_writev.3

nroff:
	@for file in $(real_man_pages); do \
	    source=`echo $$file | sed -e 's@/man[0-9]@@'`; \
	    config/md2nroff.pl --source=$$source.md; \
	done

dist-hook: libfabric.spec
	cp libfabric.spec $(distdir)
	"$(top_srcdir)/config/distscript.pl" "$(distdir)" "$(PACKAGE_VERSION)"

test:
	./util/fi_info

rpm: dist
	rpmbuild -ta libfabric-*.tar.bz2

prov_install_man_pages=
prov_dist_man_pages=

include prov/sockets/Makefile.include
include prov/verbs/Makefile.include
include prov/usnic/Makefile.include
include prov/psm/Makefile.include
include prov/psm2/Makefile.include
include prov/mxm/Makefile.include
include prov/gni/Makefile.include

man_MANS = $(real_man_pages) $(prov_install_man_pages) $(dummy_man_pages)

EXTRA_DIST = \
        NEWS.md \
        libfabric.map \
        libfabric.spec.in \
        config/distscript.pl \
        $(real_man_pages) $(prov_dist_man_pages) $(dummy_man_pages)
