AM_CPPFLAGS = \
	-I$(srcdir)/include \
	-D_GNU_SOURCE \
	-DSYSCONFDIR=\"$(sysconfdir)\" \
	-DRDMADIR=\"@rdmadir@\" \
	-DEXTDIR=\"$(pkglibdir)\"

lib_LTLIBRARIES = src/libfabric.la

pkglib_LTLIBRARIES = $(DL_PROVIDERS)

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -g -Wall

if HAVE_LD_VERSION_SCRIPT
    libfabric_version_script = -Wl,--version-script=$(srcdir)/libfabric.map
else
    libfabric_version_script =
endif

src_libfabric_la_SOURCES = \
	src/fabric.c \
	src/common.c \
	src/enosys.c \
	src/fi_tostr.c

if HAVE_SOCKETS
_sockets_files = \
	prov/sockets/src/sock_av.c \
	prov/sockets/src/sock_dom.c \
	prov/sockets/src/sock_eq.c \
	prov/sockets/src/sock_poll.c \
	prov/sockets/src/sock_rdm.c \
	prov/sockets/src/sock.c \
	prov/sockets/src/indexer.c

if HAVE_SOCKETS_DL
pkglib_LTLIBRARIES += libsockets-fi.la
libsockets_fi_la_SOURCES = $(_sockets_files)
libsockets_fi_la_LDFLAGS = -module -avoid-version -shared -export-dynamic
else
src_libfabric_la_SOURCES += $(_sockets_files)
endif

endif #HAVE_SOCKETS

if HAVE_VERBS
_verbs_files = prov/verbs/src/fi_verbs.c

if HAVE_VERBS_DL
pkglib_LTLIBRARIES += libibverbs-fi.la
libibverbs_fi_la_SOURCES = $(_verbs_files)
libibverbs_fi_la_LDFLAGS = -module -avoid-version -shared -export-dynamic -libverbs -lrdmacm
else
src_libfabric_la_SOURCES += $(_verbs_files)
endif

endif #HAVE_VERBS

if HAVE_PSM
_psm_files = \
	prov/psm/src/psmx_init.c \
	prov/psm/src/psmx_domain.c \
	prov/psm/src/psmx_cq.c \
	prov/psm/src/psmx_cntr.c \
	prov/psm/src/psmx_av.c \
	prov/psm/src/psmx_ep.c \
	prov/psm/src/psmx_cm.c \
	prov/psm/src/psmx_tagged.c \
	prov/psm/src/psmx_msg.c \
	prov/psm/src/psmx_msg2.c \
	prov/psm/src/psmx_rma.c \
	prov/psm/src/psmx_atomic.c \
	prov/psm/src/psmx_am.c \
	prov/psm/src/psmx_mr.c \
	prov/psm/src/psmx_util.c

if HAVE_PSM_DL
pkglib_LTLIBRARIES += libpsmx-fi.la
libpsmx_fi_la_SOURCES = $(_psm_files)
libpsmx_fi_la_LDFLAGS = -module -avoid-version -shared -export-dynamic -libverbs -lrdmacm
else
src_libfabric_la_SOURCES += $(_psm_files)
endif

endif #HAVE_PSM

src_libfabric_la_LDFLAGS = -version-info 1 -export-dynamic \
			   $(libfabric_version_script)
src_libfabric_la_DEPENDENCIES = $(srcdir)/libfabric.map

rdmaincludedir = $(includedir)/rdma

rdmainclude_HEADERS = \
	$(top_srcdir)/include/rdma/fabric.h \
	$(top_srcdir)/include/rdma/fi_atomic.h \
	$(top_srcdir)/include/rdma/fi_cm.h \
	$(top_srcdir)/include/rdma/fi_domain.h \
	$(top_srcdir)/include/rdma/fi_eq.h \
	$(top_srcdir)/include/rdma/fi_prov.h \
	$(top_srcdir)/include/rdma/fi_rma.h \
	$(top_srcdir)/include/rdma/fi_endpoint.h \
	$(top_srcdir)/include/rdma/fi_errno.h \
	$(top_srcdir)/include/rdma/fi_tagged.h \
	$(top_srcdir)/include/rdma/fi_trigger.h

man_MANS = \
	man/fabric.7 \
	man/fi_accept.3 \
	man/fi_alias.3 \
	man/fi_atomic.3 \
	man/fi_atomicv.3 \
	man/fi_atomicto.3 \
	man/fi_atomicmsg.3 \
	man/fi_atomic_valid.3 \
	man/fi_av.3 \
	man/fi_av_bind.3 \
	man/fi_av_insert.3 \
	man/fi_av_insertsvc.3 \
	man/fi_av_lookup.3 \
	man/fi_av_open.3 \
	man/fi_av_remove.3 \
	man/fi_av_straddr.3 \
	man/fi_cancel.3 \
	man/fi_close.3 \
	man/fi_cm.3 \
	man/fi_compare_atomic.3 \
	man/fi_compare_atomicv.3 \
	man/fi_compare_atomicto.3 \
	man/fi_compare_atomicmsg.3 \
	man/fi_compare_atomic_valid.3 \
	man/fi_connect.3 \
	man/fi_cntr.3 \
	man/fi_cntr_open.3 \
	man/fi_cntr_read.3 \
	man/fi_cntr_add.3 \
	man/fi_cntr_set.3 \
	man/fi_cntr_wait.3 \
	man/fi_cq.3 \
	man/fi_cq_open.3 \
	man/fi_cq_read.3 \
	man/fi_cq_readfrom.3 \
	man/fi_cq_readerr.3 \
	man/fi_cq_write.3 \
	man/fi_cq_sread.3 \
	man/fi_cq_sreadfrom.3 \
	man/fi_cq_strerror.3 \
	man/fi_direct.7 \
	man/fi_domain.3 \
	man/fi_domain_query.3 \
	man/fi_domain_bind.3 \
	man/fi_ep_bind.3 \
	man/fi_ep_sync.3 \
	man/fi_eq.3 \
	man/fi_eq_open.3 \
	man/fi_eq_read.3 \
	man/fi_eq_readerr.3 \
	man/fi_eq_write.3 \
	man/fi_eq_sread.3 \
	man/fi_eq_strerror.3 \
	man/fi_enable.3 \
	man/fi_endpoint.3 \
	man/fi_fabric.3 \
	man/fi_fetch_atomic.3 \
	man/fi_fetch_atomicv.3 \
	man/fi_fetch_atomicto.3 \
	man/fi_fetch_atomicmsg.3 \
	man/fi_fetch_atomic_valid.3 \
	man/fi_freeinfo.3 \
	man/fi_getinfo.3 \
	man/fi_getname.3 \
	man/fi_getopt.3 \
	man/fi_getpeer.3 \
	man/fi_inject.3 \
	man/fi_injectto.3 \
	man/fi_join.3 \
	man/fi_leave.3 \
	man/fi_listen.3 \
	man/fi_mr.3 \
	man/fi_mr_reg.3 \
	man/fi_mr_regv.3 \
	man/fi_mr_regattr.3 \
	man/fi_mr_desc.3 \
	man/fi_mr_key.3 \
	man/fi_mr_bind.3 \
	man/fi_msg.3 \
	man/fi_open.3 \
	man/fi_pendpoint.3 \
	man/fi_poll.3 \
	man/fi_poll_add.3 \
	man/fi_poll_del.3 \
	man/fi_poll_open.3 \
	man/fi_recv.3 \
	man/fi_recvv.3 \
	man/fi_recvfrom.3 \
	man/fi_recvmsg.3 \
	man/fi_reject.3 \
	man/fi_rma.3 \
	man/fi_rx_addr.3 \
	man/fi_send.3 \
	man/fi_senddata.3 \
	man/fi_senddatato.3 \
	man/fi_sendv.3 \
	man/fi_sendto.3 \
	man/fi_sendmsg.3 \
	man/fi_setopt.3 \
	man/fi_shutdown.3 \
	man/fi_tagged.3 \
	man/fi_tinject.3 \
	man/fi_tinjectto.3 \
	man/fi_trecv.3 \
	man/fi_trecvv.3 \
	man/fi_trecvfrom.3 \
	man/fi_trecvmsg.3 \
	man/fi_trigger.3 \
	man/fi_tsearch.3 \
	man/fi_tsend.3 \
	man/fi_tsenddata.3 \
	man/fi_tsenddatato.3 \
	man/fi_tsendv.3 \
	man/fi_tsendto.3 \
	man/fi_tsendmsg.3 \
	man/fi_version.3 \
	man/fi_wait.3 \
	man/fi_wait_open.3

EXTRA_DIST = libfabric.map libfabric.spec.in $(man_MANS) \
	include/fi.h \
	include/fi_enosys.h \
	include/indexer.h \
	prov/sockets/src/sock.h \
	prov/psm/src/psmx.h

dist-hook: libfabric.spec
	cp libfabric.spec $(distdir)
