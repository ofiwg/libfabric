AM_CPPFLAGS = -I$(srcdir)/include

lib_LTLIBRARIES = libibverbs/libibverbs.la \
	librdmacm/librdmacm.la \
	src/libfabric.la

rslibdir = $(libdir)/rsocket
rslib_LTLIBRARIES = librdmacm/librspreload.la

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -g -Wall -D_GNU_SOURCE 

libibverbs_libibverbs_la_CFLAGS = $(AM_CFLAGS) -DIBV_CONFIG_DIR=\"$(sysconfdir)/libibverbs.d\"
librdmacm_librdmacm_la_CFLAGS = $(AM_CFLAGS) -DSYSCONFDIR=\"$(sysconfdir)\" -DRDMADIR=\"@rdmadir@\"
librdmacm_librspreload_la_CFLAGS = $(AM_CFLAGS) -DSYSCONFDIR=\"$(sysconfdir)\" -DRDMADIR=\"@rdmadir@\"
src_libfabric_la_CFLAGS = $(AM_CFLAGS) -DSYSCONFDIR=\"$(sysconfdir)\" -DRDMADIR=\"@rdmadir@\"

if HAVE_LD_VERSION_SCRIPT
    libibverbs_version_script = -Wl,--version-script=$(srcdir)/libibverbs/src/libibverbs.map
    librdmacm_version_script = -Wl,--version-script=$(srcdir)/librdmacm/src/librdmacm.map
    libfabric_version_script = -Wl,--version-script=$(srcdir)/libfabric.map
else
    libibverbs_version_script =
    librdmacm_version_script =
    libfabric_version_script =
endif

libibverbs_libibverbs_la_SOURCES = \
	libibverbs/src/cmd.c \
	libibverbs/src/compat-1_0.c \
	libibverbs/src/device.c \
	libibverbs/src/init.c \
	libibverbs/src/marshall.c \
	libibverbs/src/memory.c \
	libibverbs/src/sysfs.c \
	libibverbs/src/verbs.c \
	libibverbs/src/enum_strs.c
libibverbs_libibverbs_la_LDFLAGS = -version-info 1 -export-dynamic \
				   $(libibverbs_version_script)
libibverbs_libibverbs_la_DEPENDENCIES = \
	$(srcdir)/libibverbs/src/libibverbs.map

librdmacm_librdmacm_la_SOURCES = \
	librdmacm/src/cma.c \
	librdmacm/src/addrinfo.c \
	librdmacm/src/acm.c \
	librdmacm/src/rsocket.c \
	librdmacm/src/indexer.c
librdmacm_librdmacm_la_LDFLAGS = -version-info 1 -export-dynamic \
                           $(librdmacm_version_script)
librdmacm_librdmacm_la_DEPENDENCIES = \
	$(srcdir)/librdmacm/src/librdmacm.map \
	$(top_builddir)/libibverbs/libibverbs.la
librdmacm_librdmacm_la_LIBADD = \
	$(top_builddir)/libibverbs/libibverbs.la

librdmacm_librspreload_la_SOURCES = \
	librdmacm/src/preload.c \
	librdmacm/src/indexer.c
librdmacm_librspreload_la_LDFLAGS = -version-info 1 -export-dynamic
librdmacm_librspreload_la_LIBADD = \
	$(top_builddir)/librdmacm/librdmacm.la
librdmacm_librspreload_la_DEPENDENCIES = \
	$(top_builddir)/librdmacm/librdmacm.la

src_libfabric_la_SOURCES = src/fabric.c \
	prov/ibverbs/src/fi_verbs.c \
	prov/sockets/src/sock_av.c \
	prov/sockets/src/sock_dom.c \
	prov/sockets/src/sock_eq.c \
	prov/sockets/src/sock_poll.c \
	prov/sockets/src/sock_rdm.c \
	prov/sockets/src/sock.c \
	prov/sockets/src/indexer.c

if HAVE_PSM
src_libfabric_la_SOURCES += prov/psm/src/psmx_init.c \
	prov/psm/src/psmx_domain.c \
	prov/psm/src/psmx_eq.c \
	prov/psm/src/psmx_cntr.c \
	prov/psm/src/psmx_av.c \
	prov/psm/src/psmx_ep.c \
	prov/psm/src/psmx_cm.c \
	prov/psm/src/psmx_tagged.c \
	prov/psm/src/psmx_msg.c \
	prov/psm/src/psmx_mr.c \
	prov/psm/src/psmx_util.c
endif

src_libfabric_la_LDFLAGS = -version-info 1 -export-dynamic \
			   $(libfabric_version_script)
src_libfabric_la_DEPENDENCIES = \
	$(srcdir)/libfabric.map \
	$(top_builddir)/librdmacm/librdmacm.la \
	$(top_builddir)/libibverbs/libibverbs.la
src_libfabric_la_LIBADD = \
	$(top_builddir)/librdmacm/librdmacm.la \
	$(top_builddir)/libibverbs/libibverbs.la

rdmaincludedir = $(includedir)/rdma
infinibandincludedir = $(includedir)/infiniband

rdmainclude_HEADERS =	$(top_srcdir)/include/rdma/fabric.h \
			$(top_srcdir)/include/rdma/fi_atomic.h \
			$(top_srcdir)/include/rdma/fi_cm.h \
			$(top_srcdir)/include/rdma/fi_domain.h \
			$(top_srcdir)/include/rdma/fi_prov.h \
			$(top_srcdir)/include/rdma/fi_rma.h \
			$(top_srcdir)/include/rdma/fi_endpoint.h \
			$(top_srcdir)/include/rdma/fi_errno.h \
			$(top_srcdir)/include/rdma/fi_tagged.h \
			$(top_srcdir)/include/rdma/fi_ucma.h \
			$(top_srcdir)/include/rdma/fi_uverbs.h \
			$(top_srcdir)/include/rdma/rdma_cma.h \
			$(top_srcdir)/include/rdma/rdma_cma_abi.h \
			$(top_srcdir)/include/rdma/rdma_verbs.h \
			$(top_srcdir)/include/rdma/rsocket.h

infinibandinclude_HEADERS = \
			$(top_srcdir)/include/infiniband/arch.h \
			$(top_srcdir)/include/infiniband/driver.h \
			$(top_srcdir)/include/infiniband/ib.h \
			$(top_srcdir)/include/infiniband/kern-abi.h \
			$(top_srcdir)/include/infiniband/marshall.h \
			$(top_srcdir)/include/infiniband/opcode.h \
			$(top_srcdir)/include/infiniband/sa.h \
			$(top_srcdir)/include/infiniband/sa-kern-abi.h \
			$(top_srcdir)/include/infiniband/verbs.h

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
	man/fi_av_close.3 \
	man/fi_av_insert.3 \
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
	man/fi_cntr_close.3 \
	man/fi_cntr_read.3 \
	man/fi_cntr_add.3 \
	man/fi_cntr_set.3 \
	man/fi_cntr_wait.3 \
	man/fi_direct.7 \
	man/fi_domain.3 \
	man/fi_domain_query.3 \
	man/fi_domain_bind.3 \
	man/fi_ep_bind.3 \
	man/fi_ep_control.3 \
	man/fi_ep_sync.3 \
	man/fi_eq.3 \
	man/fi_eq_open.3 \
	man/fi_eq_close.3 \
	man/fi_eq_control.3 \
	man/fi_eq_read.3 \
	man/fi_eq_readfrom.3 \
	man/fi_eq_readerr.3 \
	man/fi_eq_write.3 \
	man/fi_eq_condread.3 \
	man/fi_eq_condreadfrom.3 \
	man/fi_eq_strerror.3 \
	man/fi_enable.3 \
	man/fi_endpoint.3 \
	man/fi_fabric.3 \
	man/fi_feq_open.3 \
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
	man/fi_mr_unreg.3 \
	man/fi_mr_desc.3 \
	man/fi_mr_key.3 \
	man/fi_mr_bind.3 \
	man/fi_msg.3 \
	man/fi_open.3 \
	man/fi_pendpoint.3 \
	man/fi_poll.3 \
	man/fi_poll_add.3 \
	man/fi_poll_close.3 \
	man/fi_poll_del.3 \
	man/fi_poll_open.3 \
	man/fi_recv.3 \
	man/fi_recvv.3 \
	man/fi_recvfrom.3 \
	man/fi_recvmsg.3 \
	man/fi_reject.3 \
	man/fi_rma.3 \
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
	man/fi_wait.3 \
	man/fi_wait_close.3 \
	man/fi_wait_control.3 \
	man/fi_wait_open.3

EXTRA_DIST = include/fi.h libfabric.map libfabric.spec.in $(man_MANS) \
	librdmacm/src/cma.h \
	librdmacm/src/indexer.h \
	librdmacm/src/librdmacm.map \
	libibverbs/src/ibverbs.h \
	libibverbs/src/libibverbs.map \
	prov/sockets/src/sock.h \
	prov/psm/src/psmx.h

dist-hook: libfabric.spec
	cp libfabric.spec $(distdir)
