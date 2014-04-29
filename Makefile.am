AM_CPPFLAGS = -I$(srcdir)/include -I$(srcdir)/prov/ibverbs/include \
	      -I$(srcdir)/prov/rdmacm/include

lib_LTLIBRARIES = src/libfabric.la

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -g -Wall -D_GNU_SOURCE -Wno-deprecated-declarations

src_libfabric_la_CFLAGS = $(AM_CFLAGS) -DSYSCONFDIR=\"$(sysconfdir)\" -DRDMADIR=\"@rdmadir@\"

if HAVE_LD_VERSION_SCRIPT
    libfabric_version_script = -Wl,--version-script=$(srcdir)/libfabric.map
else
    libfabric_version_script =
endif

src_libfabric_la_SOURCES = src/fabric.c src/uverbs.c src/ucma.c \
	prov/ibverbs/src/cmd.c \
	prov/ibverbs/src/device.c \
	prov/ibverbs/src/enum_strs.c \
	prov/ibverbs/src/fi_verbs.c \
	prov/ibverbs/src/init.c \
	prov/ibverbs/src/marshall.c \
	prov/ibverbs/src/memory.c \
	prov/ibverbs/src/verbs.c \
	prov/rdmacm/src/acm.c \
	prov/rdmacm/src/addrinfo.c \
	prov/rdmacm/src/cma.c \
	prov/rdmacm/src/indexer.c \
	prov/rdmacm/src/rsocket.c \
	prov/mlx4/src/buf.c \
	prov/mlx4/src/cq.c \
	prov/mlx4/src/dbrec.c \
	prov/mlx4/src/mlx4.c \
	prov/mlx4/src/qp.c \
	prov/mlx4/src/srq.c \
	prov/mlx4/src/mlx4_verbs.c

if HAVE_PSM
src_libfabric_la_SOURCES += prov/psm/src/psmx_init.c \
	prov/psm/src/psmx_domain.c \
	prov/psm/src/psmx_ec.c \
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

src_libfabric_la_DEPENDENCIES =  $(srcdir)/libfabric.map

bin_PROGRAMS = \
	examples/fi_provinfo \
	examples/fi_perf

examples_fi_provinfo_SOURCES = \
	examples/provinfo.c \
	examples/shared.c
examples_fi_provinfo_LDADD = \
	$(top_builddir)/src/libfabric.la
examples_fi_perf_SOURCES = \
	examples/perf.c \
	examples/shared.c
examples_fi_perf_LDADD = \
	$(top_builddir)/src/libfabric.la

libfabricincludedir = $(includedir)/rdma
infinibandincludedir = $(includedir)/infiniband

libfabricinclude_HEADERS = $(top_srcdir)/include/rdma/fabric.h \
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
			   $(top_srcdir)/include/rdma/rdma_verbs.h \
			   $(top_srcdir)/include/rdma/rsocket.h

infinibandinclude_HEADERS = $(top_srcdir)/include/infiniband/ib.h \
			   $(top_srcdir)/include/infiniband/verbs.h

man_MANS = \
	man/fabric.7 \
	man/fi_atomic.3 \
	man/fi_av.3 \
	man/fi_cm.3 \
	man/fi_direct.7 \
	man/fi_domain.3 \
	man/fi_ec.3 \
	man/fi_endpoint.3 \
	man/fi_fabric.3 \
	man/fi_getinfo.3 \
	man/fi_mr.3 \
	man/fi_msg.3 \
	man/fi_open.3 \
	man/fi_rma.3 \
	man/fi_tagged.3

EXTRA_DIST = include/fi.h libfabric.map libfabric.spec.in $(man_MANS) \
	prov/ibverbs/include/infiniband/driver.h \
	prov/ibverbs/include/infiniband/marshall.h \
	prov/ibverbs/include/infiniband/opcode.h \
	prov/ibverbs/src/ibverbs.h \
	prov/rdmacm/src/cma.h \
	prov/rdmacm/src/indexer.h \
	prov/mlx4/src/doorbell.h \
	prov/mlx4/src/mlx4.h \
	prov/mlx4/src/mlx4-abi.h \
	prov/mlx4/wqe.h \
	examples/shared.h

dist-hook: libfabric.spec
	cp libfabric.spec $(distdir)

install-data-hook:
	cd $(DESTDIR)$(mandir)/man3 && \
	$(RM) fi_freeinfo.3 && \
	$(RM) fi_close.3 && \
	$(LN_S) fi_getinfo.3 fi_freeinfo.3 && \
	$(LN_S) fi_open.3 fi_close.3
