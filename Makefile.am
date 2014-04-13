AM_CPPFLAGS = -I$(srcdir)/include -I$(srcdir)/prov/rdmacm/include

lib_LTLIBRARIES = libibverbs/libibverbs.la src/libfabric.la

ACLOCAL_AMFLAGS = -I config
AM_CFLAGS = -g -Wall -D_GNU_SOURCE 

libibverbs_libibverbs_la_CFLAGS = $(AM_CFLAGS) -DIBV_CONFIG_DIR=\"$(sysconfdir)/libibverbs.d\"
src_libfabric_la_CFLAGS = $(AM_CFLAGS) -DSYSCONFDIR=\"$(sysconfdir)\" -DRDMADIR=\"@rdmadir@\"

if HAVE_LD_VERSION_SCRIPT
    libibverbs_version_script = -Wl,--version-script=$(srcdir)/libibverbs/src/libibverbs.map
    libfabric_version_script = -Wl,--version-script=$(srcdir)/libfabric.map
else
    libibverbs_version_script =
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
libibverbs_libibverbs_la_DEPENDENCIES =\
	$(srcdir)/libibverbs/src/libibverbs.map

src_libfabric_la_SOURCES = src/fabric.c \
	src/ucma.c \
	prov/ibverbs/fi_verbs.c \
	prov/rdmacm/src/acm.c \
	prov/rdmacm/src/addrinfo.c \
	prov/rdmacm/src/cma.c \
	prov/rdmacm/src/indexer.c \
	prov/rdmacm/src/rsocket.c

if HAVE_PSM
src_libfabric_la_SOURCES += prov/psm/src/psmx_init.c \
	prov/psm/src/psmx_domain.c \
	prov/psm/src/psmx_ec.c \
	prov/psm/src/psmx_av.c \
	prov/psm/src/psmx_ep.c \
	prov/psm/src/psmx_cm.c \
	prov/psm/src/psmx_tagged.c \
	prov/psm/src/psmx_msg.c \
	prov/psm/src/psmx_rdma.c \
	prov/psm/src/psmx_util.c
endif

src_libfabric_la_LDFLAGS = -version-info 1 -export-dynamic \
			   $(libfabric_version_script)
src_libfabric_la_DEPENDENCIES =  $(srcdir)/libfabric.map
src_libfabric_la_LIBADD = $(top_builddir)/libibverbs/libibverbs.la

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
			$(top_srcdir)/include/rdma/rdma_verbs.h \
			$(top_srcdir)/include/rdma/rsocket.h

infinibandinclude_HEADERS = \
			$(top_srcdir)/include/infiniband/arch.h \
			$(top_srcdir)/include/infiniband/driver.h \
			$(top_srcdir)/include/infiniband/kern-abi.h \
			$(top_srcdir)/include/infiniband/marshall.h \
			$(top_srcdir)/include/infiniband/opcode.h \
			$(top_srcdir)/include/infiniband/sa.h \
			$(top_srcdir)/include/infiniband/sa-kern-abi.h \
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
	man/fi_rma.3 \
	man/fi_tagged.3

EXTRA_DIST = include/fi.h libfabric.map libfabric.spec.in $(man_MANS) \
	prov/ibverbs/src/ibverbs.h \
	prov/rdmacm/src/cma.h \
	prov/rdmacm/src/indexer.h \
	prov/mlx4/src/doorbell.h \
	prov/mlx4/src/mlx4.h \
	prov/mlx4/src/mlx4-abi.h \
	prov/mlx4/wqe.h \
	libibverbs/src/ibverbs.h

dist-hook: libfabric.spec
	cp libfabric.spec $(distdir)

install-data-hook:
	cd $(DESTDIR)$(mandir)/man3 && \
	$(RM) fi_freeinfo.3 && \
	$(LN_S) fi_getinfo.3 fi_freeinfo.3
