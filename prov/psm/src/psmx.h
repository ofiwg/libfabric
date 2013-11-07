#ifndef _FI_PSM_H
#define _FI_PSM_H

#ifdef __cplusplus
extern "C" {
#endif

#if HAVE_CONFIG_H
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <rdma/fabric.h>
#include <rdma/fi_prov.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_socket.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <psm.h>
#include <psm_mq.h>

#define PFX "libfabric:psm"

#define PSMX_TIME_OUT	120

struct psmx_fid_domain {
	struct fid_domain	domain;
	psm_ep_t		psm_ep;
	psm_epid_t		psm_epid;
	psm_mq_t		psm_mq;
	pthread_t		ns_thread;
	int			ns_port;
};

struct psmx_fid_ec {
	struct fid_ec		ec;
	struct psmx_fid_domain	*domain;
	int			type;
	int 			format;
};

struct psmx_fid_av {
	struct fid_av		av;
	struct psmx_fid_domain	*domain;
	int			type;
	int			format;
	size_t			addrlen;
};

struct psmx_fid_socket {
	struct fid_socket	socket;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_ec	*ec;
	struct psmx_fid_av	*av;
	uint64_t		flags;
};

extern struct fi_ops_cm		psmx_cm_ops;
extern struct fi_ops_tagged	psmx_tagged_ops;

void	psmx_ini(void);
void	psmx_fini(void);

int	psmx_domain_open(const char *name, struct fi_info *info, uint64_t flags,
			 fid_t *fid, void *context);
int	psmx_sock_open(struct fi_info *info, fid_t *fid, void *context);
int	psmx_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context);
int	psmx_av_open(fid_t fid, struct fi_av_attr *attr, fid_t *av, void *context);

void 	*psmx_name_server(void *args);
void	*psmx_resolve_name(char *servername, psm_uuid_t uuid);
void	psmx_string_to_uuid(char *s, psm_uuid_t uuid);
int	psmx_uuid_to_port(psm_uuid_t uuid);
int	psmx_errno(int err);

#ifdef __cplusplus
}
#endif

#endif

