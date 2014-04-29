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
#include <rdma/fi_endpoint.h>
#include <rdma/fi_tagged.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_errno.h>
#include <psm.h>
#include <psm_mq.h>

#define PFX "libfabric:psm"

#define PSMX_TIME_OUT	120
#define PSMX_SUPPORTED_FLAGS (FI_BLOCK | FI_EXCL | \
			      FI_BUFFERED_SEND | FI_BUFFERED_RECV | \
			      FI_NOCOMP | FI_SIGNAL | FI_ACK | FI_CANCEL)
#define PSMX_DEFAULT_FLAGS   (0)
#define PSMX_PROTO_CAPS	     (FI_PROTO_CAP_TAGGED | FI_PROTO_CAP_MSG | \
			      FI_PROTO_CAP_RMA)

#define PSMX_OUI_INTEL	0x0002b3L
#define PSMX_PROTOCOL	0x0001

#define PSMX_NONMATCH_BIT (0x8000000000000000ULL)
#define PSMX_NOCOMP_CONTEXT ((void *)0xFFFF0000FFFF0000ULL)

struct psmx_fid_domain {
	struct fid_domain	domain;
	psm_ep_t		psm_ep;
	psm_epid_t		psm_epid;
	psm_mq_t		psm_mq;
	pthread_t		ns_thread;
	int			ns_port;

	/* certain bits in the tag space can be reserved for non tag-matching
	 * purpose. The tag-matching functions automatically treat these bits
	 * as 0. This field is a bit mask, with reserved bits valued as "1".
	 */
	uint64_t		reserved_tag_bits;
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

struct psmx_fid_ep {
	struct fid_ep		ep;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_ec	*ec;
	struct psmx_fid_av	*av;
	uint64_t		flags;
	psm_epid_t		peer_psm_epid;
	psm_epaddr_t		peer_psm_epaddr;
	int			connected;
};

struct psmx_fid_mr {
	struct fid_mr		mr;
};

extern struct fi_ops_cm		psmx_cm_ops;
extern struct fi_ops_tagged	psmx_tagged_ops;
extern struct fi_ops_msg	psmx_msg_ops;
extern struct fi_ops_rma	psmx_rma_ops;

void	psmx_ini(void);
void	psmx_fini(void);

int	psmx_domain_open(struct fi_info *info, fid_t *fid, void *context);
int	psmx_ep_open(struct fi_info *info, fid_t *fid, void *context);
int	psmx_ec_open(fid_t fid, struct fi_ec_attr *attr, fid_t *ec, void *context);
int	psmx_av_open(fid_t fid, struct fi_av_attr *attr, fid_t *av, void *context);

int	psmx_mr_reg(fid_t fid, const void *buf, size_t len,
		       struct fi_mr_attr *attr, fid_t *mr, void *context);
int	psmx_mr_regv(fid_t fid, const struct iovec *iov, size_t count,
			struct fi_mr_attr *attr, fid_t *mr, void *context);

void 	*psmx_name_server(void *args);
void	*psmx_resolve_name(const char *servername, psm_uuid_t uuid);
void	psmx_string_to_uuid(const char *s, psm_uuid_t uuid);
int	psmx_uuid_to_port(psm_uuid_t uuid);
int	psmx_errno(int err);
int	psmx_epid_to_epaddr(psm_ep_t ep, psm_epid_t epid, psm_epaddr_t *epaddr);

#ifdef __cplusplus
}
#endif

#endif

