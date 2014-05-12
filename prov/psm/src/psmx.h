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
#include <stdarg.h>
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

#ifndef MIN
#define MIN(x,y) ((x)<(y)?(x):(y))
#endif

#define PSMX_MR_SIGNATURE (0x05F109530F1D03B0ULL) /* "SFI PSM FID MR" */
#define PSMX_TIME_OUT	120
#define PSMX_SUPPORTED_FLAGS (FI_BLOCK | FI_EXCL | \
			      FI_READ | FI_WRITE | FI_RECV | FI_SEND | \
			      FI_REMOTE_READ | FI_REMOTE_WRITE | \
			      FI_BUFFERED_SEND | FI_BUFFERED_RECV | \
			      FI_MULTI_RECV | FI_REMOTE_COMPLETE | \
			      FI_EVENT | FI_REMOTE_SIGNAL | FI_CANCEL)
#define PSMX_DEFAULT_FLAGS   (0)

#define PSMX_PROTO_CAPS	     (FI_PROTO_CAP_TAGGED | FI_PROTO_CAP_MSG)

#define PSMX_OUI_INTEL	0x0002b3L
#define PSMX_PROTOCOL	0x0001

#define PSMX_MSG_BIT		(0x1ULL << 63)

enum psmx_context_type {
	PSMX_NOCOMP_CONTEXT = 1,
	PSMX_SEND_CONTEXT,
	PSMX_RECV_CONTEXT,
	PSMX_WRITE_CONTEXT,
	PSMX_READ_CONTEXT,
	PSMX_SENDIMM_CONTEXT,
	PSMX_WRITEIMM_CONTEXT,
	PSMX_ATOMIC_CONTEXT,
};

#define PSMX_CTXT_REQ(fi_context)	((fi_context)->internal[0])
#define PSMX_CTXT_TYPE(fi_context)	(*(int *)&(fi_context)->internal[1])
#define PSMX_CTXT_USER(fi_context)	((fi_context)->internal[2])
#define PSMX_CTXT_EP(fi_context)	((fi_context)->internal[3])

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

struct psmx_event {
	union {
		struct fi_eq_entry		context;
		struct fi_eq_comp_entry		comp;
		struct fi_eq_data_entry		data;
		struct fi_eq_tagged_entry	tagged;
		struct fi_eq_err_entry		err;
		struct fi_eq_cm_entry		cm;
	} eqe;
	int error;
	uint64_t source;
	struct psmx_event *next;
};

struct psmx_event_queue {
	struct psmx_event	*head;
	struct psmx_event	*tail;
};

struct psmx_fid_eq {
	struct fid_eq		eq;
	struct psmx_fid_domain	*domain;
	int 			format;
	int			entry_size;
	struct psmx_event_queue	event_queue;
	struct psmx_event	*pending_error;
};

struct psmx_fid_cntr {
	struct fid_cntr		cntr;
	struct psmx_fid_domain	*domain;
	int			events;
	int			wait_obj;
	uint64_t		flags;
	volatile uint64_t	counter;
	pthread_mutex_t		mutex;
	pthread_cond_t		cond;
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
	struct psmx_fid_eq	*eq;
	struct psmx_fid_cntr	*cntr;
	struct psmx_fid_av	*av;
	uint64_t		flags;
	int			connected;
	psm_epid_t		peer_psm_epid;
	psm_epaddr_t		peer_psm_epaddr;
	struct fi_context	sendimm_context;
	struct fi_context	writeimm_context;
	uint64_t		pending_sends;
	uint64_t		pending_writes;
	uint64_t		pending_reads;
	uint64_t		pending_atomics;
};

struct psmx_fid_mr {
	struct fid_mr		mr;
	struct psmx_fid_domain	*domain;
	struct psmx_fid_ep	*ep;
	struct psmx_fid_eq	*eq;
	uint64_t		signature;
	uint64_t		access;
	uint64_t		flags;
	size_t			iov_count;
	struct iovec		iov[0];	/* must be the last field */
};

extern struct fi_ops_mr		psmx_mr_ops;
extern struct fi_ops_cm		psmx_cm_ops;
extern struct fi_ops_tagged	psmx_tagged_ops;
extern struct fi_ops_msg	psmx_msg_ops;

void	psmx_ini(void);
void	psmx_fini(void);

int	psmx_domain_open(struct fid_fabric *fabric, struct fi_info *info,
			 struct fid_domain **fid, void *context);
int	psmx_ep_open(struct fid_domain *domain, struct fi_info *info,
		     struct fid_ep **fid, void *context);
int	psmx_eq_open(struct fid_domain *domain, struct fi_eq_attr *attr,
		     struct fid_eq **eq, void *context);
int	psmx_av_open(struct fid_domain *domain, struct fi_av_attr *attr,
		     struct fid_av **av, void *context);
int	psmx_cntr_alloc(struct fid_domain *domain, struct fi_cntr_attr *attr,
			struct fid_cntr **cntr, void *context);

void 	*psmx_name_server(void *args);
void	*psmx_resolve_name(const char *servername, psm_uuid_t uuid);
void	psmx_string_to_uuid(const char *s, psm_uuid_t uuid);
int	psmx_uuid_to_port(psm_uuid_t uuid);
int	psmx_errno(int err);
int	psmx_epid_to_epaddr(psm_ep_t ep, psm_epid_t epid, psm_epaddr_t *epaddr);
void	psmx_query_mpi(void);
void	psmx_debug(char *fmt, ...);

void	psmx_eq_enqueue_event(struct psmx_fid_eq *fid_eq, struct psmx_event *event);
struct	psmx_event *psmx_eq_create_event(struct psmx_fid_eq *fid_eq,
					void *op_context, void *buf,
					uint64_t flags, size_t len,
					uint64_t data, uint64_t tag,
					size_t olen, int err);
int	psmx_eq_poll_mq(struct psmx_fid_eq *eq,
			struct psmx_fid_domain *domain_if_null_eq);

#ifdef __cplusplus
}
#endif

#endif

