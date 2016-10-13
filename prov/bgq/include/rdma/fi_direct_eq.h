#ifndef _FI_BGQ_DIRECT_EQ_H_
#define _FI_BGQ_DIRECT_EQ_H_

#define FABRIC_DIRECT_EQ 1

#include <unistd.h>
#include <stdint.h>

#include "rdma/bgq/fi_bgq_hwi.h"

#include "rdma/bgq/fi_bgq_l2atomic.h"
#include "rdma/bgq/fi_bgq_flight_recorder.h"
#include "rdma/bgq/fi_bgq_mu.h"

#ifdef __cplusplus
extern "C" {
#endif

struct fi_bgq_cntr {
	struct fid_cntr		cntr_fid;

	struct {
		volatile uint64_t	*l2_vaddr;
		uint64_t		paddr;
		uint64_t		batid;
	} std;
	struct {
		volatile uint64_t	*l2_vaddr;
		uint64_t		paddr;
		uint64_t		batid;
	} err;

	volatile uint64_t	data[2];

	struct {
		uint64_t		ep_count;
		struct fi_bgq_ep	*ep[64];	/* TODO - check this array size */
	} progress;

	uint64_t		ep_bind_count;
	struct fi_bgq_ep	*ep[64];	/* TODO - check this array size */

	struct fi_cntr_attr	*attr;
	struct fi_bgq_domain	*domain;
};

#define FI_BGQ_CQ_CONTEXT_EXT		(0x8000000000000000ull)
#define FI_BGQ_CQ_CONTEXT_MULTIRECV	(0x4000000000000000ull)


union fi_bgq_context {
	struct fi_context		context;
	struct {
		union fi_bgq_context	*next;		// fi_cq_entry::op_context
		uint64_t		flags;		// fi_cq_msg_entry::flags
		size_t			len;		// fi_cq_msg_entry::len (only need 37 bits)
		void			*buf;		// fi_cq_data_entry::buf (unused for tagged cq's and non-multi-receive message cq's)

		uint64_t		byte_counter;	// fi_cq_data_entry::data
		union {
			uint64_t	tag;		// fi_cq_tagged_entry::tag
			union fi_bgq_context	*multi_recv_next;	// only for multi-receives
		};
		union {
			uint64_t		ignore;	// only for tagged receive
			struct fi_bgq_mu_packet	*claim;	// only for peek/claim
			void			*multi_recv_context;	// only for individual FI_MULTI_RECV's
		};
		union fi_bgq_context	*prev;
	};
};

struct fi_bgq_context_ext {
	union fi_bgq_context		bgq_context;
	struct fi_cq_err_entry		err_entry;
	struct {
		struct fi_context	*op_context;
		size_t			iov_count;
		struct iovec		*iov;
	} msg;
};


#if 0
#define	CQ_ENTRY_KIND_DEFAULT (0)
#define CQ_ENTRY_KIND_MSG (1)
union fi_cq_bgq_entry {
	struct fi_cq_entry		context;
	struct fi_cq_msg_entry		msg;
	struct fi_cq_data_entry		data;
	struct fi_cq_tagged_entry	tagged;
	struct fi_cq_err_entry		err;

	struct {
		void *			op_context;	// fi_cq_entry::op_context; a.k.a "union fi_bgq_context *"
		union {
			uint64_t	unused_1;	// fi_cq_msg_entry::flags;
			uint64_t	flags;		// for CQ_ENTRY_KIND_MSG
		};
		size_t			len;		// fi_cq_msg_entry::len; for CQ_ENTRY_KIND_DEFAULT
		union {
			void *		buf;		// fi_cq_data_entry::buf; for CQ_ENTRY_KIND_DEFAULT
			const struct iovec *msg_iov;	// for CQ_ENTRY_KIND_MSG
		};
		uint64_t		ignore;		// fi_cq_data_entry::data;
		uint64_t		tag;		// fi_cq_tagged_entry::tag
		union {
			size_t		unused_6;	// fi_cq_err_entry::olen;
			size_t		iov_count;	// for CQ_ENTRY_KIND_MSG
		};
		int			unused_7;	// fi_cq_err_entry::err;
		int			unused_8;	// fi_cq_err_entry::prov_errno;
		void *			unused_9;	// fi_cq_err_entry::err_data;
		uint16_t		entry_id;
		uint16_t		entry_kind;

		struct {
			volatile uint64_t	value;
			uint64_t		muhwi_paddr_rsh3b;	// paddr right-shifted 3 bits
		} byte_counter;
	} recv;
};
#endif

struct fi_bgq_cq {
	struct fid_cq			cq_fid;		/* must be the first field in the structure */

	struct l2atomic_fifo_consumer	err_consumer;

	struct l2atomic_lock		lock;
	union fi_bgq_context		*pending_head;
	union fi_bgq_context		*pending_tail;

	struct l2atomic_fifo_consumer	std_consumer;



	struct l2atomic_fifo_producer	err_producer;
	struct l2atomic_fifo_producer	std_producer;



	struct fi_bgq_domain	*domain;
	uint64_t		bflags;		/* fi_bgq_bind_ep_cq() */
	size_t			size;
	enum fi_cq_format	format;

	MUHWI_Descriptor_t	local_completion_model;

//	union fi_cq_bgq_entry		*entry;
//	uint64_t			entry_mask;
//	struct l2atomic_boundedcounter	entry_counter;

//	struct l2atomic_fifo		err_fifo;
//	struct l2atomic_fifo		std_fifo;

	struct {
		uint64_t		ep_count;
		struct fi_bgq_ep	*ep[64];	/* TODO - check this array size */
	} progress;

	uint64_t		ep_bind_count;
	struct fi_bgq_ep	*ep[64];		/* TODO - check this array size */

	struct fi_cq_bgq_l2atomic_data	*fifo_memptr;
	struct flight_recorder	flight_recorder;
	struct l2atomic_counter	ref_cnt;
};

#define DUMP_ENTRY_INPUT(entry)	\
({				\
	fprintf(stderr,"%s:%s():%d entry = %p\n", __FILE__, __func__, __LINE__, (entry));					\
	fprintf(stderr,"%s:%s():%d   op_context = %p\n", __FILE__, __func__, __LINE__, (entry)->tagged.op_context);		\
	fprintf(stderr,"%s:%s():%d   flags      = 0x%016lx\n", __FILE__, __func__, __LINE__, (entry)->tagged.flags);		\
	fprintf(stderr,"%s:%s():%d   len        = %zu\n", __FILE__, __func__, __LINE__, (entry)->tagged.len);			\
	fprintf(stderr,"%s:%s():%d   buf        = %p\n", __FILE__, __func__, __LINE__, (entry)->tagged.buf);			\
	fprintf(stderr,"%s:%s():%d   ignore     = 0x%016lx\n", __FILE__, __func__, __LINE__, (entry)->recv.ignore);		\
	fprintf(stderr,"%s:%s():%d   tag        = 0x%016lx\n", __FILE__, __func__, __LINE__, (entry)->tagged.tag);		\
	fprintf(stderr,"%s:%s():%d   entry_kind = %u\n", __FILE__, __func__, __LINE__, (entry)->recv.entry_kind);		\
	fprintf(stderr,"%s:%s():%d   entry_id   = %u\n", __FILE__, __func__, __LINE__, (entry)->recv.entry_id);		\
})

static inline
int fi_bgq_cq_context_append (struct fi_bgq_cq * bgq_cq,
		union fi_bgq_context * context, const int lock_required)
{
	int ret;
	ret = fi_bgq_lock_if_required(&bgq_cq->lock, lock_required);
	if (ret) return ret;

	union fi_bgq_context * tail = bgq_cq->pending_tail;
	context->next = NULL;
	if (tail) {
		context->prev = tail;
		tail->next = context;
	} else {
		context->prev = NULL;
		bgq_cq->pending_head = context;
	}
	bgq_cq->pending_tail = context;

	ret = fi_bgq_unlock_if_required(&bgq_cq->lock, lock_required);
	if (ret) return ret;

	return 0;
}


static size_t fi_bgq_cq_fill(uintptr_t output,
		union fi_bgq_context * context,
		const enum fi_cq_format format)
{
	assert((context->flags & FI_BGQ_CQ_CONTEXT_EXT)==0);

	struct fi_cq_tagged_entry * entry = (struct fi_cq_tagged_entry *) output;
	switch (format) {
	case FI_CQ_FORMAT_CONTEXT:
		if ((context->flags & FI_BGQ_CQ_CONTEXT_MULTIRECV) == 0) {	/* likely */
			entry->op_context = (void *)context;
		} else {
			entry->op_context = (void *)context->multi_recv_context;
		}
		return sizeof(struct fi_cq_entry);
		break;
	case FI_CQ_FORMAT_MSG:
		*((struct fi_cq_msg_entry *)output) = *((struct fi_cq_msg_entry *)context);
		if ((context->flags & FI_BGQ_CQ_CONTEXT_MULTIRECV) == 0) {	/* likely */
			entry->op_context = (void *)context;
		} else {
			entry->op_context = (void *)context->multi_recv_context;
		}
		return sizeof(struct fi_cq_msg_entry);
		break;
	case FI_CQ_FORMAT_DATA:
		*((struct fi_cq_data_entry *)output) = *((struct fi_cq_data_entry *)context);
		if ((context->flags & FI_BGQ_CQ_CONTEXT_MULTIRECV) == 0) {	/* likely */
			entry->op_context = (void *)context;
		} else {
			entry->op_context = (void *)context->multi_recv_context;
		}
		((struct fi_cq_data_entry *)output)->data = 0; /* bgq does not provide completion data - TODO */
		return sizeof(struct fi_cq_data_entry);
		break;
	case FI_CQ_FORMAT_TAGGED:
		*((struct fi_cq_tagged_entry *)output) = *((struct fi_cq_tagged_entry *)context);
		if ((context->flags & FI_BGQ_CQ_CONTEXT_MULTIRECV) == 0) {	/* likely */
			entry->op_context = (void *)context;
		} else {
			entry->op_context = (void *)context->multi_recv_context;
		}
		((struct fi_cq_tagged_entry *)output)->data = 0; /* bgq does not provide completion data - TODO */
		return sizeof(struct fi_cq_tagged_entry);
		break;
	default:
		assert(0);
	}

	return 0;
}

int fi_bgq_ep_progress_manual (struct fi_bgq_ep *bgq_ep);

static ssize_t fi_bgq_cq_poll(struct fid_cq *cq, void *buf, size_t count,
		fi_addr_t *src_addr, const enum fi_cq_format format,
		const int lock_required, const enum fi_progress progress)
{
	ssize_t num_entries = 0;

	struct fi_bgq_cq *bgq_cq = (struct fi_bgq_cq *)cq;

	/* check if the err fifo has anything in it and return */
	/* TODO - don't use atomic fifo to report error events in FI_PROGRESS_MANUAL mode */
	if (!l2atomic_fifo_isempty(&bgq_cq->err_consumer)) {
		errno = FI_EAVAIL;
		return -errno;
	}

	int ret;
	ret = fi_bgq_lock_if_required(&bgq_cq->lock, lock_required);
	if (ret) return ret;

	if (progress == FI_PROGRESS_MANUAL) {
		const uint64_t count = bgq_cq->progress.ep_count;
		uint64_t i;
		for (i=0; i<count; ++i) {
			fi_bgq_ep_progress_manual(bgq_cq->progress.ep[i]);
		}
	}

	uintptr_t output = (uintptr_t)buf;

	/* examine each context in the pending completion queue and, if the
	 * operation is complete, initialize the cq entry in the application
	 * buffer and remove the context from the queue. */
	union fi_bgq_context * head = bgq_cq->pending_head;
	union fi_bgq_context * tail = bgq_cq->pending_tail;
	union fi_bgq_context * context = head;
	while ((count - num_entries) > 0 && context != NULL) {

		if (context->byte_counter == 0) {
			output += fi_bgq_cq_fill(output, context, format);
			++ num_entries;

			if (context->prev)
				context->prev->next = context->next;
			else
				head = context->next;

			if (context->next)
				context->next->prev = context->prev;
			else
				tail = context->prev;
		}

		context = context->next;
	}

	if (progress == FI_PROGRESS_AUTO) {

		/* drain the std fifo and initialize the cq entries in the application
		 * buffer if the operation is complete; otherwise append to the
		 * pending completion queue */
		uint64_t value = 0;
		struct l2atomic_fifo_consumer * consumer = &bgq_cq->std_consumer;
		while ((count - num_entries) > 0 &&
				l2atomic_fifo_consume(consumer, &value) == 0) {

			/* const uint64_t flags = value & 0xE000000000000000ull; -- currently not used */

			/* convert the fifo value into a context pointer */
			context = (union fi_bgq_context *) (value << 3);

			if (context->byte_counter == 0) {
				output += fi_bgq_cq_fill(output, context, format);
				++ num_entries;
			} else {
				context->next = NULL;
				context->prev = tail;
				if (tail)
					tail->next = context;
				else
					head = context;
				tail = context;
			}
		}
	}

	/* save the updated head and tail pointers */
	bgq_cq->pending_head = head;
	bgq_cq->pending_tail = tail;

	ret = fi_bgq_unlock_if_required(&bgq_cq->lock, lock_required);
	if (ret) return ret;

	if (num_entries == 0) {
		errno = FI_EAGAIN;
		return -errno;
	}

	return num_entries;
}


static inline
ssize_t fi_bgq_cq_read_generic (struct fid_cq *cq, void *buf, size_t count,
		const enum fi_cq_format format, const int lock_required, const enum fi_progress progress)
{
	int ret;
	ret = fi_bgq_cq_poll(cq, buf, count, NULL, format, lock_required, progress);
	return ret;
}

static inline
ssize_t fi_bgq_cq_readfrom_generic (struct fid_cq *cq, void *buf, size_t count, fi_addr_t *src_addr,
		const enum fi_cq_format format, const int lock_required, const enum fi_progress progress)
{
	int ret;
	ret = fi_bgq_cq_poll(cq, buf, count, src_addr, format, lock_required, progress);
	if (ret > 0) {
		unsigned n;
		for (n=0; n<ret; ++n) src_addr[n] = FI_ADDR_NOTAVAIL;
	}

	return ret;
}

/*
 * Declare specialized functions that qualify for FABRIC_DIRECT.
 * - No locks
 * - FI_CQ_FORMAT_TAGGED
 * - FI_PROGRESS_MANUAL
 */
#define FI_BGQ_CQ_FABRIC_DIRECT_LOCK		0
#define FI_BGQ_CQ_FABRIC_DIRECT_FORMAT		FI_CQ_FORMAT_TAGGED

FI_BGQ_CQ_SPECIALIZED_FUNC(FI_BGQ_CQ_FABRIC_DIRECT_FORMAT,
		FI_BGQ_CQ_FABRIC_DIRECT_LOCK,
		FI_BGQ_FABRIC_DIRECT_PROGRESS)

#ifdef FABRIC_DIRECT
#define fi_cq_read(cq, buf, count)					\
	(FI_BGQ_CQ_SPECIALIZED_FUNC_NAME(cq_read,			\
			FI_BGQ_CQ_FABRIC_DIRECT_FORMAT,			\
			FI_BGQ_CQ_FABRIC_DIRECT_LOCK,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(cq, buf, count))

#define fi_cq_readfrom(cq, buf, count, src_addr)			\
	(FI_BGQ_CQ_SPECIALIZED_FUNC_NAME(cq_readfrom,			\
			FI_BGQ_CQ_FABRIC_DIRECT_FORMAT,			\
			FI_BGQ_CQ_FABRIC_DIRECT_LOCK,			\
			FI_BGQ_FABRIC_DIRECT_PROGRESS)			\
	(cq, buf, count, src_addr))


static inline
ssize_t fi_cq_readerr(struct fid_cq *cq, struct fi_cq_err_entry *buf,
		uint64_t flags)
{
	return cq->ops->readerr(cq, buf, flags);
}

static inline
uint64_t fi_cntr_read(struct fid_cntr *cntr)
{
	return cntr->ops->read(cntr);
}

static inline
int fi_cntr_wait(struct fid_cntr *cntr, uint64_t threshold, int timeout)
{
	return cntr->ops->wait(cntr, threshold, timeout);
}

static inline
int fi_trywait(struct fid_fabric *fabric, struct fid **fids, int count)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_wait(struct fid_wait *waitset, int timeout)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_poll(struct fid_poll *pollset, void **context, int count)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_poll_add(struct fid_poll *pollset, struct fid *event_fid,
			      uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_poll_del(struct fid_poll *pollset, struct fid *event_fid,
			      uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_eq_open(struct fid_fabric *fabric, struct fi_eq_attr *attr,
			     struct fid_eq **eq, void *context)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
ssize_t fi_eq_read(struct fid_eq *eq, uint32_t *event, void *buf,
				 size_t len, uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
ssize_t fi_eq_readerr(struct fid_eq *eq,
				    struct fi_eq_err_entry *buf, uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
ssize_t fi_eq_write(struct fid_eq *eq, uint32_t event,
				  const void *buf, size_t len, uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
ssize_t fi_eq_sread(struct fid_eq *eq, uint32_t *event, void *buf,
				  size_t len, int timeout, uint64_t flags)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
const char *fi_eq_strerror(struct fid_eq *eq, int prov_errno,
					 const void *err_data, char *buf,
					 size_t len)
{
	return NULL;		/* TODO - implement this */
}

static inline
ssize_t fi_cq_sread(struct fid_cq *cq, void *buf, size_t count,
				  const void *cond, int timeout)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
ssize_t fi_cq_sreadfrom(struct fid_cq *cq, void *buf,
				      size_t count, fi_addr_t *src_addr,
				      const void *cond, int timeout)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_cq_signal(struct fid_cq *cq)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
const char *fi_cq_strerror(struct fid_cq *cq, int prov_errno,
					 const void *err_data, char *buf,
					 size_t len)
{
	return NULL;		/* TODO - implement this */
}

static inline
uint64_t fi_cntr_readerr(struct fid_cntr *cntr)
{
	return 0;		/* TODO - implement this */
}

static inline
int fi_cntr_add(struct fid_cntr *cntr, uint64_t value)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

static inline
int fi_cntr_set(struct fid_cntr *cntr, uint64_t value)
{
	return -FI_ENOSYS;	/* TODO - implement this */
}

#endif

#ifdef __cplusplus
}
#endif

#endif /* _FI_BGQ_DIRECT_EQ_H_ */
