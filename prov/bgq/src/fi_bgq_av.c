#include "rdma/bgq/fi_bgq.h"
#include <fi_enosys.h>

#include "rdma/bgq/fi_bgq_spi.h"

#if 0
static inline
uint32_t coords_to_nodeid (Personality_t * p,
		uint32_t a, uint32_t b, uint32_t c, uint32_t d, uint32_t e)
{
        return ((((((((p->Network_Config.Acoord *
		p->Network_Config.Bnodes) + p->Network_Config.Bcoord) *
			p->Network_Config.Cnodes) + p->Network_Config.Ccoord) *
				p->Network_Config.Dnodes) + p->Network_Config.Dcoord) *
					p->Network_Config.Enodes) + p->Network_Config.Ecoord);
}
#endif

static inline
void fi_bgq_addr_initialize (union fi_bgq_addr * output,
		BG_CoordinateMapping_t * my_coords, uint32_t a, uint32_t b,
		uint32_t c, uint32_t d, uint32_t e, uint32_t t, uint32_t ppn,
		uint32_t domain_id, uint32_t domains_per_process,
		uint32_t endpoint_id, uint32_t endpoints_per_domain)
{
	const uint32_t rx_per_node = ((BGQ_MU_NUM_REC_FIFO_GROUPS-1) * BGQ_MU_NUM_REC_FIFOS_PER_GROUP) / 2;	/* each rx uses two mu reception fifos */
	const uint32_t rx_per_process = rx_per_node / ppn;
	const uint32_t rx_per_domain = rx_per_process / domains_per_process;
	const uint32_t rx_per_endpoint = rx_per_domain / endpoints_per_domain;

	output->reserved	= 0;
	output->a		= a;
	output->b		= b;
	output->c		= c;
	output->d		= d;
	output->e		= e;

	output->is_local	=
		(my_coords->a == a) &&
		(my_coords->b == b) &&
		(my_coords->c == c) &&
		(my_coords->d == d) &&
		(my_coords->e == e);

	output->fifo_map = fi_bgq_mu_calculate_fifo_map(*my_coords, a, b, c, d, e, t);

	/*
	 * The least significant bits are initially zero, which represents the
	 * 'base' reception context for a scalable endpoint, and the 'rx'
	 * field is the last field in the address structure which means that
	 * a 'base' address stored in the address vector object can be
	 * converted into a 'scalable' address by simply adding the rx index
	 * to the fi_addr_t.
	 */
	output->rx		= (rx_per_process * t) +
				(rx_per_domain * domain_id) +
				(rx_per_endpoint * endpoint_id);
}

#if 0
static inline
void fi_bgq_addr_initialize_coords (union fi_bgq_addr * output,
		BG_CoordinateMapping_t * my_coords, BG_CoordinateMapping_t * input,
		uint32_t t, uint32_t ppn)
{
	fi_bgq_addr_initialize(output, my_coords, input->a, input->b, input->c,
		input->d, input->e, t, ppn);
}
#endif

static int fi_bgq_close_av(fid_t fid)
{
	int ret;
	struct fi_bgq_av *bgq_av =
		container_of(fid, struct fi_bgq_av, av_fid);

	ret = fi_bgq_fid_check(fid, FI_CLASS_AV, "address vector");
	if (ret)
		return ret;

	if (bgq_av->map_addr) free(bgq_av->map_addr);

	ret = fi_bgq_ref_dec(&bgq_av->domain->ref_cnt, "domain");
	if (ret)
		return ret;

	ret = fi_bgq_ref_finalize(&bgq_av->ref_cnt, "address vector");
	if (ret)
		return ret;

	free(bgq_av);
	return 0;
}

/*
 * The 'addr' is a representation of the address - not a string
 *
 * 'flags' is allowed to be ignored
 * 'context' is not used ... what is the purpose?
 */
static int
fi_bgq_av_insert(struct fid_av *av, const void *addr, size_t count,
	     fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct fi_bgq_av *bgq_av =
		container_of(av, struct fi_bgq_av, av_fid);

	if (!bgq_av) {
		errno = FI_EINVAL;
		return -errno;
	}

	switch (bgq_av->type) {
	case FI_AV_TABLE:
		/* The address table is internal and the application uses a
		 * 'monotonically increasing integer' to index the table and
		 * retrieve the actual internal address
		 */
		errno = FI_ENOSYS;
		return -errno;
		break;
	case FI_AV_MAP:
		/* The address map is maintained by the application ('fi_addr') and
		 * the provider must fill in the map with the actual network
		 * address of each .
		 */
		if (!addr) {
			errno = FI_EINVAL;
			return -errno;
		}
		break;
	default:
		errno = FI_EINVAL;
		return -errno;
	}

	BG_CoordinateMapping_t * coords = (BG_CoordinateMapping_t *) addr;
	union fi_bgq_addr * output = (union fi_bgq_addr *) fi_addr;
	uint32_t ppn = Kernel_ProcessCount();
	uint32_t n;
	for (n=0; n<count; ++n) {

		fi_bgq_addr_initialize(&output[n], &bgq_av->domain->my_coords,
			coords[n].a, coords[n].b, coords[n].c,
			coords[n].d, coords[n].e, coords[n].t, ppn,
			0, 1, 0, 1);
	}

	return count;
}

static int
fi_bgq_av_insertsvc(struct fid_av *av, const char *node, const char *service,
		fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct fi_bgq_av *bgq_av =
		container_of(av, struct fi_bgq_av, av_fid);

	if (!bgq_av) {
		errno = FI_EINVAL;
		return -errno;
	}

	switch (bgq_av->type) {
	case FI_AV_TABLE:
		/* The address table is internal and the application uses a
		 * 'monotonically increasing integer' to index the table and
		 * retrieve the actual internal address
		 */
		errno = FI_ENOSYS;
		return -errno;
		break;
	case FI_AV_MAP:
		/* The address map is maintained by the application ('fi_addr') and
		 * the provider must fill in the map with the actual network
		 * address of each .
		 */

		break;
	default:
		errno = FI_EINVAL;
		return -errno;
	}

	/*
	 * convert the string representation of the node ("#.#.#.#.#.#") into
	 * torus coordinates and the 't' coordinate.
	 */
	uint32_t a, b, c, d, e, t;
	const char * node_str = (const char *) node;
	sscanf(node_str, "%u.%u.%u.%u.%u.%u", &a, &b, &c, &d, &e, &t);

	union fi_bgq_addr * output = (union fi_bgq_addr *) fi_addr;
	fi_bgq_addr_initialize (output, &bgq_av->domain->my_coords,
		a, b, c, d, e, t, Kernel_ProcessCount(), 0, 1, 0, 1);

	return 0;
}

/*
 * This is similar to "ranks to coords" syscall. The "node" is the string
 * representation of the torus coordinates of a node and the 't' coordinate,
 * such as "0.0.0.0.0.0", and the "service" is the string representation of
 * what could be considered a pami-style "client id". Currently, only a single
 * "service" per "node" is supported - the service parameter is ignored and
 * a svccnt != 1 is considered an error.
 *
 * If the "node" parameter is NULL, then the insert begins at coordinate
 * 0.0.0.0.0.0 and increments according to the default ABCDET map order until
 * "nodecnt" addresses have been inserted. In this respect, "nodecnt" is the
 * same as the job size.
 *
 * The bgq provider does not support rank reorder via mapfiles.
 */
static int
fi_bgq_av_insertsym(struct fid_av *av, const char *node, size_t nodecnt,
		const char *service, size_t svccnt,
		fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct fi_bgq_av *bgq_av =
		container_of(av, struct fi_bgq_av, av_fid);

	if (!bgq_av) {
		errno = FI_EINVAL;
		return -errno;
	}

	if (svccnt != 1) {
		fprintf(stderr, "Error. Only one 'service' per 'node' is supported by the bgq provider\n");
		errno = FI_EINVAL;
		return -errno;
	}

	switch (bgq_av->type) {
	case FI_AV_TABLE:
		/* The address table is internal and the application uses a
		 * 'monotonically increasing integer' to index the table and
		 * retrieve the actual internal address
		 */
		errno = FI_ENOSYS;
		return -errno;
		break;
	case FI_AV_MAP:
		/* The address map is maintained by the application ('fi_addr') and
		 * the provider must fill in the map with the actual network
		 * address of each .
		 */

		break;
	default:
		errno = FI_EINVAL;
		return -errno;
	}

	/*
	 * convert the string representation of the node ("#.#.#.#.#") into
	 * torus coordinates and convert the string representation of the
	 * service, a.k.a. "process", into a t coordinate.
	 */
	uint32_t a, b, c, d, e, t;
	if (node)
		sscanf(node, "%u.%u.%u.%u.%u.%u", &a, &b, &c, &d, &e, &t);
	else
		a = b = c = d = e = t = 0;

	Personality_t personality;
	int rc;
	rc = Kernel_GetPersonality(&personality, sizeof(Personality_t));
	if (rc) {
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	}
	uint32_t ppn = Kernel_ProcessCount();
	size_t node_count = personality.Network_Config.Anodes *
		personality.Network_Config.Bnodes *
		personality.Network_Config.Cnodes *
		personality.Network_Config.Dnodes *
		personality.Network_Config.Enodes *
		ppn;

	uint32_t maximum_to_insert = (node_count < nodecnt) ? node_count : nodecnt;

	int n = 0;
	uint32_t _a, _b, _c, _d, _e, _t;
	union fi_bgq_addr * output = (union fi_bgq_addr *) fi_addr;
	for (_a = a; _a < personality.Network_Config.Anodes; ++_a) {
	for (_b = b; _b < personality.Network_Config.Bnodes; ++_b) {
	for (_c = c; _c < personality.Network_Config.Cnodes; ++_c) {
	for (_d = d; _d < personality.Network_Config.Dnodes; ++_d) {
	for (_e = e; _e < personality.Network_Config.Enodes; ++_e) {
	for (_t = t; _t < ppn; ++_t) {

		if (n == maximum_to_insert) break;

		fi_bgq_addr_initialize (&output[n++],
			&bgq_av->domain->my_coords,
			_a, _b, _c, _d, _e, _t, ppn, 0, 1, 0, 1);

	}}}}}}

	return n;		
}


#if 0
static int
fi_bgq_av_insertsym_save(struct fid_av *av, const char *node, size_t nodecnt,
		const char *service, size_t svccnt,
		fi_addr_t *fi_addr, uint64_t flags, void *context)
{
	struct fi_bgq_av *bgq_av =
		container_of(av, struct fi_bgq_av, av_fid);

	if (!bgq_av) {
		errno = FI_EINVAL;
		return -errno;
	}

	switch (bgq_av->type) {
	case FI_AV_TABLE:
		/* The address table is internal and the application uses a
		 * 'monotonically increasing integer' to index the table and
		 * retrieve the actual internal address
		 */
		errno = FI_ENOSYS;
		return -errno;
		break;
	case FI_AV_MAP:
		/* The address map is maintained by the application ('fi_addr') and
		 * the provider must fill in the map with the actual network
		 * address of each .
		 */

		break;
	default:
		errno = FI_EINVAL;
		return -errno;
	}

	/*
	 * verify that a mapping file was not used.
	 *
	 * verify that the map order, such as "ABCDET", always ends in "T"
	 * (see the man page for "fi_av_insertsym").
	 */
	uint32_t cnk_rc = 0;
	char name[1024];
	uint32_t is_file = 0;
	cnk_rc = Kernel_GetMapping(1023, name, &is_file);
	if (cnk_rc) {
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	} else if (is_file) {
		fprintf(stderr, "Mapping file is not supported\n");
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	}
	fprintf(stderr, "Kernel_GetMapping() -> \"%s\"\n", name);



	Personality_t personality;
	cnk_rc = Kernel_GetPersonality(&personality, sizeof(Personality_t));
	if (cnk_rc) {
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	}
	size_t node_count = personality.Network_Config.Anodes *
		personality.Network_Config.Bnodes *
		personality.Network_Config.Cnodes *
		personality.Network_Config.Dnodes *
		personality.Network_Config.Enodes;

	uint32_t ppn = Kernel_ProcessCount();

	/*
	 * convert the string representation of the node ("#.#.#.#.#") into
	 * torus coordinates and convert the string representation of the
	 * service, a.k.a. "process", into a t coordinate.
	 */
	uint32_t a, b, c, d, e, t;
	const char * node_str = (const char *) node;
	sscanf(node_str, "%u.%u.%u.%u.%u.%u", &a, &b, &c, &d, &e);
	const char * service_str = (const char *) service;
	sscanf(service_str, "%u", &t);

	/* check input parameters to avoid buffer overrun */
	uint32_t nodeid = coords_to_nodeid(&pers, a, b, c, d, e);
	if (nodeid + nodecnt > node_count) {
fprintf(stderr, "Error. Too many nodes requested\n");
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	}
	if (t + svccnt > ppn) {	
fprintf(stderr, "Error. Too many service ports (processes) requested\n");
		errno = FI_EINVAL;	/* is this the correct errno? */
		return -errno;
	}

	/*
	 * read the ranks2coords mapping on to the stack - this is temporary
	 * as the actual fi_addr_t array is maintained by the application
	 */
	size_t mapsize = node_count * ppn;
	BG_CoordinateMapping_t map[mapsize];
	uint64_t numentries = 0;
	cnk_rc = Kernel_RanksToCoords(sizeof(map), map, &numentries);

	/*
	 * convert the 'BG_CoordinateMapping_t' addresses into 'fi_addr_t'
	 * addresses and copy into the provided memory buffer.
	 */
	uint32_t processes_to_insert = MIN(ppn-t, svccnt);
	uint32_t nodes_to_insert = MIN(node_count-nodeid, nodecnt);
	union fi_bgq_addr * fi_bgq_addr = (union fi_bgq_addr *) fi_addr;

	int entry = 0;
	for (_nodeid = nodeid; _nodeid < nodes_to_insert; ++_nodeid) {

		uint32_t _t, _rank;
		for (_t = t; _t < processes_to_insert; ++_t) {
			_rank = nodeid * ppn + _t;
			union fi_bgq_addr * output = &fi_bgq_addr[entry];
			fi_bgq_addr_initialize_coords(output,
				&bgq_av->domain->my_coords, &map[_rank],
				_t, ppn);

			++entry;
		}
	}

	return entry;
}
#endif
static int
fi_bgq_av_remove(struct fid_av *av, fi_addr_t *fi_addr, size_t count, uint64_t flags)
{
	return 0;	/* noop on bgq */
}

static int
fi_bgq_av_lookup(struct fid_av *av, fi_addr_t fi_addr, void *addr, size_t *addrlen)
{
	union fi_bgq_addr bgq_addr;
	bgq_addr.fi = fi_addr;

	BG_CoordinateMapping_t tmp;
	tmp.a = bgq_addr.a;
	tmp.b = bgq_addr.b;
	tmp.c = bgq_addr.c;
	tmp.d = bgq_addr.d;
	tmp.e = bgq_addr.e;

	const uint32_t ppn = Kernel_ProcessCount();
	const uint32_t rx_per_node = ((BGQ_MU_NUM_REC_FIFO_GROUPS-1) * BGQ_MU_NUM_REC_FIFOS_PER_GROUP) / 2;	/* each rx uses two mu reception fifos */
	const uint32_t rx_per_process = rx_per_node / ppn;
	tmp.t = (uint32_t)(bgq_addr.rx / rx_per_process);

	memcpy(addr, (const void *)&tmp, *addrlen);

	*addrlen = sizeof(BG_CoordinateMapping_t);

	return 0;
}
	
static const char *
fi_bgq_av_straddr(struct fid_av *av, const void *addr,
			char *buf, size_t *len)
{
	BG_CoordinateMapping_t * input = (BG_CoordinateMapping_t *) addr;
	snprintf(buf, *len, "%u.%u.%u.%u.%u.%u", input->a, input->b, input->c,
		input->d, input->e, input->t);

	*len = 16;	/* "aa.bb.cc.dd.e.tt" */
	return buf;
}

static struct fi_ops fi_bgq_fi_ops = {
	.size		= sizeof(struct fi_ops),
	.close		= fi_bgq_close_av,
	.bind		= fi_no_bind,
	.control	= fi_no_control,
	.ops_open	= fi_no_ops_open
};

int fi_bgq_bind_ep_av(struct fi_bgq_ep *bgq_ep,
		struct fi_bgq_av *bgq_av, uint64_t flags)
{
	if (bgq_ep->av) {
		FI_LOG(fi_bgq_global.prov, FI_LOG_DEBUG, FI_LOG_DOMAIN,
			"Address vector already bound to TX endpoint\n");
		errno = FI_EINVAL;
		return -errno;
	}

	bgq_ep->av = bgq_av;
#if 0
	enum fi_threading threading = bgq_ep->domain->threading;
fprintf(stderr, "%s:%d threading=%d\n", __FILE__, __LINE__, threading);
	switch (threading) {
	case FI_THREAD_ENDPOINT:
		assert(bgq_ep->av->type == FI_AV_MAP); // for now ....
		if (bgq_ep->av->type == FI_AV_MAP) {
			if (bgq_ep->tx_op_flags & FI_DELIVERY_COMPLETE) {
				bgq_ep->ep_fid.tagged->send = 
					&fi_bgq_tsend_thread_single_map_mpi;
			} else {
				bgq_ep->ep_fid.tagged->send = 
					&fi_bgq_tsend_thread_single_map;
			}
			bgq_ep->ep_fid.tagged->inject = 
				&fi_bgq_tinject_thread_single_map;
			bgq_ep->ep_fid.tagged->sendmsg = 
				&fi_bgq_tsendmsg_last;
		}
		break;
	case FI_THREAD_DOMAIN:
	case FI_THREAD_SAFE:
		assert(bgq_ep->av->type == FI_AV_MAP); // for now ....
		if (bgq_ep->av->type == FI_AV_MAP) {
			if (bgq_ep->tx_op_flags & FI_DELIVERY_COMPLETE) {
				bgq_ep->ep_fid.tagged->send = 
					&fi_bgq_tsend_thread_multiple_map_mpi;
			} else {
				bgq_ep->ep_fid.tagged->send = 
					&fi_bgq_tsend_thread_multiple_map;
			}
			bgq_ep->ep_fid.tagged->inject = 
				&fi_bgq_tinject_thread_multiple_map;
			bgq_ep->ep_fid.tagged->sendmsg = 
				&fi_bgq_tsendmsg_last;
		}
		break;
	default:
		errno = FI_ENOSYS;
		return -errno;
	}
#endif

	fi_bgq_ref_inc(&bgq_av->ref_cnt, "address vector");

	return 0;
}

static struct fi_ops_av fi_bgq_av_ops = {
	.size		= sizeof(struct fi_ops_av),
	.insert		= fi_bgq_av_insert,
	.insertsvc	= fi_bgq_av_insertsvc,
	.insertsym	= fi_bgq_av_insertsym,
	.remove		= fi_bgq_av_remove,
	.lookup		= fi_bgq_av_lookup,
	.straddr	= fi_bgq_av_straddr
};

int fi_bgq_av_open(struct fid_domain *dom,
		struct fi_av_attr *attr, struct fid_av **av,
		void *context)
{
	int ret;
	struct fi_bgq_av *bgq_av = NULL;

	/*
	 * Check for unsupported mappings. Currently only the default
	 * 'ABCDET' mapping is supported.
	 */
	char maporder[8];
	uint32_t isFile = 0;
	Kernel_GetMapping(sizeof(maporder), maporder, &isFile);
	if (isFile) {
		FI_LOG(fi_bgq_global.prov, FI_LOG_DEBUG, FI_LOG_AV,
				"bgq 'file' mapping is not supported\n");
		errno = FI_EINVAL;
		return -errno;
	}
	if (maporder[0] != 'A' || maporder[1] != 'B' ||
			maporder[2] != 'C' || maporder[3] != 'D' ||
			maporder[4] != 'E' || maporder[5] != 'T') {
		FI_LOG(fi_bgq_global.prov, FI_LOG_DEBUG, FI_LOG_AV,
				"bgq mapping %c%c%c%c%c%c is not supported\n",
				maporder[0], maporder[1], maporder[2],
				maporder[3], maporder[4], maporder[5]);
		errno = FI_EINVAL;
		return -errno;
	}

	if (!attr) {
		FI_LOG(fi_bgq_global.prov, FI_LOG_DEBUG, FI_LOG_AV,
				"no attr provided\n");
		errno = FI_EINVAL;
		return -errno;
	}

	ret = fi_bgq_fid_check(&dom->fid, FI_CLASS_DOMAIN, "domain");
	if (ret)
		return ret;

	bgq_av = calloc(1, sizeof(*bgq_av));
	if (!bgq_av) {
		errno = FI_ENOMEM;
		goto err;
	}

	bgq_av->av_fid.fid.fclass = FI_CLASS_AV;
	bgq_av->av_fid.fid.context= context;
	bgq_av->av_fid.fid.ops    = &fi_bgq_fi_ops;
	bgq_av->av_fid.ops 	  = &fi_bgq_av_ops;

	bgq_av->domain = (struct fi_bgq_domain *) dom;
	bgq_av->type = attr->type;

	bgq_av->map_addr = NULL;
	if (attr->name != NULL && (attr->flags & FI_READ)) {

		if (0 == attr->ep_per_node) {
			errno = FI_EINVAL;
			goto err;
		}

		assert(0 == attr->map_addr);


		Personality_t personality;
		int rc;
		rc = Kernel_GetPersonality(&personality, sizeof(Personality_t));
		if (rc) {
			errno = FI_EINVAL;
			return -errno;
		}

		const uint32_t ppn = Kernel_ProcessCount();
		const size_t node_count = personality.Network_Config.Anodes *
			personality.Network_Config.Bnodes *
			personality.Network_Config.Cnodes *
			personality.Network_Config.Dnodes *
			personality.Network_Config.Enodes;

		size_t mapsize = node_count * ppn;
		BG_CoordinateMapping_t map[mapsize];
		uint64_t ep_count;
		rc = Kernel_RanksToCoords(sizeof(map), map, &ep_count);

		// For now just 1 end point per process
		const size_t ep_per_process = 1;

		union fi_bgq_addr *addr = (union fi_bgq_addr *)malloc(sizeof(fi_addr_t)*ep_count);	/* TODO - mmap this into shared memory */

		size_t ep = 0, n = 0;
		int i;

		/* Call the fi_bgq_addr_initialize for the exact processes in the block. */
		for (i=0;i<ep_count;i++) {
			fi_bgq_addr_initialize (&addr[n++],
				&bgq_av->domain->my_coords,
				map[i].a, map[i].b, map[i].c, map[i].d, map[i].e, map[i].t, ppn, 0, 1,
				ep, ep_per_process);
		}

		bgq_av->map_addr = (void *)addr;
		attr->map_addr = (void *)addr;
	}

	*av = &bgq_av->av_fid;

	fi_bgq_ref_init(&bgq_av->domain->fabric->node, &bgq_av->ref_cnt, "address vector");
	fi_bgq_ref_inc(&bgq_av->domain->ref_cnt, "domain");

	return 0;
err:
	if (bgq_av)
		free(bgq_av);
	return -errno;
}
