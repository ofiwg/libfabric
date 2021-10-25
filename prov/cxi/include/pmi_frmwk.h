/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/* These are initialized by pmi_populate_av() */
extern int pmi_numranks;
extern int pmi_rank;
extern char pmi_hostname[256];
extern struct cxip_addr *pmi_nids;

char *cxit_node;
char *cxit_service;
uint64_t cxit_flags;
struct fi_info *cxit_fi_hints;
struct fi_info *cxit_fi;

struct fid_fabric *cxit_fabric;
struct fid_domain *cxit_domain;
struct fi_cxi_dom_ops *dom_ops;

struct fid_ep *cxit_ep;
struct fi_eq_attr cxit_eq_attr;
uint64_t cxit_eq_bind_flags;
struct fid_eq *cxit_eq;

struct fi_cq_attr cxit_rx_cq_attr;
uint64_t cxit_rx_cq_bind_flags;
struct fid_cq *cxit_rx_cq;

struct fi_cq_attr cxit_tx_cq_attr;
uint64_t cxit_tx_cq_bind_flags;
struct fid_cq *cxit_tx_cq;

fi_addr_t cxit_ep_fi_addr;

struct fi_cntr_attr cxit_cntr_attr;
struct fid_cntr *cxit_send_cntr;
struct fid_cntr *cxit_recv_cntr;
struct fid_cntr *cxit_read_cntr;
struct fid_cntr *cxit_write_cntr;
struct fid_cntr *cxit_rem_cntr;

struct fi_av_attr cxit_av_attr;
struct fid_av *cxit_av;

int cxit_n_ifs;
struct fid_av_set *cxit_av_set;
struct fid_mc *cxit_mc;
fi_addr_t cxit_mc_addr;

void pmi_free_libfabric(void);
int pmi_init_libfabric(void);
int pmi_populate_av(fi_addr_t **fiaddr, size_t *size);
int pmi_errmsg(int ret, const char *fmt, ...)
	__attribute__((format(__printf__, 2, 3)));
int pmi_log0(const char *fmt, ...)
	__attribute__((format(__printf__, 1, 2)));

int pmi_trace_enable(bool enable);
