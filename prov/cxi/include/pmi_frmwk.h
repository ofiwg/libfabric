/*
 * (c) Copyright 2021 Hewlett Packard Enterprise Development LP
 */

/* These are initialized by pmi_populate_av() */
extern int pmi_numranks;
extern int pmi_rank;
extern uint32_t *pmi_nids;

void pmi_free_libfabric(void);
int pmi_init_libfabric(void);
int pmi_populate_av(void);
int pmi_enable_libfabric(void);
