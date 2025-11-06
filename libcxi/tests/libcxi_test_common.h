/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018, 2020 Hewlett Packard Enterprise Development LP */

#ifndef __LIBCXI_TEST_COMMON_H__
#define __LIBCXI_TEST_COMMON_H__

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-prototypes"
#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#pragma GCC diagnostic pop

#include "libcxi.h"

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

enum eqe_fmt {
	EQE_INIT_SHORT,
	EQE_INIT_LONG,
	EQE_TRIG_SHORT,
	EQE_TRIG_LONG,
	EQE_CMD_FAIL,
	EQE_TGT_LONG,
	EQE_TGT_SHORT,
	EQE_ENET,
	EQE_EQ_SWITCH,
};

enum gpu_copy_dir {
	 /* These match the cudaMemcpyHostToDevice/cudaMemcpyDeviceToHost */
	to_device = 1,
	to_host = 2,
};

enum buf_location {
	on_host = 0,
	on_device,
};

struct cxil_test_data {
	void *addr;
	size_t len;
	struct cxi_md *md;
};

struct mem_window {
	size_t length;
	uint8_t *buffer;
	struct cxi_md *md;
	struct cxi_md_hints hints;
	enum buf_location loc;
	bool is_device;
	bool use_dmabuf;
};

extern uint32_t dev_id;
extern uint32_t vni;
extern uint32_t vni_excp;
extern uint32_t domain_pid;
extern struct cxil_dev *dev;
extern struct cxil_lni *lni;
extern struct cxi_cp *cp;
extern struct cxi_cp *excp;
extern struct cxil_domain *domain;
extern struct cxil_domain *domain_excp;
extern struct cxi_cq *transmit_cmdq;
extern struct cxi_cq *transmit_cmdq_excp;
extern struct cxi_cq *target_cmdq;
extern struct cxil_wait_obj *wait_obj;
extern struct cxi_eq_attr transmit_eq_attr;
extern size_t transmit_eq_buf_len;
extern void *transmit_eq_buf;
extern struct cxi_md *transmit_eq_md;
extern struct cxi_eq *transmit_evtq;
extern struct cxi_eq_attr target_eq_attr;
extern size_t target_eq_buf_len;
extern void *target_eq_buf;
extern struct cxi_md *target_eq_md;
extern struct cxi_eq *target_evtq;
extern struct cxil_pte *rx_pte;
extern struct cxil_pte_map *rx_pte_map;
extern struct cxil_test_data *test_data;
extern struct cxi_ct *ct;
extern struct cxi_cq *trig_cmdq;
extern struct c_ct_writeback *wb;

extern int test_data_len;

/* Used by most tests */
void devinfo_setup(void);
void devinfo_teardown(void);
void dev_setup(void);
void dev_teardown(void);
void lni_setup(void);
void lni_teardown(void);
void cp_setup(void);
void cp_teardown(void);
void domain_setup(void);
void domain_teardown(void);

/* Used by PTE tests */
void pte_setup(void);
void pte_teardown(void);
void test_data_setup(void);
void test_data_teardown(void);

/* Used by data transfer tests */
void data_xfer_setup(void);
void data_xfer_teardown(void);
void ptlte_setup(uint32_t pid_idx, bool matching, bool exclusive_cp);
void ptlte_teardown(void);
bool is_netsim(void);

/* Used by counting_events. */
void counting_event_setup(void);
void counting_event_teardown(void);
void expect_ct_values(struct cxi_ct *ct, uint64_t success, uint8_t failure);

void process_eqe(struct cxi_eq *evtq, enum eqe_fmt fmt, uint32_t type,
		 uint64_t id, union c_event *event_out);

void append_le(const struct cxil_pte *pte,
	       struct mem_window *mem_win,
	       enum c_ptl_list list,
	       uint32_t buffer_id,
	       uint64_t match_bits,
	       uint64_t ignore_bits,
	       uint32_t match_id,
	       uint64_t min_free,
	       bool event_success_disable,
	       bool event_unlink_disable,
	       bool use_once,
	       bool manage_local,
	       bool no_truncate,
	       bool op_put,
	       bool op_get);
void append_le_sync(const struct cxil_pte *pte,
		    struct mem_window *mem_win,
		    enum c_ptl_list list,
		    uint32_t buffer_id,
		    uint64_t match_bits,
		    uint64_t ignore_bits,
		    uint32_t match_id,
		    uint64_t min_free,
		    bool event_success_disable,
		    bool event_unlink_disable,
		    bool use_once,
		    bool manage_local,
		    bool no_truncate,
		    bool op_put,
		    bool op_get,
		    union c_event *event);

/* Generic Append interfaces, expand as necessary */
void unlink_le(const struct cxil_pte *pte, enum c_ptl_list list,
	       uint32_t buffer_id);
void unlink_le_sync(const struct cxil_pte *pte, enum c_ptl_list list,
		    uint32_t buffer_id);

int wait_for_event(struct cxil_wait_obj *wait);

/* Generic Put interfaces, expand as necessary */
void do_put(struct mem_window mem_win, size_t len, uint64_t r_off,
	    uint64_t l_off, uint32_t pid_idx, bool restricted,
	    uint64_t match_bits, uint64_t user_ptr, uint32_t initiator,
	    bool exclusive_cp);
void do_put_sync(struct mem_window mem_win, size_t len, uint64_t r_off,
		 uint64_t l_off, uint32_t pid_idx, bool restricted,
		 uint64_t match_bits, uint64_t user_ptr, uint32_t initiator,
		 bool exclusive_cp);

/* Generic Get interfaces, expand as necessary */
void do_get(struct mem_window mem_win, size_t len, uint64_t r_off,
	     uint32_t pid_idx, bool restricted, uint64_t match_bits,
	     uint64_t user_ptr, uint32_t initiator, struct cxi_eq *evtq);
void do_get_sync(struct mem_window mem_win, size_t len, uint64_t r_off,
		 uint32_t pid_idx, bool restricted, uint64_t match_bits,
		 uint64_t user_ptr, uint32_t initiator, struct cxi_eq *evtq);

void alloc_iobuf(size_t len, struct mem_window *win, uint32_t prot);
void free_iobuf(struct mem_window *win);
void alloc_map_devicebuf(size_t len, struct mem_window *win, uint32_t prot);
void alloc_map_hostbuf(size_t len, struct mem_window *win, uint32_t prot);
void free_unmap_devicebuf(struct mem_window *win);
void memcpy_device_to_host(void *dest, struct mem_window *win);
void memcpy_host_to_device(struct mem_window *win, void *src);
void memset_device(struct mem_window *win, int value, size_t count);

int gpu_lib_init(void);
void gpu_lib_fini(void);
int hip_lib_init(void);
void hip_lib_fini(void);
int cuda_lib_init(void);
void cuda_lib_fini(void);
void ze_fini(void);
int ze_init(void);
extern int (*gpu_malloc)(struct mem_window *win);
extern int (*gpu_host_alloc)(struct mem_window *win);
extern int (*gpu_free)(void *devPtr);
extern int (*gpu_host_free)(void *p);
extern int (*gpu_memset)(void *devPtr, int value, size_t count);
extern int (*gpu_memcpy)(void *dst, const void *src, size_t count,
			 enum gpu_copy_dir dir);
extern int (*gpu_props)(struct mem_window *win, void **base, size_t *size);
extern int (*gpu_close_fd)(int dma_buf_fd);

#define ONE_GB (1024*1024*1024)
#define TWO_MB (2*1024*1024)
int check_huge_pg_free(size_t hp_size, uint32_t npg_needed);
int huge_pg_setup(size_t hp_size, uint32_t npg);
bool is_vm(void);
extern int s_page_size;


#endif /* __LIBCXI_TEST_COMMON_H__ */
