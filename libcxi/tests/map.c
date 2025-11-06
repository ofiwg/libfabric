/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause */
/* Copyright 2018, 2020, 2024 Hewlett Packard Enterprise Development LP */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <linux/mman.h>
#include <fcntl.h>
#include <sched.h>

#include "libcxi_test_common.h"
#include "libcxi_priv.h"

#define N_HUGEPGS (1)
#define MIN_LEN (1024UL * 4)
#define DEF_LEN (1024UL * 1024 * 1)
#define MAX_LEN (1024UL * 1024 * 64)

#define WIN_LENGTH (0x1000)
#define GET_LENGTH (16)
#define PUT_LENGTH (16)
#define PUT_BUFFER_ID 0xb0f
#define GET_BUFFER_ID 0xa0e

//#define DEBUG_PRINT
#ifdef DEBUG_PRINT
#define DPRINT(...) cr_log_info(__VA_ARGS__)
#else
#define DPRINT(...)
#endif

uint64_t odp_request_count;

static uint64_t get_odp_request_count(void)
{
	int rc;
	uint64_t val;

	rc = cxil_read_cntr(dev, C_CNTR_ATU_ODP_REQUESTS_0, &val, NULL);
	cr_expect_eq(rc, 0);

	return val;
}

static uint64_t get_odp_request_diff(void)
{
	uint64_t diff;
	uint64_t count;

	count = get_odp_request_count();
	diff = count - odp_request_count;
	odp_request_count = count;

	return diff;
}

void odp_setup(void)
{
	data_xfer_setup();
	odp_request_count = get_odp_request_count();
}

void odp_teardown(void)
{
	data_xfer_teardown();
}

static size_t get_len(int randomize, size_t default_len)
{
	if (randomize)
		return (rand() % (MAX_LEN + 1 - MIN_LEN)) + MIN_LEN;

	return default_len;
}

static void print_good(int good_start, int bad_start, struct mem_window *dest)
{
	printf("%p - %p len:0x%-8lx ok\n",
	       dest->buffer + (good_start * sizeof(__u64)),
	       dest->buffer + (bad_start * sizeof(__u64)),
	       (bad_start - good_start) * sizeof(__u64));
}

static void print_bad(int good_start, int bad_start, bool sas, __u64 *dest_buf,
		      struct mem_window *dest)
{
	printf("%p - %p len:0x%-8lx bad value:%llx (%s)\n",
	       dest->buffer + (bad_start * sizeof(__u64)),
	       dest->buffer + (good_start * sizeof(__u64)),
	       (good_start - bad_start) * sizeof(__u64),
	       dest_buf[bad_start], sas ? "same" : "uniq");
}

static int check_data(__u64 *src_buf, __u64 *dest_buf, int count,
		       struct mem_window *dest)
{
	int i;
	bool errors = false;
	bool any_errors = false;
	int bad_start = 0;
	int good_start = 0;

	for (i = 0; i < count; i++) {
		if (src_buf[i] == dest_buf[i]) {
			if (errors) {
				int j;
				bool sas = true;

				good_start = i;

				for (j = bad_start; j < good_start; j++) {
					if (dest_buf[bad_start] !=
								dest_buf[j]) {
						sas = false;
						break;
					}
				}

				print_bad(good_start, bad_start, sas, dest_buf,
					  dest);
				errors = false;
			}
		} else if (!errors) {
			bad_start = i;
			errors = true;
			any_errors = true;

			if (good_start < bad_start)
				print_good(good_start, bad_start, dest);
		}
	}

	if (errors) {
		printf("Done :%llx %04lx - %04lx\n",
		       dest_buf[bad_start],
		       bad_start * sizeof(__u64),
		       i * sizeof(__u64));
		errors = false;
	}

	return any_errors;
}

static void alloc_map_buf(size_t len, struct mem_window *win, int flags)
{
	int rc;

	memset(&win->md, 0, sizeof(win->md));
	win->length = len;
	win->buffer = aligned_alloc(s_page_size, win->length);
	win->loc = on_host;

	cr_assert_not_null(win->buffer, "Failed to allocate iobuf");
	memset(win->buffer, 0, win->length);

	rc = cxil_map(lni, win->buffer, win->length,
		      flags, NULL, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

void do_transfer(struct mem_window *src, struct mem_window *dest)
{
	int i;
	int ret;
	int pid_idx = 0;
	__u64 *sbuf = NULL;
	__u64 *dbuf = NULL;
	__u64 *src_buf;
	__u64 *dest_buf = NULL;
	size_t len = dest->length / sizeof(__u64);

	ptlte_setup(pid_idx, false, false);
	append_le_sync(rx_pte, dest, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);

	if (src->loc == on_host) {
		src_buf = (__u64 *)src->buffer;

		for (i = 0; i < len; i++)
			src_buf[i] = i;
	} else {
		sbuf = malloc(src->length);
		cr_assert_neq(sbuf, NULL);

		for (i = 0; i < len; i++)
			sbuf[i] = i;

		memcpy_host_to_device(src, sbuf);
		src_buf = sbuf;
	}

	if (dest->loc == on_device) {
		dbuf = calloc(1, dest->length);
		cr_assert_neq(dbuf, NULL);
		dest_buf = dbuf;

		gpu_memset(dest->buffer, 0x55, dest->length);
	} else {
		dest_buf = (__u64 *)dest->buffer;
		memset(dest->buffer, 0xa5, dest->length);
	}

	do_put_sync(*src, dest->length, 0, 0, pid_idx, true, 0, 0, 0, false);

	if (dest->loc == on_device)
		memcpy_device_to_host(dbuf, dest);

	ret = check_data(src_buf, dest_buf, len, dest);
	cr_assert_not(ret, "Check data failed.");

	free(sbuf);
	free(dbuf);
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	ptlte_teardown();
}

void odp_check_event_rc(struct cxi_eq *evtq, int rc)
{
	const union c_event *event;

	while (!(event = cxi_eq_get_event(evtq)))
		sched_yield();

	if (event->hdr.return_code != C_RC_NO_EVENT)
		cr_log_info("proceess_event: type:%d size:%d rc:%d\n",
			    event->hdr.event_type, event->tgt_long.event_size,
			    event->hdr.return_code);

	if (event->hdr.return_code > C_RC_OK) {
		/* Netsim returns C_RC_NO_TRANSLATION instead of
		 * C_RC_PAGE_REQ_ERROR.
		 */
		if (is_netsim() &&  rc == C_RC_PAGE_REQ_ERROR)
			rc = C_RC_NO_TRANSLATION;

		cr_assert_eq(event->hdr.return_code, rc,
			  "Return code expected %d, got %d", rc,
			  event->hdr.return_code);
	}

	cxi_eq_ack_events(evtq);
}

TestSuite(map, .init = lni_setup, .fini = lni_teardown);

#define TASK_SIZE 0x40000000000UL

Test(map, all_vas, .disabled = true)
{
	int ret;
	uint32_t map_flags = (CXI_MAP_READ | CXI_MAP_WRITE |
			CXI_MAP_ATS);
	struct cxi_md *md;
	char *va;

	ret = cxil_map(lni, 0, -1, map_flags, NULL, &md);
	cr_assert(!ret);
	cxil_unmap(md);

	va = (char *)TASK_SIZE - 0x1000;
	map_flags = (CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_PIN);
	ret = cxil_map(lni, va, 0x2000, map_flags, NULL, &md);
	cr_assert(ret == -EINVAL);
}

struct basic_map_params {
	int test;
	int flags;
	int map_expected;
	int unmap_expected;
};

ParameterizedTestParameters(map, basic_map_test)
{
	size_t param_sz;

	static struct basic_map_params params[] = {
		{.test = 1,
		 .flags = 0,
		 .map_expected = 0,
		 .unmap_expected = 0},
		{.test = 2,
		 .flags = CXI_MAP_PIN,
		 .map_expected = 0,
		 .unmap_expected = 0},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct basic_map_params, params, param_sz);
}

ParameterizedTest(struct basic_map_params *param, map, basic_map_test)
{
	int rc = 0;
	size_t len = 4 * 1024;
	uint64_t *va;
	struct cxi_md *md = NULL;
	struct cxil_lni_priv *lni_priv = NULL;

	cr_log_info("test %d\n", param->test);

	lni_priv = container_of(lni, struct cxil_lni_priv, lni);
	if (lni_priv->dev->dev.info.cassini_version == CASSINI_1_0 && param->flags == 0) 
		cr_skip_test("Cassini 1.0 does not support ODP");

	va = aligned_alloc(s_page_size, len);
	cr_assert_neq(va, 0, "allocation failed");

	rc = cxil_map(lni, va, len, param->flags, NULL, &md);
	cr_assert_eq(rc, param->map_expected, "cxil_map failed rc:%d", rc);

	rc = cxil_unmap(md);
	cr_assert_eq(rc, param->unmap_expected, "cxil_unmap failed rc:%d", rc);

	free(va);
}

struct map_params {
	int test;
	int pin;
	size_t default_len;
	int nbuffers;
	int random_len;
	int random_pin;
};

ParameterizedTestParameters(map, map_test)
{
	size_t param_sz;

	static struct map_params params[] = {
		{.test = 1,
		 .pin = 1,
		 .default_len = DEF_LEN,
		 .nbuffers = 800,
		 .random_len = 0,
		 .random_pin = 0},
		{.test = 2,
		 .pin = 1,
		 .default_len = DEF_LEN,
		 .nbuffers = 27,
		 .random_len = 0,
		 .random_pin = 0},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct map_params, params, param_sz);
}

ParameterizedTest(struct map_params *param, map, map_test)
{
	int i;
	int rc = 0;
	int rc1 = 0;
	size_t len;
	uint64_t *va;
	uint32_t flags;
	int nallocd = 0;
	long total_allocd = 0;
	struct cxi_md *md;
	struct cxil_test_data *my_data;

	cr_log_info("test %d\n", param->test);
	flags = CXI_MAP_WRITE | CXI_MAP_READ | CXI_MAP_PIN;

	my_data = calloc(param->nbuffers, sizeof(*my_data));
	cr_assert_neq(my_data, NULL);

	srand(42);

	for (i = 0; i < param->nbuffers; i++) {
		len = get_len(param->random_len, param->default_len);

		rc = posix_memalign((void *)&va, s_page_size, len);
		if (rc != 0) {
			cr_log_info("Allocation failed for len %lx\n", len);
			break;
		}

		my_data[i].addr = va;
		my_data[i].len = len;
		total_allocd += len;
		DPRINT("addr:%p len:0x%08lx\n", va, len);
	}

	cr_log_info("Allocated %d buffers of %d, total memory %ld\n", i,
		    param->nbuffers, total_allocd);

	nallocd = i;

	for (i = 0; i < nallocd; i++) {
		rc = cxil_map(lni, my_data[i].addr, my_data[i].len, flags,
			      NULL, &md);
		cr_assert_eq(rc, 0, "map failed rc:%d\n", rc);

		my_data[i].md = md;

		if ((uint64_t)my_data[i].addr != md->va) {
			cr_log_info("va was adjusted va:%lx iova:%llx\n",
				    (uint64_t)va, md->va);
			break;
		}
	}

	cr_log_info("mapped %d of %d buffers\n", i, nallocd);
	// cr_assert_neq(i, nallocd, "%d map failed\n", i);

	for (i = 0; i < nallocd; i++) {
		if (my_data[i].md) {
			rc1 = cxil_unmap(my_data[i].md);
			cr_assert_eq(rc1, 0, "unmap failed rc:%d\n", rc1);
		}

		free((void *)my_data[i].addr);

		if (rc1)
			break;
	}

	DPRINT("Freed %d buffers\n", i);
	free(my_data);
}

TestSuite(map_xfer, .init = data_xfer_setup, .fini = data_xfer_teardown);

struct map_xfer_params {
	int test;
	size_t buf_len;
	size_t increment;
	size_t xfer_len;
};

ParameterizedTestParameters(map_xfer, puts)
{
	size_t param_sz;

	/* TODO: enable ATU caching after emulator is fixed */
	static struct map_xfer_params params[] = {
		{.test = 1,
		 .buf_len = 0x1000,
		 .increment = 0,
		 .xfer_len = 0x1000},
		{.test = 2,
		 .buf_len = 0x2000,
		 .increment = 0,
		 .xfer_len = 0x2000},
		{.test = 3,
		 .buf_len = 0x4000,
		 .increment = 0x1000,
		 .xfer_len = 0x1000},
		{.test = 4,
		 .buf_len = 0x8000,
		 .increment = 0,
		 .xfer_len = 0x8000},
		{.test = 5,
		 .buf_len = 0x10000,
		 .increment = 0,
		 .xfer_len = 0x10000},
		{.test = 6,
		 .buf_len = 0x10000,
		 .increment = 0,
		 .xfer_len = 0x10000},
		{.test = 7,
		 .buf_len = 0x40000,
		 .increment = 0,
		 .xfer_len = 0x40000},
		{.test = 8,
		 .buf_len = 0x80000,
		 .increment = 0x8000,
		 .xfer_len = 0x1000},
		{.test = 9,
		 .buf_len = 0x100000,
		 .increment = 0,
		 .xfer_len = 0x100000},
		{.test = 10,
		 .buf_len = 0x200000,
		 .increment = 0,
		 .xfer_len = 0x200000},
		{.test = 11,
		 .buf_len = 0x40000000,
		 .increment = 0x10000000,
		 .xfer_len = 0x10000},
		{.test = 12,
		 .buf_len = 0x80000000,
		 .increment = 0x8000000,
		 .xfer_len = 0x1000},
		{.test = 13,
		 .buf_len =  0x100000000,
		 .increment = 0x10000000,
		 .xfer_len = 0x10000},
		{.test = 14,
		 .buf_len = 0x100000000,
		 .increment = 0x10000000,
		 .xfer_len = 0x1000},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct map_xfer_params, params, param_sz);
}

ParameterizedTest(struct map_xfer_params *param, map_xfer, puts)
{
	int rc;
	int i = 0;
	size_t len;
	int off = 0;
	int pid_idx = 0;
	int errors = 0;
	int map_flags = CXI_MAP_PIN | CXI_MAP_NOCACHE;
	size_t inc = !param->increment ? param->xfer_len : param->increment;
	struct mem_window snd_mem = {};
	struct mem_window rma_mem = {};
	size_t xfer_len = param->xfer_len;

	/* Allocate buffers */
	len = param->buf_len;

	cr_log_info("test %d, len %ld inc %ld\n", param->test, len, inc);

	snd_mem.length = len;
	snd_mem.buffer = aligned_alloc(s_page_size, snd_mem.length);
	if (!snd_mem.buffer)
		cr_skip_test("Not enough memory to allocate %ld bytes", len);

	rma_mem.length = xfer_len;
	rma_mem.buffer = aligned_alloc(s_page_size, rma_mem.length);
	cr_assert_not_null(rma_mem.buffer, "RMA buffer allocation failed.");

	/* Initialize Send Memory */
	for (int i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

	rc = cxil_map(lni, rma_mem.buffer, rma_mem.length,
		      map_flags | CXI_MAP_WRITE, NULL, &rma_mem.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	rc = cxil_map(lni, snd_mem.buffer, snd_mem.length,
		      map_flags | CXI_MAP_READ, NULL, &snd_mem.md);
	cr_assert_eq(rc, 0, "Send MD cxil_map() failed %d", rc);

	/* Initialize RMA PtlTE and Post RMA Buffer */
	ptlte_setup(pid_idx, false, false);
	append_le(rx_pte, &rma_mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		  0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false, true,
		  true, true);
	process_eqe(target_evtq, EQE_TGT_LONG, C_EVENT_LINK,
		    PUT_BUFFER_ID, NULL);

	for (off = 0; off < param->buf_len; off += inc) {
		memset(rma_mem.buffer, 0, xfer_len);
		do_put_sync(snd_mem, xfer_len, 0, off, pid_idx, true, 0, 0, 0,
			    false);

		/* Validate Source and Destination Data Match */
		for (i = 0; i < param->xfer_len; i++) {
			cr_expect_eq(snd_mem.buffer[off + i],
				     rma_mem.buffer[i],
				     "%d Data mismatch: idx %2d - %02x != %02x",
				     param->test, off + i,
				     snd_mem.buffer[off + i],
				     rma_mem.buffer[i]);
			if (snd_mem.buffer[off + i] != rma_mem.buffer[i])
				errors++;

			if (errors > 10)
				goto error;
		}
	}

error:
	/* Clean up PTE and RMA buffer */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	ptlte_teardown();

	/* Unmap Memory */
	rc = cxil_unmap(snd_mem.md);
	cr_expect_eq(rc, 0, "Unmap of send MD Failed %d", rc);
	rc = cxil_unmap(rma_mem.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	/* Deallocate buffers */
	free(snd_mem.buffer);
	free(rma_mem.buffer);
}

#define PAGE_SHIFT 12
#define HPAGE_SHIFT 21
#define PAGE_SIZE (1 << PAGE_SHIFT)
#define HPAGE_SIZE (1 << HPAGE_SHIFT)
#define PAGEMAP_PRESENT(ent)	(((ent) & (1ull << 63)) != 0)
#define PAGEMAP_PFN(ent)	((ent) & ((1ull << 55) - 1))
#define HPAGE_PFNS(hp_order) (1 << (hp_order - PAGE_SHIFT))

int pagemap_fd;

void is_hp(void *ptr, int hp_order)
{
	int i;
	uint64_t ent;
	uint64_t prev;
	uintptr_t addr = (uintptr_t)ptr >> (PAGE_SHIFT - 3);
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);

	cr_assert_geq(pagemap_fd, 0, "open pagemap failed %d", pagemap_fd);

	if (!hp_order)
		hp_order = ffs(s_page_size) - 1;

	for (i = 0; i < HPAGE_PFNS(hp_order); i++) {
		if (pread(pagemap_fd, &ent, sizeof(ent), addr) != sizeof(ent))
			cr_assert("read pagemap");

		if (!PAGEMAP_PRESENT(ent)) {
			cr_log_info("%s NP ptr:%p addr:%lx ent[%d]:%lx\n",
				    __func__, ptr, addr, i, ent);
			close(pagemap_fd);
			return;
		}

		/* check first entry for alignment */
		if (!i && (PAGEMAP_PFN(ent) & (HPAGE_PFNS(hp_order) - 1))) {
			cr_log_info("%s NA ent[%d]:%lx\n", __func__, i, ent);
			close(pagemap_fd);
			return;
		}

		/* check if contiguous with previous entry */
		if (i && ((PAGEMAP_PFN(prev) + 1) != PAGEMAP_PFN(ent))) {
			cr_log_info("%s NC prev:%lx ent[%d]:%lx\n", __func__,
				    prev, i, ent);
			close(pagemap_fd);
			return;
		}

		prev = ent;
		addr += sizeof(addr);
	}

	close(pagemap_fd);
	cr_log_info("mmap'd %d hugepage\n", hp_order);
}

int thp(int nr_hugepages)
{
	int rc;
	void *ptr;
	size_t len;
	size_t ram;
	uint64_t ent[2];
	int flags = CXI_MAP_PIN;
	struct mem_window mmap_m = {};
	struct mem_window rma_mem = {};

	ram = sysconf(_SC_PHYS_PAGES);
	if (ram > SIZE_MAX / sysconf(_SC_PAGESIZE) / 4)
		ram = SIZE_MAX / 4;
	else
		ram *= sysconf(_SC_PAGESIZE);

	if (nr_hugepages == 0)
		len = ram / 4;
	else
		len = nr_hugepages << HPAGE_SHIFT;

	cr_log_info("mapping len %lx\n", len);

	ptr = mmap(NULL, len + HPAGE_SIZE, PROT_READ | PROT_WRITE,
			MAP_ANONYMOUS | MAP_NORESERVE | MAP_PRIVATE, -1, 0);
	cr_assert_neq(ptr, MAP_FAILED, "initial mmap");

	rc = posix_memalign(&ptr, HPAGE_SIZE, len);
	cr_assert_eq(rc, 0, "posix_memalign");

	if (madvise(ptr, len, MADV_HUGEPAGE))
		cr_assert("MADV_HUGEPAGE");

	if (pread(pagemap_fd, ent, sizeof(ent),
			(uintptr_t)ptr >> (PAGE_SHIFT - 3)) != sizeof(ent))
		cr_assert("read pagemap");

	if (PAGEMAP_PRESENT(ent[0]) && PAGEMAP_PRESENT(ent[1]) &&
	    PAGEMAP_PFN(ent[0]) + 1 == PAGEMAP_PFN(ent[1]) &&
	    !(PAGEMAP_PFN(ent[0]) & ((1 << (HPAGE_SHIFT - PAGE_SHIFT)) - 1)))
		cr_log_info("Mapped transparent hugepage\n");

	rma_mem.length = s_page_size * 32;
	rma_mem.buffer = aligned_alloc(s_page_size, rma_mem.length);
	cr_assert_not_null(rma_mem.buffer, "allocation failed.");

	rc = cxil_map(lni, rma_mem.buffer, rma_mem.length,
		      flags | CXI_MAP_WRITE, NULL, &rma_mem.md);
	cr_assert_eq(rc, 0, "map failed %d", rc);

	mmap_m.buffer = ptr;
	mmap_m.length = HPAGE_SIZE;

	rc = cxil_map(lni, mmap_m.buffer, mmap_m.length, flags | CXI_MAP_READ,
		      NULL, &mmap_m.md);
	cr_assert_eq(rc, 0, "map failed %d", rc);

	do_transfer(&mmap_m, &rma_mem);

	rc = cxil_unmap(mmap_m.md);
	cr_assert_eq(rc, 0, "Unmap Failed %d", rc);

	rc = cxil_unmap(rma_mem.md);
	cr_assert_eq(rc, 0, "Unmap Failed %d", rc);
	free(rma_mem.buffer);

	free(ptr);

	return 0;
}

Test(map_xfer, thp, .disabled = false, .timeout = 10)
{
	pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	cr_assert_geq(pagemap_fd, 0, "open pagemap failed %d", pagemap_fd);

	thp(200);
	thp(100);
	thp(8);

	close(pagemap_fd);
}

Test(map, notifier1)
{
	int rc;
	size_t len = 1UL * 1024 * 1024;
	int flags = CXI_MAP_PIN | CXI_MAP_WRITE;
	struct mem_window mem1 = {};
	struct mem_window mem2 = {};

	mem1.length = len;
	mem1.buffer = aligned_alloc(s_page_size, mem1.length);
	cr_assert_not_null(mem1.buffer, "Buffer allocation failed.");

	rc = cxil_map(lni, mem1.buffer, mem1.length, flags, NULL, &mem1.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	mem2.length = len * 2;
	mem2.buffer = aligned_alloc(s_page_size, mem2.length);
	cr_assert_not_null(mem2.buffer, "Buffer allocation failed.");

	rc = cxil_map(lni, mem2.buffer, mem2.length, flags, NULL, &mem2.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	/* Deallocate buffer before unmapping */
	free(mem1.buffer);

	rc = cxil_unmap(mem1.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	/* Deallocate buffer before unmapping */
	free(mem2.buffer);

	rc = cxil_unmap(mem2.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);
}

Test(map, notifier2)
{
	int rc;
	size_t len = 1UL * 1024 * 1024;
	size_t map_len = len / 2;
	int flags = CXI_MAP_PIN | CXI_MAP_WRITE;
	struct mem_window mem1 = {};
	struct mem_window mem2 = {};

	/*
	 * allocate memory and map parts of it in separate MDs
	 */
	mem1.length = len;
	mem1.buffer = aligned_alloc(s_page_size, mem1.length);
	cr_assert_not_null(mem1.buffer, "Buffer allocation failed.");

	cr_log_info("allocated %p to %p\n", mem1.buffer, mem1.buffer + len);

	mem2.buffer = mem1.buffer + map_len;

	rc = cxil_map(lni, mem1.buffer, map_len, flags, NULL, &mem1.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	rc = cxil_map(lni, mem2.buffer, map_len, flags, NULL, &mem2.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	/* Deallocate buffer before unmapping */
	free(mem1.buffer);

	rc = cxil_unmap(mem1.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	rc = cxil_unmap(mem2.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);
}

Test(map_xfer, simple)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	size_t len = 8UL * 1024 * 1024;

	/* Allocate buffers */
	alloc_iobuf(len, &dst_mem, CXI_MAP_WRITE);
	alloc_iobuf(len, &src_mem, CXI_MAP_READ);

	do_transfer(&src_mem, &dst_mem);

	free_iobuf(&src_mem);
	free_iobuf(&dst_mem);
}

Test(map, overlap)
{
	int rc;
	uint8_t *mmap_buf;
	size_t map_len = 8 * 1024;
	struct mem_window mem1 = {.length = map_len};
	struct mem_window mem2 = {.length = map_len * 2};
	size_t mmap_len = 8UL * 1024 * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	uint8_t *addr = (uint8_t *)0x100000000;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
	int flags = CXI_MAP_PIN | CXI_MAP_WRITE | CXI_MAP_READ;

	mmap_buf = mmap(addr, mmap_len, prot, mmap_flags, 0, 0);
	if (mmap_buf != addr)
		cr_skip_test("Could not mmap memory at %p", addr);
	cr_assert_neq(mmap_buf, MAP_FAILED, "mmap failed. %p", mmap_buf);
	cr_log_info("mmapped %p to %p\n", mmap_buf, mmap_buf + mmap_len);

	/* map parts of the mmapped range in separate MDs */
	/* in top half of mmapped region */
	mem1.buffer = addr + (mmap_len / 2);
	/* cross over the 4 GB boundary */
	mem2.buffer = mem1.buffer - (mem2.length / 2);

	cr_log_info("b1:%p b2:%p\n", mem1.buffer, mem2.buffer);

	rc = cxil_map(lni, mem1.buffer, mem1.length, flags, NULL, &mem1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_map(lni, mem2.buffer, mem2.length, flags, NULL, &mem2.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_unmap(mem1.md);
	cr_expect_eq(rc, 0, "Unmap of MD Failed %d", rc);

	rc = cxil_unmap(mem2.md);
	cr_expect_eq(rc, 0, "Unmap of MD Failed %d", rc);

	munmap(mmap_buf, mmap_len);
}

struct region_params {
	int test;
	uint8_t *addr;
	size_t  mmap_len;
	size_t  map_len;
	size_t  offset;
};

ParameterizedTestParameters(map, region)
{
	size_t param_sz;

	static struct region_params params[] = {
		{.test = 1,
		 .addr = (uint8_t *)0x00100000,
		 .mmap_len = 0x1000,
		 .map_len = 0x1000,
		 .offset = 0},
		{.test = 2,
		 .addr = (uint8_t *)0x100000000, // 4 GB
		 .mmap_len = 8UL * ONE_GB,
		 .map_len = 8 * 1024,
		 .offset = 0},
		{.test = 3,
		 .addr = (uint8_t *)0x100000000,
		 .mmap_len = 8UL * ONE_GB,
		 .map_len = 3UL * ONE_GB + TWO_MB,
		 .offset = 1UL * ONE_GB},
		{.test = 4,
		 .addr = (uint8_t *)0x100000000,
		 .mmap_len = 8UL * ONE_GB,
		 .map_len = 5UL * ONE_GB,
		 .offset = 0},
		{.test = 5,
		 .addr = (uint8_t *)0x200000000,
		 .mmap_len = 8UL * ONE_GB,
		 .map_len = 5UL * ONE_GB,
		 .offset = 0},
		{.test = 6,
		 .addr = (uint8_t *)0x100000000,
		 .mmap_len = 8UL * ONE_GB,
		 .map_len = 4UL * ONE_GB - (4 * TWO_MB),
		 .offset = 4 * TWO_MB},
		{.test = 7,
		 .addr = (uint8_t *)0x100000000,
		 .mmap_len = 16UL * ONE_GB,
		 .map_len = 8UL * ONE_GB,
		 .offset = 1UL * ONE_GB},
		{.test = 8,
		 .addr = (uint8_t *)0x100000000,
		 .mmap_len = 4UL * ONE_GB,
		 .map_len = 5UL * ONE_GB,
		 .offset = 0UL * ONE_GB},
	};

	param_sz = ARRAY_SIZE(params);
	return cr_make_param_array(struct region_params, params, param_sz);
}

ParameterizedTest(struct region_params *param, map, region)
{
	int rc;
	uint8_t *mmap_buf;
	struct mem_window mem = {.length = param->map_len};
	size_t mmap_len = param->mmap_len;
	size_t map_len = param->map_len;
	int prot = PROT_WRITE | PROT_READ;
	uint8_t *addr = param->addr;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS; // | MAP_FIXED;
	int flags = CXI_MAP_PIN | CXI_MAP_WRITE | CXI_MAP_READ;

	mmap_buf = mmap(addr, mmap_len, prot, mmap_flags, 0, 0);
	if (mmap_buf != addr)
		cr_skip_test("Could not mmap memory at %p", addr);
	cr_assert_neq(mmap_buf, MAP_FAILED, "mmap failed. %p", mmap_buf);

	/* map parts of the mmapped range in separate MDs */
	mem.length = map_len;
	/* in top half of mmapped region */
	mem.buffer = addr + param->offset;

	rc = cxil_map(lni, mem.buffer, mem.length, flags, NULL, &mem.md);
	if (mmap_len >= map_len)
		cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);
	else
		cr_assert_neq(rc, 0, "RMA MD cxil_map() should fail %d", rc);

	if (!rc) {
		rc = cxil_unmap(mem.md);
		cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);
	}

	munmap(mmap_buf, mmap_len);
}

Test(map_xfer, notifier)
{
	int rc;
	// int errors = 0;
	int pid_idx = 0;
	size_t len = 1UL * 1024 * 128;
	struct mem_window snd_mem;
	struct mem_window rma_mem;
	// const union c_event *event;

	alloc_iobuf(len, &rma_mem, CXI_MAP_WRITE);
	alloc_iobuf(len, &snd_mem, CXI_MAP_READ);

	/*
	 * Deallocate buffer before sending.
	 * Causes MMU notifier to invalidate mapped memory.
	 */
	free(rma_mem.buffer);

	ptlte_setup(pid_idx, false, false);

	append_le_sync(rx_pte, &rma_mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);

	cr_log_info("initialize data\n");
	for (int i = 0; i < snd_mem.length; i++)
		snd_mem.buffer[i] = i;

#if 0
	/*
	 * The emulator gets an translation error but doesn't return an event
	 * indicating the error so the test hangs. Enable this when the
	 * emulator is updated.
	 */
	cr_log_info("send data\n");
	do_put_sync(snd_mem, len, 0, 0, pid_idx, true, 0, 0, 0);

	/* will probably need to look for an error event */
	do_put(snd_mem, len, 0, 0, pid_idx, true, 0, 0, 0);
	process_eqe(transmit_evtq, EQE_INIT_SHORT, C_EVENT_ERROR,
		    user_ptr, NULL);

	while (!(event = cxi_eq_get_event(transmit_evtq)))
		sched_yield();

	cr_log_info("checking data\n");
	/* Validate Source and Destination Data Match */
	for (int i = 0; i < len; i++) {
		cr_expect_eq(snd_mem.buffer[i], rma_mem.buffer[i],
			     "Data mismatch: idx %2d - %02x != %02x",
			     i, snd_mem.buffer[i],
			     rma_mem.buffer[i]);
		if (snd_mem.buffer[i] != rma_mem.buffer[i])
			errors++;

		if (errors > 10)
			goto error;
	}

	cr_log_info("data received\n");

error:
#endif
	/* Clean up PTE and RMA buffer */
	unlink_le_sync(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	ptlte_teardown();

	rc = cxil_unmap(snd_mem.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	/* Deallocate buffer before unmapping */

	rc = cxil_unmap(rma_mem.md);
	cr_expect_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	free(snd_mem.buffer);
}

Test(map_xfer, hugepage)
{
	int rc;
	size_t len = 8UL * 1024 * 1024;
	size_t mmap_len = 8UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
			 MAP_HUGE_1GB;
	int flags = CXI_MAP_PIN;
	struct mem_window rma_mem = {};
	struct mem_window mmap_m = {};
	struct cxi_md_hints hints = {};
	int page_shift = 18;
	int huge_shift = 30;
	int hp_cnt;

	hp_cnt = huge_pg_setup(ONE_GB, N_HUGEPGS);
	if (hp_cnt < 0)
		cr_skip_test("1 GB hugepage not available");
	check_huge_pg_free(ONE_GB, 1);

	cr_log_info("ps:%d hs:%d\n", page_shift, huge_shift);
	mmap_m.buffer = mmap(NULL, mmap_len, prot, mmap_flags, 0, 0);
	cr_assert_not_null(mmap_m.buffer, "mmap failed. %p", mmap_m.buffer);
	cr_assert_neq(mmap_m.buffer, MAP_FAILED, "mmap failed. %p",
		      mmap_m.buffer);

	mmap_m.length = len;
	hints.page_shift = page_shift;
	hints.huge_shift = huge_shift;
	rc = cxil_map(lni, mmap_m.buffer, mmap_m.length,
		      flags | CXI_MAP_READ, &hints, &mmap_m.md);
	cr_assert_eq(rc, 0, "cxil_map() of mmap_m failed %d", rc);

	rma_mem.length = len;
	rma_mem.buffer = aligned_alloc(s_page_size, rma_mem.length);
	cr_assert_not_null(rma_mem.buffer, "RMA buffer allocation failed.");

	rc = cxil_map(lni, rma_mem.buffer, rma_mem.length,
		      flags | CXI_MAP_WRITE, &hints, &rma_mem.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	cr_log_info("rma_mem iova:%llx len:%lx\n", rma_mem.md->iova,
		    rma_mem.md->len);

	do_transfer(&mmap_m, &rma_mem);

	rc = cxil_unmap(rma_mem.md);
	cr_assert_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	free(rma_mem.buffer);

	rc = cxil_unmap(mmap_m.md);
	cr_assert_eq(rc, 0, "Unmap of mmap MD Failed %d", rc);

	cr_log_info("munmap\n");
	munmap(mmap_m.buffer, mmap_len);

	huge_pg_setup(ONE_GB, hp_cnt);
}

void hugepage_test(int page_shift, int huge_shift, void *addr)
{
	int rc;
	size_t len = 8UL * 1024 * 1024;
	int flags = CXI_MAP_PIN;
	struct mem_window mmap_m = {};
	struct mem_window rma_mem = {};
	struct cxi_md_hints hints = {};

	mmap_m.length = len;
	mmap_m.buffer = addr;
	hints.page_shift = page_shift;
	hints.huge_shift = huge_shift;

	rc = cxil_map(lni, mmap_m.buffer, mmap_m.length,
		      flags | CXI_MAP_READ, &hints, &mmap_m.md);
	cr_assert_eq(rc, 0, "cxil_map() of mmap_m failed %d", rc);

	rma_mem.length = len;
	rma_mem.buffer = aligned_alloc(s_page_size, rma_mem.length);
	cr_assert_not_null(rma_mem.buffer, "RMA buffer allocation failed.");

	rc = cxil_map(lni, rma_mem.buffer, rma_mem.length,
		      flags | CXI_MAP_WRITE, &hints, &rma_mem.md);
	cr_assert_eq(rc, 0, "RMA MD cxil_map() failed %d", rc);

	do_transfer(&mmap_m, &rma_mem);

	rc = cxil_unmap(rma_mem.md);
	cr_assert_eq(rc, 0, "Unmap of RMA MD Failed %d", rc);

	free(rma_mem.buffer);

	rc = cxil_unmap(mmap_m.md);
	cr_assert_eq(rc, 0, "Unmap of mmap MD Failed %d", rc);
}

#define PAGE_SHIFT_MIN 12
#define PAGE_SHIFT_MAX (PAGE_SHIFT_MIN + 7)
#define HUGE_SHIFT_MIN 21
#define HUGE_SHIFT_MAX (HUGE_SHIFT_MIN + 10)

TestSuite(map_hp);
Test(map_hp, hugepage_suite, .timeout = 15)
{
	int i;
	int ps;
	int hs;
	int rc;
	void *buffer;
	size_t len = 1024UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB |
			 MAP_HUGE_1GB;
	int hp_cnt;

	hp_cnt = huge_pg_setup(ONE_GB, N_HUGEPGS);
	if (hp_cnt < 0)
		cr_skip_test("1 GB hugepage not available");
	check_huge_pg_free(ONE_GB, N_HUGEPGS);

	data_xfer_setup();

	for (i = 0, hs = HUGE_SHIFT_MIN; hs < HUGE_SHIFT_MAX; hs += 2) {
		for (ps = PAGE_SHIFT_MIN; ps < PAGE_SHIFT_MAX; ps += 2) {
			if ((hs - ps) > 15)
				continue;

			cr_log_info("ps:%d hs:%d\n", ps, hs);

			buffer = mmap(NULL, len, prot, mmap_flags, 0, 0);
			cr_assert_not_null(buffer, "mmap failed. %p", buffer);
			cr_assert_neq(buffer, MAP_FAILED,
				      "mmap failed. ret:%p errno:%d", buffer,
				      errno);

			hugepage_test(ps, hs, buffer);

			rc = munmap(buffer, len);
			cr_assert_eq(rc, 0, "munmap %d, errno %d", rc, errno);

			/* Each cxil_map of a new page size uses an lac.
			 * There are only 8 lacs available per LNI.
			 */
			if (++i % 6 == 0) {
				data_xfer_teardown();
				data_xfer_setup();
			}
		}

	}

	huge_pg_setup(ONE_GB, hp_cnt);
	data_xfer_teardown();
}

void mmap_map(struct mem_window *win, size_t len, int hp_order, int map_flags)
{
	int rc;
	int prot = PROT_WRITE | PROT_READ;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB;

	mmap_flags |= hp_order << MAP_HUGE_SHIFT;
	map_flags |= CXI_MAP_PIN;

	memset(&win->md, 0, sizeof(win->md));
	win->loc = on_host;
	win->length = len;
	win->length = len;
	win->buffer = mmap(NULL, len, prot, mmap_flags, 0, 0);
	cr_assert_not_null(win->buffer, "mmap failed. %p", win->buffer);

	/* skip test if hugepages are unavailable */
	if (win->buffer == MAP_FAILED)
		cr_skip_test("mmap failed. ret:%p errno:%d", win->buffer,
			     -errno);

	memset(win->buffer, 0, win->length);
	rc = cxil_map(lni, win->buffer, win->length, map_flags, NULL, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

void unmap_mem(struct mem_window *win)
{
	int rc = cxil_unmap(win->md);

	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);
	munmap(win->buffer, win->length);
}

Test(map_hp, hugepage_suite2, .timeout = 15)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	size_t len = 8UL * 1024 * 1024;
	size_t page_size = 2 * 1024 * 1024;
	int hp_cnt;
	int hp_order;
	int npages = len / page_size;

	for (hp_order = 21; hp_order < 31; hp_order++) {
		page_size = 1 << hp_order;
		npages = len / page_size;
		npages = !npages ? 1 : npages;
		npages *= 2;
		cr_log_info("page_size:%ld npages:%d\n", page_size, npages);
		hp_cnt = huge_pg_setup(page_size, npages);
		if (hp_cnt < 0)
			break;

		data_xfer_setup();

		cr_log_info("page_size:%ld hp_order:%d\n", page_size, hp_order);
		mmap_map(&dst_mem, len, hp_order, CXI_MAP_WRITE);
		cr_log_info("cxil_page_size:%ld\n",
			    cxil_page_size(dst_mem.buffer));
		mmap_map(&src_mem, len, hp_order, CXI_MAP_READ);
		cr_log_info("cxil_page_size:%ld\n",
			    cxil_page_size(src_mem.buffer));

		do_transfer(&src_mem, &dst_mem);

		unmap_mem(&src_mem);
		unmap_mem(&dst_mem);

		huge_pg_setup(page_size, hp_cnt);
		data_xfer_teardown();
	}
}

#define HIGH_BASE 0x30000000000
#define HIGH_BPTR ((void *)HIGH_BASE)

int mmap_hp(struct mem_window *win, void *addr, int prot, size_t len,
	    int hp_order, bool touch)
{
	int hp_cnt = 0;
	int npages;
	size_t page_size = 1 << hp_order;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;

	npages = len / page_size;
	npages = !npages ? 1 : npages;

	cr_log_info("page_size:%ld npages:%d\n", page_size, npages);

	if (hp_order > 12) {
		hp_cnt = huge_pg_setup(page_size, npages * 2);
		mmap_flags |= MAP_HUGETLB | (hp_order << MAP_HUGE_SHIFT);
	}

	win->loc = on_host;
	win->length = len;
	win->buffer = mmap(addr, len, prot, mmap_flags, 0, 0);
	cr_assert_not_null(win->buffer, "mmap failed. %p", win->buffer);
	if (addr)
		cr_assert_eq(win->buffer, addr, "failed to use addr (%p) hint (%p)",
			     addr, win->buffer);

	/* skip test if hugepages are unavailable */
	if (win->buffer == MAP_FAILED) {
		cr_skip_test("mmap failed errno:%d", -errno);
		perror("mmap");
	}

	if (touch) {
		memset(win->buffer, 0, win->length);
		is_hp(win->buffer, hp_order);
	}

	return hp_cnt;
}

void reg_win(struct mem_window *win, struct cxi_md_hints *hints, int flags)
{
	int rc;

	if (!hints)
		memset(&win->md, 0, sizeof(win->md));

	rc = cxil_map(lni, win->buffer, win->length, flags, hints, &win->md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
}

Test(map_hp, odp_hp0)
{
	int hp_order;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	struct cxi_md_hints hints = {};
	size_t len = 256UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;

	data_xfer_setup();

	hp_order = 21;
	mmap_hp(&src_mem, NULL, prot, len, hp_order, true);
	cr_log_info("src buf page size:%ld\n", cxil_page_size(src_mem.buffer));

	mmap_hp(&dst_mem, NULL, prot, len, hp_order, false);
	cr_log_info("dst buf page size:%ld\n", cxil_page_size(dst_mem.buffer));

	reg_win(&src_mem, NULL, CXI_MAP_FAULT | CXI_MAP_READ);
	hints.huge_shift = hp_order;
	reg_win(&dst_mem, &hints, CXI_MAP_WRITE);

	do_transfer(&src_mem, &dst_mem);

	unmap_mem(&src_mem);
	unmap_mem(&dst_mem);

	data_xfer_teardown();
}

/*
 * Mmap two contiguous buffers with different hugepage sizes
 */
Test(map_hp, odp_hp1)
{
	int rc;
	void *map_buf;
	struct cxi_md *md;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	size_t len = 256UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	int hp_order;

	data_xfer_setup();

	hp_order = 24;
	mmap_hp(&src_mem, NULL, prot, len, hp_order, true);
	cr_log_info("src buf:%p page size:%ld\n", src_mem.buffer,
		    cxil_page_size(src_mem.buffer));

	hp_order = 22;
	mmap_hp(&dst_mem, NULL, prot, len, hp_order, true);
	cr_log_info("dst buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));

	if (dst_mem.buffer < src_mem.buffer) {
		map_buf = dst_mem.buffer;
		cr_assert_eq(dst_mem.buffer + len, src_mem.buffer,
			     "buffers not contiguous");
	} else {
		map_buf = src_mem.buffer;
		cr_assert_eq(src_mem.buffer + len, dst_mem.buffer,
			     "buffers not contiguous");
	}

	rc = cxil_map(lni, map_buf, src_mem.length + dst_mem.length,
		      CXI_MAP_READ | CXI_MAP_WRITE, NULL, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_assert_eq(hp_order, md->huge_shift, "huge_shift incorrect %d",
		     md->huge_shift);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	src_mem.md = md;
	dst_mem.md = md;

	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(dst_mem.buffer, dst_mem.length);
	munmap(src_mem.buffer, src_mem.length);

	data_xfer_teardown();
}

/*
 * Mmap three contiguous buffers with no hugepages in the middle one
 * therefore the AC is configured using the default hugepage size.
 */
Test(map_hp, odp_hp2)
{
	int rc;
	void *map_buf;
	struct cxi_md *md;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	struct mem_window tmp_mem;
	size_t len = 256UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	int hp_order;

	data_xfer_setup();

	hp_order = 24;
	mmap_hp(&src_mem, NULL, prot, len, hp_order, true);
	cr_log_info("src buf:%p page size:%ld\n", src_mem.buffer,
		    cxil_page_size(src_mem.buffer));

	hp_order = 0;
	mmap_hp(&tmp_mem, NULL, prot, len, hp_order, true);
	cr_log_info("src buf:%p page size:%ld\n", tmp_mem.buffer,
		    cxil_page_size(tmp_mem.buffer));

	hp_order = 24;
	mmap_hp(&dst_mem, NULL, prot, len, hp_order, false);
	cr_log_info("dst buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));

	map_buf = dst_mem.buffer;

	rc = cxil_map(lni, map_buf, src_mem.length * 3,
		      CXI_MAP_READ | CXI_MAP_WRITE, NULL, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_assert_eq(21, md->huge_shift, "huge_shift incorrect %d",
		     md->huge_shift);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	src_mem.md = md;
	dst_mem.md = md;

	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(dst_mem.buffer, dst_mem.length);
	munmap(src_mem.buffer, src_mem.length);

	data_xfer_teardown();
}

void dump_vma_range(const uintptr_t addr, size_t len)
{
	int ret;
	FILE *file;
	char mode[5];
	char buf[BUFSIZ];
	char rest[BUFSIZ];
	char mapsf[BUFSIZ];
	uintptr_t low;
	uintptr_t high;
	uintptr_t end = addr + len;
	pid_t pid = getpid();

	snprintf(mapsf, sizeof(mapsf), "/proc/%d/maps", pid);
	file = fopen(mapsf, "r");

	while (fgets(buf, sizeof(buf), file) != NULL) {
		ret = sscanf(buf, "%lx-%lx %4c %s", &low, &high, mode, rest);
		cr_assert_eq(ret, 4, "failed to get values. ret %d", ret);

		if ((low <= addr && high > addr) ||
				(end > low && end <= high)) {
			cr_log_info("addr:%jx end:%jx len:%lx vma %jx-%jx %s %s\n",
				    addr, end, len, low, high, mode, rest);
		} else if (low > end) {
			break;
		}
	}

	fclose(file);
}

TestSuite(odp, .init = odp_setup, .fini = odp_teardown, .timeout = 5);

Test(odp, odp_success)
{
	struct mem_window src_mem;
	struct mem_window dst_mem;
	uint32_t flags = CXI_MAP_READ | CXI_MAP_WRITE;
	size_t len = 1UL * 1024 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	ulong none_odp_requests;
	ulong fault_odp_requests;
	ulong prefetch_odp_requests;

	/* Verify ODP */
	alloc_map_buf(len, &src_mem, flags | CXI_MAP_PIN);
	alloc_map_buf(len, &dst_mem, flags);

	do_transfer(&src_mem, &dst_mem);

	free_iobuf(&dst_mem);
	none_odp_requests = get_odp_request_diff();

	/* Verify ODP + CXI_MAP_FAULT */
	mmap_hp(&dst_mem, HIGH_BPTR, prot, len, 12, false);
	reg_win(&dst_mem, NULL, flags | CXI_MAP_FAULT);

	do_transfer(&src_mem, &dst_mem);
	unmap_mem(&dst_mem);

	/* check for no odp requests */
	fault_odp_requests = get_odp_request_diff();
	cr_assert_eq(0, fault_odp_requests, "Should have no odp requests have %ld\n",
		     fault_odp_requests);

	/* Verify ODP + CXI_MAP_PREFETCH */
	mmap_hp(&dst_mem, HIGH_BPTR, prot, len, 12, false);
	/* Touch half of the pages */
	memset(dst_mem.buffer, 0, len / 2);
	reg_win(&dst_mem, NULL, flags | CXI_MAP_PREFETCH);

	do_transfer(&src_mem, &dst_mem);
	unmap_mem(&dst_mem);

	/* check for some odp requests */
	prefetch_odp_requests = get_odp_request_diff();

	/* Netsim does not count ODP requests */
	if (!is_netsim())
		cr_assert_lt(prefetch_odp_requests, none_odp_requests,
			     "prefetch requests (%ld) should be less than none requests (%ld)\n",
			     prefetch_odp_requests, none_odp_requests);
	free_iobuf(&src_mem);
}

ParameterizedTestParameters(odp, odp_fail)
{
	size_t param_sz;
	static int extra_flag[] = {0, CXI_MAP_PREFETCH};

	param_sz = ARRAY_SIZE(extra_flag);
	return cr_make_param_array(int, extra_flag, param_sz);
}

/* ODP host memory failures:
 *    Invalid flag combinations:
 *        CXI_MAP_PIN & CXI_MAP_FAULT
 *        CXI_MAP_PIN & CXI_MAP_PREFETCH
 *    VA not backed by VMA
 *    VA not backed by VMA and CXI_MAP_PREFETCH
 */
ParameterizedTest(int *extra_flag, odp, odp_fail)
{
	int rc;
	int pid_idx = 0;
	union c_event *event;
	struct mem_window mem;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	uint32_t flags = CXI_MAP_READ | CXI_MAP_WRITE;
	size_t len = 1UL * 1024 * 1024;

	mem.length = s_page_size;
	mem.buffer = aligned_alloc(s_page_size, mem.length);

	/* Invalid combination of flags - pin and fault */
	rc = cxil_map(lni, mem.buffer, mem.length,
		      flags | CXI_MAP_PIN | CXI_MAP_FAULT, NULL,
		      &mem.md);
	cr_assert_eq(rc, -EINVAL, "cxil_map() should fail invalid flags %d",
		     rc);

	/* Invalid combination of flags - pin and prefetch */
	rc = cxil_map(lni, mem.buffer, mem.length,
		      flags | CXI_MAP_PIN | CXI_MAP_PREFETCH, NULL,
		      &mem.md);
	cr_assert_eq(rc, -EINVAL, "cxil_map() should fail invalid flags %d",
		     rc);
	free(mem.buffer);

	alloc_map_buf(len, &src_mem, flags | CXI_MAP_PIN);

	dst_mem.loc = on_host;
	dst_mem.length = len;
	dst_mem.buffer = HIGH_BPTR;

	/* Page request error will be returned when VA is not backed by VMA. */
	reg_win(&dst_mem, NULL, flags | *extra_flag);

	ptlte_setup(pid_idx, false, false);
	append_le_sync(rx_pte, &dst_mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);

	/* Put will cause a page request error */
	do_put(src_mem, dst_mem.length, 0, 0, pid_idx, true, 0, 0, 0, false);
	odp_check_event_rc(transmit_evtq, C_RC_PAGE_REQ_ERROR);

	/* consume the rest of the target events */
	while ((event = (union c_event *)cxi_eq_peek_event(target_evtq))) {
		odp_check_event_rc(target_evtq, C_RC_OK);
		sched_yield();
	}

	unlink_le(rx_pte, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID);
	odp_check_event_rc(target_evtq, C_RC_OK);
	ptlte_teardown();

	rc = cxil_unmap(dst_mem.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	free_iobuf(&src_mem);
}

/* ODP CXI_MAP_FAULT registration error if the VA range is not backed by VMA. */
Test(odp, odp_reg_fault_fail)
{
	int rc;
	struct mem_window m1;
	struct mem_window m2;
	size_t len = 16UL * 1024;
	uint32_t flags = CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_FAULT;
	int prot = PROT_WRITE | PROT_READ;

	m1.loc = on_host;
	m1.length = len;
	m1.buffer = HIGH_BPTR;

	/* No VMA */
	rc = cxil_map(lni, m1.buffer, m1.length, flags, NULL, &m1.md);
	cr_assert_eq(rc, -EINVAL, "cxil_map() failed %d", rc);

	/* VMA does not cover the complete range being mapped */
	mmap_hp(&m1, HIGH_BPTR, prot, len, 12, false);
	rc = cxil_map(lni, m1.buffer, m1.length * 2, flags, NULL, &m1.md);
	cr_assert_neq(rc, 0, "cxil_map() should fail %d", rc);

	/* Consecutive VMAs covering region should succeed */
	mmap_hp(&m2, (void *)(HIGH_BASE + len), prot, len, 12, false);
	rc = cxil_map(lni, m1.buffer, m1.length * 2, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(m1.buffer, m1.length);
	munmap(m2.buffer, m2.length);
}

/* ODP CXI_MAP_PREFETCH registration success scenarios */
Test(odp, odp_reg_prefetch)
{
	int rc;
	struct mem_window m1;
	struct mem_window m2;
	size_t len = 16UL * 1024;
	uint32_t flags = CXI_MAP_READ | CXI_MAP_WRITE | CXI_MAP_PREFETCH;
	int prot = PROT_WRITE | PROT_READ;

	m1.loc = on_host;
	m1.length = len;
	m1.buffer = HIGH_BPTR;

	/* No VMA */
	rc = cxil_map(lni, m1.buffer, m1.length, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	/* VMA does not cover the complete range being mapped */
	mmap_hp(&m1, HIGH_BPTR, prot, len, 12, false);
	rc = cxil_map(lni, m1.buffer, len * 2, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	/* Consecutive VMAs covering region should succeed */
	mmap_hp(&m2, (void *)(HIGH_BASE + len), prot, len, 12, false);
	rc = cxil_map(lni, m1.buffer, len * 2, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	/* Range ending with no VMA is ok */
	rc = cxil_map(lni, m1.buffer, len * 3, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(m1.buffer, m1.length);
	munmap(m2.buffer, m2.length);
}

/* Test cxil_update_md */
Test(odp, odp_update_md)
{
	int rc;
	struct mem_window m1;
	size_t len = 16UL * 1024;
	uint32_t flags = CXI_MAP_READ | CXI_MAP_WRITE;
	int prot = PROT_WRITE | PROT_READ;

	m1.loc = on_host;
	m1.length = len;
	m1.buffer = HIGH_BPTR;

	/* No VMA */
	rc = cxil_map(lni, m1.buffer, m1.length, flags, NULL, &m1.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	/* VA in range but no VMA */
	rc = cxil_update_md(m1.md, HIGH_BPTR, len, CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	mmap_hp(&m1, HIGH_BPTR, prot, len, 12, false);

	/* VA out of range */
	rc = cxil_update_md(m1.md, (void *)(HIGH_BASE + len), len,
			    CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	/* Update part of range */
	rc = cxil_update_md(m1.md, HIGH_BPTR, len / 2, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	/* Update full range */
	rc = cxil_update_md(m1.md, HIGH_BPTR, len / 2, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	rc = cxil_unmap(m1.md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(m1.buffer, m1.length);
}

/* mmap multiple ranges leaving holes between the VMAs */
Test(map_xfer, odp_fault)
{
	int rc;
	void *map_buf;
	struct cxi_md *md;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	struct mem_window tmp_mem;
	size_t map_len;
	size_t len = 1UL * 1024 * 1024;
	int hp_order;
	int prot = PROT_WRITE | PROT_READ;

	hp_order = 12;

	mmap_hp(&src_mem, NULL, prot, len, hp_order, true);
	cr_log_info("src buf:%p page size:%ld\n", src_mem.buffer,
		    cxil_page_size(src_mem.buffer));
	dump_vma_range((uintptr_t)src_mem.buffer, src_mem.length);

	mmap_hp(&tmp_mem, NULL, prot, len, hp_order, false);
	cr_log_info("src buf:%p page size:%ld\n", tmp_mem.buffer,
		    cxil_page_size(tmp_mem.buffer));
	dump_vma_range((uintptr_t)tmp_mem.buffer, tmp_mem.length);

	mmap_hp(&dst_mem, NULL, prot, len, hp_order, false);
	cr_log_info("dst buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));
	dump_vma_range((uintptr_t)dst_mem.buffer, dst_mem.length);

	map_buf = dst_mem.buffer;
	map_len = src_mem.buffer - dst_mem.buffer + len;
	cr_log_info("map:%p len:%lx, %lx\n", map_buf, map_len,
		    src_mem.buffer - dst_mem.buffer);
	dump_vma_range((uintptr_t)map_buf, map_len);

	rc = cxil_map(lni, map_buf, map_len,
		      CXI_MAP_PREFETCH | CXI_MAP_READ | CXI_MAP_WRITE, NULL,
		      &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	src_mem.md = md;
	dst_mem.md = md;

	dst_mem.length = 64 * 1024;
	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(dst_mem.buffer, dst_mem.length);
	munmap(src_mem.buffer, src_mem.length);
}

ParameterizedTestParameters(map_xfer, odp_fault2)
{
	size_t param_sz;
	static int order_values[] = {12, 21, 23, 25};

	param_sz = ARRAY_SIZE(order_values);
	return cr_make_param_array(int, order_values, param_sz);
}

/* With one mmap'd range, fault in separate parts for source and dest
 * with various page sizes.
 */
ParameterizedTest(int *order, map_xfer, odp_fault2)
{
	int rc;
	struct cxi_md *md;
	struct mem_window mem;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	int hp_order = *order;
	size_t mmap_len = 4UL << hp_order;
	size_t xfer_len = mmap_len / 2;
	int prot = PROT_WRITE | PROT_READ;

	mmap_hp(&mem, NULL, prot, mmap_len, hp_order, false);
	cr_log_info("buf:%p page size:%ld\n", mem.buffer,
		    cxil_page_size(mem.buffer));
	dump_vma_range((uintptr_t)mem.buffer, mem.length);

	rc = cxil_map(lni, mem.buffer, mmap_len,
		      CXI_MAP_READ | CXI_MAP_WRITE, NULL, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	src_mem.buffer = mem.buffer;
	src_mem.length = xfer_len;
	src_mem.loc = mem.loc;
	src_mem.md = md;

	dst_mem.buffer = mem.buffer + mmap_len - xfer_len;
	dst_mem.length = xfer_len;
	dst_mem.loc = mem.loc;
	dst_mem.md = md;

	/* Check invalid values */
	/* No MD */
	rc = cxil_update_md(NULL, dst_mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	/* NULL va */
	rc = cxil_update_md(md, NULL, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	/* Length outside of MD */
	rc = cxil_update_md(md, dst_mem.buffer, xfer_len + 1, CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	/* No CXI_MAP_FAULT */
	rc = cxil_update_md(md, dst_mem.buffer, xfer_len, 0);
	cr_assert_eq(rc, -EINVAL, "map_buf() success %d", rc);

	/* Now fault in pages */
	rc = cxil_update_md(md, src_mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	rc = cxil_update_md(md, dst_mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(mem.buffer, mem.length);
}

/* Register range with hint and fault separate ranges */
Test(map_xfer, odp_fault3)
{
	int rc;
	int hp_order = 25;
	struct cxi_md *md;
	struct mem_window mem;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	struct cxi_md_hints hints = {};
	size_t mmap_len = 4UL << hp_order;
	size_t xfer_len = mmap_len / 2;
	int prot = PROT_WRITE | PROT_READ;
	int flags = CXI_MAP_READ | CXI_MAP_WRITE;

	/* Register with a hugepage hint before mmaping */
	hints.huge_shift = hp_order;
	rc = cxil_map(lni, HIGH_BPTR, mmap_len, flags, &hints, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_assert_eq(hp_order, md->huge_shift, "huge_shift incorrect %d",
		     md->huge_shift);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	/* Check for registration with hugepage missmatch */
	mmap_hp(&mem, HIGH_BPTR, prot, xfer_len, hp_order - 2, false);
	rc = cxil_update_md(md, mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, -EINVAL, "map_buf() should get -EINVAL got:%d", rc);
	munmap(mem.buffer, mem.length);

	mmap_hp(&src_mem, HIGH_BPTR, prot, xfer_len, hp_order, false);
	cr_log_info("buf:%p page size:%ld\n", src_mem.buffer,
		    cxil_page_size(src_mem.buffer));
	dump_vma_range((uintptr_t)src_mem.buffer, src_mem.length);

	mmap_hp(&dst_mem, (void *)(HIGH_BASE + xfer_len), prot, xfer_len,
		hp_order, false);
	cr_log_info("buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));
	dump_vma_range((uintptr_t)dst_mem.buffer, dst_mem.length);

	rc = cxil_update_md(md, src_mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	rc = cxil_update_md(md, dst_mem.buffer, xfer_len, CXI_MAP_FAULT);
	cr_assert_eq(rc, 0, "map_buf() failed %d", rc);

	src_mem.md = md;
	dst_mem.md = md;

	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(src_mem.buffer, src_mem.length);
	munmap(dst_mem.buffer, dst_mem.length);
}

/* Test small (less than default PRI alignment) buffers allocated within a
 * sparse region
 */
Test(map_xfer, odp_fault4)
{
	int rc;
	int hp_order = 21;
	struct cxi_md *md;
	struct mem_window src_mem;
	struct mem_window dst_mem;
	struct cxi_md_hints hints = {};
	size_t mmap_len = 4UL << hp_order;
	size_t xfer_len = 8 * 1024;
	int prot = PROT_WRITE | PROT_READ;
	int flags = CXI_MAP_READ | CXI_MAP_WRITE;

	/* Register a large addr range before mmaping */
	rc = cxil_map(lni, HIGH_BPTR, mmap_len, flags, &hints, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);
	cr_log_info("ps:%d hs:%d\n", md->page_shift, md->huge_shift);

	/* mmap some small unaligned on 64k buffers in the addr range */
	mmap_hp(&dst_mem, (void *)(HIGH_BASE + xfer_len * 5), prot, xfer_len,
		12, false);
	cr_log_info("buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));
	dump_vma_range((uintptr_t)dst_mem.buffer, dst_mem.length);

	mmap_hp(&src_mem, (void *)(HIGH_BASE + xfer_len * 10), prot, xfer_len,
		12, false);
	cr_log_info("buf:%p page size:%ld\n", src_mem.buffer,
		    cxil_page_size(src_mem.buffer));
	dump_vma_range((uintptr_t)src_mem.buffer, src_mem.length);

	src_mem.md = md;
	dst_mem.md = md;

	do_transfer(&src_mem, &dst_mem);

	munmap(dst_mem.buffer, dst_mem.length);

	/* test aligned buffer with less than 64k length */
	mmap_hp(&dst_mem, HIGH_BPTR, prot, xfer_len, 12, false);
	cr_log_info("buf:%p page size:%ld\n", dst_mem.buffer,
		    cxil_page_size(dst_mem.buffer));
	dump_vma_range((uintptr_t)dst_mem.buffer, dst_mem.length);

	do_transfer(&src_mem, &dst_mem);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(src_mem.buffer, src_mem.length);
	munmap(dst_mem.buffer, dst_mem.length);
}

size_t get_ac_size(void)
{
	int fd;
	int ret;
	uint32_t value;
	char buf[128];
	char *path = "/sys/module/cxi_ss1/parameters/ac_size_dbl";

	fd = open(path, O_RDONLY);
	if (fd < 0)
		return 0;

	ret = read(fd, buf, 8);
	cr_assert_geq(ret, 0, "read failed %d", ret);

	ret = sscanf(buf, "%u", &value);
	cr_assert_eq(ret, 1, "failed to get value. ret %d", ret);

	close(fd);

	return (1UL * 1024 * 1024 * 1024) << value;
}

Test(map, iova_init)
{
	int rc;
	uint8_t *mmap_buf;
	struct cxi_md *md;
	struct mem_window mem2 = {};
	size_t ac_size = get_ac_size();
	size_t mmap_len = 2UL * 1024 * 1024;
	int flags = CXI_MAP_WRITE | CXI_MAP_READ;
	int prot = PROT_WRITE | PROT_READ;
	int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;

	mmap_buf = mmap(NULL, mmap_len, prot, mmap_flags, 0, 0);
	if (!mmap_buf || (uintptr_t)mmap_buf == 0xffffffffffffffff)
		cr_log_info("Could not mmap memory 0x%lx bytes", mmap_len);

	/* Fill up an AC. Registration will succeed since we are using
	 * ODP but not touching the memory. We just need an iova
	 * that fills the AC.
	 */
	rc = cxil_map(lni, mmap_buf, ac_size, flags, NULL, &md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	/* This registration will fail to get an allocation from the first
	 * AC. (This will cause a kernel oops if the IOVA is not initialized
	 * properly.) The driver will then allocate from a new AC.
	 */
	alloc_map_buf(4096, &mem2, flags);

	free_iobuf(&mem2);

	rc = cxil_unmap(md);
	cr_expect_eq(rc, 0, "cxil_unmap() failed %d", rc);

	munmap(mmap_buf, mmap_len);
}

#if defined(HAVE_HIP_SUPPORT) || defined(HAVE_CUDA_SUPPORT) || defined(HAVE_ZE_SUPPORT)
Test(map_xfer, device1)
{
	int rc;
	size_t len = 64UL * 1024;
	struct mem_window buf;
	uint32_t flags = CXI_MAP_PIN | CXI_MAP_WRITE | CXI_MAP_READ;

	buf.length = len;
	buf.buffer = calloc(1, len);
	cr_assert_not_null(buf.buffer, "Failed to allocate iobuf");

	/* try mapping a normal buffer with the DEVICE flag */
	rc = cxil_map(lni, buf.buffer, buf.length,
		      CXI_MAP_DEVICE | flags, NULL, &buf.md);
	cr_assert_neq(rc, 0, "cxil_map() should have failed %d", rc);
	free(buf.buffer);

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	rc = gpu_malloc(&buf);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	/* try mapping a device buffer without the DEVICE flag */
	rc = cxil_map(lni, buf.buffer, buf.length,
		      CXI_MAP_PIN | CXI_MAP_READ, NULL, &buf.md);
	cr_assert_neq(rc, 0, "cxil_map() should have failed %d", rc);

	gpu_free(buf.buffer);

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device_map_multiple)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

/* Map the exact same buffer multiple times. */
ParameterizedTest(bool *use_dmabuf, map_xfer, device_map_multiple)
{
	int rc;
	struct mem_window buf = {.use_dmabuf = *use_dmabuf};
	struct cxi_md *mds[10];
	int i;
	void *base_addr;
	size_t size;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	buf.length = 64UL * 1024;
	rc = gpu_malloc(&buf);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	rc = gpu_props(&buf, &base_addr, &size);
	cr_assert(rc == 0);

	/* Map the buffer multiple times to ensure the stack can handle this. */
	for (i = 0; i < 10; i++) {
		rc = cxil_map(lni, buf.buffer, buf.length,
			      CXI_MAP_PIN | CXI_MAP_READ | CXI_MAP_DEVICE,
			      &buf.hints, &mds[i]);
		cr_assert_eq(rc, 0, "cxil_map() should have failed %d", rc);
	}

	for (i = 0; i < 10; i++) {
		rc = cxil_unmap(mds[i]);
		cr_assert_eq(rc, 0, "cxil_unmap() should have failed %d", rc);
	}

	if (buf.hints.dmabuf_valid)
		gpu_close_fd(buf.hints.dmabuf_fd);

	gpu_free(buf.buffer);

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device2)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

ParameterizedTest(bool *use_dmabuf, map_xfer, device2)
{
	int rc;
	struct mem_window src_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window dst_mem;
	size_t len = 64UL * 1024;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	/* Allocate buffers */
	alloc_iobuf(len, &dst_mem, CXI_MAP_WRITE);
	alloc_map_devicebuf(len, &src_mem, CXI_MAP_READ);

	do_transfer(&src_mem, &dst_mem);

	free_unmap_devicebuf(&src_mem);
	free_iobuf(&dst_mem);

	gpu_lib_fini();
}

Test(map_xfer, device3)
{
	int rc;
	size_t len;
	struct mem_window src_mem;
	struct mem_window dst_mem;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("host to gpu\n");

	for (len = 0x8000; len < 0x100000000; len <<= 1) {
		printf("len:%lx\n", len);

		alloc_map_devicebuf(len, &dst_mem, CXI_MAP_WRITE);
		alloc_iobuf(len, &src_mem, CXI_MAP_READ);

		do_transfer(&src_mem, &dst_mem);

		free_iobuf(&src_mem);
		free_unmap_devicebuf(&dst_mem);
	}

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device4)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

ParameterizedTest(bool *use_dmabuf, map_xfer, device4)
{
	int rc;
	size_t len;
	struct mem_window src_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window dst_mem = {};

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("gpu to host\n");

	for (len = 0x8000; len < 0x100000000; len <<= 1) {
		printf("len:%lx\n", len);

		alloc_map_devicebuf(len, &src_mem, CXI_MAP_READ);
		alloc_iobuf(len, &dst_mem, CXI_MAP_WRITE);

		do_transfer(&src_mem, &dst_mem);

		free_unmap_devicebuf(&src_mem);
		free_iobuf(&dst_mem);
	}

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device5)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

ParameterizedTest(bool *use_dmabuf, map_xfer, device5)
{
	int rc;
	size_t len;
	struct mem_window src_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window dst_mem = {.use_dmabuf = *use_dmabuf};

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("gpu to gpu\n");

	for (len = 0x8000; len < 0x10000000; len <<= 1) {
		printf("len:%lx\n", len);

		alloc_map_devicebuf(len, &dst_mem, CXI_MAP_WRITE);
		alloc_map_devicebuf(len, &src_mem, CXI_MAP_READ);

		do_transfer(&src_mem, &dst_mem);

		free_unmap_devicebuf(&src_mem);
		free_unmap_devicebuf(&dst_mem);
	}

	gpu_lib_fini();
}

Test(map_xfer, device6)
{
	struct mem_window src_mem = {.use_dmabuf = true};
	struct mem_window dst_mem = {};
	size_t len = 64UL * 1024;
	int rc;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("host visible gpu to host\n");

	alloc_iobuf(len, &dst_mem, CXI_MAP_WRITE);
	alloc_map_hostbuf(len, &src_mem, CXI_MAP_READ);

	do_transfer(&src_mem, &dst_mem);

	free_unmap_devicebuf(&src_mem);
	free_iobuf(&dst_mem);

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device7)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

ParameterizedTest(bool *use_dmabuf, map_xfer, device7)
{
	struct mem_window src_mem = {.use_dmabuf = false};
	struct mem_window dst_mem = {.use_dmabuf = *use_dmabuf};
	size_t len = 64UL * 1024;
	int rc;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("gpu to host visible gpu\n");

	alloc_map_devicebuf(len, &dst_mem, CXI_MAP_WRITE);
	alloc_map_hostbuf(len, &src_mem, CXI_MAP_READ);

	do_transfer(&src_mem, &dst_mem);

	free_unmap_devicebuf(&src_mem);
	free_unmap_devicebuf(&dst_mem);

	gpu_lib_fini();
}

ParameterizedTestParameters(map_xfer, device8)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

ParameterizedTest(bool *use_dmabuf, map_xfer, device8)
{
	int rc;
	size_t len;
	struct mem_window src_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window dst_mem = {.use_dmabuf = *use_dmabuf};

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	printf("hostbuf to hostbuf\n");

	for (len = 0x8000; len < 0x10000000; len <<= 1) {
		printf("len:%lx\n", len);

		alloc_map_hostbuf(len, &dst_mem, CXI_MAP_WRITE);
		alloc_map_hostbuf(len, &src_mem, CXI_MAP_READ);

		do_transfer(&src_mem, &dst_mem);

		free_unmap_devicebuf(&src_mem);
		free_unmap_devicebuf(&dst_mem);
	}
	gpu_lib_fini();
}

#define MAP_OFFSET_LARGE_BUF_SIZE 1048576U
#define MAP_OFFSET_SMALL_BUF_SIZE 1234U
#define MAP_OFFSET_SRC_OFFSET 9873U
#define MAP_OFFSET_DST_OFFSET 32175U

ParameterizedTestParameters(map_xfer, device_map_offset)
{
	size_t param_sz;
	static bool param[] = {false, true};

	param_sz = ARRAY_SIZE(param);
	return cr_make_param_array(bool, param, param_sz);
}

/* Map only a small offset of a buffer and use if for transfers. */
ParameterizedTest(bool *use_dmabuf, map_xfer, device_map_offset)
{
	struct mem_window src_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window dst_mem = {.use_dmabuf = *use_dmabuf};
	struct mem_window src_md_mem = {};
	struct mem_window dst_md_mem = {};
	void *base_addr;
	size_t size;
	int rc;

	rc = gpu_lib_init();
	if (rc)
		cr_skip_test("No GPU detected\n");

	/* Over allocate the source and destination buffer. */
	src_mem.length = MAP_OFFSET_LARGE_BUF_SIZE;
	rc = gpu_malloc(&src_mem);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	dst_mem.length = MAP_OFFSET_LARGE_BUF_SIZE;
	rc = gpu_malloc(&dst_mem);
	cr_assert_eq(rc, 0, "gpu_malloc() failed %d", rc);

	/* Map the source buffer at an offset. */
	rc = gpu_props(&src_mem, &base_addr, &size);
	cr_assert(rc == 0);

	src_mem.hints.dmabuf_offset += MAP_OFFSET_SRC_OFFSET;

	src_md_mem.buffer =
		(void *)((uintptr_t)src_mem.buffer + MAP_OFFSET_SRC_OFFSET);
	src_md_mem.length = MAP_OFFSET_SMALL_BUF_SIZE;
	src_md_mem.loc = on_device;

	rc = cxil_map(lni, src_md_mem.buffer, src_md_mem.length,
		      CXI_MAP_WRITE | CXI_MAP_READ | CXI_MAP_PIN |
		      CXI_MAP_DEVICE, &src_mem.hints, &src_md_mem.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	/* Map the destination buffer at an offset. */
	rc = gpu_props(&dst_mem, &base_addr, &size);
	cr_assert(rc == 0 || rc == -ENOSYS);

	dst_mem.hints.dmabuf_offset += MAP_OFFSET_DST_OFFSET;

	dst_md_mem.buffer =
		(void *)((uintptr_t)dst_mem.buffer + MAP_OFFSET_DST_OFFSET);
	dst_md_mem.length = MAP_OFFSET_SMALL_BUF_SIZE;
	dst_md_mem.loc = on_device;

	rc = cxil_map(lni, dst_md_mem.buffer, dst_md_mem.length,
		      CXI_MAP_WRITE | CXI_MAP_READ | CXI_MAP_PIN |
		      CXI_MAP_DEVICE, &dst_mem.hints, &dst_md_mem.md);
	cr_assert_eq(rc, 0, "cxil_map() failed %d", rc);

	do_transfer(&src_md_mem, &dst_md_mem);

	rc = cxil_unmap(dst_md_mem.md);
	cr_assert_eq(rc, 0, "cxil_unmap() failed %d", rc);

	rc = cxil_unmap(src_md_mem.md);
	cr_assert_eq(rc, 0, "cxil_unmap() failed %d", rc);

	if (dst_mem.hints.dmabuf_valid)
		gpu_close_fd(dst_mem.hints.dmabuf_fd);
	if (src_mem.hints.dmabuf_valid)
		gpu_close_fd(src_mem.hints.dmabuf_fd);

	gpu_free(dst_mem.buffer);
	gpu_free(src_mem.buffer);

	gpu_lib_fini();
}

#endif /* HAVE_HIP_SUPPORT || HAVE_CUDA_SUPPORT || HAVE_ZE_SUPPORT */
