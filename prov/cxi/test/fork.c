/*
 * (C) Copyright 2023 Hewlett Packard Enterprise Development LP
 *
 * This software is available to you under a choice of one of two
 * licenses.  You may choose to be licensed under the terms of the GNU
 * General Public License (GPL) Version 2, available from the file
 * COPYING in the main directory of this source tree, or the
 * BSD license below:
 *
 *     Redistribution and use in source and binary forms, with or
 *     without modification, are permitted provided that the following
 *     conditions are met:
 *
 *      - Redistributions of source code must retain the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer.
 *
 *      - Redistributions in binary form must reproduce the above
 *        copyright notice, this list of conditions and the following
 *        disclaimer in the documentation and/or other materials
 *        provided with the distribution.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
 * BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
 * ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <ctype.h>

#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <pthread.h>

#include "cxip.h"
#include "cxip_test_common.h"

#define SECRET 0xFFU
#define XFER_SIZE 257U
#define INIT_BUF_VALUE 0xAAU
#define INIT_BUF_OFFSET 127U
#define TGT_BUF_VALUE 0xFFU
#define TGT_BUF_OFFSET 3215U
#define RKEY 0x1U
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)

struct linux_version {
	union {
		struct {
			unsigned int kernel;
			unsigned int major;
			unsigned int minor;
		};
		unsigned int ver[3];
	};
};

/* Needs to be marked volatile to prevent hangs due to compiler optimization. */
static volatile bool child_process_block = true;

static void signal_handler(int sig)
{
	child_process_block = false;
}

static int get_kernel_version(struct linux_version *version)
{
	FILE *fd;
	char buf[256] = {};
	size_t ret;
	char *p = buf;
	int i = 0;

	/* uname() appears to not work. Extract information from version file.
	 */
	fd = fopen("/proc/version", "r");
	if (fd == NULL)
		return -errno;

	ret = fread(buf, 1, 256, fd);

	fclose(fd);

	if (ret) {
		while (*p) {
			if (isdigit(*p)) {
				version->ver[i] = strtol(p, &p, 10);
				i++;
			} else {
				p++;
			}

			if (i == 3)
				return 0;
		}
	}

	return -EIO;
}

static void fork_test_runner(bool odp, bool huge_page, bool fork_safe)
{
	long page_size;
	uint8_t *buf;
	uint8_t *init_buf;
	uint8_t *tgt_buf;
	int ret;
	struct fid_mr *mr;
	int status;
	struct fi_cq_tagged_entry cqe;
	pid_t pid;
	struct linux_version ver;
	int i = 0;
	int flags = MAP_PRIVATE | MAP_ANONYMOUS;

	if (odp) {
		ret = setenv("FI_CXI_FORCE_ODP", "1", 1);
		cr_assert_eq(ret, 0, "Failed to set FI_CXI_FORCE_ODP %d", -errno);
	}

	if (fork_safe) {
		ret = setenv("CXI_FORK_SAFE", "1", 1);
		cr_assert_eq(ret, 0, "Failed to set CXI_FORK_SAFE %d", -errno);

		if (huge_page) {
			ret = setenv("CXI_FORK_SAFE_HP", "1", 1);
			cr_assert_eq(ret, 0, "Failed to set CXI_FORK_SAFE %d", -errno);
		}
	}

	cxit_setup_msg();

	ret = get_kernel_version(&ver);
	cr_assert_eq(ret, 0, "Failed to get kernel version %d", ret);

	signal(SIGUSR1, signal_handler);

	/* Single map is used for page aliasing with child process and RDMA. */
	if (huge_page) {
		page_size = 2 * 1024 * 1024;
		flags |= MAP_HUGETLB | MAP_HUGE_2MB;
	} else {
		page_size = sysconf(_SC_PAGESIZE);
	}

	buf = mmap(NULL, page_size, PROT_READ | PROT_WRITE, flags, -1, 0);
	cr_assert(buf != MAP_FAILED, "mmap failed");

	memset(buf, 0, page_size);

	/* This secret is passed to the child process. Child process will verify
	 * it receives this secret.
	 */
	buf[0] = SECRET;
	init_buf = buf + INIT_BUF_OFFSET;
	tgt_buf = buf + TGT_BUF_OFFSET;

	/* Register the buffer. The behavior of the child buffer depends upon
	 * the following
	 * - If CXI_FORK_SAFE is set, madvise(MADV_DONTFORK) will be issued
	 * against the page. This will cause the child to segfault.
	 * - If ODP is not used and the kernel < 5.12, the child process will
	 * get its data, and the parent process will have data corruption.
	 * - If ODP is not used and the kernel >= 5.12, the child process will
	 * get its data, and the parent process will not have data corruption.
	 * - If ODP is used, the child process will get its data, and the parent
	 * process will not have data corruption.
	 */
	ret = fi_mr_reg(cxit_domain, tgt_buf, XFER_SIZE, FI_REMOTE_WRITE, 0,
			RKEY, 0, &mr, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_reg failed %d", ret);

	ret = fi_mr_bind(mr, &cxit_ep->fid, 0);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_bind failed %d", ret);

	ret = fi_mr_enable(mr);
	cr_assert_eq(ret, FI_SUCCESS, "fi_mr_enable failed %d", ret);

	pid = fork();
	cr_assert(pid != -1, "fork() failed");

	if (pid == 0) {
		while (child_process_block)
			sched_yield();

		/* If CXI_FORK_SAFE is set (i.e. fork_safe is true), this will
		 * segfault.
		 */
		if (buf[0] == SECRET)
			_exit(EXIT_SUCCESS);

		/* This should never happen. */
		_exit(EXIT_FAILURE);
	}

	/* Writing this these buffers will trigger COW.  Unless
	* madvise(MADV_DONTFORK) was called, parent process will get new page.
	*/
	memset(init_buf, INIT_BUF_VALUE, XFER_SIZE);
	memset(tgt_buf, TGT_BUF_VALUE, XFER_SIZE);

	ofi_sfence();

	/* Unblock the child process. */
	kill(pid, SIGUSR1);

	ret = fi_write(cxit_ep, init_buf, XFER_SIZE, NULL, cxit_ep_fi_addr, 0,
		       RKEY, NULL);
	cr_assert_eq(ret, FI_SUCCESS, "fi_write failed %d", ret);

	ret = cxit_await_completion(cxit_tx_cq, &cqe);
	cr_assert_eq(ret, 1, "fi_cq_read failed %d", ret);

	validate_tx_event(&cqe, FI_RMA | FI_WRITE, NULL);

	if ((ver.kernel >= 5 && ver.major >= 12) || odp || fork_safe) {
		for (i = 0; i < XFER_SIZE; i++)
			cr_assert_eq(init_buf[i], tgt_buf[i], "data corruption with fork");
	} else {
		for (i = 0; i < XFER_SIZE; i++)
			cr_assert_neq(init_buf[i], tgt_buf[i], "Missing data corruption with fork");
	}

	waitpid(pid, &status, 0);

	if (fork_safe) {
		cr_assert_eq(WIFSIGNALED(status), true, "Child was not terminated by signal: is_exit=%d exit=%d is_sig=%d sig=%d",
			     WIFEXITED(status), WEXITSTATUS(status),
			     WIFSIGNALED(status), WTERMSIG(status));
		cr_assert_eq(WTERMSIG(status), SIGSEGV, "Child signal was not SIGSEGV");
	} else {
		cr_assert_eq(WIFEXITED(status), true, "Child was not terminated by exit: is_exit=%d exit=%d is_sig=%d sig=%d",
			     WIFEXITED(status), WEXITSTATUS(status),
			     WIFSIGNALED(status), WTERMSIG(status));
		cr_assert_eq(WEXITSTATUS(status), EXIT_SUCCESS, "Child process had data corruption");
	}

	fi_close(&mr->fid);
	munmap(buf, page_size);

	cxit_teardown_msg();
}

TestSuite(fork, .timeout = CXIT_DEFAULT_TIMEOUT);

/* No ODP, no fork safe variables, and system page size. On kernels before 5.12,
 * parent process should have data corruption. Child process should not have
 * data corruption and should not segfault.
 */
Test(fork, page_aliasing_no_odp_no_fork_safe_system_page_size)
{
	fork_test_runner(false, false, false);
}

/* ODP, no fork safe variables, and system page size. Parent process should not
 * have data corruption regardless of kernel version. Child process should not
 * have data corruption and should not segfault.
 */
Test(fork, page_aliasing_odp_no_fork_safe_system_page_size)
{
	fork_test_runner(true, false, false);
}

/* No ODP, no fork safe variables, and system page size. Parent process should
 * not have data corruption regardless of kernel version. Child process should
 * segfault since parent called MADV_DONTFORK on virtual address range.
 */
Test(fork, page_aliasing_no_odp_fork_safe_system_page_size)
{
	fork_test_runner(false, false, true);
}

/* No ODP, no fork safe variables, and 2MiB page size. On kernels before 5.12,
 * parent process should have data corruption. Child process should not have
 * data corruption and should not segfault.
 */
Test(fork, page_aliasing_no_odp_no_fork_safe_huge_page)
{
	fork_test_runner(false, true, false);
}

/* ODP, no fork safe variables, and 2MiB page size. Parent process should not
 * have data corruption regardless of kernel version. Child process should not
 * have data corruption and should not segfault.
 */
Test(fork, page_aliasing_odp_no_fork_safe_huge_page)
{
	fork_test_runner(true, true, false);
}

/* No ODP, no fork safe variables, and 2MiB page size. Parent process should
 * not have data corruption regardless of kernel version. Child process should
 * segfault since parent called MADV_DONTFORK on virtual address range.
 */
Test(fork, page_aliasing_no_odp_fork_safe_huge_page)
{
	fork_test_runner(false, true, true);
}
