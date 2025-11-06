/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
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
#include <unistd.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

#include "libcxi_test_common.h"
#include "uapi/misc/cxi.h"

#include <pthread.h>

#define SECRET 0xFFU
#define XFER_SIZE 257U
#define INIT_BUF_VALUE 0xAAU
#define INIT_BUF_OFFSET 127U
#define TGT_BUF_VALUE 0xFFU
#define TGT_BUF_OFFSET 3215U
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#define PUT_BUFFER_ID 0xb0f

TestSuite(fork);

/* Needs to be marked volatile to prevent hangs due to compiler optimization. */
static volatile bool child_process_block = true;

static void signal_handler(int sig)
{
	child_process_block = false;
}

static void fork_test_runner(bool fork_safe)
{
	long page_size;
	uint8_t *buf;
	uint8_t *init_buf;
	uint8_t *tgt_buf;
	int ret;
	struct mem_window mem;
	int pid_idx = 0;
	int status;
	pid_t pid;
	int i = 0;

	if (fork_safe) {
		ret = setenv("CXI_FORK_SAFE", "1", 1);
		cr_assert_eq(ret, 0, "Failed to set CXI_FORK_SAFE %d", -errno);
	}

	data_xfer_setup();

	signal(SIGUSR1, signal_handler);

	/* Single page is used and is allocated page aligned and cleared */
	page_size = sysconf(_SC_PAGESIZE);
	alloc_iobuf(page_size, &mem, CXI_MAP_READ | CXI_MAP_WRITE);
	buf = mem.buffer;

	/* This secret is passed to the child process. Child process will verify
	 * it receives this secret.
	 */
	buf[0] = SECRET;
	init_buf = buf + INIT_BUF_OFFSET;
	tgt_buf = buf + TGT_BUF_OFFSET;

	ptlte_setup(pid_idx, false, false);
	append_le_sync(rx_pte, &mem, C_PTL_LIST_PRIORITY, PUT_BUFFER_ID,
		       0, 0, CXI_MATCH_ID_ANY, 0, true, false, false, false,
		       false, true, false, NULL);
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

	sfence();

	/* Unblock the child process. */
	kill(pid, SIGUSR1);

	do_put_sync(mem, XFER_SIZE, TGT_BUF_OFFSET, INIT_BUF_OFFSET,
		    pid_idx, true, 0, 0, 0, false);
	ptlte_teardown();

	if (cxil_is_copy_on_fork() || fork_safe) {
		for (i = 0; i < XFER_SIZE; i++)
			cr_assert_eq(init_buf[i], tgt_buf[i],
				     "data corruption with fork");
	} else {
		for (i = 0; i < XFER_SIZE; i++)
			cr_assert_neq(init_buf[i], tgt_buf[i],
				      "Missing data corruption with fork");
	}

	waitpid(pid, &status, 0);

	if (!cxil_is_copy_on_fork() && fork_safe) {
		cr_assert_eq(WIFSIGNALED(status), true,
			     "Child was not terminated by signal: is_exit=%d exit=%d is_sig=%d sig=%d",
			     WIFEXITED(status), WEXITSTATUS(status),
			     WIFSIGNALED(status), WTERMSIG(status));
		cr_assert_eq(WTERMSIG(status), SIGSEGV,
			     "Child signal was not SIGSEGV");
	} else {
		cr_assert_eq(WIFEXITED(status), true,
			     "Child was not terminated by exit: is_exit=%d exit=%d is_sig=%d sig=%d",
			     WIFEXITED(status), WEXITSTATUS(status),
			     WIFSIGNALED(status), WTERMSIG(status));
		cr_assert_eq(WEXITSTATUS(status), EXIT_SUCCESS,
			     "Child process had data corruption");
	}

	free_iobuf(&mem);
	data_xfer_teardown();
}

/* Verify default copy-on-fork kernel operation without fork environment
 * variables.
 * COF Supported: Parent will not detect corruption. Child exits.
 * COF Unsupported: Parent will detect corruption. Child exits.
 */
Test(fork, page_aliasing_no_fork_safe)
{
	fork_test_runner(false);
}

/* Verify that if kernel copy-on-fork is supported that fork
 * environment variables are overridden and madvise is not called.
 * COF Supported: Parent will not detect corruption. Child exits.
 * COF Unsupported: Parent will not detect corruption. Child segfaults.
 */
Test(fork, page_aliasing_fork_safe)
{
	fork_test_runner(true);
}
