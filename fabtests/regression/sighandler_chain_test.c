/*
 * Copyright (c) 2026 Perplexity AI.  All rights reserved.
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

/*
 * Regression test for
 * "prov/shm: Properly chain the original signal handlers"
 * https://github.com/ofiwg/libfabric/pull/11915
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <rdma/fabric.h>
#include <signal.h>
#include "shared.h"

static volatile sig_atomic_t got_signal = 0;
static struct sigaction prev_action;

/*
 * Mimics signal-hook-registry's handler: chains to previous handler first,
 * then runs registered actions.
 * https://github.com/vorner/signal-hook/blob/v0.4.3/signal-hook-registry/src/lib.rs#L390-L428
 */
static void signal_hook_handler(int signum, siginfo_t *info, void *ucontext)
{
	uintptr_t fptr = (uintptr_t)prev_action.sa_sigaction;

	if (fptr != 0 && fptr != (uintptr_t)SIG_DFL &&
	    fptr != (uintptr_t)SIG_IGN) {
		if (prev_action.sa_flags & SA_SIGINFO)
			prev_action.sa_sigaction(signum, info, ucontext);
		else
			((void (*)(int))fptr)(signum);
	}

	got_signal = 1;
}

static void install_signal_hook_handler(void)
{
	struct sigaction act;

	memset(&act, 0, sizeof(act));
	act.sa_sigaction = signal_hook_handler;
	act.sa_flags = SA_SIGINFO;

	sigaction(SIGTERM, &act, &prev_action);
}

int main(int argc, char **argv)
{
	int child;
	int status;
	int op;

	opts = INIT_OPTS;
	opts.options |= FT_OPT_SKIP_ADDR_EXCH;

	if ((child = fork())) {
		usleep(5000000);
		kill(child, SIGKILL);

		waitpid(child, &status, 0);
		if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
			printf("Pass: outer signal handler survived\n");
			exit(0);
		}
		if (WIFSIGNALED(status))
			printf("Fail: prov destroyed the outer signal "
			       "handler!\n");
		printf("Fail: child killed by signal %d (%s) "
		       "and exited with status %d\n",
		       WIFSIGNALED(status) ? WTERMSIG(status) : 0,
		       WIFSIGNALED(status) ? strsignal(WTERMSIG(status))
					   : "n/a",
		       WIFEXITED(status) ? WEXITSTATUS(status) : -1);
		exit(EXIT_FAILURE);
	} else {
		hints = fi_allocinfo();
		if (!hints)
			exit(EXIT_FAILURE);

		while ((op = getopt(argc, argv, "p:h")) != -1) {
			switch (op) {
			case 'p':
				hints->fabric_attr->prov_name = strdup(optarg);
				break;
			case '?':
			case 'h':
				FT_PRINT_OPTS_USAGE("-p <provider>",
					"specific provider name eg shm, efa");
				return EXIT_FAILURE;
			}
		}
		hints->caps = FI_MSG;
		hints->mode = FI_CONTEXT | FI_CONTEXT2;
		if (ft_init_fabric()) {
			ft_freehints(hints);
			exit(EXIT_FAILURE);
		}

		install_signal_hook_handler();
		raise(SIGTERM);

		if (got_signal)
			_exit(0);

		_exit(EXIT_FAILURE);
	}
}
