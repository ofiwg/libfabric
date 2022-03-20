/*
 * Copyright (C) 2021 Cornelis Networks.
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
#ifndef _FI_PROV_OPX_DEBUG_COUNTERS_H_
#define _FI_PROV_OPX_DEBUG_COUNTERS_H_

#include <stdio.h>
#include <stdint.h>
#include <string.h>

struct fi_opx_debug_counters {
	uint64_t	mp_eager_send_first_packets;
	uint64_t	mp_eager_send_nth_packets;
	uint64_t	mp_eager_send_first_force_cr;
	uint64_t	mp_eager_send_nth_force_cr;
	uint64_t	mp_eager_send_fall_back_to_rzv;
	uint64_t	mp_eager_send_full_replay_buffer_rx_poll;

	uint64_t	mp_eager_recv_max_ue_queue_length;
	uint64_t	mp_eager_recv_max_mq_queue_length;
	uint64_t	mp_eager_recv_first_packets;
	uint64_t	mp_eager_recv_nth_packets;
	uint64_t	mp_eager_recv_completed_process_context;
	uint64_t	mp_eager_recv_completed_eager_first;
	uint64_t	mp_eager_recv_completed_eager_nth;
	uint64_t	mp_eager_recv_truncation;
	uint64_t	mp_eager_recv_nth_no_match;
	uint64_t	mp_eager_recv_nth_match;
};

static inline
void dump_mem(void *address, uint64_t lenth) {
	fprintf(stderr, "### Dumping %lu bytes of memory at addr %p...\n", lenth, address);

	fprintf(stderr, "Address                   QW0              QW1              QW2            QW3\n");
	uint64_t *mem = (uint64_t *) address;
	for (uint64_t i = 0; i < lenth; i += 32) {
		fprintf(stderr, "%016lX    %016lX  %016lX  %016lX  %016lX\n", *mem, mem[0], mem[1], mem[2], mem[3]);
		mem += 4;
	}
	fprintf(stderr, "#############################################\n");
}

static inline
void fi_opx_init_debug_counters(struct fi_opx_debug_counters *counters) {
	memset(counters, 0, sizeof(struct fi_opx_debug_counters));
}

static inline
void fi_opx_print_debug_counters(struct fi_opx_debug_counters *counters) {

	pid_t pid = getpid();
	fprintf(stderr, "(%d) ### DEBUG COUNTERS ###\n", pid);

	fprintf(stderr, "(%d) ### mp_eager_send_first_packets                %lu\n", pid, counters->mp_eager_send_first_packets);
	fprintf(stderr, "(%d) ### mp_eager_send_nth_packets                  %lu\n", pid, counters->mp_eager_send_nth_packets);
	fprintf(stderr, "(%d) ### mp_eager_send_first_force_cr               %lu\n", pid, counters->mp_eager_send_first_force_cr);
	fprintf(stderr, "(%d) ### mp_eager_send_nth_force_cr                 %lu\n", pid, counters->mp_eager_send_nth_force_cr);
	fprintf(stderr, "(%d) ### mp_eager_send_fall back to rzv             %lu\n", pid, counters->mp_eager_send_fall_back_to_rzv);
	fprintf(stderr, "(%d) ### mp_eager_send_full replay buffer rx poll   %lu\n", pid, counters->mp_eager_send_full_replay_buffer_rx_poll);

	fprintf(stderr, "(%d) ### mp_eager_recv_max_ue_queue_length          %lu\n", pid, counters->mp_eager_recv_max_ue_queue_length);
	fprintf(stderr, "(%d) ### mp_eager_recv_max_mq_queue_length          %lu\n", pid, counters->mp_eager_recv_max_mq_queue_length);
	fprintf(stderr, "(%d) ### mp_eager_recv_first_packets                %lu\n", pid, counters->mp_eager_recv_first_packets);
	fprintf(stderr, "(%d) ### mp_eager_recv_nth_packets                  %lu\n", pid, counters->mp_eager_recv_nth_packets);
	fprintf(stderr, "(%d) ### mp_eager_recv_completed_process_context    %lu\n", pid, counters->mp_eager_recv_completed_process_context);
	fprintf(stderr, "(%d) ### mp_eager_recv_completed_eager_first        %lu\n", pid, counters->mp_eager_recv_completed_eager_first);
	fprintf(stderr, "(%d) ### mp_eager_recv_completed_eager_nth          %lu\n", pid, counters->mp_eager_recv_completed_eager_nth);
	fprintf(stderr, "(%d) ### mp_eager_recv_nth_no_match                 %lu\n", pid, counters->mp_eager_recv_nth_no_match);
	fprintf(stderr, "(%d) ### mp_eager_recv_nth_match                    %lu\n", pid, counters->mp_eager_recv_nth_match);


	uint64_t total_completed = counters->mp_eager_recv_completed_process_context +
				counters->mp_eager_recv_completed_eager_first +
				counters->mp_eager_recv_completed_eager_nth;
	fprintf(stderr, "(%d) ### mp_eager_recv_total_completed -----------> %lu\n", pid, total_completed);
	fprintf(stderr, "(%d) ### mp_eager_recv_truncation                   %lu\n", pid, counters->mp_eager_recv_truncation);

	if (counters->mp_eager_recv_first_packets != total_completed) {
		fprintf(stderr, "!!!!! TOTAL COMPLETED %lu != mp_eager_first %lu!!!!! \n",
				total_completed, counters->mp_eager_recv_first_packets);
	}
}

static inline
void fi_opx_debug_counters_max_of(uint64_t *counter, uint64_t candidate) {
	if (candidate > (*counter)) (*counter) = candidate;
}

#endif
