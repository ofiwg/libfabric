/*
 * Copyright (c) 2017-2018 Intel Corporation. All rights reserved.
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

#pragma once

#include <stdlib.h>
#include <stdbool.h>

#include "pattern/user.h"
#include "test/user.h"


/*
 * -----------------------------------------------------------------------------
 * CORE API
 * -----------------------------------------------------------------------------
 *
 * Core combines a test and pattern, and drives the callback routines in each.
 * Arguments are positional in argv.  The arguments are:
 *
 * argv[0]: Provider Name (%128s): The name of the provider to use for
 * transport.  Corresponds to the OFI option "prov_name".
 *
 * argv[1]: Domains per Process (%zu): The number of OFI domains that should be
 * created per process.
 *
 * argv[2]: Endpoints per Domain (%zu): The number of OFI endpoints to create
 * per domain.
 *
 * argv[3]: Transfers per EP Pair (%zu): The number of transfers between each
 * endpoint pair that should occur (i.e., repetitions).
 *
 * argv[4]: Window Size (%zu): Window size is directly related to the amount of
 * memory each process is allowed to consume.  The exact consumption depends on
 * the specific test implementation.  Operations which require a context
 * (FI_CONTEXT) or will generate a completion in a completion queue will consume
 * resources controlled by window size.
 *
 * argv[5]: Callback Order (char *): Test callback ordering refers to whether or
 * not the TX test callbacks or the RX test callbacks are executed before the
 * other.  This allows testing of expected vs. unexpected data transfers, but at
 * a not-insignificant runtime cost.  Possible values are:
 * "CALLBACK_ORDER_NONE", "CALLBACK_ORDER_EXPECTED", and
 * "CALLBACK_ORDER_UNEXPECTED".  No test callback ordering guarantees generally
 * means whatever is the most convenient execution order in terms of the pattern
 * generator, but can only safely be used with the FI_MSG and FI_TAGGED
 * interfaces.  Data transfers within the FI_RMA/FI_ATOMIC scope can only safely
 * be run with expected test callback ordering, or internal errors may be
 * generated and reported.
 *
 * argv[6-9]: Completion Object Sharing (%u):  Whether or not the TX counters,
 * RX counters, TX completion queue, and RX completion queue respectively should
 * be shared.  Counter or completion queue sharing is a concept that has no
 * bearing on the logic specific to any test implementations, but can have
 * significant impacts on the logic paths taken by the fabric provider, and is
 * thus provided as a configurable option.  A shared counter or completion queue
 * (1) will be shared by all endpoints in a domain, whereas an unshared counter
 * (0) will be bound to only a single endpoint.
 *
 * num_mpi_ranks: The number of MPI ranks in the job (i.e., the total number of
 * processes).
 *
 * our_mpi_rank: This process's MPI rank.
 */

/* type signature of address exchange callback that harness passes to core */
typedef int (address_exchange_t) (void *my_address, void *addresses, int size, int count);

/* synchronization barrier callback signature */
typedef void (barrier_t) ();

int core(
		const int argc,
		char * const *argv,
		const int num_mpi_ranks,
		const int our_mpi_rank,
		address_exchange_t address_exchange,
		barrier_t barrier);

void hpcs_error(const char* format, ...);
void hpcs_verbose(const char* format, ...);
