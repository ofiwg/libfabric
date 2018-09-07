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
 */

/* wrapper around MPI callbacks and job metadata */
struct job {
	/* type signature of address exchange callback that harness passes to core */
	int (*address_exchange) (void *my_address,
				void *addresses,
				 int size, int count);
	/* synchronization barrier callback signature */
	void (*barrier) ();
	size_t			rank;
	size_t			ranks;
};

/*
 * Core combines a test and pattern, and drives the callback routines in each.
 *
 * Core doesn't interact with MPI directly, rather it's passed some data about
 *     the MPI environment and callback functions that invoke MPI functionality.
 *
 * Arguments:
 *
 *  argc,argv: command line arguments passed from main (in harness.c).  This
 *     includes the pattern and test arguments, which are separated out by core.
 *
 *  num_mpi_ranks: number of ranks in MPI job.
 *
 *  our_mpi_rank: MPI rank of current process.
 *
 *  address_exchange: MPI allgather function used to share host addresses.
 *
 *  barrier: MPI barrier function.
 */

int core(const int argc, char * const *argv, struct job *job);

void hpcs_error(const char* format, ...);
void hpcs_verbose(const char* format, ...);
