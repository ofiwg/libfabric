/*
 * Copyright (c) 2019 Intel Corporation. All rights reserved.
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

#include <pattern.h>
#include <core.h>

static int pattern_next(int *cur)
{

	if ((pm_job.my_rank == 0 ? pm_job.num_ranks - 1 : pm_job.my_rank - 1) == *cur){
		return -ENODATA;
	}
	else{
		if (pm_job.my_rank == 0)
			*cur = pm_job.num_ranks - 1;
		else 			
			*cur = pm_job.my_rank - 1;
		return 0; 
	}
}

static int pattern_current(int *cur)
{
	if ((pm_job.my_rank + 1) % pm_job.num_ranks == *cur){
		return -ENODATA;
	} else {
		*cur = (pm_job.my_rank + 1) % pm_job.num_ranks;
		return 0; 
	}
}

struct pattern_ops ring_ops = {
	.name = "ring",
	.next_source = pattern_next,
	.next_target = pattern_current,	
};
