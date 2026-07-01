/*
 * Copyright (c) 2004-2007 The Trustees of Indiana University and Indiana
 *                         University Research and Technology
 *                         Corporation.  All rights reserved.
 * Copyright (c) 2004-2009 The University of Tennessee and The University
 *                         of Tennessee Research Foundation.  All rights
 *                         reserved.
 * Copyright (c) 2004-2005 High Performance Computing Center Stuttgart,
 *                         University of Stuttgart.  All rights reserved.
 * Copyright (c) 2004-2005 The Regents of the University of California.
 *                         All rights reserved.
 * Copyright (c) 2006-2007 Voltaire. All rights reserved.
 * Copyright (c) 2009-2010 Cisco Systems, Inc.  All rights reserved.
 * Copyright (c) 2010-2018 Los Alamos National Security, LLC.
 *                         All rights reserved.
 * Copyright (c) 2020      Google, LLC. All rights reserved.
 * Copyright (c) Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 *  * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * - Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * - Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer listed
 *   in this license in the documentation and/or other materials
 *   provided with the distribution.
 *
 * - Neither the name of the copyright holders nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * The copyright holders provide no reassurances that the source code
 * provided does not infringe any patent, copyright, or any other
 * intellectual property rights of third parties.  The copyright holders
 * disclaim any liability to any recipient for claims brought against
 * recipient by any third party for infringement of that parties
 * intellectual property rights.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * ----------------[Copyright from inclusion of MPICH code]----------------
 *
 * The following is a notice of limited availability of the code, and disclaimer
 * which must be included in the prologue of the code and in all source listings
 * of the code.
 *
 * Copyright Notice
 *  + 2002 University of Chicago
 *
 * Permission is hereby granted to use, reproduce, prepare derivative works, and
 * to redistribute to others.  This software was authored by:
 *
 * Mathematics and Computer Science Division
 * Argonne National Laboratory, Argonne IL 60439
 *
 * (and)
 *
 * Department of Computer Science
 * University of Illinois at Urbana-Champaign
 *
 *
 * 			      GOVERNMENT LICENSE
 *
 * Portions of this material resulted from work developed under a U.S.
 * Government Contract and are subject to the following license: the Government
 * is granted for itself and others acting on its behalf a paid-up,
 * nonexclusive, irrevocable worldwide license in this computer software to
 * reproduce, prepare derivative works, and perform publicly and display
 * publicly.
 *
 * 				  DISCLAIMER
 *
 * This computer code material was prepared, in part, as an account of work
 * sponsored by an agency of the United States Government.  Neither the United
 * States, nor the University of Chicago, nor any of their employees, makes any
 * warranty express or implied, or assumes any legal liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately owned rights.
 */
/*
 * Copyright (c) 2023 Amazon.com, Inc. or its affiliates. All rights reserved.
 * Copyright (c) Intel Corporation. All rights reserved.
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
 * Multi Writer, Single Reader FIFO Queue (Not Thread Safe)
 * This implementation of this Queue is a one directional linked list
 * with head/tail pointers where every pointer is a relative offset
 * into the Shared Memory Region.
 */

#ifndef _SMR_FIFO_H_
#define _SMR_FIFO_H_

#include "smr_atom.h"
#include <stdint.h>
#include <assert.h>

#define SMR_FIFO_FREE (-3)
#define OFI_CACHE_LINE_SIZE (64)

static inline int64_t smr_local_to_offset(uintptr_t base, uintptr_t local_ptr)
{
	return (int64_t)(local_ptr - base);
}

static inline uintptr_t smr_offset_to_local(uintptr_t base, int64_t offset)
{
	return (uintptr_t) base + offset;
}

struct smr_fifo {
	uintptr_t	head;
	uint8_t		pad[OFI_CACHE_LINE_SIZE - sizeof(uintptr_t)];
	uintptr_t	tail;
};

static inline void smr_fifo_init(struct smr_fifo *fifo)
{
	fifo->head = SMR_FIFO_FREE;
	fifo->tail = SMR_FIFO_FREE;
}

static inline void smr_fifo_write(struct smr_fifo *fifo, uintptr_t peer_base,
				  uintptr_t cmd)
{
	uintptr_t ptr_offset = smr_local_to_offset(peer_base, cmd);
	uintptr_t prev_offset;
	uintptr_t prev_cmd;

	assert(fifo->head != 0 && fifo->tail != 0);

	// First entry in cmd must be free for use as next pointer in the FIFO
	*((void **) cmd) = (void *) SMR_FIFO_FREE;

	atomic_wmb();
	prev_offset = atomic_swap_ptr(&fifo->tail, ptr_offset);
	atomic_rmb();

	assert(prev_offset != ptr_offset);

	prev_cmd = smr_offset_to_local(peer_base, prev_offset);
	if (prev_offset != SMR_FIFO_FREE)
		*((void **) prev_cmd) = (void *) ptr_offset;
	else
		fifo->head = ptr_offset;

	atomic_wmb();
}

static inline void *smr_fifo_read(struct smr_fifo *fifo, uintptr_t base)
{
	uintptr_t cmd, cmd_offset, next_offset;

	assert(fifo->head != 0 && fifo->tail != 0);

	if (fifo->head == SMR_FIFO_FREE)
		return NULL;

	atomic_rmb();

	cmd_offset = fifo->head;
	fifo->head = SMR_FIFO_FREE;

	cmd = smr_offset_to_local(base, cmd_offset);
	next_offset = (uintptr_t) *((void **) cmd);
	assert(next_offset != cmd_offset && cmd && next_offset);

	if (next_offset == SMR_FIFO_FREE) {
		atomic_rmb();
		if (!atomic_compare_exchange(&fifo->tail, &cmd_offset,
					     SMR_FIFO_FREE)) {
			while ((uintptr_t) *((void **) cmd) == SMR_FIFO_FREE) {
				atomic_rmb();
			}
			fifo->head = (uintptr_t) *((void **) cmd);
		}
	} else {
		fifo->head = next_offset;
	}

	atomic_wmb();
	return (void *)cmd;
}

#endif /* _SMR_FIFO_H_ */
