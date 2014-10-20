/*
 * Copyright (c) 2014 Intel Corporation.  All rights reserved.
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
 *
 */

#if !defined(RBUF_H)
#define RBUF_H

#if HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#include <fi.h>


struct ringbuf {
	size_t		size;
	size_t		size_mask;
	size_t		rcnt;
	size_t		wcnt;
	void		*buf;
};

static inline int rbinit(struct ringbuf *rb, size_t size)
{
	rb->size = roundup_power_of_two(size);
	rb->size_mask = rb->size - 1;
	rb->rcnt = 0;
	rb->wcnt = 0;
	rb->buf = calloc(1, rb->size);
	if (!rb->buf)
		return -ENOMEM;
	return 0;
}

static inline void rbfree(struct ringbuf *rb)
{
	free(rb->buf);
}

static inline int rbfull(struct ringbuf *rb)
{
	return rb->wcnt - rb->rcnt >= rb->size;
}

static inline int rbempty(struct ringbuf *rb)
{
	return rb->wcnt == rb->rcnt;
}

static inline size_t rbused(struct ringbuf *rb)
{
	return rb->wcnt - rb->rcnt;
}

static inline size_t rbavail(struct ringbuf *rb)
{
	return rb->size - rbused(rb);
}

static inline void rbwrite(struct ringbuf *rb, void *buf, size_t len)
{
	memcpy(rb->buf + (rb->wcnt & rb->size_mask), buf, len);
	rb->wcnt += len;
}

static inline void rbpeek(struct ringbuf *rb, void *buf, size_t len)
{
	memcpy(buf, rb->buf + (rb->rcnt & rb->size_mask), len);
}

static inline void rbread(struct ringbuf *rb, void *buf, size_t len)
{
	rbpeek(rb, buf, len);
	rb->rcnt += len;
}

#endif /* RBUF_H */
