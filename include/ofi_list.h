/*
 * Copyright (c) 2011-2015 Intel Corporation.  All rights reserved.
 * Copyright (c) 2016 Cray Inc.  All rights reserved.
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

#ifndef _OFI_LIST_H_
#define _OFI_LIST_H_

#include "config.h"

#include <sys/types.h>
#include <stdlib.h>

#include <ofi_signal.h>


/*
 * Double-linked list
 */
struct dlist_entry {
	struct dlist_entry	*next;
	struct dlist_entry	*prev;
};

#define DLIST_INIT(addr) { addr, addr }
#define DEFINE_LIST(name) struct dlist_entry name = DLIST_INIT(&name)

static inline void dlist_init(struct dlist_entry *head)
{
	head->next = head;
	head->prev = head;
}

static inline int dlist_empty(struct dlist_entry *head)
{
	return head->next == head;
}

static inline void
dlist_insert_after(struct dlist_entry *item, struct dlist_entry *head)
{
	item->next = head->next;
	item->prev = head;
	head->next->prev = item;
	head->next = item;
}

static inline void
dlist_insert_before(struct dlist_entry *item, struct dlist_entry *head)
{
	dlist_insert_after(item, head->prev);
}

#define dlist_insert_head dlist_insert_after
#define dlist_insert_tail dlist_insert_before

static inline void dlist_remove(struct dlist_entry *item)
{
	item->prev->next = item->next;
	item->next->prev = item->prev;
}

static inline void dlist_remove_init(struct dlist_entry *item)
{
	dlist_remove(item);
	dlist_init(item);
}

#define dlist_pop_front(head, type, container, member)			\
	do {								\
		container = container_of((head)->next, type, member);	\
		dlist_remove((head)->next);				\
	} while (0)

#define dlist_foreach(head, item) 						\
	for ((item) = (head)->next; (item) != (head); (item) = (item)->next)

#define dlist_foreach_container(head, type, container, member)			\
	for ((container) = container_of((head)->next, type, member);		\
	     &((container)->member) != (head);					\
	     (container) = container_of((container)->member.next,		\
					type, member))

#define dlist_foreach_safe(head, item, tmp)					\
	for ((item) = (head)->next, (tmp) = (item)->next; (item) != (head);	\
             (item) = (tmp), (tmp) = (item)->next)

#define dlist_foreach_container_safe(head, type, container, member, tmp)	\
	for ((container) = container_of((head)->next, type, member),		\
	     (tmp) = (container)->member.next;					\
	     &((container)->member) != (head);					\
	     (container) = container_of((tmp), type, member),			\
	     (tmp) = (container)->member.next)

typedef int dlist_func_t(struct dlist_entry *item, const void *arg);

static inline struct dlist_entry *
dlist_find_first_match(struct dlist_entry *head, dlist_func_t *match,
		       const void *arg)
{
	struct dlist_entry *item;

	dlist_foreach(head, item) {
		if (match(item, arg))
			return item;
	}

	return NULL;
}

static inline struct dlist_entry *
dlist_remove_first_match(struct dlist_entry *head, dlist_func_t *match,
			 const void *arg)
{
	struct dlist_entry *item;

	item = dlist_find_first_match(head, match, arg);
	if (item)
		dlist_remove(item);

	return item;
}

/* splices list at the front of the list 'head'
 *
 * BEFORE:
 * head:      HEAD->a->b->c->HEAD
 * to_splice: HEAD->d->e->HEAD
 *
 * AFTER:
 * head:      HEAD->d->e->a->b->c->HEAD
 * to_splice: HEAD->HEAD (empty list)
 */
static inline void dlist_splice_head(struct dlist_entry *head,
				     struct dlist_entry *to_splice)
{
	if (dlist_empty(to_splice))
		return;

	/* hook first element of 'head' to last element of 'to_splice' */
	head->next->prev = to_splice->prev;
	to_splice->prev->next = head->next;

	/* put first element of 'to_splice' as first element of 'head' */
	head->next = to_splice->next;
	head->next->prev = head;

	/* set list to empty */
	dlist_init(to_splice);
}

/* splices list at the back of the list 'head'
 *
 * BEFORE:
 * head:      HEAD->a->b->c->HEAD
 * to_splice: HEAD->d->e->HEAD
 *
 * AFTER:
 * head:      HEAD->a->b->c->d->e->HEAD
 * to_splice: HEAD->HEAD (empty list)
 */
static inline void dlist_splice_tail(struct dlist_entry *head,
				     struct dlist_entry *to_splice)
{
	dlist_splice_head(head->prev, to_splice);
}

/*
 * Single-linked list
 */
struct slist_entry {
	struct slist_entry	*next;
};

struct slist {
	struct slist_entry	*head;
	struct slist_entry	*tail;
};

static inline void slist_init(struct slist *list)
{
	list->head = list->tail = NULL;
}

static inline int slist_empty(struct slist *list)
{
	return !list->head;
}

static inline void slist_insert_head(struct slist_entry *item, struct slist *list)
{
	if (slist_empty(list))
		list->tail = item;
	else
		item->next = list->head;

	list->head = item;
}

static inline void slist_insert_tail(struct slist_entry *item, struct slist *list)
{
	if (slist_empty(list))
		list->head = item;
	else
		list->tail->next = item;

	list->tail = item;
}

static inline struct slist_entry *slist_remove_head(struct slist *list)
{
	struct slist_entry *item;

	item = list->head;
	if (list->head == list->tail)
		slist_init(list);
	else
		list->head = item->next;
#if ENABLE_DEBUG
	if (item) {
		item->next = NULL;
	}
#endif
	return item;
}

#define slist_foreach(list, item, prev)				\
	for ((prev) = NULL, (item) = (list)->head; (item); 	\
			(prev) = (item), (item) = (item)->next)


typedef int slist_func_t(struct slist_entry *item, const void *arg);

static inline struct slist_entry *
slist_find_first_match(const struct slist *list, slist_func_t *match,
			const void *arg)
{
	struct slist_entry *item;
	for (item = list->head; item; item = item->next) {
		if (match(item, arg))
			return item;
	}

	return NULL;
}

static inline void slist_remove(struct slist *list,
		struct slist_entry *item, struct slist_entry *prev)
{
	if (prev)
		prev->next = item->next;
	else
		list->head = item->next;

	if (!item->next)
		list->tail = prev;
}

static inline struct slist_entry *
slist_remove_first_match(struct slist *list, slist_func_t *match, const void *arg)
{
	struct slist_entry *item, *prev;

	slist_foreach(list, item, prev) {
		if (match(item, arg)) {
			slist_remove(list, item, prev);
			return item;
		}
	}

	return NULL;
}

/*
 * Double-linked list with blocking wait-until-avail support
 */

struct dlistfd_head {
	struct dlist_entry list;
	struct fd_signal   signal;
};

static inline int dlistfd_head_init(struct dlistfd_head *head)
{
	dlist_init(&head->list);
	return fd_signal_init(&head->signal);
}

static inline void dlistfd_head_free(struct dlistfd_head *head)
{
	fd_signal_free(&head->signal);
}

static inline int dlistfd_empty(struct dlistfd_head *head)
{
	return dlist_empty(&head->list);
}

static inline void dlistfd_signal(struct dlistfd_head *head)
{
	fd_signal_set(&head->signal);
}

static inline void dlistfd_reset(struct dlistfd_head *head)
{
	if (dlistfd_empty(head))
		fd_signal_reset(&head->signal);
}

static inline void
dlistfd_insert_head(struct dlist_entry *item, struct dlistfd_head *head)
{
	dlist_insert_after(item, &head->list);
	dlistfd_signal(head);
}

static inline void
dlistfd_insert_tail(struct dlist_entry *item, struct dlistfd_head *head)
{
	dlist_insert_before(item, &head->list);
	dlistfd_signal(head);
}

static inline void dlistfd_remove(struct dlist_entry *item, struct dlistfd_head *head)
{
	dlist_remove(item);
	dlistfd_reset(head);
}

static inline int dlistfd_wait_avail(struct dlistfd_head *head, int timeout)
{
	int ret;

	if (!dlistfd_empty(head))
		return 1;

	ret = fd_signal_poll(&head->signal, timeout);
	return ret ? ret : !dlistfd_empty(head);
}

#endif /* _OFI_LIST_H_ */
