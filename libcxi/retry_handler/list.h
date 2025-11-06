/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2020 Hewlett Packard Enterprise Development LP
 */

/* Adapted from linux's list.h */

struct list_head {
	struct list_head *next;
	struct list_head *prev;
};

static inline void init_list_head(struct list_head *list)
{
	list->next = list;
	list->prev = list;
}

static inline void __list_add(struct list_head *new, struct list_head *prev,
			      struct list_head *next)
{
	next->prev = new;
	new->next = next;
	new->prev = prev;
	prev->next = new;
}

static inline void list_add(struct list_head *new, struct list_head *head)
{
	__list_add(new, head, head->next);
}

static inline void list_add_tail(struct list_head *new, struct list_head *head)
{
	__list_add(new, head->prev, head);
}

static inline void __list_del(struct list_head *prev, struct list_head *next)
{
	next->prev = prev;
	prev->next = next;
}

static inline void list_del(struct list_head *entry)
{
	__list_del(entry->prev, entry->next);
}

static inline void list_del_init(struct list_head *entry)
{
	list_del(entry);
	init_list_head(entry);
}

static inline int list_empty(const struct list_head *head)
{
	return head->next == head;
}

#define container_of(ptr, type, member) ({ \
			const typeof(((type *)0)->member)*__mptr = (ptr); \
			(type *)((char *)__mptr - offsetof(type, member)); })

#define list_entry(ptr, type, member) \
	container_of(ptr, type, member)

#define list_first_entry(ptr, type, member)	\
	list_entry((ptr)->next, type, member)

#define list_next_entry(pos, member)		\
	list_entry((pos)->member.next, typeof(*(pos)), member)

#define list_for_each_entry(pos, head, member)              \
	for (pos = list_first_entry(head, typeof(*pos), member);       \
	     &pos->member != (head);				       \
	     pos = list_next_entry(pos, member))

#define list_for_each_entry_safe(pos, n, head, member)			\
	for (pos = list_first_entry(head, typeof(*pos), member),	\
	     n = list_next_entry(pos, member);				\
	     &pos->member != (head);					\
	     pos = n, n = list_next_entry(n, member))

#define list_first_entry_or_null(ptr, type, member) ({ \
	struct list_head *head__ = (ptr);	 \
	struct list_head *pos__ = head__->next;	 \
	pos__ != head__ ? list_entry(pos__, type, member) : NULL; \
})
