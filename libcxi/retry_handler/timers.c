/* SPDX-License-Identifier: GPL-2.0-only or BSD-2-Clause
 * Copyright 2019 Hewlett Packard Enterprise Development LP
 */

/*  Timer implementation for Cassini retry handler */

#include "rh.h"

static void check_timer_list(struct retry_handler *rh);

static void timer_cb(uv_timer_t *handle)
{
	struct retry_handler *rh = handle->data;

	check_timer_list(rh);
}

/* Reset the timer watcher to 'timeout'. 'timeout' is absolute. */
static void set_timer(uint64_t timeout)
{
	uint64_t now;

	uv_update_time(loop);
	now = uv_now(loop);

	if (timeout <= now)
		timeout = 1;
	else
		timeout -= now;

	if (uv_is_active((uv_handle_t *)&timer_watcher))
		uv_timer_stop(&timer_watcher);

	uv_timer_start(&timer_watcher, timer_cb, timeout, 0);
}

/* Check expired timers on the waiting list */
static void check_timer_list(struct retry_handler *rh)
{
	struct timer_list *timer;
	struct timer_list *tmp;
	uint64_t now;

	uv_update_time(loop);
	now = uv_now(loop);

	rh->timer_generation++;

	list_for_each_entry_safe(timer, tmp, &rh->timeout_list.list, list) {
		/* The list is in time ascending order, so exit the
		 * function as timer hasn't expired.
		 */
		if (timer->timeout_ms > now)
			break;

		/* If the timer was added this generation (ie. its
		 * handler re-scheduled itself), skip it. Some work on
		 * the EQs may be needed to help progress first.
		 */
		if (timer->generation == rh->timer_generation)
			continue;

		/* Remove the timer from the list. It may be
		 * re-inserted with a new timeout, or disposed of.
		 */
		timer_del(timer);

		timer->func(rh, timer);
	}

	/* Program the timer to fire on the first entry, if there is
	 * one left.
	 */
	timer = list_first_entry_or_null(&rh->timeout_list.list,
					 struct timer_list, list);
	if (timer)
		set_timer(timer->timeout_ms);
}

/* Add a new timer in the timer list, in ascending order */
void timer_add(struct retry_handler *rh, struct timer_list *new,
	       const struct timeval *timeout)
{
	struct timer_list *iter;
	uint64_t now;
	struct timer_list *head = &rh->timeout_list;
	bool inserted = false;

	assert(new->func != NULL);

	uv_update_time(loop);
	now = uv_now(loop);

	/* It is possible that entry was already on the timer
	 * list. Reinitialize it just in case.
	 */
	timer_del(new);

	new->generation = rh->timer_generation;
	new->timeout_ms = now + timeout->tv_sec * 1000 + timeout->tv_usec / 1000;

	/* insert it */
	list_for_each_entry(iter, &head->list, list) {
		if (iter->timeout_ms > new->timeout_ms) {
			/* Insert before the iterator */
			list_add_tail(&new->list, &iter->list);
			inserted = true;

			break;
		}
	}

	if (!inserted) {
		/* Either the list was empty, or it goes at the end of the
		 * list
		 */
		list_add_tail(&new->list, &head->list);
	}

	/* If the new timer was added at the head, reset the event
	 * loop timer.
	 */
	if (head->list.next == &new->list)
		set_timer(new->timeout_ms);
}

/* Delete a timer from a timer list */
void timer_del(struct timer_list *timer)
{
	list_del_init(&timer->list);
}

/* Check if timer list has an associated timer */
bool timer_is_set(struct timer_list *timer)
{
	return !list_empty(&timer->list);
}
