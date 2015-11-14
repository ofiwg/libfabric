/*
 * Copyright (c) 2015 Cray Inc. All rights reserved.
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

#include "rdma/fabric.h"
#include "rdma/fi_tagged.h"

#include "gnix_tags.h"

#include "gnix.h"
#include "gnix_util.h"

struct gnix_tag_storage_ops list_ops;
struct gnix_tag_storage_ops hlist_ops;
struct gnix_tag_storage_ops kdtree_ops;

struct gnix_tag_search_element {
	uint64_t tag;
	uint64_t ignore;
	void *context;
	uint64_t flags;
	int use_src_addr_matching;
	struct gnix_address *addr;
};

/**
 * @brief converts gnix_tag_list_element to gnix_fab_req
 *
 * @param elem  slist element embedded in a gnix_fab_req
 * @return pointer to gnix_fab_req
 */
static inline struct gnix_fab_req *__to_gnix_fab_req(
		struct gnix_tag_list_element *elem)
{
	struct gnix_fab_req *req;

	req = container_of(elem, struct gnix_fab_req, msg.tle);

	return req;
}

/**
 * @brief determines if a req matches the address parameters
 *
 * @param addr_to_find    address to find in tag storage
 * @param addr            stored address
 * @return 0 if the request does not match the parameters, 1 otherwise
 */
static inline int __req_matches_addr_params(
		struct gnix_address *addr_to_find,
		struct gnix_address *addr)
{
	return (GNIX_ADDR_UNSPEC(*addr) ||
			GNIX_ADDR_EQUAL(*addr_to_find, *addr));
}

int _gnix_req_matches_params(
		struct gnix_fab_req *req,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		int use_src_addr_matching,
		struct gnix_address *addr,
		int matching_posted)
{
	int valid_request;

	/* adding some error checking to the first condition
	 *
	 * if the context is null, then FI_PEEK | FI_CLAIM should fail
	 */
	if ((flags & FI_CLAIM) && (flags & FI_PEEK))
		valid_request = (context != NULL &&
				context != req->msg.tle.context);
	else if ((flags & FI_CLAIM) && !(flags & FI_PEEK))
		valid_request = (req->msg.tle.context != NULL &&
				context == req->msg.tle.context);
	else
		valid_request = req->msg.tle.context == NULL;

	/* shortcut */
	if (!valid_request)
		return valid_request;

	if (use_src_addr_matching && matching_posted) {
		/* if matching posted, flip the arguments so that the unspec check
		 *   is done on the request in the tag store and not the address
		 *   that was passed into the function
		 */
		valid_request &= __req_matches_addr_params(addr, &req->addr);
	} else if (use_src_addr_matching && !matching_posted) {
		valid_request &= __req_matches_addr_params(&req->addr, addr);
	}

	return valid_request && ((req->msg.tag & ~ignore) == (tag & ~ignore));
}

static int __req_matches_context(struct slist_entry *entry, const void *arg)
{
	struct gnix_tag_list_element *tle;
	struct gnix_fab_req *req;

	tle = container_of(entry, struct gnix_tag_list_element, free);
	req = __to_gnix_fab_req(tle);

	return req->user_context == arg;
}

/* used to match elements in the posted lists */
int _gnix_match_posted_tag(struct slist_entry *entry, const void *arg)
{
	const struct gnix_tag_search_element *s_elem = arg;
	struct gnix_tag_list_element *tle;
	struct gnix_fab_req *req;

	tle = container_of(entry, struct gnix_tag_list_element, free);
	req = __to_gnix_fab_req(tle);

	return _gnix_req_matches_params(req, s_elem->tag, req->msg.ignore,
			s_elem->flags, s_elem->context,
			s_elem->use_src_addr_matching,
			s_elem->addr, 1);
}

/* used to match elements in the unexpected lists */
int _gnix_match_unexpected_tag(struct slist_entry *entry, const void *arg)
{
	const struct gnix_tag_search_element *s_elem = arg;
	struct gnix_tag_list_element *tle;
	struct gnix_fab_req *req;

	tle = container_of(entry, struct gnix_tag_list_element, free);
	req = __to_gnix_fab_req(tle);

	return _gnix_req_matches_params(req, s_elem->tag, s_elem->ignore,
			s_elem->flags, s_elem->context,
			s_elem->use_src_addr_matching,
			s_elem->addr, 0);
}

/* default attributes for tag storage objects */
static struct gnix_tag_storage_attr default_attr = {
		.type = GNIX_TAG_AUTOSELECT,
		.use_src_addr_matching = 0,
};

/**
 * @brief peeks into a tag list to find the first match using given parameters
 *
 * @param ts           pointer to gnix_tag_storage_object
 * @param tag          tag to find
 * @param ignore       bits to ignore in tags
 * @param list         slist to search
 * @param flags        fi_tagged flags
 * @param context      fi_context associated with tag
 * @param addr         gnix_address to find
 * @param addr_ignore  bits to ignore in address
 * @return NULL, if no match is found,
 *         a non-NULL value, if a match is found
 */
static inline struct gnix_tag_list_element *__tag_list_peek_first_match(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		struct slist *list,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct slist_entry *current;
	struct gnix_tag_search_element s_elem = {
			.tag = tag,
			.ignore = ignore,
			.flags = flags,
			.context = context,
			.use_src_addr_matching = ts->attr.use_src_addr_matching,
			.addr = addr,
	};

	/* search the list for a matching element. stop at the first match */
	for (current = list->head; current != NULL; current = current->next) {
		if (ts->match_func(current, &s_elem))
			return (struct gnix_tag_list_element *) current;
	}

	return NULL;
}

/**
 * @brief finds and removes the first match in a tag list
 *
 * @param ts           pointer to gnix_tag_storage_object
 * @param tag          tag to find
 * @param ignore       bits to ignore in tags
 * @param list         slist to search
 * @param flags        fi_tagged flags
 * @param context      fi_context associated with tag
 * @param addr         gnix_address to find
 * @param addr_ignore  bits to ignore in address
 * @return NULL, if no match is found,
 *         a non-NULL value, if a match is found
 */
static inline struct gnix_tag_list_element *__tag_list_find_element(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		struct slist *list,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct gnix_tag_search_element s_elem = {
			.tag = tag,
			.ignore = ignore,
			.flags = flags,
			.context = context,
			.use_src_addr_matching = ts->attr.use_src_addr_matching,
			.addr = addr,
	};

	/* search the list for a matching element. stop at the first match */
	return (struct gnix_tag_list_element *)
			slist_remove_first_match(list,
					ts->match_func, &s_elem);
}

/**
 * @brief checks attributes for invalid values
 *
 * @param attr  attributes to be checked
 * @return -FI_EINVAL, if attributes contain invalid values
 *         FI_SUCCESS, otherwise
 */
static inline int __check_for_invalid_attributes(
		struct gnix_tag_storage_attr *attr)
{
	if (attr->type < 0 || attr->type >= GNIX_TAG_MAXTYPES)
		return -FI_EINVAL;

	return FI_SUCCESS;
}

int _gnix_tag_storage_init(
		struct gnix_tag_storage *ts,
		struct gnix_tag_storage_attr *attr,
		int (*match_func)(struct slist_entry *, const void *))
{
	int ret;
	struct gnix_tag_storage_attr *attributes = &default_attr;

	if (ts->state == GNIX_TS_STATE_INITIALIZED) {
		GNIX_WARN(FI_LOG_EP_CTRL,
				"attempted to initialize already active tag storage\n");
		return -FI_EINVAL;
	}

	if (attr) {
		if (__check_for_invalid_attributes(attr)) {
			GNIX_WARN(FI_LOG_EP_CTRL,
					"invalid attributes passed in to init\n");
			return -FI_EINVAL;
		}

		attributes = attr;
	}



	/* copy attributes */
	memcpy(&ts->attr, attributes, sizeof(struct gnix_tag_storage_attr));

	switch (ts->attr.type) {
	case GNIX_TAG_AUTOSELECT:
	case GNIX_TAG_LIST:
		ts->ops = &list_ops;
		break;
	case GNIX_TAG_HLIST:
		ts->ops = &hlist_ops;
		break;
	case GNIX_TAG_KDTREE:
		ts->ops = &kdtree_ops;
		break;
	default:
		GNIX_WARN(FI_LOG_EP_CTRL, "failed to find valid tag type\n");
		assert(0);
	}

	ret = ts->ops->init(ts);
	if (ret) {
		GNIX_WARN(FI_LOG_EP_CTRL,
				"failed to initialize at ops->init\n");
		return ret;
	}

	/* a different type of matching behavior is required for unexpected
	 *   messages
	 */
	ts->match_func = match_func;

	atomic_initialize(&ts->seq, 1);
	ts->gen = 0;
	ts->state = GNIX_TS_STATE_INITIALIZED;

	return FI_SUCCESS;
}

int _gnix_tag_storage_destroy(struct gnix_tag_storage *ts)
{
	int ret;

	if (ts->state != GNIX_TS_STATE_INITIALIZED)
		return -FI_EINVAL;

	ret = ts->ops->fini(ts);
	if (ret)
		return ret;

	ts->state = GNIX_TS_STATE_DESTROYED;

	return FI_SUCCESS;
}

/* not implemented operations */
static int __gnix_tag_no_init(struct gnix_tag_storage *ts)
{
	return -FI_ENOSYS;
}

static int __gnix_tag_no_fini(struct gnix_tag_storage *ts)
{
	return -FI_ENOSYS;
}

static int __gnix_tag_no_insert_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		struct gnix_fab_req *req)
{
	return -FI_ENOSYS;
}


static struct gnix_fab_req *__gnix_tag_no_peek_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	return NULL;
}

static struct gnix_fab_req *__gnix_tag_no_remove_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	return NULL;
}

static struct gnix_fab_req *__gnix_tag_no_remove_req_by_context(
		struct gnix_tag_storage *ts,
		void *context)
{
	return NULL;
}

/* list operations */

static int __gnix_tag_list_init(struct gnix_tag_storage *ts)
{
	slist_init(&ts->list.list);

	return FI_SUCCESS;
}

static int __gnix_tag_list_fini(struct gnix_tag_storage *ts)
{
	if (!slist_empty(&ts->list.list))
		return -FI_EAGAIN;

	return FI_SUCCESS;
}

static int __gnix_tag_list_insert_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		struct gnix_fab_req *req)
{
	struct gnix_tag_list_element *element;

	element = &req->msg.tle;
	element->free.next = NULL;
	element->context = NULL;
	slist_insert_tail(&element->free, &ts->list.list);

	return FI_SUCCESS;
}

static struct gnix_fab_req *__gnix_tag_list_peek_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct gnix_tag_list_element *element;

	element = __tag_list_peek_first_match(ts, tag, ignore,
			&ts->list.list, flags, context, addr);

	if (!element)
		return NULL;

	return __to_gnix_fab_req(element);
}

static struct gnix_fab_req *__gnix_tag_list_remove_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct gnix_tag_list_element *element;
	struct gnix_fab_req *req;

	element = __tag_list_find_element(ts, tag, ignore, &ts->list.list,
			flags, context, addr);
	if (!element)
		return NULL;

	req = __to_gnix_fab_req(element);
	element->free.next = NULL;

	return req;
}

static struct gnix_fab_req *__gnix_tag_list_remove_req_by_context(
		struct gnix_tag_storage *ts,
		void *context)
{
	struct gnix_tag_list_element *element;
	struct gnix_fab_req *req;

	element = (struct gnix_tag_list_element *)
			slist_remove_first_match(&ts->list.list,
			__req_matches_context, context);

	if (!element)
			return NULL;

	req = __to_gnix_fab_req(element);
	element->free.next = NULL;

	return req;
}

/* ignore is only used on inserting into posted tag storages
 * addr_ignore is only used on inserting into post tag storages with
 * use_src_addr_matching enabled
 */
int _gnix_insert_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		struct gnix_fab_req *req,
		uint64_t ignore)
{
	int ret;

	GNIX_INFO(FI_LOG_EP_CTRL, "inserting a message by tag, "
				"ts=%p tag=%llx req=%p\n", ts, tag, req);
	req->msg.tag = tag;
	if (ts->match_func == _gnix_match_posted_tag) {
		req->msg.ignore = ignore;
	}

	ret = ts->ops->insert_tag(ts, tag, req);

	GNIX_INFO(FI_LOG_EP_CTRL, "ret=%i\n", ret);

	return ret;
}

/*
 * ignore parameter is not used for posted tag storages
 */
static struct gnix_fab_req *__remove_by_tag_and_addr(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct gnix_fab_req *ret;

	/* assuming that flags and context are correct */
	GNIX_INFO(FI_LOG_EP_CTRL, "removing a message by tag, "
			"ts=%p tag=%llx ignore=%llx flags=%llx context=%p "
			"addr=%p\n",
			ts, tag, ignore, flags, context, addr);
	ret = ts->ops->remove_tag(ts, tag, ignore, flags, context, addr);
	GNIX_INFO(FI_LOG_EP_CTRL, "ret=%p\n", ret);

	return ret;
}

static struct gnix_fab_req *__peek_by_tag_and_addr(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	struct gnix_fab_req *ret;

	/* assuming that flags and context are correct */
	GNIX_INFO(FI_LOG_EP_CTRL, "peeking a message by tag, "
			"ts=%p tag=%llx ignore=%llx flags=%llx context=%p "
			"addr=%p\n",
			ts, tag, ignore, flags, context, addr);

	ret =  ts->ops->peek_tag(ts, tag, ignore, flags, context, addr);

	if (ret != NULL && (flags & FI_CLAIM)) {
		ret->msg.tle.context = context;
	}

	GNIX_INFO(FI_LOG_EP_CTRL, "ret=%p\n", ret);

	return ret;
}

struct gnix_fab_req *_gnix_match_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr)
{
	if ((flags & FI_PEEK) && !(flags & FI_DISCARD))
		return __peek_by_tag_and_addr(ts, tag, ignore, flags,
				context, addr);
	else
		return __remove_by_tag_and_addr(ts, tag, ignore, flags,
				context, addr);
}

struct gnix_fab_req *_gnix_remove_req_by_context(
		struct gnix_tag_storage *ts,
		void *context)
{
	return ts->ops->remove_req_by_context(ts, context);
}

struct gnix_tag_storage_ops list_ops = {
		.init = __gnix_tag_list_init,
		.fini = __gnix_tag_list_fini,
		.insert_tag = __gnix_tag_list_insert_tag,
		.peek_tag = __gnix_tag_list_peek_tag,
		.remove_tag = __gnix_tag_list_remove_tag,
		.remove_req_by_context = __gnix_tag_list_remove_req_by_context,
};

struct gnix_tag_storage_ops hlist_ops = {
		.init = __gnix_tag_no_init,
		.fini = __gnix_tag_no_fini,
		.insert_tag = __gnix_tag_no_insert_tag,
		.remove_tag = __gnix_tag_no_remove_tag,
		.peek_tag = __gnix_tag_no_peek_tag,
		.remove_req_by_context = __gnix_tag_no_remove_req_by_context,
};

struct gnix_tag_storage_ops kdtree_ops = {
		.init = __gnix_tag_no_init,
		.fini = __gnix_tag_no_fini,
		.insert_tag = __gnix_tag_no_insert_tag,
		.remove_tag = __gnix_tag_no_remove_tag,
		.peek_tag = __gnix_tag_no_peek_tag,
		.remove_req_by_context = __gnix_tag_no_remove_req_by_context,
};
