/*
 * Copyright (c) 2015-2016 Cray Inc. All rights reserved.
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
 * Examples:
 *
 * Init:
 *
 * _gnix_posted_tag_storage_init(&ep->posted_recvs, NULL);
 * _gnix_unexpected_tag_storage_init(&ep->unexpected_recvs, NULL);
 *
 * On receipt of a message:
 *
 * fastlock_acquire(&ep->tag_lock);
 * req = _gnix_remove_by_tag(&ep->posted_recvs, msg->tag, 0);
 * if (!req)
 *     _gnix_insert_by_tag(&ep->unexpected_recvs, msg->tag, msg->req);
 * fastlock_release(&ep->tag_lock);
 *
 * On post of receive:
 *
 * fastlock_acquire(&ep->tag_lock);
 * tag_req = _gnix_remove_by_tag(&ep->unexpected_recvs,
 *           req->tag, req->ignore);
 * if (!tag_req)
 *     _gnix_insert_by_tag(&ep->posted_recvs, tag, req);
 * fastlock_release(&ep->tag_lock);
 *
 */

#ifndef PROV_GNI_SRC_GNIX_TAGS_H_
#define PROV_GNI_SRC_GNIX_TAGS_H_

#include <stdlib.h>
#include <fi.h>
#include <fi_list.h>

#include "gnix_util.h"

/* enumerations */

/**
 * Enumeration for determining the underlying data structure for a
 * tag storage.
 *
 * Using auto will choose one of list, hlist or kdtree based on mem_tag_format.
 */
enum {
	GNIX_TAG_AUTOSELECT = 0,//!< GNIX_TAG_AUTOSELECT
	GNIX_TAG_LIST,          //!< GNIX_TAG_LIST
	GNIX_TAG_HLIST,         //!< GNIX_TAG_HLIST
	GNIX_TAG_KDTREE,        //!< GNIX_TAG_KDTREE
	GNIX_TAG_MAXTYPES,      //!< GNIX_TAG_MAXTYPES
};

/**
 * Enumeration for the tag storage states
 */
enum {
	GNIX_TS_STATE_UNINITIALIZED = 0,//!< GNIX_TS_STATE_UNINITIALIZED
	GNIX_TS_STATE_INITIALIZED,      //!< GNIX_TS_STATE_INITIALIZED
	GNIX_TS_STATE_DESTROYED,        //!< GNIX_TS_STATE_DESTROYED
};

/* forward declarations */
struct gnix_tag_storage;
struct gnix_fab_req;
struct gnix_address;

/* structure declarations */
/**
 * @brief Function dispatch table for the different types of underlying structures
 * used in the tag storage.
 *
 * @var insert_tag    insert a request into the tag storage
 * @var remove_tag    remove a request from the tag storage
 * @var peek_tag      probe tag storage for a specific tag
 * @var init          performs specific initialization based on underlying
 *                     data structure
 * @var fini          performs specific finalization based on underlying
 *                     data structure
 */
struct gnix_tag_storage_ops {
	int (*insert_tag)(struct gnix_tag_storage *ts, uint64_t tag,
			struct gnix_fab_req *req);
	struct gnix_fab_req *(*remove_tag)(struct gnix_tag_storage *ts,
			uint64_t tag, uint64_t ignore,
			uint64_t flags, void *context,
			struct gnix_address *addr);
	struct gnix_fab_req *(*peek_tag)(struct gnix_tag_storage *ts,
			uint64_t tag, uint64_t ignore,
			uint64_t flags, void *context,
			struct gnix_address *addr);
	int (*init)(struct gnix_tag_storage *ts);
	int (*fini)(struct gnix_tag_storage *ts);
	struct gnix_fab_req *(*remove_req_by_context)(struct gnix_tag_storage *ts,
			void *context);
};

/**
 * @note The sequence and generation numbers will be used in the future for
 *       optimizing the search with branch and bound.
 */
struct gnix_tag_list_element {
	 /* entry to the next element in the list */
	struct slist_entry free;
    /* has element been claimed with FI_CLAIM? */
	int claimed;
    /* associated fi_context with claimed element */
	void *context;
	/* sequence number */
	uint32_t seq;
	/* generation number */
	uint32_t gen;
};

/**
 * @note The type field is based on the GNIX_TAG_* enumerations listed above
 */
struct gnix_tag_storage_attr {
	/* one of 'auto', 'list', 'hlist' or 'kdtree' */
	int type;
	/* should the tag storage check addresses? */
	int use_src_addr_matching;
};

/**
 * @note Unused. This will be used in the future for the init heuristic when
 *         performing auto detection based on the mem_tag_format.
 */
struct gnix_tag_field {
	uint64_t mask;
	uint64_t len;
};

/**
 * @note Unused. This will be used in the future for the init heuristic when
 *         performing auto detection based on the mem_tag_format.
 */
struct gnix_tag_format {
	int field_cnt;
	struct gnix_tag_field *fields;
};

struct gnix_tag_list {
	struct slist list;
};

struct gnix_tag_hlist {
	struct dlist_entry *array;
	int elements;
};

struct gnix_tag_kdtree {

};

/**
 * @brief gnix tag storage structure
 *
 * Used to store gnix_fab_requests by tag, and optionally, by address.
 *
 * @var seq         sequence counter for elements
 * @var state       state of the tag storage structure
 * @var gen         generation counter for elements
 * @var match_func  matching function for the tag storage, either posted or
 *                    unexpected
 * @var attr        tag storage attributes
 * @var ops         function dispatch table for underlying data structures
 * @var tag_format  unused. used during init for determining what type of
 *                  data structure to use for storing data
 */
struct gnix_tag_storage {
	atomic_t seq;
	int state;
	int gen;
	int (*match_func)(struct slist_entry *entry, const void *arg);
	struct gnix_tag_storage_attr attr;
	struct gnix_tag_storage_ops *ops;
	struct gnix_tag_format tag_format;
	union {
		struct gnix_tag_list list;
		struct gnix_tag_hlist hlist;
		struct gnix_tag_kdtree kdtree;
	};
};

/* function declarations */
/**
 * @brief generic matching function for posted and unexpected tag storages
 *
 * @param req                     gnix fabric request to match
 * @param tag                     tag to match
 * @param ignore                  bits to ignore in the tags
 * @param flags                   fi_tagged flags
 * @param context                 fi_context to match in request
 * @param uses_src_addr_matching  should we check addresses?
 * @param addr                    gnix address to match
 * @param matching_posted         is matching on a posted tag storage?
 * @return 1 if this request matches the parameters, 0 otherwise
 */
int _gnix_req_matches_params(
		struct gnix_fab_req *req,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		int use_src_addr_matching,
		struct gnix_address *addr,
		int matching_posted);

/**
 * @brief matching function for unexpected tag storages
 *
 * @param entry  slist entry pointing to the request to search
 * @param arg    search parameters as a gnix_tag_search_element
 * @return 1 if this request matches the parameters, 0 otherwise
 */
int _gnix_match_unexpected_tag(struct slist_entry *entry, const void *arg);

/**
 * @brief matching function for posted tag storages
 *
 * @param entry  slist entry pointing to the request to search
 * @param arg    search parameters as a gnix_tag_search_element
 * @return 1 if this request matches the parameters, 0 otherwise
 */
int _gnix_match_posted_tag(struct slist_entry *entry, const void *arg);

/**
 * @brief base initialization function for tag storages
 * @note  This function should never be called directly. It is exposed for the
 *        purpose of allowing the test suite to reinitialize tag storages
 *        without knowing what type of tag storage is being reinitialized
 *
 * @param ts          tag storage pointer
 * @param attr        tag storage attributes
 * @param match_func  match function to be used on individual list elements
 * @return -FI_EINVAL, if any invalid parameters were given
 *         FI_SUCCESS, otherwise
 */
int _gnix_tag_storage_init(
		struct gnix_tag_storage *ts,
		struct gnix_tag_storage_attr *attr,
		int (*match_func)(struct slist_entry *, const void *));

/**
 * @brief initialization function for posted tag storages
 *
 * @param ts          tag storage pointer
 * @param attr        tag storage attributes
 * @param match_func  match function to be used on individual list elements
 * @return -FI_EINVAL, if any invalid parameters were given
 *         FI_SUCCESS, otherwise
 */
static inline int _gnix_posted_tag_storage_init(
		struct gnix_tag_storage *ts,
		struct gnix_tag_storage_attr *attr)
{
	return _gnix_tag_storage_init(ts, attr, _gnix_match_posted_tag);
}

/**
 * @brief initialization function for unexpected tag storages
 *
 * @param ts          tag storage pointer
 * @param attr        tag storage attributes
 * @param match_func  match function to be used on individual list elements
 * @return -FI_EINVAL, if any invalid parameters were given
 *         FI_SUCCESS, otherwise
 */
static inline int _gnix_unexpected_tag_storage_init(
		struct gnix_tag_storage *ts,
		struct gnix_tag_storage_attr *attr)
{
	return _gnix_tag_storage_init(ts, attr, _gnix_match_unexpected_tag);
}

/**
 * @brief destroys a tag storage and releases any held memory
 *
 * @param ts
 * @return -FI_EINVAL, if the tag storage is in a bad state
 *         -FI_EAGAIN, if there are tags remaining in the tag storage
 *         FI_SUCCESS, otherwise
 */
int _gnix_tag_storage_destroy(struct gnix_tag_storage *ts);

/**
 * @brief inserts a gnix_fab_req into the tag storage
 *
 * @param ts           pointer to the tag storage
 * @param tag          tag associated with fab request
 * @param req          gnix fabric request
 * @param ignore       bits to ignore in tag (only applies to posted)
 * @param addr_ignore  bits to ignore in addr (only applies to posted)
 * @return
 *
 * @note if ts is a posted tag storage, 'req->ignore_bits' will be set to
 *         the value of 'ignore'.
 *
 * @note if ts is a posted tag storage and ts->attr.use_src_addr_matching
 *         is enabled, 'req->addr_ignore_bits' will be set to the value
 *         of 'addr_ignore'.
 */
int _gnix_insert_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		struct gnix_fab_req *req,
		uint64_t ignore);


/**
 * @brief matches at a request from the tag storage by tag and address
 *
 * @param ts           pointer to the tag storage
 * @param tag          tag to remove
 * @param ignore       bits to ignore in tag
 * @param flags        fi_tagged flags
 * @param context      fi_context associated with tag
 * @param addr         gnix_address associated with tag
 * @param addr_ignore  bits to ignore in address
 * @return NULL, if no entry found that matches parameters
 *         otherwise, a non-null value pointing to a gnix_fab_req
 *
 * @note ignore parameter is not used for posted tag storages
 * @note addr_ignore parameter is not used for posted tag storages
 * @note if FI_CLAIM is not provided in flags, the call is an implicit removal
 *       of the tag
 */
struct gnix_fab_req *_gnix_match_tag(
		struct gnix_tag_storage *ts,
		uint64_t tag,
		uint64_t ignore,
		uint64_t flags,
		void *context,
		struct gnix_address *addr);

struct gnix_fab_req *_gnix_remove_req_by_context(
		struct gnix_tag_storage *ts,
		void *context);

/* external symbols */



#endif /* PROV_GNI_SRC_GNIX_TAGS_H_ */
