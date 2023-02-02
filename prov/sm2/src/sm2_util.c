/*
 * Copyright (c) 2016-2021 Intel Corporation. All rights reserved.
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

#include "config.h"
#include "sm2_common.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>


struct dlist_entry sm2_ep_name_list;
DEFINE_LIST(sm2_ep_name_list);
pthread_mutex_t sm2_ep_list_lock = PTHREAD_MUTEX_INITIALIZER;

void sm2_cleanup(void)
{
	struct sm2_ep_name *ep_name;
	struct dlist_entry *tmp;

	pthread_mutex_lock(&sm2_ep_list_lock);
	dlist_foreach_container_safe(&sm2_ep_name_list, struct sm2_ep_name,
				     ep_name, entry, tmp)
		free(ep_name);
	pthread_mutex_unlock(&sm2_ep_list_lock);
}

static void sm2_peer_addr_init(struct sm2_addr *peer)
{
	memset(peer->name, 0, SM2_NAME_MAX);
	peer->id = -1;
}

size_t sm2_calculate_size_offsets(size_t tx_count, size_t rx_count,
				  size_t *cmd_offset, size_t *resp_offset,
				  size_t *inject_offset, size_t *sar_offset,
				  size_t *peer_offset, size_t *name_offset,
				  size_t *sock_offset)
{
	size_t cmd_queue_offset, resp_queue_offset, inject_pool_offset;
	size_t sar_pool_offset, peer_data_offset, ep_name_offset;
	size_t tx_size, rx_size, total_size, sock_name_offset;

	tx_size = roundup_power_of_two(tx_count);
	rx_size = roundup_power_of_two(rx_count);

	/* Align cmd_queue offset to 128-bit boundary. */
	cmd_queue_offset = ofi_get_aligned_size(sizeof(struct sm2_region), 16);
	resp_queue_offset = cmd_queue_offset + sizeof(struct sm2_cmd_queue) +
			    sizeof(struct sm2_cmd) * rx_size;
	inject_pool_offset = resp_queue_offset + sizeof(struct sm2_resp_queue) +
			     sizeof(struct sm2_resp) * tx_size;
	sar_pool_offset = inject_pool_offset +
		freestack_size(sizeof(struct sm2_inject_buf), rx_size);
	peer_data_offset = sar_pool_offset +
		freestack_size(sizeof(struct sm2_sar_buf), SM2_MAX_PEERS);
	ep_name_offset = peer_data_offset + sizeof(struct sm2_peer_data) *
		SM2_MAX_PEERS;

	sock_name_offset = ep_name_offset + SM2_NAME_MAX;

	if (cmd_offset)
		*cmd_offset = cmd_queue_offset;
	if (resp_offset)
		*resp_offset = resp_queue_offset;
	if (inject_offset)
		*inject_offset = inject_pool_offset;
	if (sar_offset)
		*sar_offset = sar_pool_offset;
	if (peer_offset)
		*peer_offset = peer_data_offset;
	if (name_offset)
		*name_offset = ep_name_offset;
	if (sock_offset)
		*sock_offset = sock_name_offset;

	total_size = sock_name_offset + SM2_SOCK_NAME_MAX;

	/*
 	 * Revisit later to see if we really need the size adjustment, or
 	 * at most align to a multiple of a page size.
 	 */
	total_size = roundup_power_of_two(total_size);

	return total_size;
}

static int sm2_retry_map(const char *name, int *fd)
{
	char tmp[NAME_MAX];
	struct sm2_region *old_shm;
	struct stat sts;
	int shm_pid;

	*fd = shm_open(name, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	if (*fd < 0)
		return -errno;

	old_shm = mmap(NULL, sizeof(*old_shm), PROT_READ | PROT_WRITE,
		       MAP_SHARED, *fd, 0);
	if (old_shm == MAP_FAILED)
		goto err;

        /* No backwards compatibility for now. */
	if (old_shm->version != SM2_VERSION) {
		munmap(old_shm, sizeof(*old_shm));
		goto err;
	}
	shm_pid = old_shm->pid;
	munmap(old_shm, sizeof(*old_shm));

	if (!shm_pid)
		return FI_SUCCESS;

	memset(tmp, 0, sizeof(tmp));
	snprintf(tmp, sizeof(tmp), "/proc/%d", shm_pid);

	if (stat(tmp, &sts) == -1 && errno == ENOENT)
		return FI_SUCCESS;

err:
	close(*fd);
	shm_unlink(name);
	return -FI_EBUSY;
}

static void sm2_lock_init(pthread_spinlock_t *lock)
{
	pthread_spin_init(lock, PTHREAD_PROCESS_SHARED);
}

/* TODO: Determine if aligning SMR data helps performance */
int sm2_create(const struct fi_provider *prov, struct sm2_map *map,
	       const struct sm2_attr *attr, struct sm2_region *volatile *smr)
{
	struct sm2_ep_name *ep_name;
	size_t total_size, cmd_queue_offset, peer_data_offset;
	size_t resp_queue_offset, inject_pool_offset, name_offset;
	size_t sar_pool_offset, sock_name_offset;
	int fd, ret, i;
	void *mapped_addr;
	size_t tx_size, rx_size;

	tx_size = roundup_power_of_two(attr->tx_count);
	rx_size = roundup_power_of_two(attr->rx_count);
	total_size = sm2_calculate_size_offsets(tx_size, rx_size, &cmd_queue_offset,
					&resp_queue_offset, &inject_pool_offset,
					&sar_pool_offset, &peer_data_offset,
					&name_offset, &sock_name_offset);

	fd = shm_open(attr->name, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
	if (fd < 0) {
		if (errno != EEXIST) {
			FI_WARN(prov, FI_LOG_EP_CTRL,
				"shm_open error (%s): %s\n",
				attr->name, strerror(errno));
			return -errno;
		}

		ret = sm2_retry_map(attr->name, &fd);
		if (ret) {
			FI_WARN(prov, FI_LOG_EP_CTRL, "shm file in use (%s)\n",
				attr->name);
			return ret;
		}
		FI_WARN(prov, FI_LOG_EP_CTRL,
			"Overwriting shm from dead process (%s)\n", attr->name);
	}

	ep_name = calloc(1, sizeof(*ep_name));
	if (!ep_name) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "calloc error\n");
		ret = -FI_ENOMEM;
		goto close;
	}
	strncpy(ep_name->name, (char *)attr->name, SM2_NAME_MAX - 1);
	ep_name->name[SM2_NAME_MAX - 1] = '\0';

	pthread_mutex_lock(&sm2_ep_list_lock);
	dlist_insert_tail(&ep_name->entry, &sm2_ep_name_list);

	ret = ftruncate(fd, total_size);
	if (ret < 0) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "ftruncate error\n");
		ret = -errno;
		goto remove;
	}

	mapped_addr = mmap(NULL, total_size, PROT_READ | PROT_WRITE,
			   MAP_SHARED, fd, 0);
	if (mapped_addr == MAP_FAILED) {
		FI_WARN(prov, FI_LOG_EP_CTRL, "mmap error\n");
		ret = -errno;
		goto remove;
	}

	close(fd);

	if (attr->flags & SM2_FLAG_HMEM_ENABLED) {
		ret = ofi_hmem_host_register(mapped_addr, total_size);
		if (ret)
			FI_WARN(prov, FI_LOG_EP_CTRL,
				"unable to register shm with iface\n");
	}

	ep_name->region = mapped_addr;
	pthread_mutex_unlock(&sm2_ep_list_lock);

	*smr = mapped_addr;
	sm2_lock_init(&(*smr)->lock);
	ofi_atomic_initialize32(&(*smr)->signal, 0);

	(*smr)->map = map;
	(*smr)->version = SM2_VERSION;

	(*smr)->flags = attr->flags;
#ifdef HAVE_ATOMICS
	(*smr)->flags |= SM2_FLAG_ATOMIC;
#endif
#if ENABLE_DEBUG
	(*smr)->flags |= SM2_FLAG_DEBUG;
#endif

	(*smr)->base_addr = *smr;

	(*smr)->total_size = total_size;
	(*smr)->cmd_queue_offset = cmd_queue_offset;
	(*smr)->resp_queue_offset = resp_queue_offset;
	(*smr)->inject_pool_offset = inject_pool_offset;
	(*smr)->sar_pool_offset = sar_pool_offset;
	(*smr)->peer_data_offset = peer_data_offset;
	(*smr)->name_offset = name_offset;
	(*smr)->sock_name_offset = sock_name_offset;
	(*smr)->cmd_cnt = rx_size;
	/* Limit of 1 outstanding SAR message per peer */
	(*smr)->sar_cnt = SM2_MAX_PEERS;
	(*smr)->max_sar_buf_per_peer = SM2_BUF_BATCH_MAX;

	sm2_cmd_queue_init(sm2_cmd_queue(*smr), rx_size);
	sm2_resp_queue_init(sm2_resp_queue(*smr), tx_size);
	smr_freestack_init(sm2_inject_pool(*smr), rx_size,
			sizeof(struct sm2_inject_buf));
	smr_freestack_init(sm2_sar_pool(*smr), SM2_MAX_PEERS,
			sizeof(struct sm2_sar_buf));
	for (i = 0; i < SM2_MAX_PEERS; i++) {
		sm2_peer_addr_init(&sm2_peer_data(*smr)[i].addr);
		sm2_peer_data(*smr)[i].sar_status = 0;
		sm2_peer_data(*smr)[i].name_sent = 0;
	}

	strncpy((char *) sm2_name(*smr), attr->name, total_size - name_offset);

	/* Must be set last to signal full initialization to peers */
	(*smr)->pid = getpid();
	return 0;

remove:
	dlist_remove(&ep_name->entry);
	pthread_mutex_unlock(&sm2_ep_list_lock);
	free(ep_name);
close:
	close(fd);
	shm_unlink(attr->name);
	return ret;
}

void sm2_free(struct sm2_region *smr)
{
	if (smr->flags & SM2_FLAG_HMEM_ENABLED)
		(void) ofi_hmem_host_unregister(smr);
	shm_unlink(sm2_name(smr));
	munmap(smr, smr->total_size);
}

static int sm2_name_compare(struct ofi_rbmap *map, void *key, void *data)
{
	struct sm2_map *sm2_map;

	sm2_map = container_of(map, struct sm2_map, rbmap);

	return strncmp(sm2_map->peers[(uintptr_t) data].peer.name,
		       (char *) key, SM2_NAME_MAX);
}

int sm2_map_create(const struct fi_provider *prov, int peer_count,
		   uint16_t flags, struct sm2_map **map)
{
	int i;

	(*map) = calloc(1, sizeof(struct sm2_map));
	if (!*map) {
		FI_WARN(prov, FI_LOG_DOMAIN, "failed to create SHM region group\n");
		return -FI_ENOMEM;
	}

	for (i = 0; i < peer_count; i++) {
		sm2_peer_addr_init(&(*map)->peers[i].peer);
		(*map)->peers[i].fiaddr = FI_ADDR_UNSPEC;
	}
	(*map)->flags = flags;

	ofi_rbmap_init(&(*map)->rbmap, sm2_name_compare);
	ofi_spin_init(&(*map)->lock);

	return 0;
}

static int sm2_match_name(struct dlist_entry *item, const void *args)
{
	return !strcmp(container_of(item, struct sm2_ep_name, entry)->name,
		       (char *) args);
}

int sm2_map_to_region(const struct fi_provider *prov, struct sm2_map *map,
		      int64_t id)
{
	struct sm2_peer *peer_buf = &map->peers[id];
	struct sm2_region *peer;
	size_t size;
	int fd, ret = 0;
	struct stat sts;
	struct dlist_entry *entry;
	const char *name = sm2_no_prefix(peer_buf->peer.name);
	char tmp[SM2_PATH_MAX];

	if (peer_buf->region)
		return FI_SUCCESS;

	pthread_mutex_lock(&sm2_ep_list_lock);
	entry = dlist_find_first_match(&sm2_ep_name_list, sm2_match_name, name);
	if (entry) {
		peer_buf->region = container_of(entry, struct sm2_ep_name,
						entry)->region;
		pthread_mutex_unlock(&sm2_ep_list_lock);
		return FI_SUCCESS;
	}
	pthread_mutex_unlock(&sm2_ep_list_lock);

	fd = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);
	if (fd < 0) {
		FI_WARN_ONCE(prov, FI_LOG_AV, "shm_open error\n");
		return -errno;
	}

	memset(tmp, 0, sizeof(tmp));
	snprintf(tmp, sizeof(tmp), "%s%s", SM2_DIR, name);
	if (stat(tmp, &sts) == -1) {
		ret = -errno;
		goto out;
	}

	if (sts.st_size < sizeof(*peer)) {
		ret = -FI_ENOENT;
		goto out;
	}

	peer = mmap(NULL, sizeof(*peer), PROT_READ | PROT_WRITE,
		    MAP_SHARED, fd, 0);
	if (peer == MAP_FAILED) {
		FI_WARN(prov, FI_LOG_AV, "mmap error\n");
		ret = -errno;
		goto out;
	}

	if (!peer->pid) {
		FI_WARN(prov, FI_LOG_AV, "peer not initialized\n");
		munmap(peer, sizeof(*peer));
		ret = -FI_ENOENT;
		goto out;
	}

	size = peer->total_size;
	munmap(peer, sizeof(*peer));

	peer = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	peer_buf->region = peer;

	if (map->flags & SM2_FLAG_HMEM_ENABLED) {
		ret = ofi_hmem_host_register(peer, peer->total_size);
		if (ret)
			FI_WARN(prov, FI_LOG_EP_CTRL,
				"unable to register shm with iface\n");
	}

out:
	close(fd);
	return ret;
}

void sm2_map_to_endpoint(struct sm2_region *region, int64_t id)
{
	struct sm2_peer_data *local_peers;

	if (region->map->peers[id].peer.id < 0)
		return;

	local_peers = sm2_peer_data(region);

	strncpy(local_peers[id].addr.name,
		region->map->peers[id].peer.name, SM2_NAME_MAX - 1);
	local_peers[id].addr.name[SM2_NAME_MAX - 1] = '\0';
}

void sm2_unmap_from_endpoint(struct sm2_region *region, int64_t id)
{
	struct sm2_region *peer_smr;
	struct sm2_peer_data *local_peers, *peer_peers;
	int64_t peer_id;

	local_peers = sm2_peer_data(region);

	memset(local_peers[id].addr.name, 0, SM2_NAME_MAX);
	peer_id = region->map->peers[id].peer.id;
	if (peer_id < 0)
		return;

	peer_smr = sm2_peer_region(region, id);
	peer_peers = sm2_peer_data(peer_smr);

	peer_peers[peer_id].addr.id = -1;
	peer_peers[peer_id].name_sent = 0;
}

void sm2_exchange_all_peers(struct sm2_region *region)
{
	int64_t i;
	for (i = 0; i < SM2_MAX_PEERS; i++)
		sm2_map_to_endpoint(region, i);
}

int sm2_map_add(const struct fi_provider *prov, struct sm2_map *map,
		const char *name, int64_t *id)
{
	struct ofi_rbnode *node;
	int tries = 0, ret = 0;

	ofi_spin_lock(&map->lock);
	ret = ofi_rbmap_insert(&map->rbmap, (void *) name,
			       (void *) (intptr_t) *id, &node);
	if (ret) {
		assert(ret == -FI_EALREADY);
		*id = (intptr_t) node->data;
		ofi_spin_unlock(&map->lock);
		return 0;
	}

	while (map->peers[map->cur_id].peer.id != -1 &&
	       tries < SM2_MAX_PEERS) {
		if (++map->cur_id == SM2_MAX_PEERS)
			map->cur_id = 0;
		tries++;
	}

	assert(map->cur_id < SM2_MAX_PEERS && tries < SM2_MAX_PEERS);
	*id = map->cur_id;
	node->data = (void *) (intptr_t) *id;
	strncpy(map->peers[*id].peer.name, name, SM2_NAME_MAX);
	map->peers[*id].peer.name[SM2_NAME_MAX - 1] = '\0';
	map->peers[*id].region = NULL;

	ret = sm2_map_to_region(prov, map, *id);
	if (!ret)
		map->peers[*id].peer.id = *id;

	map->num_peers++;
	ofi_spin_unlock(&map->lock);
	return ret == -ENOENT ? 0 : ret;
}

void sm2_map_del(struct sm2_map *map, int64_t id)
{
	struct dlist_entry *entry;

	if (id >= SM2_MAX_PEERS || id < 0 || map->peers[id].peer.id < 0)
		return;

	pthread_mutex_lock(&sm2_ep_list_lock);
	entry = dlist_find_first_match(&sm2_ep_name_list, sm2_match_name,
				       sm2_no_prefix(map->peers[id].peer.name));
	pthread_mutex_unlock(&sm2_ep_list_lock);

	ofi_spin_lock(&map->lock);
	if (!entry) {
		if (map->flags & SM2_FLAG_HMEM_ENABLED)
			(void) ofi_hmem_host_unregister(map->peers[id].region);
		munmap(map->peers[id].region, map->peers[id].region->total_size);
	}

	(void) ofi_rbmap_find_delete(&map->rbmap,
				     (void *) map->peers[id].peer.name);

	map->peers[id].fiaddr = FI_ADDR_UNSPEC;
	map->peers[id].peer.id = -1;
	map->num_peers--;

	ofi_spin_unlock(&map->lock);
}

void sm2_map_free(struct sm2_map *map)
{
	int64_t i;

	for (i = 0; i < SM2_MAX_PEERS; i++)
		sm2_map_del(map, i);

	ofi_rbmap_cleanup(&map->rbmap);
	free(map);
}

struct sm2_region *sm2_map_get(struct sm2_map *map, int64_t id)
{
	if (id < 0 || id >= SM2_MAX_PEERS)
		return NULL;

	return map->peers[id].region;
}
