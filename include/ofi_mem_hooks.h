/* -*- Mode: C; c-basic-offset:8 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2016      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include <inttypes.h>
#include <stdbool.h>

#include <ofi.h>
#include <ofi_atom.h>
#include <ofi_lock.h>
#include <ofi_list.h>
#include <ofi_tree.h>

/*
 * HOOKS  memory mechanism structs
 */
#define OFI_BASE_PATCHER_MAX_PATCH 32

struct ofi_patcher_base_patch {
	struct dlist_entry super;
	char		*patch_symbol;
	uintptr_t	patch_value;
	uintptr_t	patch_orig;
	unsigned char	patch_data[OFI_BASE_PATCHER_MAX_PATCH];
	unsigned char	patch_orig_data[OFI_BASE_PATCHER_MAX_PATCH];
	unsigned	patch_data_size;
	void * (*patch_restore)(struct ofi_patcher_base_patch *);
};

struct ofi_patcher_base_module {
	struct dlist_entry			patch_list;
	fastlock_t				patch_list_mutex;
	int * (*patch_finit)(void);
	int * (*patch_symbol)(const char *func_symbol_name, uintptr_t func_new_addr,
                                                   	    uintptr_t *func_old_addr);
	int * (*patch_address)(uintptr_t func_addr, uintptr_t func_new_addr);
};

struct ofi_patcher_patch {
	struct ofi_patcher_base_patch	super;
	struct dlist_entry		patch_got_list;
};

struct ofi_patcher_patch_got {
	struct dlist_entry super;
	void **got_entry;
	void *got_orig;
};

int ofi_patcher_patch_symbol(const char *symbol_name,
			     uintptr_t replacement, uintptr_t *orig);
int patcher_open (void);
