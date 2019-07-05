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
#define SIZEOF_VOID_P				8
struct ofi_patcher_base_patch_t;

typedef void (*ofi_patcher_base_restore_fn_t) (struct ofi_patcher_base_patch_t *);

typedef struct ofi_patcher_base_patch_t {
	struct dlist_entry super;
	/** name symbol to patch */
	char		*patch_symbol;
	/** address of function to call instead */
	uintptr_t	patch_value;
	/** original address of function */
	uintptr_t	patch_orig;
	/** patch data */
	unsigned char	patch_data[OFI_BASE_PATCHER_MAX_PATCH];
	/** original data */
	unsigned char	patch_orig_data[OFI_BASE_PATCHER_MAX_PATCH];
	/** size of patch data */
	unsigned	patch_data_size;
	/** function to undo the patch */
	ofi_patcher_base_restore_fn_t	patch_restore;
} ofi_patcher_base_patch_t;

typedef int (*ofi_patcher_base_patch_symbol_fn_t)(const char *func_symbol_name, uintptr_t func_new_addr,
                                                   uintptr_t *func_old_addr);
typedef int (*ofi_patcher_base_patch_address_fn_t)(uintptr_t func_addr, uintptr_t func_new_addr);
typedef int (*ofi_patcher_base_init_fn_t) (void);
typedef int (*ofi_patcher_base_fini_fn_t) (void);

typedef struct ofi_patcher_base_module_t {
	/** list of patches */
	struct dlist_entry			patch_list;
	 /** lock for patch list */
	fastlock_t				patch_list_mutex;
	ofi_patcher_base_init_fn_t		patch_init;
	/** function to call when patcher is unloaded. this function
	  * MUST clean up all active patches. can be NULL. */
	ofi_patcher_base_fini_fn_t		patch_fini;
	/** hook a symbol. may be NULL */
	ofi_patcher_base_patch_symbol_fn_t	patch_symbol;
	/** hook a function pointer. may be NULL */
	ofi_patcher_base_patch_address_fn_t	patch_address;
} ofi_patcher_base_module_t;

struct ofi_patcher_linux_patch_t {
	ofi_patcher_base_patch_t	super;
	struct dlist_entry		patch_got_list;
};
typedef struct ofi_patcher_linux_patch_t ofi_patcher_linux_patch_t;

struct ofi_patcher_linux_patch_got_t {
	struct dlist_entry super;
	void **got_entry;
	void *got_orig;
};
typedef struct ofi_patcher_linux_patch_got_t ofi_patcher_linux_patch_got_t;

int ofi_patcher_linux_init (void);
int ofi_patcher_linux_patch_symbol(const char *symbol_name, uintptr_t replacement, uintptr_t *orig);
int patcher_open (void);