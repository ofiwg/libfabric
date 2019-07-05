#include <elf.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <ofi_mr_hooks.h>
#include <stdio.h>

/*
	Here is the main mechanism of patcher applying
*/
 typedef struct ofi_patcher_linux_dl_iter_context {
	ofi_patcher_linux_patch_t *patch;
	bool remove;
	int status;
} ofi_patcher_linux_dl_iter_context_t;

static void *(*orig_dlopen) (const char *, int);
static void *ofi_patcher_linux_dlopen(const char *filename, int flag);
static int ofi_patcher_linux_apply_patch (struct ofi_patcher_linux_patch_t *patch);

 ofi_patcher_base_module_t ofi_patcher_linux_module = {
	.patch_init = ofi_patcher_linux_init,
	.patch_symbol = ofi_patcher_linux_patch_symbol,
};

static const ElfW(Phdr) *
ofi_patcher_linux_get_phdr_dynamic(const ElfW(Phdr) *phdr,
				   uint16_t phnum, int phent)
{
	for (uint16_t i = 0 ; i < phnum ; ++i, phdr = (ElfW(Phdr)*)((intptr_t) phdr + phent)) {
		if (phdr->p_type == PT_DYNAMIC) {
			return phdr;
		}
	}

	return NULL;
}

static void *ofi_patcher_linux_dlopen(const char *filename, int flag)
{
	void *handle;
	ofi_patcher_linux_patch_t  *patch;
	handle = orig_dlopen(filename, flag);
	struct dlist_entry *item;

	if (handle != NULL) {
		fastlock_acquire(&ofi_patcher_linux_module.patch_list_mutex);

		dlist_foreach(&ofi_patcher_linux_module.patch_list, item) {
			patch = container_of(item, ofi_patcher_linux_patch_t, super.super);
			if (!patch->super.patch_data_size) {

				ofi_patcher_linux_apply_patch (patch);
			}
		}
		fastlock_release(&ofi_patcher_linux_module.patch_list_mutex);
	}

	return handle;
}

static void *ofi_patcher_linux_get_dynentry(ElfW(Addr) base,
					    const ElfW(Phdr) *pdyn, ElfW(Sxword) type)
{
	for (ElfW(Dyn) *dyn = (ElfW(Dyn)*)(base + pdyn->p_vaddr); dyn->d_tag; ++dyn) {
		if (dyn->d_tag == type) {
			return (void *) (uintptr_t) dyn->d_un.d_val;
		}
	}

	return NULL;
}

static void * ofi_patcher_linux_get_got_entry (ElfW(Addr) base, const ElfW(Phdr) *phdr, int16_t phnum,
                                               int phent, const char *symbol)
{
	const ElfW(Phdr) *dphdr;
	void *jmprel, *strtab;
	ElfW(Sym)  *symtab;
	size_t pltrelsz;

	dphdr = ofi_patcher_linux_get_phdr_dynamic (phdr, phnum, phent);

	jmprel = ofi_patcher_linux_get_dynentry (base, dphdr, DT_JMPREL);
	symtab = (ElfW(Sym) *) ofi_patcher_linux_get_dynentry (base, dphdr, DT_SYMTAB);
	strtab = ofi_patcher_linux_get_dynentry (base, dphdr, DT_STRTAB);
	pltrelsz = (size_t) (uintptr_t) ofi_patcher_linux_get_dynentry (base, dphdr, DT_PLTRELSZ);

	for (ElfW(Rela) *reloc = jmprel; (intptr_t) reloc < (intptr_t) jmprel + pltrelsz; ++reloc) {
#if SIZEOF_VOID_P == 8
	uint32_t relsymidx = ELF64_R_SYM(reloc->r_info);
#else
	uint32_t relsymidx = ELF32_R_SYM(reloc->r_info);
#endif
	char *elf_sym = (char *) strtab + symtab[relsymidx].st_name;

		if (0 == strcmp (symbol, elf_sym)) {
			return (void *)(base + reloc->r_offset);
		}
	}

	return NULL;
}

int ofi_getpagesize(void)
{
	static int page_size = -1;

	if (page_size != -1) {
		/* testing in a loop showed sysconf() took ~5
		 usec vs ~0.3 usec with it cached*/
		return page_size;
	}

#ifdef HAVE_GETPAGESIZE
	return page_size = getpagesize();
#elif defined(_SC_PAGESIZE )
	return page_size = sysconf(_SC_PAGESIZE);
#elif defined(_SC_PAGE_SIZE)
	return page_size = sysconf(_SC_PAGE_SIZE);
#else
	return page_size = 65536; /* safer to overestimate than under */
#endif
}

static int
ofi_patcher_linux_modify_got (ElfW(Addr) base, const ElfW(Phdr) *phdr,
			      const char *phname, int16_t phnum, int phent,
			      ofi_patcher_linux_dl_iter_context_t *ctx)
{
	long page_size = ofi_getpagesize();
	void **entry, *page;
	int ret;
	dlist_init(&ctx->patch->patch_got_list);

	entry = ofi_patcher_linux_get_got_entry (base, phdr, phnum, phent,
						 ctx->patch->super.patch_symbol);
	if (entry == NULL) {
		return FI_SUCCESS;
	}

	page = (void *)((intptr_t)entry & ~(page_size - 1));
	ret = mprotect(page, page_size, PROT_READ|PROT_WRITE);
	if (ret < 0) {
		/* FI_DBG(&core_prov, FI_LOG_MR, "failed to modify GOT page");*/

		return -FI_EOPNOTSUPP;
	}
	struct dlist_entry *tmp;
	struct dlist_entry *item;
	if (!ctx->remove) {
		if (*entry != (void *) ctx->patch->super.patch_value) {
			ofi_patcher_linux_patch_got_t *patch_got = (ofi_patcher_linux_patch_got_t *)
								   malloc(sizeof(ofi_patcher_linux_patch_got_t));
			if (NULL == patch_got) {
				return -FI_EOVERFLOW;
			}

		 
			/*	FI_DBG(&core_prov, FI_LOG_MR, "patch %p (%s): modifying 
			got entry %p. original value %p. new value %p\n");*/

			patch_got->got_entry = entry;
			patch_got->got_orig = *entry;

			dlist_insert_tail(&patch_got->super, &ctx->patch->patch_got_list);
			*entry = (void *)ctx->patch->super.patch_value;
		}
	} else {
		ofi_patcher_linux_patch_got_t *patch_got;
		dlist_foreach_safe(&ctx->patch->patch_got_list,item, tmp) {
			patch_got = container_of(item, ofi_patcher_linux_patch_got_t, super);
			if (patch_got->got_entry == entry) {

				/* FI_DBG(&core_prov, FI_LOG_MR, "patch %p (%s): modifying 
				got entry %p. original value %p. new value %p\n");*/

				if (*entry == (void *) ctx->patch->super.patch_value) {
						*entry = patch_got->got_orig;
				}
				dlist_remove(&patch_got->super);
				free(patch_got);
				break;
			}
		}
	}
	return FI_SUCCESS;
}

static int ofi_patcher_linux_get_aux_phent (void)
{
	return getauxval(AT_PHENT);
}

static int ofi_patcher_linux_phdr_iterator(struct dl_phdr_info *info,
					   size_t size, void *data)
{
	ofi_patcher_linux_dl_iter_context_t *ctx = data;
	int phent;

	phent = ofi_patcher_linux_get_aux_phent();
	if (phent <= 0) {
		/* FI_DBG(&core_prov, FI_LOG_MR, "Iteration failed");*/
		ctx->status = -FI_EOPNOTSUPP;
		return -1;
	}

	ctx->status = ofi_patcher_linux_modify_got (info->dlpi_addr, info->dlpi_phdr,
						info->dlpi_name, info->dlpi_phnum,
						phent, ctx);
	if (ctx->status == FI_SUCCESS) {
		return 0; /* continue iteration and patch all objects */
	} else {
		return -1; /* stop iteration if got a real error */
	}
}

static int ofi_patcher_linux_apply_patch (ofi_patcher_linux_patch_t *patch)
{
	ofi_patcher_linux_dl_iter_context_t ctx = {
		.patch    = patch,
		.remove   = false,
		.status   = FI_SUCCESS,
	};

	(void) dl_iterate_phdr(ofi_patcher_linux_phdr_iterator, &ctx);

	return ctx.status;
}

static int ofi_patcher_linux_remove_patch (ofi_patcher_linux_patch_t *patch)
{
	ofi_patcher_linux_dl_iter_context_t ctx = {
		.patch    = patch,
		.remove   = true,
		.status   = FI_SUCCESS,
	};

	/* Avoid locks here because we don't modify ELF data structures.
	* Worst case the same symbol will be written more than once.
	*/
	(void) dl_iterate_phdr(ofi_patcher_linux_phdr_iterator, &ctx);

	return ctx.status;
}

static inline int ofi_patcher_linux_get_orig(const char *symbol, void *replacement)
{
	const char *error;
	void *func_ptr;

	func_ptr = dlsym(RTLD_DEFAULT, symbol);
	if (func_ptr == replacement) {
		(void)dlerror();
		func_ptr = dlsym(RTLD_NEXT, symbol);
		if (func_ptr == NULL) {
			error = dlerror();
			/* FI_DBG(&core_prov, FI_LOG_MR, 
			"get orig failed: %s\n", error); */
		}
	}

	return FI_SUCCESS;
}

static inline uintptr_t ofi_patcher_base_addr_text (uintptr_t addr) {
	return addr;
}

int ofi_patcher_linux_patch_symbol(const char *symbol_name, 
				   uintptr_t replacement, uintptr_t *orig)
{
	int ret;
	ofi_patcher_linux_patch_t* patch = (ofi_patcher_linux_patch_t *)
					   malloc(sizeof(ofi_patcher_linux_patch_t));
	if (NULL == patch) {
		return -FI_EOVERFLOW;
	}

	patch->super.patch_symbol = strdup (symbol_name);
	if (NULL == patch->super.patch_symbol) {
		free(patch);
		return -FI_EOVERFLOW;
	}
	
	patch->super.patch_value = ofi_patcher_base_addr_text (replacement);
	patch->super.patch_restore = (ofi_patcher_base_restore_fn_t) ofi_patcher_linux_remove_patch;

	fastlock_acquire(&ofi_patcher_linux_module.patch_list_mutex);
	do {
	
		ret = ofi_patcher_linux_apply_patch(patch);
		if (FI_SUCCESS != ret) {
			free(patch);
			break;
		}
		*orig = ofi_patcher_linux_get_orig(patch->super.patch_symbol ,
						   (void *) replacement);
		if (FI_SUCCESS != ret) {
			free(patch);
			break;
		}
		dlist_insert_tail(&ofi_patcher_linux_module.patch_list,
				  &patch->super.super);
	} while(0);

	fastlock_release(&ofi_patcher_linux_module.patch_list_mutex);

	return ret;
}

static int ofi_patcher_linux_install_dlopen (void)
{
	return ofi_patcher_linux_patch_symbol ("dlopen", 
					      (uintptr_t) ofi_patcher_linux_dlopen,
                                              (uintptr_t *) &orig_dlopen);
}

int ofi_patcher_linux_init (void)
{
	return ofi_patcher_linux_install_dlopen ();
}

/*SYS_MMAP, SYS_MUNMAP and etc. handling,
*the main patcher initilization
*/

#define memory_patcher_syscall __syscall

#if defined (SYS_mmap)

static void *(*original_mmap)(void *, size_t, int, int, int, off_t);

static void *_intercept_mmap(void *start, size_t length,
			     int prot, int flags, int fd, off_t offset)
{
	void *result = 0;
	result = original_mmap (start, length, prot, flags, fd, offset);
	return result;
}

static void *intercept_mmap(void *start, size_t
			    length, int prot, int flags, int fd, off_t offset)
{
	void *result = _intercept_mmap (start, length, prot, flags, fd, offset);
	return result;
}
#endif

int patcher_open (void)
{	
	int ret;

	struct dlist_entry patch_list = patch_list;
	fastlock_t patch_list_mutex = patch_list_mutex;

	dlist_init(&patch_list);
	fastlock_init(&patch_list_mutex);

	ret = ofi_patcher_linux_module.patch_init ();
	if (FI_SUCCESS != ret) {
		return ret;
	}

#if defined (SYS_mmap)
	rc = ofi_patcher_linux_module.patch_symbol ("mmap", (uintptr_t) intercept_mmap,
						   (uintptr_t *) &original_mmap);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif

	return ret;
}
