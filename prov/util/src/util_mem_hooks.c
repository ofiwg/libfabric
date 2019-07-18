/* -*- Mode: C; c-basic-offset:8 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2015.  ALL RIGHTS RESERVED.
 * Copyright (c) 2016      Los Alamos National Security, LLC. All rights
 *                         reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
/*
 * Copied from OpenUCX
 */

/*#if HAVE_PATCH_UNMAP*/
#include <elf.h>
#include <sys/auxv.h>
#include <sys/mman.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <ofi_mem_hooks.h>
#include <stdio.h>

/*
	Here is the main mechanism of patcher applying
*/


static void *(*orig_dlopen) (const char *, int);
static void *ofi_patcher_dlopen(const char *filename, int flag);
static int ofi_patcher_apply_patch (struct ofi_patcher_patch *patch);

struct ofi_patcher_base_module	ofi_patcher_module;
struct ofi_patcher_dl_iter_context ofi_patcher_dl_iter_context;

static const ElfW(Phdr) *
ofi_patcher_get_phdr_dynamic(const ElfW(Phdr) *phdr,
				   uint16_t phnum, int phent)
{
	for (uint16_t i = 0 ; i < phnum ; ++i, phdr = (ElfW(Phdr)*)((intptr_t) phdr + phent)) {
		if (phdr->p_type == PT_DYNAMIC) {
			return phdr;
		}
	}

	return NULL;
}

static void *ofi_patcher_dlopen(const char *filename, int flag)
{
	void *handle;
	struct ofi_patcher_patch  *patch;
	handle = orig_dlopen(filename, flag);
	struct dlist_entry *item;

	if (handle != NULL) {
		fastlock_acquire(&ofi_patcher_module.patch_list_mutex);

		dlist_foreach(&ofi_patcher_module.patch_list, item) {
			patch = container_of(item, struct ofi_patcher_patch, super.super);
			if (!patch->super.patch_data_size) {

				ofi_patcher_apply_patch (patch);
			}
		}
		fastlock_release(&ofi_patcher_module.patch_list_mutex);
	}

	return handle;
}

static void *ofi_patcher_get_dynentry(ElfW(Addr) base,
					    const ElfW(Phdr) *pdyn, ElfW(Sxword) type)
{
	for (ElfW(Dyn) *dyn = (ElfW(Dyn)*)(base + pdyn->p_vaddr); dyn->d_tag; ++dyn) {
		if (dyn->d_tag == type) {
			return (void *) (uintptr_t) dyn->d_un.d_val;
		}
	}

	return NULL;
}

static void * ofi_patcher_get_got_entry (ElfW(Addr) base, const ElfW(Phdr) *phdr, int16_t phnum,
                                               int phent, const char *symbol)
{
	const ElfW(Phdr) *dphdr;
	void *jmprel, *strtab;
	uint32_t relsymidx;
	ElfW(Sym)  *symtab;
	size_t pltrelsz;

	dphdr = ofi_patcher_get_phdr_dynamic (phdr, phnum, phent);

	jmprel = ofi_patcher_get_dynentry (base, dphdr, DT_JMPREL);
	symtab = (ElfW(Sym) *) ofi_patcher_get_dynentry (base, dphdr, DT_SYMTAB);
	strtab = ofi_patcher_get_dynentry (base, dphdr, DT_STRTAB);
	pltrelsz = (size_t) (uintptr_t) ofi_patcher_get_dynentry (base, dphdr, DT_PLTRELSZ);

	for (ElfW(Rela) *reloc = jmprel; (intptr_t) reloc < (intptr_t) jmprel + pltrelsz; ++reloc) {
		if (sizeof(void*) == 8)
			relsymidx = ELF64_R_SYM(reloc->r_info);
		else
			relsymidx = ELF32_R_SYM(reloc->r_info);

		char *elf_sym = (char *) strtab + symtab[relsymidx].st_name;

			if (0 == strcmp (symbol, elf_sym)) {
				return (void *)(base + reloc->r_offset);
			}
	}

	return NULL;
}

static int
ofi_patcher_modify_got (ElfW(Addr) base, const ElfW(Phdr) *phdr,
			      const char *phname, int16_t phnum, int phent,
			      struct ofi_patcher_dl_iter_context *ctx)
{
	long page_size = ofi_get_page_size();
	void **entry, *page;
	int ret;
/*	dlist_init(&ctx->patch->patch_got_list);*/
	struct dlist_entry *tmp;
	struct dlist_entry *item;

	entry = ofi_patcher_get_got_entry (base, phdr, phnum, phent,
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

	if (!ctx->remove) {
		if (*entry != (void *) ctx->patch->super.patch_value) {
			struct ofi_patcher_patch_got *patch_got = (struct ofi_patcher_patch_got *)
								   malloc(sizeof(struct ofi_patcher_patch_got));
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
		struct ofi_patcher_patch_got *patch_got;
		dlist_foreach_safe(&ctx->patch->patch_got_list,item, tmp) {
			patch_got = container_of(item, struct ofi_patcher_patch_got, super);
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

static int ofi_patcher_phdr_iterator(struct dl_phdr_info *info,
					   size_t size, void *data)
{
	struct ofi_patcher_dl_iter_context *ctx = data;
	int phent;

	phent = getauxval(AT_PHENT);
	if (phent <= 0) {
		/* FI_DBG(&core_prov, FI_LOG_MR, "Iteration failed");*/
		ctx->status = -FI_EOPNOTSUPP;
		return -1;
	}

	ctx->status = ofi_patcher_modify_got (info->dlpi_addr, info->dlpi_phdr,
						info->dlpi_name, info->dlpi_phnum,
						phent, ctx);
	if (ctx->status == FI_SUCCESS) {
		return 0; /* continue iteration and patch all objects */
	} else {
		return -1; /* stop iteration if got a real error */
	}
}

static int ofi_patcher_apply_patch(struct ofi_patcher_patch *patch)
{
	struct ofi_patcher_dl_iter_context ctx = {
		.patch    = patch,
		.remove   = false,
		.status   = FI_SUCCESS,
	};

	(void) dl_iterate_phdr(ofi_patcher_phdr_iterator, &ctx);

	return ctx.status;
}

static int ofi_patcher_remove_patch(struct ofi_patcher_patch *patch)
{
	struct ofi_patcher_dl_iter_context ctx = {
		.patch    = patch,
		.remove   = true,
		.status   = FI_SUCCESS,
	};

	/* Avoid locks here because we don't modify ELF data structures.
	* Worst case the same symbol will be written more than once.
	*/
	(void) dl_iterate_phdr(ofi_patcher_phdr_iterator, &ctx);

	return ctx.status;
}

static inline int ofi_patcher_get_orig(const char *symbol, void *replacement)
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
			return -FI_EOVERFLOW;
		}
	}

	return FI_SUCCESS;
}

int ofi_patcher_patch_symbol(const char *symbol_name, 
			     uintptr_t replacement, uintptr_t *orig)
{
	int ret;
	struct ofi_patcher_patch* patch = (struct ofi_patcher_patch *)
					   malloc(sizeof(struct ofi_patcher_patch));
	if (NULL == patch) {
		return -FI_EOVERFLOW;
	}
	
	patch->super.patch_symbol = strdup (symbol_name);
	if (NULL == patch->super.patch_symbol) {
		free(patch);
		return -FI_EOVERFLOW;
	}
	
	patch->super.patch_value = replacement;
	patch->super.patch_restore = (void*) ofi_patcher_remove_patch;

	/* Take lock first to handle a possible race where dlopen() is called
	 * from another thread and we may end up not patching it.
	 */
	fastlock_acquire(&ofi_patcher_module.patch_list_mutex);
	
	ret = ofi_patcher_apply_patch(patch);
	if (FI_SUCCESS != ret) {
		goto unlock;
	}

	*orig = ofi_patcher_get_orig(patch->super.patch_symbol ,
				    (void *) replacement);
	if (FI_SUCCESS != ret) {
		goto unlock;
	}
	dlist_insert_tail(&ofi_patcher_module.patch_list,
			  &patch->super.super);


unlock:
	free(patch);
	fastlock_release(&ofi_patcher_module.patch_list_mutex);

	return ret;
}

/*SYS_MMAP, SYS_MUNMAP and etc. handling,
*the main patcher initilization
*/

/* calling __syscall is preferred on some systems when some arguments may be 64-bit. it also
 * has the benefit of having an off_t return type */
//#define memory_patcher_syscall __syscall
//#else
#define memory_patcher_syscall syscall
//#endif

/*SYS_MMAP*/
#if defined (SYS_mmap)

static void *(*original_mmap)(void *, size_t, int, int, int, off_t);

static void *_intercept_mmap(void *start, size_t length,
			     int prot, int flags, int fd, off_t offset)
{
	void *result = 0;

	/*not implemented yet*/
	/*if ((flags & MAP_FIXED) && (start != NULL)) {
		opal_mem_hooks_release_hook (start, length, true);
	}*/
	if (!original_mmap) {
		result = (void*)(intptr_t) memory_patcher_syscall(SYS_mmap, start, length,
								  prot, flags, fd, offset);
	} else {
			result = original_mmap (start, length, prot, flags, fd, offset);
	}

	return result;
}

static void *intercept_mmap(void *start, size_t
			    length, int prot, int flags, int fd, off_t offset)
{
	void *result = _intercept_mmap (start, length, prot, flags, fd, offset);

	return result;
}
#endif

/*SYS_MUNMAP*/
#if defined (SYS_munmap)

static int (*original_munmap) (void *, size_t);

static int _intercept_munmap(void *start, size_t length)
{
	int result = 0;
	/* could be in a malloc implementation */
	/* opal_mem_hooks_release_hook (start, length, true);*/

	if (!original_munmap) {
		result = memory_patcher_syscall(SYS_munmap, start, length);
	} else {
		result = original_munmap(start, length);
	}

	return result;
}

static int intercept_munmap(void *start, size_t length)
{
	int result = _intercept_munmap (start, length);

	return result;
}

#endif

/*SYS_MREMAP*/
#if defined (SYS_mremap)

#if defined(__linux__)
/* on linux this function has an optional extra argument but ... can not be used here because it
 * causes issues when intercepting a 4-argument mremap call */
static void *(*original_mremap) (void *, size_t, size_t, int, void *);
#else
/* mremap has a different signature on BSD systems */
static void *(*original_mremap) (void *, size_t, void *, size_t, int);
#endif

#if defined(__linux__)
static void *_intercept_mremap (void *start, size_t oldlen, size_t newlen, int flags, void *new_address)
#else
static void *_intercept_mremap (void *start, size_t oldlen, void *new_address, size_t newlen, int flags)
#endif
{
	void *result = 0;
#if defined(__linux__)
        result = original_mremap (start, oldlen, newlen, flags, new_address);
#else
        result = original_mremap (start, oldlen, new_address, newlen, flags);
#endif
        return result;
}

#if defined(__linux__)
static void *intercept_mremap (void *start, size_t oldlen, size_t newlen, int flags, void *new_address)
{
    void *result = _intercept_mremap (start, oldlen, newlen, flags, new_address);
    return result;
}
#else
static void *intercept_mremap (void *start, size_t oldlen, void *new_address, size_t newlen, int flags)
{
    void *result = _intercept_mremap (start, oldlen, new_address, newlen, flags);
    return result;
}
#endif

#endif

/*SYS_MADVISE*/
#if defined (SYS_madvise)

static int (*original_madvise) (void *, size_t, int);

static int _intercept_madvise (void *start, size_t length, int advice)
{
	int result = 0;
	result = original_madvise (start, length, advice);
	return result;
}

static int intercept_madvise (void *start, size_t length, int advice)
{
	int result = _intercept_madvise (start, length, advice);
	return result;
}

#endif

/*HAS_SHMAT*/
#if HAS_SHMAT && defined(__linux__)
static void *(*original_shmat)(int shmid, const void *shmaddr, int shmflg);

static void *_intercept_shmat(int shmid, const void *shmaddr, int shmflg)
{
	void *result = 0;
        result = original_shmat (shmid, shmaddr, shmflg);
	return result;
}

static void* intercept_shmat (int shmid, const void * shmaddr, int shmflg)
{

	void *result = _intercept_shmat (shmid, shmaddr, shmflg);
	return result;
}
#endif

/*HAS_SHMDT*/
#if HAS_SHMDT && defined(__linux__)
static int (*original_shmdt) (const void *);

static int _intercept_shmdt (const void *shmaddr)
{
	int result;
        result = original_shmdt (shmaddr);
	return result;
}

static int intercept_shmdt (const void *shmaddr)
{
	int result = _intercept_shmdt (shmaddr);
	return result;
}
#endif

#if defined SYS_brk

#ifdef HAVE___CURBRK
extern void *__curbrk; /* in libc */
#endif

static int (*original_brk) (void *);

static int _intercept_brk (void *addr)
{
	int result = 0;
        result = original_brk (addr);
	return result;
}

static int intercept_brk (void *addr)
{
    
	int result = _intercept_brk (addr);
	return result;
}
#endif

int ofi_patcher_handler(void)
{	
	int ret;

	/*we'd understand what we init here*/
	/*struct dlist_entry patch_list 	= patch_list;
	fastlock_t patch_list_mutex 	= patch_list_mutex;

	dlist_init(&patch_list);
	fastlock_init(&patch_list_mutex);*/

	ret = ofi_patcher_patch_symbol("dlopen", (uintptr_t) ofi_patcher_dlopen,
                                      (uintptr_t *) &orig_dlopen);
	if (FI_SUCCESS != ret) {
		return ret;
	}

/*#if defined (SYS_mmap)
	rc = ofi_patcher_patch_symbol("mmap", (uintptr_t) intercept_mmap,
				     (uintptr_t *) &original_mmap);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif

#if defined (SYS_munmap)
	ret = ofi_patcher_patch_symbol("munmap", (uintptr_t)intercept_munmap,
				      (uintptr_t *) &original_munmapp);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif


#if defined (SYS_mremap)
	ret = ofi_patcher_patch_symbol("mremap", (uintptr_t)intercept_mremap,
				      (uintptr_t *) &original_mremap);
	if (FI_SUCCESS != ret) {
		return ret;
	}

#endif

#if defined (SYS_madvise)
	ret = ofi_patcher_patch_symbol("madvise", (uintptr_t)intercept_madvise,
				      (uintptr_t *) &original_madvise);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif

#if HAS_SHMAT && defined(__linux__)
	ret = ofi_patcher_patch_symbol("shmat", (uintptr_t) intercept_shmat,
				      (uintptr_t *) &original_shmat);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif

#if HAS_SHMDT && defined(__linux__)
	ret = ofi_patcher_patch_symbol("shmdt", (uintptr_t) intercept_shmdt,
				      (uintptr_t *) &original_shmdt);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif

#if defined (SYS_brk)
	ret = ofi_patcher_patch_symbol("brk", (uintptr_t)intercept_brk,
				      (uintptr_t *) &original_brk);
	if (FI_SUCCESS != ret) {
		return ret;
	}
#endif*/

	return ret;
}
/*#end //HAVE_PATCH_UNMAP*/
