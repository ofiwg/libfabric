#ifndef HOOK_PROV_H
#define HOOK_PROV_H

#include <ofi.h>
#include "ofi_hook.h"

int hook_bind(struct fid *fid, struct fid *bfid, uint64_t flags);
int hook_control(struct fid *fid, int command, void *arg);
int hook_ops_open(struct fid *fid, const char *name,
			 uint64_t flags, void **ops, void *context);
int hook_close(struct fid *fid);

#endif /* HOOK_PROV_H */
