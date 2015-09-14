#ifndef _MLXM_MQ_STORAGE_H
#define _MLXM_MQ_STORAGE_H
#include "mlxm.h"

static inline
int mlxm_find_mq(struct mlxm_mq_storage *storage,
                 uint16_t id, mxm_mq_h *mq) {
        struct mlxm_mq_entry *mq_e = NULL;
        HASH_FIND(hh, storage->hash, &id, sizeof(uint16_t), mq_e);
        if (mq_e) {
                *mq = mq_e->mq;
                return 1;
        } else {
                return 0;
        }
};

static inline
int mlxm_mq_add_to_storage(struct mlxm_mq_storage *storage,
                           uint16_t id, mxm_mq_h *mq) {
        int mxm_err;
        struct mlxm_mq_entry *mq_entry;
        mq_entry = (struct mlxm_mq_entry*)malloc(sizeof(*mq_entry));
        mxm_err = mxm_mq_create(mlxm_globals.mxm_context,
                                id,
                                &mq_entry->mq);
        if (mxm_err) {
                FI_WARN(&mlxm_prov,FI_LOG_CORE,
                        "mxm_mq_create failed: mq_id %d, errno %d:%s\n",
                        id, mxm_err, mxm_error_string(mxm_err));
                return mlxm_errno(mxm_err);
        }
        FI_INFO(&mlxm_prov,FI_LOG_CORE,
                "MXM mq created, id 0x%x, %p\n",id , mq_entry->mq);

        mq_entry->mq_key = id;
        HASH_ADD(hh, storage->hash, mq_key, sizeof(uint16_t), mq_entry);
        *mq = mq_entry->mq;
        return 0;
};

static inline
void mlxm_mq_storage_init() {
        mlxm_globals.mq_storage.hash = NULL;
}

static inline
void mlxm_mq_storage_fini() {
        struct mlxm_mq_entry *mq_e, *tmp;
        HASH_ITER(hh, mlxm_globals.mq_storage.hash, mq_e, tmp) {
                mxm_mq_destroy(mq_e->mq);
                HASH_DEL(mlxm_globals.mq_storage.hash,
                         mq_e);
                free(mq_e);
        }
}

#endif
