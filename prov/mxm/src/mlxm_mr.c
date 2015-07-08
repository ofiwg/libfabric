#include "mlxm.h"
#include "fi_enosys.h"

static int mlxm_mr_reg(struct fid *domain, const void *buf, size_t len,
                       uint64_t access, uint64_t offset, uint64_t requested_key,
                       uint64_t flags, struct fid_mr **mr, void *context) {
        mlxm_fid_domain_t *domain_priv;
        mlxm_fid_mr_t *mr_priv = NULL;
	uint64_t key;
        int err;
        domain_priv = container_of(domain, mlxm_fid_domain_t, domain);

        mr_priv = (mlxm_fid_mr_t *) calloc(1, sizeof(*mr_priv) + sizeof(struct iovec));
	if (!mr_priv)
		return -ENOMEM;

	mr_priv->mr.fid.fclass = FI_CLASS_MR;
	mr_priv->mr.fid.context = context;
        // mr_priv->mr.fid.ops = &mlxm_fi_ops;
        mr_priv->mr.mem_desc = mr_priv;
        key = (uint64_t)(uintptr_t)mr_priv;
        mr_priv->mr.key = key;
	mr_priv->domain = domain_priv;
        mr_priv->iov_count = 1;
	mr_priv->iov[0].iov_base = (void *)buf;
	mr_priv->iov[0].iov_len = len;

        // err = mxm_mem_map(domain_priv->mxm_context, (void**)&buf, &len, 0, 0, 0);
        // if (MXM_OK != err) {
        //         fprintf(stderr,"Failed to register memory: %s", mxm_error_string(err));
        //         goto error_out;
        // }

        err = mxm_mem_get_key(mlxm_globals.mxm_context, (void*)buf,
                              &mr_priv->mxm_key);
        if (MXM_OK != err) {
                FI_WARN(&mlxm_prov,FI_LOG_MR,
                        "Failed to get memory key: %s", mxm_error_string(err));
                goto error_out;
        }


	*mr = &mr_priv->mr;

        return 0;
error_out:
        if (mr_priv)
                free(mr_priv);
        return FI_ENOKEY;

}


struct fi_ops_mr mlxm_mr_ops = {
	.size = sizeof(struct fi_ops_mr),
	.reg = mlxm_mr_reg,
        .regv = fi_no_mr_regv,
        .regattr = fi_no_mr_regattr,
};
