#[cfg(test)]
mod unit_tests {
    use ofi_libfabric_sys::bindgen::*;
    use paste;
    use std::ffi::CString;
    use std::ptr;

    /// Macro to generate dynamic linkage tests for Libfabric functions.
    ///
    /// This macro creates a test function which verifies that a provided function pointer
    /// (ex: fi_send) is not null, nor errors out with "ld: symbol(s) not found" upon its reference.
    macro_rules! test_function_linkage {
        ($func_name:ident) => {
            paste::paste! {
                #[test]
                fn [<test_ $func_name _linkage>]() {
                    let func_ptr = $func_name as *const ();
                    assert!(!func_ptr.is_null(),
                           concat!(stringify!($func_name), " function pointer should not be null."));
                }
            }
        };
        // Support for multiple functions at once.
        ($($func_name:ident),+ $(,)?) => {
            $(test_function_linkage!($func_name);)+
        };
    }

    /// Test core libfabric API function linkage, via the above macro.
    test_function_linkage!(
        get_fid_ptr,
        fi_getinfo,
        fi_freeinfo,
        fi_dupinfo,
        fi_param_get_str,
        fi_param_get_int,
        fi_param_get_bool,
        fi_param_get_size_t,
        fi_allocinfo,
        fi_close,
        fi_control,
        fi_alias,
        fi_get_val,
        fi_set_val,
        fi_open_ops,
        fi_set_ops,
        fi_passive_ep,
        fi_endpoint,
        fi_endpoint2,
        fi_scalable_ep,
        fi_ep_bind,
        fi_pep_bind,
        fi_scalable_ep_bind,
        fi_enable,
        fi_cancel,
        fi_setopt,
        fi_getopt,
        fi_ep_alias,
        fi_tx_context,
        fi_rx_context,
        fi_rx_size_left,
        fi_tx_size_left,
        fi_stx_context,
        fi_srx_context,
        fi_recv,
        fi_recvv,
        fi_recvmsg,
        fi_send,
        fi_sendv,
        fi_sendmsg,
        fi_inject,
        fi_senddata,
        fi_injectdata,
        fi_atomic,
        fi_atomicv,
        fi_atomicmsg,
        fi_inject_atomic,
        fi_fetch_atomic,
        fi_fetch_atomicv,
        fi_fetch_atomicmsg,
        fi_compare_atomic,
        fi_compare_atomicv,
        fi_compare_atomicmsg,
        fi_atomicvalid,
        fi_fetch_atomicvalid,
        fi_compare_atomicvalid,
        fi_query_atomic,
        fi_domain,
        fi_domain2,
        fi_domain_bind,
        fi_cq_open,
        fi_cntr_open,
        fi_wait_open,
        fi_poll_open,
        fi_mr_reg,
        fi_mr_regv,
        fi_mr_regattr,
        fi_mr_desc,
        fi_mr_key,
        fi_mr_raw_attr,
        fi_mr_map_raw,
        fi_mr_unmap_key,
        fi_mr_bind,
        fi_mr_refresh,
        fi_mr_enable,
        fi_av_open,
        fi_av_bind,
        fi_av_insert,
        fi_av_insertsvc,
        fi_av_insertsym,
        fi_av_remove,
        fi_av_lookup,
        fi_av_straddr,
        fi_av_insert_auth_key,
        fi_av_lookup_auth_key,
        fi_av_set_user_id,
        fi_rx_addr,
        fi_group_addr,
        fi_setname,
        fi_getname,
        fi_getpeer,
        fi_listen,
        fi_connect,
        fi_accept,
        fi_reject,
        fi_shutdown,
        fi_join,
        fi_mc_addr,
        fi_av_set,
        fi_av_set_union,
        fi_av_set_intersect,
        fi_av_set_diff,
        fi_av_set_insert,
        fi_av_set_remove,
        fi_av_set_addr,
        fi_join_collective,
        fi_barrier,
        fi_barrier2,
        fi_broadcast,
        fi_alltoall,
        fi_allreduce,
        fi_allgather,
        fi_reduce_scatter,
        fi_reduce,
        fi_scatter,
        fi_gather,
        fi_query_collective,
        fi_trywait,
        fi_wait,
        fi_poll,
        fi_poll_add,
        fi_poll_del,
        fi_eq_open,
        fi_eq_read,
        fi_eq_readerr,
        fi_eq_write,
        fi_eq_sread,
        fi_eq_strerror,
        fi_cq_read,
        fi_cq_readfrom,
        fi_cq_readerr,
        fi_cq_sread,
        fi_cq_sreadfrom,
        fi_cq_signal,
        fi_cq_strerror,
        fi_cntr_read,
        fi_cntr_readerr,
        fi_cntr_add,
        fi_cntr_adderr,
        fi_cntr_set,
        fi_cntr_seterr,
        fi_cntr_wait,
        fi_export_fid,
        fi_import_fid,
        fi_import,
        fi_import_log,
        fi_profile_reset,
        fi_profile_query_vars,
        fi_profile_query_events,
        fi_profile_read_u64,
        fi_profile_register_callback,
        fi_profile_start_reads,
        fi_profile_end_reads,
        fi_profile_open,
        fi_profile_close,
        fi_read,
        fi_readv,
        fi_readmsg,
        fi_write,
        fi_writev,
        fi_writemsg,
        fi_inject_write,
        fi_writedata,
        fi_inject_writedata,
        fi_trecv,
        fi_trecvv,
        fi_trecvmsg,
        fi_tsend,
        fi_tsendv,
        fi_tsendmsg,
        fi_tinject,
        fi_tsenddata,
        fi_tinjectdata,
    );

    /// Test successful instantiation of core Libfabric structures.
    #[test]
    fn test_struct_definitions() {
        // FI info.
        let _info: fi_info = unsafe { std::mem::zeroed() };

        // Control resources.
        let _fid_fabric: fid_fabric = unsafe { std::mem::zeroed() };
        let _fid_domain: fid_domain = unsafe { std::mem::zeroed() };
        let _fid_ep: fid_ep = unsafe { std::mem::zeroed() };
        let _fid_cq: fid_cq = unsafe { std::mem::zeroed() };
        let _fid_eq: fid_eq = unsafe { std::mem::zeroed() };
        let _fid_mr: fid_mr = unsafe { std::mem::zeroed() };

        // Control resource attributes.
        let _fabric_attr: fi_fabric_attr = unsafe { std::mem::zeroed() };
        let _domain_attr: fi_domain_attr = unsafe { std::mem::zeroed() };
        let _ep_attr: fi_ep_attr = unsafe { std::mem::zeroed() };
        let _cq_attr: fi_cq_attr = unsafe { std::mem::zeroed() };
        let _eq_attr: fi_eq_attr = unsafe { std::mem::zeroed() };
        let _fi_mr_attr: fi_mr_attr = unsafe { std::mem::zeroed() };
    }

    /// Example use case of the library.
    #[test]
    fn test_get_info() {
        unsafe {
            // Configure hints.
            let hints = fi_allocinfo();
            assert_eq!(hints.is_null(), false);

            (*hints).caps = FI_MSG as u64;
            (*hints).mode = FI_CONTEXT;
            (*(*hints).ep_attr).type_ = fi_ep_type_FI_EP_RDM;
            (*(*hints).domain_attr).mr_mode = FI_MR_LOCAL as i32;
            let prov_name = CString::new("tcp").unwrap();
            (*(*hints).fabric_attr).prov_name = prov_name.into_raw() as *mut i8;

            // Get Fabric info based on the hints.
            let mut info_ptr = ptr::null_mut();
            let version = fi_version();
            let ret = fi_getinfo(
                version,
                ptr::null_mut(),
                ptr::null_mut(),
                0,
                hints,
                &mut info_ptr,
            );
            assert_eq!(ret, 0);

            // Free the info structure returned by fi_getinfo.
            if !info_ptr.is_null() {
                fi_freeinfo(info_ptr);
            }

            // Free the hints structure we allocated.
            fi_freeinfo(hints);
        }
    }
}
