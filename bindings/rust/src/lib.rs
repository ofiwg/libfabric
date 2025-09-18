// Suppress expected warnings from bindgen-generated code.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(deref_nullptr)]
#![allow(improper_ctypes)]
#![allow(unnecessary_transmutes)]
#![allow(unsafe_op_in_unsafe_fn)]

// Dump the bindgen generated code, and export as public module named "bindings".
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
