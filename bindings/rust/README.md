# ofi-libfabric-sys

The official distribution of lightweight Rust bindings for Libfabric - a communication API for high-performance parallel and distributed applications, by the OFI Working Group.

### Motivation

Increasing number of HPC networking code is being written in Rust. Naturally, to
support Libfabric usage in Rust, there needs a proper Rust library that wraps
Libfabric APIs written in C. This practice is commonly referred as a Rust
binding / FFI (foreign function interface).

This library builds a lightweight Rust binding via bindgen. Lightweight, meaning
there's no additional abstraction on top of the automatically generated code via
bindgen, aside from the `wrapper.[ch]` which is strictly used to support
`static inline` functions to be properly bound, by introducing a new translation
unit upon compilation.

### Build

```
// Not strictly necessary for the build process, but necessary for running the built library.
LD_LIBRARY_PATH={your_directory_containing_libfabric.so}

// Clean the existing build.
cargo clean

// [Recommended] Build, using the already installed Libfabric.
PKG_CONFIG_PATH={your_directory_containing_libfabric.pc} cargo build

// [Only for library development] Build, using the Libfabric binary that is compiled on-the-fly.
// The vendored option is not supported for library import scenario (imported under Cargo.toml).
// Rather, it is meant for developers building the ofi-libfabric-sys library under the libfabric repo.
cargo build --features vendored

// Unit-tests.
cargo test

// Unit-tests with ASAN enabled.
cargo test --features asan
```

### How to use the library

Add the crate dependency under your Rust application's `Cargo.toml` file. Then;

```rust
use ofi_libfabric_sys::bindgen as ffi;
use std::ffi::CString;
use std::ptr;

fn test_get_info() {
    unsafe {
        // Configure hints.
        let hints = ffi::fi_allocinfo();
        assert_eq!(hints.is_null(), false);

        (*hints).caps = ffi::FI_MSG as u64;
        (*hints).mode = ffi::FI_CONTEXT;
        (*(*hints).ep_attr).type_ = ffi::fi_ep_type_FI_EP_RDM;
        (*(*hints).domain_attr).mr_mode = ffi::FI_MR_LOCAL as i32;
        let prov_name = CString::new("tcp").unwrap();
        (*(*hints).fabric_attr).prov_name = prov_name.into_raw() as *mut i8;

        // Get Fabric info based on the hints.
        let mut info_ptr = ptr::null_mut();
        let version = ffi::fi_version();
        let ret = ffi::fi_getinfo(
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
            ffi::fi_freeinfo(info_ptr);
        }

        // Free the hints structure we allocated.
        ffi::fi_freeinfo(hints);
    }
}
```

### Files

- `build.rs`: The actual build script for the bindgen.
- `src/lib.rs`: The generated binding is copy-pasted programmatically and
  publicly exported under `bindgen` namespace.
- `wrapper.[ch]`: Wrapper source files that simply calls the static inline
  functions. This way, an isolated translation unit for each static inline
  function is made, for which the Rust bindgen is able to link against it.
- `tests/unit_test.rs`: Unit tests.
