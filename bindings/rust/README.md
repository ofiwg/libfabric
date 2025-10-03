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
// Clean the existing build.
cargo clean

// Build, using the Libfabric binary that is compiled on-the-fly.
cargo build --features vendored

// Build, using the already installed Libfabric.
// You may be required to set `PKG_CONFIG_PATH` env variable to point towards `libfabric.pkg` file.
cargo build

// Unit-tests.
cargo test

// Unit-tests with ASAN enabled.
cargo test --features asan
```

### How to use the library

Add the crate dependency under your Rust application's `Cargo.toml` file. Then;

```rust
use bindings as ffi;

fn test_get_info() {
    unsafe {
        // Configure hints.
        let hints = ffi::fi_allocinfo();
        assert_eq!(hints.is_null(), false);

        (*hints).caps = ffi::FI_MSG as u64;
        (*hints).mode = ff::FI_CONTEXT;
        (*(*hints).ep_attr).type_ = ffi::fi_ep_type_FI_EP_RDM;
        (*(*hints).domain_attr).mr_mode = ffi::FI_MR_LOCAL as i32;
        let prov_name = CString::new("efa").unwrap();
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
  publicly exported under `bindings` namespace.
- `wrapper.[ch]`: Wrapper source files that simply calls the static inline
  functions. This way, an isolated translation unit for each static inline
  function is made, for which the Rust bindgen is able to link against it.
- `tests/unit_test.rs`: Unit tests.
