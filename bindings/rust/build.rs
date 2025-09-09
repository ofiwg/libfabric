use bindgen::callbacks::ItemInfo;
use bindgen::callbacks::ItemKind;
use bindgen::callbacks::ParseCallbacks;
use std::env;
use std::path::Path;
use std::path::PathBuf;

#[derive(Debug)]
struct RenameFunctions;

// Rename function callback, such that those static inline functions are replaced without the "wrap_" prefix.
// This way, the library is able to export such functions under `fi_xyz()`, rather than `wrap_fi_xyz()`.
impl ParseCallbacks for RenameFunctions {
    // This is to remove the prefix, from `wrap_fi_send()` --> `fi_send()` for Rust function.
    fn item_name(&self, original_name: ItemInfo<'_>) -> Option<String> {
        original_name.name.strip_prefix("wrap_").map(String::from)
    }

    // Explicitly use link_name for those marked with "wrap_" prefix, which indicates for static inline wrapper.
    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        match item_info.kind {
            ItemKind::Function => {
                if item_info.name.starts_with("wrap_") {
                    return Some(String::from(item_info.name));
                }
                None
            }
            _ => None,
        }
    }
}

fn main() {
    #[cfg(windows)]
    compile_error!("This binding isn't compatible with Windows.");

    // Link asan library.
    let asan = cfg!(feature = "asan");
    if asan {
        println!("cargo:rustc-link-lib=asan");
    }

    // Link libfabric library.
    println!("cargo:rustc-link-lib=fabric");

    // Conditional reference of header files from the source code, versus from the already installed library,
    // based on the vendor feature flag (ex: cargo build --features vendored).
    let vendored = cfg!(feature = "vendored");
    let include_paths = match vendored {
        true => {
            let libfabric_par_dir = Path::new("../../../");
            vec![
                libfabric_par_dir.join("libfabric"),
                libfabric_par_dir.join("libfabric").join("include"),
                libfabric_par_dir
                    .join("libfabric")
                    .join("include")
                    .join("rdma"),
                libfabric_par_dir
                    .join("libfabric")
                    .join("include")
                    .join("rdma")
                    .join("providers"),
            ]
        }
        false => {
            let lib = pkg_config::Config::new().probe("libfabric").unwrap();
            assert_eq!(1, lib.include_paths.len());
            vec![
                lib.include_paths[0].clone(),
                lib.include_paths[0].join("rdma"),
                lib.include_paths[0].join("rdma").join("providers"),
            ]
        }
    };
    include_paths
        .iter()
        .enumerate()
        .for_each(|(i, x)| eprintln!("include_paths[{}]: {}", i, x.display()));

    // Compiles the wrapper.[ch].
    //
    // This generates a libwrapper.a, which is statically linked against your Rust application code.
    // Then, from the statically linked single executable, libfabric.so is dynamically called via libwrapper.
    //
    // The goal of the wrapper.[ch] is to create translation unit for "static inline" functions, such that they can be properly FFI'ed.
    // TODO: https://github.com/rust-lang/rust-bindgen/discussions/2405
    let mut builder = cc::Build::new();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("failed to get current directory");
    builder.file(format!("{manifest_dir}/wrapper.c"));
    for path in &include_paths {
        builder.include(format!("{}", path.display()));
    }
    if asan {
        builder.flag("-fsanitize=address");
        builder.flag("-fsanitize-recover=address");
    }
    builder.compile("wrapper");

    // Finally, build the Rust binding.
    let builder = bindgen::Builder::default().header("wrapper.h").clang_args(
        include_paths
            .iter()
            .map(|dir| format!("-I{}", dir.display())),
    );
    let bindings = builder
        .clang_arg("-fno-inline-functions")
        .clang_arg("-Wno-error=implicit-function-declaration")
        .clang_arg("-Wno-error=int-conversion")
        .parse_callbacks(Box::new(RenameFunctions))
        .generate_inline_functions(false)
        .wrap_static_fns(false)
        .derive_default(true)
        .derive_debug(true)
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
