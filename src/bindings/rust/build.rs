use bindgen::callbacks::ItemInfo;
use bindgen::callbacks::ItemKind;
use bindgen::callbacks::ParseCallbacks;
use std::env;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

#[derive(Debug)]
struct RenameFunctions;

// Rename function callback, such that those static inline functions are replaced without the "_" prefix.
// This way, the library is able to export such functions under fi_xyz(), and not _fi_xyz().
impl ParseCallbacks for RenameFunctions {
    // This is to remove _fi_send() --> fi_send() for Rust function.
    fn item_name(&self, original_name: ItemInfo<'_>) -> Option<String> {
        original_name.name.strip_prefix("_").map(String::from)
    }

    // This is to revert fi_xyz() --> _fi_xyz(), only for extern link_name.
    fn generated_link_name_override(&self, item_info: ItemInfo<'_>) -> Option<String> {
        // Explicitly use link_name for those marked with "_" prefix, which indicates for static inline wrapper.
        match item_info.kind {
            ItemKind::Function => {
                if item_info.name.starts_with("_") {
                    return Some(String::from(item_info.name));
                }
                None
            }
            _ => None,
        }
    }
}

fn build_libfabric(out_dir: &PathBuf, libfabric_rsync_dir: &PathBuf) {
    // Build the libfabric on the fly, such that its header files under the installation path could be referred by the bindgen.
    let libfabric_par_dir = Path::new("../../../../");
    let install_dir = out_dir.join("install");

    // Rsync the libfabric codebase into OUT_DIR, as the compilation process should never modify the source directory.
    if !libfabric_rsync_dir.exists() {
        Command::new("rsync")
            .arg("-av")
            .arg("--exclude='libfabric'")
            .arg(format!(
                "{}",
                libfabric_par_dir.join("libfabric").display()
            ))
            .arg(format!("{}/", out_dir.display()))
            .status()
            .expect("Failed to copy vendor directory");
    }

    // Run autogen.
    eprintln!("Run autogen.sh as part of libfabric compilation.");
    assert!(Command::new("sh")
        .current_dir(&libfabric_rsync_dir)
        .arg("autogen.sh")
        .status()
        .unwrap()
        .success());

    // Run configure.
    //
    // Note that the binary is compiled on-the-fly to extract relevant header files under the compilation folder.
    // The arguments provided for such compilation is not important, as the Libfabric API interface stays the same.
    // During run-time, a separate user compiled library should be dynamically loaded via exporting the 'LD_LIBRARY_PATH' environment variable.
    eprintln!("Run configure as part of libfabric compilation.");
    assert!(Command::new("sh")
        .current_dir(&libfabric_rsync_dir)
        .arg("configure")
        .arg(format!("--prefix={}", install_dir.display()))
        .status()
        .unwrap()
        .success());

    // Run make install.
    eprintln!("Run make install as part of libfabric compilation.");
    assert!(Command::new("make")
        .current_dir(&libfabric_rsync_dir)
        .arg("-j")
        .arg("install")
        .status()
        .unwrap()
        .success());

    eprintln!("Libfabric successfully compiled.");
}

fn main() {
    #[cfg(windows)]
    compile_error!("This binding isn't compatible with Windows.");

    // Conditional compilation from the source code versus refer to the already installed library,
    // based on the vendor feature flag (ex: cargo build --features vendored).
    let vendored = cfg!(feature = "vendored");
    let include_paths = match vendored {
        true => {
            // Vendored option, build the libfabric based on the available source code.
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let install_dir = out_dir.join("install");
            let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
            let libfabric_rsync_dir = out_dir.join("libfabric");
            build_libfabric(&out_dir, &libfabric_rsync_dir);
            // Tell cargo to look for the compiled shared library under the specified directory.
            // Under below directories, we will build the libfabric.so on the fly.
            println!(
                "{}",
                format!("cargo:rustc-link-search={}/lib", install_dir.display())
            );
            vec![
                libfabric_rsync_dir.clone(),
                libfabric_rsync_dir.clone().join("include"),
                libfabric_rsync_dir.clone().join("include").join("rdma"),
                libfabric_rsync_dir.clone().join("install").join("include"),
                libfabric_rsync_dir
                    .clone()
                    .join("install")
                    .join("include")
                    .join("rdma"),
            ]
        }
        false => {
            // Use the system's installed libfabric instead.
            println!("cargo:rustc-link-lib=fabric");
            let lib = pkg_config::Config::new().probe("libfabric").unwrap();
            lib.include_paths
        }
    };

    // Compile the wrapper.c/h.
    // The goal of the wrapper.c/h is to create translation unit for "static inline" functions, such that they can be properly FFI'ed.
    let mut builder = cc::Build::new();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").expect("failed to get current directory");
    builder.file(format!("{manifest_dir}/wrapper.c"));
    for path in &include_paths {
        builder.include(format!("{}", path.display()));
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
