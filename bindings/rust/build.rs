use bindgen::callbacks::ItemInfo;
use bindgen::callbacks::ItemKind;
use bindgen::callbacks::ParseCallbacks;
use std::env;
use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

// Global variables for common directories.
static CARGO_MANIFEST_DIR: OnceLock<PathBuf> = OnceLock::new();
static CARGO_WORKSPACE_DIR: OnceLock<PathBuf> = OnceLock::new();

fn get_cargo_manifest_dir() -> &'static PathBuf {
    CARGO_MANIFEST_DIR.get_or_init(|| PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()))
}

// Use CARGO_MANIFEST_DIR to get absolute path to the crate, then navigate to libfabric root.
// The conditional directory walking supports for both `cargo build` and `cargo publish` scenarios.
//
// For the case of cargo build:
// manifest_dir = {your_libfabric_directory}/bindings/rust
//
// For the case of cargo publish:
// manifest_dir = {your_libfabric_directory}/target/package/ofi-libfabric-sys-x.y.z
fn get_cargo_workspace_dir() -> &'static PathBuf {
    CARGO_WORKSPACE_DIR.get_or_init(|| {
        let manifest_dir = get_cargo_manifest_dir();
        manifest_dir
            .ancestors()
            .skip(2) // Skip the 0th (myself) and 1st ancestor directories.
            .take(2) // Check for both 2nd and 3rd ancestor directories.
            .find(|dir| dir.join("Cargo.toml").exists())
            .unwrap_or_else(|| panic!("Could not find parent directory with Cargo.toml"))
            .to_path_buf()
    })
}

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

fn build_libfabric(install_dir: &PathBuf) {
    // Build the libfabric.so on the fly, such that its symbols can be accessed during the Rust binding compilation.
    // This way, the libfabric.so library can later be dynamically linked during run-time.
    // Else, you will receive the following error; = note: ld: cannot find -lfabric
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let libfabric_rsync_dir = out_dir.join("vendored");
    let cargo_manifest_dir = get_cargo_manifest_dir();
    let cargo_workspace_dir = get_cargo_workspace_dir();

    println!(
        "cargo:warning=cargo_manifest_dir {}",
        cargo_manifest_dir.display()
    );
    println!(
        "cargo:warning=cargo_workspace_dir {}",
        cargo_workspace_dir.display()
    );
    println!(
        "cargo:warning=libfabric_rsync_dir {}",
        libfabric_rsync_dir.display()
    );
    println!("cargo:warning=install_dir {}", install_dir.display());
    println!("cargo:warning=out_dir {}", out_dir.display());

    // Rsync the libfabric codebase into OUT_DIR, as the compilation process should never modify the source directory.
    if !libfabric_rsync_dir.exists() {
        println!("cargo:warning=Running rsync as part of libfabric compilation.");
        Command::new("rsync")
            .arg("-av")
            .arg("--exclude=vendored")
            .arg(format!("{}/", cargo_workspace_dir.display()))
            .arg(&libfabric_rsync_dir)
            .status()
            .expect("Failed to copy vendor directory");
    }

    // Run autogen.
    println!("cargo:warning=Running autogen.sh as part of libfabric compilation.");
    assert!(
        Command::new("sh")
            .current_dir(&libfabric_rsync_dir)
            .arg("autogen.sh")
            .status()
            .unwrap()
            .success()
    );

    // Run configure.
    //
    // Note that the binary is compiled on-the-fly to extract relevant header files under the compilation folder.
    // The arguments provided for such compilation is not important, as the Libfabric API interface stays the same.
    // During run-time, a separate user compiled library should be dynamically loaded via exporting the 'LD_LIBRARY_PATH' environment variable.
    println!("cargo:warning=Running configure as part of libfabric compilation.");
    assert!(
        Command::new("sh")
            .current_dir(&libfabric_rsync_dir)
            .arg("configure")
            .arg(format!("--prefix={}", install_dir.display()))
            .status()
            .unwrap()
            .success()
    );

    // Run make install.
    println!("cargo:warning=Running make install as part of libfabric compilation.");
    assert!(
        Command::new("make")
            .current_dir(&libfabric_rsync_dir)
            .arg("-j")
            .arg("install")
            .status()
            .unwrap()
            .success()
    );

    println!("cargo:warning=Libfabric successfully compiled.");
}

fn main() {
    #[cfg(not(target_os = "linux"))]
    compile_error!("This binding is only compatible with Linux.");

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
    println!(
        "cargo:warning=Building the binding with vendored: {}",
        vendored
    );

    let (lib_path, include_paths) = match vendored {
        true => {
            // Vendored option, build the libfabric based on the available source code.
            let install_dir = PathBuf::from(env::var("OUT_DIR").unwrap()).join("install");
            build_libfabric(&install_dir);

            // Return relevant paths using global variables.
            let libfabric_dir = get_cargo_workspace_dir();

            (
                // Provide static link search path.
                // Vendored option, thus should refer to the compiled library's installation path.
                install_dir.join("lib"),
                vec![
                    libfabric_dir.clone(),
                    libfabric_dir.join("include"),
                    libfabric_dir.join("include").join("rdma"),
                    libfabric_dir.join("include").join("rdma").join("providers"),
                ],
            )
        }
        false => {
            // Non-vendored option, and thus should refer to the already installed library's path.
            let lib = pkg_config::Config::new().probe("libfabric").unwrap();
            assert_eq!(1, lib.include_paths.len());
            assert_eq!(1, lib.link_paths.len());

            // Return relevant paths.
            (
                // Provide static link search path.
                // Non-vendored option, and thus should refer to the already installed library's path.
                lib.link_paths[0].clone(),
                vec![
                    lib.include_paths[0].clone(),
                    lib.include_paths[0].join("rdma"),
                    lib.include_paths[0].join("rdma").join("providers"),
                ],
            )
        }
    };

    println!("cargo:rustc-link-search=native={}", lib_path.display());
    println!("cargo:warning=Library link path: {}", lib_path.display());
    include_paths
        .iter()
        .enumerate()
        .for_each(|(i, x)| println!("cargo:warning=include_paths[{}]: {}", i, x.display()));

    // Compiles the wrapper.[ch].
    //
    // This generates a libwrapper.a, which is statically linked against your Rust application code.
    // Then, from the statically linked single executable, libfabric.so is dynamically called via libwrapper.
    //
    // The goal of the wrapper.[ch] is to create translation unit for "static inline" functions, such that they can be properly FFI'ed.
    // TODO: https://github.com/rust-lang/rust-bindgen/discussions/2405
    let mut builder = cc::Build::new();
    let cargo_manifest_dir = get_cargo_manifest_dir().display();
    builder.file(format!("{cargo_manifest_dir}/wrapper.c"));
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
