use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = PathBuf::from(&manifest_dir).join("../../sdk/include/geniex.h");

    println!("cargo:rerun-if-env-changed=CARGO_GENIEX_LIB_DIR");
    if let Ok(lib_dir) = env::var("CARGO_GENIEX_LIB_DIR") {
        println!("cargo:rustc-link-search=native={}", lib_dir);
    } else {
        let pkg_lib = PathBuf::from(&manifest_dir).join("../../sdk/pkg-geniex/lib");
        if pkg_lib.exists() {
            println!("cargo:rustc-link-search=native={}", pkg_lib.display());
        }
        let build_lib = PathBuf::from(&manifest_dir).join("../../sdk/build/src");
        if build_lib.exists() {
            println!("cargo:rustc-link-search=native={}", build_lib.display());
        }
    }
    println!("cargo:rustc-link-lib=geniex");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_str().unwrap())
        .generate_comments(false)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");

    // On Windows, copy geniex.dll to target_dir so binaries run without DLL PATH errors
    if cfg!(target_os = "windows") {
        if let Some(target_dir) = out_path.ancestors().nth(3) {
            let search_dirs = if let Ok(lib_dir) = env::var("CARGO_GENIEX_LIB_DIR") {
                vec![PathBuf::from(lib_dir)]
            } else {
                vec![
                    PathBuf::from(&manifest_dir).join("../../sdk/pkg-geniex/lib"),
                    PathBuf::from(&manifest_dir).join("../../sdk/build/src"),
                ]
            };

            for search_dir in search_dirs {
                let dll_src = search_dir.join("geniex.dll");
                if dll_src.exists() {
                    let _ = std::fs::copy(&dll_src, target_dir.join("geniex.dll"));
                }
                // Copy llama_cpp plugin DLLs if available
                let llama_plugin_dir = search_dir.join("llama_cpp");
                if llama_plugin_dir.exists() {
                    let plugin_target = target_dir.join("llama_cpp");
                    let _ = std::fs::create_dir_all(&plugin_target);
                    if let Ok(entries) = std::fs::read_dir(&llama_plugin_dir) {
                        for entry in entries.flatten() {
                            if entry.path().extension().map_or(false, |ext| ext == "dll") {
                                let _ = std::fs::copy(entry.path(), plugin_target.join(entry.file_name()));
                            }
                        }
                    }
                }
            }
        }
    }
}
