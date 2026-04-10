// build.rs
fn main() {
    #[cfg(feature = "cuda")]
    {
        // Print version requirement warning
        println!("cargo:warning=cuTENSOR 2.0+ is REQUIRED. Version 1.x uses a different API and will NOT work.");
        println!("cargo:warning=Install cuTENSOR 2.0+: conda install -c nvidia cutensor-cu12 (for CUDA 12)");
        println!("cargo:warning=Or download from: https://developer.nvidia.com/cutensor-downloads");

        let lib_path = if let Ok(path) = std::env::var("CUTENSOR_PATH") {
            println!("cargo:warning=Using CUTENSOR_PATH={}", path);
            path
        } else if let Ok(cuda) = std::env::var("CUDA_PATH") {
            let path = format!("{}/lib64", cuda);
            println!("cargo:warning=Using CUDA_PATH/lib64={}", path);
            path
        } else {
            println!("cargo:warning=Using default path /usr/local/cuda/lib64");
            "/usr/local/cuda/lib64".to_string()
        };

        println!("cargo:rustc-link-search=native={}", lib_path);
        println!("cargo:rustc-link-lib=dylib=cutensor");
        println!("cargo:rerun-if-env-changed=CUTENSOR_PATH");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");

        // Check if library exists (either libcutensor.so or libcutensor.so.2 for cuTENSOR 2.x)
        let lib_file = format!("{}/libcutensor.so", lib_path);
        let lib_file_v2 = format!("{}/libcutensor.so.2", lib_path);
        if !std::path::Path::new(&lib_file).exists() && !std::path::Path::new(&lib_file_v2).exists()
        {
            println!(
                "cargo:warning=libcutensor.so not found at {}. Linking may fail.",
                lib_path
            );
            println!("cargo:warning=Set CUTENSOR_PATH to the directory containing libcutensor.so");
            println!("cargo:warning=For pip-installed cuTENSOR: pip install cutensor-cu12, then create symlink:");
            println!("cargo:warning=  ln -s libcutensor.so.2 $CUTENSOR_PATH/libcutensor.so");
        } else if std::path::Path::new(&lib_file_v2).exists()
            && !std::path::Path::new(&lib_file).exists()
        {
            println!("cargo:warning=Found libcutensor.so.2 but not libcutensor.so - you may need to create a symlink:");
            println!(
                "cargo:warning=  ln -s libcutensor.so.2 {}/libcutensor.so",
                lib_path
            );
        }
    }
}
