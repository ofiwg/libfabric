# Reusable library functions for libfabric Nix packaging
{ lib }:

{
  # Combine CUDA header packages into a single derivation
  # cuda_runtime.h needs headers from cuda_cccl, crt/ from cuda_nvcc, nvml.h from cuda_nvml_dev
  mkCudaHeaders = pkgs: pkgs.symlinkJoin {
    name = "cuda-headers-combined";
    paths = [
      pkgs.cudaPackages.cuda_cudart
      pkgs.cudaPackages.cuda_cccl
      pkgs.cudaPackages.cuda_nvcc
      pkgs.cudaPackages.cuda_nvml_dev.include
    ];
  };

  # Get ROCm runtime if available on the platform (x86_64-linux only)
  getRocmRuntime = pkgs:
    if pkgs.stdenv.hostPlatform.isx86_64
    then pkgs.rocmPackages.clr
    else null;

  # Conditionally return a value only on Linux
  linuxOnly = pkgs: value:
    if pkgs.stdenv.hostPlatform.isLinux then value else null;

  # Generate cross-compiled package attributes with a prefix
  # packages: attrset of packages
  # prefix: string prefix for attribute names (e.g., "cross-aarch64-linux")
  # names: list of package names to include
  mkCrossPackageAttrs = { packages, prefix, names }:
    lib.listToAttrs (map (name: {
      name = "${prefix}-${name}";
      value = packages.${name};
    }) names);

  # Create a cross-compilation dev shell
  mkCrossShell = { crossPkgs, crossPackages, arch }:
    crossPkgs.mkShell {
      name = "libfabric-cross-${arch}";
      packages = with crossPkgs; [
        autoconf
        automake
        libtool
        pkg-config
        libuuid
        curl
        numactl
        libnl
        rdma-core
      ];
      inputsFrom = [ crossPackages.libfabric ];
      shellHook = ''
        echo "Cross-compilation shell: ${arch}"
        echo "Target: ${crossPkgs.stdenv.hostPlatform.config}"
      '';
    };

  # Dependencies that need native packages when cross-compiling
  # Returns an attrset to merge with callPackage args
  crossDepsForLibcxi = { nativePkgs, mkCudaHeaders }:
    {
      lm_sensors = nativePkgs.lm_sensors;
      criterion = nativePkgs.criterion;
      fuse = nativePkgs.fuse;
      cuda_cudart = mkCudaHeaders nativePkgs;
      level-zero = nativePkgs.level-zero;
      rocm-runtime =
        if nativePkgs.stdenv.hostPlatform.isx86_64
        then nativePkgs.rocmPackages.clr
        else null;
    };

  crossDepsForLibfabric = { nativePkgs, mkCudaHeaders }:
    {
      lttng-ust = nativePkgs.lttng-ust;
      rdma-core = nativePkgs.rdma-core;
      liburing = nativePkgs.liburing;
      libnl = nativePkgs.libnl;
      # DSA/idxd is x86-only (uses immintrin.h)
      idxd-config = null;
      cuda_cudart = mkCudaHeaders nativePkgs;
      gdrcopy = nativePkgs.cudaPackages.gdrcopy or null;
      rocm-runtime =
        if nativePkgs.stdenv.hostPlatform.isx86_64
        then nativePkgs.rocmPackages.clr
        else null;
      level-zero = nativePkgs.level-zero;
    };
}
