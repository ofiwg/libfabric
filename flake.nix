{
  description = "libfabric - Open Fabric Interfaces";

  inputs = {
    flake-parts.url = "https://flakehub.com/f/hercules-ci/flake-parts/0.1.424";
    nixpkgs.url = "https://flakehub.com/f/DeterminateSystems/nixpkgs-weekly/0.1.918255";
  };

  outputs = inputs@{ flake-parts, nixpkgs, ... }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];

      perSystem = { system, lib, pkgs, ... }:
        let
          # Import our reusable library
          fabLib = import ./contrib/nix/lib { inherit lib; };

          isLinux = pkgs.stdenv.hostPlatform.isLinux;
          isDarwin = pkgs.stdenv.hostPlatform.isDarwin;

          # Local source filtered to only include relevant files
          libfabricSrc = lib.fileset.toSource {
            root = ./.;
            fileset = lib.fileset.unions [
              ./src ./include ./prov ./util ./man ./fabtests
              ./configure.ac ./Makefile.am ./autogen.sh ./config
              (lib.fileset.fileFilter (file: lib.hasSuffix ".in" file.name) ./.)
              (lib.fileset.fileFilter (file: lib.hasSuffix ".m4" file.name) ./.)
              (lib.fileset.fileFilter (file: file.name == "Makefile.include" || file.name == "Makefile.am") ./.)
            ];
          };

          # Cross-compilation setup (Darwin -> Linux)
          crossTargets = lib.optionalAttrs isDarwin {
            aarch64-linux = {
              crossPkgs = pkgs.pkgsCross.aarch64-multiplatform;
              nativePkgs = import nixpkgs { system = "aarch64-linux"; config.allowUnfree = true; };
            };
            x86_64-linux = {
              crossPkgs = pkgs.pkgsCross.gnu64;
              nativePkgs = import nixpkgs { system = "x86_64-linux"; config.allowUnfree = true; };
            };
          };

          # Build package set for a given pkgs
          mkPackages = p: { isCross ? false, nativePkgs ? null }:
            let
              cudaHeaders = fabLib.mkCudaHeaders p;
              rocmRuntime = fabLib.getRocmRuntime p;

              cassini-headers' = p.callPackage ./contrib/nix/pkgs/cassini-headers { };
              cxi-driver-headers' = p.callPackage ./contrib/nix/pkgs/cxi-driver-headers { };

              libcxi' = p.callPackage ./contrib/nix/pkgs/libcxi ({
                cassini-headers = cassini-headers';
                cxi-driver-headers = cxi-driver-headers';
                cuda_cudart = cudaHeaders;
                level-zero = p.level-zero;
                rocm-runtime = rocmRuntime;
              } // lib.optionalAttrs isCross (fabLib.crossDepsForLibcxi {
                inherit nativePkgs;
                mkCudaHeaders = fabLib.mkCudaHeaders;
              }));

              accel-config' = p.callPackage ./contrib/nix/pkgs/accel-config {
                systemdLibs = if isCross then nativePkgs.systemdLibs else p.systemdLibs or null;
              };

              xpmem' = p.callPackage ./contrib/nix/pkgs/xpmem { };
            in {
              cassini-headers = cassini-headers';
              libcxi = libcxi';
              accel-config = accel-config';
              xpmem = xpmem';
              libfabric = p.callPackage ./contrib/nix/pkgs/libfabric ({
                src = libfabricSrc;
                # Disable OPX - requires Cornelis proprietary headers (hfi1dv.h) not in nixpkgs
                enableProvOpx = false;
                # Linux-only dependencies
                numactl = fabLib.linuxOnly p p.numactl;
                libnl = fabLib.linuxOnly p p.libnl;
                rdma-core = fabLib.linuxOnly p p.rdma-core;
                liburing = fabLib.linuxOnly p p.liburing;
                lttng-ust = fabLib.linuxOnly p p.lttng-ust;
                libpsm2 = fabLib.linuxOnly p p.libpsm2;
                cassini-headers = fabLib.linuxOnly p cassini-headers';
                cxi-driver-headers = fabLib.linuxOnly p cxi-driver-headers';
                libcxi = fabLib.linuxOnly p libcxi';
                json_c = fabLib.linuxOnly p p.json_c;
                idxd-config = fabLib.linuxOnly p accel-config';
                cuda_cudart = fabLib.linuxOnly p cudaHeaders;
                gdrcopy = fabLib.linuxOnly p (p.cudaPackages.gdrcopy or null);
                rocm-runtime = fabLib.linuxOnly p p.rocmPackages.clr;
                level-zero = fabLib.linuxOnly p p.level-zero;
              } // lib.optionalAttrs isCross (fabLib.crossDepsForLibfabric {
                inherit nativePkgs;
                mkCudaHeaders = fabLib.mkCudaHeaders;
              }));
            };

          # Build package sets
          nativePackages = mkPackages pkgs { };

          crossPackageSets = lib.mapAttrs (_: { crossPkgs, nativePkgs }:
            mkPackages crossPkgs { isCross = true; inherit nativePkgs; }
          ) crossTargets;

          # Package names to expose
          packageNames = [ "libfabric" "cassini-headers" "libcxi" "accel-config" "xpmem" ];

          # Development shell inputs
          commonBuildInputs = with pkgs; [
            autoconf automake libtool pkg-config libuuid curl
          ] ++ lib.optionals isLinux [
            numactl libnl rdma-core liburing lttng-ust json_c
            libpsm2 ucx hwloc level-zero
            nativePackages.cassini-headers nativePackages.libcxi
            nativePackages.accel-config nativePackages.xpmem
          ] ++ [ cmocka ];

          devShellPackages = commonBuildInputs ++ (with pkgs; [
            clang-tools bear
          ] ++ lib.optionals isLinux [ gdb valgrind ]
            ++ lib.optionals isDarwin [ lldb ]);

        in {
          _module.args.pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

          packages = {
            inherit (nativePackages) libfabric;
            default = nativePackages.libfabric;
          } // lib.optionalAttrs isLinux {
            inherit (nativePackages) cassini-headers libcxi accel-config xpmem;
          } // lib.concatMapAttrs (arch: packages:
            fabLib.mkCrossPackageAttrs {
              inherit packages;
              prefix = "cross-${arch}";
              names = packageNames;
            }
          ) crossPackageSets;

          devShells = {
            default = pkgs.mkShell {
              name = "libfabric-dev";
              packages = devShellPackages;
              inputsFrom = [ nativePackages.libfabric ];
              shellHook = ''
                echo "libfabric development shell (${system})"
                echo ""
                echo "Build commands:"
                echo "  ./autogen.sh      - Generate configure script"
                echo "  ./configure       - Configure the build"
                echo "  make              - Build libfabric"
                echo "  make check        - Run tests"
                echo ""
              '' + lib.optionalString isLinux ''
                echo "Provider dependencies available:"
                echo "  - rdma-core (verbs, efa)"
                echo "  - libpsm2 (psm2)"
                echo "  - libcxi + cassini-headers (cxi)"
                echo "  - ucx (ucx)"
                echo "  - level-zero (ze/GPU)"
              '' + lib.optionalString isDarwin ''
                echo "Darwin build (limited providers: tcp, udp, rxm, sockets)"
                echo ""
                echo "Cross-compilation targets available:"
                echo "  nix build .#cross-aarch64-linux-libfabric"
                echo "  nix build .#cross-x86_64-linux-libfabric"
              '';
              CFLAGS = "-g -O0";
            };
            ci = pkgs.mkShell {
              name = "libfabric-ci";
              packages = commonBuildInputs;
            };
          } // lib.mapAttrs' (arch: { crossPkgs, ... }: {
            name = "cross-${arch}";
            value = fabLib.mkCrossShell {
              inherit crossPkgs arch;
              crossPackages = crossPackageSets.${arch};
            };
          }) crossTargets;

          checks = { inherit (nativePackages) libfabric; };
        };
    };
}
