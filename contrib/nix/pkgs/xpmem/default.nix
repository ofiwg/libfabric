{
  lib,
  stdenv,
  fetchFromGitHub,
  autoreconfHook,
  pkg-config,
}:

stdenv.mkDerivation rec {
  pname = "xpmem";
  version = "2.7.3-unstable-2024-09-11";

  src = fetchFromGitHub {
    owner = "hpc";
    repo = "xpmem";
    rev = "3bcab55479489fdd93847fa04c58ab16e9c0b3fd";
    hash = "sha256-MnuuUSgqjlrW0cfw3wZa2DA5MPgtjHwWsnD5OR/QMoo=";
  };

  nativeBuildInputs = [
    autoreconfHook
    pkg-config
  ];

  # Only build the userspace library, not the kernel module
  configureFlags = [
    "--disable-kernel-module"
  ];

  enableParallelBuilding = true;

  meta = with lib; {
    description = "Cross-partition memory (XPMEM) userspace library";
    longDescription = ''
      XPMEM is a Linux kernel module that enables a process to map the
      memory of another process into its virtual address space. This
      enables upper-level protocols such as MPI and SHMEM to perform
      single-copy address-space to address-space transfers.

      This package only provides the userspace library. The kernel module
      must be installed separately.
    '';
    homepage = "https://github.com/hpc/xpmem";
    license = with licenses; [ gpl2Only lgpl21Only ];
    platforms = platforms.linux;
    maintainers = with maintainers; [ ];
  };
}
