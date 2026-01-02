{
  lib,
  stdenv,
  pkg-config,
  autoreconfHook,
  libpsm2 ? null,
  libuuid,
  numactl ? null,
  libnl ? null,
  rdma-core ? null,
  valgrind ? null,
  cmocka,
  lttng-ust ? null,
  liburing ? null,
  curl,

  # Source - can be overridden to use local source
  src,
  version ? "dev",

  # Optional dependencies
  cuda_cudart ? null,
  gdrcopy ? null,
  rocm-runtime ? null,
  level-zero ? null,
  cassini-headers ? null,
  cxi-driver-headers ? null,
  libcxi ? null,
  json_c ? null,
  idxd-config ? null,
  enableValgrind ? false,
  enableLttng ? lttng-ust != null,

  # Builtin providers.
  enableProvRxm ? true,
  enableProvTcp ? true,
  enableProvUdp ? true,
  enableProvSockets ? true,

  # Real/hardware providers.
  enableProvEfa ? stdenv.hostPlatform.isLinux,
  enableProvOpx ? stdenv.hostPlatform.isLinux && stdenv.hostPlatform.isx86_64,
  enableProvPsm2 ? stdenv.hostPlatform.isLinux && stdenv.hostPlatform.isx86_64,
  enableProvPsm3 ? stdenv.hostPlatform.isLinux && stdenv.hostPlatform.isx86_64,
  enableProvUsnic ? stdenv.hostPlatform.isLinux,
  enableProvVerbs ? stdenv.hostPlatform.isLinux,
  enableProvShm ? stdenv.hostPlatform.isLinux,
  enableProvCxi ? stdenv.hostPlatform.isLinux && libcxi != null && cassini-headers != null && cxi-driver-headers != null,

  # GPU memory support - enabled when packages are provided
  enableHmemCuda ? stdenv.hostPlatform.isLinux && cuda_cudart != null,
  enableCudaGDRCopy ? stdenv.hostPlatform.isLinux && enableHmemCuda && gdrcopy != null,
  enableHmemRocr ? stdenv.hostPlatform.isLinux && rocm-runtime != null,
  enableHmemZe ? stdenv.hostPlatform.isLinux && level-zero != null,
}:

stdenv.mkDerivation {
  pname = "libfabric";
  inherit src version;

  enableParallelBuilding = true;

  outputs = [
    "out"
    "dev"
    "man"
  ];

  nativeBuildInputs = [
    pkg-config
    autoreconfHook
  ];

  buildInputs = [
    libuuid
    curl
  ]
  ++ lib.optionals (numactl != null) [ numactl ]
  ++ lib.optionals (libnl != null) [ libnl ]
  ++ lib.optionals (rdma-core != null) [ rdma-core ]
  ++ lib.optionals (liburing != null) [ liburing ]
  ++ lib.optionals enableLttng [ lttng-ust ]
  ++ lib.optionals enableProvPsm2 [ libpsm2 ]
  ++ lib.optionals enableValgrind [ valgrind ]
  ++ lib.optionals enableHmemCuda [ cuda_cudart ]
  ++ lib.optionals enableCudaGDRCopy [ gdrcopy ]
  ++ lib.optionals enableHmemRocr [ rocm-runtime ]
  ++ lib.optionals enableHmemZe [ level-zero ]
  ++ lib.optionals enableProvCxi [
    libcxi
    cassini-headers
    cxi-driver-headers
    json_c
  ]
  ++ lib.optionals (enableProvShm && idxd-config != null) [ idxd-config ];

  checkInputs = [ cmocka ];

  # Disable CXI multinode tests - they require static linking against libfabric
  # which doesn't work when CXI provider is built as a plugin (HAVE_CXI_DL=1)
  # We use sed to remove the entire multinode test blocks (lines 55-94 approximately)
  postPatch = lib.optionalString enableProvCxi ''
    # Remove multinode test definitions from Makefile.include
    # These tests link statically against libfabric which fails when CXI is a plugin
    sed -i '/^# Stand-alone srun tests/,/^if HAVE_CRITERION/{ /^if HAVE_CRITERION/!d }' prov/cxi/Makefile.include
  '';

  configureFlags = [
    (lib.enableFeature enableProvEfa "efa")
    (lib.enableFeature enableProvOpx "opx")
    (lib.enableFeatureAs enableProvPsm2 "psm2" (lib.getLib libpsm2))
    (lib.enableFeature enableProvPsm3 "psm3")
    (lib.enableFeature enableProvRxm "rxm")
    (lib.enableFeature enableProvTcp "tcp")
    (lib.enableFeature enableProvUdp "udp")
    (lib.enableFeature enableProvUsnic "usnic")
    (lib.enableFeature enableProvVerbs "verbs")
    (lib.enableFeature enableProvShm "shm")
    (lib.enableFeature enableProvCxi "cxi")
    (lib.enableFeature enableProvSockets "sockets")
    (lib.withFeatureAs enableValgrind "valgrind" valgrind)
    # Use dlopen for GPU libs - avoids linking issues during cross-compilation
    # and allows runtime detection of GPU support
    (lib.withFeatureAs enableHmemCuda "cuda" (lib.getDev cuda_cudart))
    (lib.enableFeature enableHmemCuda "cuda-dlopen")
    (lib.withFeatureAs enableCudaGDRCopy "gdrcopy" (lib.getDev gdrcopy))
    (lib.withFeatureAs enableHmemRocr "rocr" (lib.getDev rocm-runtime))
    (lib.withFeatureAs enableHmemZe "ze" (lib.getDev level-zero))
    (lib.withFeatureAs (enableProvShm && idxd-config != null) "idxd" (
      if idxd-config != null then idxd-config else null
    ))
  ] ++ lib.optionals (libnl != null) [
    "--with-libnl=${lib.getDev libnl}"
  ];

  meta = with lib; {
    homepage = "https://ofiwg.github.io/libfabric/";
    description = "Open Fabric Interfaces";
    license = with licenses; [
      gpl2
      bsd2
    ];
    platforms = platforms.all;
    maintainers = with maintainers; [
      bzizou
      sielicki
    ];
  };
}
