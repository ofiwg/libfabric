{
  lib,
  stdenv,
  fetchFromGitHub,
  autoreconfHook,
  pkg-config,
  python3,
  libuuid,
  numactl,
  cassini-headers,
  cxi-driver-headers,
  json_c,
  libconfig,
  libuv,
  fuse,
  libyaml,
  libnl,
  criterion,
  lm_sensors ? null,
  # GPU/hmem support
  cuda_cudart ? null,
  level-zero ? null,
  rocm-runtime ? null,
}:

stdenv.mkDerivation rec {
  pname = "libcxi";
  version = "13.0.0";

  src = fetchFromGitHub {
    owner = "HewlettPackard";
    repo = "shs-libcxi";
    rev = "release/shs-${version}";
    hash = "sha256-zkVlIwrUKvtzuKwgk4yfAvmoehh5JSWFdZBeBINuFdU=";
  };

  nativeBuildInputs = [
    autoreconfHook
    pkg-config
    python3
  ];

  buildInputs = [
    criterion
    libuuid
    numactl
    cassini-headers
    cxi-driver-headers
    json_c
    libconfig
    libuv
    fuse
    libyaml
    libnl
  ] ++ lib.optionals (lm_sensors != null) [
    lm_sensors
  ] ++ lib.optionals (cuda_cudart != null) [
    cuda_cudart
  ] ++ lib.optionals (level-zero != null) [
    level-zero
  ] ++ lib.optionals (rocm-runtime != null) [
    rocm-runtime
  ];

  configureFlags = [
    "--with-cassini-headers=${cassini-headers}"
    "--without-systemd"
    "--disable-tests"
    # Set udev rules directory to output path
    "--with-udevrulesdir=${placeholder "out"}/lib/udev/rules.d"
  ] ++ lib.optionals (lm_sensors == null) [
    "--disable-libsensors"
  ] ++ lib.optionals (cuda_cudart != null) [
    "--with-cuda=${cuda_cudart}"
  ] ++ lib.optionals (level-zero != null) [
    "--with-ze=${level-zero}"
  ] ++ lib.optionals (rocm-runtime != null) [
    "--with-rocm=${rocm-runtime}"
  ];

  postPatch = ''
    # Fix path to csr_defs.json in the python script
    substituteInPlace utils/cxi_dump_csrs.py \
      --replace-fail "/usr/share/cassini-headers/csr_defs.json" \
                     "${cassini-headers}/share/cassini-headers/csr_defs.json" \
      --replace-fail "../cassini-headers/install/share/cassini-headers/csr_defs.json" \
                     "${cassini-headers}/share/cassini-headers/csr_defs.json"
  '';

  # The configure script doesn't have --disable-libsensors, we need to patch
  preConfigure = lib.optionalString (lm_sensors == null) ''
    # Remove sensors check if lm_sensors is not available
    substituteInPlace configure.ac --replace-fail \
      'AC_CHECK_HEADERS([sensors/sensors.h], ,' \
      'dnl AC_CHECK_HEADERS([sensors/sensors.h], ,'
  '';

  enableParallelBuilding = true;
  doCheck = false;

  # Disable test building by patching Makefile.am
  postAutoreconf = ''
    # Remove tests from SUBDIRS and conditional test building
    if [ -f Makefile.am ]; then
      sed -i 's/SUBDIRS.*tests/SUBDIRS =/' Makefile.am || true
      sed -i '/^if.*HAVE_CRITERION/,/^endif/d' Makefile.am || true
    fi
  '';

  # Patch libtool to skip ldconfig during cross-compilation
  # (ldconfig is a target binary that can't run on the host)
  preBuild = ''
    sed -i 's|finish_cmds=.*ldconfig.*|finish_cmds=""|' libtool || true
  '';

  postInstall = ''
    rm -f $out/lib/*.la
  '';

  meta = with lib; {
    description = "Low-level interface library for HPE Cassini high-speed NIC (Slingshot)";
    homepage = "https://github.com/HewlettPackard/shs-libcxi";
    license = with licenses; [ gpl2 bsd2 ];
    platforms = platforms.linux;
    maintainers = with maintainers; [ ];
  };
}
