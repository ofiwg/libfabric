{
  lib,
  stdenv,
  fetchFromGitHub,
  autoreconfHook,
  pkg-config,
  libuuid,
  json_c,
  systemdLibs ? null,
  kmod,
}:

stdenv.mkDerivation rec {
  pname = "accel-config";
  version = "4.1.9";

  src = fetchFromGitHub {
    owner = "intel";
    repo = "idxd-config";
    rev = "accel-config-v${version}";
    hash = "sha256-uV6cOha+g5fZBq+ucrNmfUW+3gQzX9BKGYvYaeCYv40=";
  };

  nativeBuildInputs = [
    autoreconfHook
    pkg-config
  ];

  buildInputs = [
    libuuid
    json_c
    kmod
  ] ++ lib.optionals (systemdLibs != null) [
    systemdLibs
  ];

  configureFlags = [
    "--disable-test"
    "--disable-docs"
  ] ++ lib.optionals (systemdLibs == null) [
    "--disable-systemd"
  ];

  # Create version.m4 which is normally generated from git, and make
  # git-version-gen a no-op so it doesn't overwrite during build
  postPatch = ''
    echo "m4_define([GIT_VERSION], [${version}])" > version.m4
    cat > git-version-gen <<'EOF'
    #!/bin/sh
    exit 0
    EOF
    chmod +x git-version-gen
  '';

  enableParallelBuilding = true;

  meta = with lib; {
    description = "Utility for controlling and configuring Intel DSA and IAA accelerators";
    homepage = "https://github.com/intel/idxd-config";
    license = licenses.lgpl21Plus;
    platforms = platforms.linux;
    maintainers = with maintainers; [ ];
  };
}
